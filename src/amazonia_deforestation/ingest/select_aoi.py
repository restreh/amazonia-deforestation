"""Selección dirigida por datos del AOI de trabajo (~5.000 km²).

Los municipios objetivo suman ~30.000 km², muy por encima del alcance de la
propuesta. Este módulo ubica una ventana del área objetivo, dentro de los
municipios, que maximiza la deforestación capturada según Hansen GFC del año
objetivo. Así el AOI queda anclado al núcleo activo, no a un rectángulo
arbitrario.

Procedimiento:
    1. Lee el límite municipal y la pérdida Hansen del año objetivo sobre su bbox.
    2. Enmascara la pérdida a los polígonos municipales.
    3. Agrega a una grilla gruesa y desliza una ventana del área objetivo,
       eligiendo la de mayor pérdida total (suma por imagen integral).
    4. Devuelve el bbox de esa ventana y la fracción de pérdida capturada.

Ejecución:
    python scripts/select_aoi.py
"""

from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import from_bounds

from amazonia_deforestation.ingest.hansen import tiles_for_bbox, version_tag


def best_window_sum(grid: np.ndarray, wy: int, wx: int) -> tuple[int, int, int]:
    """Top-left (r0, c0) y total de la ventana wy×wx de mayor suma, por imagen integral."""
    h, w = grid.shape
    wy = min(wy, h)
    wx = min(wx, w)
    ii = np.zeros((h + 1, w + 1), dtype=np.int64)
    ii[1:, 1:] = grid.cumsum(0).cumsum(1)
    best_total, best_r, best_c = -1, 0, 0
    for r0 in range(0, h - wy + 1):
        r1 = r0 + wy
        col_sums = (ii[r1, wx:w + 1] - ii[r0, wx:w + 1]
                    - ii[r1, 0:w - wx + 1] + ii[r0, 0:w - wx + 1])
        c0 = int(np.argmax(col_sums))
        total = int(col_sums[c0])
        if total > best_total:
            best_total, best_r, best_c = total, r0, c0
    return best_r, best_c, best_total


def select_aoi(config: dict, boundary_path: Path) -> dict:
    """Selecciona el AOI de trabajo sobre el núcleo de deforestación dentro de los municipios."""
    hansen = config["data_sources"]["hansen_gfc"]
    target_year = config["temporal"]["target_year"]
    year_code = target_year - 2000
    target_area = config["aoi"]["target_area_km2"]

    muni = gpd.read_file(boundary_path).to_crs("EPSG:4326")
    muni_union = muni.union_all() if hasattr(muni, "union_all") else muni.unary_union
    bbox = list(muni.total_bounds)  # (minx, miny, maxx, maxy)

    tiles = tiles_for_bbox(bbox)
    if len(tiles) > 1:
        raise NotImplementedError(f"El bbox municipal cruza varios tiles Hansen {tiles}.")
    url = f"/vsicurl/{hansen['base_url']}/Hansen_{version_tag(hansen['version'])}_lossyear_{tiles[0]}.tif"

    with rasterio.open(url) as src:
        window = from_bounds(*bbox, src.transform)
        lossyear = src.read(1, window=window)
        wt = src.window_transform(window)

    loss = (lossyear == year_code).astype(np.uint32)
    # Enmascara a los polígonos municipales.
    muni_mask = rasterize(
        [(geom, 1) for geom in muni.geometry],
        out_shape=loss.shape, transform=wt, fill=0, dtype="uint8",
    )
    loss *= muni_mask
    total_loss = int(loss.sum())

    # Agregación a grilla gruesa de ~1 km.
    px_deg_x = wt.a
    px_deg_y = -wt.e
    mid_lat = (bbox[1] + bbox[3]) / 2
    px_m_x = px_deg_x * 111_320 * math.cos(math.radians(mid_lat))
    px_m_y = px_deg_y * 110_540
    cell = max(1, round(1000 / px_m_y))
    h, w = loss.shape
    hc, wc = h // cell, w // cell
    grid = loss[:hc * cell, :wc * cell].reshape(hc, cell, wc, cell).sum(axis=(1, 3))

    # Ventana objetivo en celdas (cuadrada).
    side_km = math.sqrt(target_area)
    cell_km_x = px_m_x * cell / 1000
    cell_km_y = px_m_y * cell / 1000
    wx = max(1, round(side_km / cell_km_x))
    wy = max(1, round(side_km / cell_km_y))

    r0, c0, captured = best_window_sum(grid, wy, wx)

    # Coordenadas geográficas de la ventana.
    px_x0, px_y0 = c0 * cell, r0 * cell
    px_x1, px_y1 = (c0 + wx) * cell, (r0 + wy) * cell
    lon_min, lat_max = wt * (px_x0, px_y0)
    lon_max, lat_min = wt * (px_x1, px_y1)
    aoi_bbox = [round(float(lon_min), 5), round(float(lat_min), 5),
                round(float(lon_max), 5), round(float(lat_max), 5)]
    area_km2 = float((wx * cell_km_x) * (wy * cell_km_y))

    print(f"Tile Hansen: {tiles[0]}")
    print(f"Pérdida {target_year} dentro de los municipios: {total_loss:,} píxeles (30 m)")
    print(f"AOI seleccionado bbox (lon_min, lat_min, lon_max, lat_max): {aoi_bbox}")
    print(f"Área del AOI: {area_km2:,.0f} km^2")
    frac = captured / total_loss if total_loss else 0
    print(f"Pérdida capturada por el AOI: {captured:,} píxeles ({frac:.1%} del total municipal)")
    print("\nActualiza config/config.yaml -> aoi.bbox_geographic con el bbox de arriba.")
    return {"bbox_geographic": aoi_bbox, "area_km2": area_km2, "captured_fraction": frac}
