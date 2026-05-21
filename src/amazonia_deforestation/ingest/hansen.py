"""Lectura de Hansen Global Forest Change v1.12 sobre el área de interés.

Lee por ventana (vía /vsicurl/, sin descargar el tile completo de 10x10 grados)
las capas de Hansen GFC y deriva la etiqueta binaria de deforestación para el
año objetivo. Hansen GFC está en EPSG:4326, igual que el bbox del AOI, por lo
que no requiere reproyección.

Ejecución:
    python scripts/download_labels.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds


def hansen_tile_name(lon_min: float, lat_max: float) -> str:
    """Nombre del tile Hansen (granularidad de 10 grados) que contiene una esquina.

    El nombre usa el borde superior (norte) y el borde occidental (oeste) del
    tile, p. ej. '10N_080W' cubre de 0 a 10 N y de 80 a 70 W.
    """
    top = int(math.ceil(lat_max / 10.0) * 10)
    west = int(math.floor(lon_min / 10.0) * 10)
    lat_band = f"{abs(top):02d}{'N' if top >= 0 else 'S'}"
    lon_band = f"{abs(west):03d}{'W' if west < 0 else 'E'}"
    return f"{lat_band}_{lon_band}"


def tiles_for_bbox(bbox: list[float]) -> list[str]:
    """Tiles Hansen necesarios para cubrir el bbox (lon_min, lat_min, lon_max, lat_max)."""
    lon_min, lat_min, lon_max, lat_max = bbox
    tiles = set()
    # Recorre las esquinas y bordes en pasos de 10 grados para cubrir cruces de tile.
    lons = sorted({lon_min, lon_max})
    lats = sorted({lat_min, lat_max})
    for lo in lons:
        for la in lats:
            tiles.add(hansen_tile_name(lo, la))
    return sorted(tiles)


def read_layer_window(base_url: str, version: str, layer: str, tile: str, bbox: list[float]):
    """Lee la ventana del bbox de una capa Hansen. Devuelve (array, transform, crs, profile)."""
    lon_min, lat_min, lon_max, lat_max = bbox
    url = f"/vsicurl/{base_url}/Hansen_{version_tag(version)}_{layer}_{tile}.tif"
    with rasterio.open(url) as src:
        window = from_bounds(lon_min, lat_min, lon_max, lat_max, src.transform)
        data = src.read(1, window=window)
        transform = src.window_transform(window)
        profile = src.profile.copy()
        profile.update(
            height=data.shape[0],
            width=data.shape[1],
            transform=transform,
            count=1,
            compress="deflate",
        )
    return data, transform, src.crs, profile


def version_tag(version: str) -> str:
    """Convierte 'v1.12' al tag usado en los nombres de archivo: 'GFC-2024-v1.12'."""
    return f"GFC-2024-{version}"


def write_geotiff(path: Path, data: np.ndarray, profile: dict) -> None:
    """Escribe un arreglo 2D como GeoTIFF de una banda."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out_profile = profile.copy()
    out_profile.update(dtype=str(data.dtype), count=1)
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(data, 1)


def build_label(config: dict, out_dir: Path) -> Path:
    """Deriva la etiqueta binaria de deforestación del año objetivo y la guarda.

    label = 1 donde lossyear == (target_year - 2000); 0 en el resto.
    """
    hansen = config["data_sources"]["hansen_gfc"]
    bbox = config["aoi"]["bbox_geographic"]
    target_year = config["temporal"]["target_year"]
    year_code = target_year - 2000  # 2024 -> 24

    tiles = tiles_for_bbox(bbox)
    if len(tiles) > 1:
        raise NotImplementedError(
            f"El AOI cruza varios tiles Hansen {tiles}; falta el mosaico. "
            "Ajustar el bbox o implementar la unión de tiles."
        )
    tile = tiles[0]

    lossyear, transform, crs, profile = read_layer_window(
        hansen["base_url"], hansen["version"], "lossyear", tile, bbox
    )
    label = (lossyear == year_code).astype("uint8")

    label_path = out_dir / f"label_loss_{target_year}.tif"
    write_geotiff(label_path, label, profile)

    n_pos = int(label.sum())
    n_tot = int(label.size)
    print(f"Tile Hansen: {tile}")
    print(f"Ventana del AOI: {label.shape[0]} x {label.shape[1]} píxeles")
    print(f"Píxeles de pérdida {target_year}: {n_pos} de {n_tot} "
          f"(prevalencia {n_pos / n_tot:.4%})")
    print(f"Etiqueta guardada en {label_path}")
    return label_path
