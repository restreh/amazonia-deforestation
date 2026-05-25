"""Refinamiento del area de interes a los limites municipales.

Carga los poligonos municipales (de un archivo DANE/IGAC indicado en la
configuracion o, en su defecto, del proyecto abierto geoBoundaries), filtra a
los municipios objetivo, calcula su union, su bounding box y su area, y guarda
el limite recortado. Sirve tambien de base para la segmentacion territorial de
resultados que exige la propuesta.

Ejecucion:
    python scripts/refine_aoi.py
"""

from __future__ import annotations

import json
import unicodedata
import urllib.request
from pathlib import Path

import geopandas as gpd

GEOBOUNDARIES_API = "https://www.geoboundaries.org/api/current/gbOpen/COL/ADM2/"

# Nombres de columna candidatos para el nombre del municipio segun la fuente.
NAME_COLUMNS = ("shapeName", "MPIO_CNMBR", "mpio_cnmbr", "NOMBRE_MPI", "nombre", "name")


def _normalize(text: str) -> str:
    """Minuscula sin acentos ni espacios sobrantes, para comparar nombres."""
    norm = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return norm.lower().strip()


def load_boundaries(config: dict) -> gpd.GeoDataFrame:
    """Lee los limites municipales desde el archivo configurado o desde geoBoundaries."""
    path = config["aoi"].get("boundaries_path")
    if path and Path(path).exists():
        return gpd.read_file(path)

    with urllib.request.urlopen(GEOBOUNDARIES_API, timeout=60) as resp:
        meta = json.load(resp)
    return gpd.read_file(meta["gjDownloadURL"])


def _name_column(gdf: gpd.GeoDataFrame) -> str:
    for col in NAME_COLUMNS:
        if col in gdf.columns:
            return col
    raise KeyError(
        f"No se encontro columna de nombre municipal. Columnas disponibles: {list(gdf.columns)}"
    )


def refine_aoi(config: dict, out_dir: Path) -> dict:
    """Filtra a los municipios objetivo, calcula bbox y area, y guarda el limite."""
    gdf = load_boundaries(config)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs("EPSG:4326")

    name_col = _name_column(gdf)
    targets = {_normalize(m) for m in config["aoi"]["municipalities"]}
    selected = gdf[gdf[name_col].map(_normalize).isin(targets)].copy()

    if selected.empty:
        raise ValueError(
            f"Ningun municipio coincide con {targets} en la columna '{name_col}'. "
            f"Ejemplos en la fuente: {sorted(gdf[name_col].map(_normalize).unique())[:10]}"
        )

    union = selected.union_all() if hasattr(selected, "union_all") else selected.unary_union
    bounds = union.bounds  # (minx, miny, maxx, maxy) en EPSG:4326
    work_crs = config["processing"]["output_crs"]
    area_km2 = gpd.GeoSeries([union], crs="EPSG:4326").to_crs(work_crs).area.iloc[0] / 1e6

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "aoi_municipalities.geojson"
    selected.to_file(out_path, driver="GeoJSON")

    bbox = [round(bounds[0], 5), round(bounds[1], 5), round(bounds[2], 5), round(bounds[3], 5)]
    print(f"Municipios encontrados: {selected[name_col].tolist()}")
    print(f"bbox refinado (lon_min, lat_min, lon_max, lat_max): {bbox}")
    print(f"Area de la union: {area_km2:,.0f} km^2")
    print(f"Limite guardado en {out_path}")
    print("\nActualiza config/config.yaml -> aoi.bbox_geographic con el bbox de arriba.")
    return {"bbox_geographic": bbox, "area_km2": area_km2, "boundary_path": str(out_path)}
