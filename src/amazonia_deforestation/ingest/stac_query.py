"""Consulta de disponibilidad de Sentinel-2 L2A via STAC Earth-Search.

Primer paso del entendimiento de los datos: cuantifica cuantas escenas hay
sobre el area de interes por trimestre y con que nubosidad. No requiere
credenciales de AWS; la API STAC es publica y de solo lectura.

Ejecucion:
    python scripts/check_availability.py
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pystac_client import Client


@dataclass
class SceneRecord:
    """Resumen de una escena Sentinel-2 devuelta por la consulta STAC."""

    item_id: str
    datetime: str
    cloud_cover: float
    tile: str
    quarter: str


def open_catalog(stac_url: str) -> Client:
    """Abre el catalogo STAC en la URL indicada."""
    return Client.open(stac_url)


def search_quarter(
    catalog: Client,
    collection: str,
    bbox: list[float],
    start: str,
    end: str,
    quarter_id: str,
    cloud_cover_max: float,
) -> list[SceneRecord]:
    """Busca escenas Sentinel-2 en una ventana temporal y devuelve sus registros."""
    search = catalog.search(
        collections=[collection],
        bbox=bbox,
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": cloud_cover_max}},
    )
    records: list[SceneRecord] = []
    for item in search.items():
        props = item.properties
        records.append(
            SceneRecord(
                item_id=item.id,
                datetime=str(item.datetime),
                cloud_cover=float(props.get("eo:cloud_cover", float("nan"))),
                tile=str(props.get("grid:code", props.get("s2:mgrs_tile", "NA"))),
                quarter=quarter_id,
            )
        )
    return records


def availability_report(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construye el reporte de disponibilidad por escena y el resumen por trimestre.

    Devuelve (detalle, resumen):
        detalle  una fila por escena (id, fecha, nubosidad, tile, trimestre).
        resumen  una fila por trimestre con conteos y estadisticos de nubosidad.
    """
    s2 = config["data_sources"]["sentinel2"]
    bbox = config["aoi"]["bbox_geographic"]
    cloud_max = config["temporal"]["cloud_cover_max"]

    catalog = open_catalog(s2["stac_url"])
    all_records: list[SceneRecord] = []
    for q in config["temporal"]["composite_quarters"]:
        all_records.extend(
            search_quarter(
                catalog=catalog,
                collection=s2["collection"],
                bbox=bbox,
                start=q["start"],
                end=q["end"],
                quarter_id=q["id"],
                cloud_cover_max=cloud_max,
            )
        )

    detail = pd.DataFrame([r.__dict__ for r in all_records])
    if detail.empty:
        return detail, pd.DataFrame()

    summary = (
        detail.groupby("quarter")
        .agg(
            scenes=("item_id", "count"),
            tiles=("tile", "nunique"),
            cloud_min=("cloud_cover", "min"),
            cloud_median=("cloud_cover", "median"),
            cloud_max=("cloud_cover", "max"),
            scenes_lt20=("cloud_cover", lambda s: int((s < 20).sum())),
        )
        .reset_index()
    )
    return detail, summary
