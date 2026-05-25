"""Alineacion de la etiqueta Hansen a la grilla de trabajo del AOI.

Reproyecta y remuestrea la capa lossyear de Hansen GFC (30 m, EPSG:4326) a la
grilla exacta de una composicion de referencia (20 m, EPSG:32618), por vecino
mas cercano, y deriva la etiqueta binaria de perdida del ano objetivo. Asi la
etiqueta coincide pixel a pixel con las composiciones e indices.

Ejecucion:
    python scripts/align_label.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from rasterio.windows import from_bounds

from amazonia_deforestation.ingest.hansen import tiles_for_bbox, version_tag


def align_label(config, reference_path, out_path):
    """Alinea la etiqueta de perdida del ano objetivo a la grilla de la composicion."""
    hansen = config["data_sources"]["hansen_gfc"]
    bbox = config["aoi"]["bbox_geographic"]
    target_year = config["temporal"]["target_year"]
    year_code = target_year - 2000

    # Grilla destino tomada de la composicion de referencia.
    with rasterio.open(reference_path) as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        height, width = ref.height, ref.width
        ref_profile = ref.profile.copy()

    # Lectura por ventana de lossyear sobre el AOI (con un pequeno margen).
    tile = tiles_for_bbox(bbox)[0]
    url = f"/vsicurl/{hansen['base_url']}/Hansen_{version_tag(hansen['version'])}_lossyear_{tile}.tif"
    buf = 0.02
    win_bounds = (bbox[0] - buf, bbox[1] - buf, bbox[2] + buf, bbox[3] + buf)
    with rasterio.open(url) as src:
        window = from_bounds(*win_bounds, src.transform)
        lossyear = src.read(1, window=window)
        src_transform = src.window_transform(window)
        src_crs = src.crs

    # Reproyeccion a la grilla destino por vecino mas cercano.
    dst = np.zeros((height, width), dtype=lossyear.dtype)
    reproject(
        source=lossyear,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
    )
    label = (dst == year_code).astype("uint8")

    out_profile = ref_profile.copy()
    out_profile.update(count=1, dtype="uint8", nodata=0, compress="deflate")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **out_profile) as dst_ds:
        dst_ds.write(label, 1)
        dst_ds.set_band_description(1, f"loss_{target_year}")

    n_pos = int(label.sum())
    n_tot = int(label.size)
    print("Grilla destino: " + str(height) + " x " + str(width) + " px a 20 m")
    print("Perdida " + str(target_year) + ": " + format(n_pos, ",") + " de " + format(n_tot, ",")
          + " (prevalencia " + format(n_pos / n_tot, ".4%") + ")")
    print("Etiqueta alineada en " + str(out_path))
    return out_path
