"""Indices espectrales sobre las composiciones trimestrales.

Calcula NDVI, NBR y NDWI a partir de un GeoTIFF de composicion multibanda.
Las bandas se localizan por su descripcion (nombre de asset), no por posicion,
para ser robusto al orden. Los indices son cocientes normalizados, asi que no
dependen de la escala de los valores crudos.

Pares de bandas:
    NDVI = (nir - red)    / (nir + red)       vigor de la vegetacion
    NBR  = (nir - swir22) / (nir + swir22)    area quemada / perdida de cobertura
    NDWI = (green - nir)  / (green + nir)      agua superficial (McFeeters)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

INDEX_BANDS = {
    "ndvi": ("nir", "red"),
    "nbr": ("nir", "swir22"),
    "ndwi": ("green", "nir"),
}


def normalized_difference(a, b):
    """Diferencia normalizada (a - b) / (a + b), con NaN donde el denominador es 0."""
    den = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        nd = (a - b) / den
    nd = np.where(den == 0, np.nan, nd)
    return nd.astype("float32")


def compute_indices(composite_path, out_path):
    """Lee una composicion y escribe un GeoTIFF multibanda con NDVI, NBR y NDWI."""
    with rasterio.open(composite_path) as src:
        name_to_idx = {desc: i for i, desc in enumerate(src.descriptions, start=1) if desc}
        needed = {b for pair in INDEX_BANDS.values() for b in pair}
        missing = needed - set(name_to_idx)
        if missing:
            raise KeyError("Faltan bandas en la composicion: " + str(sorted(missing)))
        data = {name: src.read(name_to_idx[name]).astype("float32") for name in needed}
        profile = src.profile.copy()

    results = {name: normalized_difference(data[a], data[b]) for name, (a, b) in INDEX_BANDS.items()}

    profile.update(count=len(results), dtype="float32", compress="deflate", nodata=np.nan)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        for i, (name, arr) in enumerate(results.items(), start=1):
            dst.write(arr, i)
            dst.set_band_description(i, name)
    return list(results.keys())
