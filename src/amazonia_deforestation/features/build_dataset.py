"""Ensamblaje de la tabla de caracteristicas por pixel.

Para cada pixel se construye un vector de atributos a partir de:
  - bandas espectrales de la composicion mediana (10) y del percentil 25 (10),
  - indices NDVI, NBR, NDWI (derivados de la mediana),
  - atributos contextuales (media, desviacion estandar y textura GLCM: contraste,
    homogeneidad, entropia) en ventanas 3 y 5, calculados sobre las bandas de la
    mediana y sobre los indices,
todo repetido en los cuatro trimestres como contexto estacional.

El percentil 25 entra solo como valor base (sin contextuales) para no duplicar la
dimensionalidad de la textura. Las contextuales se calculan sobre la mediana y los
indices, que son las capas con mayor relacion senal/ruido.

Procesamiento por capa: cada banda/indice se lee densamente, se calculan sus
contextuales sobre toda la grilla y se extraen los valores en los pixeles pedidos
(coords). Asi solo hay una capa densa en memoria a la vez y la tabla final contiene
unicamente los pixeles solicitados.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from amazonia_deforestation.features.contextual import glcm_textures, local_mean_std


def _layer_columns(layer, name, rows, cols, windows, levels, contextual, value_range=None):
    """Columnas de una capa: valor base y, si contextual, media/desv/GLCM por ventana."""
    out = {name: layer[rows, cols].astype("float32")}
    if not contextual:
        return out
    for w in windows:
        mean, std = local_mean_std(layer, w)
        out[f"{name}_mean_w{w}"] = mean[rows, cols]
        out[f"{name}_std_w{w}"] = std[rows, cols]
        tex = glcm_textures(layer, w, levels, value_range)
        for metric, arr in tex.items():
            out[f"{name}_{metric}_w{w}"] = arr[rows, cols]
    return out


def _add_bands(data, path, prefix, rows, cols, windows, levels, contextual):
    """Lee cada banda de un raster y agrega sus columnas a data."""
    with rasterio.open(path) as src:
        names = src.descriptions
        for bi in range(1, src.count + 1):
            bname = names[bi - 1] or f"b{bi}"
            layer = src.read(bi).astype("float64")
            data.update(_layer_columns(layer, f"{prefix}_{bname}", rows, cols,
                                       windows, levels, contextual))


def build_feature_table(config, composites_dir, indices_dir, coords,
                        levels=16, quarters=None, verbose=True):
    """Tabla de features (DataFrame) para los pixeles en coords (array N x 2: row, col)."""
    windows = config["features"]["contextual_windows"]
    if quarters is None:
        quarters = [q["id"] for q in config["temporal"]["composite_quarters"]]
    rows = coords[:, 0]
    cols = coords[:, 1]

    data: dict[str, np.ndarray] = {}
    for qid in quarters:
        if verbose:
            print("Trimestre " + qid + " ...", flush=True)
        med = Path(composites_dir) / ("composite_" + qid + "_median.tif")
        p25 = Path(composites_dir) / ("composite_" + qid + "_p25.tif")
        idx = Path(indices_dir) / ("indices_" + qid + ".tif")
        _add_bands(data, med, qid + "_med", rows, cols, windows, levels, contextual=True)
        _add_bands(data, p25, qid + "_p25", rows, cols, windows, levels, contextual=False)
        _add_bands(data, idx, qid, rows, cols, windows, levels, contextual=True)
    return pd.DataFrame(data)
