"""Prediccion densa por franjas (tiles) para los baseline.

Recorre el AOI por bloques con un halo (margen) suficiente para que los atributos
contextuales de los pixeles del borde del tile se calculen con sus vecinos reales.
Para cada tile construye los 612 atributos con la misma logica de ensamblaje del
entrenamiento (mismas funciones, mismos nombres de columna), reordena segun el
encabezado de la tabla de entrenamiento y predice la probabilidad por pixel.

El GLCM cuantiza cada capa con un rango global (percentiles 2-98 del raster
completo), el mismo que vio el entrenamiento; por eso los rangos se precomputan una
vez sobre los rasters completos y se pasan a cada tile, garantizando que los atributos
de prediccion coincidan exactamente con los de entrenamiento.

Por defecto solo predice los tiles que tocan bloques de validacion o prueba, para
acotar el computo. El resultado es un raster de probabilidad; donde no se predijo, NaN.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import rasterio
from rasterio.windows import Window

from amazonia_deforestation.features.build_dataset import _layer_columns


def feature_order(features_parquet):
    """Orden canonico de atributos, tomado del encabezado de la tabla de entrenamiento."""
    names = [f.name for f in pq.ParquetFile(str(features_parquet)).schema_arrow]
    return [n for n in names if n not in ("row", "col", "label")]


def _band_range(path, bi):
    """Percentiles (2, 98) de una banda completa, sobre pixeles validos."""
    with rasterio.open(path) as src:
        a = src.read(bi).astype("float64")   # float64 como en el entrenamiento (percentil identico)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return (0.0, 1.0)
    lo, hi = np.percentile(a, [2, 98])
    return (float(lo), float(hi))


def compute_ranges(comp_dir, idx_dir, quarters):
    """Rango de cuantizacion global por capa con GLCM (mediana e indices), clave = nombre de columna base."""
    ranges = {}
    for qid in quarters:
        med = Path(comp_dir) / ("composite_" + qid + "_median.tif")
        idx = Path(idx_dir) / ("indices_" + qid + ".tif")
        with rasterio.open(med) as src:
            names = src.descriptions
        for bi, bname in enumerate(names):
            ranges[qid + "_med_" + (bname or f"b{bi+1}")] = _band_range(med, bi + 1)
        with rasterio.open(idx) as src:
            inames = src.descriptions
        for bi, iname in enumerate(inames):
            ranges[qid + "_" + (iname or f"i{bi+1}")] = _band_range(idx, bi + 1)
    return ranges


def _quarter_columns(comp_dir, idx_dir, qid, rdwin, rows, cols, windows, levels, ranges):
    """Columnas de un trimestre sobre el arreglo leido en ventana (mismo orden que el entrenamiento)."""
    data = {}
    with rasterio.open(Path(comp_dir) / ("composite_" + qid + "_median.tif")) as src:
        names = src.descriptions
        arr = src.read(window=rdwin).astype("float64")
    for bi, bname in enumerate(names):
        name = qid + "_med_" + (bname or f"b{bi+1}")
        data.update(_layer_columns(arr[bi], name, rows, cols, windows, levels, True,
                                   ranges.get(name)))
    with rasterio.open(Path(comp_dir) / ("composite_" + qid + "_p25.tif")) as src:
        pnames = src.descriptions
        parr = src.read(window=rdwin).astype("float64")
    for bi, bname in enumerate(pnames):
        data[qid + "_p25_" + (bname or f"b{bi+1}")] = parr[bi][rows, cols].astype("float32")
    with rasterio.open(Path(idx_dir) / ("indices_" + qid + ".tif")) as src:
        inames = src.descriptions
        iarr = src.read(window=rdwin).astype("float64")
    for bi, iname in enumerate(inames):
        name = qid + "_" + (iname or f"i{bi+1}")
        data.update(_layer_columns(iarr[bi], name, rows, cols, windows, levels, True,
                                   ranges.get(name)))
    return data


def tile_features(config, comp_dir, idx_dir, r0, r1, c0, c1, halo, shape, levels, ranges):
    """DataFrame de atributos para todos los pixeles del tile interior [r0:r1, c0:c1]."""
    H, W = shape
    windows = config["features"]["contextual_windows"]
    quarters = [q["id"] for q in config["temporal"]["composite_quarters"]]
    rr0, rr1 = max(0, r0 - halo), min(H, r1 + halo)
    cc0, cc1 = max(0, c0 - halo), min(W, c1 + halo)
    rdwin = Window(cc0, rr0, cc1 - cc0, rr1 - rr0)
    ir = np.arange(r0, r1) - rr0
    ic = np.arange(c0, c1) - cc0
    gr, gc = np.meshgrid(ir, ic, indexing="ij")
    rows, cols = gr.ravel(), gc.ravel()
    data = {}
    for qid in quarters:
        data.update(_quarter_columns(comp_dir, idx_dir, qid, rdwin, rows, cols, windows, levels, ranges))
    return pd.DataFrame(data)


def predict_raster(config, model, feat_order, comp_dir, idx_dir, split_path, out_path,
                  eval_codes=(2, 3), tile=512, levels=16, verbose=True):
    """Predice probabilidad por pixel en los tiles que tocan eval_codes; escribe un raster."""
    halo = max(config["features"]["contextual_windows"]) // 2 + 1   # +1: el GLCM usa vecinos a distancia 1 dentro de la ventana
    quarters = [q["id"] for q in config["temporal"]["composite_quarters"]]
    if verbose:
        print("Precomputando rangos de cuantizacion globales...", flush=True)
    ranges = compute_ranges(comp_dir, idx_dir, quarters)

    with rasterio.open(split_path) as src:
        shape = (src.height, src.width)
        profile = src.profile.copy()
        split = src.read(1)
    H, W = shape

    out = np.full(shape, np.nan, dtype="float32")
    n_done = 0
    for r0 in range(0, H, tile):
        for c0 in range(0, W, tile):
            r1, c1 = min(H, r0 + tile), min(W, c0 + tile)
            if not np.isin(split[r0:r1, c0:c1], eval_codes).any():
                continue
            df = tile_features(config, comp_dir, idx_dir, r0, r1, c0, c1, halo, shape, levels, ranges)
            X = df[feat_order].to_numpy("float32")
            proba = model.predict_proba(X)[:, 1].astype("float32")
            out[r0:r1, c0:c1] = proba.reshape(r1 - r0, c1 - c0)
            n_done += 1
            if verbose:
                print("tile (" + str(r0) + "," + str(c0) + ") [" + str(n_done) + "]", flush=True)

    profile.update(count=1, dtype="float32", nodata=float("nan"), compress="deflate")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out, 1)
        dst.set_band_description(1, "deforestation_probability")
    if verbose:
        print("Tiles predichos: " + str(n_done) + " | raster en " + str(out_path))
    return out_path
