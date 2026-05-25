"""Diagnostico de dependencia espacial: I de Moran y semivariograma empirico.

Cuantifica la autocorrelacion espacial de las capas del proyecto. El I de Moran
sobre rasters se calcula por convolucion con un kernel de 8 vecinos (reina), que
es equivalente a la formula clasica para pesos de grilla y escala a millones de
pixeles. El semivariograma empirico se estima por muestreo de pares de puntos y
sirve para fijar el rango espacial, base del tamano de bloque de validacion.

El I de Moran sobre los residuos del baseline (puntos muestreados) se calculara
con esda/PySAL cuando exista el modelo baseline.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial.distance import pdist

QUEEN = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype="float64")


def morans_i(arr, mask=None):
    """I de Moran global de un raster 2D con pesos de 8 vecinos (reina)."""
    a = arr.astype("float64")
    if mask is None:
        mask = np.isfinite(a)
    else:
        mask = mask & np.isfinite(a)

    valid = mask.astype("float64")
    mean = a[mask].mean()
    zc = np.where(mask, a - mean, 0.0)

    lag = ndimage.convolve(zc, QUEEN, mode="constant", cval=0.0)
    wcount = ndimage.convolve(valid, QUEEN, mode="constant", cval=0.0)

    num = np.sum(zc[mask] * lag[mask])
    weights = np.sum(wcount[mask])
    denom = np.sum(zc[mask] ** 2)
    n = int(mask.sum())
    if denom == 0 or weights == 0:
        return float("nan")
    return (n / weights) * (num / denom)


def empirical_variogram(arr, pixel_size_m, mask=None, n_samples=4000,
                        n_bins=20, max_lag_m=None, seed=0):
    """Semivariograma empirico por muestreo de pares. Devuelve (lags_m, semivarianza)."""
    rng = np.random.default_rng(seed)
    a = arr.astype("float64")
    if mask is None:
        mask = np.isfinite(a)
    ys, xs = np.where(mask)
    if len(ys) > n_samples:
        idx = rng.choice(len(ys), size=n_samples, replace=False)
        ys, xs = ys[idx], xs[idx]
    vals = a[ys, xs]
    coords = np.column_stack([xs * pixel_size_m, ys * pixel_size_m]).astype("float64")

    dist = pdist(coords)
    sqdiff = pdist(vals.reshape(-1, 1), metric="sqeuclidean")
    if max_lag_m is None:
        max_lag_m = dist.max() / 2
    keep = dist <= max_lag_m
    dist, sqdiff = dist[keep], sqdiff[keep]

    edges = np.linspace(0, max_lag_m, n_bins + 1)
    which = np.digitize(dist, edges)
    lags, semiv = [], []
    for b in range(1, n_bins + 1):
        sel = which == b
        if sel.sum() > 0:
            lags.append((edges[b - 1] + edges[b]) / 2)
            semiv.append(0.5 * sqdiff[sel].mean())
    return np.array(lags), np.array(semiv)


def estimate_range_m(lags, semiv, sill_fraction=0.95):
    """Rango espacial: primer lag donde la semivarianza alcanza sill_fraction del sill."""
    if len(semiv) == 0:
        return float("nan")
    sill = np.nanmax(semiv)
    threshold = sill_fraction * sill
    reached = np.where(semiv >= threshold)[0]
    return float(lags[reached[0]]) if len(reached) else float(lags[-1])


def spectral_separability(composite_path, label_path):
    """Media y desviacion por banda para clase 0 (permanencia) y 1 (perdida)."""
    import rasterio

    with rasterio.open(composite_path) as src:
        names = list(src.descriptions)
        bands = src.read().astype("float64")  # (band, y, x)
    with rasterio.open(label_path) as lsrc:
        label = lsrc.read(1)

    stats = {}
    for i, name in enumerate(names):
        band = bands[i]
        for cls in (0, 1):
            sel = (label == cls) & np.isfinite(band)
            key = (name or f"band{i+1}", cls)
            stats[key] = (float(np.nanmean(band[sel])), float(np.nanstd(band[sel])))
    return stats
