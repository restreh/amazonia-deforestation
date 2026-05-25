"""Atributos contextuales por pixel: media, desviacion estandar y textura GLCM.

Calcula, para una capa (banda o indice), resumenes de la vecindad en ventanas
cuadradas. Media y desviacion estandar son baratas (filtro de caja). Las metricas
de textura GLCM (contraste, homogeneidad, entropia) se calculan de forma densa y
eficiente, sin recorrer pixel por pixel:

- Contraste y homogeneidad no requieren la matriz de co-ocurrencia completa: son
  esperanzas sobre la vecindad de funciones de la diferencia de niveles entre un
  pixel y su vecino, asi que salen con un filtro de caja por desplazamiento.
- La entropia si necesita la distribucion de pares, que se acumula con un filtro
  de caja por cada par de niveles (L x L), manteniendo solo acumuladores.

Convencion de ventana: para cada pixel de la ventana (como centro) se cuentan sus
pares de co-ocurrencia con los 8 vecinos a distancia 1, y esos conteos se agregan
sobre la ventana. Agrupar los 8 vecinos hace la co-ocurrencia simetrica e
isotropica, coherente con la vecindad de 8 usada en la I de Moran.

Manejo de NaN: un pixel sin dato (nube/sombra enmascarada) no aporta
a la ventana, y un par cuenta solo si ambos extremos son validos.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

# Vecindad de 8 (distancia 1). Agrupar los 8 desplazamientos da una GLCM simetrica.
NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _box_sum(arr: np.ndarray, size: int) -> np.ndarray:
    """Suma en ventana cuadrada de lado size, via filtro de caja (media * area)."""
    return uniform_filter(arr, size=size, mode="constant", cval=0.0) * (size * size)


def local_mean_std(layer: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Media y desviacion estandar de la vecindad, ignorando NaN.

    Devuelve arrays del mismo tamano. Donde la ventana no tiene pixeles validos,
    el resultado es NaN.
    """
    valid = np.isfinite(layer)
    filled = np.where(valid, layer, 0.0)
    n = _box_sum(valid.astype(np.float64), size)
    s = _box_sum(filled, size)
    s2 = _box_sum(filled * filled, size)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s / n
        var = s2 / n - mean * mean
        var = np.where(var < 0, 0.0, var)  # corrige ruido numerico
        std = np.sqrt(var)
    empty = n < 0.5   # n es un conteo; <0.5 == 0 robusto al residuo de punto flotante
    mean[empty] = np.nan
    std[empty] = np.nan
    return mean.astype("float32"), std.astype("float32")


def _quantize(layer: np.ndarray, levels: int, value_range=None) -> np.ndarray:
    """Cuantiza la capa a [0, levels-1]. value_range=(lo,hi) fija el rango; si es None
    usa los percentiles robustos (2-98) de la propia capa. NaN -> -1."""
    valid = np.isfinite(layer)
    if not valid.any():
        return np.full(layer.shape, -1, dtype=np.int16)
    if value_range is None:
        lo, hi = np.percentile(layer[valid], [2, 98])
    else:
        lo, hi = value_range
    if hi <= lo:
        hi = lo + 1.0
    scaled = (layer - lo) / (hi - lo)
    q = np.floor(scaled * levels)
    q = np.clip(q, 0, levels - 1)
    return np.where(valid, q, -1).astype(np.int16)


def _shift(arr: np.ndarray, dy: int, dx: int, fill) -> np.ndarray:
    """Desplaza arr por (dy, dx) rellenando el borde con fill (sin envolver)."""
    out = np.full_like(arr, fill)
    ys_src = slice(max(0, -dy), arr.shape[0] - max(0, dy))
    xs_src = slice(max(0, -dx), arr.shape[1] - max(0, dx))
    ys_dst = slice(max(0, dy), arr.shape[0] - max(0, -dy))
    xs_dst = slice(max(0, dx), arr.shape[1] - max(0, -dx))
    out[ys_dst, xs_dst] = arr[ys_src, xs_src]
    return out


def glcm_textures(layer: np.ndarray, size: int, levels: int = 16, value_range=None) -> dict:
    """Contraste, homogeneidad y entropia GLCM densos sobre la vecindad de 8.

    layer: capa float con NaN como nodato.
    size: lado de la ventana (p. ej. 3 o 5).
    levels: niveles de cuantizacion.
    Devuelve {'contrast', 'homogeneity', 'entropy'} como arrays float32; NaN donde
    la ventana no tiene pares validos.
    """
    q = _quantize(layer, levels, value_range)
    valid = q >= 0
    qf = q.astype(np.float64)

    sum_count = np.zeros(layer.shape, dtype=np.float64)     # pares validos en la ventana
    sum_contrast = np.zeros(layer.shape, dtype=np.float64)
    sum_homog = np.zeros(layer.shape, dtype=np.float64)
    pair_codes = []   # codigo de par (i*levels + j) por desplazamiento; -1 si invalido
    for dy, dx in NEIGHBORS:
        qn = _shift(qf, dy, dx, np.nan)
        vn = _shift(valid, dy, dx, False)
        both = valid & vn
        diff = np.where(both, qf - qn, 0.0)
        sum_count += _box_sum(both.astype(np.float64), size)
        sum_contrast += _box_sum(np.where(both, diff * diff, 0.0), size)
        sum_homog += _box_sum(np.where(both, 1.0 / (1.0 + diff * diff), 0.0), size)
        qn_int = np.where(np.isfinite(qn), qn, 0).astype(np.int32)
        pair_codes.append(np.where(both, q.astype(np.int32) * levels + qn_int, -1))

    with np.errstate(invalid="ignore", divide="ignore"):
        contrast = sum_contrast / sum_count
        homogeneity = sum_homog / sum_count

    # Entropia: -sum_k p_k log p_k, con p_k = conteo_k / total, acumulada por codigo.
    entropy = np.zeros(layer.shape, dtype=np.float64)
    for k in range(levels * levels):
        ind = np.zeros(layer.shape, dtype=np.float64)
        for code in pair_codes:
            ind += (code == k)
        if not ind.any():
            continue
        count_k = _box_sum(ind, size)
        with np.errstate(invalid="ignore", divide="ignore"):
            p = count_k / sum_count
            entropy += np.where(p > 0, -p * np.log(p), 0.0)

    empty = sum_count < 0.5   # idem: 0 pares validos, robusto al residuo flotante
    for arr in (contrast, homogeneity, entropy):
        arr[empty] = np.nan
    return {
        "contrast": contrast.astype("float32"),
        "homogeneity": homogeneity.astype("float32"),
        "entropy": entropy.astype("float32"),
    }
