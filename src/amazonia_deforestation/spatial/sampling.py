"""Muestreo balanceado de pixeles de entrenamiento.

La prevalencia de perdida es ~2.8 %, demasiado baja para entrenar modelos por
pixel sin sesgo hacia la clase negativa. Este modulo toma todos los pixeles
positivos de los bloques de entrenamiento y un numero controlado de negativos
por positivo, restringido a esos mismos bloques para no filtrar informacion de
validacion o prueba.

Validacion y prueba se evaluan de forma densa sobre todos sus pixeles (no se
submuestrean), de modo que las metricas reflejan la prevalencia real.

Salida: tabla de pixeles muestreados (row, col, label) en formato Parquet, lista
para extraer vectores de caracteristicas en esos indices.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

TRAIN_CODE = 1


def sample_training_pixels(label: np.ndarray, split_map: np.ndarray,
                           neg_per_pos: int, seed: int):
    """Indices (row, col, label) de positivos + negativos balanceados en bloques de train."""
    rng = np.random.default_rng(seed)
    train = split_map == TRAIN_CODE
    pos_idx = np.flatnonzero((train & (label > 0)).ravel())
    neg_idx = np.flatnonzero((train & (label == 0)).ravel())

    n_pos = pos_idx.size
    n_neg_target = min(neg_idx.size, n_pos * neg_per_pos)
    neg_sample = rng.choice(neg_idx, size=n_neg_target, replace=False)

    sel = np.concatenate([pos_idx, neg_sample])
    rng.shuffle(sel)
    rows, cols = np.unravel_index(sel, label.shape)
    labels = label.ravel()[sel]
    return rows.astype("int32"), cols.astype("int32"), labels.astype("uint8")


def build_training_sample(config: dict, label_path: Path, split_path: Path,
                          out_path: Path) -> Path:
    """Construye y guarda la tabla de muestreo balanceado de entrenamiento."""
    s = config["modeling"]["sampling"]
    neg_per_pos = s["neg_per_pos"]
    seed = s["seed"]

    with rasterio.open(label_path) as src:
        label = src.read(1)
    with rasterio.open(split_path) as src:
        split_map = src.read(1)

    rows, cols, labels = sample_training_pixels(label, split_map, neg_per_pos, seed)
    df = pd.DataFrame({"row": rows, "col": cols, "label": labels})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    n = len(df)
    n_pos = int((df["label"] > 0).sum())
    print("Negativos por positivo (objetivo): " + str(neg_per_pos))
    print("Pixeles muestreados: " + format(n, ",") + " (positivos " + format(n_pos, ",")
          + ", " + format(n_pos / n if n else 0, ".2%") + ")")
    print("Tabla de entrenamiento en " + str(out_path))
    return out_path
