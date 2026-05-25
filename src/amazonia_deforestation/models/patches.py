"""Dataset de recortes (patches) para el U-Net.

Construye recortes cuadrados (por defecto 256x256) sobre la grilla de trabajo. Cada
recorte apila como canales las bandas de la composicion mediana y los indices de los
cuatro trimestres (10 + 3 por trimestre = 52 canales). La mascara objetivo es la
etiqueta binaria de deforestacion alineada a 20 m.

Para respetar la particion espacial por bloques sin fuga de etiqueta, cada recorte
trae un peso por pixel igual a 1 en los pixeles del split correspondiente y 0 en el
resto; la perdida y las metricas se calculan solo donde el peso es 1. Asi un pixel de
validacion o prueba nunca aporta a la perdida de entrenamiento.

Las composiciones tienen NaN donde la nube fue enmascarada; los canales se escalan y
los NaN se rellenan con 0, valor neutro tras el escalado.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

TRAIN_CODE, VAL_CODE, TEST_CODE = 1, 2, 3
REFLECTANCE_SCALE = 10000.0   # numero digital Sentinel-2 a reflectancia aproximada
N_SPECTRAL = 10               # bandas de la composicion mediana


def channel_spec(config, composites_dir, indices_dir):
    """Lista ordenada de canales (ruta, banda, nombre): mediana + indices por trimestre."""
    quarters = [q["id"] for q in config["temporal"]["composite_quarters"]]
    spec = []
    for qid in quarters:
        med = Path(composites_dir) / ("composite_" + qid + "_median.tif")
        idx = Path(indices_dir) / ("indices_" + qid + ".tif")
        with rasterio.open(med) as src:
            names = src.descriptions
        for bi in range(1, N_SPECTRAL + 1):
            spec.append((med, bi, qid + "_med_" + (names[bi - 1] or f"b{bi}"), "spectral"))
        with rasterio.open(idx) as src:
            inames = src.descriptions
        for bi in range(1, src.count + 1):
            spec.append((idx, bi, qid + "_" + (inames[bi - 1] or f"i{bi}"), "index"))
    return spec


def read_stack(spec, window):
    """Lee los canales en la ventana dada y devuelve un arreglo (C, H, W) escalado, NaN->0."""
    chans = []
    for path, bi, _name, kind in spec:
        with rasterio.open(path) as src:
            a = src.read(bi, window=window).astype("float32")
        if kind == "spectral":
            a = a / REFLECTANCE_SCALE
        a = np.nan_to_num(a, nan=0.0)
        chans.append(a)
    return np.stack(chans, axis=0)


def build_patch_index(split_path, size, stride):
    """Posiciones (r0, c0) de recortes en la grilla y el conteo de pixeles por split."""
    with rasterio.open(split_path) as src:
        split = src.read(1)
    h, w = split.shape
    records = []
    for r0 in range(0, h - size + 1, stride):
        for c0 in range(0, w - size + 1, stride):
            sub = split[r0:r0 + size, c0:c0 + size]
            records.append({
                "r0": r0, "c0": c0,
                "n_train": int((sub == TRAIN_CODE).sum()),
                "n_val": int((sub == VAL_CODE).sum()),
                "n_test": int((sub == TEST_CODE).sum()),
            })
    return records, (h, w)


def select_patches(records, split, min_pixels=1):
    """Filtra los recortes que contienen al menos min_pixels del split indicado."""
    key = {"train": "n_train", "val": "n_val", "test": "n_test"}[split]
    return [r for r in records if r[key] >= min_pixels]


class PatchDataset:
    """Dataset de recortes para PyTorch. Devuelve (imagen C x H x W, etiqueta H x W, peso H x W).

    El peso es 1 en los pixeles del split objetivo y 0 en el resto. Se importa torch de
    forma diferida para poder construir el indice de recortes sin torch instalado.
    """

    def __init__(self, config, composites_dir, indices_dir, label_path, split_path,
                 records, split, size):
        self.spec = channel_spec(config, composites_dir, indices_dir)
        self.label_path = str(label_path)
        self.split_path = str(split_path)
        self.records = records
        self.code = {"train": TRAIN_CODE, "val": VAL_CODE, "test": TEST_CODE}[split]
        self.size = size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        import torch
        r = self.records[i]
        win = Window(r["c0"], r["r0"], self.size, self.size)
        image = read_stack(self.spec, win)
        with rasterio.open(self.label_path) as src:
            label = src.read(1, window=win).astype("float32")
        with rasterio.open(self.split_path) as src:
            split = src.read(1, window=win)
        weight = (split == self.code).astype("float32")
        return (torch.from_numpy(image),
                torch.from_numpy(label),
                torch.from_numpy(weight))
