"""Dataset de recortes (patches) para el U-Net.

Construye recortes cuadrados (por defecto 256x256) sobre la grilla de trabajo. Cada
recorte apila como canales las bandas de la composicion mediana y los indices de los
cuatro trimestres (10 + 3 por trimestre = 52 canales). La mascara objetivo es la
etiqueta binaria de deforestacion alineada a 20 m.

Opcionalmente se anaden 4 canales de mascara de validez (uno por trimestre), con 1
donde la observacion es real y 0 donde habia NaN. Eso permite que el modelo distinga
"sin dato" de "reflectancia cero", que con el relleno simple eran indistinguibles. Con
mascaras activas el numero de canales pasa de 52 a 56.

Para respetar la particion espacial por bloques sin fuga de etiqueta, cada recorte
trae un peso por pixel igual a 1 en los pixeles del split correspondiente y 0 en el
resto; la perdida y las metricas se calculan solo donde el peso es 1. Asi un pixel de
validacion o prueba nunca aporta a la perdida de entrenamiento.

Durante entrenamiento se aplican aumentaciones (flips H/V y rot90) consistentes sobre
imagen, etiqueta y peso. En validacion y prueba no se aumenta.

Si se proporcionan estadisticas por canal (media y desviacion calculadas sobre train
con scripts/compute_channel_stats.py), read_stack estandariza cada canal espectral o
indice por (x - mu) / sigma. Los NaN se rellenan con la media del canal antes de
estandarizar, asi quedan en 0 tras el escalado. Los canales de validez no se tocan.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

TRAIN_CODE, VAL_CODE, TEST_CODE = 1, 2, 3
REFLECTANCE_SCALE = 10000.0   # numero digital Sentinel-2 a reflectancia aproximada
N_SPECTRAL = 10               # bandas de la composicion mediana


def channel_spec(config, composites_dir, indices_dir, validity_masks=False):
    """Lista ordenada de canales (ruta, banda, nombre, tipo).

    Orden: por trimestre, primero las 10 bandas espectrales y luego los 3 indices. Si
    validity_masks=True se anaden al final 4 canales (uno por trimestre) con tipo
    "validity": valen 1 donde hay observacion real y 0 donde habia NaN.
    """
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
    if validity_masks:
        for qid in quarters:
            med = Path(composites_dir) / ("composite_" + qid + "_median.tif")
            spec.append((med, 1, qid + "_validity", "validity"))
    return spec


def read_stack(spec, window, stats=None):
    """Lee los canales en la ventana dada y devuelve un arreglo (C, H, W).

    Canales "spectral" se dividen por REFLECTANCE_SCALE y "index" se dejan en su
    escala nativa [-1, 1]. Canales "validity" devuelven 1 donde la lectura no es NaN
    y 0 donde si lo es.

    Si stats=(means, stds) viene dado, cada canal espectral o indice se estandariza
    por (x - mu) / sigma, rellenando antes los NaN con mu del canal. Si stats es None,
    los NaN se rellenan con 0 (comportamiento previo).
    """
    if stats is not None:
        means, stds = stats
    chans = []
    for ci, (path, bi, _name, kind) in enumerate(spec):
        with rasterio.open(path) as src:
            a = src.read(bi, window=window).astype("float32")
        if kind == "validity":
            chans.append((~np.isnan(a)).astype("float32"))
            continue
        if kind == "spectral":
            a = a / REFLECTANCE_SCALE
        if stats is not None:
            mu = float(means[ci])
            sigma = float(stds[ci]) if float(stds[ci]) > 0.0 else 1.0
            a = np.where(np.isnan(a), mu, a)
            a = (a - mu) / sigma
        else:
            a = np.nan_to_num(a, nan=0.0)
        chans.append(a.astype("float32"))
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


def _augment(image, label, weight, rng):
    """Aplica flips H/V y rot90 aleatorios de forma consistente.

    image: (C, H, W); label y weight: (H, W). Se rota sobre los dos ultimos ejes.
    """
    if rng.random() < 0.5:
        image = image[:, :, ::-1]
        label = label[:, ::-1]
        weight = weight[:, ::-1]
    if rng.random() < 0.5:
        image = image[:, ::-1, :]
        label = label[::-1, :]
        weight = weight[::-1, :]
    k = int(rng.integers(0, 4))
    if k:
        image = np.rot90(image, k=k, axes=(1, 2))
        label = np.rot90(label, k=k)
        weight = np.rot90(weight, k=k)
    return (np.ascontiguousarray(image),
            np.ascontiguousarray(label),
            np.ascontiguousarray(weight))


class PatchDataset:
    """Dataset de recortes para PyTorch. Devuelve (imagen C x H x W, etiqueta H x W, peso H x W).

    El peso es 1 en los pixeles del split objetivo y 0 en el resto. Se importa torch de
    forma diferida para poder construir el indice de recortes sin torch instalado. Si
    augment=True (solo train), aplica flips H/V y rot90 aleatorios consistentes.
    """

    def __init__(self, config, composites_dir, indices_dir, label_path, split_path,
                 records, split, size, validity_masks=False, augment=False,
                 stats=None, seed=None):
        self.spec = channel_spec(config, composites_dir, indices_dir,
                                 validity_masks=validity_masks)
        self.label_path = str(label_path)
        self.split_path = str(split_path)
        self.records = records
        self.code = {"train": TRAIN_CODE, "val": VAL_CODE, "test": TEST_CODE}[split]
        self.size = size
        self.augment = bool(augment)
        self.stats = stats  # (means, stds) o None
        # rng por instancia; los workers del DataLoader se reseteanan en _ensure_rng
        self._seed = seed
        self._rng = None
        self._worker = None

    def _ensure_rng(self):
        import os
        try:
            import torch
            info = torch.utils.data.get_worker_info()
            wid = info.id if info is not None else -1
        except Exception:
            wid = -1
        if self._rng is None or self._worker != wid:
            base = self._seed if self._seed is not None else int.from_bytes(os.urandom(4), "little")
            self._rng = np.random.default_rng((base, wid))
            self._worker = wid

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        import torch
        r = self.records[i]
        win = Window(r["c0"], r["r0"], self.size, self.size)
        image = read_stack(self.spec, win, stats=self.stats)
        with rasterio.open(self.label_path) as src:
            label = src.read(1, window=win).astype("float32")
        with rasterio.open(self.split_path) as src:
            split = src.read(1, window=win)
        weight = (split == self.code).astype("float32")
        if self.augment:
            self._ensure_rng()
            image, label, weight = _augment(image, label, weight, self._rng)
        return (torch.from_numpy(image),
                torch.from_numpy(label),
                torch.from_numpy(weight))
