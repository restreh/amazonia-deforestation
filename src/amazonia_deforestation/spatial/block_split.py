"""Particion espacial por bloques para validacion sin fuga.

La deforestacion esta fuertemente autocorrelacionada en el espacio (I de Moran
~0.84, rango ~3.4 km en este AOI), por lo que una particion aleatoria por pixel
mezclaria vecinos casi identicos entre entrenamiento y prueba e inflaria las
metricas. Este modulo divide el AOI en bloques cuadrados de lado >= rango y
asigna bloques completos a entrenamiento, validacion y prueba.

La asignacion estratifica por presencia de pixeles positivos: los bloques con y
sin perdida se reparten por separado segun las proporciones objetivo, de modo
que cada conjunto reciba positivos aun con prevalencia baja.

Salida: un raster de la misma grilla que la etiqueta con codigos
1=train, 2=val, 3=test, alineado pixel a pixel.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

SPLIT_CODES = {"train": 1, "val": 2, "test": 3}


def block_side_px(block_size_km: float, pixel_size_m: float) -> int:
    """Lado del bloque en pixeles para un tamano en km y una resolucion en metros."""
    return max(1, round(block_size_km * 1000.0 / pixel_size_m))


def assign_block_ids(shape: tuple[int, int], block_px: int) -> np.ndarray:
    """Id de bloque por pixel: indice plano del bloque que contiene cada pixel."""
    h, w = shape
    n_cols = int(np.ceil(w / block_px))
    rows = np.arange(h) // block_px
    cols = np.arange(w) // block_px
    return (rows[:, None] * n_cols + cols[None, :]).astype(np.int64)


def _partition(ids: np.ndarray, ratios: dict, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reparte una lista de ids en train/val/test segun ratios (suma ~1)."""
    ids = np.array(ids, dtype=np.int64)
    rng.shuffle(ids)
    n = ids.size
    n_train = int(round(ratios["train"] * n))
    n_val = int(round(ratios["val"] * n))
    return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]


def split_blocks(block_ids: np.ndarray, label: np.ndarray, ratios: dict, seed: int) -> dict:
    """Mapa block_id -> 'train'|'val'|'test', estratificado por presencia de positivos."""
    rng = np.random.default_rng(seed)
    flat_ids = block_ids.ravel()
    flat_lab = (label.ravel() > 0)
    n_blocks = int(flat_ids.max()) + 1
    pos_per_block = np.bincount(flat_ids, weights=flat_lab, minlength=n_blocks)
    present = np.unique(flat_ids)
    pos_blocks = present[pos_per_block[present] > 0]
    neg_blocks = present[pos_per_block[present] == 0]

    mapping: dict[int, str] = {}
    for group in (pos_blocks, neg_blocks):
        tr, va, te = _partition(group, ratios, rng)
        for b in tr:
            mapping[int(b)] = "train"
        for b in va:
            mapping[int(b)] = "val"
        for b in te:
            mapping[int(b)] = "test"
    return mapping


def build_split_map(block_ids: np.ndarray, mapping: dict) -> np.ndarray:
    """Raster de codigos de split (uint8) por busqueda vectorizada sobre block_ids."""
    lut = np.zeros(int(block_ids.max()) + 1, dtype="uint8")
    for b, name in mapping.items():
        lut[b] = SPLIT_CODES[name]
    return lut[block_ids]


def summarize(split_map: np.ndarray, label: np.ndarray) -> list[tuple[str, int, int, float]]:
    """Conteo de pixeles, positivos y prevalencia por split."""
    rows = []
    for name, code in SPLIT_CODES.items():
        m = split_map == code
        n = int(m.sum())
        pos = int(label[m].sum()) if n else 0
        prev = pos / n if n else 0.0
        rows.append((name, n, pos, prev))
    return rows


def make_split(config: dict, label_path: Path, out_path: Path) -> Path:
    """Construye y guarda la particion por bloques sobre la etiqueta alineada."""
    ratios = config["modeling"]["split_ratios"]
    seed = config["modeling"]["split_seed"]
    block_km = config["aoi"]["block_size_km"]
    pixel_m = config["processing"]["working_resolution_m"]

    with rasterio.open(label_path) as src:
        label = src.read(1)
        profile = src.profile.copy()

    block_px = block_side_px(block_km, pixel_m)
    block_ids = assign_block_ids(label.shape, block_px)
    mapping = split_blocks(block_ids, label, ratios, seed)
    split_map = build_split_map(block_ids, mapping)

    profile.update(count=1, dtype="uint8", nodata=0, compress="deflate")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(split_map, 1)
        dst.set_band_description(1, "split_1train_2val_3test")

    counts = {name: sum(1 for v in mapping.values() if v == name) for name in SPLIT_CODES}
    print("Lado de bloque: " + str(block_px) + " px (" + format(block_km, ".1f") + " km)")
    print("Bloques: " + str(len(mapping)))
    print(f"{'split':<6} {'bloques':>8} {'pixeles':>12} {'positivos':>12} {'prevalencia':>12}")
    for name, n, pos, prev in summarize(split_map, label):
        print(f"{name:<6} {counts[name]:>8} {n:>12,} {pos:>12,} {prev:>11.4%}")
    print("Mapa de particion en " + str(out_path))
    return out_path
