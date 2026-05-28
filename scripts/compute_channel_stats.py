"""Calcula media y desviacion por canal sobre los recortes de entrenamiento.

Uso (desde la raiz del repositorio):
    python scripts/compute_channel_stats.py

Recorre los recortes 256x256 cuyo split objetivo es train y acumula sumas y sumas de
cuadrados por canal, ignorando los pixeles NaN. Las bandas espectrales se escalan por
REFLECTANCE_SCALE antes de acumular (igual que en read_stack). Los canales de validez
no se estandarizan, se les fija media 0 y desviacion 1 para que el escalado quede como
identidad.

Salida: data/interim/channel_stats.json con campos means, stds, n_channels,
validity_masks, channel_names y channel_kinds. Lo consumen train_unet.py y
predict_unet.py cuando modeling.unet.standardize es true en config.yaml.

Dependencias: rasterio, numpy, pyyaml.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import rasterio  # noqa: E402
import yaml  # noqa: E402
from rasterio.windows import Window  # noqa: E402

from amazonia_deforestation.models.patches import (  # noqa: E402
    REFLECTANCE_SCALE, build_patch_index, channel_spec, select_patches)


def main() -> None:
    config = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    u = config["modeling"]["unet"]
    comp = ROOT / "data" / "processed" / "composites"
    idx = ROOT / "data" / "processed" / "indices"
    split = ROOT / "data" / "interim" / "split_blocks.tif"
    size = u["window_size"]
    use_validity = bool(u.get("validity_masks", False))

    spec = channel_spec(config, comp, idx, validity_masks=use_validity)
    records, _ = build_patch_index(split, size, u["patch_stride"])
    tr = select_patches(records, "train", u["min_patch_pixels"])
    print("Recortes train: " + str(len(tr)) + " | canales: " + str(len(spec)))

    n = len(spec)
    sums = np.zeros(n, dtype="float64")
    sqs = np.zeros(n, dtype="float64")
    counts = np.zeros(n, dtype="float64")

    for i, r in enumerate(tr):
        win = Window(r["c0"], r["r0"], size, size)
        for ci, (path, bi, _name, kind) in enumerate(spec):
            if kind == "validity":
                continue
            with rasterio.open(path) as src:
                a = src.read(bi, window=win).astype("float64")
            if kind == "spectral":
                a = a / REFLECTANCE_SCALE
            m = ~np.isnan(a)
            if not m.any():
                continue
            v = a[m]
            sums[ci] += float(v.sum())
            sqs[ci] += float((v * v).sum())
            counts[ci] += float(v.size)
        if (i + 1) % 50 == 0 or (i + 1) == len(tr):
            print("  procesados " + str(i + 1) + " / " + str(len(tr)), flush=True)

    means = np.zeros(n, dtype="float64")
    stds = np.ones(n, dtype="float64")
    for ci, (_p, _b, _name, kind) in enumerate(spec):
        if kind == "validity" or counts[ci] < 1:
            continue
        mu = sums[ci] / counts[ci]
        var = sqs[ci] / counts[ci] - mu * mu
        if var < 1e-8:
            var = 1e-8
        means[ci] = float(mu)
        stds[ci] = float(np.sqrt(var))

    out = {
        "means": means.tolist(),
        "stds": stds.tolist(),
        "n_channels": n,
        "validity_masks": use_validity,
        "channel_names": [s[2] for s in spec],
        "channel_kinds": [s[3] for s in spec],
    }
    out_path = ROOT / "data" / "interim" / "channel_stats.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Stats guardadas en " + str(out_path))


if __name__ == "__main__":
    main()
