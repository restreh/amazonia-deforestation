"""Prediccion densa de la U-Net sobre los bloques de validacion y prueba.

Uso (desde la raiz del repositorio, tras train_unet.py):
    python scripts/predict_unet.py

Recorre el AOI con ventanas de 256x256 solapadas y promedia la probabilidad por pixel
para evitar artefactos de borde. Solo procesa las ventanas que tocan bloques de val o
prueba. Escribe data/processed/predictions/proba_unet.tif, en el mismo formato que la
prediccion del baseline, para que la evaluacion y la comparacion sean directas.

Dependencias: torch, segmentation-models-pytorch, rasterio, numpy, pyyaml.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import rasterio  # noqa: E402
import yaml  # noqa: E402

from amazonia_deforestation.models.patches import channel_spec, read_stack  # noqa: E402
from amazonia_deforestation.models.unet import build_unet  # noqa: E402


def main() -> None:
    import torch
    from rasterio.windows import Window

    config = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    u = config["modeling"]["unet"]
    comp = ROOT / "data" / "processed" / "composites"
    idx = ROOT / "data" / "processed" / "indices"
    split_path = ROOT / "data" / "interim" / "split_blocks.tif"
    ckpt_path = ROOT / "models" / "unet.pt"
    if not ckpt_path.exists():
        print("Falta models/unet.pt. Corre primero scripts/train_unet.py")
        return

    size = u["window_size"]
    stride = size // 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device)
    use_validity = bool(ckpt.get("validity_masks", u.get("validity_masks", False)))
    spec = channel_spec(config, comp, idx, validity_masks=use_validity)
    if len(spec) != ckpt["in_channels"]:
        raise RuntimeError(
            "Inconsistencia de canales: spec=" + str(len(spec))
            + " vs checkpoint=" + str(ckpt["in_channels"])
            + ". Revisa validity_masks en config.yaml.")
    model = build_unet(ckpt["in_channels"], ckpt.get("encoder", u["encoder"]),
                       encoder_weights=None).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with rasterio.open(split_path) as src:
        shape = (src.height, src.width)
        profile = src.profile.copy()
        split = src.read(1)
    H, W = shape

    acc = np.zeros(shape, dtype="float32")
    cnt = np.zeros(shape, dtype="float32")
    eval_codes = (2, 3)
    rows = list(range(0, H - size + 1, stride)) + [H - size]
    cols = list(range(0, W - size + 1, stride)) + [W - size]
    n = 0
    with torch.no_grad():
        for r0 in sorted(set(rows)):
            for c0 in sorted(set(cols)):
                if not np.isin(split[r0:r0 + size, c0:c0 + size], eval_codes).any():
                    continue
                win = Window(c0, r0, size, size)
                image = read_stack(spec, win)
                x = torch.from_numpy(image).unsqueeze(0).to(device)
                prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
                acc[r0:r0 + size, c0:c0 + size] += prob
                cnt[r0:r0 + size, c0:c0 + size] += 1.0
                n += 1

    out = np.full(shape, np.nan, dtype="float32")
    m = cnt > 0
    out[m] = acc[m] / cnt[m]
    profile.update(count=1, dtype="float32", nodata=float("nan"), compress="deflate")
    out_dir = ROOT / "data" / "processed" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "proba_unet.tif"
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out, 1)
        dst.set_band_description(1, "deforestation_probability")
    print("Ventanas predichas: " + str(n) + " | raster en " + str(out_path))


if __name__ == "__main__":
    main()
