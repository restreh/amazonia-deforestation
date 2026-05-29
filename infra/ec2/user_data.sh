#!/usr/bin/env bash
# Script de bootstrap que corre como root al arrancar la instancia t3.medium.
# Instala dependencias minimas, descarga modelo y composites desde S3, corre la
# inferencia U-Net densa sobre todo el AOI midiendo wall-time, publica el resultado
# en s3://bucket/benchmarks/ y termina la instancia.
set -euxo pipefail

BUCKET="amazonia-deforestation-data-363918845645"
AWS_REGION="us-west-2"
WORKDIR="/opt/amazonia"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULT_KEY="benchmarks/benchmark_${TIMESTAMP}.json"
LOG_KEY="benchmarks/benchmark_${TIMESTAMP}.log"

exec > >(tee /var/log/benchmark.log) 2>&1
echo "== Inicio del benchmark $TIMESTAMP =="

# GDAL no viene en los repos por defecto de AL2023; las wheels de rasterio traen su
# propio GDAL embebido, asi que no se instala a nivel de sistema.
dnf install -y python3.12 python3.12-pip gcc git tar gzip
ln -sf /usr/bin/python3.12 /usr/local/bin/python
ln -sf /usr/bin/pip3.12 /usr/local/bin/pip

mkdir -p "$WORKDIR" && cd "$WORKDIR"

pip install --upgrade pip
pip install --no-cache-dir \
    "numpy==1.26.4" "rasterio==1.4.3" "pyyaml==6.0.2" "boto3==1.35.50" \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "torch==2.4.1+cpu" "segmentation-models-pytorch==0.3.4"

echo "== Descargando modelo y rasters =="
aws s3 cp "s3://$BUCKET/models/unet.pt" "$WORKDIR/unet.pt"
aws s3 sync "s3://$BUCKET/derived/composites/" "$WORKDIR/composites/" --no-progress
aws s3 sync "s3://$BUCKET/derived/indices/" "$WORKDIR/indices/" --no-progress
aws s3 cp "s3://$BUCKET/derived/interim/split_blocks.tif" "$WORKDIR/split_blocks.tif"
aws s3 cp "s3://$BUCKET/derived/interim/label_2024_20m.tif" "$WORKDIR/label.tif"

cat > "$WORKDIR/benchmark.py" <<'PY'
"""Inferencia U-Net densa sobre el AOI con timing."""
import json, os, time, sys
import numpy as np
import rasterio
from rasterio.windows import Window
import torch
import segmentation_models_pytorch as smp

W = "/opt/amazonia"
QUARTERS = ["2024Q1", "2024Q2", "2024Q3", "2024Q4"]
N_SPECTRAL = 10
REFLECTANCE_SCALE = 10000.0
WINDOW = 256
STRIDE = 128

def build_unet(in_channels, encoder, encoder_weights=None, classes=1):
    return smp.Unet(encoder_name=encoder, encoder_weights=encoder_weights,
                    in_channels=in_channels, classes=classes, activation=None)

print("Cargando modelo")
ckpt = torch.load(os.path.join(W, "unet.pt"), map_location="cpu", weights_only=False)
in_ch = int(ckpt["in_channels"])
validity = bool(ckpt.get("validity_masks", False))
model = build_unet(in_ch, ckpt.get("encoder", "resnet34"))
model.load_state_dict(ckpt["state_dict"])
model.eval()
means = np.asarray(ckpt.get("means", []), dtype="float32")
stds = np.asarray(ckpt.get("stds", []), dtype="float32")

def quarter_paths(q):
    return (os.path.join(W, f"composites/tile=AOI_caqueta/quarter={q}/aggregation=median/composite.tif"),
            os.path.join(W, f"indices/tile=AOI_caqueta/quarter={q}/indices.tif"))

# Forma de la grilla via split_blocks
with rasterio.open(os.path.join(W, "split_blocks.tif")) as src:
    H, W_ = src.height, src.width
    profile = src.profile.copy()
print(f"Grilla {H}x{W_}")

def read_window(win):
    chans = []
    for q in QUARTERS:
        cp, ip = quarter_paths(q)
        with rasterio.open(cp) as src:
            for bi in range(1, N_SPECTRAL + 1):
                a = src.read(bi, window=win).astype("float32") / REFLECTANCE_SCALE
                chans.append(a)
        with rasterio.open(ip) as src:
            for bi in range(1, src.count + 1):
                chans.append(src.read(bi, window=win).astype("float32"))
    if validity:
        for q in QUARTERS:
            cp, _ = quarter_paths(q)
            with rasterio.open(cp) as src:
                a = src.read(1, window=win).astype("float32")
            chans.append((~np.isnan(a)).astype("float32"))
    img = np.stack(chans, axis=0)
    if means.size > 0:
        n = means.size
        for c in range(n):
            mu = float(means[c]); sigma = float(stds[c]) if float(stds[c]) > 0 else 1.0
            ch = img[c]
            ch = np.where(np.isnan(ch), mu, ch)
            img[c] = (ch - mu) / sigma
        for c in range(n, img.shape[0]):
            img[c] = np.nan_to_num(img[c], nan=0.0)
    return img

acc = np.zeros((H, W_), dtype="float32")
cnt = np.zeros((H, W_), dtype="float32")
rows = list(range(0, H - WINDOW + 1, STRIDE)) + [H - WINDOW]
cols = list(range(0, W_ - WINDOW + 1, STRIDE)) + [W_ - WINDOW]
rows = sorted(set(rows)); cols = sorted(set(cols))
n_windows = len(rows) * len(cols)
print(f"Ventanas: {n_windows}")

t0 = time.monotonic()
with torch.no_grad():
    for i, r0 in enumerate(rows):
        for c0 in cols:
            win = Window(c0, r0, WINDOW, WINDOW)
            x = read_window(win)
            xt = torch.from_numpy(x).unsqueeze(0)
            p = torch.sigmoid(model(xt))[0, 0].numpy()
            acc[r0:r0+WINDOW, c0:c0+WINDOW] += p
            cnt[r0:r0+WINDOW, c0:c0+WINDOW] += 1.0
        print(f"  fila {i+1}/{len(rows)}  t={time.monotonic()-t0:.1f}s", flush=True)

out = np.full((H, W_), np.nan, dtype="float32")
m = cnt > 0
out[m] = acc[m] / cnt[m]
profile.update(count=1, dtype="float32", nodata=float("nan"), compress="deflate")
with rasterio.open(os.path.join(W, "proba_unet_bench.tif"), "w", **profile) as dst:
    dst.write(out, 1)

elapsed = time.monotonic() - t0
result = {
    "instance_type": "t3.medium", "model": "unet",
    "windows": n_windows, "grid_h": H, "grid_w": W_,
    "elapsed_seconds": round(elapsed, 2),
    "elapsed_minutes": round(elapsed / 60.0, 2),
    "under_10_minutes": elapsed < 600,
}
print(json.dumps(result, indent=2))
with open(os.path.join(W, "result.json"), "w") as f:
    json.dump(result, f, indent=2)
PY

echo "== Corriendo benchmark =="
python "$WORKDIR/benchmark.py"

echo "== Publicando resultado =="
aws s3 cp "$WORKDIR/result.json" "s3://$BUCKET/$RESULT_KEY"
aws s3 cp "$WORKDIR/proba_unet_bench.tif" "s3://$BUCKET/inference/proba_unet_t3medium_${TIMESTAMP}.tif"
aws s3 cp /var/log/benchmark.log "s3://$BUCKET/$LOG_KEY"

echo "== Terminando instancia =="
shutdown -h now
