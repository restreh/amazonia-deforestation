"""Handler de Lambda para inferencia U-Net sobre una ventana del raster.

El handler espera un evento JSON con la ubicacion S3 del modelo (unet.pt con stats
embebidos), los prefijos S3 de composites e indices del proyecto, la ventana de
trabajo en coordenadas de raster (row0, col0, row1, col1) y el URI de salida. Lee
solo la ventana solicitada usando GDAL/VSI sobre los objetos S3 (sin descargar la
imagen completa), corre la U-Net en CPU y sube el raster de probabilidad al destino.

El modelo se cachea en /tmp tras la primera invocacion para que las llamadas warm
reutilicen el checkpoint.

Evento de ejemplo:
{
  "model_uri": "s3://amazonia-deforestation-data-363918845645/models/unet.pt",
  "composites_prefix": "s3://amazonia-deforestation-data-363918845645/derived/composites/tile=AOI_caqueta/",
  "indices_prefix": "s3://amazonia-deforestation-data-363918845645/derived/indices/tile=AOI_caqueta/",
  "quarters": ["2024Q1", "2024Q2", "2024Q3", "2024Q4"],
  "bbox": [0, 0, 256, 256],
  "output_uri": "s3://amazonia-deforestation-data-363918845645/inference/proba_window_0_0.tif"
}
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import urllib.parse
from pathlib import Path

# Permite importar amazonia_deforestation.* (copiado al lado del handler en la imagen)
sys.path.insert(0, os.path.join(os.environ.get("LAMBDA_TASK_ROOT", "."), "src"))

import boto3  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from rasterio.windows import Window  # noqa: E402

from amazonia_deforestation.models.unet import build_unet  # noqa: E402
from amazonia_deforestation.models.patches import (  # noqa: E402
    N_SPECTRAL, REFLECTANCE_SCALE)

# Configuracion de GDAL para lectura eficiente desde S3
os.environ.setdefault("CPL_VSIL_CURL_USE_HEAD", "NO")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff")
os.environ.setdefault("AWS_REGION", "us-west-2")

MODEL_CACHE = "/tmp/unet.pt"
_model_cache = {"model": None, "means": None, "stds": None, "in_ch": None}
s3 = boto3.client("s3")


def parse_s3(uri):
    assert uri.startswith("s3://"), "URI debe empezar con s3://"
    p = uri[5:].split("/", 1)
    return p[0], p[1]


def vsi_uri(s3_uri):
    """Convierte s3://bucket/key a /vsis3/bucket/key (formato GDAL VSI)."""
    bucket, key = parse_s3(s3_uri)
    return "/vsis3/" + bucket + "/" + key


def ensure_model(model_uri):
    """Descarga el modelo a /tmp si no esta cacheado y lo carga en memoria."""
    import torch
    if _model_cache["model"] is not None:
        return _model_cache
    if not os.path.exists(MODEL_CACHE):
        bucket, key = parse_s3(model_uri)
        print("Descargando modelo " + model_uri)
        s3.download_file(bucket, key, MODEL_CACHE)
    ckpt = torch.load(MODEL_CACHE, map_location="cpu", weights_only=False)
    model = build_unet(ckpt["in_channels"], ckpt.get("encoder", "resnet34"),
                       encoder_weights=None)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    _model_cache["model"] = model
    _model_cache["means"] = np.asarray(ckpt.get("means", []), dtype="float32")
    _model_cache["stds"] = np.asarray(ckpt.get("stds", []), dtype="float32")
    _model_cache["in_ch"] = int(ckpt["in_channels"])
    _model_cache["validity"] = bool(ckpt.get("validity_masks", False))
    print("Modelo cargado | canales " + str(_model_cache["in_ch"]))
    return _model_cache


def read_channels(comp_prefix, idx_prefix, quarters, window, validity):
    """Lee los canales (espectral, indice y validez) en la ventana via VSI/S3."""
    chans = []
    for q in quarters:
        comp_uri = comp_prefix.rstrip("/") + "/quarter=" + q + "/aggregation=median/composite.tif"
        with rasterio.open(vsi_uri(comp_uri)) as src:
            for bi in range(1, N_SPECTRAL + 1):
                a = src.read(bi, window=window).astype("float32") / REFLECTANCE_SCALE
                chans.append(a)
        idx_uri = idx_prefix.rstrip("/") + "/quarter=" + q + "/indices.tif"
        with rasterio.open(vsi_uri(idx_uri)) as src:
            for bi in range(1, src.count + 1):
                chans.append(src.read(bi, window=window).astype("float32"))
    if validity:
        # Una mascara de validez por trimestre basada en NaN de la primera banda espectral
        for q in quarters:
            comp_uri = comp_prefix.rstrip("/") + "/quarter=" + q + "/aggregation=median/composite.tif"
            with rasterio.open(vsi_uri(comp_uri)) as src:
                a = src.read(1, window=window).astype("float32")
            chans.append((~np.isnan(a)).astype("float32"))
    return np.stack(chans, axis=0)


def standardize(image, means, stds):
    """(x - mu) / sigma por canal; sustituye NaN por mu antes de estandarizar."""
    out = image.copy()
    n_no_validity = means.size  # los canales de validez no tienen stats
    for c in range(n_no_validity):
        mu = float(means[c])
        sigma = float(stds[c]) if float(stds[c]) > 0 else 1.0
        ch = out[c]
        ch = np.where(np.isnan(ch), mu, ch)
        out[c] = (ch - mu) / sigma
    # Validez: NaN -> 0
    for c in range(n_no_validity, out.shape[0]):
        out[c] = np.nan_to_num(out[c], nan=0.0)
    return out


def lambda_handler(event, context):
    import torch

    t0 = time.monotonic()
    cache = ensure_model(event["model_uri"])
    model, means, stds = cache["model"], cache["means"], cache["stds"]
    in_ch, validity = cache["in_ch"], cache["validity"]

    row0, col0, row1, col1 = event["bbox"]
    win = Window(col0, row0, col1 - col0, row1 - row0)
    quarters = event.get("quarters", ["2024Q1", "2024Q2", "2024Q3", "2024Q4"])

    image = read_channels(event["composites_prefix"], event["indices_prefix"],
                          quarters, win, validity)
    if image.shape[0] != in_ch:
        return {"statusCode": 400,
                "body": "Canales leidos " + str(image.shape[0]) + " != esperado " + str(in_ch)}
    if means.size > 0:
        image = standardize(image, means, stds)

    with torch.no_grad():
        x = torch.from_numpy(image).unsqueeze(0)
        proba = torch.sigmoid(model(x))[0, 0].numpy()

    # Lee transform y CRS de la primera banda para reproyectar la salida
    comp_uri = event["composites_prefix"].rstrip("/") + "/quarter=" + quarters[0] + "/aggregation=median/composite.tif"
    with rasterio.open(vsi_uri(comp_uri)) as src:
        win_transform = rasterio.windows.transform(win, src.transform)
        crs = src.crs

    out_path = "/tmp/proba_window.tif"
    profile = {
        "driver": "GTiff", "count": 1, "dtype": "float32",
        "width": proba.shape[1], "height": proba.shape[0],
        "crs": crs, "transform": win_transform,
        "compress": "deflate", "nodata": float("nan"),
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(proba, 1)
        dst.set_band_description(1, "deforestation_probability_unet")

    bucket, key = parse_s3(event["output_uri"])
    s3.upload_file(out_path, bucket, key)

    elapsed = time.monotonic() - t0
    return {
        "statusCode": 200,
        "body": json.dumps({
            "output_uri": event["output_uri"],
            "elapsed_seconds": round(elapsed, 3),
            "shape": list(proba.shape),
            "mean_proba": float(proba.mean()),
            "max_proba": float(proba.max()),
        }),
    }
