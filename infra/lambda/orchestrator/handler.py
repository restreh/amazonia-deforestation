"""Handler de Lambda de orquestacion. Disparado por EventBridge.

Recorre la grilla del AOI y dispara el Lambda de inferencia U-Net (asincrono) para
cada ventana. Tras la lluvia de invocaciones, las predicciones por ventana quedan en
s3://<bucket>/inference/scheduled/<timestamp>/proba_<row0>_<col0>.tif. Un paso
posterior fuera de Lambda (mosaiqueado) las une si se desea un raster denso.

Evento EventBridge tipico (cron trimestral):
{
  "quarter": "2024Q4",     # opcional; si no, se calcula del trimestre vigente
  "stride": 128,           # opcional; por defecto media ventana
  "window": 256            # opcional
}

Variables de entorno requeridas:
  PROJECT_BUCKET        nombre del bucket S3 del proyecto
  INFERENCE_FUNCTION    nombre del Lambda de inferencia
  GRID_HEIGHT           alto del raster del AOI en pixeles (por defecto 3533)
  GRID_WIDTH            ancho del raster del AOI en pixeles (por defecto 3556)
"""

from __future__ import annotations

import datetime as dt
import json
import os

import boto3

lambda_client = boto3.client("lambda")

BUCKET = os.environ.get("PROJECT_BUCKET")
INFERENCE_FN = os.environ.get("INFERENCE_FUNCTION", "amazonia-deforestation-unet-inference")
H = int(os.environ.get("GRID_HEIGHT", "3533"))
W = int(os.environ.get("GRID_WIDTH", "3556"))


def current_quarter():
    """Devuelve el id del trimestre vigente, p.ej. 2024Q4."""
    today = dt.date.today()
    q = (today.month - 1) // 3 + 1
    return f"{today.year}Q{q}"


def grid_windows(h, w, window, stride):
    rows = list(range(0, h - window + 1, stride)) + [h - window]
    cols = list(range(0, w - window + 1, stride)) + [w - window]
    rows = sorted(set(rows))
    cols = sorted(set(cols))
    return [(r, c) for r in rows for c in cols]


def lambda_handler(event, context):
    if not BUCKET:
        return {"statusCode": 500, "body": "Falta PROJECT_BUCKET"}

    quarter = event.get("quarter") or current_quarter()
    window = int(event.get("window", 256))
    stride = int(event.get("stride", 128))
    timestamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    windows = grid_windows(H, W, window, stride)

    out_prefix = f"inference/scheduled/{timestamp}/"
    invocations = 0
    for r0, c0 in windows:
        payload = {
            "model_uri": f"s3://{BUCKET}/models/unet.pt",
            "composites_prefix": f"s3://{BUCKET}/derived/composites/tile=AOI_caqueta/",
            "indices_prefix": f"s3://{BUCKET}/derived/indices/tile=AOI_caqueta/",
            "quarters": [quarter],
            "bbox": [r0, c0, r0 + window, c0 + window],
            "output_uri": f"s3://{BUCKET}/{out_prefix}proba_{r0:04d}_{c0:04d}.tif",
        }
        lambda_client.invoke(
            FunctionName=INFERENCE_FN,
            InvocationType="Event",     # asincrono
            Payload=json.dumps(payload).encode("utf-8"),
        )
        invocations += 1

    return {
        "statusCode": 200,
        "body": json.dumps({
            "quarter": quarter,
            "windows": len(windows),
            "invocations": invocations,
            "output_prefix": f"s3://{BUCKET}/{out_prefix}",
        }),
    }
