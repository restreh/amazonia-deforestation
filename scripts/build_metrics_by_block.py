"""Tabla agregada de metricas por bloque espacial y modelo, para Athena.

Uso (desde la raiz del repositorio, tras evaluate_baseline.py de los modelos):
    python scripts/build_metrics_by_block.py
    python scripts/build_metrics_by_block.py --models xgboost unet ensemble random_forest

Para cada bloque del raster split_blocks.tif (train/val/test) y cada modelo evaluado,
agrega los conteos de la matriz de confusion (TP, FP, FN, TN) al umbral calibrado en
validacion, las metricas derivadas (precision, recall, F1, IoU), la prevalencia de la
etiqueta, las hectareas predichas vs Hansen y la probabilidad media. Escribe un
Parquet pequeno con block_id, split_code y model como dimensiones.

Esta es la tabla principal que registra Glue/Athena para consultas analiticas en el
informe (metricas por bloque, prevalencia por particion, hectareas por municipio
agregables si se cruza con un mapa de bloques).

Salida: data/processed/metrics_by_block/part.parquet.

Dependencias: pandas, pyarrow, rasterio, numpy, pyyaml.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
import yaml  # noqa: E402

from amazonia_deforestation.spatial.block_split import (  # noqa: E402
    assign_block_ids, block_side_px)

PIXEL_HA = 0.04   # 20 m x 20 m
SPLIT_CODES = {0: "outside", 1: "train", 2: "val", 3: "test"}


def load_proba_and_threshold(name):
    p = ROOT / "data" / "processed" / "predictions" / ("proba_" + name + ".tif")
    e = ROOT / "data" / "interim" / ("eval_" + name + ".json")
    with rasterio.open(p) as src:
        proba = src.read(1)
    thr = float(json.loads(e.read_text(encoding="utf-8"))["threshold"])
    return proba, thr


def main() -> None:
    ap = argparse.ArgumentParser(description="Tabla de metricas por bloque y modelo")
    ap.add_argument("--models", nargs="+",
                    default=["xgboost", "random_forest", "unet",
                             "ensemble", "unet_imagenet"])
    args = ap.parse_args()

    config = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    label_path = ROOT / "data" / "interim" / "label_2024_20m.tif"
    split_path = ROOT / "data" / "interim" / "split_blocks.tif"

    with rasterio.open(label_path) as src:
        label = src.read(1)
    with rasterio.open(split_path) as src:
        split = src.read(1)
    block_px = block_side_px(config["aoi"]["block_size_km"],
                             config["processing"]["working_resolution_m"])
    bids = assign_block_ids(label.shape, block_px)

    # Bloques validos: los que tienen al menos un pixel de cualquier split
    valid = split > 0
    unique_blocks = np.unique(bids[valid])
    print("Bloques con pixeles asignados: " + str(unique_blocks.size))

    rows = []
    for m in args.models:
        try:
            proba, thr = load_proba_and_threshold(m)
        except FileNotFoundError:
            print("Saltando " + m + " (sin proba o eval)")
            continue
        print("Procesando " + m + " | umbral " + format(thr, ".4f"))

        finite = np.isfinite(proba)
        pred = (proba >= thr) & finite

        for b in unique_blocks:
            mb = (bids == b) & finite
            if not mb.any():
                continue
            split_code = int(np.bincount(split[mb], minlength=4).argmax())
            y = label[mb]
            p = pred[mb]
            tp = int(((p == 1) & (y == 1)).sum())
            fp = int(((p == 1) & (y == 0)).sum())
            fn = int(((p == 0) & (y == 1)).sum())
            tn = int(((p == 0) & (y == 0)).sum())
            n = tp + fp + fn + tn
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            mean_proba = float(proba[mb].mean())
            rows.append({
                "block_id": int(b),
                "split_code": SPLIT_CODES.get(split_code, "outside"),
                "model": m, "threshold": thr,
                "n_pixels": n,
                "n_positives": int((y == 1).sum()),
                "prevalence": float((y == 1).mean()),
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": prec, "recall": rec, "f1": f1, "iou": iou,
                "mean_proba": mean_proba,
                "predicted_ha": float(p.sum()) * PIXEL_HA,
                "truth_ha": float((y == 1).sum()) * PIXEL_HA,
            })

    df = pd.DataFrame(rows)
    out_dir = ROOT / "data" / "processed" / "metrics_by_block"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "part.parquet"
    df.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
    print("Filas: " + format(len(df), ",")
          + " | modelos: " + str(df["model"].nunique())
          + " | bloques: " + str(df["block_id"].nunique()))
    print("Salida: " + str(out_path)
          + " (" + format(out_path.stat().st_size / 1024, ".1f") + " KB)")


if __name__ == "__main__":
    main()
