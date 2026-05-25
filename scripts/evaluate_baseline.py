"""Evaluacion honesta del baseline a prevalencia real, sobre los bloques de val/test.

Uso (desde la raiz del repositorio, tras predict.py):
    python scripts/evaluate_baseline.py            # XGBoost (por defecto)
    python scripts/evaluate_baseline.py --model random_forest

Lee el raster de probabilidad, la etiqueta y la particion. Calibra el umbral sobre
los bloques de validacion (maximizando F1) y mide una sola vez sobre los de prueba:
metricas de pixel (precision, recall, F1, IoU, AUC-ROC, AUC-PR) y de poligono
(precision, recall, F1 e IoU medio). Contrasta con los criterios de exito del config.

Dependencias: scikit-learn, scipy, rasterio, numpy, pyyaml.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import rasterio  # noqa: E402
import yaml  # noqa: E402

from amazonia_deforestation.evaluation.metrics import (  # noqa: E402
    calibrate_threshold, pixel_metrics, polygon_metrics,
)

VAL_CODE, TEST_CODE = 2, 3


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluacion del baseline (pixel y poligono)")
    ap.add_argument("--model", default="xgboost", choices=["xgboost", "random_forest"])
    args = ap.parse_args()

    config = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    ev = config["evaluation"]
    proba_path = ROOT / "data" / "processed" / "predictions" / ("proba_" + args.model + ".tif")
    label_path = ROOT / "data" / "interim" / "label_2024_20m.tif"
    split_path = ROOT / "data" / "interim" / "split_blocks.tif"
    for p in (proba_path, label_path, split_path):
        if not p.exists():
            print("Falta " + str(p) + ". Corre primero predict.py")
            return

    with rasterio.open(proba_path) as src:
        proba = src.read(1)
    with rasterio.open(label_path) as src:
        label = src.read(1)
    with rasterio.open(split_path) as src:
        split = src.read(1)

    val = (split == VAL_CODE) & np.isfinite(proba)
    test = (split == TEST_CODE) & np.isfinite(proba)
    print("Pixeles val: " + format(int(val.sum()), ",")
          + " | test: " + format(int(test.sum()), ","))

    thr, f1_val = calibrate_threshold(label[val].astype(np.uint8), proba[val])
    print("Umbral calibrado en validacion: " + format(thr, ".4f")
          + " (F1_val " + format(f1_val, ".4f") + ")")

    px = pixel_metrics(label[test].astype(np.uint8), proba[test], thr)
    print("\n-- Metricas de pixel (prueba, prevalencia real " + format(px["prevalence"], ".2%") + ") --")
    for k in ["precision", "recall", "f1", "iou", "auc_roc", "auc_pr"]:
        print(f"  {k:10s} {px[k]:.4f}")

    pred2d = (proba >= thr) & test
    true2d = (label == 1) & test
    poly = polygon_metrics(pred2d, true2d,
                           iou_threshold=ev["polygon_iou_threshold"],
                           min_area=ev["min_polygon_pixels"],
                           morphology=ev["morphology"])
    print("\n-- Metricas de poligono (prueba, IoU>=" + str(ev["polygon_iou_threshold"]) + ") --")
    for k in ["polygon_precision", "polygon_recall", "polygon_f1", "mean_iou_matched"]:
        print(f"  {k:20s} {poly[k]:.4f}")
    print(f"  parches: pred={poly['n_pred']} true={poly['n_true']} emparejados={poly['tp']}")

    sc = ev["success_criteria"]
    checks = {
        "f1_pixel >= " + str(sc["f1_pixel_min"]): px["f1"] >= sc["f1_pixel_min"],
        "iou_polygon >= " + str(sc["iou_polygon_min"]): (
            poly["mean_iou_matched"] >= sc["iou_polygon_min"]
            if np.isfinite(poly["mean_iou_matched"]) else False),
    }
    print("\n-- Criterios de exito --")
    for k, v in checks.items():
        print("  " + ("CUMPLE  " if v else "NO CUMPLE") + "  " + k)

    report = {"model": args.model, "threshold": thr, "f1_val": f1_val,
              "pixel": px, "polygon": poly, "success": checks}
    out = ROOT / "data" / "interim" / ("eval_" + args.model + ".json")
    out.write_text(json.dumps(report, indent=2))
    print("\nReporte guardado en " + str(out))


if __name__ == "__main__":
    main()
