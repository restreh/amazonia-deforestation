"""Barrido de pesos del ensamble XGBoost + U-Net y reporte comparativo.

Uso (desde la raiz del repositorio, tras predict.py y predict_unet.py):
    python scripts/sweep_ensemble.py
    python scripts/sweep_ensemble.py --weights 0.3 0.4 0.5 0.6 0.7

Para cada peso de XGBoost en la lista (el peso del U-Net se calcula como 1 - w_xgb),
construye proba_ensemble.tif, lo evalua con evaluate_baseline.py y guarda el reporte
en data/interim/eval_ensemble_wXX.json. Imprime una tabla resumen con las metricas
clave para decidir el peso optimo. La ultima ejecucion deja proba_ensemble.tif con la
combinacion que maximiza F1_val.

Dependencias: rasterio, numpy, pyyaml, scikit-learn (las usan los scripts invocados).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd):
    """Ejecuta un comando y suprime stdout, deja pasar los errores."""
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(res.stdout)
        sys.stderr.write(res.stderr)
        raise RuntimeError("Fallo el comando: " + " ".join(cmd))


def main() -> None:
    ap = argparse.ArgumentParser(description="Barrido de pesos del ensamble XGBoost + U-Net")
    ap.add_argument("--weights", nargs="+", type=float,
                    default=[0.3, 0.4, 0.5, 0.6, 0.7],
                    help="lista de pesos para XGBoost (el resto va al U-Net)")
    args = ap.parse_args()

    py = sys.executable
    interim = ROOT / "data" / "interim"
    eval_path = interim / "eval_ensemble.json"
    results = []

    for w_xgb in args.weights:
        w_unet = round(1.0 - float(w_xgb), 4)
        print("\n=== w_xgb=" + format(w_xgb, ".2f")
              + " | w_unet=" + format(w_unet, ".2f") + " ===")
        run([py, "scripts/build_ensemble.py",
             "--w-xgboost", str(w_xgb), "--w-unet", str(w_unet)])
        run([py, "scripts/evaluate_baseline.py", "--model", "ensemble"])
        rep = json.loads(eval_path.read_text(encoding="utf-8"))
        suffix = "w" + format(int(round(w_xgb * 100)), "02d")
        shutil.copy(eval_path, interim / ("eval_ensemble_" + suffix + ".json"))
        results.append({
            "w_xgb": w_xgb, "w_unet": w_unet,
            "threshold": rep["threshold"], "f1_val": rep["f1_val"],
            "f1_pixel": rep["pixel"]["f1"], "auc_pr": rep["pixel"]["auc_pr"],
            "precision_pixel": rep["pixel"]["precision"],
            "recall_pixel": rep["pixel"]["recall"],
            "polygon_f1": rep["polygon"]["polygon_f1"],
            "polygon_precision": rep["polygon"]["polygon_precision"],
            "polygon_recall": rep["polygon"]["polygon_recall"],
            "mean_iou_matched": rep["polygon"]["mean_iou_matched"],
            "tp": rep["polygon"]["tp"], "n_pred": rep["polygon"]["n_pred"],
        })

    # Tabla resumen
    cols = ["w_xgb", "w_unet", "thr", "f1_val", "f1_px", "auc_pr",
            "prec_px", "rec_px", "poly_f1", "poly_prec", "poly_rec",
            "iou_match", "tp", "n_pred"]
    print("\n=== Resumen del barrido ===")
    print(" | ".join(c.rjust(9) for c in cols))
    for r in results:
        row = [format(r["w_xgb"], ".2f"), format(r["w_unet"], ".2f"),
               format(r["threshold"], ".4f"), format(r["f1_val"], ".4f"),
               format(r["f1_pixel"], ".4f"), format(r["auc_pr"], ".4f"),
               format(r["precision_pixel"], ".4f"), format(r["recall_pixel"], ".4f"),
               format(r["polygon_f1"], ".4f"), format(r["polygon_precision"], ".4f"),
               format(r["polygon_recall"], ".4f"), format(r["mean_iou_matched"], ".4f"),
               str(r["tp"]), str(r["n_pred"])]
        print(" | ".join(v.rjust(9) for v in row))

    best = max(results, key=lambda r: r["f1_val"])
    print("\nMejor por f1_val: w_xgb=" + format(best["w_xgb"], ".2f")
          + " | w_unet=" + format(best["w_unet"], ".2f")
          + " | f1_val=" + format(best["f1_val"], ".4f")
          + " | f1_test=" + format(best["f1_pixel"], ".4f")
          + " | auc_pr=" + format(best["auc_pr"], ".4f")
          + " | polygon_f1=" + format(best["polygon_f1"], ".4f"))

    # Dejar proba_ensemble.tif con la combinacion ganadora
    run([py, "scripts/build_ensemble.py",
         "--w-xgboost", str(best["w_xgb"]), "--w-unet", str(best["w_unet"])])
    run([py, "scripts/evaluate_baseline.py", "--model", "ensemble"])
    print("\nproba_ensemble.tif y eval_ensemble.json corresponden ahora al peso ganador.")


if __name__ == "__main__":
    main()
