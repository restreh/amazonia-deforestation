"""Prediccion densa del baseline sobre los bloques de validacion y prueba.

Uso (desde la raiz del repositorio, tras regenerar features y reentrenar):
    python scripts/predict.py            # XGBoost (por defecto)
    python scripts/predict.py --model random_forest

Recalcula los 612 atributos densamente por franjas (con halo) en los tiles que tocan
bloques de val/test y escribe un raster de probabilidad. 
Correr en la maquina que tiene los datos.

Dependencias: scikit-learn, xgboost, joblib, rasterio, scipy, pandas, pyarrow, pyyaml.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from amazonia_deforestation.models.predict import feature_order, predict_raster  # noqa: E402


def load_model(name, models_dir):
    if name == "xgboost":
        from xgboost import XGBClassifier
        m = XGBClassifier()
        m.load_model(str(models_dir / "xgboost.json"))
        return m
    import joblib
    return joblib.load(models_dir / "random_forest.joblib")


def main() -> None:
    ap = argparse.ArgumentParser(description="Prediccion densa del baseline sobre val/test")
    ap.add_argument("--model", default="xgboost", choices=["xgboost", "random_forest"])
    ap.add_argument("--tile", type=int, default=512, help="lado del tile en pixeles")
    args = ap.parse_args()

    config = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    feats = ROOT / "data" / "processed" / "features" / "train_features.parquet"
    split = ROOT / "data" / "interim" / "split_blocks.tif"
    comp_dir = ROOT / "data" / "processed" / "composites"
    idx_dir = ROOT / "data" / "processed" / "indices"
    models_dir = ROOT / "models"
    for p in (feats, split):
        if not p.exists():
            print("Falta " + str(p) + ". Corre primero los pasos previos.")
            return

    feat_order = feature_order(feats)
    model = load_model(args.model, models_dir)
    out_dir = ROOT / "data" / "processed" / "predictions"
    out_path = out_dir / ("proba_" + args.model + ".tif")
    print("Modelo: " + args.model + " | atributos: " + str(len(feat_order)))
    predict_raster(config, model, feat_order, comp_dir, idx_dir, split, out_path,
                  eval_codes=(2, 3), tile=args.tile)


if __name__ == "__main__":
    main()
