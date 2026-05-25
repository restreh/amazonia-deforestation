"""Entrena los baseline (Random Forest y XGBoost) con validacion cruzada espacial.

Uso (desde la raiz del repositorio, tras build_features.py):
    python scripts/train_baselines.py                 # ambos modelos
    python scripts/train_baselines.py --models rf      # solo Random Forest
    python scripts/train_baselines.py --models xgb     # solo XGBoost

Ambos modelos entrenan sobre el mismo conjunto (config max_train_rows = rf_max_rows)
para una comparacion pareada justa. Cada modelo guarda sus metricas de CV en disco
(data/interim/cv_<modelo>.csv) apenas las calcula; el resumen se arma con lo que haya,
de modo que correr un modelo despues no borra los resultados del otro.

Registra en MLflow si esta instalado. 
Dependencias: scikit-learn, xgboost, pandas, pyarrow, rasterio, pyyaml (mlflow opcional).
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
import yaml  # noqa: E402

from amazonia_deforestation.spatial.block_split import block_side_px  # noqa: E402
from amazonia_deforestation.models.baselines import (  # noqa: E402
    feature_importance, load_features, block_ids, make_random_forest,
    make_xgboost, spatial_cv, stratified_indices, subsample,
)

METRICS = ["precision", "recall", "f1", "iou", "auc_roc", "auc_pr"]
ALIASES = {"rf": "random_forest", "xgb": "xgboost"}


class _NoMlflow:
    """Sustituto inerte si MLflow no esta instalado."""
    def set_tracking_uri(self, *a, **k): pass
    def set_experiment(self, *a, **k): pass
    def log_params(self, *a, **k): pass
    def log_param(self, *a, **k): pass
    def log_metric(self, *a, **k): pass
    def start_run(self, *a, **k):
        import contextlib
        return contextlib.nullcontext()


def write_summary(out_dir):
    """Arma baseline_cv_summary.json con los cv_<modelo>.csv presentes."""
    summary = {}
    for name in ("xgboost", "random_forest"):
        f = out_dir / ("cv_" + name + ".csv")
        if f.exists():
            cv = pd.read_csv(f)
            summary[name] = {k: float(cv[k].mean()) for k in METRICS}
    (out_dir / "baseline_cv_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_one(name, factory, params, X, y, groups, folds, mlflow, models_dir, out_dir, feat_cols):
    """Corre CV espacial, ajuste final, importancias y guardado para un modelo."""
    print("\n=== " + name + " : CV espacial " + str(folds) + " pliegues ===")
    print("Filas: " + format(len(y), ",") + " | positivos: " + format(int(y.sum()), ","))
    with mlflow.start_run(run_name=name):
        mlflow.log_params({str(k): v for k, v in params.items()})
        mlflow.log_param("n_rows", int(len(y)))
        cv = spatial_cv(factory, X, y, groups, folds)
        cv.to_csv(out_dir / ("cv_" + name + ".csv"), index=False)
        print(cv.round(4).to_string(index=False))
        for met in METRICS:
            mlflow.log_metric(met + "_mean", float(cv[met].mean()))
            mlflow.log_metric(met + "_std", float(cv[met].std()))
        print("media -> " + "  ".join(met + " " + format(cv[met].mean(), ".4f") for met in METRICS))

        final = factory()
        final.fit(X, y)
        imp = feature_importance(final, feat_cols, name)
        imp.to_csv(out_dir / ("importance_" + name + ".csv"), index=False)
        if name == "xgboost":
            final.save_model(str(models_dir / "xgboost.json"))
        else:
            import joblib
            joblib.dump(final, models_dir / "random_forest.joblib")
        print("Top 10 atributos: " + ", ".join(imp["feature"].head(10)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Entrena baseline RF/XGBoost con CV espacial")
    ap.add_argument("--models", default="all",
                    help="all | xgboost | random_forest (alias: xgb, rf)")
    args = ap.parse_args()
    requested = ["xgboost", "random_forest"] if args.models == "all" \
        else [ALIASES.get(args.models, args.models)]

    config = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    feats = ROOT / "data" / "processed" / "features" / "train_features.parquet"
    split = ROOT / "data" / "interim" / "split_blocks.tif"
    if not feats.exists():
        print("Falta train_features.parquet. Corre primero scripts/build_features.py")
        return

    m = config["modeling"]
    print("Cargando features...")
    df, feat_cols = load_features(feats)
    df = subsample(df, m.get("max_train_rows"), m["split_seed"])
    print("Filas: " + format(len(df), ",") + " | atributos: " + str(len(feat_cols))
          + " | positivos: " + format(int(df["label"].sum()), ","))

    with rasterio.open(split) as src:
        shape = (src.height, src.width)
    block_px = block_side_px(config["aoi"]["block_size_km"],
                             config["processing"]["working_resolution_m"])
    groups = block_ids(df["row"].to_numpy(), df["col"].to_numpy(), shape, block_px)
    print("Bloques distintos en entrenamiento: " + str(len(np.unique(groups))))

    X = df[feat_cols].to_numpy(dtype="float32")
    y = df["label"].to_numpy().astype(np.uint8)
    del df
    gc.collect()
    spw = (len(y) - int(y.sum())) / max(1, int(y.sum()))
    folds = m.get("cv_folds", 5)

    try:
        import mlflow
        mlflow.set_tracking_uri(m["mlflow_tracking_uri"])
        mlflow.set_experiment("baselines")
    except Exception:
        print("MLflow no disponible; se omite el registro.")
        mlflow = _NoMlflow()

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    out_dir = ROOT / "data" / "interim"

    if "xgboost" in requested:
        run_one("xgboost", lambda: make_xgboost(m["xgboost"], scale_pos_weight=spw),
                m["xgboost"], X, y, groups, folds, mlflow, models_dir, out_dir, feat_cols)

    if "random_forest" in requested:
        rf_idx = stratified_indices(y, m.get("rf_max_rows"), m["split_seed"])
        if len(rf_idx) == len(y):
            X_rf, y_rf, g_rf = X, y, groups          # mismo conjunto: sin copia extra
        else:
            X_rf, y_rf, g_rf = X[rf_idx], y[rf_idx], groups[rf_idx]
            del X
            gc.collect()
        run_one("random_forest", lambda: make_random_forest(m["random_forest"]),
                m["random_forest"], X_rf, y_rf, g_rf, folds, mlflow, models_dir, out_dir, feat_cols)

    summary = write_summary(out_dir)
    print("\nResumen CV (medias):\n" + json.dumps(summary, indent=2))
    print("Modelos en " + str(models_dir) + " | metricas e importancias en " + str(out_dir))


if __name__ == "__main__":
    main()
