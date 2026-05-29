"""Validacion cruzada aleatoria vs espacial sobre los baselines.

Uso (desde la raiz del repositorio, tras build_features.py):
    python scripts/compare_cv.py
    python scripts/compare_cv.py --models xgboost
    python scripts/compare_cv.py --folds 5 --seed 42

Para cada modelo (XGBoost y opcionalmente Random Forest) corre dos esquemas de
validacion cruzada sobre el mismo subconjunto de entrenamiento.

  - Aleatoria, KFold estandar (no controla por dependencia espacial). Es la
    referencia comparable con literatura previa que no controla por dependencia.
  - Espacial, GroupKFold por bloque de 5 km (sin fuga espacial). Es la estimacion
    honesta del desempeno esperado en zonas no observadas.

Reporta el promedio y la desviacion por metrica para cada esquema, y la diferencia
(aleatoria - espacial) como el "optimismo" inducido por la autocorrelacion. Esa
diferencia esta comprometida en la propuesta como un resultado del proyecto.

Salida: data/interim/compare_cv.json y tabla resumen por stdout.

Dependencias: scikit-learn, xgboost, pandas, pyarrow, rasterio, pyyaml.
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
    block_ids, load_features, make_random_forest, make_xgboost,
    pixel_metrics, stratified_indices, subsample,
)

METRICS = ["precision", "recall", "f1", "iou", "auc_roc", "auc_pr"]


def random_cv(make_model, X, y, n_splits, seed, threshold=0.5):
    """Validacion cruzada KFold aleatoria. No controla por dependencia espacial."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, average_precision_score
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = []
    for k, (tr, te) in enumerate(skf.split(X, y)):
        model = make_model()
        model.fit(X[tr], y[tr])
        proba = model.predict_proba(X[te])[:, 1]
        m = pixel_metrics(y[te], proba, threshold)
        m["auc_roc"] = float(roc_auc_score(y[te], proba))
        m["auc_pr"] = float(average_precision_score(y[te], proba))
        m["fold"] = k
        m["n_test"] = int(len(te))
        out.append(m)
    return pd.DataFrame(out)


def spatial_cv(make_model, X, y, groups, n_splits, threshold=0.5):
    """Validacion cruzada por bloques. Sin fuga espacial."""
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score, average_precision_score
    gkf = GroupKFold(n_splits=n_splits)
    out = []
    for k, (tr, te) in enumerate(gkf.split(X, y, groups)):
        model = make_model()
        model.fit(X[tr], y[tr])
        proba = model.predict_proba(X[te])[:, 1]
        m = pixel_metrics(y[te], proba, threshold)
        m["auc_roc"] = float(roc_auc_score(y[te], proba))
        m["auc_pr"] = float(average_precision_score(y[te], proba))
        m["fold"] = k
        m["n_test"] = int(len(te))
        out.append(m)
    return pd.DataFrame(out)


def summarize(df):
    """Promedio y desviacion por metrica."""
    return {k: {"mean": float(df[k].mean()), "std": float(df[k].std())}
            for k in METRICS}


def main() -> None:
    ap = argparse.ArgumentParser(description="CV aleatoria vs espacial sobre baselines")
    ap.add_argument("--models", nargs="+", default=["xgboost"],
                    choices=["xgboost", "random_forest"],
                    help="modelos a comparar (RF es mas lento)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    config = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    feats = ROOT / "data" / "processed" / "features" / "train_features.parquet"
    split_path = ROOT / "data" / "interim" / "split_blocks.tif"
    if not feats.exists():
        print("Falta " + str(feats) + ". Corre primero scripts/build_features.py")
        return

    m = config["modeling"]
    print("Cargando features...")
    df, feat_cols = load_features(feats)
    df = subsample(df, m.get("max_train_rows"), m["split_seed"])
    print("Filas: " + format(len(df), ",") + " | atributos: " + str(len(feat_cols))
          + " | positivos: " + format(int(df["label"].sum()), ","))

    with rasterio.open(split_path) as src:
        shape = (src.height, src.width)
    block_px = block_side_px(config["aoi"]["block_size_km"],
                             config["processing"]["working_resolution_m"])
    groups = block_ids(df["row"].to_numpy(), df["col"].to_numpy(), shape, block_px)
    n_groups = int(np.unique(groups).size)
    print("Bloques distintos en entrenamiento: " + str(n_groups))

    X = df[feat_cols].to_numpy(dtype="float32")
    y = df["label"].to_numpy().astype(np.uint8)
    del df
    gc.collect()
    spw = (len(y) - int(y.sum())) / max(1, int(y.sum()))

    summary = {"folds": args.folds, "seed": args.seed,
               "n_rows": int(len(y)), "n_positives": int(y.sum()),
               "n_blocks": n_groups, "models": {}}

    for name in args.models:
        print("\n=== " + name + " ===")
        if name == "xgboost":
            factory = lambda: make_xgboost(m["xgboost"], scale_pos_weight=spw)
        else:
            rf_idx = stratified_indices(y, m.get("rf_max_rows"), m["split_seed"])
            X_use, y_use, g_use = (X[rf_idx], y[rf_idx], groups[rf_idx]) \
                if len(rf_idx) < len(y) else (X, y, groups)
            factory = lambda: make_random_forest(m["random_forest"])

        if name == "xgboost":
            X_use, y_use, g_use = X, y, groups

        print("Aleatoria (StratifiedKFold)")
        cv_rand = random_cv(factory, X_use, y_use, args.folds, args.seed)
        print(cv_rand[METRICS + ["n_test"]].round(4).to_string(index=False))
        s_rand = summarize(cv_rand)

        print("\nEspacial (GroupKFold por bloque)")
        cv_spat = spatial_cv(factory, X_use, y_use, g_use, args.folds)
        print(cv_spat[METRICS + ["n_test"]].round(4).to_string(index=False))
        s_spat = summarize(cv_spat)

        diffs = {k: float(s_rand[k]["mean"] - s_spat[k]["mean"]) for k in METRICS}
        summary["models"][name] = {"random": s_rand, "spatial": s_spat, "optimism": diffs}

        print("\nOptimismo (aleatoria - espacial)")
        for k in METRICS:
            print("  " + k.ljust(10) + " random " + format(s_rand[k]["mean"], ".4f")
                  + "  espacial " + format(s_spat[k]["mean"], ".4f")
                  + "  diferencia " + format(diffs[k], "+.4f"))

    out = ROOT / "data" / "interim" / "compare_cv.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nReporte guardado en " + str(out))


if __name__ == "__main__":
    main()
