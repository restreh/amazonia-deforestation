"""Entrenamiento de los baseline (Random Forest y XGBoost) por pixel.

Entrena sobre la tabla de features de entrenamiento con validacion cruzada
espacial por bloques: los pliegues se forman agrupando por bloque (GroupKFold),
de modo que ningun bloque aparece a la vez en ajuste y evaluacion. Esto evita la
fuga por autocorrelacion espacial que inflaria las metricas (I de Moran ~0.84).

Random Forest no admite NaN; se imputa por mediana con un transformador propio que
calcula la mediana columna a columna, sin el pico de memoria del imputador de
scikit-learn (que ordena con arreglos enmascarados). XGBoost maneja NaN de forma
nativa y no se imputa.

Las metricas a nivel de pixel se calculan sobre el pliegue retenido. Nota: la tabla
de entrenamiento esta balanceada (~9% de positivos), asi que F1/IoU aqui son
relativos y sirven para comparar modelos y atributos; las metricas a prevalencia
real se obtienen luego sobre los bloques de val/test (prediccion densa).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

EXCLUDE = ("row", "col", "label")


class MedianImputer(BaseEstimator, TransformerMixin):
    """Imputacion por mediana, calculada columna a columna (memoria acotada)."""

    def fit(self, X, y=None):
        n = X.shape[1]
        med = np.empty(n, dtype=np.float64)
        for j in range(n):
            col = X[:, j]
            v = col[~np.isnan(col)]
            med[j] = np.median(v) if v.size else 0.0
        self.medians_ = med
        return self

    def transform(self, X):
        X = X.copy()
        for j in range(X.shape[1]):
            col = X[:, j]
            m = np.isnan(col)
            if m.any():
                col[m] = self.medians_[j]
        return X


def load_features(path):
    """Carga la tabla y devuelve (DataFrame, lista de columnas de atributos)."""
    df = pd.read_parquet(path)
    feat_cols = [c for c in df.columns if c not in EXCLUDE]
    return df, feat_cols


def block_ids(rows, cols, raster_shape, block_px):
    """Id de bloque por pixel a partir de (row, col), igual que en la particion."""
    n_cols = int(np.ceil(raster_shape[1] / block_px))
    return (rows // block_px) * n_cols + (cols // block_px)


def subsample(df, max_rows, seed):
    """Submuestra filas de un DataFrame preservando la proporcion de clases."""
    if max_rows is None or len(df) <= max_rows:
        return df
    frac = max_rows / len(df)
    return (df.groupby("label", group_keys=False)
              .sample(frac=frac, random_state=seed)
              .reset_index(drop=True))


def stratified_indices(y, max_rows, seed):
    """Indices de una submuestra que preserva la proporcion de clases."""
    if max_rows is None or len(y) <= max_rows:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    frac = max_rows / len(y)
    parts = []
    for cls in np.unique(y):
        ci = np.flatnonzero(y == cls)
        k = max(1, int(round(len(ci) * frac)))
        parts.append(rng.choice(ci, size=k, replace=False))
    out = np.concatenate(parts)
    rng.shuffle(out)
    return out


def pixel_metrics(y_true, proba, threshold):
    """Precision, recall, F1 e IoU a un umbral dado."""
    from sklearn.metrics import precision_recall_fscore_support, jaccard_score
    yhat = (proba >= threshold).astype(np.uint8)
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_true, yhat, average="binary", zero_division=0)
    iou = jaccard_score(y_true, yhat, zero_division=0)
    return {"precision": float(pr), "recall": float(rc), "f1": float(f1), "iou": float(iou)}


def make_random_forest(params):
    """Pipeline imputacion por mediana + Random Forest (sklearn no admite NaN)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    rf = RandomForestClassifier(
        n_estimators=params.get("n_estimators", 300),
        max_depth=params.get("max_depth"),
        max_features=params.get("max_features", "sqrt"),
        min_samples_leaf=params.get("min_samples_leaf", 50),
        max_samples=params.get("max_samples", 0.5),
        class_weight=params.get("class_weight", "balanced_subsample"),
        n_jobs=params.get("n_jobs", -1),
        random_state=42,
    )
    return Pipeline([("impute", MedianImputer()), ("rf", rf)])


def make_xgboost(params, scale_pos_weight=1.0):
    """Clasificador XGBoost con manejo nativo de NaN."""
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=params.get("n_estimators", 400),
        max_depth=params.get("max_depth", 6),
        learning_rate=params.get("learning_rate", 0.1),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        tree_method=params.get("tree_method", "hist"),
        n_jobs=params.get("n_jobs", -1),
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
    )


def spatial_cv(make_model, X, y, groups, n_splits, threshold=0.5):
    """Validacion cruzada por bloques. Devuelve DataFrame de metricas por pliegue."""
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


def feature_importance(model, feat_cols, kind):
    """Importancia de atributos ordenada (RF: impureza; XGB: ganancia)."""
    if kind == "random_forest":
        imp = model.named_steps["rf"].feature_importances_
    else:
        imp = model.feature_importances_
    return (pd.DataFrame({"feature": feat_cols, "importance": imp})
              .sort_values("importance", ascending=False)
              .reset_index(drop=True))
