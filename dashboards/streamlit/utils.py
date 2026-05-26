"""Utilidades compartidas del tablero: rutas, cargadores tolerantes a archivos faltantes."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]   # dashboards/streamlit/utils.py -> raiz del repo
DATA = ROOT / "data"
PATHS = {
    "config":          ROOT / "config" / "config.yaml",
    "municipios":      DATA / "external" / "aoi_municipalities.geojson",
    "etiqueta":        DATA / "interim" / "label_2024_20m.tif",
    "split":           DATA / "interim" / "split_blocks.tif",
    "diagnosticos":    DATA / "interim" / "spatial_diagnostics.txt",
    "disponibilidad":  DATA / "interim" / "s2_availability_summary.csv",
    "cv_summary":      DATA / "interim" / "baseline_cv_summary.json",
    "eval_xgboost":    DATA / "interim" / "eval_xgboost.json",
    "eval_unet":       DATA / "interim" / "eval_unet.json",
    "cv_xgboost":      DATA / "interim" / "cv_xgboost.csv",
    "cv_rf":           DATA / "interim" / "cv_random_forest.csv",
    "imp_xgboost":     DATA / "interim" / "importance_xgboost.csv",
    "imp_rf":          DATA / "interim" / "importance_random_forest.csv",
    "proba_xgboost":   DATA / "processed" / "predictions" / "proba_xgboost.tif",
    "proba_unet":      DATA / "processed" / "predictions" / "proba_unet.tif",
}


def warn_if_missing(key: str) -> bool:
    p = PATHS[key]
    if not p.exists():
        st.info(f"Aun no esta disponible {p.relative_to(ROOT)}; este panel se completa cuando exista.")
        return False
    return True


@st.cache_data(show_spinner=False)
def load_config() -> dict:
    import yaml
    return yaml.safe_load(PATHS["config"].read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def read_json(key: str):
    p = PATHS[key]
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@st.cache_data(show_spinner=False)
def read_csv(key: str):
    import pandas as pd
    p = PATHS[key]
    return pd.read_csv(p) if p.exists() else None
