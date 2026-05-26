"""Modelos y metricas: baseline (RF, XGBoost) y U-Net cuando exista."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
from utils import load_config, read_csv, read_json, warn_if_missing

st.title("Modelos y metricas")

cfg = load_config()
sc = cfg["evaluation"]["success_criteria"]

st.subheader("Comparacion por validacion cruzada espacial (medias)")
if warn_if_missing("cv_summary"):
    summary = read_json("cv_summary")
    st.dataframe(pd.DataFrame(summary).T)

st.subheader("Metricas a prevalencia real en prueba")
cols = st.columns(2)
for col, key, name in [(cols[0], "eval_xgboost", "XGBoost (baseline)"),
                       (cols[1], "eval_unet", "U-Net")]:
    with col:
        st.markdown(f"**{name}**")
        if warn_if_missing(key):
            data = read_json(key)
            st.json(data.get("pixel", {}), expanded=False)
            st.json(data.get("polygon", {}), expanded=False)
            # TODO: comparar con sc (f1_pixel_min, iou_polygon_min) y mostrar CUMPLE/NO CUMPLE.

st.subheader("Importancia de atributos (XGBoost)")
if warn_if_missing("imp_xgboost"):
    imp = read_csv("imp_xgboost").head(20)
    st.dataframe(imp)
    # TODO: barra horizontal (plotly) con los 20 atributos mas importantes.
