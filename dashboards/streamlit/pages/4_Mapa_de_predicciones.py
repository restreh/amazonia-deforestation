"""Mapa de probabilidad de deforestacion con umbral interactivo."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from utils import read_json, warn_if_missing, PATHS

st.title("Mapa de predicciones")

opciones = []
if PATHS["proba_xgboost"].exists(): opciones.append("xgboost")
if PATHS["proba_unet"].exists(): opciones.append("unet")
if not opciones:
    st.info("Aun no hay rasters de probabilidad. Apareceran al ejecutar predict.py / predict_unet.py.")
    st.stop()

modelo = st.selectbox("Modelo", opciones)
eval_key = "eval_" + modelo
eval_data = read_json(eval_key) if PATHS[eval_key].exists() else None
default_thr = float(eval_data["threshold"]) if eval_data else 0.5
umbral = st.slider("Umbral de decision", 0.0, 1.0, default_thr, 0.01)
st.caption(f"Umbral calibrado en validacion: {default_thr:.3f}")

# TODO: renderizar proba_<modelo>.tif con leafmap (m.add_raster).
# TODO: capa opcional con TP/FP/FN cruzando con label_2024_20m.tif y split_blocks.tif==3 (test).
# TODO: mostrar hectareas predichas por encima del umbral (cada pixel = 0.04 ha).
