"""Datos y area de interes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from utils import load_config, read_csv, warn_if_missing, PATHS

st.title("Datos y area de interes")

cfg = load_config()
aoi = cfg["aoi"]; tmp = cfg["temporal"]
st.write(f"AOI: {aoi['name']} | bbox: {aoi['bbox_geographic']}")
st.write(f"Municipios: {', '.join(aoi['municipalities'])}")
st.write(f"Trimestres: {', '.join(q['id'] for q in tmp['composite_quarters'])}")

st.subheader("Mapa del AOI y municipios")
# TODO: renderizar el bbox y los poligonos municipales con leafmap.
warn_if_missing("municipios")

st.subheader("Disponibilidad de Sentinel-2 por trimestre")
if warn_if_missing("disponibilidad"):
    df = read_csv("disponibilidad")
    st.dataframe(df)
    # TODO: barra (plotly) con escenas por trimestre.

st.subheader("Prevalencia de la etiqueta de deforestacion")
# TODO: leer label_2024_20m.tif (rasterio), calcular fraccion de 1s y mostrar st.metric.
warn_if_missing("etiqueta")
