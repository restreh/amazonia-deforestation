"""Tablero del proyecto amazonia-deforestation (pagina de inicio).

Ejecucion desde la raiz del repositorio:
    streamlit run dashboards/streamlit/streamlit_app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
from utils import load_config

st.set_page_config(page_title="Amazonia: deforestacion", layout="wide")
st.title("Deteccion temprana de deforestacion en la Amazonia colombiana")

cfg = load_config()
aoi = cfg["aoi"]; tmp = cfg["temporal"]
st.markdown(
    "Clasificacion binaria por pixel sobre el arco amazonico (Caqueta), con "
    "composiciones trimestrales de Sentinel-2 y Hansen Global Forest Change "
    "como referencia. Este tablero resume datos, diagnosticos y resultados."
)

c1, c2, c3 = st.columns(3)
c1.metric("Ano objetivo", tmp["target_year"])
c2.metric("Area objetivo (km2)", aoi["target_area_km2"])
c3.metric("Bloque de validacion (km)", aoi["block_size_km"])

st.markdown(
    "Navega por las paginas en la barra lateral: Datos y AOI, Diagnostico "
    "espacial, Modelos y metricas, Mapa de predicciones, Por municipio."
)
