"""Agregaciones por municipio (zonal stats)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from utils import warn_if_missing

st.title("Por municipio")

st.markdown(
    "Hectareas detectadas por el modelo vs Hansen, por municipio. Util para "
    "comparar la deteccion con el reporte institucional del SMByC-IDEAM."
)

# TODO: zonal stats por municipio (rasterstats o rasterio.mask) sobre
# label_2024_20m.tif (Hansen) y proba_<modelo>.tif binarizado al umbral calibrado.
# TODO: tabla ordenable + choropleth (plotly o leafmap).

warn_if_missing("municipios")
warn_if_missing("etiqueta")
warn_if_missing("proba_xgboost")
