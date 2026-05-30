"""Tablero del proyecto amazonia-deforestation (pagina de inicio).

Ejecucion desde la raiz del repositorio:
    streamlit run dashboards/streamlit/streamlit_app.py
"""
import os
import sys
from pathlib import Path

# ── Fix PROJ version conflict (Windows) ────────────────────────────────────────
# rasterio 1.4+ bundlea PROJ 9.6 (DATABASE.LAYOUT.VERSION.MINOR=5).
# pyproj 3.6/3.7 bundlea PROJ 9.3–9.5 (MINOR=2–4). Al importarse, pyproj
# sobreescribe PROJ_DATA con su directorio (MINOR demasiado bajo), haciendo
# que rasterio falle con CRSError al intentar resolver cualquier EPSG.
# Solución: apuntar tanto los env vars como el contexto interno de pyproj al
# proj_data de rasterio (MINOR=5) antes de cualquier operación de CRS.
try:
    import rasterio as _rio
    import pyproj as _pp
    _proj_dir = str((Path(_rio.__file__).parent / "proj_data").resolve())
    if (Path(_proj_dir) / "proj.db").exists():
        os.environ["PROJ_DATA"] = _proj_dir
        os.environ["PROJ_LIB"]  = _proj_dir
        _pp.datadir.set_data_dir(_proj_dir)
except Exception:
    pass
# ──────────────────────────────────────────────────────────────────────────────

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
c1.metric("Ano objetivo",               tmp["target_year"])
c2.metric("Area objetivo (km2)",        aoi["target_area_km2"])
c3.metric("Bloque de validacion (km)",  aoi["block_size_km"])

st.markdown(
    "Navega por las paginas en la barra lateral: Datos y AOI, Diagnostico "
    "espacial, Modelos y metricas, Mapa de predicciones, Por municipio."
)
