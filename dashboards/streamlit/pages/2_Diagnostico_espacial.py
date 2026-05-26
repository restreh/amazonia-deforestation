"""Diagnostico espacial: I de Moran, semivariograma, particion por bloques."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from utils import warn_if_missing, PATHS

st.title("Diagnostico espacial")

st.markdown(
    "La deforestacion esta fuertemente autocorrelacionada en el espacio, por lo "
    "que la validacion se hace en bloques de 5 km (no por pixel) para evitar "
    "fuga de informacion entre entrenamiento y prueba."
)

st.subheader("Resumen de diagnosticos")
if warn_if_missing("diagnosticos"):
    txt = PATHS["diagnosticos"].read_text(encoding="utf-8")
    st.text(txt)
    # TODO: extraer y mostrar I de Moran, rango y tamano de bloque en st.metric().

st.subheader("Particion por bloques (train, val, test)")
# TODO: renderizar split_blocks.tif con leafmap (m.add_raster) coloreado por codigo.
warn_if_missing("split")
