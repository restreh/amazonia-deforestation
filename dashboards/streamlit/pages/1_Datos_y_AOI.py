"""Datos y area de interes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import geopandas as gpd
import numpy as np
import rasterio
import plotly.express as px
import leafmap.foliumap as leafmap
from utils import load_config, read_csv, warn_if_missing, PATHS

st.set_page_config(page_title="Datos y AOI", layout="wide")
st.title("📍 Datos y Área de Interés")
st.markdown(
    "Caracterización del área de estudio (~5.023 km² en el Caquetá) "
    "y disponibilidad de imágenes Sentinel-2 por trimestre."
)

cfg = load_config()
aoi = cfg["aoi"]
tmp = cfg["temporal"]

# ── 1. Mapa AOI y municipios ──────────────────────────────────────────────────
st.subheader("Área de interés — Núcleo activo del Caquetá")

if warn_if_missing("municipios"):
    @st.cache_data(show_spinner=False)
    def cargar_municipios():
        return gpd.read_file(PATHS["municipios"]).to_crs(epsg=4326)

    gdf = cargar_municipios()
    cy  = gdf.geometry.centroid.y.mean()
    cx  = gdf.geometry.centroid.x.mean()

    m = leafmap.Map(center=[cy, cx], zoom=9)
    m.add_basemap("CartoDB.Positron")
    m.add_gdf(
        gdf,
        layer_name="Municipios",
        style={"color": "#1b4332", "fillColor": "#52b788", "fillOpacity": 0.20, "weight": 2},
        hover_style={"fillOpacity": 0.45},
    )
    m.to_streamlit(height=460)
    st.caption("Fuente: IGAC — Cartagena del Chairá y San Vicente del Caguán, Caquetá.")

st.divider()

# ── 2. Disponibilidad Sentinel-2 por trimestre ────────────────────────────────
st.subheader("Disponibilidad de escenas Sentinel-2 por trimestre")

if warn_if_missing("disponibilidad"):
    df_s2 = read_csv("disponibilidad")

    fig = px.bar(
        df_s2,
        x="quarter",
        y="scenes",
        text="scenes",
        labels={"quarter": "Trimestre", "scenes": "Escenas disponibles"},
        color="scenes",
        color_continuous_scale="Greens",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        plot_bgcolor="white",
        yaxis_title="Número de escenas",
        coloraxis_showscale=False,
        xaxis_title="",
    )

    idx_max = df_s2["scenes"].idxmax()
    fig.add_annotation(
        x=df_s2.loc[idx_max, "quarter"],
        y=df_s2.loc[idx_max, "scenes"],
        text="☀️ Más despejado",
        showarrow=True,
        arrowhead=2,
        yshift=10,
        font={"color": "#1b4332"},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "La nubosidad sigue la estacionalidad amazónica: "
        "el tercer trimestre (jul–sep) es el más despejado y el segundo (abr–jun) el más nublado."
    )

st.divider()

# ── 3. Prevalencia de la etiqueta ─────────────────────────────────────────────
st.subheader("Prevalencia de la clase deforestación — Etiqueta Hansen GFC 2024")

if warn_if_missing("etiqueta"):
    @st.cache_data(show_spinner=False)
    def calcular_prevalencia():
        with rasterio.open(PATHS["etiqueta"]) as src:
            arr    = src.read(1)
            nodata = src.nodata
        mascara   = (arr != nodata) if nodata is not None else np.ones(arr.shape, bool)
        total     = int(mascara.sum())
        positivos = int(((arr == 1) & mascara).sum())
        return total, positivos

    total, positivos = calcular_prevalencia()
    prevalencia      = positivos / total * 100 if total > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Píxeles válidos",          f"{total:,}")
    c2.metric("Píxeles deforestados",     f"{positivos:,}")
    c3.metric("Prevalencia clase positiva", f"{prevalencia:.2f} %")
    c4.metric("Área deforestada (ha)",    f"{positivos * 0.04:,.0f} ha")

    fig_d = px.pie(
        names=["Deforestación (clase 1)", "Sin pérdida (clase 0)"],
        values=[positivos, total - positivos],
        hole=0.55,
        color_discrete_sequence=["#d62728", "#2ca02c"],
    )
    fig_d.update_traces(textinfo="percent+label")
    fig_d.update_layout(showlegend=False, margin=dict(t=20, b=20))
    st.plotly_chart(fig_d, use_container_width=True)
    st.caption(
        "Desbalance severo (~1–3 % de clase positiva): motiva el uso de "
        "pérdida focal y muestreo balanceado. Cada píxel = 20×20 m = 0,04 ha."
    )
