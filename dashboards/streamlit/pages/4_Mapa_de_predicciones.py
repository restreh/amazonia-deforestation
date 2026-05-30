import io, base64, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import streamlit as st
import rasterio
import rasterio.warp
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from utils import load_config, read_json, warn_if_missing, PATHS

st.set_page_config(page_title="Mapa de Predicciones", layout="wide")
st.title("🛰️ Mapa de Predicciones")
st.markdown("Probabilidad de deforestación píxel a píxel y análisis de errores "
            "(TP / FP / FN) sobre el conjunto de prueba.")

# ── Helper: raster → folium ImageOverlay (sin localtileserver) ────────────────
@st.cache_data(show_spinner=False)
def raster_a_overlay(ruta: str, cmap_name: str, vmin: float, vmax: float,
                     max_px: int = 700, opacidad: float = 0.75,
                     layer_name: str = "Raster"):
    with rasterio.open(ruta) as src:
        factor = max(1, max(src.width, src.height) // max_px)
        data   = src.read(1, out_shape=(max(1, src.height // factor),
                                        max(1, src.width  // factor)),
                          resampling=rasterio.enums.Resampling.nearest).astype(np.float32)
        nodata = src.nodata
        bounds = rasterio.warp.transform_bounds(src.crs, "EPSG:4326", *src.bounds)

    if nodata is not None:
        data = np.where(data == nodata, np.nan, data)

    cmap = plt.get_cmap(cmap_name)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    rgba = cmap(norm(data))
    if nodata is not None:
        rgba[np.isnan(data), 3] = 0

    buf = io.BytesIO()
    plt.imsave(buf, (rgba * 255).astype(np.uint8), format="png")
    buf.seek(0)
    img_b64 = "data:image/png;base64," + base64.b64encode(buf.read()).decode()

    west, south, east, north = bounds
    overlay = folium.raster_layers.ImageOverlay(
        image=img_b64, bounds=[[south, west], [north, east]],
        opacity=opacidad, name=layer_name,
    )
    return overlay, [(south + north) / 2, (west + east) / 2]

# ── Selector de modelo ────────────────────────────────────────────────────────
opciones: dict[str, Path] = {}
if PATHS["proba_xgboost"].exists():
    opciones["XGBoost (baseline)"] = PATHS["proba_xgboost"]
if PATHS["proba_unet"].exists():
    opciones["U-Net (modelo principal)"] = PATHS["proba_unet"]

if not opciones:
    st.warning("⚠️ Aún no hay rasters de predicción disponibles. "
               "Se generan al ejecutar `scripts/predict.py`.")
    st.stop()

col_sel, col_umbral = st.columns([2, 2])
with col_sel:
    modelo_sel = st.selectbox("Modelo", list(opciones.keys()))
ruta_proba = opciones[modelo_sel]

umbral_default = 0.5
eval_key  = "eval_xgboost" if "XGBoost" in modelo_sel else "eval_unet"
eval_data = read_json(eval_key)
if eval_data:
    umbral_default = float(eval_data.get("threshold", 0.5))

with col_umbral:
    umbral = st.slider("Umbral de decisión (τ)", 0.0, 1.0, umbral_default, 0.01,
                       help="Mayor τ → menos falsas alarmas · Menor τ → menos omisiones.")

# ── Mapa de probabilidades ────────────────────────────────────────────────────
st.subheader(f"Probabilidad de deforestación — {modelo_sel}")

overlay_proba, centro = raster_a_overlay(
    str(ruta_proba), "RdYlGn_r", 0.0, 1.0, layer_name=f"Prob. {modelo_sel}"
)
m = folium.Map(location=centro, zoom_start=10, tiles="CartoDB dark_matter")
overlay_proba.add_to(m)

# ── Capa TP / FP / FN (opcional) ─────────────────────────────────────────────
mostrar_errores = st.checkbox("Mostrar capa TP / FP / FN sobre el conjunto de prueba")
conteos: dict = {}

if mostrar_errores and warn_if_missing("split") and warn_if_missing("etiqueta"):

    @st.cache_data(show_spinner=False)
    def calcular_errores(ruta_proba: str, umbral: float):
        with rasterio.open(ruta_proba) as src:
            proba  = src.read(1).astype(np.float32)
            perfil = src.profile.copy()
            nd     = src.nodata
        if nd is not None:
            proba = np.where(proba == nd, np.nan, proba)
        with rasterio.open(PATHS["etiqueta"]) as src:
            etiqueta = src.read(1)
        with rasterio.open(PATHS["split"]) as src:
            split = src.read(1)
        pred = (proba >= umbral).astype(np.uint8)
        test = split == 3
        resultado = np.zeros_like(pred, dtype=np.uint8)
        resultado[test & (pred == 1) & (etiqueta == 1)] = 1  # TP
        resultado[test & (pred == 1) & (etiqueta == 0)] = 2  # FP
        resultado[test & (pred == 0) & (etiqueta == 1)] = 3  # FN
        tp = int((resultado == 1).sum())
        fp = int((resultado == 2).sum())
        fn = int((resultado == 3).sum())
        perfil.update(dtype=rasterio.uint8, count=1, nodata=0)
        return resultado, perfil, tp, fp, fn

    resultado, perfil, tp, fp, fn = calcular_errores(str(ruta_proba), umbral)
    conteos = {"TP": tp, "FP": fp, "FN": fn}

    # Renderizar capa de errores: 0=transp, 1=verde(TP), 2=naranja(FP), 3=rojo(FN)
    tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
    with rasterio.open(tmp.name, "w", **perfil) as dst:
        dst.write(resultado, 1)

    overlay_err, _ = raster_a_overlay(
        tmp.name,
        cmap_name="tab10", vmin=0, vmax=3,
        opacidad=0.70, layer_name="TP / FP / FN"
    )
    overlay_err.add_to(m)

folium.LayerControl().add_to(m)
st_folium(m, height=520, use_container_width=True)

st.caption("🟩 verde = baja probabilidad · 🟥 rojo = alta probabilidad. "
           "Errores: 🟩 TP · 🟧 FP (falsa alarma) · 🟥 FN (omisión).")

st.divider()

# ── Indicadores cuantitativos ─────────────────────────────────────────────────
st.subheader("Indicadores cuantitativos")

@st.cache_data(show_spinner=False)
def hectareas(ruta: str, umbral: float) -> float:
    with rasterio.open(ruta) as src:
        p  = src.read(1).astype(np.float32)
        nd = src.nodata
    if nd is not None:
        p = np.where(p == nd, np.nan, p)
    return float(np.nansum(p >= umbral)) * 0.04

ha   = hectareas(str(ruta_proba), umbral)
cols = st.columns(4)
cols[0].metric(f"Hectáreas detectadas (τ={umbral:.2f})", f"{ha:,.0f} ha",
               help="Píxeles con probabilidad ≥ τ × 0,04 ha/píxel.")

if conteos:
    f1v = (2*conteos["TP"]) / (2*conteos["TP"] + conteos["FP"] + conteos["FN"] + 1e-9)
    cols[1].metric("TP — Detecciones correctas", f"{conteos['TP']:,}")
    cols[2].metric("FP — Falsas alarmas",         f"{conteos['FP']:,}", delta_color="inverse")
    cols[3].metric("FN — Omisiones",               f"{conteos['FN']:,}", delta_color="inverse")
    st.metric("F1 en prueba (τ actual)", f"{f1v:.3f}")
