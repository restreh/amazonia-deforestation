"""Mapa de predicciones con capas conmutables (cinco modelos + Hansen),
umbral interactivo y diagnostico TP/FP/FN sobre el conjunto de prueba."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import streamlit as st
import rasterio
import folium
from streamlit_folium import st_folium

from utils import (PATHS, read_json, warn_if_missing, raster_a_overlay,
                   NOMBRES_MODELO, MODELOS_CANDIDATOS, MODELO_CONTROL,
                   fmt_int, fmt_hectareas, PIXEL_HA)
from theme import (aplicar_tema_plotly, aplicar_estilos_streamlit, takeaway,
                   lead, CMAP_PROBA, CMAP_ERROR)


aplicar_tema_plotly()
aplicar_estilos_streamlit()

st.title("Mapa de predicciones")
lead(
    "Probabilidad de deforestación por píxel sobre los bloques de validación "
    "y prueba, capas conmutables para los cinco modelos y la etiqueta de "
    "Hansen, y diagnóstico de errores TP/FP/FN al umbral elegido. La "
    "leyenda de color es consistente con el resto del tablero."
)


# ── Seleccion de modelo y umbral ────────────────────────────────────────────
disponibles = []
for m in MODELOS_CANDIDATOS + [MODELO_CONTROL]:
    p = PATHS["proba_" + m]
    if p.exists():
        disponibles.append(m)

if not disponibles:
    st.warning("No hay rasters de probabilidad. Corre `predict.py` y/o "
               "`predict_unet.py` y `build_ensemble.py`.")
    st.stop()

c1, c2 = st.columns([2, 2])
with c1:
    modelo_id = st.selectbox(
        "Modelo a visualizar",
        options=disponibles,
        format_func=lambda m: NOMBRES_MODELO[m],
        index=disponibles.index("ensemble") if "ensemble" in disponibles else 0,
        help="El ensamble es el modelo final por F1 píxel; U-Net es el más "
             "calibrado en hectáreas totales (ver Modelos y métricas).",
    )

ev = read_json("eval_" + modelo_id) or {}
umbral_calibrado = float(ev.get("threshold", 0.5))

with c2:
    umbral = st.slider(
        "Umbral de decisión τ",
        0.0, 1.0, umbral_calibrado, 0.01,
        help="Calibrado por defecto sobre validación maximizando F1 a "
             "prevalencia real. Mayor τ → menos falsas alarmas, menos recall.",
    )

st.caption(f"Umbral calibrado para {NOMBRES_MODELO[modelo_id]}: "
           f"{umbral_calibrado:.3f}")


# ── Mapa con capas conmutables ──────────────────────────────────────────────
st.subheader(f"Probabilidad — {NOMBRES_MODELO[modelo_id]}")

overlay_proba, centro = raster_a_overlay(
    str(PATHS["proba_" + modelo_id]),
    colormap=CMAP_PROBA, vmin=0.0, vmax=1.0,
    layer_name=f"Probabilidad {NOMBRES_MODELO[modelo_id]}",
)

m = folium.Map(location=centro, zoom_start=10, tiles="CartoDB dark_matter")
overlay_proba.add_to(m)

# Hansen como capa de referencia conmutable
if warn_if_missing("etiqueta"):
    overlay_hansen, _ = raster_a_overlay(
        str(PATHS["etiqueta"]), colormap=["#0e1117", "#d62828"],
        vmin=0, vmax=1, opacidad=0.70, layer_name="Hansen GFC 2024",
    )
    overlay_hansen.add_to(m)


# ── Capa TP/FP/FN opcional ──────────────────────────────────────────────────
mostrar_errores = st.checkbox(
    "Diagnóstico TP / FP / FN sobre el conjunto de prueba",
    help="Cruce de la predicción binarizada al umbral con la etiqueta de "
         "Hansen, restringido a los bloques de prueba (split == 3).",
)

conteos = {}
if mostrar_errores:

    @st.cache_data(show_spinner=False)
    def calcular_errores(ruta_proba: str, umbral: float):
        with rasterio.open(ruta_proba) as src:
            proba = src.read(1).astype("float32")
            perfil = src.profile.copy()
            nd = src.nodata
        if nd is not None:
            proba = np.where(proba == nd, np.nan, proba)
        with rasterio.open(PATHS["etiqueta"]) as src:
            etiqueta = src.read(1)
        with rasterio.open(PATHS["split"]) as src:
            split = src.read(1)
        pred = np.where(np.isnan(proba), 0, (proba >= umbral).astype("uint8"))
        test = split == 3
        resultado = np.zeros_like(pred, dtype="uint8")
        resultado[test & (pred == 1) & (etiqueta == 1)] = 1   # TP
        resultado[test & (pred == 1) & (etiqueta == 0)] = 2   # FP
        resultado[test & (pred == 0) & (etiqueta == 1)] = 3   # FN
        tp = int((resultado == 1).sum())
        fp = int((resultado == 2).sum())
        fn = int((resultado == 3).sum())
        tn = int((test & (pred == 0) & (etiqueta == 0)).sum())
        perfil.update(dtype="uint8", count=1, nodata=0)
        tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        with rasterio.open(tmp.name, "w", **perfil) as dst:
            dst.write(resultado, 1)
        return tmp.name, tp, fp, fn, tn

    ruta_err, tp, fp, fn, tn = calcular_errores(
        str(PATHS["proba_" + modelo_id]), umbral)
    conteos = {"TP": tp, "FP": fp, "FN": fn, "TN": tn}
    overlay_err, _ = raster_a_overlay(
        ruta_err, colormap=CMAP_ERROR, vmin=0, vmax=3, opacidad=0.85,
        layer_name="TP / FP / FN")
    overlay_err.add_to(m)

folium.LayerControl(collapsed=False, position="topright").add_to(m)
st_folium(m, height=520, use_container_width=True, returned_objects=[])

st.caption(
    "Colores: probabilidad de deforestación con escala verde (0) → rojo (1). "
    "TP en verde, FP en naranja, FN en rojo. Capa Hansen conmutable como "
    "referencia institucional."
)

st.divider()


# ── Indicadores cuantitativos ───────────────────────────────────────────────
st.subheader("Indicadores al umbral actual")

@st.cache_data(show_spinner=False)
def hectareas_predichas(ruta: str, umbral: float) -> float:
    with rasterio.open(ruta) as src:
        p = src.read(1).astype("float32")
        nd = src.nodata
    if nd is not None:
        p = np.where(p == nd, np.nan, p)
    return float(np.nansum(p >= umbral)) * PIXEL_HA


ha = hectareas_predichas(str(PATHS["proba_" + modelo_id]), umbral)

if conteos:
    tp, fp, fn = conteos["TP"], conteos["FP"], conteos["FN"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    cols = st.columns(5)
    cols[0].metric("Hectáreas predichas (AOI)", fmt_hectareas(ha))
    cols[1].metric("F1 píxel (prueba, τ actual)", f"{f1:.3f}")
    cols[2].metric("TP", fmt_int(tp))
    cols[3].metric("FP — falsas alarmas", fmt_int(fp), delta_color="inverse")
    cols[4].metric("FN — omisiones", fmt_int(fn), delta_color="inverse")

    takeaway(
        f"Subir τ por encima de <strong>{umbral_calibrado:.2f}</strong> reduce los "
        "FP (falsas alarmas) y aumenta los FN (alertas que dejas pasar). "
        "Bajar τ es lo opuesto. El equilibrio operativo depende del costo "
        "relativo de un equipo de campo enviado en balde vs el costo de no "
        "detectar pérdida activa."
    )
else:
    c1, c2 = st.columns(2)
    c1.metric("Hectáreas predichas (AOI)", fmt_hectareas(ha))
    c2.metric("Umbral", f"{umbral:.3f}")
