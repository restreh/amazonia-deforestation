"""Analisis por municipio: hectareas detectadas vs Hansen, choropleth y
coeficiente de concordancia agregado, con selector entre los cinco modelos."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import streamlit as st
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.mask
import plotly.express as px

from utils import (PATHS, read_json, warn_if_missing, NOMBRES_MODELO,
                   MODELOS_CANDIDATOS, MODELO_CONTROL, fmt_hectareas, PIXEL_HA)
from theme import (aplicar_tema_plotly, aplicar_estilos_streamlit, takeaway,
                   lead, COLORES, COLORES_MODELO)


aplicar_tema_plotly()
aplicar_estilos_streamlit()

st.title("Análisis por municipio")
lead(
    "Agregación por municipio de las hectáreas detectadas contra las "
    "reportadas por Hansen GFC 2024. Permite comparar la calibración total "
    "de cada modelo y elegir el más fiable cuando lo que importa es el "
    "número agregado, no la posición píxel a píxel."
)

# ── Selectores ──────────────────────────────────────────────────────────────
disponibles = [m for m in MODELOS_CANDIDATOS + [MODELO_CONTROL]
               if PATHS["proba_" + m].exists()]
if not disponibles:
    st.warning("No hay rasters de probabilidad disponibles.")
    st.stop()

c_a, c_b = st.columns([2, 2])
with c_a:
    modelo_id = st.selectbox(
        "Modelo",
        disponibles,
        format_func=lambda m: NOMBRES_MODELO[m],
        index=disponibles.index("ensemble") if "ensemble" in disponibles else 0,
    )

ev = read_json("eval_" + modelo_id) or {}
umbral_calibrado = float(ev.get("threshold", 0.5))

with c_b:
    umbral = st.slider("Umbral de decisión τ", 0.0, 1.0,
                       umbral_calibrado, 0.01)

st.caption(f"Umbral calibrado del modelo: {umbral_calibrado:.3f}")

# ── Calculo: zonal stats con reproyeccion correcta ──────────────────────────
for key in ("municipios", "etiqueta"):
    if not warn_if_missing(key):
        st.stop()


@st.cache_data(show_spinner="Calculando hectáreas por municipio…")
def hectareas_por_municipio(modelo_id: str, umbral: float):
    proba_path = PATHS["proba_" + modelo_id]
    with rasterio.open(PATHS["etiqueta"]) as src_et:
        crs_raster = src_et.crs

    gdf = gpd.read_file(PATHS["municipios"]).to_crs(crs_raster)
    name_field = next((c for c in ("MpNombre", "mpio_cnmbr", "nombre",
                                   "NOMBRE_MPI") if c in gdf.columns),
                       gdf.columns[0])

    filas = []
    with rasterio.open(PATHS["etiqueta"]) as src_et, \
         rasterio.open(str(proba_path)) as src_pr:
        for _, row in gdf.iterrows():
            geom = [row.geometry.__geo_interface__]
            # etiqueta Hansen
            et, _ = rasterio.mask.mask(src_et, geom, crop=True, all_touched=True)
            et = et[0]
            et_mask = et > 0
            et_positivos = int(((et == 1) & et_mask).sum())

            # probabilidad
            pr, _ = rasterio.mask.mask(src_pr, geom, crop=True,
                                        all_touched=True, filled=False)
            pr = np.ma.filled(pr[0].astype("float32"), np.nan)
            pred = (pr >= umbral) & np.isfinite(pr)
            pred_positivos = int(pred.sum())

            area_pixeles = int(np.isfinite(pr).sum())
            area_ha = area_pixeles * PIXEL_HA
            filas.append({
                "Municipio": str(row[name_field]).title(),
                "Hansen (ha)": et_positivos * PIXEL_HA,
                "Modelo (ha)": pred_positivos * PIXEL_HA,
                "Área del AOI en el municipio (ha)": area_ha,
            })

    df = pd.DataFrame(filas)
    df["Diferencia (ha)"] = df["Modelo (ha)"] - df["Hansen (ha)"]
    df["Razón modelo / Hansen"] = df["Modelo (ha)"] / df["Hansen (ha)"]
    df["% deforestado (Hansen)"] = df["Hansen (ha)"] / df["Área del AOI en el municipio (ha)"]
    return df


df = hectareas_por_municipio(modelo_id, umbral)

# ── Tabla principal ────────────────────────────────────────────────────────
st.subheader("Hectáreas por municipio")

st.dataframe(
    df.style.format({
        "Hansen (ha)":            "{:,.0f}",
        "Modelo (ha)":            "{:,.0f}",
        "Diferencia (ha)":        "{:+,.0f}",
        "Razón modelo / Hansen":  "{:.3f}",
        "% deforestado (Hansen)": "{:.2%}",
        "Área del AOI en el municipio (ha)": "{:,.0f}",
    }),
    use_container_width=True,
)

st.divider()

# ── Choropleth con tasa Hansen ──────────────────────────────────────────────
st.subheader("Choropleth — tasa de pérdida en el municipio (Hansen 2024)")

gdf_plot = gpd.read_file(PATHS["municipios"]).to_crs(epsg=4326)
name_field = next((c for c in ("MpNombre", "mpio_cnmbr", "nombre", "NOMBRE_MPI")
                   if c in gdf_plot.columns), gdf_plot.columns[0])
gdf_plot["Municipio"] = gdf_plot[name_field].astype(str).str.title()
gdf_plot = gdf_plot.merge(df[["Municipio", "% deforestado (Hansen)",
                              "Hansen (ha)", "Modelo (ha)"]],
                          on="Municipio", how="left")

fig = px.choropleth_mapbox(
    gdf_plot,
    geojson=gdf_plot.geometry.__geo_interface__,
    locations=gdf_plot.index, color="% deforestado (Hansen)",
    color_continuous_scale="Reds",
    range_color=[0, max(0.05, gdf_plot["% deforestado (Hansen)"].max() * 1.1)],
    mapbox_style="carto-darkmatter",
    center={"lat": gdf_plot.geometry.centroid.y.mean(),
            "lon": gdf_plot.geometry.centroid.x.mean()},
    zoom=8.5, opacity=0.7,
    hover_name="Municipio",
    hover_data={"Hansen (ha)": ":,.0f", "Modelo (ha)": ":,.0f",
                "% deforestado (Hansen)": ":.2%"},
)
fig.update_layout(margin=dict(l=0, r=0, t=20, b=0), height=420)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Barras comparativas ─────────────────────────────────────────────────────
st.subheader("Hansen vs. modelo por municipio")

df_long = df.melt(id_vars="Municipio",
                  value_vars=["Hansen (ha)", "Modelo (ha)"],
                  var_name="Fuente", value_name="Hectáreas")
fig = px.bar(df_long, x="Municipio", y="Hectáreas", color="Fuente",
             barmode="group", text_auto=".0f",
             color_discrete_map={"Hansen (ha)": COLORES["referencia"],
                                 "Modelo (ha)": COLORES_MODELO.get(modelo_id, COLORES["predicho"])})
fig.update_layout(xaxis_title="", yaxis_title="Hectáreas", height=320,
                  legend_title_text="")
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Calibracion total y CCC del modelo ─────────────────────────────────────
ccc = read_json("concordance_lin") or {}
ccc_modelo = ccc.get("models", {}).get(modelo_id, {})

c1, c2, c3 = st.columns(3)
c1.metric("Total Hansen (ha)", fmt_hectareas(df["Hansen (ha)"].sum()))
c2.metric("Total modelo (ha)", fmt_hectareas(df["Modelo (ha)"].sum()),
          delta=f"{df['Diferencia (ha)'].sum():+,.0f} ha".replace(",", "."),
          delta_color="inverse" if df["Diferencia (ha)"].sum() > 0 else "normal",
          help="Positivo = sobreestimación. Negativo = subestimación.")
c3.metric("CCC de Lin por bloque",
          f"{ccc_modelo.get('ccc_block', float('nan')):.3f}"
          if ccc_modelo.get("ccc_block") else "—",
          help="Concordancia entre hectáreas predichas y Hansen, agregadas "
               "por bloque de 5 km. Cercano a 1 → calibración casi perfecta.")

razon = ccc_modelo.get("total_predicted_ha_aoi", 0) / ccc_modelo.get(
    "total_truth_ha_aoi", 1) if ccc_modelo else None

takeaway(
    f"Para totales agregados, la razón predicho/Hansen del modelo "
    f"seleccionado es <strong>{razon:.3f}</strong>" + (
        " (casi perfecta calibración)" if razon and abs(razon - 1) < 0.05
        else f" ({'sobreestimación' if razon and razon > 1 else 'subestimación'} "
             f"de {abs(razon - 1) * 100:.1f} %)" if razon else ""
    ) + ". Para alertas accionables municipio a municipio, el ensamble suele "
        "ser el mejor balance entre cobertura y precisión; para reportes "
        "agregados, el U-Net está más cerca del total Hansen."
)
