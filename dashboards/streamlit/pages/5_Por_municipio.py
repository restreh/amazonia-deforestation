"""Analisis por municipio: hectareas Hansen vs modelo, tabla ordenable y coroplético."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
import rasterio.mask
import plotly.express as px
from shapely.geometry import mapping
from utils import read_json, warn_if_missing, PATHS

st.set_page_config(page_title="Por Municipio", layout="wide")
st.title("🏘️ Análisis por Municipio")
st.markdown(
    "Hectáreas de deforestación reportadas por Hansen GFC 2024 vs. detectadas por el modelo, "
    "desagregadas por municipio."
)

# Verificar archivos mínimos
for key in ("municipios", "etiqueta", "proba_xgboost"):
    if not warn_if_missing(key):
        st.stop()

# Umbral calibrado
umbral_default = 0.5
eval_data = read_json("eval_xgboost")
if eval_data:
    umbral_default = float(eval_data.get("threshold", 0.5))

umbral = st.slider("Umbral de decisión (τ)", 0.0, 1.0, umbral_default, 0.01,
                   help="Mismo umbral que en la página de Mapa de Predicciones.")

# ── Cómputo de estadísticas por municipio ─────────────────────────────────────
@st.cache_data(show_spinner=False)
def calcular_stats(umbral: float):
    # Leer GeoJSON y reproyectar al CRS del raster de etiqueta
    # BUG FIX: los rasters están en UTM; las geometrías deben coincidir en CRS
    # antes de llamar rasterio.mask, y el área debe calcularse en CRS métrico.
    gdf_4326 = gpd.read_file(PATHS["municipios"]).to_crs(epsg=4326)

    with rasterio.open(PATHS["etiqueta"]) as src_et:
        crs_raster = src_et.crs

    # Reproyectar al CRS del raster para el enmascaramiento
    gdf_utm = gdf_4326.to_crs(crs_raster)

    col_nombre = next(
        (c for c in gdf_utm.columns
         if c.lower() in ("nombre", "nombre_mpi", "mpio_cnmbr", "municipio")),
        gdf_utm.columns[0],
    )

    registros = []
    with rasterio.open(PATHS["etiqueta"]) as src_et, \
         rasterio.open(PATHS["proba_xgboost"]) as src_pr:

        for idx, fila in gdf_utm.iterrows():
            geom   = [mapping(fila.geometry)]
            nombre = str(fila[col_nombre])
            try:
                et, _ = rasterio.mask.mask(src_et, geom, crop=True, all_touched=True)
                pr, _ = rasterio.mask.mask(src_pr, geom, crop=True, all_touched=True)

                pr_f  = pr[0].astype(np.float32)
                nd_pr = src_pr.nodata
                if nd_pr is not None:
                    pr_f = np.where(pr_f == nd_pr, np.nan, pr_f)

                nd_et   = src_et.nodata
                mascara = (et[0] != nd_et) if nd_et is not None else np.ones(et[0].shape, bool)

                ha_hansen = float((et[0][mascara] == 1).sum()) * 0.04
                ha_modelo = float(np.nansum(pr_f[mascara] >= umbral)) * 0.04

                # BUG FIX: área en CRS métrico (m²→km²), no en grados
                area_km2 = fila.geometry.area / 1e6

                registros.append({
                    "Municipio":       nombre,
                    "Área (km²)":      round(area_km2, 1),
                    "Hansen (ha)":     round(ha_hansen, 1),
                    "Modelo (ha)":     round(ha_modelo, 1),
                    "Diferencia (ha)": round(ha_modelo - ha_hansen, 1),
                    "% Hansen / Área": round(ha_hansen / (area_km2 * 100) * 100, 2)
                                       if area_km2 > 0 else 0.0,
                    # Guardar geometría en 4326 para el mapa
                    "geometry":        gdf_4326.loc[idx, "geometry"],
                })
            except Exception:
                continue

    df = pd.DataFrame(registros).sort_values("Hansen (ha)", ascending=False)
    return df, gdf_4326, col_nombre

with st.spinner("Calculando estadísticas por municipio…"):
    df_mun, gdf, col_nombre = calcular_stats(umbral)

if df_mun.empty:
    st.error("No se pudieron calcular estadísticas. Verifica que el GeoJSON y los rasters coincidan.")
    st.stop()

# ── 1. Tabla ordenable ────────────────────────────────────────────────────────
st.subheader("Tabla de deforestación por municipio")

def _color_diff(val):
    if not isinstance(val, (int, float)):
        return ""
    return "color:#d62728" if val > 0 else ("color:#2d6a4f" if val < 0 else "")

st.dataframe(
    df_mun.drop(columns=["geometry"]).style
          .applymap(_color_diff, subset=["Diferencia (ha)"])
          .format({"Área (km²)": "{:,.1f}", "Hansen (ha)": "{:,.1f}",
                   "Modelo (ha)": "{:,.1f}", "Diferencia (ha)": "{:+,.1f}",
                   "% Hansen / Área": "{:.2f} %"}),
    use_container_width=True,
    height=280,
)
st.caption("Diferencia = Modelo − Hansen. Rojo → sobredetección · Verde → subdetección.")

st.divider()

# ── 2. Mapa coroplético ───────────────────────────────────────────────────────
st.subheader("Mapa coroplético — Hectáreas detectadas por el modelo")

# BUG FIX: construir GeoDataFrame con los datos de df_mun para que el merge
# y los IDs del GeoJSON sean consistentes, evitando el mismatch de featureidkey.
gdf_plot = gpd.GeoDataFrame(df_mun, geometry="geometry", crs="EPSG:4326")
# Usar el nombre del municipio como clave —es único y evita el problema de IDs enteros vs string
geojson_dict = gdf_plot.__geo_interface__
# Añadir propiedad "Municipio" como id en cada feature para que plotly pueda matchear
for i, feat in enumerate(geojson_dict["features"]):
    feat["id"] = gdf_plot.iloc[i]["Municipio"]

cy = gdf_plot.geometry.centroid.y.mean()
cx = gdf_plot.geometry.centroid.x.mean()

fig_c = px.choropleth_mapbox(
    gdf_plot,
    geojson=geojson_dict,
    locations="Municipio",       # columna con el valor que coincide con feat["id"]
    color="Modelo (ha)",
    hover_name="Municipio",
    hover_data={"Hansen (ha)": True, "Modelo (ha)": True, "Diferencia (ha)": True},
    mapbox_style="carto-positron",
    center={"lat": cy, "lon": cx}, zoom=8,
    color_continuous_scale="YlOrRd",
    labels={"Modelo (ha)": "Ha detectadas"}, opacity=0.72,
)
fig_c.update_layout(margin={"r": 0, "t": 10, "l": 0, "b": 0})
st.plotly_chart(fig_c, use_container_width=True)

st.divider()

# ── 3. Barras comparativas ────────────────────────────────────────────────────
st.subheader("Comparación Hansen vs. Modelo por municipio")

df_long = df_mun[["Municipio", "Hansen (ha)", "Modelo (ha)"]].melt(
    id_vars="Municipio", var_name="Fuente", value_name="Hectáreas")
fig_b = px.bar(df_long, x="Municipio", y="Hectáreas", color="Fuente",
               barmode="group", text_auto=".0f",
               color_discrete_map={"Hansen (ha)": "#1b4332", "Modelo (ha)": "#52b788"})
fig_b.update_layout(plot_bgcolor="white", xaxis_tickangle=-20,
                    legend_title="Fuente", yaxis_title="Hectáreas", xaxis_title="")
st.plotly_chart(fig_b, use_container_width=True)

st.divider()

# ── 4. Totales ────────────────────────────────────────────────────────────────
st.subheader("Totales del área de interés")

c1, c2, c3 = st.columns(3)
c1.metric("Total Hansen (ha)",  f"{df_mun['Hansen (ha)'].sum():,.0f}")
c2.metric("Total Modelo (ha)",  f"{df_mun['Modelo (ha)'].sum():,.0f}")
diff = df_mun["Modelo (ha)"].sum() - df_mun["Hansen (ha)"].sum()
c3.metric("Diferencia neta (ha)", f"{diff:+,.0f}",
          delta_color="inverse" if diff > 0 else "normal",
          help="Positivo = sobredetección · Negativo = subdetección respecto a Hansen GFC.")
