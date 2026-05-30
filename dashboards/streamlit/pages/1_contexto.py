"""Contexto y datos: por que importa Caqueta, datos de entrada, prevalencia honesta."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import plotly.express as px

from utils import (load_config, read_csv, warn_if_missing,
                   leer_geojson, estadisticas_etiqueta,
                   fmt_int, fmt_pct, fmt_hectareas)
from theme import aplicar_tema_plotly, aplicar_estilos_streamlit, takeaway, lead


aplicar_tema_plotly()
aplicar_estilos_streamlit()

st.title("Contexto y datos")
lead(
    "El proyecto se ubica en el arco amazónico colombiano, en dos municipios "
    "del Caquetá (Cartagena del Chairá y San Vicente del Caguán). El "
    "departamento del Caquetá encabezó el incremento trimestral de pérdida en "
    "el último Boletín del IDEAM. Esta página enmarca el problema y "
    "caracteriza los datos."
)

cfg = load_config()
aoi = cfg["aoi"]
tmp = cfg["temporal"]

# ── 1. El contexto: por qué Caquetá ───────────────────────────────────────
st.subheader("¿Por qué Caquetá?")

c1, c2, c3 = st.columns(3)
c1.metric("Deforestación nacional 2024",
          "107.000 ha",
          delta="+35 % vs 2023",
          delta_color="inverse",
          help="Boletín 42 del IDEAM (SMByC).")
c2.metric("Incremento Caquetá Q4-2025",
          "+9.078 ha",
          delta="vs mismo trimestre 2024",
          delta_color="inverse",
          help="Boletín 45 del IDEAM. El mayor incremento entre departamentos.")
c3.metric("Núcleos activos en la Amazonía",
          "21",
          help="Identificados en el Boletín 45 del IDEAM. Cuemaní está en el AOI.")

takeaway(
    "El AOI captura aproximadamente <strong>54 %</strong> de la pérdida municipal "
    "2024 en una ventana de 5.023 km². El SMByC publica boletines trimestrales; "
    "este sistema se concibe como complemento académico con código abierto y "
    "ciclo de actualización más corto."
)

st.divider()

# ── 2. Mapa del AOI ────────────────────────────────────────────────────────
st.subheader("Área de interés")

if warn_if_missing("municipios"):
    import leafmap.foliumap as leafmap

    @st.cache_data(show_spinner=False)
    def cargar_municipios():
        return leer_geojson("municipios").to_crs(epsg=4326)

    gdf = cargar_municipios()
    cy = gdf.geometry.centroid.y.mean()
    cx = gdf.geometry.centroid.x.mean()

    m = leafmap.Map(center=[cy, cx], zoom=9, draw_control=False,
                    measure_control=False, fullscreen_control=False)
    m.add_basemap("CartoDB.DarkMatter")
    m.add_gdf(
        gdf,
        layer_name="Municipios del AOI",
        style={"color": "#52b788", "fillColor": "#52b788",
               "fillOpacity": 0.15, "weight": 2},
        hover_style={"fillOpacity": 0.40},
    )
    m.to_streamlit(height=420)
    st.caption("Cartagena del Chairá (norte) y San Vicente del Caguán (sur). "
               "IGAC · Marco Geoestadístico Nacional.")

st.divider()

# ── 3. Disponibilidad Sentinel-2 ────────────────────────────────────────────
st.subheader("Disponibilidad de escenas Sentinel-2 por trimestre")

if warn_if_missing("disponibilidad"):
    df_s2 = read_csv("disponibilidad")
    col_quarter = next((c for c in df_s2.columns if "quarter" in c.lower()
                        or "trimestre" in c.lower()), df_s2.columns[0])
    col_scenes = next((c for c in df_s2.columns if "scene" in c.lower()
                       or "escena" in c.lower()), df_s2.columns[1])

    fig = px.bar(df_s2, x=col_quarter, y=col_scenes, text=col_scenes,
                 color=col_scenes, color_continuous_scale="Greens",
                 labels={col_quarter: "Trimestre", col_scenes: "Escenas"})
    fig.update_traces(textposition="outside")
    fig.update_layout(coloraxis_showscale=False, xaxis_title="",
                      yaxis_title="Escenas disponibles", height=320)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "La nubosidad sigue la estacionalidad amazónica: el tercer trimestre "
        "(jul–sep) es el más despejado y el segundo (abr–jun) el más nublado. "
        "Las composiciones trimestrales por mediana y percentil 25 permiten "
        "tener observaciones utilizables durante todo el año."
    )

st.divider()

# ── 4. Prevalencia honesta ──────────────────────────────────────────────────
st.subheader("Prevalencia de la clase deforestación — Hansen GFC 2024")

if warn_if_missing("etiqueta") and warn_if_missing("split"):
    stats = estadisticas_etiqueta()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Píxeles en el AOI", fmt_int(stats["total_pixeles"]))
    c2.metric("Píxeles deforestados", fmt_int(stats["positivos"]))
    c3.metric("Prevalencia", fmt_pct(stats["prevalencia"]))
    c4.metric("Hectáreas perdidas", fmt_hectareas(stats["hectareas_deforestadas"]))

    fig = px.pie(
        names=["Cobertura preservada (clase 0)", "Pérdida de cobertura (clase 1)"],
        values=[stats["total_pixeles"] - stats["positivos"], stats["positivos"]],
        hole=0.6,
        color_discrete_sequence=["#2d6a4f", "#d62828"],
    )
    fig.update_traces(textinfo="percent+label",
                      textfont=dict(color="#f8f9fa", size=13))
    fig.update_layout(showlegend=False, margin=dict(t=10, b=10), height=340)
    st.plotly_chart(fig, use_container_width=True)

    takeaway(
        f"Solo {fmt_pct(stats['prevalencia'])} de los píxeles del AOI corresponden a la "
        "clase positiva. Este <strong>desbalance severo</strong> condiciona "
        "tres decisiones del modelado: pérdida focal (Lin et al., 2017), "
        "muestreo balanceado durante el entrenamiento y umbral calibrado en "
        "validación a la prevalencia real, en lugar del umbral 0,5 por defecto."
    )

st.divider()

# ── 5. Fuentes de datos ────────────────────────────────────────────────────
with st.expander("Detalle de las fuentes de datos"):
    st.markdown(
        "- **Sentinel-2 L2A** (Copernicus / ESA). Reflectancia de superficie "
        "corregida atmosféricamente, 13 bandas espectrales, revisita combinada "
        "S2A+S2B de 5 días. Acceso vía STAC Earth-Search sobre el bucket "
        "público `sentinel-cogs` en `us-west-2`.\n"
        "- **Hansen Global Forest Change v1.12** (GLAD, Universidad de Maryland). "
        "Mapas globales de cambio en cobertura arbórea a 30 m, actualizados "
        "anualmente. Capa `lossyear` para identificar el año de pérdida.\n"
        "- **SMByC-IDEAM**. Boletines trimestrales de alertas tempranas para "
        "validación cualitativa y contextualización institucional.\n"
        "- **IGAC**. Límites administrativos para acotar el AOI y agregar por "
        "municipio."
    )
