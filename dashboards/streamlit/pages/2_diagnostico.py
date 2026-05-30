"""Diagnostico espacial: I de Moran, semivariograma, particion por bloques y
optimismo entre validacion aleatoria y espacial."""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.express as px
import pandas as pd

from utils import (read_text, read_json, warn_if_missing, url,
                   raster_a_overlay)
from theme import (aplicar_tema_plotly, aplicar_estilos_streamlit,
                   takeaway, lead, CMAP_SPLIT, COLORES, COLORES_MODELO)


aplicar_tema_plotly()
aplicar_estilos_streamlit()

st.title("Diagnóstico espacial")
lead(
    "Los píxeles de teledetección no son independientes: vecinos comparten "
    "condiciones de iluminación, atmósfera y cobertura. Si validamos con un "
    "muestreo aleatorio, los conjuntos de entrenamiento y prueba quedan "
    "entrelazados y el desempeño se sobreestima. Esta página cuantifica la "
    "autocorrelación, muestra la partición que la controla y demuestra "
    "empíricamente el sesgo."
)

# ── 1. Indicadores del diagnostico ──────────────────────────────────────────
st.subheader("Cuánto se parecen los píxeles vecinos")

texto = read_text("diagnosticos")
moran = re.search(r"Moran I[^:]*:\s*([-\d.]+)", texto)
rango = re.search(r"Rango[^:]*:\s*([\d.]+)", texto)
bloque = re.search(r"Bloque[^:]*:\s*([\d.]+)", texto)

c1, c2, c3 = st.columns(3)
c1.metric("I de Moran (etiqueta)",
          moran.group(1) if moran else "—",
          help="Cercano a 1 → autocorrelación positiva fuerte. "
               "El valor observado, 0.84, indica que un píxel deforestado "
               "tiende a estar rodeado por otros deforestados.")
c2.metric("Rango espacial",
          f"{rango.group(1)} m" if rango else "—",
          help="Distancia a la que la correlación se vuelve despreciable. "
               "Es el insumo principal para dimensionar los bloques.")
c3.metric("Lado del bloque de validación",
          f"{bloque.group(1)} km" if bloque else "—",
          help="Mayor que el rango espacial para que la distancia mínima "
               "entre entrenamiento y prueba sea suficiente.")

takeaway(
    "El I de Moran de 0.84 confirma autocorrelación positiva fuerte; el rango "
    "del semivariograma (3.4 km) define un piso para el tamaño del bloque. "
    "Por eso la partición es de <strong>5 km</strong>, no de 1 km ni aleatoria por píxel."
)

st.divider()

# ── 2. Mapa de la particion ──────────────────────────────────────────────────
st.subheader("Partición espacial 70 / 15 / 15 por bloque")

col_mapa, col_leyenda = st.columns([3, 1])
with col_leyenda:
    st.markdown(
        f"<div style='line-height: 2.2;'>"
        f"<span style='color:{CMAP_SPLIT[1]};font-weight:600'>■</span> Entrenamiento (70 %)<br>"
        f"<span style='color:{CMAP_SPLIT[2]};font-weight:600'>■</span> Validación (15 %)<br>"
        f"<span style='color:{CMAP_SPLIT[3]};font-weight:600'>■</span> Prueba (15 %)"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.info(
        "Los bloques se asignan **completos** a cada conjunto. La distancia "
        "mínima entre prueba y entrenamiento supera el rango del semivariograma."
    )

with col_mapa:
    if warn_if_missing("split"):
        overlay, centro = raster_a_overlay(
            url("split"),
            colormap=CMAP_SPLIT,
            vmin=0, vmax=3, opacidad=0.75,
            layer_name="Partición",
        )
        m = folium.Map(location=centro, zoom_start=10, tiles="CartoDB dark_matter")
        overlay.add_to(m)
        folium.LayerControl(collapsed=False).add_to(m)
        st_folium(m, height=460, use_container_width=True,
                  returned_objects=[])

st.divider()

# ── 3. Optimismo: el resultado documentado en la propuesta ───────────────────
st.subheader("Optimismo de la validación aleatoria vs. espacial")

st.markdown(
    "La propuesta compromete reportar las métricas bajo dos esquemas en paralelo. "
    "La diferencia es, por sí misma, un resultado del proyecto: cuantifica el "
    "sesgo de no controlar por dependencia espacial."
)

compare = read_json("compare_cv")
if compare:
    filas = []
    for modelo, datos in compare.get("models", {}).items():
        for metrica, gap in datos.get("optimism", {}).items():
            filas.append({"Modelo": modelo.replace("_", " ").title(),
                          "Métrica": metrica,
                          "Aleatoria": datos["random"][metrica]["mean"],
                          "Espacial":  datos["spatial"][metrica]["mean"],
                          "Optimismo (Δ)": gap})
    if filas:
        df = pd.DataFrame(filas)
        metricas_orden = ["precision", "recall", "f1", "iou", "auc_roc", "auc_pr"]
        df["Métrica"] = pd.Categorical(df["Métrica"], categories=metricas_orden,
                                       ordered=True)
        df = df.sort_values(["Modelo", "Métrica"])

        c_a, c_b = st.columns([3, 2])
        with c_a:
            df_plot = df.melt(id_vars=["Modelo", "Métrica"],
                              value_vars=["Aleatoria", "Espacial"],
                              var_name="Esquema CV", value_name="Valor")
            fig = px.bar(df_plot, x="Métrica", y="Valor", color="Esquema CV",
                         facet_col="Modelo", barmode="group", text_auto=".3f",
                         color_discrete_map={"Aleatoria": COLORES["alerta"],
                                             "Espacial": COLORES["bosque"]})
            fig.update_layout(yaxis_range=[0, 1.05], height=420,
                              legend_title_text="")
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            st.plotly_chart(fig, use_container_width=True)
        with c_b:
            tabla = df.pivot_table(index=["Modelo", "Métrica"],
                                   values="Optimismo (Δ)", aggfunc="first")
            st.dataframe(
                tabla.style.format("{:+.3f}")
                     .background_gradient(cmap="Reds", vmin=0, vmax=0.10),
                use_container_width=True,
            )
            takeaway(
                "El optimismo es máximo en <strong>AUC-PR</strong> "
                "(+0.05 a +0.07) y en <strong>recall</strong> (+0.05 a +0.06). "
                "Reportar solo la validación aleatoria sobreestimaría el "
                "desempeño esperado en zonas no observadas."
            )

st.divider()

with st.expander("¿Por qué este sesgo importa? — Roberts et al. 2017, Karasiak et al. 2022"):
    st.markdown(
        "El supuesto de independencia de la validación cruzada estándar se "
        "viola cuando los datos tienen estructura espacial. Cualquier modelo "
        "evaluado bajo esa contaminación rinde bien sobre conjuntos similares "
        "a los de entrenamiento, pero su capacidad de generalización a zonas "
        "no observadas se sobreestima. El procedimiento honesto es la "
        "partición por bloques de tamaño mayor o igual al rango espacial, y "
        "el reporte paralelo de ambas estimaciones para documentar el sesgo."
    )
