"""Tablero del proyecto amazonia-deforestation.

Entrypoint del tablero. Resuelve el conflicto de versiones de PROJ entre
rasterio y pyproj en Windows, configura la navegacion con titulos en espanol
con tildes, define el tema visual y renderiza la pagina de inicio (resumen
ejecutivo del proyecto).

Ejecucion desde la raiz del repositorio:
    streamlit run dashboards/streamlit/streamlit_app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ── Fix PROJ version conflict (Windows) ─────────────────────────────────────
# rasterio 1.4+ trae PROJ 9.6; pyproj 3.6/3.7 trae PROJ 9.3-9.5. Al importarse,
# pyproj sobreescribe PROJ_DATA con su directorio y rasterio falla al resolver
# cualquier EPSG. Solucion: apuntar PROJ_DATA al proj_data de rasterio antes de
# cualquier operacion de CRS.
try:
    import rasterio as _rio
    import pyproj as _pp
    _proj_dir = str((Path(_rio.__file__).parent / "proj_data").resolve())
    if (Path(_proj_dir) / "proj.db").exists():
        os.environ["PROJ_DATA"] = _proj_dir
        os.environ["PROJ_LIB"] = _proj_dir
        _pp.datadir.set_data_dir(_proj_dir)
except Exception:
    pass
# ────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

st.set_page_config(
    page_title="Amazonía colombiana — Detección temprana de deforestación",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded",
)


def home():
    """Pagina de inicio: resumen ejecutivo con narrativa, no listado de datos."""
    from utils import (load_config, read_json, estadisticas_etiqueta,
                       fmt_hectareas, fmt_pct, NOMBRES_MODELO)
    from theme import (aplicar_tema_plotly, aplicar_estilos_streamlit,
                       takeaway, lead, COLORES_MODELO)
    import plotly.express as px
    import pandas as pd

    aplicar_tema_plotly()
    aplicar_estilos_streamlit()

    cfg = load_config()
    aoi = cfg["aoi"]
    año = cfg["temporal"]["target_year"]

    st.title("Detección temprana de deforestación en la Amazonía colombiana")
    lead(
        f"Sentinel-2 y aprendizaje automático sobre {aoi['target_area_km2']:,} km² "
        f"del Caquetá ({', '.join(aoi['municipalities'])}). Cuatro modelos candidatos "
        f"comparados con bootstrap espacial por bloques (B=1.000) y prueba de McNemar "
        f"pareada. Año objetivo: {año}.".replace(",", ".")
    )

    # ── Hero metrics: el dolor, no la solucion ──────────────────────────────
    stats = estadisticas_etiqueta()
    eval_ens = read_json("eval_ensemble") or {}
    eval_xgb = read_json("eval_xgboost") or {}

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            "Hectáreas perdidas en el AOI (Hansen 2024)",
            fmt_hectareas(stats["hectareas_deforestadas"]),
            help="Referencia institucional para evaluar el modelo.",
        )
    with c2:
        st.metric(
            "Prevalencia clase positiva",
            fmt_pct(stats["prevalencia"]),
            help="Fracción de píxeles con pérdida de cobertura. El desbalance "
                 "severo motiva pérdida focal y bloques espaciales.",
        )
    with c3:
        f1 = eval_ens.get("pixel", {}).get("f1")
        delta_xgb = (eval_ens.get("pixel", {}).get("f1") or 0) - (eval_xgb.get("pixel", {}).get("f1") or 0)
        st.metric(
            "F1 píxel — Modelo final (ensamble)",
            f"{f1:.3f}" if f1 else "—",
            delta=f"{delta_xgb:+.3f} vs baseline XGBoost" if f1 else None,
        )
    with c4:
        iou = eval_ens.get("polygon", {}).get("mean_iou_matched")
        st.metric(
            "IoU polígono emparejado (ensamble)",
            f"{iou:.3f}" if iou else "—",
            delta="≥ 0.40 ✓" if iou and iou >= 0.40 else "criterio < 0.40",
            delta_color="normal" if iou and iou >= 0.40 else "inverse",
        )

    takeaway(
        "La región del arco amazónico colombiano perdió <strong>107.000 hectáreas en 2024</strong>, "
        "35 % más que en 2023 según el IDEAM. El proyecto entrena un sistema de detección "
        "temprana sobre un núcleo activo del Caquetá y demuestra que el ensamble por "
        "promedio ponderado XGBoost + U-Net supera a los individuales en la mayoría "
        "de métricas operativas, manteniendo intervalos de confianza espaciales honestos."
    )

    st.divider()

    # ── Comparacion sintetica: una vista de los 4 modelos sobre la metrica clave ──
    st.subheader("Cuatro modelos candidatos — F1 píxel sobre prueba")
    filas = []
    for key in ("xgboost", "random_forest", "unet", "ensemble"):
        ev = read_json("eval_" + key)
        if ev and "pixel" in ev:
            filas.append({
                "modelo_id": key,
                "Modelo": NOMBRES_MODELO[key],
                "F1": ev["pixel"]["f1"],
                "AUC-PR": ev["pixel"]["auc_pr"],
                "Polygon F1": ev["polygon"]["polygon_f1"],
            })
    if filas:
        df = pd.DataFrame(filas).set_index("Modelo")
        c_a, c_b = st.columns([2, 1])
        with c_a:
            df_plot = df.reset_index().melt(id_vars=["Modelo", "modelo_id"],
                                            var_name="Métrica", value_name="Valor")
            color_map = {NOMBRES_MODELO[k]: COLORES_MODELO[k]
                         for k in ("xgboost", "random_forest", "unet", "ensemble")}
            fig = px.bar(df_plot, x="Métrica", y="Valor", color="Modelo",
                         barmode="group", text_auto=".3f",
                         color_discrete_map=color_map)
            fig.update_layout(yaxis_range=[0, max(0.65, df_plot["Valor"].max() * 1.15)],
                              legend_title_text="", height=380)
            st.plotly_chart(fig, use_container_width=True)
        with c_b:
            mejor = df["F1"].idxmax()
            st.markdown(
                f"**Ganador F1 píxel.** {mejor}\n\n"
                f"**Ganador AUC-PR.** {df['AUC-PR'].idxmax()}\n\n"
                f"**Ganador Polygon F1.** {df['Polygon F1'].idxmax()}\n\n"
                f"\nLa elección del modelo final depende de la métrica operativa. "
                f"Para alertas, polygon F1 es lo más útil; para ranking de bloques, "
                f"AUC-PR; para totales agregados, el coeficiente de concordancia de Lin."
            )

    st.divider()

    # ── Navegacion guiada por narrativa ──
    st.subheader("Ruta de lectura del tablero")
    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown("**1. Contexto y datos**\n\n"
                    "Por qué importa Caquetá. Disponibilidad de Sentinel-2 por "
                    "trimestre, prevalencia honesta de la etiqueta y mapa del AOI.")
        st.markdown("**2. Diagnóstico espacial**\n\n"
                    "I de Moran, variograma y por qué validamos por bloques. Optimismo "
                    "de la validación aleatoria comparado con la espacial.")
    with g2:
        st.markdown("**3. Modelos y métricas**\n\n"
                    "Comparación de los cuatro candidatos con bootstrap espacial, "
                    "McNemar pareado y CCC de Lin.")
        st.markdown("**4. Mapa de predicciones**\n\n"
                    "Capas conmutables (cinco modelos + Hansen), umbral ajustable, "
                    "TP/FP/FN sobre el conjunto de prueba.")
    with g3:
        st.markdown("**5. Análisis por municipio**\n\n"
                    "Hectáreas detectadas vs Hansen por municipio. Calibración "
                    "agregada y concordancia.")
        st.markdown("**6. Despliegue Big Data**\n\n"
                    "Arquitectura AWS, consultas Athena ejecutadas, tiempos Lambda "
                    "y benchmark t3.medium.")


pagina_inicio = st.Page(home, title="Inicio", icon="🏠", default=True, url_path="inicio")
pagina_contexto = st.Page("pages/1_contexto.py", title="Contexto y datos",
                          icon="🌎", url_path="contexto")
pagina_diagnostico = st.Page("pages/2_diagnostico.py", title="Diagnóstico espacial",
                             icon="🧭", url_path="diagnostico")
pagina_modelos = st.Page("pages/3_modelos.py", title="Modelos y métricas",
                         icon="📊", url_path="modelos")
pagina_mapa = st.Page("pages/4_mapa.py", title="Mapa de predicciones",
                      icon="🗺️", url_path="mapa")
pagina_municipios = st.Page("pages/5_municipios.py", title="Análisis por municipio",
                            icon="🏘️", url_path="municipios")
pagina_despliegue = st.Page("pages/6_despliegue.py", title="Despliegue Big Data",
                            icon="☁️", url_path="despliegue")

nav = st.navigation([
    pagina_inicio,
    pagina_contexto,
    pagina_diagnostico,
    pagina_modelos,
    pagina_mapa,
    pagina_municipios,
    pagina_despliegue,
], position="sidebar")

# Pie de pagina compartido en el sidebar
with st.sidebar:
    st.divider()
    st.caption("**Equipo CLRCV** · Maestría en Ciencia de Datos y Analítica")
    st.caption("Datos: Sentinel-2 L2A · Hansen GFC v1.12 · IDEAM SMByC")

nav.run()
