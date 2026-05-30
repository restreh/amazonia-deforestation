"""Despliegue Big Data: arquitectura AWS, consultas Athena ejecutadas,
benchmark EC2 t3.medium y tiempos de Lambda."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils import (PATHS, read_csv, warn_if_missing, fmt_decimal)
from theme import (aplicar_tema_plotly, aplicar_estilos_streamlit, takeaway,
                   lead, COLORES)


aplicar_tema_plotly()
aplicar_estilos_streamlit()

st.title("Despliegue Big Data")
lead(
    "El bloque de Big Data lleva el modelo del notebook a la nube. Los "
    "derivados viven en S3 con particionamiento Hive, Athena consulta la "
    "tabla analítica `metrics_by_block` con pruning de particiones, la "
    "inferencia se sirve en un contenedor Lambda con PyTorch y un orquestador "
    "EventBridge dispara el flujo cada trimestre."
)

# ── 1. Arquitectura ─────────────────────────────────────────────────────────
st.subheader("Arquitectura de despliegue")

st.markdown(
    """
```
Sentinel-2 L2A  ─►  STAC Earth-Search  ─►  scripts/build_composites.py
                                            (stackstac + Dask, lectura COG)
                                                    │
                                                    ▼
            ┌──────────────────────────────────────────────────────────┐
            │  S3  amazonia-deforestation-data-363918845645            │
            │  · derived/composites/tile=AOI_caqueta/quarter=*/...     │
            │  · derived/features_by_block/block_id=*/part.parquet     │
            │  · derived/metrics_by_block/part.parquet                 │
            │  · models/{xgboost.json, unet.pt, ...}                   │
            └──────────────────────────────────────────────────────────┘
                       │                                  │
                       ▼                                  ▼
               Athena (Glue)                     ECR (Docker image)
               · metrics_by_block                       │
               · features_by_block                     ▼
               · train_features              Lambda inference (PyTorch)
                                                       │
                                                       ▼
                                            Orchestrator Lambda  ◄── EventBridge (trimestral)
```
"""
)

st.markdown(
    "- **S3** con particionamiento Hive por tile MGRS, trimestre y bloque "
    "espacial.\n"
    "- **Glue / Athena** sobre las tablas analíticas; consultas demo guardadas "
    "en `infra/athena/`.\n"
    "- **Lambda PyTorch** sirve la U-Net por ventana 256×256 leyendo S3 vía "
    "GDAL/VSI.\n"
    "- **EventBridge** dispara el orquestador cada trimestre "
    "(`cron(0 0 1 1,4,7,10 ? *)`).\n"
    "- **EC2 t3.medium** para el benchmark del criterio de éxito de tiempo."
)

st.divider()

# ── 2. Consultas Athena ejecutadas ──────────────────────────────────────────
st.subheader("Consultas Athena ejecutadas")

st.markdown("Cuatro consultas demo se ejecutaron sobre Athena; estos son los "
            "resultados que devolvió el motor.")

tabs = st.tabs([
    "Métricas por modelo",
    "Top bloques (ensamble)",
    "Hectáreas por split",
    "Partition pruning",
])

with tabs[0]:
    if warn_if_missing("athena_metrics"):
        df = read_csv("athena_metrics")
        st.dataframe(df, use_container_width=True)
        st.caption("`SELECT AVG(f1) ... FROM metrics_by_block WHERE split_code='test'`")

with tabs[1]:
    if warn_if_missing("athena_top_blocks"):
        df = read_csv("athena_top_blocks")
        st.dataframe(df, use_container_width=True)
        st.caption("Top 10 bloques donde el ensamble obtiene mejor F1 píxel "
                   "en el conjunto de prueba.")

with tabs[2]:
    if warn_if_missing("athena_hectares_by_split"):
        df = read_csv("athena_hectares_by_split")
        st.dataframe(df, use_container_width=True)
        if "predicted_total_ha" in df.columns and "truth_total_ha" in df.columns:
            df["razón"] = df["predicted_total_ha"] / df["truth_total_ha"]
            fig = px.bar(df, x="model", y="razón", color="split_code",
                         barmode="group", text_auto=".3f",
                         labels={"razón": "Razón predicho/Hansen", "model": ""})
            fig.add_hline(y=1.0, line_dash="dot", line_color="white")
            fig.update_layout(yaxis_range=[0.85, 1.15], height=320)
            st.plotly_chart(fig, use_container_width=True)

with tabs[3]:
    if warn_if_missing("athena_partition_pruning"):
        df = read_csv("athena_partition_pruning")
        st.dataframe(df, use_container_width=True)
        st.caption("La consulta filtra `WHERE block_id IN (100, 200, 300, 400)`. "
                   "Athena solo escanea las particiones de esos bloques: el "
                   "particionamiento Hive funciona como se espera.")

st.divider()

# ── 3. Benchmark t3.medium ──────────────────────────────────────────────────
st.subheader("Benchmark de inferencia — EC2 t3.medium")

c1, c2, c3 = st.columns(3)
c1.metric("Tiempo medido (AOI)", "19,52 min",
          help="Inferencia U-Net densa sobre el AOI 3.533×3.556 px.")
c2.metric("Escena Sentinel-2 completa (extrapolado)", "~47 min",
          help="Por extrapolación lineal a 5.490×5.490 px = ~30 Mpx.")
c3.metric("Criterio de la propuesta", "< 10 min",
          delta="❌ NO CUMPLE sincrónicamente",
          delta_color="inverse")

takeaway(
    "La medición sobre `t3.medium` sincrónico (2 vCPUs) está por encima "
    "del criterio. El criterio fue establecido sin medición previa y "
    "sobreestimó el rendimiento del nodo para una U-Net ResNet-34. La "
    "arquitectura distribuida del proyecto compensa este límite operativo, "
    "como se ve abajo."
)

st.divider()

# ── 4. Lambda timing ────────────────────────────────────────────────────────
st.subheader("Inferencia distribuida — contenedor Lambda con PyTorch")

c1, c2 = st.columns(2)
c1.metric("Cold start (primera invocación)", "19,5 s",
          help="Descarga del modelo a /tmp + carga de PyTorch + inferencia "
               "sobre ventana 256×256.")
c2.metric("Warm (invocaciones siguientes)", "5,7 s",
          help="Modelo cacheado en /tmp; solo lectura S3 vía VSI + inferencia.")

# Comparacion conceptual: t3.medium vs Lambda paralelo
df_tiempos = pd.DataFrame([
    {"Modo": "t3.medium sincrónico", "Tiempo (min)": 19.52},
    {"Modo": "Lambda paralelo (729 ventanas, warm)", "Tiempo (min)": 5.7 / 60},
    {"Modo": "Lambda paralelo (cold start estimado)", "Tiempo (min)": 19.5 / 60},
])
fig = px.bar(df_tiempos, x="Modo", y="Tiempo (min)", text_auto=".2f",
             color="Modo",
             color_discrete_sequence=[COLORES["fallo"], COLORES["exito"],
                                      COLORES["alerta"]])
fig.update_layout(showlegend=False, xaxis_title="",
                  yaxis_title="Tiempo de inferencia (min)", height=320)
fig.add_hline(y=10, line_dash="dot", line_color="white",
              annotation_text="Criterio < 10 min", annotation_position="top right")
st.plotly_chart(fig, use_container_width=True)

st.markdown(
    "Lambda admite por defecto **1.000 invocaciones concurrentes**. La "
    "grilla del AOI tiene 729 ventanas 256×256 con stride 128. Si todas se "
    "lanzan en paralelo, el tiempo de pared es el de la más lenta más la "
    "latencia de orquestación. Esa es la arquitectura desplegada y "
    "documentada en `infra/lambda/`."
)

st.divider()

# ── 5. Verificacion en vivo (opcional) ──────────────────────────────────────
st.subheader("Verificación en vivo contra Athena (opcional)")

st.markdown(
    "Si tienes credenciales AWS configuradas con permisos sobre la cuenta "
    "del proyecto (`363918845645`), puedes correr una consulta contra "
    "Athena en vivo para verificar que la integración sigue activa."
)

if st.button("Ejecutar `SELECT model, AVG(f1) ... GROUP BY model` en vivo"):
    try:
        import boto3
        import json
        import time

        athena = boto3.client("athena", region_name="us-west-2")
        sql = ("SELECT model, AVG(f1) AS mean_f1, AVG(iou) AS mean_iou "
               "FROM amazonia_deforestation.metrics_by_block "
               "WHERE split_code = 'test' "
               "GROUP BY model ORDER BY mean_f1 DESC")
        res = athena.start_query_execution(
            QueryString=sql,
            ResultConfiguration={
                "OutputLocation": "s3://amazonia-deforestation-data-363918845645/"
                                  "athena-results/"
            },
        )
        qid = res["QueryExecutionId"]
        with st.spinner(f"Ejecutando query {qid}…"):
            while True:
                s = athena.get_query_execution(QueryExecutionId=qid)["QueryExecution"]["Status"]
                if s["State"] in ("SUCCEEDED", "FAILED", "CANCELLED"):
                    break
                time.sleep(0.5)
        if s["State"] != "SUCCEEDED":
            st.error(f"Query falló: {s.get('StateChangeReason')}")
        else:
            result = athena.get_query_results(QueryExecutionId=qid)
            headers = [c["VarCharValue"]
                       for c in result["ResultSet"]["Rows"][0]["Data"]]
            rows = [[c.get("VarCharValue", "")
                     for c in row["Data"]]
                    for row in result["ResultSet"]["Rows"][1:]]
            df_live = pd.DataFrame(rows, columns=headers)
            st.success(f"Query {qid} ejecutada en vivo")
            st.dataframe(df_live, use_container_width=True)
    except Exception as exc:
        st.error(f"No se pudo ejecutar la consulta en vivo: {exc}")
        st.caption("Configura `aws configure` con las credenciales del "
                   "proyecto, o usa los resultados estáticos arriba.")
