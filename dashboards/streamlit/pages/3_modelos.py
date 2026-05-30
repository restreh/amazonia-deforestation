"""Modelos y metricas: cuatro candidatos + control ImageNet, criterios de exito,
bootstrap espacial, McNemar pareado, CCC de Lin e importancia de atributos."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils import (load_config, read_json, read_csv, warn_if_missing,
                   NOMBRES_MODELO, MODELOS_CANDIDATOS, MODELO_CONTROL,
                   fmt_decimal)
from theme import (aplicar_tema_plotly, aplicar_estilos_streamlit, takeaway,
                   lead, COLORES, COLORES_MODELO)


aplicar_tema_plotly()
aplicar_estilos_streamlit()

st.title("Modelos y métricas")
lead(
    "Cuatro modelos candidatos y un experimento de control. La elección del "
    "modelo final se hace contra evidencia estadística pareada (McNemar) y "
    "contra intervalos de confianza con bootstrap espacial por bloques "
    "(B = 1.000) que respeta la autocorrelación, no contra puntos individuales."
)

cfg = load_config()
sc = cfg["evaluation"]["success_criteria"]
F1_MIN = sc["f1_pixel_min"]
IOU_MIN = sc["iou_polygon_min"]


# ── Carga ordenada de evaluaciones ──────────────────────────────────────────
def evaluacion(modelo_id: str) -> dict | None:
    return read_json("eval_" + modelo_id)


modelos = [(m, evaluacion(m)) for m in MODELOS_CANDIDATOS + [MODELO_CONTROL]]
modelos = [(m, ev) for m, ev in modelos if ev]
if not modelos:
    st.warning("No hay reportes de evaluación. Corre `evaluate_baseline.py` "
               "para cada modelo.")
    st.stop()

# ── 1. Tabla canonica de metricas ────────────────────────────────────────────
st.subheader("Comparación principal — prueba a prevalencia real 1,83 %")


def fila(modelo_id: str, ev: dict) -> dict:
    p = ev.get("pixel", {})
    po = ev.get("polygon", {})
    return {
        "Modelo": NOMBRES_MODELO[modelo_id],
        "_id": modelo_id,
        "F1 píxel": p.get("f1"),
        "Precision": p.get("precision"),
        "Recall": p.get("recall"),
        "IoU píxel": p.get("iou"),
        "AUC-PR": p.get("auc_pr"),
        "AUC-ROC": p.get("auc_roc"),
        "Polygon F1": po.get("polygon_f1"),
        "Polygon precision": po.get("polygon_precision"),
        "Polygon recall": po.get("polygon_recall"),
        "Mean IoU emparejado": po.get("mean_iou_matched"),
        "Umbral": ev.get("threshold"),
    }


df = pd.DataFrame([fila(m, ev) for m, ev in modelos])
df_mostrar = df.drop(columns=["_id"]).set_index("Modelo")


def estilo_celda(val, col_name):
    if not isinstance(val, (int, float)):
        return ""
    if col_name == "F1 píxel" and val < F1_MIN:
        return "color:#ff6b6b"
    if col_name == "Mean IoU emparejado" and val >= IOU_MIN:
        return "color:#52b788;font-weight:600"
    return ""


# Resaltar el mejor por columna (excepto Umbral)
def resaltar_max(s):
    if s.name == "Umbral" or s.dtype == object:
        return [""] * len(s)
    mejor = s.max()
    return ["background-color: rgba(82, 183, 136, 0.18); font-weight: 600;"
            if v == mejor else "" for v in s]


styled = (df_mostrar.style
          .format("{:.3f}", na_rep="—")
          .apply(resaltar_max, axis=0)
          .apply(lambda col: [estilo_celda(v, col.name) for v in col], axis=0))
st.dataframe(styled, use_container_width=True)
st.caption("En verde, el mejor valor por métrica. El umbral viene calibrado "
           "en validación maximizando F1 a la prevalencia real.")

st.divider()

# ── 2. Criterios de exito de la propuesta ────────────────────────────────────
st.subheader(f"Criterios de éxito  ·  F1 píxel ≥ {F1_MIN}  ·  IoU polígono ≥ {IOU_MIN}")

cols = st.columns(len(modelos))
for col, (mid, ev) in zip(cols, modelos):
    p = ev.get("pixel", {})
    po = ev.get("polygon", {})
    f1v = p.get("f1") or 0.0
    iouv = po.get("mean_iou_matched") or 0.0
    f1_ok = f1v >= F1_MIN
    iou_ok = iouv >= IOU_MIN
    estado_f1 = "✅" if f1_ok else "❌"
    estado_iou = "✅" if iou_ok else "❌"
    col.markdown(f"**{NOMBRES_MODELO[mid]}**")
    col.markdown(f"{estado_f1}  F1 píxel · `{f1v:.3f}`")
    col.markdown(f"{estado_iou}  IoU polígono · `{iouv:.3f}`")

takeaway(
    "El criterio <strong>IoU polígono ≥ 0,40</strong> lo cumplen los cuatro candidatos. "
    "El criterio <strong>F1 píxel ≥ 0,70</strong> no lo cumple ninguno y es coherente con la "
    "literatura comparable para deforestación amazónica con etiqueta Hansen "
    "(rango 0,45–0,65 en Adarme et al. 2022 y Maretto et al. 2020). Esta es "
    "una observación metodológica documentada en el informe, no un fracaso."
)

st.divider()

# ── 3. Comparacion visual ───────────────────────────────────────────────────
st.subheader("Barras comparativas — operativas (F1, IoU, AUC-PR)")
metricas_visual = ["F1 píxel", "Recall", "Precision", "AUC-PR",
                   "Polygon F1", "Polygon recall", "Mean IoU emparejado"]
df_long = df.melt(id_vars="Modelo", value_vars=metricas_visual,
                  var_name="Métrica", value_name="Valor").dropna()

color_map = {NOMBRES_MODELO[k]: COLORES_MODELO[k]
             for k in MODELOS_CANDIDATOS + [MODELO_CONTROL]}
fig = px.bar(df_long, x="Métrica", y="Valor", color="Modelo",
             barmode="group", text_auto=".3f", color_discrete_map=color_map)
fig.update_layout(yaxis_range=[0, 0.7], height=380, legend_title_text="")
fig.add_hline(y=F1_MIN, line_dash="dot", line_color=COLORES["fallo"],
              annotation_text=f"F1 píxel ≥ {F1_MIN}", annotation_position="top left")
fig.add_hline(y=IOU_MIN, line_dash="dot", line_color=COLORES["exito"],
              annotation_text=f"IoU polígono ≥ {IOU_MIN}", annotation_position="bottom left")
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── 4. Bootstrap espacial por bloques ────────────────────────────────────────
st.subheader("Bootstrap espacial por bloques — IC 95 %")

st.markdown(
    "Remuestreo de bloques de prueba con reposición, B = 1.000 iteraciones. "
    "Los intervalos respetan la autocorrelación; el bootstrap píxel a píxel "
    "los habría dado mucho más estrechos pero engañosamente."
)

boot = read_json("bootstrap_spatial")
if boot:
    metricas_boot = ["f1", "iou", "auc_pr"]
    filas = []
    for mid, datos in boot.get("models", {}).items():
        for met in metricas_boot:
            if met in datos and datos[met]:
                filas.append({
                    "Modelo": NOMBRES_MODELO.get(mid, mid),
                    "Métrica": {"f1": "F1 píxel", "iou": "IoU píxel",
                                "auc_pr": "AUC-PR"}[met],
                    "p2.5":  datos[met]["p2_5"],
                    "p50":   datos[met]["p50"],
                    "p97.5": datos[met]["p97_5"],
                })
    if filas:
        df_boot = pd.DataFrame(filas)
        for metrica_nombre in ["F1 píxel", "IoU píxel", "AUC-PR"]:
            sub = df_boot[df_boot["Métrica"] == metrica_nombre].sort_values("p50")
            fig = go.Figure()
            for _, row in sub.iterrows():
                fig.add_trace(go.Scatter(
                    x=[row["p2.5"], row["p97.5"]],
                    y=[row["Modelo"], row["Modelo"]],
                    mode="lines",
                    line=dict(color=color_map.get(row["Modelo"], COLORES["alerta"]),
                              width=6),
                    showlegend=False,
                    hovertemplate=(f"{row['Modelo']}<br>"
                                   f"p2.5 = {row['p2.5']:.3f}<br>"
                                   f"p97.5 = {row['p97.5']:.3f}<extra></extra>"),
                ))
                fig.add_trace(go.Scatter(
                    x=[row["p50"]], y=[row["Modelo"]],
                    mode="markers",
                    marker=dict(color="white", size=11, line=dict(width=1.5,
                                                                  color=color_map.get(row["Modelo"], COLORES["alerta"]))),
                    showlegend=False,
                    hovertemplate=f"mediana = {row['p50']:.3f}<extra></extra>",
                ))
            fig.update_layout(title=f"<b>{metrica_nombre}</b>",
                              xaxis_title="Valor",
                              height=240,
                              margin=dict(t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

    takeaway(
        "Las IC al 95 % de F1, IoU y AUC-PR se <strong>traslapan</strong> entre XGBoost, "
        "U-Net y ensamble. El ensamble es el mejor en mediana, pero la "
        "incertidumbre por heterogeneidad entre bloques abarca a los otros. "
        "Random Forest queda por debajo de forma más marcada."
    )

st.divider()

# ── 5. McNemar pareado ──────────────────────────────────────────────────────
st.subheader("Prueba de McNemar pareada a nivel de píxel")

st.markdown(
    "Compara aciertos pareados de cada par de modelos en el conjunto de "
    "prueba. **Caveat metodológico**: con 1,86 M de píxeles la potencia "
    "estadística es enorme y diferencias minúsculas de accuracy salen "
    "significativas. Por eso el bootstrap espacial es el control de realidad."
)

mc = read_json("mcnemar")
if mc:
    pares = mc.get("pairs", [])
    df_mc = pd.DataFrame([{
        "Par": f"{NOMBRES_MODELO.get(p['a'], p['a'])}  vs  {NOMBRES_MODELO.get(p['b'], p['b'])}",
        "Δ accuracy": p["acc_a"] - p["acc_b"],
        "n01": p["n01"], "n10": p["n10"],
        "p-valor exacto": p["p_exact"],
        "Significativo 5%": "✅" if p["significant_at_0.05"] else "—",
        "Ganador": NOMBRES_MODELO.get(p.get("winner") or "", "—"),
    } for p in pares])
    st.dataframe(
        df_mc.style.format({"Δ accuracy": "{:+.5f}",
                            "p-valor exacto": "{:.2e}"})
            .background_gradient(subset=["p-valor exacto"], cmap="Greens_r"),
        use_container_width=True,
    )

st.divider()

# ── 6. CCC de Lin: calibracion agregada ─────────────────────────────────────
st.subheader("Coeficiente de concordancia de Lin — hectáreas por bloque vs Hansen")

st.markdown(
    "El CCC combina exactitud y precisión en una sola métrica para "
    "comparar dos series. Aplicado a hectáreas predichas por bloque "
    "(modelo) contra Hansen, dice qué tan bien el modelo **reproduce** la "
    "referencia institucional cuando se agregan los píxeles."
)

ccc = read_json("concordance_lin")
if ccc:
    rows = []
    for mid, d in ccc.get("models", {}).items():
        rows.append({
            "Modelo": NOMBRES_MODELO.get(mid, mid),
            "CCC bloque": d.get("ccc_block"),
            "Pearson bloque": d.get("pearson_block"),
            "Total predicho (ha)": d.get("total_predicted_ha_aoi"),
            "Total Hansen (ha)": d.get("total_truth_ha_aoi"),
        })
    df_ccc = pd.DataFrame(rows)
    df_ccc["Razón predicho/Hansen"] = (df_ccc["Total predicho (ha)"]
                                       / df_ccc["Total Hansen (ha)"])

    c_a, c_b = st.columns([3, 2])
    with c_a:
        fig = px.bar(df_ccc.sort_values("CCC bloque", ascending=True),
                     x="CCC bloque", y="Modelo", orientation="h",
                     text_auto=".3f", color="Modelo",
                     color_discrete_map=color_map)
        fig.update_layout(height=320, showlegend=False,
                          xaxis_range=[0.8, 1.0])
        st.plotly_chart(fig, use_container_width=True)
    with c_b:
        st.dataframe(
            df_ccc.set_index("Modelo")
                  [["CCC bloque", "Razón predicho/Hansen"]]
                  .style.format({"CCC bloque": "{:.3f}",
                                 "Razón predicho/Hansen": "{:.3f}"})
                  .background_gradient(cmap="Greens", subset=["CCC bloque"],
                                       vmin=0.80, vmax=1.0),
            use_container_width=True,
        )
        takeaway(
            "El <strong>U-Net</strong> tiene el mayor CCC por bloque (0,945) y la razón "
            "predicho/Hansen más cercana a 1 (0,979). El ensamble es muy "
            "competitivo en F1 pero sobreestima ~6 %. Es un trade-off real "
            "que vale la pena reportar."
        )

st.divider()

# ── 7. Importancia de atributos ─────────────────────────────────────────────
st.subheader("Importancia de atributos — Top 20 XGBoost")

if warn_if_missing("imp_xgboost"):
    df_imp = read_csv("imp_xgboost")
    col_imp = next((c for c in df_imp.columns if c.lower() in
                    ("importance", "gain", "weight", "score")), df_imp.columns[-1])
    col_feat = next((c for c in df_imp.columns if c.lower() in
                     ("feature", "name", "variable")), df_imp.columns[0])
    top = df_imp.nlargest(20, col_imp).sort_values(col_imp)
    fig = px.bar(top, x=col_imp, y=col_feat, orientation="h",
                 text_auto=".3f", color=col_imp,
                 color_continuous_scale="Greens",
                 labels={col_imp: "Importancia", col_feat: ""})
    fig.update_layout(coloraxis_showscale=False, height=520)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Los atributos contextuales y de GLCM consistentemente "
               "aparecen entre los más informativos para distinguir "
               "deforestación reciente de cobertura preservada.")
