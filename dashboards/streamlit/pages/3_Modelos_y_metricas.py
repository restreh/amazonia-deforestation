"""Modelos y metricas: tabla comparativa, criterios de exito e importancia de atributos."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import plotly.express as px
from utils import load_config, read_json, read_csv, warn_if_missing, PATHS

st.set_page_config(page_title="Modelos y Métricas", layout="wide")
st.title("📊 Modelos y Métricas de Evaluación")
st.markdown(
    "Comparación de los modelos baseline (Random Forest y XGBoost) y el modelo "
    "principal (U-Net) sobre la partición espacial de prueba."
)

cfg       = load_config()
criterios = cfg.get("evaluation", {}).get("success_criteria", {})
F1_MIN    = float(criterios.get("f1_pixel",    0.70))
IOU_MIN   = float(criterios.get("iou_polygon", 0.40))

# ── Cargar resultados disponibles ─────────────────────────────────────────────
modelos: dict[str, dict] = {}
if warn_if_missing("eval_xgboost"):
    modelos["XGBoost"] = read_json("eval_xgboost")
if PATHS["eval_unet"].exists():
    modelos["U-Net"] = read_json("eval_unet")

if not modelos:
    st.stop()

# ── Normalizar claves del JSON ────────────────────────────────────────────────
# El JSON puede usar nombres distintos (precision / test_precision / prec / etc.)
# Esta función busca la primera clave que coincida con algún patrón.
def _buscar(datos: dict, *patrones: str):
    """Devuelve el primer valor numérico cuya clave contenga alguno de los patrones."""
    for k, v in datos.items():
        k_lower = k.lower()
        for p in patrones:
            if p in k_lower:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
    return None

METRICAS_DEF = {
    "Precisión":      ("precis",),
    "Recall":         ("recall",),
    "F1 (píxel)":     ("f1",),
    "IoU (píxel)":    ("iou_pix", "iou_pixel", "pixel_iou"),
    "AUC-ROC":        ("auc_roc", "roc_auc", "auroc"),
    "AUC-PR":         ("auc_pr", "pr_auc", "ap"),
    "IoU (polígono)": ("iou_pol", "polygon_iou", "poly_iou"),
}

filas = []
for nombre, datos in modelos.items():
    fila = {"Modelo": nombre}
    for etiqueta, patrones in METRICAS_DEF.items():
        v = _buscar(datos, *patrones)
        fila[etiqueta] = round(v, 3) if v is not None else None
    filas.append(fila)

df_met = pd.DataFrame(filas).set_index("Modelo")

# ── 1. Tabla comparativa ──────────────────────────────────────────────────────
st.subheader("Métricas por modelo")

def _color(val, col):
    if not isinstance(val, float):
        return ""
    if col == "F1 (píxel)":
        return "color:#4ade80;font-weight:bold" if val >= F1_MIN  else "color:#f87171;font-weight:bold"
    if col == "IoU (polígono)":
        return "color:#4ade80;font-weight:bold" if val >= IOU_MIN else "color:#f87171;font-weight:bold"
    return ""

st.dataframe(
    df_met.style
          .apply(lambda col: [_color(v, col.name) for v in col], axis=0)
          .format("{:.3f}", na_rep="—"),
    use_container_width=True,
)

# Expander de depuración: muestra el JSON crudo por si alguna métrica sale "—"
with st.expander("🔍 Ver JSON completo de evaluación (para depuración)"):
    for nombre, datos in modelos.items():
        st.markdown(f"**{nombre}**")
        st.json(datos)

st.divider()

# ── 2. Criterios de éxito ─────────────────────────────────────────────────────
st.subheader(f"Criterios de éxito  (F1 píxel ≥ {F1_MIN} · IoU polígono ≥ {IOU_MIN})")

cols_ui = st.columns(len(modelos) * 2)
for i, (nombre, datos) in enumerate(modelos.items()):
    f1v  = df_met.loc[nombre, "F1 (píxel)"]   or 0.0
    iouv = df_met.loc[nombre, "IoU (polígono)"] or 0.0
    cols_ui[i*2  ].metric(f"{nombre} — F1 (píxel)",
                          f"{f1v:.3f}" if f1v else "—",
                          delta="✅ CUMPLE" if f1v and f1v >= F1_MIN else f"❌ NO CUMPLE  (mín {F1_MIN})",
                          delta_color="normal" if f1v and f1v >= F1_MIN else "inverse")
    cols_ui[i*2+1].metric(f"{nombre} — IoU (polígono)",
                          f"{iouv:.3f}" if iouv else "—",
                          delta="✅ CUMPLE" if iouv and iouv >= IOU_MIN else f"❌ NO CUMPLE  (mín {IOU_MIN})",
                          delta_color="normal" if iouv and iouv >= IOU_MIN else "inverse")

st.divider()

# ── 3. Barras comparativas ────────────────────────────────────────────────────
st.subheader("Comparación visual de métricas")

COLORES = {"XGBoost": "#1b4332", "Random Forest": "#52b788", "U-Net": "#1565c0"}

# Solo incluir métricas que tengan al menos un valor real
metricas_con_datos = [
    m for m in METRICAS_DEF.keys()
    if df_met[m].notna().any()
]

if metricas_con_datos:
    df_long = (
        df_met[metricas_con_datos].reset_index()
        .melt(id_vars="Modelo", var_name="Métrica", value_name="Valor")
        .dropna(subset=["Valor"])
    )
    fig = px.bar(df_long, x="Métrica", y="Valor", color="Modelo", barmode="group",
                 range_y=[0, 1.05], text_auto=".3f", color_discrete_map=COLORES)
    fig.add_hline(y=F1_MIN, line_dash="dot", line_color="#d62728",
                  annotation_text=f"F1 mín {F1_MIN}", annotation_position="top left")
    fig.update_layout(plot_bgcolor="white", yaxis_title="Valor",
                      legend_title="Modelo", xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Las claves del JSON de evaluación no coincidieron con los nombres esperados. "
            "Revisa el JSON en el expander de arriba.", icon="⚠️")

st.divider()

# ── 4. Sesgo validación aleatoria vs. espacial ────────────────────────────────
st.subheader("Sesgo por dependencia espacial — Validación aleatoria vs. espacial")

filas_bias = []
for nombre, datos in modelos.items():
    f1r = _buscar(datos, "f1_random", "random_f1", "f1_rand")
    f1s = _buscar(datos, "f1_spatial", "spatial_f1", "f1_spat") or df_met.loc[nombre, "F1 (píxel)"]
    if f1r is not None and f1s is not None:
        filas_bias.append({"Modelo": nombre,
                           "F1 aleatorio": round(f1r, 3),
                           "F1 espacial":  round(f1s, 3),
                           "Sesgo (Δ)":    round(f1r - f1s, 3)})

if filas_bias:
    st.dataframe(pd.DataFrame(filas_bias).set_index("Modelo"), use_container_width=True)
    st.caption("La diferencia cuantifica el optimismo introducido por la autocorrelación espacial.")
else:
    st.info("Disponible cuando los JSON incluyan claves de validación aleatoria.", icon="ℹ️")

st.divider()

# ── 5. Importancia de atributos ───────────────────────────────────────────────
st.subheader("Importancia de atributos — Top 20 (XGBoost)")

if warn_if_missing("imp_xgboost"):
    df_imp = read_csv("imp_xgboost")
    # La columna de importancia puede llamarse 'importance', 'gain', 'weight', etc.
    col_imp = next((c for c in df_imp.columns if c.lower() in
                    ("importance", "gain", "weight", "score", "value")), df_imp.columns[-1])
    col_feat = next((c for c in df_imp.columns if c.lower() in
                     ("feature", "feature_name", "name", "variable")), df_imp.columns[0])
    df_imp = df_imp.nlargest(20, col_imp).sort_values(col_imp)
    fig_imp = px.bar(df_imp, x=col_imp, y=col_feat, orientation="h",
                     labels={col_imp: "Importancia", col_feat: "Atributo"},
                     color=col_imp, color_continuous_scale="Greens", text_auto=".3f")
    fig_imp.update_layout(plot_bgcolor="white", coloraxis_showscale=False,
                          yaxis_title="", xaxis_title="Importancia (XGBoost)")
    st.plotly_chart(fig_imp, use_container_width=True)
