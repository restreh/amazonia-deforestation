"""Sistema de diseno del tablero: paleta de colores y tema plotly.

Decisiones de diseno y referencias:

  - Alto cociente dato/tinta y supresion de elementos no informativos
    (gridlines pesadas, fondos, ejes redundantes). Tufte (2001).
  - Asignacion semantica fija de color: cada concepto (perdida, preservacion,
    referencia institucional, prediccion) y cada modelo conserva su color en
    todas las paginas, para que el lector aprenda el codigo una sola vez.
    Few (2006) y Cairo (2016).
  - Codificacion visual jerarquizada por precision perceptual: posicion y
    longitud para magnitudes; color y forma solo para categorias.
    Cleveland & McGill (1984).
  - Accesibilidad para daltonismo: la combinacion rojo-verde no es el unico
    canal informativo; metricas criticas se acompanan de iconos y deltas
    explicitos.
  - Narrativa guiada por pagina: lead introductorio, metricas principales,
    visualizacion principal y un mensaje destacado (`takeaway`) que sintetiza
    una sola idea por seccion. Knaflic (2015).

Referencias:
  Cairo, A. (2016). The truthful art. New Riders.
  Cleveland, W. S., & McGill, R. (1984). Graphical perception. Journal of the
    American Statistical Association, 79(387), 531-554.
  Few, S. (2006). Information dashboard design. O'Reilly.
  Knaflic, C. N. (2015). Storytelling with data. Wiley.
  Tufte, E. R. (2001). The visual display of quantitative information
    (2.a ed.). Graphics Press.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio


# Paleta semantica. Una sola fuente para que el lector aprenda el codigo de
# colores y la lectura sea consistente entre paginas.
COLORES = {
    "bosque":         "#2d6a4f",   # cobertura forestal
    "perdida":        "#d62828",   # deforestacion
    "alerta":         "#f77f00",   # alerta intermedia
    "predicho":       "#4361ee",   # prediccion del modelo
    "referencia":     "#888888",   # Hansen / referencia institucional
    "exito":          "#52b788",   # criterio cumplido
    "fallo":          "#e63946",   # criterio incumplido
    "neutro":         "#adb5bd",
    "fondo":          "#0e1117",   # match con tema dark de streamlit
    "tinta":          "#f8f9fa",   # texto sobre fondo oscuro
    "tinta_suave":    "#9aa0a6",
}

# Color asignado a cada modelo. Mismo orden y misma asignacion en TODA la app.
COLORES_MODELO = {
    "xgboost":        "#1f77b4",
    "random_forest":  "#9b6f3a",
    "unet":           "#9467bd",
    "ensemble":       "#d62828",
    "unet_imagenet":  "#7f7f7f",
}

# Paletas continuas. La de probabilidad va de verde a rojo (preservacion a
# perdida); evita usar el rojo-verde tradicional simultaneo para clases.
CMAP_PROBA = "RdYlGn_r"   # 0 verde -> 1 rojo
CMAP_SPLIT = ["#0e1117", "#52b788", "#f9c74f", "#e63946"]  # 0 transp, 1 train, 2 val, 3 test
CMAP_ERROR = ["#0e1117", "#2d6a4f", "#f77f00", "#d62828"]  # 0 transp, 1 TP, 2 FP, 3 FN


def aplicar_tema_plotly():
    """Registra y aplica un tema plotly Tufte-friendly como default.

    Llamar una vez al inicio de cada pagina (idempotente).
    """
    template = go.layout.Template()
    template.layout = go.Layout(
        font=dict(family="Inter, sans-serif", color=COLORES["tinta"], size=14),
        title=dict(font=dict(size=18, color=COLORES["tinta"]), x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=[COLORES_MODELO["xgboost"], COLORES_MODELO["random_forest"],
                  COLORES_MODELO["unet"], COLORES_MODELO["ensemble"],
                  COLORES["alerta"]],
        xaxis=dict(
            showgrid=False, zeroline=False, showline=True,
            linecolor=COLORES["tinta_suave"], linewidth=1,
            ticks="outside", tickcolor=COLORES["tinta_suave"],
            tickfont=dict(color=COLORES["tinta_suave"]),
            title=dict(font=dict(color=COLORES["tinta_suave"])),
        ),
        yaxis=dict(
            showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=False,
            tickcolor=COLORES["tinta_suave"], tickfont=dict(color=COLORES["tinta_suave"]),
            title=dict(font=dict(color=COLORES["tinta_suave"])),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORES["tinta_suave"]),
        ),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    pio.templates["amazonia_dark"] = template
    pio.templates.default = "amazonia_dark"


def aplicar_estilos_streamlit():
    """Inyecta CSS minimo para Tufte: reduce chart junk, ajusta tipografia."""
    import streamlit as st
    st.markdown(
        """
        <style>
        /* Espaciado y tipografia */
        .main .block-container { padding-top: 2rem; padding-bottom: 3rem; }
        h1 { letter-spacing: -0.02em; line-height: 1.1; }
        h2 { letter-spacing: -0.01em; margin-top: 1.5rem; }
        h3 { color: rgba(248,249,250,0.92); margin-top: 1.2rem; }

        /* Metricas mas legibles */
        [data-testid="stMetricValue"] { font-size: 2rem; font-weight: 600; }
        [data-testid="stMetricLabel"] { color: rgba(154,160,166,1); font-size: 0.85rem;
                                         text-transform: uppercase; letter-spacing: 0.05em; }
        [data-testid="stMetricDelta"] { font-size: 0.95rem; }

        /* Reduce decorado de st.dataframe */
        div[data-testid="stDataFrame"] thead { background: rgba(255,255,255,0.04); }

        /* Banner principal */
        .takeaway { background: rgba(214, 40, 40, 0.08); border-left: 4px solid #d62828;
                    padding: 0.9rem 1.2rem; border-radius: 4px; margin: 0.5rem 0 1rem 0; }
        .takeaway p { margin: 0; color: rgba(248,249,250,0.95); }
        .takeaway strong { color: #ffba08; }

        .lead { color: rgba(154,160,166,1); font-size: 1.05rem; max-width: 60ch;
                line-height: 1.6; margin-bottom: 1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def takeaway(texto_html: str):
    """Bloque destacado para el mensaje principal de cada pagina."""
    import streamlit as st
    st.markdown(f'<div class="takeaway"><p>{texto_html}</p></div>',
                unsafe_allow_html=True)


def lead(texto: str):
    """Parrafo introductorio de cada pagina con estilo discreto."""
    import streamlit as st
    st.markdown(f'<p class="lead">{texto}</p>', unsafe_allow_html=True)
