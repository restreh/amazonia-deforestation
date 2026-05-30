"""Diagnostico espacial: I de Moran, semivariograma y particion por bloques."""
import re, io, base64, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import numpy as np
import streamlit as st
import rasterio
import rasterio.warp
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from utils import warn_if_missing, PATHS

st.set_page_config(page_title="Diagnóstico Espacial", layout="wide")
st.title("🗺️ Diagnóstico Espacial")
st.markdown(
    "La autocorrelación espacial de las observaciones satelitales viola el supuesto de "
    "independencia de la validación cruzada estándar. Aquí se diagnostica y se controla "
    "mediante una partición por bloques."
)

# ── Helper: raster → folium ImageOverlay (sin localtileserver) ────────────────
@st.cache_data(show_spinner=False)
def raster_a_overlay(ruta: str, colores: list[str], vmin: float, vmax: float,
                     max_px: int = 600, opacidad: float = 0.65):
    """
    Lee un raster local, aplica un colormap matplotlib y devuelve
    (folium.ImageOverlay, centro_lat, centro_lon).
    No requiere localtileserver.
    """
    with rasterio.open(ruta) as src:
        factor   = max(1, max(src.width, src.height) // max_px)
        out_h    = max(1, src.height // factor)
        out_w    = max(1, src.width  // factor)
        data     = src.read(1, out_shape=(out_h, out_w),
                            resampling=rasterio.enums.Resampling.nearest).astype(np.float32)
        nodata   = src.nodata
        bounds   = rasterio.warp.transform_bounds(src.crs, "EPSG:4326", *src.bounds)

    if nodata is not None:
        data = np.where(data == nodata, np.nan, data)

    # Construir colormap a partir de la lista de colores
    cmap = plt.matplotlib.colors.ListedColormap(colores) if len(colores) > 1 \
           else plt.get_cmap(colores[0])
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    rgba = cmap(norm(data))                            # H × W × 4
    if nodata is not None:
        rgba[np.isnan(data), 3] = 0                    # transparente donde hay nodata

    buf = io.BytesIO()
    plt.imsave(buf, (rgba * 255).astype(np.uint8), format="png")
    buf.seek(0)
    img_b64 = "data:image/png;base64," + base64.b64encode(buf.read()).decode()

    west, south, east, north = bounds
    overlay = folium.raster_layers.ImageOverlay(
        image   = img_b64,
        bounds  = [[south, west], [north, east]],
        opacity = opacidad,
        name    = "Raster",
    )
    centro  = [(south + north) / 2, (west + east) / 2]
    return overlay, centro

# ── 1. Métricas del diagnóstico ───────────────────────────────────────────────
st.subheader("Indicadores de autocorrelación — I de Moran y semivariograma")

if warn_if_missing("diagnosticos"):
    @st.cache_data(show_spinner=False)
    def cargar_diagnostico():
        return PATHS["diagnosticos"].read_text(encoding="utf-8")

    texto = cargar_diagnostico()
    pat_moran  = re.search(r"(?:I\s+de\s+Moran|Moran['\s]*I)\s*[=:]\s*([-\d.]+)", texto, re.I)
    pat_rango  = re.search(r"[Rr]ango\s*[=:]\s*([\d.]+)\s*m",  texto)
    pat_bloque = re.search(r"[Bb]loque\s*[=:]\s*([\d.]+)\s*km", texto)
    pat_pval   = re.search(r"[Pp][-_]?valor?\s*[=:]\s*([\d.eE+\-]+)", texto)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("I de Moran",               pat_moran.group(1)            if pat_moran  else "—",
              help="Cercano a 1 → autocorrelación positiva fuerte.")
    c2.metric("Rango del semivariograma", f"{pat_rango.group(1)} m"     if pat_rango  else "—",
              help="Distancia a la que la dependencia espacial se vuelve despreciable.")
    c3.metric("Tamaño de bloque",         f"{pat_bloque.group(1)} km"   if pat_bloque else "—",
              help="Bloques asignados completos a cada conjunto.")
    c4.metric("p-valor (I de Moran)",      pat_pval.group(1)            if pat_pval   else "—")

    with st.expander("📄 Ver texto completo del diagnóstico"):
        st.text(texto)

st.divider()

# ── 2. Mapa de partición por bloques ─────────────────────────────────────────
st.subheader("Partición espacial por bloques (70 % entrenamiento · 15 % validación · 15 % prueba)")

col_mapa, col_leyenda = st.columns([3, 1])

with col_leyenda:
    st.markdown("**Leyenda**")
    st.markdown("🟩 **1 — Entrenamiento** (70 %)\n\n🟨 **2 — Validación** (15 %)\n\n🟥 **3 — Prueba** (15 %)")
    st.info("Los bloques se asignan completos para que la distancia mínima entre "
            "entrenamiento y prueba supere el rango del semivariograma.", icon="ℹ️")

with col_mapa:
    if warn_if_missing("split"):
        # Colores: 1=verde, 2=amarillo, 3=rojo  (nodata/0=transparente)
        overlay, centro = raster_a_overlay(
            str(PATHS["split"]),
            colores=["#ffffff", "#52b788", "#f9c74f", "#e63946"],  # 0=blanco transp, 1,2,3
            vmin=0, vmax=3, opacidad=0.70
        )
        m = folium.Map(location=centro, zoom_start=9, tiles="CartoDB positron")
        overlay.add_to(m)
        folium.LayerControl().add_to(m)
        st_folium(m, height=500, use_container_width=True)

st.caption("Partición con BlockKFold (scikit-learn + verstack/spacv). "
           "La validación aleatoria pixel a pixel habría generado sobreestimación del desempeño "
           "(Karasiak et al., 2022; Ploton et al., 2020).")

st.divider()

with st.expander("📚 ¿Por qué importa la autocorrelación espacial?"):
    st.markdown("""
    Los píxeles vecinos comparten condiciones de iluminación, atmosféricas y de cobertura
    (primera ley de Tobler), lo que viola el supuesto de independencia estándar.

    1. **Sesgo optimista**: la diferencia entre F1 aleatorio y F1 espacial cuantifica
       el optimismo introducido por la dependencia y es, por sí misma, un resultado del proyecto.
    2. **Información aprovechable**: el proyecto la explota mediante atributos contextuales
       (ventanas 3×3 y 5×5, textura GLCM) y, en la U-Net, a través de su campo receptivo creciente.
    """)
