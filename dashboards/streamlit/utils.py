"""Utilidades compartidas del tablero: rutas, cargadores tolerantes a archivos faltantes,
helpers de raster y formato. La carpinteria que cada pagina reutiliza.
"""

from __future__ import annotations

import io
import json
import base64
from pathlib import Path

import numpy as np
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
PRED = DATA / "processed" / "predictions"
INTERIM = DATA / "interim"
EXTERNAL = DATA / "external"


# ---------------------------------------------------------------------------
# Catalogo unico de rutas. Una sola fuente de verdad para que las paginas no
# tengan que armar paths a mano.
# ---------------------------------------------------------------------------
PATHS = {
    # Configuracion y dominio
    "config":            ROOT / "config" / "config.yaml",
    "municipios":        EXTERNAL / "aoi_municipalities.geojson",

    # Etiqueta y particion
    "etiqueta":          INTERIM / "label_2024_20m.tif",
    "split":             INTERIM / "split_blocks.tif",
    "diagnosticos":      INTERIM / "spatial_diagnostics.txt",
    "disponibilidad":    INTERIM / "s2_availability_summary.csv",

    # Probabilidades densas por modelo
    "proba_xgboost":         PRED / "proba_xgboost.tif",
    "proba_random_forest":   PRED / "proba_random_forest.tif",
    "proba_unet":            PRED / "proba_unet.tif",
    "proba_ensemble":        PRED / "proba_ensemble.tif",
    "proba_unet_imagenet":   PRED / "proba_unet_imagenet.tif",

    # Reportes de evaluacion por modelo
    "eval_xgboost":         INTERIM / "eval_xgboost.json",
    "eval_random_forest":   INTERIM / "eval_random_forest.json",
    "eval_unet":            INTERIM / "eval_unet.json",
    "eval_ensemble":        INTERIM / "eval_ensemble.json",
    "eval_unet_imagenet":   INTERIM / "eval_unet_imagenet.json",

    # Comparacion estadistica
    "mcnemar":           INTERIM / "mcnemar.json",
    "bootstrap_spatial": INTERIM / "bootstrap_spatial.json",
    "concordance_lin":   INTERIM / "concordance_lin.json",
    "compare_cv":        INTERIM / "compare_cv.json",

    # Importancia de atributos y CV resumen
    "cv_summary":        INTERIM / "baseline_cv_summary.json",
    "cv_xgboost":        INTERIM / "cv_xgboost.csv",
    "cv_rf":             INTERIM / "cv_random_forest.csv",
    "imp_xgboost":       INTERIM / "importance_xgboost.csv",
    "imp_rf":            INTERIM / "importance_random_forest.csv",

    # Big Data
    "athena_metrics":            INTERIM / "athena_q_metrics_by_model.csv",
    "athena_top_blocks":         INTERIM / "athena_q_top_blocks_ensemble.csv",
    "athena_hectares_by_split":  INTERIM / "athena_q_hectares_by_split.csv",
    "athena_partition_pruning":  INTERIM / "athena_q_features_partition_pruning.csv",
    "metrics_by_block":          DATA / "processed" / "metrics_by_block" / "part.parquet",

    # Composiciones (para comparador pre/post)
    "comp_2024Q1_median":  DATA / "processed" / "composites" / "composite_2024Q1_median.tif",
    "comp_2024Q4_median":  DATA / "processed" / "composites" / "composite_2024Q4_median.tif",
}


# Nombres canonicos y orden de presentacion de modelos. Una sola fuente para
# que toda la UI cuente la misma historia.
MODELOS_CANDIDATOS = ["xgboost", "random_forest", "unet", "ensemble"]
MODELO_CONTROL = "unet_imagenet"
NOMBRES_MODELO = {
    "xgboost":        "XGBoost",
    "random_forest":  "Random Forest",
    "unet":           "U-Net",
    "ensemble":       "Ensamble 0.7/0.3",
    "unet_imagenet":  "U-Net (control ImageNet)",
}


# ---------------------------------------------------------------------------
# Cargadores tolerantes
# ---------------------------------------------------------------------------
def warn_if_missing(key: str) -> bool:
    """Devuelve True si el archivo existe; si no, muestra un aviso amable."""
    p = PATHS[key]
    if not p.exists():
        st.info(f"Aún no está disponible `{p.relative_to(ROOT)}`; "
                "este panel se completa cuando exista.")
        return False
    return True


@st.cache_data(show_spinner=False)
def load_config() -> dict:
    import yaml
    return yaml.safe_load(PATHS["config"].read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def read_json(key: str):
    p = PATHS[key]
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@st.cache_data(show_spinner=False)
def read_csv(key: str):
    import pandas as pd
    p = PATHS[key]
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def read_parquet(key: str):
    import pandas as pd
    p = PATHS[key]
    return pd.read_parquet(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def read_text(key: str) -> str:
    p = PATHS[key]
    return p.read_text(encoding="utf-8") if p.exists() else ""


# ---------------------------------------------------------------------------
# Helpers de raster
# ---------------------------------------------------------------------------
PIXEL_HA = 0.04   # 20 m x 20 m = 400 m^2 = 0.04 ha


@st.cache_data(show_spinner=False)
def estadisticas_etiqueta() -> dict:
    """Conteo total, positivos y prevalencia de la etiqueta de Hansen.

    label_2024_20m.tif tiene `nodata=0` por convencion de GeoTIFF, pero en
    realidad 0 significa "sin deforestacion". Para la prevalencia honesta
    contamos sobre todos los pixeles del AOI (intersectados con split>0).
    """
    import rasterio
    with rasterio.open(PATHS["etiqueta"]) as src:
        et = src.read(1)
    with rasterio.open(PATHS["split"]) as src:
        sp = src.read(1)
    en_aoi = sp > 0
    total = int(en_aoi.sum())
    positivos = int(((et == 1) & en_aoi).sum())
    prevalencia = positivos / total if total > 0 else 0.0
    hectareas = positivos * PIXEL_HA
    return {
        "total_pixeles": total,
        "positivos": positivos,
        "prevalencia": prevalencia,
        "hectareas_deforestadas": hectareas,
    }


@st.cache_data(show_spinner=False)
def raster_a_overlay(ruta: str, colormap: str | list[str], vmin: float, vmax: float,
                     max_px: int = 700, opacidad: float = 0.75,
                     layer_name: str = "Raster"):
    """Lee un raster, le aplica un colormap y devuelve un folium.ImageOverlay listo
    para apilar en un mapa, mas el centro geografico para encuadrar.

    Funciona sin localtileserver: genera un PNG base64 en memoria. Reproyecta
    los limites desde el CRS del raster a EPSG:4326 para que folium lo ubique.
    """
    import rasterio
    import rasterio.warp
    import folium
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    with rasterio.open(ruta) as src:
        factor = max(1, max(src.width, src.height) // max_px)
        data = src.read(
            1,
            out_shape=(max(1, src.height // factor), max(1, src.width // factor)),
            resampling=rasterio.enums.Resampling.nearest,
        ).astype(np.float32)
        nodata = src.nodata
        bounds = rasterio.warp.transform_bounds(src.crs, "EPSG:4326", *src.bounds)

    if nodata is not None:
        data = np.where(data == nodata, np.nan, data)

    if isinstance(colormap, list):
        cmap = ListedColormap(colormap)
    else:
        cmap = plt.get_cmap(colormap)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    rgba = cmap(norm(np.nan_to_num(data, nan=vmin)))
    rgba[np.isnan(data), 3] = 0   # transparencia donde haya NaN

    buf = io.BytesIO()
    plt.imsave(buf, (rgba * 255).astype(np.uint8), format="png")
    buf.seek(0)
    img_b64 = "data:image/png;base64," + base64.b64encode(buf.read()).decode()

    west, south, east, north = bounds
    overlay = folium.raster_layers.ImageOverlay(
        image=img_b64,
        bounds=[[south, west], [north, east]],
        opacity=opacidad,
        name=layer_name,
    )
    return overlay, [(south + north) / 2, (west + east) / 2]


# ---------------------------------------------------------------------------
# Helpers de formato
# ---------------------------------------------------------------------------
def fmt_int(n: float | int) -> str:
    return f"{int(n):,}".replace(",", ".")


def fmt_decimal(x: float, n: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{n}f}"


def fmt_pct(x: float, n: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x*100:.{n}f} %"


def fmt_hectareas(x: float) -> str:
    return f"{x:,.0f} ha".replace(",", ".")
