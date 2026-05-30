"""Utilidades compartidas del tablero: rutas, cargadores tolerantes a archivos
faltantes, helpers de raster y formato.

El tablero puede leer de dos fuentes, configurable via la variable de entorno
`AMAZONIA_DATA_SOURCE`:

  - `local` (por defecto): lee de `data/...` en el repo. Usado en desarrollo
    y cuando se distribuye el repo con artefactos.
  - `s3`: lee de `s3://amazonia-deforestation-data-363918845645/...`
    anonimamente. Usado por Streamlit Cloud y cualquier despliegue donde no
    se quieran versionar los rasters en git. Los prefijos `derived/`,
    `models/` y `metrics/` del bucket son public-read.
"""

from __future__ import annotations

import io
import json
import os
import base64
from pathlib import Path
from typing import Union

import numpy as np
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
S3_BUCKET = "amazonia-deforestation-data-363918845645"
S3_REGION = "us-west-2"

# Decide la fuente una sola vez y propagala como env vars para que GDAL/
# rasterio puedan leer s3 anonimamente sin firmar.
DATA_SOURCE = os.environ.get("AMAZONIA_DATA_SOURCE", "local").lower()
if DATA_SOURCE == "s3":
    # GDAL respeta estas variables para todas las operaciones VSI/s3.
    os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
    os.environ.setdefault("AWS_REGION", S3_REGION)
    os.environ.setdefault("AWS_DEFAULT_REGION", S3_REGION)
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("CPL_VSIL_CURL_USE_HEAD", "NO")


# ---------------------------------------------------------------------------
# Catalogo unico de rutas. Una sola fuente de verdad. Cada entrada tiene la
# misma forma de URL relativa al bucket; la funcion `url(key)` la materializa
# como Path local o como s3:// segun la fuente.
# ---------------------------------------------------------------------------
_S3_PATHS = {
    # Configuracion y dominio
    "config":            (ROOT / "config" / "config.yaml", None),   # solo local
    "municipios":        (ROOT / "data" / "external" / "aoi_municipalities.geojson",
                          "derived/external/aoi_municipalities.geojson"),

    # Etiqueta y particion (subidas bajo derived/interim/)
    "etiqueta":          (ROOT / "data" / "interim" / "label_2024_20m.tif",
                          "derived/interim/label_2024_20m.tif"),
    "split":             (ROOT / "data" / "interim" / "split_blocks.tif",
                          "derived/interim/split_blocks.tif"),

    # Reportes y CSVs que viven en metrics/
    "diagnosticos":      (ROOT / "data" / "interim" / "spatial_diagnostics.txt",
                          "metrics/spatial_diagnostics.txt"),
    "disponibilidad":    (ROOT / "data" / "interim" / "s2_availability_summary.csv",
                          "metrics/s2_availability_summary.csv"),
    "cv_summary":        (ROOT / "data" / "interim" / "baseline_cv_summary.json",
                          "metrics/baseline_cv_summary.json"),
    "eval_xgboost":      (ROOT / "data" / "interim" / "eval_xgboost.json",
                          "metrics/eval_xgboost.json"),
    "eval_random_forest": (ROOT / "data" / "interim" / "eval_random_forest.json",
                           "metrics/eval_random_forest.json"),
    "eval_unet":         (ROOT / "data" / "interim" / "eval_unet.json",
                          "metrics/eval_unet.json"),
    "eval_ensemble":     (ROOT / "data" / "interim" / "eval_ensemble.json",
                          "metrics/eval_ensemble.json"),
    "eval_unet_imagenet": (ROOT / "data" / "interim" / "eval_unet_imagenet.json",
                           "metrics/eval_unet_imagenet.json"),
    "mcnemar":           (ROOT / "data" / "interim" / "mcnemar.json",
                          "metrics/mcnemar.json"),
    "bootstrap_spatial": (ROOT / "data" / "interim" / "bootstrap_spatial.json",
                          "metrics/bootstrap_spatial.json"),
    "concordance_lin":   (ROOT / "data" / "interim" / "concordance_lin.json",
                          "metrics/concordance_lin.json"),
    "compare_cv":        (ROOT / "data" / "interim" / "compare_cv.json",
                          "metrics/compare_cv.json"),
    "cv_xgboost":        (ROOT / "data" / "interim" / "cv_xgboost.csv",
                          "metrics/cv_xgboost.csv"),
    "cv_rf":             (ROOT / "data" / "interim" / "cv_random_forest.csv",
                          "metrics/cv_random_forest.csv"),
    "imp_xgboost":       (ROOT / "data" / "interim" / "importance_xgboost.csv",
                          "metrics/importance_xgboost.csv"),
    "imp_rf":            (ROOT / "data" / "interim" / "importance_random_forest.csv",
                          "metrics/importance_random_forest.csv"),

    # Probabilidades densas por modelo
    "proba_xgboost":          (ROOT / "data" / "processed" / "predictions" / "proba_xgboost.tif",
                               "derived/predictions/model=xgboost/proba.tif"),
    "proba_random_forest":    (ROOT / "data" / "processed" / "predictions" / "proba_random_forest.tif",
                               "derived/predictions/model=random_forest/proba.tif"),
    "proba_unet":             (ROOT / "data" / "processed" / "predictions" / "proba_unet.tif",
                               "derived/predictions/model=unet/proba.tif"),
    "proba_ensemble":         (ROOT / "data" / "processed" / "predictions" / "proba_ensemble.tif",
                               "derived/predictions/model=ensemble/proba.tif"),
    "proba_unet_imagenet":    (ROOT / "data" / "processed" / "predictions" / "proba_unet_imagenet.tif",
                               "derived/predictions/model=unet_imagenet/proba.tif"),

    # Big Data / Athena (estaticos en el repo, no se duplican en S3 metrics)
    "athena_metrics":            (ROOT / "data" / "interim" / "athena_q_metrics_by_model.csv",
                                   "metrics/athena_q_metrics_by_model.csv"),
    "athena_top_blocks":         (ROOT / "data" / "interim" / "athena_q_top_blocks_ensemble.csv",
                                   "metrics/athena_q_top_blocks_ensemble.csv"),
    "athena_hectares_by_split":  (ROOT / "data" / "interim" / "athena_q_hectares_by_split.csv",
                                   "metrics/athena_q_hectares_by_split.csv"),
    "athena_partition_pruning":  (ROOT / "data" / "interim" / "athena_q_features_partition_pruning.csv",
                                   "metrics/athena_q_features_partition_pruning.csv"),
}


def url(key: str) -> Union[Path, str]:
    """Devuelve la URL del recurso segun la fuente. Path local o `s3://...`."""
    local, s3_key = _S3_PATHS[key]
    if DATA_SOURCE == "s3" and s3_key is not None:
        return f"s3://{S3_BUCKET}/{s3_key}"
    return local


# Mantengo `PATHS` para compatibilidad: en modo local es identico al diccionario
# original; en modo s3 contiene `s3://...` strings.
PATHS = {k: url(k) for k in _S3_PATHS}


# Nombres canonicos y orden de presentacion de modelos.
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
def _exists(u: Union[Path, str]) -> bool:
    """True si el recurso existe (local o s3 anonimo)."""
    if isinstance(u, Path):
        return u.exists()
    # s3://; HEAD anonimo
    try:
        import fsspec
        fs = fsspec.filesystem("s3", anon=True)
        return fs.exists(u.replace("s3://", ""))
    except Exception:
        return False


def _open_text(u: Union[Path, str]):
    """Abre el recurso como texto (read mode)."""
    if isinstance(u, Path):
        return open(u, "r", encoding="utf-8")
    import fsspec
    return fsspec.open(u, mode="r", encoding="utf-8", anon=True).open()


def warn_if_missing(key: str) -> bool:
    """Devuelve True si existe; si no, muestra un aviso amable."""
    u = url(key)
    if _exists(u):
        return True
    label = (u if isinstance(u, str)
             else u.relative_to(ROOT))
    st.info(f"Aún no está disponible `{label}`; este panel se completa cuando exista.")
    return False


@st.cache_data(show_spinner=False)
def load_config() -> dict:
    import yaml
    return yaml.safe_load(_open_text(PATHS["config"]).read())


@st.cache_data(show_spinner=False)
def read_json(key: str):
    u = url(key)
    if not _exists(u):
        return None
    with _open_text(u) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def read_csv(key: str):
    import pandas as pd
    u = url(key)
    if not _exists(u):
        return None
    if isinstance(u, str) and u.startswith("s3://"):
        return pd.read_csv(u, storage_options={"anon": True})
    return pd.read_csv(u)


@st.cache_data(show_spinner=False)
def read_parquet(key: str):
    import pandas as pd
    u = url(key)
    if not _exists(u):
        return None
    if isinstance(u, str) and u.startswith("s3://"):
        return pd.read_parquet(u, storage_options={"anon": True})
    return pd.read_parquet(u)


@st.cache_data(show_spinner=False)
def read_text(key: str) -> str:
    u = url(key)
    if not _exists(u):
        return ""
    with _open_text(u) as f:
        return f.read()


def _rasterio_open(u: Union[Path, str]):
    """rasterio.open que funciona con local o s3 anonimo via VSI/s3."""
    import rasterio
    if isinstance(u, Path):
        return rasterio.open(str(u))
    if u.startswith("s3://"):
        # rasterio entiende /vsis3/ con AWS_NO_SIGN_REQUEST=YES
        vsi = u.replace("s3://", "/vsis3/")
        return rasterio.open(vsi)
    return rasterio.open(u)


# Aliases publicos para usar desde las paginas.
def existe(key: str) -> bool:
    """True si el recurso de la key existe (local o s3)."""
    return _exists(url(key))


def abrir_raster(key_or_url):
    """rasterio.open transparente sobre key del catalogo o URL/Path directo."""
    if isinstance(key_or_url, str) and key_or_url in _S3_PATHS:
        return _rasterio_open(url(key_or_url))
    return _rasterio_open(key_or_url)


def leer_geojson(key: str):
    """gpd.read_file que funciona con local o s3 anonimo."""
    import geopandas as gpd
    u = url(key)
    if isinstance(u, Path):
        return gpd.read_file(u)
    # s3 anonimo: descargamos a tempfile y leemos. Es chico.
    import fsspec
    import tempfile
    suffix = "." + u.split(".")[-1] if "." in u else ""
    with fsspec.open(u, mode="rb", anon=True) as src:
        data = src.read()
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(data)
    tmp.close()
    return gpd.read_file(tmp.name)


# ---------------------------------------------------------------------------
# Helpers de raster
# ---------------------------------------------------------------------------
PIXEL_HA = 0.04   # 20 m x 20 m = 400 m^2 = 0.04 ha


@st.cache_data(show_spinner=False)
def estadisticas_etiqueta() -> dict:
    """Conteo total, positivos y prevalencia honesta de la etiqueta de Hansen.

    `label_2024_20m.tif` tiene `nodata=0` (convencion GeoTIFF), pero 0 en
    realidad significa "sin deforestacion". Acotamos al AOI usando
    `split > 0` para no descartar pixeles validos.
    """
    with _rasterio_open(url("etiqueta")) as src:
        et = src.read(1)
    with _rasterio_open(url("split")) as src:
        sp = src.read(1)
    en_aoi = sp > 0
    total = int(en_aoi.sum())
    positivos = int(((et == 1) & en_aoi).sum())
    prevalencia = positivos / total if total > 0 else 0.0
    return {
        "total_pixeles": total,
        "positivos": positivos,
        "prevalencia": prevalencia,
        "hectareas_deforestadas": positivos * PIXEL_HA,
    }


@st.cache_data(show_spinner="Renderizando raster...")
def raster_a_overlay(ruta: str, colormap, vmin: float, vmax: float,
                     max_px: int = 700, opacidad: float = 0.75,
                     layer_name: str = "Raster"):
    """Lee un raster (local o s3) y devuelve un folium.ImageOverlay con un PNG
    base64 generado a partir del colormap dado. No requiere localtileserver."""
    import rasterio
    import rasterio.warp
    import folium
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    with _rasterio_open(ruta) as src:
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
    rgba[np.isnan(data), 3] = 0

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
def fmt_int(n) -> str:
    return f"{int(n):,}".replace(",", ".")


def fmt_decimal(x, n: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{n}f}"


def fmt_pct(x, n: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x*100:.{n}f} %"


def fmt_hectareas(x) -> str:
    return f"{x:,.0f} ha".replace(",", ".")
