"""Composiciones trimestrales de Sentinel-2 sobre el AOI.

Construye, por trimestre, un mosaico libre de nubes agregando las escenas
disponibles con mediana y percentil 25, tras enmascarar nubes y sombras con la
capa SCL. Usa stackstac + Dask para leer por ventana (Cloud-Optimized GeoTIFF)
sin descargar escenas completas, a la resolucion de trabajo definida en config.

Lectura en float64 (admite NaN como relleno); el resultado mediana/p25 se guarda
en float32 para no inflar el disco. Valores crudos (numero digital, escala 1e-4
= reflectancia); los indices son cocientes y no dependen de esa escala.

Disenado para memoria acotada: se procesa un trimestre a la vez y se escribe a
disco antes de pasar al siguiente.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401  (registra el accesor .rio en xarray)
import stackstac
from pystac_client import Client

# Mapeo de bandas Sentinel-2 a las claves de asset de Earth-Search.
ASSET_MAP = {
    "B02": "blue", "B03": "green", "B04": "red",
    "B05": "rededge1", "B06": "rededge2", "B07": "rededge3",
    "B08": "nir", "B8A": "nir08", "B11": "swir16", "B12": "swir22",
    "SCL": "scl",
}

# Clases SCL que se conservan: 4 vegetacion, 5 no-vegetado, 6 agua, 7 no clasificado.
SCL_KEEP = [4, 5, 6, 7]


def search_items(stac_url, collection, bbox, start, end, cloud_max):
    """Devuelve los items STAC de Sentinel-2 para una ventana temporal y el AOI."""
    catalog = Client.open(stac_url)
    search = catalog.search(
        collections=[collection],
        bbox=bbox,
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": cloud_max}},
    )
    return list(search.items())


def build_quarter_composite(items, bands, bbox, resolution, epsg, chunksize=512):
    """Compone un trimestre: devuelve (mediana, p25) como DataArrays (band, y, x)."""
    spectral_assets = [ASSET_MAP[b] for b in bands if b != "SCL"]
    stack_assets = spectral_assets + ["scl"]

    cube = stackstac.stack(
        items,
        assets=stack_assets,
        epsg=epsg,
        resolution=resolution,
        bounds_latlon=bbox,
        chunksize=chunksize,
        dtype="float64",
        fill_value=np.nan,
        rescale=False,
    )
    scl = cube.sel(band="scl")
    keep = scl.isin(SCL_KEEP)
    spectral = cube.sel(band=spectral_assets).where(keep).chunk({"time": -1})

    median = spectral.median(dim="time", skipna=True).astype("float32")
    p25 = (spectral.quantile(0.25, dim="time", skipna=True)
           .drop_vars("quantile", errors="ignore").astype("float32"))

    for arr in (median, p25):
        arr.rio.write_crs(f"EPSG:{epsg}", inplace=True)
    return median, p25


def write_composite(arr, path, band_names):
    """Escribe un DataArray (band, y, x) como GeoTIFF multibanda con nombres de banda."""
    import rasterio

    path.parent.mkdir(parents=True, exist_ok=True)
    arr = arr.rio.write_nodata(np.nan)
    arr.rio.to_raster(path, driver="GTiff", compress="deflate")
    with rasterio.open(path, "r+") as dst:
        for i, name in enumerate(band_names, start=1):
            dst.set_band_description(i, name)


def build_all_quarters(config, out_dir, scene_limit=None):
    """Construye y guarda las composiciones de todos los trimestres del config."""
    s2 = config["data_sources"]["sentinel2"]
    bands = [b for b in s2["bands"] if b != "SCL"]
    bbox = config["aoi"]["bbox_geographic"]
    cloud_max = config["temporal"]["cloud_cover_max"]
    resolution = config["processing"]["working_resolution_m"]
    epsg = int(config["processing"]["output_crs"].split(":")[1])
    band_assets = [ASSET_MAP[b] for b in bands]

    for q in config["temporal"]["composite_quarters"]:
        qid = q["id"]
        print("\n=== Trimestre " + qid + " ===")
        items = search_items(s2["stac_url"], s2["collection"], bbox, q["start"], q["end"], cloud_max)
        if scene_limit:
            items = items[:scene_limit]
        print("Escenas: " + str(len(items)))
        if not items:
            print("Sin escenas; se omite.")
            continue

        median, p25 = build_quarter_composite(items, bands, bbox, resolution, epsg)
        for stat, arr in (("median", median), ("p25", p25)):
            path = out_dir / ("composite_" + qid + "_" + stat + ".tif")
            print("Calculando y escribiendo " + path.name)
            write_composite(arr, path, band_assets)
            print("  shape " + str(tuple(arr.shape)) + " -> " + str(path))
