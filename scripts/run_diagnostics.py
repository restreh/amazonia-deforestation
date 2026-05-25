"""Diagnostico de dependencia espacial sobre las capas del AOI.

Calcula el I de Moran y el semivariograma de la etiqueta de perdida (y de NDVI
si existen los indices), estima el rango espacial y sugiere el tamano de bloque
para la validacion espacial.

Uso (desde la raiz del repositorio, tras alinear la etiqueta):
    python scripts/run_diagnostics.py

Dependencias minimas: rasterio, numpy, scipy, pyyaml.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import rasterio  # noqa: E402
import yaml  # noqa: E402

from amazonia_deforestation.spatial.diagnostics import (  # noqa: E402
    empirical_variogram, estimate_range_m, morans_i,
)


def _read(path, band=1):
    with rasterio.open(path) as src:
        return src.read(band).astype("float64")


def main():
    config = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    res = config["processing"]["working_resolution_m"]

    label_path = ROOT / "data" / "interim" / "label_2024_20m.tif"
    if not label_path.exists():
        print("Falta la etiqueta alineada. Corre primero scripts/align_label.py")
        return

    print("=== I de Moran ===")
    label = _read(label_path)
    mi_label = morans_i(label)
    print("Etiqueta de perdida 2024: I = " + format(mi_label, ".4f"))

    indices_dir = ROOT / "data" / "processed" / "indices"
    ndvi = None
    idx_files = sorted(indices_dir.glob("indices_*.tif"))
    if idx_files:
        with rasterio.open(idx_files[0]) as src:
            names = list(src.descriptions)
            if "ndvi" in names:
                ndvi = src.read(names.index("ndvi") + 1).astype("float64")
        if ndvi is not None:
            print("NDVI (" + idx_files[0].name + "): I = " + format(morans_i(ndvi), ".4f"))

    print("\n=== Semivariograma y rango espacial ===")
    lags, sv = empirical_variogram(label, pixel_size_m=res, n_samples=6000,
                                   n_bins=25, max_lag_m=10000, seed=0)
    rng_label = estimate_range_m(lags, sv)
    print("Etiqueta: rango ~ " + format(rng_label, ",.0f") + " m")
    if ndvi is not None:
        lags_n, sv_n = empirical_variogram(ndvi, pixel_size_m=res, n_samples=6000,
                                           n_bins=25, max_lag_m=10000, seed=0)
        rng_ndvi = estimate_range_m(lags_n, sv_n)
        print("NDVI: rango ~ " + format(rng_ndvi, ",.0f") + " m")
        rng_used = max(rng_label, rng_ndvi)
    else:
        rng_used = rng_label

    block_km = max(5.0, math.ceil(rng_used / 1000))
    print("\n=== Recomendacion de bloque de validacion ===")
    print("Rango espacial de referencia: ~ " + format(rng_used, ",.0f") + " m")
    print("Tamano de bloque sugerido: " + str(block_km) + " km (>= rango; minimo 5 km de la propuesta)")

    out = ROOT / "data" / "interim" / "spatial_diagnostics.txt"
    out.write_text(
        "Moran I etiqueta: " + format(mi_label, ".4f") + "\n"
        "Rango espacial (m): " + format(rng_used, ".0f") + "\n"
        "Bloque sugerido (km): " + str(block_km) + "\n",
        encoding="utf-8",
    )
    print("\nResumen guardado en " + str(out))


if __name__ == "__main__":
    main()
