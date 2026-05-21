"""Calcula NDVI, NBR y NDWI para cada composicion trimestral (mediana).

Uso (desde la raiz del repositorio):
    python scripts/build_indices.py

Opera sobre los GeoTIFF en data/processed/composites y escribe en
data/processed/indices. Dependencias minimas: rasterio, numpy.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amazonia_deforestation.features.indices import compute_indices  # noqa: E402


def main():
    comp_dir = ROOT / "data" / "processed" / "composites"
    out_dir = ROOT / "data" / "processed" / "indices"
    medians = sorted(comp_dir.glob("composite_*_median.tif"))
    if not medians:
        print("No hay composiciones en " + str(comp_dir) + ". Corre primero build_composites.py")
        return
    for path in medians:
        out_path = out_dir / path.name.replace("composite_", "indices_").replace("_median", "")
        names = compute_indices(path, out_path)
        print("Indices " + str(names) + " -> " + out_path.name)
    print("\nIndices en " + str(out_dir))


if __name__ == "__main__":
    main()
