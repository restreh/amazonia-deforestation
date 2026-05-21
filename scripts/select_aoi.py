"""Selecciona el AOI de trabajo (~5.000 km²) sobre el núcleo de deforestación.

Uso (desde la raíz del repositorio, tras correr refine_aoi.py):
    python scripts/select_aoi.py

Requiere internet. Dependencias mínimas:
    pip install rasterio geopandas numpy pyyaml
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from amazonia_deforestation.ingest.select_aoi import select_aoi  # noqa: E402


def main() -> None:
    config = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    boundary_path = ROOT / "data" / "external" / "aoi_municipalities.geojson"
    if not boundary_path.exists():
        print("Falta data/external/aoi_municipalities.geojson. Corre primero scripts/refine_aoi.py")
        return
    print("Seleccionando AOI sobre el núcleo de deforestación...")
    select_aoi(config, boundary_path)


if __name__ == "__main__":
    main()
