"""Refina el AOI a los límites municipales y reporta bbox y área.

Uso (desde la raíz del repositorio):
    python scripts/refine_aoi.py

Requiere internet (descarga límites de geoBoundaries si no se da un archivo local).
Dependencias mínimas:
    pip install geopandas pyyaml
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from amazonia_deforestation.ingest.boundaries import refine_aoi  # noqa: E402


def main() -> None:
    config = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    out_dir = ROOT / "data" / "external"
    print("Cargando límites municipales...")
    refine_aoi(config, out_dir)


if __name__ == "__main__":
    main()
