"""Alinea la etiqueta Hansen a la grilla de 20 m del AOI.

Uso (desde la raiz del repositorio, tras tener al menos una composicion):
    python scripts/align_label.py

Dependencias minimas: rasterio, numpy, pyyaml.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from amazonia_deforestation.data.labels import align_label  # noqa: E402


def main():
    config = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    comp_dir = ROOT / "data" / "processed" / "composites"
    refs = sorted(comp_dir.glob("composite_*_median.tif"))
    if not refs:
        print("Falta una composicion de referencia. Corre primero build_composites.py")
        return
    out_path = ROOT / "data" / "interim" / "label_2024_20m.tif"
    align_label(config, refs[0], out_path)


if __name__ == "__main__":
    main()
