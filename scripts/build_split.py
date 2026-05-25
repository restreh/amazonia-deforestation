"""Particion espacial por bloques sobre la etiqueta alineada.

Uso (desde la raiz del repositorio, tras align_label.py):
    python scripts/build_split.py

Dependencias minimas: rasterio, numpy, pyyaml.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from amazonia_deforestation.spatial.block_split import make_split  # noqa: E402


def main() -> None:
    config = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    label_path = ROOT / "data" / "interim" / "label_2024_20m.tif"
    if not label_path.exists():
        print("Falta data/interim/label_2024_20m.tif. Corre primero scripts/align_label.py")
        return
    out_path = ROOT / "data" / "interim" / "split_blocks.tif"
    print("Construyendo particion espacial por bloques...")
    make_split(config, label_path, out_path)


if __name__ == "__main__":
    main()
