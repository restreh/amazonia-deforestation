"""Muestreo balanceado de pixeles de entrenamiento.

Uso (desde la raiz del repositorio, tras build_split.py):
    python scripts/build_training_sample.py

Dependencias minimas: rasterio, numpy, pandas, pyarrow, pyyaml.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from amazonia_deforestation.spatial.sampling import build_training_sample  # noqa: E402


def main() -> None:
    config = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    label_path = ROOT / "data" / "interim" / "label_2024_20m.tif"
    split_path = ROOT / "data" / "interim" / "split_blocks.tif"
    for p in (label_path, split_path):
        if not p.exists():
            print("Falta " + str(p) + ". Corre primero los pasos previos.")
            return
    out_path = ROOT / "data" / "interim" / "train_sample.parquet"
    print("Muestreando pixeles de entrenamiento...")
    build_training_sample(config, label_path, split_path, out_path)


if __name__ == "__main__":
    main()
