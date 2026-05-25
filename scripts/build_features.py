"""Construye la tabla de features de entrenamiento por pixel.

Uso (desde la raiz del repositorio, tras build_training_sample.py y build_indices.py):
    python scripts/build_features.py

Calcula bandas, indices y atributos contextuales (media, desv., GLCM) en los cuatro
trimestres para los pixeles muestreados en data/interim/train_sample.parquet.

Dependencias: rasterio, scipy, numpy, pandas, pyarrow, pyyaml.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from amazonia_deforestation.features.build_dataset import build_feature_table  # noqa: E402


def main() -> None:
    config = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    sample_path = ROOT / "data" / "interim" / "train_sample.parquet"
    composites_dir = ROOT / "data" / "processed" / "composites"
    indices_dir = ROOT / "data" / "processed" / "indices"
    if not sample_path.exists():
        print("Falta data/interim/train_sample.parquet. Corre primero build_training_sample.py")
        return

    sample = pd.read_parquet(sample_path)
    coords = sample[["row", "col"]].to_numpy()
    print("Pixeles a procesar: " + format(len(coords), ","))

    table = build_feature_table(config, composites_dir, indices_dir, coords)
    table.insert(0, "row", sample["row"].to_numpy())
    table.insert(1, "col", sample["col"].to_numpy())
    table["label"] = sample["label"].to_numpy()

    out_dir = ROOT / "data" / "processed" / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "train_features.parquet"
    table.to_parquet(out_path, index=False)

    n_feat = table.shape[1] - 3  # excluye row, col, label
    print("Tabla: " + format(table.shape[0], ",") + " filas x " + str(n_feat) + " atributos")
    print("Guardada en " + str(out_path))


if __name__ == "__main__":
    main()
