"""Deriva una tabla Parquet particionada por bloque espacial para Athena.

Uso (desde la raiz del repositorio):
    python scripts/build_features_by_block.py

Lee data/processed/features/train_features.parquet (2.86 M filas, 612 atributos +
row, col, label), agrega la columna block_id calculada con el mismo block_side_px que
la particion de validacion, y escribe la tabla en formato Hive bajo
data/processed/features_by_block/block_id=*/part.parquet con compresion Snappy.

Esta tabla es la que registra Glue/Athena para consultas analiticas particionadas, en
cumplimiento del compromiso de Big Data de la propuesta. Cada particion tiene
~18 k filas (2.86 M / 158 bloques). El parquet original tambien se conserva como
referencia y se sube a S3 con upload_to_s3.py.

Dependencias: pandas, pyarrow, rasterio, pyyaml.
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
import yaml  # noqa: E402

from amazonia_deforestation.spatial.block_split import block_side_px  # noqa: E402
from amazonia_deforestation.models.baselines import block_ids  # noqa: E402


def main() -> None:
    config = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    feats = ROOT / "data" / "processed" / "features" / "train_features.parquet"
    split_path = ROOT / "data" / "interim" / "split_blocks.tif"
    out_dir = ROOT / "data" / "processed" / "features_by_block"

    if not feats.exists():
        print("Falta " + str(feats) + ". Corre primero scripts/build_features.py")
        return

    t0 = time.time()
    print("Leyendo " + str(feats))
    df = pd.read_parquet(feats)
    print("Filas: " + format(len(df), ",") + " | columnas: " + str(df.shape[1])
          + " | t " + format(time.time() - t0, ".1f") + " s")

    with rasterio.open(split_path) as src:
        shape = (src.height, src.width)
    block_px = block_side_px(config["aoi"]["block_size_km"],
                             config["processing"]["working_resolution_m"])
    print("Lado de bloque: " + str(block_px) + " px")

    df["block_id"] = block_ids(df["row"].to_numpy(), df["col"].to_numpy(),
                               shape, block_px).astype("int64")
    n_blocks = int(df["block_id"].nunique())
    print("Bloques distintos: " + str(n_blocks))

    if out_dir.exists():
        print("Borrando salida previa: " + str(out_dir))
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    t1 = time.time()
    df.to_parquet(out_dir, engine="pyarrow", partition_cols=["block_id"],
                  compression="snappy", index=False)
    print("Escritura completa en " + str(out_dir)
          + " | t " + format(time.time() - t1, ".1f") + " s")

    # Resumen
    n_parts = sum(1 for _ in out_dir.glob("block_id=*"))
    total_size = sum(p.stat().st_size for p in out_dir.rglob("*.parquet"))
    print("Particiones: " + str(n_parts)
          + " | tamano comprimido: " + format(total_size / 2**20, ".1f") + " MB")
    print("Listo. Sigue: scripts/upload_to_s3.py --only features_by_block")


if __name__ == "__main__":
    main()
