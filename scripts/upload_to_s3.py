"""Sube los artefactos derivados del proyecto a S3 con particionamiento Hive.

Uso (desde la raiz del repositorio, con credenciales AWS configuradas):
    python scripts/upload_to_s3.py                       # subida completa
    python scripts/upload_to_s3.py --dry-run             # solo lista lo que subiria
    python scripts/upload_to_s3.py --only models metrics # solo grupos especificos

Mapea cada archivo local a su prefijo bajo s3://<project_bucket>/ siguiendo el
particionamiento por tile MGRS, trimestre, agregacion y modelo. Es idempotente:
si el objeto ya existe en S3 con el mismo tamano se omite. El bucket viene del
config.yaml (modeling.aws.project_bucket).

Grupos disponibles para --only:
  composites    rasters de composiciones trimestrales (median y p25)
  indices       rasters de indices NDVI/NBR/NDWI por trimestre
  interim       label, split, channel_stats
  features      train_features.parquet (3.2 GB) y train_sample.parquet
  predictions   probabilidades densas por modelo
  models        modelos entrenados (xgboost, rf, unet, unet_imagenet)
  metrics       eval/mcnemar/bootstrap/concordance/compare_cv y diagnosticos

Dependencias: pyyaml. AWS CLI v2 disponible en el PATH.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

TILE = "AOI_caqueta"   # un solo tile MGRS en el AOI; placeholder Hive
GROUPS = ("composites", "indices", "interim", "features", "features_by_block",
          "metrics_by_block", "predictions", "models", "metrics")


def build_mapping(root, bucket):
    """Devuelve lista de (local_path, s3_uri, group) para todos los artefactos."""
    mapping = []
    base = "s3://" + bucket + "/"

    # composites
    pat = re.compile(r"composite_(2024Q[1-4])_(median|p25)\.tif$")
    for p in sorted((root / "data" / "processed" / "composites").glob("composite_*.tif")):
        m = pat.search(p.name)
        if not m:
            continue
        quarter, agg = m.group(1), m.group(2)
        key = (f"derived/composites/tile={TILE}/quarter={quarter}/"
               f"aggregation={agg}/composite.tif")
        mapping.append((p, base + key, "composites"))

    # indices
    pat = re.compile(r"indices_(2024Q[1-4])\.tif$")
    for p in sorted((root / "data" / "processed" / "indices").glob("indices_*.tif")):
        m = pat.search(p.name)
        if not m:
            continue
        quarter = m.group(1)
        key = f"derived/indices/tile={TILE}/quarter={quarter}/indices.tif"
        mapping.append((p, base + key, "indices"))

    # interim: label, split, channel_stats
    interim_files = ["label_2024_20m.tif", "split_blocks.tif", "channel_stats.json"]
    for name in interim_files:
        p = root / "data" / "interim" / name
        if p.exists():
            mapping.append((p, base + f"derived/interim/{name}", "interim"))

    # features
    for name in ("train_features.parquet", "train_sample.parquet"):
        for d in (root / "data" / "processed" / "features",
                  root / "data" / "interim"):
            p = d / name
            if p.exists():
                mapping.append((p, base + f"derived/features/{name}", "features"))
                break

    # features_by_block: tabla Hive-particionada para Athena (directorio recursivo)
    fb_dir = root / "data" / "processed" / "features_by_block"
    if fb_dir.exists() and any(fb_dir.iterdir()):
        mapping.append((fb_dir, base + "derived/features_by_block/",
                        "features_by_block"))

    # metrics_by_block: tabla agregada por bloque y modelo (directorio con 1 parquet)
    mb_dir = root / "data" / "processed" / "metrics_by_block"
    if mb_dir.exists() and any(mb_dir.iterdir()):
        mapping.append((mb_dir, base + "derived/metrics_by_block/",
                        "metrics_by_block"))

    # predictions
    pat = re.compile(r"proba_([a-z_]+)\.tif$")
    for p in sorted((root / "data" / "processed" / "predictions").glob("proba_*.tif")):
        m = pat.search(p.name)
        if not m:
            continue
        model = m.group(1)
        key = f"derived/predictions/model={model}/proba.tif"
        mapping.append((p, base + key, "predictions"))

    # models
    for p in sorted((root / "models").glob("*")):
        if p.suffix in (".pt", ".json", ".joblib", ".pkl"):
            mapping.append((p, base + f"models/{p.name}", "models"))

    # metrics: todos los json, csv y txt de data/interim que no sean rasters ni features
    skip = {"label_2024_20m.tif", "split_blocks.tif", "channel_stats.json",
            "train_sample.parquet"}
    for p in sorted((root / "data" / "interim").iterdir()):
        if p.is_dir() or p.name in skip or p.name.startswith("."):
            continue
        if p.suffix in (".json", ".csv", ".txt"):
            mapping.append((p, base + f"metrics/{p.name}", "metrics"))

    return mapping


def s3_object_size(s3_uri):
    """Devuelve el tamano del objeto en bytes, o None si no existe."""
    parts = s3_uri[5:].split("/", 1)
    bucket, key = parts[0], parts[1]
    res = subprocess.run(
        ["aws", "s3api", "head-object", "--bucket", bucket, "--key", key],
        capture_output=True, text=True)
    if res.returncode != 0:
        return None
    return int(json.loads(res.stdout).get("ContentLength", 0))


def total_size(path):
    """Tamano de un archivo o directorio (suma recursiva)."""
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def upload_one(local_path, s3_uri):
    """Sube un archivo (aws s3 cp) o un directorio (aws s3 sync)."""
    if local_path.is_dir():
        subprocess.run(["aws", "s3", "sync", str(local_path), s3_uri,
                        "--size-only"], check=True)
    else:
        subprocess.run(["aws", "s3", "cp", str(local_path), s3_uri], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sube derivados a S3 con Hive partitioning")
    ap.add_argument("--dry-run", action="store_true",
                    help="solo lista los archivos y destinos, no sube")
    ap.add_argument("--only", nargs="+", choices=GROUPS,
                    help="restringe a uno o varios grupos")
    args = ap.parse_args()

    config = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    bucket = config["aws"]["project_bucket"]
    if not bucket or bucket.startswith("<"):
        print("Falta config.aws.project_bucket en config.yaml")
        return

    mapping = build_mapping(ROOT, bucket)
    if args.only:
        wanted = set(args.only)
        mapping = [m for m in mapping if m[2] in wanted]
    if not mapping:
        print("Nada para subir.")
        return

    total = sum(total_size(p) for p, _, _ in mapping)
    print("Bucket: " + bucket)
    print("Entradas a considerar: " + str(len(mapping))
          + " | tamano total local: " + format(total / 2**30, ".2f") + " GB")
    print("Grupos: " + ", ".join(sorted(set(g for _, _, g in mapping))))
    print()

    n_skipped = 0
    n_uploaded = 0
    bytes_uploaded = 0
    for local_path, s3_uri, group in mapping:
        size_local = total_size(local_path)
        # Directorios siempre se sincronizan (sync hace el skip-if-exists por archivo)
        if local_path.is_file():
            size_remote = s3_object_size(s3_uri)
            if size_remote == size_local:
                print("  SKIP  [" + group + "] " + local_path.name + " (ya en S3)")
                n_skipped += 1
                continue
        action = "DRY  " if args.dry_run else "UP   "
        kind = "dir " if local_path.is_dir() else "file"
        print(f"  {action} [{group}] {kind} {local_path.name}"
              f" -> {s3_uri}  ({size_local / 2**20:.1f} MB)")
        if not args.dry_run:
            upload_one(local_path, s3_uri)
            n_uploaded += 1
            bytes_uploaded += size_local

    print()
    print("Saltados: " + str(n_skipped)
          + " | subidos: " + str(n_uploaded)
          + " | bytes subidos: " + format(bytes_uploaded / 2**30, ".2f") + " GB")


if __name__ == "__main__":
    main()
