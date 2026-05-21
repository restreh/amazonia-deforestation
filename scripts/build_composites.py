"""Construye las composiciones trimestrales de Sentinel-2 sobre el AOI.

Uso (desde la raíz del repositorio):
    python scripts/build_composites.py            # corrida completa
    python scripts/build_composites.py --limit 8  # prueba: 8 escenas por trimestre

Recomendado correr en SageMaker Studio Lab (más RAM). Requiere internet.
Dependencias (subconjunto de requirements.txt):
    pip install stackstac pystac-client rioxarray rasterio dask numpy pyyaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from amazonia_deforestation.data.composites import build_all_quarters  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Composiciones trimestrales Sentinel-2")
    parser.add_argument("--limit", type=int, default=None,
                        help="máximo de escenas por trimestre (modo prueba)")
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    out_dir = ROOT / "data" / "processed" / "composites"
    if args.limit:
        print(f"Modo prueba: {args.limit} escenas por trimestre.")
    build_all_quarters(config, out_dir, scene_limit=args.limit)
    print(f"\nComposiciones en {out_dir}")


if __name__ == "__main__":
    main()
