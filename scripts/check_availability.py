"""Reporta la disponibilidad y nubosidad de Sentinel-2 sobre el AOI por trimestre.

Uso (desde la raiz del repositorio):
    python scripts/check_availability.py

Requiere internet. Dependencias minimas:
    pip install pystac-client pandas pyyaml
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from amazonia_deforestation.ingest.stac_query import availability_report  # noqa: E402


def main() -> None:
    config = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))

    print("Consultando STAC Earth-Search sobre el AOI...")
    detail, summary = availability_report(config)

    if detail.empty:
        print("No se encontraron escenas. Revisa el bbox y el filtro de nubosidad.")
        return

    out_dir = ROOT / "data" / "interim"
    out_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out_dir / "s2_availability_detail.csv", index=False)
    summary.to_csv(out_dir / "s2_availability_summary.csv", index=False)

    print("\n=== Resumen por trimestre ===")
    print(summary.to_string(index=False))
    print(f"\nDetalle por escena: {len(detail)} filas.")
    print(f"Archivos guardados en {out_dir}")


if __name__ == "__main__":
    main()
