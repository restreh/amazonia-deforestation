"""Deriva la etiqueta binaria de deforestacion de Hansen GFC sobre el AOI.

Uso (desde la raiz del repositorio):
    python scripts/download_labels.py

Lee por ventana sin descargar el tile completo. Requiere internet.
Dependencias minimas:
    pip install rasterio numpy pyyaml
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from amazonia_deforestation.ingest.hansen import build_label  # noqa: E402


def main() -> None:
    config = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    out_dir = ROOT / "data" / "interim"
    print("Leyendo Hansen GFC sobre el AOI...")
    build_label(config, out_dir)


if __name__ == "__main__":
    main()
