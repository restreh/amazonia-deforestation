"""Carga de la configuracion central del proyecto desde config/config.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Raiz del repositorio = dos niveles por encima de este archivo.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Lee y devuelve la configuracion del proyecto como diccionario."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


if __name__ == "__main__":
    import json

    print(json.dumps(load_config(), indent=2, ensure_ascii=False))
