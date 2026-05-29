"""Construye el raster de probabilidad del ensamble por promedio ponderado.

Uso (desde la raiz del repositorio, tras predict.py y predict_unet.py):
    python scripts/build_ensemble.py
    python scripts/build_ensemble.py --w-xgboost 0.6 --w-unet 0.4

Lee data/processed/predictions/proba_xgboost.tif y proba_unet.tif, los promedia pixel
a pixel con pesos configurables y escribe proba_ensemble.tif en el mismo formato. Los
pesos se normalizan a sumar 1. Los pixeles con NaN en cualquiera de los dos rasters
quedan en NaN en la salida, asi la evaluacion solo considera pixeles cubiertos por
ambos modelos.

La idea es que XGBoost (gradient boosting sobre 612 atributos contextuales) y U-Net
(red convolucional sobre 56 canales crudos) tienen sesgos distintos, asi que sus
errores no estan correlacionados al 100% y el promedio suele subir las metricas de
ambos. Es el caso tipico del capitulo de ensamble.

Dependencias: rasterio, numpy, pyyaml.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import rasterio  # noqa: E402


def check_aligned(a, b, name_a, name_b):
    """Verifica que dos datasets de rasterio coincidan en forma, CRS y transform."""
    if (a.height, a.width) != (b.height, b.width):
        raise RuntimeError(name_a + " y " + name_b + " tienen formas distintas: "
                           + str((a.height, a.width)) + " vs " + str((b.height, b.width)))
    if str(a.crs) != str(b.crs):
        raise RuntimeError(name_a + " y " + name_b + " tienen CRS distintos: "
                           + str(a.crs) + " vs " + str(b.crs))
    if tuple(a.transform)[:6] != tuple(b.transform)[:6]:
        raise RuntimeError(name_a + " y " + name_b + " tienen transform distintos.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Construye proba_ensemble.tif por promedio ponderado")
    ap.add_argument("--w-xgboost", type=float, default=0.5,
                    help="peso para proba_xgboost (defecto 0.5)")
    ap.add_argument("--w-unet", type=float, default=0.5,
                    help="peso para proba_unet (defecto 0.5)")
    args = ap.parse_args()

    pred_dir = ROOT / "data" / "processed" / "predictions"
    p_xgb = pred_dir / "proba_xgboost.tif"
    p_unet = pred_dir / "proba_unet.tif"
    p_out = pred_dir / "proba_ensemble.tif"
    for p in (p_xgb, p_unet):
        if not p.exists():
            print("Falta " + str(p) + ". Corre primero predict.py y/o predict_unet.py")
            return

    total = args.w_xgboost + args.w_unet
    if total <= 0:
        print("Los pesos deben sumar > 0.")
        return
    w_xgb = args.w_xgboost / total
    w_unet = args.w_unet / total
    print("Pesos normalizados | XGBoost: " + format(w_xgb, ".4f")
          + " | U-Net: " + format(w_unet, ".4f"))

    with rasterio.open(p_xgb) as src_xgb, rasterio.open(p_unet) as src_unet:
        check_aligned(src_xgb, src_unet, "proba_xgboost", "proba_unet")
        profile = src_xgb.profile.copy()
        xgb = src_xgb.read(1)
        unet = src_unet.read(1)

    # NaN en cualquiera -> NaN en el ensamble (mantiene la convencion del evaluador)
    mask_valid = np.isfinite(xgb) & np.isfinite(unet)
    out = np.full(xgb.shape, np.nan, dtype="float32")
    out[mask_valid] = (w_xgb * xgb[mask_valid] + w_unet * unet[mask_valid]).astype("float32")

    profile.update(count=1, dtype="float32", nodata=float("nan"), compress="deflate")
    with rasterio.open(p_out, "w", **profile) as dst:
        dst.write(out, 1)
        dst.set_band_description(1, "deforestation_probability_ensemble")

    n_valid = int(mask_valid.sum())
    n_total = int(mask_valid.size)
    print("Pixeles validos en ambos rasters: " + format(n_valid, ",")
          + " de " + format(n_total, ",")
          + " (" + format(100.0 * n_valid / max(1, n_total), ".2f") + " %)")
    print("Raster guardado en " + str(p_out))


if __name__ == "__main__":
    main()
