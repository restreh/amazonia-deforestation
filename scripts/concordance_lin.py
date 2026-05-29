"""Coeficiente de concordancia de Lin (CCC) entre hectareas predichas y de referencia.

Uso (desde la raiz del repositorio, tras evaluar cada modelo):
    python scripts/concordance_lin.py
    python scripts/concordance_lin.py --models xgboost unet ensemble random_forest

La propuesta compromete reportar Lin's CCC entre hectareas detectadas por trimestre y
municipio del modelo y del SMByC-IDEAM. Como el modelo aqui predice anualmente sobre
Hansen 2024 (no por trimestre) y el AOI cubre solo dos municipios, la version literal
no es factible con la salida actual. Se reporta en su lugar:

  1. CCC a nivel de bloque espacial (5 km) entre hectareas predichas y hectareas de
     referencia (Hansen), sobre bloques de validacion y prueba. Es una serie larga y
     metodologicamente robusta de la concordancia interna del modelo con su etiqueta.
  2. Totales anuales por municipio del AOI (Cartagena del Chaira, San Vicente del
     Caguan) y de la AOI completa, para comparacion cualitativa con cifras IDEAM si
     el usuario las dispone (se pueden anexar manualmente al reporte).

Salida: data/interim/concordance_lin.json y tabla resumen por stdout.

Dependencias: rasterio, numpy, geopandas, pyyaml.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Asegura UTF-8 en stdout para nombres de municipios con tildes
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

import numpy as np  # noqa: E402
import rasterio  # noqa: E402
import yaml  # noqa: E402

from amazonia_deforestation.spatial.block_split import (  # noqa: E402
    assign_block_ids, block_side_px)

VAL_CODE, TEST_CODE = 2, 3
PIXEL_HA = 0.04  # 20 m x 20 m = 400 m^2 = 0.04 ha


def lin_ccc(x, y):
    """Coeficiente de concordancia de Lin entre dos vectores."""
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = ((x - mx) * (y - my)).mean()
    denom = vx + vy + (mx - my) ** 2
    return float(2 * cov / denom) if denom > 0 else float("nan")


def load_proba_and_threshold(name):
    proba_path = ROOT / "data" / "processed" / "predictions" / ("proba_" + name + ".tif")
    eval_path = ROOT / "data" / "interim" / ("eval_" + name + ".json")
    if not proba_path.exists():
        raise FileNotFoundError("Falta " + str(proba_path))
    if not eval_path.exists():
        raise FileNotFoundError("Falta " + str(eval_path))
    with rasterio.open(proba_path) as src:
        proba = src.read(1)
    thr = float(json.loads(eval_path.read_text(encoding="utf-8"))["threshold"])
    return proba, thr


def aggregate_per_block(values, bids, mask):
    """Suma de values por bloque sobre los pixeles con mask=True."""
    flat_vals = values[mask].astype("int64")
    flat_bids = bids[mask]
    out = np.bincount(flat_bids, weights=flat_vals)
    return out


def per_municipality_hectares(pred_bin, label, split, label_path, geojson_path):
    """Hectareas predichas y de referencia por municipio dentro del AOI."""
    try:
        import geopandas as gpd
        from rasterio.features import geometry_mask
    except ImportError:
        return None

    with rasterio.open(label_path) as src:
        transform = src.transform
        shape = (src.height, src.width)
        raster_crs = src.crs

    gdf = gpd.read_file(geojson_path).to_crs(raster_crs)
    name_field = next((c for c in ("MpNombre", "mpio_cnmbr", "nombre", "NOMBRE_MPI")
                       if c in gdf.columns), gdf.columns[0])
    test_or_val = (split == VAL_CODE) | (split == TEST_CODE)
    rows = []
    for _, r in gdf.iterrows():
        mask = ~geometry_mask([r.geometry], out_shape=shape, transform=transform,
                              invert=False)
        eval_in_mun = mask & test_or_val
        pred_ha = float(pred_bin[eval_in_mun].sum()) * PIXEL_HA
        true_ha = float((label[eval_in_mun] == 1).sum()) * PIXEL_HA
        rows.append({"municipio": str(r[name_field]),
                     "pixeles_eval": int(eval_in_mun.sum()),
                     "predicted_ha": pred_ha, "truth_ha": true_ha})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Lin's CCC entre hectareas predichas y de referencia")
    ap.add_argument("--models", nargs="+",
                    default=["xgboost", "unet", "ensemble"])
    args = ap.parse_args()

    config = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    label_path = ROOT / "data" / "interim" / "label_2024_20m.tif"
    split_path = ROOT / "data" / "interim" / "split_blocks.tif"
    geojson = ROOT / "data" / "external" / "aoi_municipalities.geojson"

    with rasterio.open(label_path) as src:
        label = src.read(1)
    with rasterio.open(split_path) as src:
        split = src.read(1)
    block_px = block_side_px(config["aoi"]["block_size_km"],
                             config["processing"]["working_resolution_m"])
    bids = assign_block_ids(label.shape, block_px)

    eval_mask = (split == VAL_CODE) | (split == TEST_CODE)
    print("Pixeles val+test: " + format(int(eval_mask.sum()), ","))

    truth_per_block_full = aggregate_per_block((label == 1).astype("int8"), bids, eval_mask)
    truth_ha_full = truth_per_block_full * PIXEL_HA
    valid_blocks = np.where(truth_per_block_full + 0 >= 0)[0]  # todos los bloques tocados
    # filtra solo bloques con al menos un pixel de evaluacion
    n_eval_per_block = aggregate_per_block(np.ones_like(label, dtype="int8"), bids, eval_mask)
    valid_blocks = np.where(n_eval_per_block > 0)[0]
    truth_ha = truth_ha_full[valid_blocks]
    print("Bloques con pixeles val+test: " + str(valid_blocks.size))

    summary = {"models": {}, "block_side_km": float(config["aoi"]["block_size_km"]),
               "n_blocks": int(valid_blocks.size)}

    print("\n=== Lin's CCC por bloque (hectareas predichas vs Hansen) ===")
    print("  modelo   |    CCC  | mean_pred | mean_true | corr_pearson")

    for m in args.models:
        proba, thr = load_proba_and_threshold(m)
        pred_bin = ((proba >= thr) & np.isfinite(proba)).astype("int8")
        pred_per_block = aggregate_per_block(pred_bin, bids, eval_mask)
        pred_ha = pred_per_block[valid_blocks] * PIXEL_HA

        ccc = lin_ccc(pred_ha, truth_ha)
        pearson = float(np.corrcoef(pred_ha, truth_ha)[0, 1]) if pred_ha.std() > 0 and truth_ha.std() > 0 else float("nan")
        mp = float(pred_ha.mean())
        mt = float(truth_ha.mean())

        mun = per_municipality_hectares(pred_bin, label, split, label_path, geojson) \
            if geojson.exists() else None
        total_pred = float(pred_bin[eval_mask].sum()) * PIXEL_HA
        total_true = float((label[eval_mask] == 1).sum()) * PIXEL_HA

        summary["models"][m] = {
            "ccc_block": ccc, "pearson_block": pearson,
            "mean_predicted_ha_per_block": mp, "mean_truth_ha_per_block": mt,
            "total_predicted_ha_aoi": total_pred, "total_truth_ha_aoi": total_true,
            "by_municipality": mun,
        }

        print("  " + m.rjust(8) + " | " + format(ccc, ".4f").rjust(7)
              + " | " + format(mp, ".2f").rjust(9)
              + " | " + format(mt, ".2f").rjust(9)
              + " | " + format(pearson, ".4f").rjust(12))

    print("\n=== Hectareas totales val+test del AOI ===")
    print("  modelo   | predichas |   Hansen | razon p/H")
    for m in args.models:
        d = summary["models"][m]
        ratio = d["total_predicted_ha_aoi"] / d["total_truth_ha_aoi"] \
            if d["total_truth_ha_aoi"] > 0 else float("nan")
        print("  " + m.rjust(8) + " | "
              + format(d["total_predicted_ha_aoi"], ".0f").rjust(9) + " | "
              + format(d["total_truth_ha_aoi"], ".0f").rjust(8) + " | "
              + format(ratio, ".3f").rjust(9))

    # Tabla por municipio (la primera disponible)
    first = next((m for m in args.models if summary["models"][m]["by_municipality"]), None)
    if first is not None:
        muns = [r["municipio"] for r in summary["models"][first]["by_municipality"]]
        print("\n=== Hectareas val+test por municipio ===")
        header = "  modelo   | " + " | ".join(name.rjust(12) for name in muns)
        print(header)
        for m in args.models:
            row = summary["models"][m]["by_municipality"]
            if not row:
                continue
            cells = [format(r["predicted_ha"], ".0f").rjust(12) for r in row]
            print("  " + m.rjust(8) + " | " + " | ".join(cells))
        truth_cells = [format(r["truth_ha"], ".0f").rjust(12)
                       for r in summary["models"][first]["by_municipality"]]
        print("  " + "Hansen".rjust(8) + " | " + " | ".join(truth_cells))

    out = ROOT / "data" / "interim" / "concordance_lin.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nReporte guardado en " + str(out))


if __name__ == "__main__":
    main()
