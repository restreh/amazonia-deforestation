"""Bootstrap espacial por bloques para intervalos de confianza de F1, IoU y AUC-PR.

Uso (desde la raiz del repositorio, tras evaluar cada modelo):
    python scripts/bootstrap_spatial.py
    python scripts/bootstrap_spatial.py --models xgboost unet ensemble random_forest --B 1000

Sobre el conjunto de prueba, remuestrea bloques espaciales con reposicion B veces y
recomputa F1, IoU, precision, recall y AUC-PR a nivel de pixel para cada modelo. El
remuestreo por bloques (no por pixel) respeta la autocorrelacion espacial; el bootstrap
estandar por pixel sobreestima el poder estadistico porque trata pixeles vecinos como
independientes (Roberts et al., 2017; Ploton et al., 2020; Karasiak et al., 2022).

Salida: data/interim/bootstrap_spatial.json con percentiles 2.5%, 50%, 97.5% por
modelo y por metrica, y una tabla resumen por stdout.

Dependencias: rasterio, numpy, scipy, scikit-learn, pyyaml.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import rasterio  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

from amazonia_deforestation.spatial.block_split import (  # noqa: E402
    assign_block_ids, block_side_px)

TEST_CODE = 3


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


def percentiles(arr):
    a = np.asarray(arr, dtype="float64")
    return {
        "p2_5": float(np.nanpercentile(a, 2.5)),
        "p50": float(np.nanpercentile(a, 50.0)),
        "p97_5": float(np.nanpercentile(a, 97.5)),
        "mean": float(np.nanmean(a)),
        "std": float(np.nanstd(a)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Bootstrap espacial por bloques sobre el conjunto de prueba")
    ap.add_argument("--models", nargs="+",
                    default=["xgboost", "unet", "ensemble"])
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--auc-pr-B", type=int, default=200,
                    help="iteraciones para AUC-PR (mas lento que F1/IoU)")
    args = ap.parse_args()

    config = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    label_path = ROOT / "data" / "interim" / "label_2024_20m.tif"
    split_path = ROOT / "data" / "interim" / "split_blocks.tif"
    with rasterio.open(label_path) as src:
        label = src.read(1)
    with rasterio.open(split_path) as src:
        split = src.read(1)

    probas, thresholds = {}, {}
    for m in args.models:
        p, t = load_proba_and_threshold(m)
        probas[m] = p
        thresholds[m] = t
        print("Modelo " + m + " | umbral " + format(t, ".4f"))

    block_px = block_side_px(config["aoi"]["block_size_km"],
                             config["processing"]["working_resolution_m"])
    bids = assign_block_ids(label.shape, block_px)
    print("Lado de bloque: " + str(block_px) + " px")

    test_mask = split == TEST_CODE
    for m in args.models:
        test_mask &= np.isfinite(probas[m])
    n_test = int(test_mask.sum())
    print("Pixeles de prueba con cobertura en todos los modelos: " + format(n_test, ","))

    flat_bids = bids[test_mask]
    flat_label = label[test_mask].astype("int8")
    unique_blocks = np.unique(flat_bids)
    n_blocks = unique_blocks.size
    print("Bloques de prueba: " + str(n_blocks))

    # Indices de pixeles por bloque para muestreo eficiente
    order = np.argsort(flat_bids, kind="stable")
    sorted_bids = flat_bids[order]
    block_starts = np.searchsorted(sorted_bids, unique_blocks, side="left")
    block_ends = np.searchsorted(sorted_bids, unique_blocks, side="right")
    pixel_index_by_block = [order[block_starts[i]:block_ends[i]] for i in range(n_blocks)]

    # Predicciones binarias por modelo sobre pixeles de prueba ordenados por bloque
    flat_proba_by_model = {m: probas[m][test_mask] for m in args.models}
    flat_pred_by_model = {m: (flat_proba_by_model[m] >= thresholds[m]).astype("int8")
                          for m in args.models}

    # Conteos por bloque por modelo para F1/IoU/precision/recall sin recomputar el binario
    counts = {m: np.zeros((n_blocks, 4), dtype="int64") for m in args.models}
    for i, idx in enumerate(pixel_index_by_block):
        y = flat_label[idx]
        for m in args.models:
            p = flat_pred_by_model[m][idx]
            tp = int(((p == 1) & (y == 1)).sum())
            fp = int(((p == 1) & (y == 0)).sum())
            fn = int(((p == 0) & (y == 1)).sum())
            tn = int(((p == 0) & (y == 0)).sum())
            counts[m][i, :] = (tp, fp, fn, tn)

    rng = np.random.default_rng(args.seed)
    metrics_runs = {m: {"precision": [], "recall": [], "f1": [], "iou": []} for m in args.models}
    t0 = time.time()
    for b in range(args.B):
        sample = rng.integers(0, n_blocks, size=n_blocks)
        for m in args.models:
            agg = counts[m][sample].sum(axis=0)
            tp, fp, fn, tn = int(agg[0]), int(agg[1]), int(agg[2]), int(agg[3])
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            metrics_runs[m]["precision"].append(prec)
            metrics_runs[m]["recall"].append(rec)
            metrics_runs[m]["f1"].append(f1)
            metrics_runs[m]["iou"].append(iou)
        if (b + 1) % max(1, args.B // 10) == 0:
            dt = time.time() - t0
            print("  F1/IoU bootstrap " + str(b + 1) + "/" + str(args.B)
                  + " | " + format(dt, ".1f") + " s", flush=True)

    # AUC-PR: se computa con probabilidades; mas lento, se reduce B
    auc_runs = {m: [] for m in args.models}
    t0 = time.time()
    for b in range(args.auc_pr_B):
        sample = rng.integers(0, n_blocks, size=n_blocks)
        idx = np.concatenate([pixel_index_by_block[i] for i in sample])
        y_b = flat_label[idx]
        if y_b.sum() == 0:
            continue
        for m in args.models:
            p_b = flat_proba_by_model[m][idx]
            ap = float(average_precision_score(y_b, p_b))
            auc_runs[m].append(ap)
        if (b + 1) % max(1, args.auc_pr_B // 10) == 0:
            dt = time.time() - t0
            print("  AUC-PR bootstrap " + str(b + 1) + "/" + str(args.auc_pr_B)
                  + " | " + format(dt, ".1f") + " s", flush=True)

    # Resumen
    summary = {"B": args.B, "auc_pr_B": args.auc_pr_B, "n_test_pixels": n_test,
               "n_test_blocks": int(n_blocks), "block_px": int(block_px),
               "thresholds": thresholds, "models": {}}
    print("\n=== Bootstrap espacial (IC 95 %) ===")
    print("  modelo   |  metric  |    p2.5 |    p50  |   p97.5 |   mean  |   std")
    for m in args.models:
        d = {}
        for k, v in metrics_runs[m].items():
            d[k] = percentiles(v)
        d["auc_pr"] = percentiles(auc_runs[m]) if auc_runs[m] else None
        summary["models"][m] = d
        for k in ("precision", "recall", "f1", "iou", "auc_pr"):
            stats_k = d[k]
            if stats_k is None:
                continue
            print("  " + m.rjust(8) + " | " + k.ljust(8)
                  + " | " + format(stats_k["p2_5"], ".4f").rjust(7)
                  + " | " + format(stats_k["p50"], ".4f").rjust(7)
                  + " | " + format(stats_k["p97_5"], ".4f").rjust(7)
                  + " | " + format(stats_k["mean"], ".4f").rjust(7)
                  + " | " + format(stats_k["std"], ".4f").rjust(7))

    out = ROOT / "data" / "interim" / "bootstrap_spatial.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nReporte guardado en " + str(out))


if __name__ == "__main__":
    main()
