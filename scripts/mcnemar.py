"""Prueba de McNemar pareada entre modelos sobre el conjunto de prueba.

Uso (desde la raiz del repositorio, tras evaluar cada modelo):
    python scripts/mcnemar.py
    python scripts/mcnemar.py --models xgboost unet ensemble random_forest

Para cada par de modelos, construye una tabla 2x2 de aciertos en los pixeles del
conjunto de prueba (split == 3) que tengan probabilidad finita en ambos rasters. La
prediccion binaria de cada modelo se deriva con su umbral calibrado en validacion,
que se lee de data/interim/eval_<modelo>.json. Calcula el estadistico de McNemar con
correccion de continuidad y un p-valor exacto via binomial, y reporta si la
diferencia es significativa al 5 %.

La prueba de McNemar trata los pixeles como independientes; en presencia de
autocorrelacion espacial esto sobreestima el poder estadistico. Para una lectura
honesta se complementa con el bootstrap espacial por bloques (scripts/bootstrap_spatial.py).

Salida: data/interim/mcnemar.json y una tabla resumen por stdout.

Dependencias: rasterio, numpy, scipy, pyyaml.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from scipy import stats  # noqa: E402

TEST_CODE = 3


def load_model(name, label, split, valid_mask):
    """Devuelve la prediccion binaria de un modelo sobre los pixeles validos."""
    proba_path = ROOT / "data" / "processed" / "predictions" / ("proba_" + name + ".tif")
    eval_path = ROOT / "data" / "interim" / ("eval_" + name + ".json")
    if not proba_path.exists():
        raise FileNotFoundError("Falta " + str(proba_path))
    if not eval_path.exists():
        raise FileNotFoundError("Falta " + str(eval_path)
                                + ". Corre evaluate_baseline.py --model " + name)
    with rasterio.open(proba_path) as src:
        proba = src.read(1)
    thr = float(json.loads(eval_path.read_text(encoding="utf-8"))["threshold"])
    return proba, thr


def main() -> None:
    ap = argparse.ArgumentParser(description="Prueba de McNemar pareada entre modelos")
    ap.add_argument("--models", nargs="+",
                    default=["xgboost", "unet", "ensemble"],
                    help="lista de modelos a comparar")
    args = ap.parse_args()

    label_path = ROOT / "data" / "interim" / "label_2024_20m.tif"
    split_path = ROOT / "data" / "interim" / "split_blocks.tif"
    with rasterio.open(label_path) as src:
        label = src.read(1)
    with rasterio.open(split_path) as src:
        split = src.read(1)

    # Carga todas las probabilidades primero, para enmascarar pixeles con cobertura
    # en todos los modelos.
    probas = {}
    thresholds = {}
    for m in args.models:
        proba, thr = load_model(m, label, split, None)
        probas[m] = proba
        thresholds[m] = thr
        print("Modelo " + m + " | umbral " + format(thr, ".4f"))

    test = split == TEST_CODE
    valid = test.copy()
    for m in args.models:
        valid &= np.isfinite(probas[m])
    n_test = int(valid.sum())
    print("Pixeles de prueba con cobertura en todos los modelos: " + format(n_test, ","))

    y_true = label[valid].astype(np.int8)
    correct = {}
    for m in args.models:
        y_pred = (probas[m][valid] >= thresholds[m]).astype(np.int8)
        correct[m] = (y_pred == y_true).astype(np.int8)
        acc = float(correct[m].mean())
        print("  " + m + " | accuracy " + format(acc, ".4f"))

    # Tabla pareada
    pairs = []
    print("\n=== McNemar pareado ===")
    print("  modelo_a vs modelo_b |     n01 |     n10 |    chi2 |  p_chi2 | p_exact | sig 5%")
    for i, a in enumerate(args.models):
        for b in args.models[i + 1:]:
            ca = correct[a].astype(bool)
            cb = correct[b].astype(bool)
            n01 = int(((~ca) & cb).sum())   # a falla, b acierta
            n10 = int((ca & (~cb)).sum())   # a acierta, b falla
            n00 = int(((~ca) & (~cb)).sum())
            n11 = int((ca & cb).sum())

            disc = n01 + n10
            if disc == 0:
                chi2 = 0.0
                p_chi2 = 1.0
                p_exact = 1.0
            else:
                # Continuidad de Yates
                chi2 = (abs(n01 - n10) - 1) ** 2 / disc
                p_chi2 = float(stats.chi2.sf(chi2, df=1))
                # Binomial exacta (mas conservadora con disc pequeno)
                k = min(n01, n10)
                p_exact = float(stats.binomtest(k, disc, 0.5).pvalue)

            sig = p_exact < 0.05
            print("  " + a.rjust(8) + " vs " + b.ljust(11)
                  + " | " + format(n01, ",").rjust(7)
                  + " | " + format(n10, ",").rjust(7)
                  + " | " + format(chi2, ".2f").rjust(7)
                  + " | " + format(p_chi2, ".4f").rjust(7)
                  + " | " + format(p_exact, ".4f").rjust(7)
                  + " | " + ("SI" if sig else "no"))
            pairs.append({
                "a": a, "b": b,
                "acc_a": float(correct[a].mean()),
                "acc_b": float(correct[b].mean()),
                "n00": n00, "n01": n01, "n10": n10, "n11": n11,
                "chi2": float(chi2), "p_chi2": p_chi2, "p_exact": p_exact,
                "significant_at_0.05": sig,
                "winner": (a if n10 > n01 else b) if disc > 0 else None,
            })

    out = {"models": args.models, "thresholds": thresholds,
           "n_test_pixels": n_test, "pairs": pairs}
    out_path = ROOT / "data" / "interim" / "mcnemar.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nReporte guardado en " + str(out_path))


if __name__ == "__main__":
    main()
