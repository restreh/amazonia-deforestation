"""Metricas de evaluacion a nivel de pixel y de poligono.

Pixel: precision, recall, F1, IoU a un umbral dado, y metricas independientes del
umbral (AUC-ROC, AUC-PR). El umbral se calibra sobre validacion maximizando F1, a
prevalencia real, y luego se aplica una sola vez sobre prueba.

Poligono: las predicciones se binarizan, se limpian con operaciones morfologicas
(cierre y apertura) y se reconstruyen en parches conectados (componentes de 8
vecinos); cada parche se compara con los parches de la etiqueta por IoU, con un
umbral de coincidencia (0.3 por defecto), de forma analoga a deteccion de objetos.
"""

from __future__ import annotations

import numpy as np


def calibrate_threshold(y, proba):
    """Umbral que maximiza F1 sobre (y, proba). Devuelve (umbral, f1)."""
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y, proba)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    # thr tiene longitud len(prec)-1; f1[:-1] corresponde a cada umbral.
    k = int(np.argmax(f1[:-1])) if len(thr) else 0
    return float(thr[k]) if len(thr) else 0.5, float(f1[k]) if len(thr) else 0.0


def pixel_metrics(y, proba, threshold):
    """Precision, recall, F1, IoU al umbral dado, mas AUC-ROC y AUC-PR (sin umbral)."""
    from sklearn.metrics import (precision_recall_fscore_support, jaccard_score,
                                 roc_auc_score, average_precision_score)
    yhat = (proba >= threshold).astype(np.uint8)
    pr, rc, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
    return {
        "threshold": float(threshold),
        "precision": float(pr),
        "recall": float(rc),
        "f1": float(f1),
        "iou": float(jaccard_score(y, yhat, zero_division=0)),
        "auc_roc": float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else float("nan"),
        "auc_pr": float(average_precision_score(y, proba)) if len(np.unique(y)) > 1 else float("nan"),
        "prevalence": float(y.mean()),
        "n_pixels": int(y.size),
    }


def _components(mask, min_area, morphology):
    """Etiqueta componentes conexas (8-vecinos), tras morfologia opcional y filtro de area minima."""
    from scipy.ndimage import label as cc_label, binary_closing, binary_opening
    st = np.ones((3, 3), dtype=bool)
    if morphology:
        mask = binary_opening(binary_closing(mask, st), st)
    lab, n = cc_label(mask, structure=np.ones((3, 3), dtype=int))
    if min_area > 1 and n > 0:
        sizes = np.bincount(lab.ravel())
        small = np.where(sizes < min_area)[0]
        small = small[small > 0]
        if small.size:
            mask = mask & ~np.isin(lab, small)
            lab, n = cc_label(mask, structure=np.ones((3, 3), dtype=int))
    return lab, n


def polygon_metrics(pred_mask, true_mask, iou_threshold=0.3, min_area=6, morphology=True):
    """Metricas por poligono: precision, recall, F1 e IoU medio de los emparejados.

    pred_mask / true_mask: arreglos 2D booleanos sobre la region de evaluacion.
    Empareja parches predichos con parches verdaderos por IoU (asignacion voraz).
    """
    pl, n_pred = _components(pred_mask, min_area, morphology)
    tl, n_true = _components(true_mask, min_area, morphology=False)
    if n_pred == 0 or n_true == 0:
        tp = 0
        return {
            "polygon_precision": 0.0 if n_pred else float("nan"),
            "polygon_recall": 0.0 if n_true else float("nan"),
            "polygon_f1": 0.0,
            "mean_iou_matched": float("nan"),
            "n_pred": int(n_pred), "n_true": int(n_true), "tp": 0,
            "iou_threshold": float(iou_threshold),
        }

    area_pred = np.bincount(pl.ravel(), minlength=n_pred + 1)
    area_true = np.bincount(tl.ravel(), minlength=n_true + 1)
    both = (pl > 0) & (tl > 0)
    key = pl[both].astype(np.int64) * (n_true + 1) + tl[both]
    uniq, cnt = np.unique(key, return_counts=True)
    pi = (uniq // (n_true + 1)).astype(int)
    ti = (uniq % (n_true + 1)).astype(int)
    inter = cnt.astype(np.float64)
    union = area_pred[pi] + area_true[ti] - inter
    iou = inter / union

    order = np.argsort(-iou)
    used_p, used_t = set(), set()
    matched_iou = []
    for k in order:
        if iou[k] < iou_threshold:
            break
        p, t = int(pi[k]), int(ti[k])
        if p in used_p or t in used_t:
            continue
        used_p.add(p); used_t.add(t); matched_iou.append(float(iou[k]))
    tp = len(matched_iou)
    precision = tp / n_pred
    recall = tp / n_true
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "polygon_precision": float(precision),
        "polygon_recall": float(recall),
        "polygon_f1": float(f1),
        "mean_iou_matched": float(np.mean(matched_iou)) if matched_iou else float("nan"),
        "n_pred": int(n_pred), "n_true": int(n_true), "tp": int(tp),
        "iou_threshold": float(iou_threshold),
    }
