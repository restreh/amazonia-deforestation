"""U-Net con encoder ResNet-34 y perdidas focal y Dice para segmentacion binaria.

El modelo se construye con segmentation_models_pytorch. La entrada tiene tantos
canales como capas apiladas (56 por defecto con mascaras de validez) y la salida es
un mapa de logits de un canal.

La perdida focal (Lin et al., 2017) mitiga el desbalance severo de clases. La
perdida Dice (Milletari et al., 2016) optimiza el solape directo y empuja el recall
en clases raras. Ambas se aplican con un peso por pixel, asi solo los pixeles del
split objetivo (peso 1) aportan y validacion o prueba nunca contaminan el
entrenamiento. train_unet.py combina ambas con coeficientes configurables.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def build_unet(in_channels, encoder="resnet34", encoder_weights="imagenet", classes=1):
    """Construye una U-Net de segmentation_models_pytorch (salida en logits)."""
    import segmentation_models_pytorch as smp
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None,
    )


def focal_loss(logits, target, weight, alpha=0.25, gamma=2.0):
    """Perdida focal binaria con logits y peso por pixel.

    logits, target, weight tienen igual forma (N, 1, H, W). Devuelve un escalar con el
    promedio de la perdida sobre los pixeles con peso > 0.
    """
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * target + (1.0 - p) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    loss = alpha_t * (1.0 - p_t).pow(gamma) * ce
    loss = loss * weight
    denom = weight.sum().clamp(min=1.0)
    return loss.sum() / denom


def dice_loss(logits, target, weight, eps=1.0):
    """Perdida Dice binaria con peso por pixel, agregada por muestra.

    logits, target, weight tienen forma (N, 1, H, W). Calcula el Dice de cada muestra
    sobre los pixeles con peso > 0 y devuelve 1 - mean(dice_n). El eps en numerador y
    denominador estabiliza recortes con pocos positivos.
    """
    p = torch.sigmoid(logits) * weight
    t = target * weight
    dims = (1, 2, 3)
    inter = (p * t).sum(dim=dims)
    denom = p.sum(dim=dims) + t.sum(dim=dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    return (1.0 - dice).mean()


@torch.no_grad()
def masked_scores(logits, target, weight):
    """Suma de aciertos para metricas enmascaradas: devuelve (probas, target, weight) aplanados.

    Util para acumular AUC/F1 sobre los pixeles validos a lo largo de un epoch.
    """
    p = torch.sigmoid(logits).reshape(-1)
    t = target.reshape(-1)
    w = weight.reshape(-1)
    m = w > 0
    return p[m].detach().cpu().numpy(), t[m].detach().cpu().numpy()
