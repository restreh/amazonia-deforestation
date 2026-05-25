"""U-Net con encoder ResNet-34 y perdida focal para segmentacion binaria.

El modelo se construye con segmentation_models_pytorch. La entrada tiene tantos
canales como capas apiladas (52 por defecto: bandas de la mediana e indices de los
cuatro trimestres) y la salida es un mapa de logits de un canal.

La perdida focal (Lin et al., 2017) mitiga el desbalance severo de clases. Se aplica
con un peso por pixel: solo los pixeles del split objetivo (peso 1) aportan, de modo
que validacion y prueba nunca contaminan el entrenamiento.
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

    logits, target, weight: tensores de igual forma (N, 1, H, W) o (N, H, W).
    Devuelve un escalar: promedio de la perdida sobre los pixeles con peso > 0.
    """
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * target + (1.0 - p) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    loss = alpha_t * (1.0 - p_t).pow(gamma) * ce
    loss = loss * weight
    denom = weight.sum().clamp(min=1.0)
    return loss.sum() / denom


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
