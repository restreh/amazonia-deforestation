"""Modelos supervisados y función de pérdida.

Responsabilidades:
    - Baselines Random Forest y XGBoost sobre atributos tabulares por píxel.
    - U-Net con encoder ResNet-34 preentrenado (segmentation_models_pytorch).
    - Pérdida focal (alpha=0.25, gamma=2.0) para el desbalance severo de clases.
    - Registro de experimentos con MLflow.
"""
