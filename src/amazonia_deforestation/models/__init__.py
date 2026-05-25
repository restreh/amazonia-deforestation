"""Modelos supervisados y funcion de perdida.

Responsabilidades:
    - Baselines Random Forest y XGBoost sobre atributos tabulares por pixel.
    - U-Net con encoder ResNet-34 preentrenado (segmentation_models_pytorch).
    - Perdida focal (alpha=0.25, gamma=2.0) para el desbalance severo de clases.
    - Registro de experimentos con MLflow.
"""
