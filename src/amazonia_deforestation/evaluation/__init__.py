"""Evaluación y comparación estadística de modelos.

Responsabilidades:
    - Métricas a nivel de píxel: precisión, recall, F1, IoU, AUC-PR.
    - Métricas a nivel de polígono con umbral de coincidencia IoU >= 0.3.
    - Prueba de McNemar para comparación pareada de modelos.
    - Bootstrap espacial por bloques para intervalos de confianza.
    - Coeficiente de concordancia de Lin frente a cifras del SMByC-IDEAM.
"""
