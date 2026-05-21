"""amazonia_deforestation: detección de deforestación por píxel sobre el arco
amazónico colombiano con Sentinel-2 y aprendizaje automático.

Subpaquetes:
    ingest      Acceso a Sentinel-2 vía STAC y construcción de cubos perezosos.
    data        Composiciones trimestrales, máscaras SCL e índices espectrales.
    features    Atributos contextuales y métricas de textura GLCM.
    spatial     Diagnóstico (I de Moran, semivariograma) y partición por bloques.
    models      Random Forest, XGBoost, U-Net y pérdida focal.
    evaluation  Métricas a nivel de píxel y polígono, McNemar, bootstrap espacial.
    viz         Utilidades de visualización de resultados.
"""

__version__ = "0.1.0"
