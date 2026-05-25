"""amazonia_deforestation: deteccion de deforestacion por pixel sobre el arco
amazonico colombiano con Sentinel-2 y aprendizaje automatico.

Subpaquetes:
    ingest      Acceso a Sentinel-2 via STAC y construccion de cubos perezosos.
    data        Composiciones trimestrales, mascaras SCL e indices espectrales.
    features    Atributos contextuales y metricas de textura GLCM.
    spatial     Diagnostico (I de Moran, semivariograma) y particion por bloques.
    models      Random Forest, XGBoost, U-Net y perdida focal.
    evaluation  Metricas a nivel de pixel y poligono, McNemar, bootstrap espacial.
    viz         Utilidades de visualizacion de resultados.
"""

__version__ = "0.1.0"
