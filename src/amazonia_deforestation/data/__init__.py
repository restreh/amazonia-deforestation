"""Preparacion de datos: composiciones temporales y derivados espectrales.

Responsabilidades:
    - Componer trimestres por agregacion de mediana y percentil 25.
    - Aplicar la mascara de clasificacion de escena (SCL) para nubes y sombras.
    - Calcular indices espectrales (NDVI, NBR, NDWI) y apilar bandas e indices.
    - Derivar etiquetas binarias desde la capa lossyear de Hansen GFC.
"""
