"""Preparación de datos: composiciones temporales y derivados espectrales.

Responsabilidades:
    - Componer trimestres por agregación de mediana y percentil 25.
    - Aplicar la máscara de clasificación de escena (SCL) para nubes y sombras.
    - Calcular índices espectrales (NDVI, NBR, NDWI) y apilar bandas e índices.
    - Derivar etiquetas binarias desde la capa lossyear de Hansen GFC.
"""
