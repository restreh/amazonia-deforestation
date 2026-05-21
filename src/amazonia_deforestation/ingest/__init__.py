"""Acceso a Sentinel-2 vía STAC Earth-Search y construcción de cubos de datos.

Responsabilidades:
    - Consultar la API STAC (pystac-client) sobre el AOI y la ventana temporal.
    - Filtrar escenas por nubosidad y leer Cloud-Optimized GeoTIFF por ventanas.
    - Construir cubos perezosos (stackstac + Dask) sin descarga completa.
    - Descargar capas Hansen GFC y de contexto (IGAC, RUNAP) hacia data/external.
"""
