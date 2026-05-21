"""Diagnóstico y control de la dependencia espacial.

Responsabilidades:
    - Índice I de Moran global y local (esda, libpysal) sobre bandas, índices,
      etiquetas y residuos del baseline.
    - Semivariograma empírico para estimar el rango espacial.
    - Partición espacial por bloques (BlockKFold / leave-one-block-out) en
      proporciones 70-15-15.
    - Regression-kriging sobre residuos del baseline (pykrige).
"""
