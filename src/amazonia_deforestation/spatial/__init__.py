"""Diagnostico y control de la dependencia espacial.

Responsabilidades:
    - Indice I de Moran global y local (esda, libpysal) sobre bandas, indices,
      etiquetas y residuos del baseline.
    - Semivariograma empirico para estimar el rango espacial.
    - Particion espacial por bloques (BlockKFold / leave-one-block-out) en
      proporciones 70-15-15.
    - Regression-kriging sobre residuos del baseline (pykrige).
"""
