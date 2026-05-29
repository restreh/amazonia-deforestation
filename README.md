# Detección temprana de deforestación en la Amazonía colombiana

Detección de deforestación a nivel de píxel sobre el arco amazónico colombiano
(Caquetá) mediante composiciones trimestrales de Sentinel-2 y aprendizaje
automático supervisado, con Hansen Global Forest Change como referencia de
entrenamiento y evaluación.

Proyecto Integrado 1 — Maestría en Ciencia de Datos y Analítica. Articula tres
materias: Aprendizaje Automático (modelos y evaluación), Almacenamiento y
Procesamiento de Grandes Datos (arquitectura e ingeniería de datos en AWS) y
Visualización de Datos (tablero Streamlit).

## Problema

El SMByC-IDEAM reporta alertas de deforestación con periodicidad trimestral. La
latencia limita la respuesta operativa frente a núcleos activos. Este proyecto
entrena un clasificador binario por píxel que detecta polígonos de deforestación
sobre un área de interés del Caquetá, como complemento académico de acceso
abierto con un ciclo de actualización más corto.

## Enfoque

- Clasificación binaria por píxel: 1 = pérdida de cobertura arbórea, 0 = permanencia.
- Baselines: Random Forest y XGBoost sobre 612 atributos (espectrales, índices
  NDVI/NBR/NDWI y contextuales con ventanas 3×3 y 5×5, incluyendo textura GLCM).
- Modelo de aprendizaje profundo: U-Net con encoder ResNet-34 entrenado desde
  cero sobre 56 canales (10 bandas mediana + 3 índices + 1 máscara de validez,
  ×4 trimestres), pérdida combinada focal + Dice y estandarización por canal.
- Modelo final: ensamble por promedio ponderado de las probabilidades de
  XGBoost y U-Net (pesos seleccionados sobre validación, 0.7 XGBoost y 0.3 U-Net).
- Tratamiento de la autocorrelación espacial: diagnóstico con I de Moran y
  semivariograma, partición espacial por bloques de 5 km y reporte en paralelo
  de validación aleatoria vs. espacial.
- Comparación estadística entre modelos: prueba de McNemar pareada, bootstrap
  espacial por bloques (B = 1.000) y coeficiente de concordancia de Lin.

## Estructura del repositorio

```
amazonia-deforestation/
├── config/        Configuración central (config.yaml)
├── data/          raw, interim, processed, external (no versionados)
├── scripts/       Pipeline reproducible por pasos
├── src/amazonia_deforestation/
│   ├── ingest/      STAC Earth-Search, lectura COG, ingestión de Hansen
│   ├── data/        Composiciones trimestrales, máscaras SCL, índices
│   ├── features/    Atributos contextuales y métricas de textura GLCM
│   ├── spatial/     I de Moran, semivariograma, partición por bloques
│   ├── models/      Random Forest, XGBoost, U-Net, dataset de patches
│   ├── evaluation/  Métricas pixel y polígono, calibración de umbral
│   └── viz/         Utilidades de visualización
├── infra/         Infraestructura como código (IAM, S3, Lambda)
└── dashboards/    Tablero Streamlit
```

## Datos

- **Sentinel-2 L2A** vía STAC Earth-Search (`s3://sentinel-cogs`, `us-west-2`).
  Política de datos libres de Copernicus.
- **Hansen Global Forest Change v1.12** (Universidad de Maryland, GLAD).
  Licencia Creative Commons Attribution 4.0.
- **Contexto**: límites administrativos del IGAC y áreas protegidas (RUNAP).

## Reproducción

```bash
# 1. Entorno
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Configurar parámetros de área, tiempo y AWS
#    Editar config/config.yaml (AOI, target_year, project_bucket)

# 3. Credenciales AWS (lectura de datos abiertos y escritura en el bucket del proyecto)
aws configure   # región us-west-2

# 4. Pipeline de ingestión y preparación (desde la raíz del repositorio)
python scripts/refine_aoi.py             # AOI a límites municipales
python scripts/select_aoi.py             # ventana de ~5.000 km² sobre el núcleo activo
python scripts/build_composites.py       # composiciones trimestrales Sentinel-2
python scripts/build_indices.py          # NDVI, NBR, NDWI
python scripts/align_label.py            # etiqueta Hansen a la grilla de 20 m
python scripts/run_diagnostics.py        # I de Moran y semivariograma
python scripts/build_split.py            # partición espacial por bloques
python scripts/build_training_sample.py  # muestreo balanceado de píxeles
python scripts/build_features.py         # tabla de 612 atributos por píxel

# 5. Modelado y predicción de baselines (CPU)
python scripts/train_baselines.py        # Random Forest y XGBoost, CV espacial
python scripts/predict.py                # predicción densa por píxel sobre val/test
python scripts/evaluate_baseline.py      # métricas píxel y polígono

# 6. Modelado y predicción del U-Net (GPU)
python scripts/compute_channel_stats.py  # media y desviación por canal sobre train
python scripts/train_unet.py
python scripts/predict_unet.py
python scripts/evaluate_baseline.py --model unet

# 7. Ensamble por promedio ponderado XGBoost + U-Net (CPU)
python scripts/build_ensemble.py         # 0.5/0.5 por defecto
python scripts/sweep_ensemble.py         # barrido de pesos y selección por F1_val
python scripts/evaluate_baseline.py --model ensemble

# 8. Comparación estadística entre modelos (CPU)
python scripts/mcnemar.py --models xgboost unet ensemble random_forest
python scripts/bootstrap_spatial.py --models xgboost unet ensemble random_forest
python scripts/concordance_lin.py --models xgboost unet ensemble random_forest
python scripts/compare_cv.py --models xgboost random_forest
```

## Infraestructura AWS

Cómputo dentro del AWS Free Tier vigente desde julio de 2025 (200 USD de crédito
durante seis meses) en la región `us-west-2`. Almacenamiento de derivados en S3
(Parquet particionado por tile MGRS, trimestre y bloque), consulta con Athena,
orquestación e inferencia en Lambda. La política IAM se encuentra en
`infra/iam/DeforestationProjectAccess.json`.

## Licencia

Código bajo licencia MIT (ver `LICENSE`). Los datos conservan las licencias de
sus fuentes originales.

## Equipo

Gia Mariana Calle Higuita · Juan Diego Llorente Ortega · Juan José Restrepo
Higuita · Manuela Caro Villada · Jerónimo Velásquez Escobar.
