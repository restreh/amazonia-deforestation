# Detección temprana de deforestación en la Amazonía colombiana

Detección de deforestación a nivel de píxel sobre el arco amazónico colombiano
(Caquetá) mediante composiciones trimestrales de Sentinel-2 y aprendizaje
automático supervisado, con Hansen Global Forest Change como referencia de
entrenamiento y evaluación.

Proyecto Integrado 1 — Maestría en Ciencia de Datos y Analítica. Articula tres
materias: Aprendizaje Automático (modelos y evaluación), Almacenamiento y
Procesamiento de Grandes Datos (arquitectura e ingeniería de datos en AWS) y
Visualización de Datos (tableros Streamlit y Tableau).

## Problema

El SMByC-IDEAM reporta alertas de deforestación con periodicidad trimestral. La
latencia limita la respuesta operativa frente a núcleos activos. Este proyecto
entrena un clasificador binario por píxel que detecta polígonos de deforestación
sobre un área de interés del Caquetá, como complemento académico de acceso
abierto con un ciclo de actualización más corto.

## Enfoque

- Clasificación binaria por píxel: 1 = pérdida de cobertura arbórea, 0 = permanencia.
- Baselines: Random Forest y XGBoost sobre atributos espectrales, índices y
  atributos contextuales (ventanas 3×3 y 5×5, textura GLCM).
- Modelo principal: U-Net con encoder ResNet-34 preentrenado, pérdida focal para
  el desbalance severo de clases.
- Tratamiento de la autocorrelación espacial: diagnóstico con I de Moran y
  semivariograma, partición espacial por bloques y reporte en paralelo de
  validación aleatoria vs. espacial.

## Estructura del repositorio

```
amazonia-deforestation/
├── config/        Configuración central (config.yaml)
├── data/          raw, interim, processed, external (no versionados)
├── scripts/       Pipeline reproducible por pasos (ingesta, features, modelado, evaluación)
├── notebooks/     EDA y figuras para el informe (opcional)
├── src/amazonia_deforestation/
│   ├── ingest/      STAC Earth-Search, lectura COG, cubos con stackstac/Dask
│   ├── data/        Composiciones trimestrales, máscaras SCL, índices
│   ├── features/    Atributos contextuales y métricas de textura GLCM
│   ├── spatial/     I de Moran, semivariograma, partición por bloques
│   ├── models/      Random Forest, XGBoost, U-Net, pérdida focal
│   ├── evaluation/  Métricas pixel/polígono, McNemar, bootstrap espacial
│   └── viz/         Utilidades de visualización
├── infra/         Infraestructura como código (IAM, S3, Lambda)
├── dashboards/    Tablero Streamlit y tablero ejecutivo Tableau
├── docs/          Documento final e informes
└── tests/         Pruebas
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

# 4. Ejecutar el pipeline por pasos (desde la raíz del repositorio):
python scripts/refine_aoi.py            # AOI a límites municipales
python scripts/select_aoi.py            # ventana de ~5.000 km² sobre el núcleo de deforestación
python scripts/build_composites.py      # composiciones trimestrales Sentinel-2
python scripts/build_indices.py         # NDVI, NBR, NDWI
python scripts/align_label.py           # etiqueta Hansen a la grilla de 20 m
python scripts/run_diagnostics.py       # I de Moran y semivariograma
python scripts/build_split.py           # partición espacial por bloques
python scripts/build_training_sample.py # muestreo balanceado de píxeles
python scripts/build_features.py        # tabla de atributos por píxel
python scripts/train_baselines.py       # Random Forest y XGBoost (CV espacial)
python scripts/predict.py               # predicción por píxel sobre val/test
python scripts/evaluate_baseline.py     # métricas de píxel y de polígono
```

## Infraestructura AWS

Cómputo dentro del AWS Free Tier (región `us-west-2`) y fine-tuning del U-Net en
GPU. Almacenamiento de derivados en S3 (Parquet particionado por tile MGRS,
trimestre y bloque), consulta con Athena, orquestación e inferencia en Lambda.
Ver `infra/`.

## Licencia

Código bajo licencia MIT (ver `LICENSE`). Los datos conservan las licencias de
sus fuentes originales.

## Equipo

Gia Mariana Calle Higuita · Juan Diego Llorente Ortega · Juan José Restrepo
Higuita · Manuela Caro Villada · Jerónimo Velásquez Escobar.
