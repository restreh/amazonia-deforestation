# Detección temprana de deforestación en la Amazonía colombiana

Detección de deforestación a nivel de píxel sobre el arco amazónico colombiano
(Caquetá) mediante composiciones trimestrales de Sentinel-2 y aprendizaje
automático supervisado, con Hansen Global Forest Change como referencia de
entrenamiento y evaluación.

Proyecto Integrador 1 — Maestría en Ciencia de Datos y Analítica. Articula tres
materias: Aprendizaje Automático (modelos y evaluación), Almacenamiento y
Procesamiento de Grandes Datos (arquitectura e ingeniería de datos en AWS) y
Visualización de Datos (tablero Streamlit).

**Tablero público**: https://amazonia-deforestation.streamlit.app/

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
│   └── evaluation/  Métricas pixel y polígono, calibración de umbral
├── infra/         Infraestructura AWS (IAM, S3, Athena, Lambda, EC2)
└── dashboards/    Tablero Streamlit (entrypoint y páginas)
```

## Datos

- **Sentinel-2 L2A** vía STAC Earth-Search (`s3://sentinel-cogs`, `us-west-2`).
  Política de datos libres de Copernicus.
- **Hansen Global Forest Change v1.12** (Universidad de Maryland, GLAD).
  Licencia Creative Commons Attribution 4.0.
- **Contexto**: límites administrativos del IGAC.

## Reproducción del pipeline completo

```bash
# 1. Entorno
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Configurar parámetros de área, tiempo y AWS
#    Editar config/config.yaml (AOI, target_year, project_bucket)

# 3. Credenciales AWS (lectura de datos abiertos y escritura en el bucket del proyecto)
aws configure   # región us-west-2

# 4. Ingestión y preparación
python scripts/refine_aoi.py             # AOI a límites municipales
python scripts/select_aoi.py             # ventana de ~5.000 km² sobre el núcleo activo
python scripts/check_availability.py     # escenas Sentinel-2 por trimestre
python scripts/download_labels.py        # capas Hansen GFC para el bbox
python scripts/build_composites.py       # composiciones trimestrales Sentinel-2
python scripts/build_indices.py          # NDVI, NBR, NDWI
python scripts/align_label.py            # etiqueta Hansen a la grilla de 20 m
python scripts/run_diagnostics.py        # I de Moran y semivariograma
python scripts/build_split.py            # partición espacial por bloques
python scripts/build_training_sample.py  # muestreo balanceado de píxeles
python scripts/build_features.py         # tabla de 612 atributos por píxel

# 5. Baselines (CPU)
python scripts/train_baselines.py        # Random Forest y XGBoost, CV espacial
python scripts/predict.py                # predicción densa por píxel sobre val/test
python scripts/predict.py --model random_forest
python scripts/evaluate_baseline.py
python scripts/evaluate_baseline.py --model random_forest

# 6. U-Net (GPU; se corre en máquina aparte)
python scripts/compute_channel_stats.py  # media y desviación por canal sobre train
python scripts/train_unet.py
python scripts/predict_unet.py
python scripts/evaluate_baseline.py --model unet

# 7. Ensamble (CPU)
python scripts/build_ensemble.py
python scripts/sweep_ensemble.py --weights 0.3 0.4 0.5 0.6 0.7
python scripts/evaluate_baseline.py --model ensemble

# 8. Comparación estadística (CPU)
python scripts/mcnemar.py --models xgboost unet ensemble random_forest
python scripts/bootstrap_spatial.py --models xgboost unet ensemble random_forest
python scripts/concordance_lin.py --models xgboost unet ensemble random_forest
python scripts/compare_cv.py --models xgboost random_forest

# 9. Big Data en AWS (S3 + Glue/Athena + Lambda + EC2)
python scripts/upload_to_s3.py                              # ~5,4 GB con partición Hive
python scripts/build_metrics_by_block.py
python scripts/upload_to_s3.py --only metrics_by_block
python scripts/build_features_by_block.py
python scripts/upload_to_s3.py --only features_by_block
python scripts/setup_athena.py                              # base Glue, tablas, consultas demo
bash infra/lambda/inference/build_and_deploy.sh             # contenedor U-Net en Lambda
bash infra/lambda/inference/test_invoke.sh
bash infra/lambda/orchestrator/build_and_deploy.sh          # orquestador + cron trimestral
bash infra/ec2/run_benchmark.sh                             # benchmark t3.medium

# 10. Tablero
streamlit run dashboards/streamlit/streamlit_app.py         # modo local
```

## Tablero Streamlit

El tablero tiene seis páginas: resumen ejecutivo, contexto y datos, diagnóstico
espacial, modelos y métricas (con bootstrap espacial, McNemar y CCC),
mapa de predicciones con capas conmutables, análisis por municipio y
despliegue Big Data.

Soporta dos fuentes de datos según la variable de entorno
`AMAZONIA_DATA_SOURCE`.

| Valor | Fuente | Caso de uso |
|---|---|---|
| `local` (por defecto) | `data/...` en el repo | Desarrollo y revisión con artefactos en mano |
| `s3` | `s3://amazonia-deforestation-data-363918845645/...` anónimo | Streamlit Cloud y cualquier despliegue sin AWS configurado |

Los prefijos `derived/`, `models/` y `metrics/` del bucket son lectura pública;
el resto queda privado.

### Despliegue en Streamlit Community Cloud

1. Cuenta en https://share.streamlit.io con un GitHub que tenga acceso al repo.
2. `Create app` → `Use existing repo`.
   - Repository: `restreh/amazonia-deforestation`.
   - Branch: `main`.
   - Main file path: `dashboards/streamlit/streamlit_app.py`.
3. `Advanced settings`.
   - Python version: `3.12`.
   - Python dependencies file: `dashboards/streamlit/requirements.txt`.
   - Secrets.
     ```toml
     AMAZONIA_DATA_SOURCE = "s3"
     AWS_REGION = "us-west-2"
     AWS_NO_SIGN_REQUEST = "YES"
     ```
4. `Deploy`. La primera build tarda 5-8 minutos por `rasterio` y `geopandas`.

Costo: Streamlit Community es gratis (1 GB RAM, dormida tras 7 días de
inactividad). El egress de S3 hacia GCP cuesta 0,09 USD/GB; una sesión completa
del tablero transfiere ~0,3 GB.

### Correr el tablero localmente apuntando a S3

```bash
AMAZONIA_DATA_SOURCE=s3 AWS_NO_SIGN_REQUEST=YES AWS_REGION=us-west-2 \
  streamlit run dashboards/streamlit/streamlit_app.py
```

## Infraestructura AWS

Cómputo dentro del AWS Free Tier vigente desde julio de 2025 (200 USD de crédito
durante seis meses) en la región `us-west-2`. Almacenamiento de derivados en S3
con particionamiento Hive por tile MGRS, trimestre y agregación. Tabla analítica
`metrics_by_block` en Apache Parquet, consultable desde Athena vía Glue.
Inferencia servida en un contenedor Lambda con PyTorch CPU y orquestación
trimestral con EventBridge. Benchmark del criterio de éxito sobre EC2
`t3.medium`. La política IAM del grupo `data-science-team` está en
`infra/iam/DeforestationProjectAccess.json`, la política de bucket público para
los prefijos del tablero está en `infra/s3/public_read_policy.json` y los
scripts de despliegue en `infra/lambda/{inference,orchestrator}/` y `infra/ec2/`.
Flujo extremo a extremo, costos y procedimiento de limpieza en `infra/README.md`.

## Licencia

Código bajo licencia MIT (ver `LICENSE`). Los datos conservan las licencias de
sus fuentes originales.

## Equipo

Gia Mariana Calle Higuita · Juan Diego Llorente Ortega · Juan José Restrepo
Higuita · Manuela Caro Villada · Jerónimo Velásquez Escobar.
