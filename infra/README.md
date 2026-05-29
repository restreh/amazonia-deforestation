# Infraestructura del proyecto

Capa de AWS del proyecto `amazonia-deforestation`. Articula el almacenamiento en S3,
las consultas analíticas en Athena (sobre Glue), la inferencia en Lambda con un
contenedor PyTorch y el benchmark del criterio de éxito de tiempo de inferencia
sobre una instancia EC2 `t3.medium`. Todo opera en `us-west-2` para evitar
costos de transferencia entre regiones (los buckets públicos de Sentinel-2 viven
ahí).

## Componentes

```
infra/
├── iam/
│   └── DeforestationProjectAccess.json    Política de proyecto, adjunta al grupo data-science-team
├── s3/                                    Estructura del bucket (placeholder)
├── athena/                                SQL de las consultas demo (lo crea setup_athena.py)
├── lambda/inference/
│   ├── Dockerfile                         Imagen Lambda con PyTorch CPU
│   ├── handler.py                         Inferencia U-Net sobre una ventana via VSI/S3
│   ├── requirements.txt
│   ├── build_and_deploy.sh                Build, ECR push, create/update Lambda function
│   └── test_invoke.sh                     Invocación de prueba sobre una ventana 256×256
└── ec2/
    ├── run_benchmark.sh                   Lanza una t3.medium, mide y termina la instancia
    └── user_data.sh                       Bootstrap que corre en la instancia
```

## Recursos en AWS

- **Cuenta:** `363918845645`.
- **Región:** `us-west-2`.
- **Bucket del proyecto:** `s3://amazonia-deforestation-data-363918845645/`.
- **Grupo IAM:** `data-science-team` (con la política `DeforestationProjectAccess`).
- **Glue database:** `amazonia_deforestation` (creada por `scripts/setup_athena.py`).
- **Lambda function:** `amazonia-deforestation-unet-inference` (creada por
  `infra/lambda/inference/build_and_deploy.sh`).
- **ECR repository:** `amazonia-deforestation-inference`.

## Convenciones de S3

El bucket sigue la convención de prefijos del `config.yaml` y particiona los
derivados al estilo Hive por tile MGRS, trimestre y agregación. Como el AOI cubre un
solo tile MGRS, se usa el placeholder `tile=AOI_caqueta`.

```
s3://amazonia-deforestation-data-363918845645/
├── derived/
│   ├── composites/tile=AOI_caqueta/quarter=2024Q{1-4}/aggregation={median,p25}/composite.tif
│   ├── indices/tile=AOI_caqueta/quarter=2024Q{1-4}/indices.tif
│   ├── interim/{label_2024_20m,split_blocks}.tif, channel_stats.json
│   ├── features/train_features.parquet, train_sample.parquet
│   ├── features_by_block/block_id=*/...                  # opcional, vía CTAS en Athena
│   ├── metrics_by_block/part.parquet                     # tabla analítica para Athena
│   └── predictions/model={xgboost,random_forest,unet,ensemble,unet_imagenet}/proba.tif
├── models/{xgboost.json, random_forest.joblib, unet.pt, unet_imagenet.pt}
├── metrics/                                              # eval/mcnemar/bootstrap/concordance JSONs
├── inference/                                            # rasters de salida del Lambda
├── benchmarks/                                           # JSON y log del benchmark t3.medium
└── athena-results/                                       # query results de Athena
```

## Flujo extremo a extremo

```bash
# 1. Subida de derivados a S3
python scripts/upload_to_s3.py

# 2. Agregar la tabla analítica por bloque y reenviar
python scripts/build_metrics_by_block.py
python scripts/upload_to_s3.py --only metrics_by_block

# 3. Crear base Glue, tablas Athena y correr las consultas demo
python scripts/setup_athena.py
#    Opcional: particionar features por bloque con CTAS dentro de Athena
python scripts/setup_athena.py --ctas-features-by-block

# 4. Construir e implementar el Lambda de inferencia
bash infra/lambda/inference/build_and_deploy.sh

# 5. Invocación de prueba sobre una ventana 256×256
bash infra/lambda/inference/test_invoke.sh

# 6. Benchmark del criterio de éxito sobre EC2 t3.medium
bash infra/ec2/run_benchmark.sh
```

## Costos

- **S3.** ~5.4 GB almacenados. ~$0.12 / mes en `us-west-2` (Standard).
- **Athena.** Consultas demo escanean <1 MB de `metrics_by_block`. ~$0 (el costo es
  $5 / TB escaneados).
- **Lambda.** Función con imagen ~1.5 GB. Sin costo si no se invoca; cada invocación
  cuesta ~$0.0001 por la combinación memoria + tiempo (4 GB × 5 s aprox.).
- **EC2 t3.medium.** ~$0.04 / hora en `us-west-2`. El benchmark dura menos de una
  hora; costo total esperado < $0.10.

Total del despliegue demo: aproximadamente $0.25 / mes en estado estable y < $1 por
ejecución completa del flujo. Se mantiene dentro del crédito inicial de $200 del
Free Tier vigente.

## Limpieza

Para liberar todos los recursos creados (útil al cerrar el ciclo académico):

```bash
# Lambda y ECR
aws lambda delete-function --function-name amazonia-deforestation-unet-inference --region us-west-2
aws ecr delete-repository --repository-name amazonia-deforestation-inference --force --region us-west-2

# Glue/Athena
python scripts/setup_athena.py --drop

# S3 (¡borra todo el contenido del bucket!)
aws s3 rm s3://amazonia-deforestation-data-363918845645/ --recursive

# Roles IAM creados por los scripts
aws iam delete-role --role-name amazonia-deforestation-lambda-role
aws iam delete-role --role-name amazonia-deforestation-ec2-benchmark
aws iam delete-instance-profile --instance-profile-name amazonia-deforestation-ec2-benchmark
```
