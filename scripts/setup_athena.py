"""Crea la base de datos Glue y las tablas Athena del proyecto.

Uso (desde la raiz del repositorio, con credenciales AWS configuradas y los
artefactos subidos a S3 con upload_to_s3.py):
    python scripts/setup_athena.py
    python scripts/setup_athena.py --queries-only   # solo ejecuta las consultas demo
    python scripts/setup_athena.py --drop           # borra todo y vuelve a crear

Crea:
  - base de datos Glue: amazonia_deforestation
  - tabla externa metrics_by_block sobre derived/metrics_by_block/ (poblada)
  - tabla externa train_features sobre derived/features/ (parquet unico)
  - tabla externa features_by_block sobre derived/features_by_block/ (Hive-particionada
    por block_id, ~158 particiones)

Ejecuta tres consultas de ejemplo y guarda los resultados como CSV en
data/interim/athena_<query>.csv:
  q_metrics_by_model.sql        promedio de F1, IoU, precision, recall por modelo en test
  q_top_blocks_ensemble.sql     bloques con mayor F1 para el ensamble en test
  q_hectares_by_split.sql       hectareas predichas y de Hansen por split y modelo

Dependencias: boto3, pandas, pyyaml.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import boto3  # noqa: E402
import yaml  # noqa: E402

config = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
REGION = config["aws"]["region"]
BUCKET = config["aws"]["project_bucket"]
DATABASE = "amazonia_deforestation"
ATHENA_OUTPUT = f"s3://{BUCKET}/{config['aws']['prefixes']['athena_results']}"

athena = boto3.client("athena", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)


def run_query(sql, wait=True):
    """Lanza una consulta en Athena, espera y devuelve el QueryExecutionId."""
    res = athena.start_query_execution(
        QueryString=sql,
        ResultConfiguration={"OutputLocation": ATHENA_OUTPUT})
    qid = res["QueryExecutionId"]
    if not wait:
        return qid
    while True:
        s = athena.get_query_execution(QueryExecutionId=qid)["QueryExecution"]["Status"]
        if s["State"] in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
        time.sleep(0.5)
    if s["State"] != "SUCCEEDED":
        raise RuntimeError("Query " + qid + " " + s["State"]
                           + ": " + s.get("StateChangeReason", "unknown"))
    return qid


def fetch_results(qid):
    """Descarga el CSV de Athena para una consulta exitosa y devuelve filas como lista."""
    info = athena.get_query_execution(QueryExecutionId=qid)["QueryExecution"]
    s3_uri = info["ResultConfiguration"]["OutputLocation"]
    key = s3_uri[len("s3://" + BUCKET + "/"):]
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    text = obj["Body"].read().decode("utf-8")
    return list(csv.reader(io.StringIO(text)))


def create_database():
    print("Creando base " + DATABASE)
    run_query(f"CREATE DATABASE IF NOT EXISTS {DATABASE}")


def drop_all():
    print("Borrando tablas y base " + DATABASE)
    for t in ("metrics_by_block", "train_features", "features_by_block"):
        try:
            run_query(f"DROP TABLE IF EXISTS {DATABASE}.{t}")
        except Exception as e:
            print("  aviso al borrar " + t + ": " + str(e))
    try:
        run_query(f"DROP DATABASE IF EXISTS {DATABASE} CASCADE")
    except Exception as e:
        print("  aviso al borrar base: " + str(e))


def create_metrics_table():
    print("Creando tabla " + DATABASE + ".metrics_by_block")
    sql = f"""
    CREATE EXTERNAL TABLE IF NOT EXISTS {DATABASE}.metrics_by_block (
        block_id INT,
        split_code STRING,
        model STRING,
        threshold DOUBLE,
        n_pixels BIGINT,
        n_positives BIGINT,
        prevalence DOUBLE,
        tp BIGINT, fp BIGINT, fn BIGINT, tn BIGINT,
        precision DOUBLE, recall DOUBLE, f1 DOUBLE, iou DOUBLE,
        mean_proba DOUBLE,
        predicted_ha DOUBLE,
        truth_ha DOUBLE
    )
    STORED AS PARQUET
    LOCATION 's3://{BUCKET}/derived/metrics_by_block/'
    TBLPROPERTIES ('parquet.compress'='SNAPPY')
    """
    run_query(sql)


def create_features_table():
    """Tabla externa sobre train_features.parquet (sin particion)."""
    print("Creando tabla " + DATABASE + ".train_features")
    # Schema minimo: row, col y label. Athena lee parquet por nombre de columna,
    # asi que las 612 features estan disponibles via SELECT * aunque aqui solo
    # declaremos las claves.
    sql = f"""
    CREATE EXTERNAL TABLE IF NOT EXISTS {DATABASE}.train_features (
        row INT, col INT, label TINYINT
    )
    STORED AS PARQUET
    LOCATION 's3://{BUCKET}/derived/features/'
    TBLPROPERTIES ('parquet.compress'='SNAPPY')
    """
    run_query(sql)


def create_features_by_block_table():
    """Tabla externa Hive-particionada por block_id."""
    print("Creando tabla " + DATABASE + ".features_by_block")
    sql = f"""
    CREATE EXTERNAL TABLE IF NOT EXISTS {DATABASE}.features_by_block (
        row INT, col INT, label TINYINT
    )
    PARTITIONED BY (block_id BIGINT)
    STORED AS PARQUET
    LOCATION 's3://{BUCKET}/derived/features_by_block/'
    TBLPROPERTIES ('parquet.compress'='SNAPPY')
    """
    run_query(sql)
    print("  registrando particiones (MSCK REPAIR)")
    run_query(f"MSCK REPAIR TABLE {DATABASE}.features_by_block")


QUERIES = {
    "q_metrics_by_model": f"""
        SELECT model,
               AVG(f1) AS mean_f1,
               AVG(iou) AS mean_iou,
               AVG(precision) AS mean_precision,
               AVG(recall) AS mean_recall,
               SUM(tp) AS total_tp, SUM(fp) AS total_fp,
               SUM(fn) AS total_fn, SUM(tn) AS total_tn
        FROM {DATABASE}.metrics_by_block
        WHERE split_code = 'test'
        GROUP BY model
        ORDER BY mean_f1 DESC
    """,
    "q_top_blocks_ensemble": f"""
        SELECT block_id, f1, precision, recall, n_pixels, prevalence,
               predicted_ha, truth_ha
        FROM {DATABASE}.metrics_by_block
        WHERE model = 'ensemble' AND split_code = 'test'
        ORDER BY f1 DESC
        LIMIT 10
    """,
    "q_hectares_by_split": f"""
        SELECT split_code, model,
               SUM(predicted_ha) AS predicted_total_ha,
               SUM(truth_ha) AS truth_total_ha,
               CASE WHEN SUM(truth_ha) > 0
                    THEN SUM(predicted_ha) / SUM(truth_ha)
                    ELSE NULL END AS ratio_pred_truth
        FROM {DATABASE}.metrics_by_block
        WHERE split_code IN ('val', 'test')
        GROUP BY split_code, model
        ORDER BY split_code, model
    """,
    "q_features_partition_pruning": f"""
        SELECT block_id, COUNT(*) AS n_rows, SUM(label) AS n_positives,
               AVG(CAST(label AS DOUBLE)) AS prevalence
        FROM {DATABASE}.features_by_block
        WHERE block_id IN (100, 200, 300, 400)
        GROUP BY block_id
        ORDER BY block_id
    """,
}


def run_demos():
    out_dir = ROOT / "data" / "interim"
    sql_dir = ROOT / "infra" / "athena"
    sql_dir.mkdir(parents=True, exist_ok=True)
    for name, sql in QUERIES.items():
        print("\nConsulta " + name)
        (sql_dir / (name + ".sql")).write_text(sql.strip() + "\n", encoding="utf-8")
        qid = run_query(sql)
        rows = fetch_results(qid)
        for r in rows[:15]:
            print("  " + " | ".join(r))
        if len(rows) > 15:
            print("  ... (+" + str(len(rows) - 15) + " filas)")
        csv_path = out_dir / ("athena_" + name + ".csv")
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)
        print("  guardado en " + str(csv_path))


def main() -> None:
    ap = argparse.ArgumentParser(description="Setup de Glue/Athena para el proyecto")
    ap.add_argument("--queries-only", action="store_true",
                    help="omite la creacion de base y tablas; solo corre los demos")
    ap.add_argument("--drop", action="store_true",
                    help="borra base y tablas existentes antes de crear")
    args = ap.parse_args()

    if args.drop:
        drop_all()
    if not args.queries_only:
        create_database()
        create_metrics_table()
        create_features_table()
        create_features_by_block_table()
    run_demos()
    print("\nListo. Resultados en data/interim/athena_*.csv y consultas en infra/athena/")


if __name__ == "__main__":
    main()
