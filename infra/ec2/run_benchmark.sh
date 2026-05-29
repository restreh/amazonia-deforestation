#!/usr/bin/env bash
# Provisiona una instancia EC2 t3.medium en us-west-2, corre el benchmark de
# inferencia U-Net sobre todo el AOI, escribe el tiempo en S3 y termina la instancia.
# Cumple el criterio de exito de la propuesta (< 10 min en t3.medium).
#
# Uso (desde la raiz del repositorio, con AWS CLI v2 configurada y permisos EC2):
#     bash infra/ec2/run_benchmark.sh
#
# Lee infra/ec2/user_data.sh, lo embebe como user-data, lanza la instancia, espera el
# resultado (publicado por la propia instancia en s3://bucket/benchmarks/) y termina.
set -euo pipefail

AWS_REGION="${AWS_REGION:-us-west-2}"
INSTANCE_TYPE="t3.medium"
BUCKET="amazonia-deforestation-data-363918845645"
ROLE_NAME="amazonia-deforestation-ec2-benchmark"
PROFILE_NAME="$ROLE_NAME"
KEY_NAME="${EC2_KEY_NAME:-}"          # opcional; sin SSH funcionamos por SSM/logs
RESULT_PREFIX="benchmarks/"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
USER_DATA="$REPO_ROOT/infra/ec2/user_data.sh"

echo ">>> AMI Amazon Linux 2023 mas reciente"
AMI_ID="$(aws ec2 describe-images --region "$AWS_REGION" \
    --owners amazon \
    --filters "Name=name,Values=al2023-ami-2023.*-x86_64" "Name=state,Values=available" \
    --query 'Images | sort_by(@,&CreationDate)[-1].ImageId' --output text)"
echo "AMI: $AMI_ID"

echo ">>> Asegurando rol IAM para la instancia"
TRUST='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1 || \
    aws iam create-role --role-name "$ROLE_NAME" --assume-role-policy-document "$TRUST" >/dev/null
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
aws iam attach-role-policy --role-name "$ROLE_NAME" \
    --policy-arn "arn:aws:iam::$ACCOUNT_ID:policy/DeforestationProjectAccess" >/dev/null 2>&1 || true
aws iam attach-role-policy --role-name "$ROLE_NAME" \
    --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore >/dev/null 2>&1 || true
aws iam get-instance-profile --instance-profile-name "$PROFILE_NAME" >/dev/null 2>&1 || \
    aws iam create-instance-profile --instance-profile-name "$PROFILE_NAME" >/dev/null
aws iam add-role-to-instance-profile --instance-profile-name "$PROFILE_NAME" \
    --role-name "$ROLE_NAME" >/dev/null 2>&1 || true
sleep 8   # propagacion IAM

echo ">>> Lanzando $INSTANCE_TYPE con user_data"
EXTRA_ARGS=()
if [[ -n "$KEY_NAME" ]]; then
    EXTRA_ARGS+=(--key-name "$KEY_NAME")
fi
# Leemos el script y lo pasamos como string para evitar el problema de file:// con
# rutas Git Bash en Windows. AWS CLI hace el base64-encode automaticamente.
USER_DATA_CONTENT="$(cat "$USER_DATA")"
INSTANCE_ID="$(aws ec2 run-instances --region "$AWS_REGION" \
    --image-id "$AMI_ID" --instance-type "$INSTANCE_TYPE" --count 1 \
    --iam-instance-profile "Name=$PROFILE_NAME" \
    --instance-initiated-shutdown-behavior terminate \
    --user-data "$USER_DATA_CONTENT" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=amazonia-benchmark-$TIMESTAMP},{Key=Project,Value=amazonia-deforestation}]" \
    "${EXTRA_ARGS[@]}" \
    --query 'Instances[0].InstanceId' --output text)"
echo "Instancia: $INSTANCE_ID"

echo ">>> Esperando publicacion del resultado en s3://$BUCKET/$RESULT_PREFIX..."
# El user_data calcula su propio TIMESTAMP al bootear, asi que detectamos el primer
# JSON nuevo bajo benchmarks/ usando una marca de inicio del polleo. Sale al primer
# archivo nuevo o al cumplir 60 minutos.
START_EPOCH="$(date -u +%s)"
RESULT_KEY=""
for i in $(seq 1 60); do
    CANDIDATE="$(aws s3api list-objects-v2 --bucket "$BUCKET" --prefix "$RESULT_PREFIX" \
        --query "Contents[?LastModified>='$(date -u -d @"$START_EPOCH" '+%Y-%m-%dT%H:%M:%S' 2>/dev/null || \
        date -u -r "$START_EPOCH" '+%Y-%m-%dT%H:%M:%S')'] | [?ends_with(Key, '.json')] | [0].Key" \
        --output text 2>/dev/null || true)"
    if [[ -n "$CANDIDATE" && "$CANDIDATE" != "None" ]]; then
        RESULT_KEY="$CANDIDATE"
        echo ">>> Resultado disponible tras $i min: $RESULT_KEY"
        aws s3 cp "s3://$BUCKET/$RESULT_KEY" - 2>/dev/null
        break
    fi
    sleep 60
done

echo ">>> Listo. Si la instancia no se autoaplico shutdown la termino manualmente."
aws ec2 terminate-instances --region "$AWS_REGION" --instance-ids "$INSTANCE_ID" >/dev/null || true
