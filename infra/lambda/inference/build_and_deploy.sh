#!/usr/bin/env bash
# Construye la imagen del Lambda de inferencia, la sube a ECR y crea/actualiza
# la funcion Lambda con configuracion de memoria y timeout adecuados.
#
# Uso (desde la raiz del repositorio, con Docker corriendo y AWS CLI v2 configurada):
#     bash infra/lambda/inference/build_and_deploy.sh
#
# Requisitos: Docker, AWS CLI v2, permisos para ECR y Lambda en la cuenta.
set -euo pipefail

AWS_REGION="${AWS_REGION:-us-west-2}"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
ECR_REPO="amazonia-deforestation-inference"
IMAGE_TAG="latest"
LAMBDA_NAME="amazonia-deforestation-unet-inference"
LAMBDA_ROLE_NAME="amazonia-deforestation-lambda-role"

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT

echo ">>> Preparando contexto de build en $BUILD_DIR"
cp "$REPO_ROOT/infra/lambda/inference/Dockerfile" "$BUILD_DIR/Dockerfile"
cp "$REPO_ROOT/infra/lambda/inference/requirements.txt" "$BUILD_DIR/requirements.txt"
cp "$REPO_ROOT/infra/lambda/inference/handler.py" "$BUILD_DIR/handler.py"
mkdir -p "$BUILD_DIR/src"
cp -r "$REPO_ROOT/src/amazonia_deforestation" "$BUILD_DIR/src/"

echo ">>> Login a ECR"
aws ecr describe-repositories --repository-names "$ECR_REPO" --region "$AWS_REGION" >/dev/null 2>&1 \
    || aws ecr create-repository --repository-name "$ECR_REPO" --region "$AWS_REGION" >/dev/null
aws ecr get-login-password --region "$AWS_REGION" \
    | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

echo ">>> Construyendo imagen Docker"
# Lambda requiere manifest Docker v2; desactivamos provenance y SBOM que generan
# atestaciones OCI no soportadas por el runtime de Lambda.
docker build --platform linux/amd64 --provenance=false --sbom=false \
    -t "$ECR_REPO:$IMAGE_TAG" "$BUILD_DIR"

IMAGE_URI="$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:$IMAGE_TAG"
docker tag "$ECR_REPO:$IMAGE_TAG" "$IMAGE_URI"

echo ">>> Push a ECR"
docker push "$IMAGE_URI"

echo ">>> Asegurando rol IAM para Lambda"
TRUST_POLICY='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
aws iam get-role --role-name "$LAMBDA_ROLE_NAME" >/dev/null 2>&1 || \
    aws iam create-role --role-name "$LAMBDA_ROLE_NAME" \
        --assume-role-policy-document "$TRUST_POLICY" >/dev/null
aws iam attach-role-policy --role-name "$LAMBDA_ROLE_NAME" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole >/dev/null 2>&1 || true
aws iam attach-role-policy --role-name "$LAMBDA_ROLE_NAME" \
    --policy-arn "arn:aws:iam::$ACCOUNT_ID:policy/DeforestationProjectAccess" >/dev/null 2>&1 || true
ROLE_ARN="arn:aws:iam::$ACCOUNT_ID:role/$LAMBDA_ROLE_NAME"
sleep 8   # propagacion IAM

echo ">>> Creando o actualizando funcion Lambda $LAMBDA_NAME"
if aws lambda get-function --function-name "$LAMBDA_NAME" --region "$AWS_REGION" >/dev/null 2>&1; then
    aws lambda update-function-code --function-name "$LAMBDA_NAME" \
        --image-uri "$IMAGE_URI" --region "$AWS_REGION" --publish >/dev/null
    aws lambda wait function-updated --function-name "$LAMBDA_NAME" --region "$AWS_REGION"
    aws lambda update-function-configuration --function-name "$LAMBDA_NAME" \
        --memory-size 3008 --timeout 300 --region "$AWS_REGION" >/dev/null
else
    aws lambda create-function --function-name "$LAMBDA_NAME" \
        --package-type Image --code "ImageUri=$IMAGE_URI" \
        --role "$ROLE_ARN" --memory-size 3008 --timeout 300 \
        --region "$AWS_REGION" >/dev/null
fi

echo ">>> Listo. Image URI: $IMAGE_URI"
echo "Para invocar: bash infra/lambda/inference/test_invoke.sh"
