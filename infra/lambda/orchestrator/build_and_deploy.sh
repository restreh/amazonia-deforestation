#!/usr/bin/env bash
# Construye el ZIP del Lambda de orquestacion, lo crea o actualiza, y configura
# una regla EventBridge con cron trimestral que lo dispara.
#
# Uso (desde la raiz del repositorio, con AWS CLI v2 configurada y permisos):
#     bash infra/lambda/orchestrator/build_and_deploy.sh
#
# Variables opcionales: SCHEDULE_EXPRESSION (default: cron trimestral el dia 1).
set -euo pipefail

AWS_REGION="${AWS_REGION:-us-west-2}"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
BUCKET="${PROJECT_BUCKET:-amazonia-deforestation-data-363918845645}"
LAMBDA_NAME="amazonia-deforestation-orchestrator"
INFERENCE_FN="amazonia-deforestation-unet-inference"
ROLE_NAME="amazonia-deforestation-orchestrator-role"
RULE_NAME="amazonia-deforestation-quarterly"
# Cron trimestral: minuto 0, hora 0, dia 1, meses 1/4/7/10
SCHEDULE_EXPRESSION="${SCHEDULE_EXPRESSION:-cron(0 0 1 1,4,7,10 ? *)}"

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT

echo ">>> Empaquetando handler"
cp "$REPO_ROOT/infra/lambda/orchestrator/handler.py" "$BUILD_DIR/"
(cd "$BUILD_DIR" && zip -q -r function.zip handler.py)

echo ">>> Asegurando rol IAM para el orquestador"
TRUST='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1 || \
    aws iam create-role --role-name "$ROLE_NAME" --assume-role-policy-document "$TRUST" >/dev/null
aws iam attach-role-policy --role-name "$ROLE_NAME" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole >/dev/null 2>&1 || true
INVOKE_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": "lambda:InvokeFunction",
    "Resource": "arn:aws:lambda:$AWS_REGION:$ACCOUNT_ID:function:$INFERENCE_FN"
  }]
}
EOF
)
aws iam put-role-policy --role-name "$ROLE_NAME" \
    --policy-name InvokeInference --policy-document "$INVOKE_POLICY" >/dev/null
ROLE_ARN="arn:aws:iam::$ACCOUNT_ID:role/$ROLE_NAME"
sleep 8

echo ">>> Creando o actualizando Lambda $LAMBDA_NAME"
ENV_VARS="Variables={PROJECT_BUCKET=$BUCKET,INFERENCE_FUNCTION=$INFERENCE_FN,GRID_HEIGHT=3533,GRID_WIDTH=3556}"
if aws lambda get-function --function-name "$LAMBDA_NAME" --region "$AWS_REGION" >/dev/null 2>&1; then
    aws lambda update-function-code --function-name "$LAMBDA_NAME" \
        --zip-file "fileb://$BUILD_DIR/function.zip" --region "$AWS_REGION" --publish >/dev/null
    aws lambda wait function-updated --function-name "$LAMBDA_NAME" --region "$AWS_REGION"
    aws lambda update-function-configuration --function-name "$LAMBDA_NAME" \
        --runtime python3.12 --handler handler.lambda_handler \
        --memory-size 512 --timeout 60 \
        --environment "$ENV_VARS" --region "$AWS_REGION" >/dev/null
else
    aws lambda create-function --function-name "$LAMBDA_NAME" \
        --runtime python3.12 --handler handler.lambda_handler \
        --role "$ROLE_ARN" --memory-size 512 --timeout 60 \
        --environment "$ENV_VARS" \
        --zip-file "fileb://$BUILD_DIR/function.zip" --region "$AWS_REGION" >/dev/null
fi

LAMBDA_ARN="arn:aws:lambda:$AWS_REGION:$ACCOUNT_ID:function:$LAMBDA_NAME"

echo ">>> Creando regla EventBridge ($SCHEDULE_EXPRESSION)"
aws events put-rule --name "$RULE_NAME" \
    --schedule-expression "$SCHEDULE_EXPRESSION" \
    --description "Disparador trimestral del orquestador U-Net" \
    --region "$AWS_REGION" >/dev/null
aws events put-targets --rule "$RULE_NAME" \
    --targets "Id=1,Arn=$LAMBDA_ARN" \
    --region "$AWS_REGION" >/dev/null

echo ">>> Permiso para que EventBridge invoque la Lambda"
aws lambda add-permission --function-name "$LAMBDA_NAME" \
    --statement-id "EventBridgeInvoke" --action lambda:InvokeFunction \
    --principal events.amazonaws.com \
    --source-arn "arn:aws:events:$AWS_REGION:$ACCOUNT_ID:rule/$RULE_NAME" \
    --region "$AWS_REGION" >/dev/null 2>&1 || true

echo ">>> Listo. Orquestador: $LAMBDA_ARN"
echo "Regla EventBridge: $RULE_NAME ($SCHEDULE_EXPRESSION)"
echo "Para invocar manualmente:"
echo "  aws lambda invoke --function-name $LAMBDA_NAME --region $AWS_REGION --payload '{}' /tmp/out.json"
