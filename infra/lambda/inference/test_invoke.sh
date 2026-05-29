#!/usr/bin/env bash
# Invocacion de prueba del Lambda de inferencia sobre una ventana 256x256.
#
# Uso:
#     bash infra/lambda/inference/test_invoke.sh [row0 col0 size]
set -euo pipefail

AWS_REGION="${AWS_REGION:-us-west-2}"
LAMBDA_NAME="amazonia-deforestation-unet-inference"
BUCKET="amazonia-deforestation-data-363918845645"

ROW0="${1:-1024}"
COL0="${2:-1024}"
SIZE="${3:-256}"
ROW1=$((ROW0 + SIZE))
COL1=$((COL0 + SIZE))

EVENT_FILE="$(mktemp --suffix=.json)"
RESPONSE_FILE="$(mktemp --suffix=.json)"
trap 'rm -f "$EVENT_FILE" "$RESPONSE_FILE"' EXIT

cat > "$EVENT_FILE" <<EOF
{
  "model_uri": "s3://$BUCKET/models/unet.pt",
  "composites_prefix": "s3://$BUCKET/derived/composites/tile=AOI_caqueta/",
  "indices_prefix": "s3://$BUCKET/derived/indices/tile=AOI_caqueta/",
  "quarters": ["2024Q1", "2024Q2", "2024Q3", "2024Q4"],
  "bbox": [$ROW0, $COL0, $ROW1, $COL1],
  "output_uri": "s3://$BUCKET/inference/proba_window_${ROW0}_${COL0}.tif"
}
EOF

echo ">>> Invocando Lambda con ventana [$ROW0,$COL0,$ROW1,$COL1]"
# AWS CLI en Git Bash Windows necesita ruta nativa para fileb://
EVENT_NATIVE="$EVENT_FILE"
RESPONSE_NATIVE="$RESPONSE_FILE"
if command -v cygpath >/dev/null 2>&1; then
    EVENT_NATIVE="$(cygpath -w "$EVENT_FILE")"
    RESPONSE_NATIVE="$(cygpath -w "$RESPONSE_FILE")"
fi
aws lambda invoke --function-name "$LAMBDA_NAME" --region "$AWS_REGION" \
    --cli-binary-format raw-in-base64-out \
    --payload "fileb://$EVENT_NATIVE" "$RESPONSE_NATIVE" >/dev/null

cat "$RESPONSE_FILE" | python -c "import json,sys; print(json.dumps(json.load(sys.stdin), indent=2))"
