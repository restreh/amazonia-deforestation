<#
  setup_aws.ps1
  Configuracion de AWS para el proyecto amazonia-deforestation.
  Ejecutar como root. Crea: politica IAM (y la adjunta al grupo
  data-science-team ya existente), bucket S3 y un presupuesto con alerta.

  El grupo data-science-team y los usuarios se crean aparte (ya hechos).

  Prerrequisitos:
    1. AWS CLI v2 instalado.
    2. aws configure   (credenciales root, region us-west-2, output json)

  Uso (desde infra\aws):
    pwsh -File .\setup_aws.ps1
    # Windows PowerShell: powershell -ExecutionPolicy Bypass -File .\setup_aws.ps1

  Idempotente: si un recurso ya existe, lo informa y continua.
#>

$ErrorActionPreference = "Stop"

# --- Parametros ---
$Region      = "us-west-2"
$PolicyName  = "DeforestationProjectAccess"
$TeamGroup   = "data-science-team"
$AdminEmail  = "juanjose.restrepo.higuita@gmail.com"
$BudgetLimit = 40   # USD mensuales para la alerta

$PolicyFile  = Join-Path $PSScriptRoot "..\iam\DeforestationProjectAccess.json"

function Test-AwsResource {
    param([scriptblock]$Check)
    try { & $Check 2>$null | Out-Null; return $true } catch { return $false }
}

Write-Host "== Verificando identidad ==" -ForegroundColor Cyan
$AccountId = (aws sts get-caller-identity --query Account --output text)
if (-not $AccountId) { throw "No hay credenciales validas. Corre 'aws configure' primero." }
Write-Host "Cuenta AWS: $AccountId"
$Bucket = "amazonia-deforestation-data-$AccountId"

# --- 1. Politica IAM y adjuncion al grupo ---
Write-Host "`n== Politica IAM: $PolicyName ==" -ForegroundColor Cyan
$PolicyArn = "arn:aws:iam::${AccountId}:policy/$PolicyName"
if (Test-AwsResource { aws iam get-policy --policy-arn $PolicyArn }) {
    Write-Host "Ya existe."
} else {
    aws iam create-policy --policy-name $PolicyName `
        --policy-document "file://$PolicyFile" | Out-Null
    Write-Host "Creada."
}
aws iam attach-group-policy --group-name $TeamGroup --policy-arn $PolicyArn | Out-Null
Write-Host "Adjuntada al grupo $TeamGroup."

# --- 2. Bucket S3 ---
Write-Host "`n== Bucket S3: $Bucket ==" -ForegroundColor Cyan
if (Test-AwsResource { aws s3api head-bucket --bucket $Bucket }) {
    Write-Host "Ya existe."
} else {
    aws s3api create-bucket --bucket $Bucket --region $Region `
        --create-bucket-configuration LocationConstraint=$Region | Out-Null
    aws s3api put-public-access-block --bucket $Bucket `
        --public-access-block-configuration `
        "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true" | Out-Null
    Write-Host "Creado, con acceso publico bloqueado."
}

# --- 3. Presupuesto con alerta ---
Write-Host "`n== Presupuesto AWS Budgets ($BudgetLimit USD/mes) ==" -ForegroundColor Cyan
$budgetJson = @"
{ "BudgetName": "amazonia-deforestation-monthly",
  "BudgetLimit": { "Amount": "$BudgetLimit", "Unit": "USD" },
  "TimeUnit": "MONTHLY", "BudgetType": "COST" }
"@
$notifJson = @"
[ { "Notification": { "NotificationType": "ACTUAL", "ComparisonOperator": "GREATER_THAN", "Threshold": 80 },
    "Subscribers": [ { "SubscriptionType": "EMAIL", "Address": "$AdminEmail" } ] } ]
"@
$bf = Join-Path $env:TEMP "budget.json"; $nf = Join-Path $env:TEMP "notif.json"
$budgetJson | Out-File $bf -Encoding ascii; $notifJson | Out-File $nf -Encoding ascii
try {
    aws budgets create-budget --account-id $AccountId `
        --budget "file://$bf" --notifications-with-subscribers "file://$nf" 2>$null | Out-Null
    Write-Host "Presupuesto creado; alerta al 80% a $AdminEmail."
} catch {
    Write-Host "El presupuesto ya existia o requiere revision manual." -ForegroundColor Yellow
}

Write-Host "`n== Listo ==" -ForegroundColor Green
Write-Host "Bucket: $Bucket"
Write-Host "Actualiza config/config.yaml -> aws.project_bucket con ese nombre."
