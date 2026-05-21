# Infraestructura AWS

Infraestructura como código del proyecto. Todo se ejecuta en una máquina con
salida de red hacia AWS (tu equipo o la nube), no en el sandbox del asistente.

## Orden de operaciones

La administración se hace con la cuenta root. El grupo `data-science-team` y los
usuarios ya están creados.

1. **Prerrequisitos (una vez):**
   - Instalar AWS CLI v2.
   - `aws configure` con las credenciales root, región `us-west-2`, salida `json`.

2. **Configuración:** ejecutar `aws/setup_aws.ps1`. Crea la política
   `DeforestationProjectAccess` y la adjunta al grupo `data-science-team`, crea
   el bucket S3 y un presupuesto con alerta. Es idempotente.

   ```powershell
   cd infra\aws
   pwsh -File .\setup_aws.ps1
   # Windows PowerShell: powershell -ExecutionPolicy Bypass -File .\setup_aws.ps1
   ```

3. **Tras ejecutar:** copiar el nombre del bucket al `config/config.yaml`
   (`aws.project_bucket`).

## Contenido

- `iam/DeforestationProjectAccess.json` — política de mínimo privilegio del equipo.
- `iam/README.md` — procedimiento de acceso para el equipo.
- `aws/setup_aws.ps1` — script de configuración inicial.
- `s3/`, `lambda/` — artefactos de almacenamiento e inferencia (siguientes fases).

## Nota sobre credenciales del CLI

La administración usa root. Para el pipeline y los notebooks, cada usuario del
grupo `data-science-team` puede generar sus propias access keys, con alcance
limitado por la política del grupo.
