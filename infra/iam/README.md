# Acceso IAM para el equipo

Procedimiento de acceso a la cuenta AWS del proyecto. La administración la hace
el propietario con la cuenta **root**. Los compañeros acceden mediante **IAM
Users agrupados**, con permisos de mínimo privilegio dados por la política
`DeforestationProjectAccess.json`.

## Convención de nombres

| Recurso         | Nombre                          |
|-----------------|---------------------------------|
| Grupo equipo    | `data-science-team`             |
| Política equipo | `DeforestationProjectAccess`    |
| Usuarios        | `firstname.lastname` (minúscula) |

Usuarios: `gia.calle`, `juandiego.llorente`, `juanjose.restrepo`,
`manuela.caro`, `jeronimo.velasquez`.

## Estado

El grupo `data-science-team` y los usuarios ya están creados. Falta crear la
política y adjuntarla al grupo.

## Crear y adjuntar la política

Ejecutar como root, desde esta carpeta (`infra/iam`):

```powershell
# 1. Crear la política
aws iam create-policy --policy-name DeforestationProjectAccess `
  --policy-document file://DeforestationProjectAccess.json

# 2. Account id
$acct = aws sts get-caller-identity --query Account --output text

# 3. Adjuntar al grupo
aws iam attach-group-policy --group-name data-science-team `
  --policy-arn "arn:aws:iam::${acct}:policy/DeforestationProjectAccess"
```

## Acceso de los compañeros

Cada usuario tiene acceso a consola con contraseña temporal. Para CLI o
notebooks, cada uno genera sus propias access keys tras iniciar sesión
(Security credentials → Create access key). El equipo decidió no exigir MFA.

## Notas de seguridad

- root tiene control total y no se restringe con políticas; cuidar esas
  credenciales con especial atención.
- Configurar alertas de presupuesto en AWS Budgets para vigilar el Free Tier.
- Al cerrar el proyecto, eliminar las access keys y desactivar los usuarios.
