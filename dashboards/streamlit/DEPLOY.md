# Despliegue del tablero en Streamlit Community Cloud

Este tablero soporta dos modos de lectura, controlados por la variable
de entorno `AMAZONIA_DATA_SOURCE`.

| Valor | Fuente | Caso de uso |
|---|---|---|
| `local` (default) | `data/...` en el repositorio clonado | desarrollo local, jurado con artefactos en mano |
| `s3` | `s3://amazonia-deforestation-data-363918845645/...` anónimo | Streamlit Cloud y cualquier host sin AWS configurado |

Para Streamlit Community Cloud usamos el modo `s3`. Los prefijos `derived/`,
`models/` y `metrics/` del bucket están publicados como lectura anónima y los
componentes de `rasterio`, `fsspec` y `s3fs` los leen sin credenciales.

## Pasos exactos

1. **Cuenta**. Crear cuenta en https://share.streamlit.io con el GitHub que tenga
   acceso de lectura al repo. Si el repo es público, cualquier cuenta sirve.

2. **Nueva app**. Botón `Create app` → `Use existing repo`.
   - Repository: `restreh/amazonia-deforestation`
   - Branch: `main`
   - Main file path: `dashboards/streamlit/streamlit_app.py`
   - App URL: el que quieras (queda `https://<algo>.streamlit.app`).

3. **Advanced settings**.
   - Python version: `3.12`.
   - Python dependencies file: `dashboards/streamlit/requirements.txt`
     (el `requirements.txt` de la raíz instala torch, dask, etc. que el tablero
     no necesita y haría que el deploy tarde mucho).
   - Secrets:
     ```toml
     AMAZONIA_DATA_SOURCE = "s3"
     AWS_REGION = "us-west-2"
     AWS_NO_SIGN_REQUEST = "YES"
     ```
     Los secrets se exponen al runtime como variables de entorno.

4. **Deploy**. Botón `Deploy`. La primera build tarda 5–8 minutos por las
   wheels de rasterio y geopandas.

5. **Verificación post-deploy**. Visitar la URL pública. La página de inicio
   debe cargar en 5–10 s; la primera entrada a "Mapa de predicciones" tarda
   30–60 s en bajar el raster de 40 MB de S3 y queda cacheado en la sesión.

## Costos

- **Streamlit Cloud**: gratis en el tier Community (1 GB RAM, 1 CPU, dormida
  tras 7 días de inactividad).
- **S3 egress**: el bucket está en `us-west-2`, Streamlit Cloud está en GCP, así
  que el tráfico cuenta como egress a internet ($0.09/GB). Por sesión, ~0.3 GB
  para que el tablero renderice todas las páginas. 10 sesiones de jurado salen
  por ~$0.30. Documentado en `infra/README.md`.

## Limitaciones del modo S3

- **Tamaño de los rasters**: los `proba_*.tif` pesan 40 MB cada uno. En la
  primera lectura, rasterio baja el archivo completo desde S3 antes de aplicar
  el `downsampling`. Si esto fuera un sistema de producción, conviene
  re-encodearlos como COG con overviews para que rasterio pueda leer solo el
  rango necesario.
- **Memoria**: Streamlit Cloud Community tiene ~1 GB de RAM. El tablero no
  pasa de 600–700 MB en estado estable con todos los caches calientes; está
  ajustado pero pasa.
- **El reloj del free tier duerme la app tras 7 días sin tráfico**. La primera
  visita después de la siesta tarda 30–60 s extra en arrancar el contenedor.

## Cambiar entre fuentes localmente

```powershell
# Local (default)
streamlit run dashboards/streamlit/streamlit_app.py

# S3 anónimo
$env:AMAZONIA_DATA_SOURCE = "s3"
$env:AWS_NO_SIGN_REQUEST = "YES"
$env:AWS_REGION = "us-west-2"
streamlit run dashboards/streamlit/streamlit_app.py
```

```bash
# Bash equivalente
AMAZONIA_DATA_SOURCE=s3 AWS_NO_SIGN_REQUEST=YES AWS_REGION=us-west-2 \
  streamlit run dashboards/streamlit/streamlit_app.py
```

## Política del bucket aplicada

`derived/`, `models/` y `metrics/` son lectura pública. El resto del bucket
queda privado. La política está en `infra/s3/public_read_policy.json` para
referencia futura.
