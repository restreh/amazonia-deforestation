# Imagen Lambda con PyTorch CPU para inferencia U-Net sobre ventanas de S3.
# Tamano esperado: ~1.5 GB (Lambda admite hasta 10 GB de imagen).
FROM public.ecr.aws/lambda/python:3.12

# GDAL y dependencias de sistema para rasterio
RUN dnf install -y gcc gcc-c++ make && \
    pip install --no-cache-dir --upgrade pip && \
    dnf clean all

WORKDIR ${LAMBDA_TASK_ROOT}

# Dependencias Python primero (cachea capas si no cambian)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Codigo del proyecto que el handler necesita (subset de src/)
COPY src/amazonia_deforestation/__init__.py src/amazonia_deforestation/__init__.py
COPY src/amazonia_deforestation/models/__init__.py src/amazonia_deforestation/models/__init__.py
COPY src/amazonia_deforestation/models/unet.py src/amazonia_deforestation/models/unet.py
COPY src/amazonia_deforestation/models/patches.py src/amazonia_deforestation/models/patches.py

# Handler de Lambda
COPY handler.py .

CMD ["handler.lambda_handler"]
