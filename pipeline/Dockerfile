# Imagen del pipeline batch de Spark (Tarea A, sección 7 del informe).
# Se construye desde la raíz del repositorio:
#   docker build -t limon-spark:1.0 -f pipeline/Dockerfile .
# La imagen NO lleva fotos, modelo ni código: todo eso se monta con -v al correrla.
FROM python:3.11.18-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYSPARK_PYTHON=python \
    PYSPARK_DRIVER_PYTHON=python \
    GIT_PYTHON_REFRESH=quiet

# Spark 4.0 necesita Java 17 o 21 (con el Java 23 de la PC no arranca)
RUN apt-get update \
 && apt-get install -y --no-install-recommends openjdk-17-jre-headless procps \
 && rm -rf /var/lib/apt/lists/*

# Dependencias con versiones exactas; PyTorch CPU desde download.pytorch.org (sin CUDA)
COPY pipeline/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

WORKDIR /repo
CMD ["python", "pipeline/job_batch.py", "--help"]
