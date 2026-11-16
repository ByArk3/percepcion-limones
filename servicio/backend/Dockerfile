# Backend: API FastAPI con la ResNet18 (PyTorch CPU). Sin TensorFlow.
# Se construye desde la raíz del repositorio (lo hace docker compose):
#   docker compose build backend
# El modelo NO va dentro de la imagen: se monta desde modelo/ (el .pt viene de la Release modelo-v1).
FROM python:3.11.18-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /servicio
COPY servicio/backend/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY servicio/backend/app/ ./app/
# misma clase Clasificador que usa Spark y mismas tablas que crea la base
COPY pipeline/inferencia.py ./app/inferencia.py
COPY servicio/db/init.sql ./app/init.sql

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)" || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
