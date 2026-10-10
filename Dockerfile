# Imagen base ligera basada en Python 3.10
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instalar paquetes requeridos por OpenCV headless y el subsistema de cámara de Raspberry Pi OS
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# 1. Instalar versión PyTorch CPU ligera (~180MB en lugar de varios GBs)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple

# 2. Instalar el resto de dependencias
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Se especifica api.main:app en lugar de main:app
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]