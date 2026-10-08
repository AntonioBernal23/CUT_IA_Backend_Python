# Imagen base ligera
FROM python:3.10-slim

# Evita que Python escriba archivos .pyc y fuerza logs en vivo
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Dependencias mínimas del sistema para OpenCV sin GUI
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# 1. Instalar primero PyTorch / torchvision versión CPU (~180 MB en vez de ~2.5 GB)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2. Instalar el resto de dependencias desde requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]