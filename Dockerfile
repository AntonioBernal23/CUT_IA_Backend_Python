# Imagen base
FROM python:3.10-slim

# Evita problemas con logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instalar dependencias del sistema (IMPORTANTE para OpenCV)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Crear carpeta de trabajo
WORKDIR /app

# Copiar requirements primero (cache eficiente)
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el proyecto
COPY . .

# Exponer puerto de FastAPI
EXPOSE 8000

# Comando para iniciar
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]