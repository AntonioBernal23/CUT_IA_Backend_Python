
# Imagen base: Docker selecciona automáticamente amd64 o arm64
FROM debian:bookworm-slim

ARG TARGETARCH

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Evita instalaciones automáticas de Ultralytics en runtime
ENV YOLO_AUTOINSTALL=false
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics

# 1. Dependencias base de Debian
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gnupg ca-certificates \
    python3 python3-venv python3-pip \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Picamera2 y libcamera únicamente en ARM64
RUN if [ "$TARGETARCH" = "arm64" ]; then \
        echo "Instalando dependencias de Raspberry Pi (ARM64)"; \
        curl -fsSL https://archive.raspberrypi.com/debian/raspberrypi.gpg.key \
            | gpg --dearmor -o /usr/share/keyrings/raspberrypi.gpg; \
        echo "deb [arch=arm64 signed-by=/usr/share/keyrings/raspberrypi.gpg] http://archive.raspberrypi.com/debian/ bookworm main" \
            > /etc/apt/sources.list.d/raspi.list; \
        apt-get update; \
        apt-get install -y --no-install-recommends python3-picamera2; \
    else \
        echo "Arquitectura AMD64: se omite Picamera2"; \
    fi \
    && rm -rf /var/lib/apt/lists/*

# 3. Entorno virtual que permite acceder a paquetes instalados por apt
RUN python3 -m venv --system-site-packages /venv

ENV PATH="/venv/bin:$PATH"

WORKDIR /app

# 4. Evitar que NumPy suba a 2.x
RUN echo "numpy<2" > /constraints.txt

COPY requirements.txt .

# 5. PyTorch CPU
RUN pip install --no-cache-dir -c /constraints.txt torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple

# 6. Resto de dependencias
RUN pip install --no-cache-dir -c /constraints.txt \
    -r requirements.txt lapx onnxruntime

# 7. Código del backend
COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
