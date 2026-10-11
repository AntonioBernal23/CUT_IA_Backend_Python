# Imagen base Debian Bookworm (misma base que Raspberry Pi OS Bookworm), arm64
FROM debian:bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 1. Repositorio oficial de Raspberry Pi (ahí vive python3-picamera2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gnupg ca-certificates \
    && curl -fsSL https://archive.raspberrypi.com/debian/raspberrypi.gpg.key \
       | gpg --dearmor -o /usr/share/keyrings/raspberrypi.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/raspberrypi.gpg] http://archive.raspberrypi.com/debian/ bookworm main" \
       > /etc/apt/sources.list.d/raspi.list \
    && rm -rf /var/lib/apt/lists/*

# 2. Picamera2 + libcamera por apt, y librerías que necesita OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip \
    python3-picamera2 \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. Venv que VE los paquetes de apt (picamera2, libcamera)
RUN python3 -m venv --system-site-packages /venv
ENV PATH="/venv/bin:$PATH"

WORKDIR /app

COPY requirements.txt .

# 4. PyTorch CPU ligero
RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple

# 5. Resto de dependencias. numpy<2 evita romper picamera2 (compilado con numpy 1.x)
RUN pip install --no-cache-dir "numpy<2" -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]