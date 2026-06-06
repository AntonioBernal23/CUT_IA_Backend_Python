from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response, StreamingResponse
from io import BytesIO
from pathlib import Path
import numpy as np
import cv2

import json

import os
from dotenv import load_dotenv

import socket

import threading
import time
import requests
from contextlib import asynccontextmanager

from app_core.yolo_runtime import YOLOModel
from app_core.stats import count_by_class

from fastapi import FastAPI, UploadFile, File, Security, HTTPException
from fastapi.security.api_key import APIKeyHeader

#Variables de entorno
load_dotenv("../.env")

# =========================
# CONFIG
# =========================
STREAM_URL = os.getenv("STREAM_URL") # cambia a tu cámara
SPRING_URL = os.getenv("SPRING_URL")
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY")
SEND_INTERVAL = 5  # segundos

# =========================
# ESTADO GLOBAL
# =========================
latest_frame = None
latest_annotated = None
lock = threading.Lock()
frame_event = threading.Event()

worker_started = False
running = True

# =========================
# MODELO YOLO
# =========================

import logging
logging.getLogger("ultralytics").setLevel(logging.WARNING)

yolo = YOLOModel(model_path="models/best.pt", device="cpu", conf=0.25)

api_key_header = APIKeyHeader(name="X-Internal-Key", auto_error=False)

def verify_api_key(key: str = Security(api_key_header)):
    if key != INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="API Key inválida")

# =====================
# Obtener ip de camara
# =====================
def get_stream_source(stream_value: str | None):
    if not stream_value:
        return 0  # default a webcam

    stream_value = stream_value.strip()

    # Si es número → webcam
    if stream_value.isdigit():
        return int(stream_value)

    # Si es URL → stream IP
    if stream_value.startswith(("http://", "https://", "rtsp://")):
        return stream_value

    # Cualquier otra cosa → intenta usarlo tal cual
    return stream_value

# =========================
# WORKER (cámara + YOLO + envío)
# =========================
def camera_worker():
    global latest_frame, latest_annotated, running

    print("Worker iniciado...")

    while running:
        print("Conectando a cámara...")
        cap = cv2.VideoCapture(get_stream_source(STREAM_URL))

        # Intento de apertura
        if not cap.isOpened():
            print("No se pudo abrir el stream, reintentando en 2s...")
            time.sleep(2)
            continue

        print("Conectado a la cámara")

        last_send = time.time()
        acumulado = {}

        while running:
            ret, frame = cap.read()

            # Si falla lectura, salimos para reconectar
            if not ret or frame is None:
                print("Frame perdido / stream caído, reconectando...")
                break

            # --- YOLO (una sola vez) ---
            out = yolo.detect_image(frame)
            annotated = out.get("annotated", frame)

            # Guardar frames (thread-safe)
            with lock:
                latest_frame = frame.copy()
                latest_annotated = annotated.copy()
            frame_event.set()

            # --- Stats ---
            dist = count_by_class(out.get("boxes"), out.get("names"))
            for k, v in dist.items():
                acumulado[k] = acumulado.get(k, 0) + v

            # --- Envío periódico ---
            if time.time() - last_send >= SEND_INTERVAL:
                if acumulado:
                    try:
                        response = requests.post(
                            SPRING_URL,
                            json={"counts": acumulado}, 
                            headers={"X-Internal-Key": INTERNAL_API_KEY},
                            timeout=2)
                        response.raise_for_status()
                        print(f"Enviado a Spring: {acumulado} → status: {response.status_code}")

                    except Exception as e:
                        print(f"Error enviando a Spring: {e}")
                        running = False
                        os._exit(1)
                else:
                    print("Sin detecciones en este intervalo, no se envía.")

                acumulado = {}
                last_send = time.time()

            # Pequeña pausa para no saturar CPU
            time.sleep(0.01)

        # Liberar y reintentar conexión
        cap.release()
        print("Reconectando en 1s...")
        time.sleep(1)

#Obtener ip
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # no envía nada realmente
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip

# =========================
# LIFESPAN (inicio automático)
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global worker_started

    # =========================
    # WORKER
    # =========================
    if not worker_started:
        print("Iniciando worker UNA sola vez...")
        thread = threading.Thread(target=camera_worker, daemon=True)
        thread.start()
        worker_started = True

    yield

    print("Cerrando aplicación...")

# =========================
# APP FASTAPI
# =========================
app = FastAPI(
    title="CUT IA Behavior API",
    version="0.1.0",
    lifespan=lifespan
)

# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# ESTÁTICOS
# =========================
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

FAVICON_PATH = STATIC_DIR / "favicon.ico"

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    if FAVICON_PATH.exists():
        return FileResponse(str(FAVICON_PATH), media_type="image/x-icon")
    return Response(content=b"", media_type="image/x-icon")

# =========================
# ENDPOINTS
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/labels")
def labels():
    id2label = yolo.get_names()
    return {str(k): v for k, v in id2label.items()}

# =========================
# PREDICT (imagen individual)
# =========================
@app.post("/predict/image")
async def predict_image(
    file: UploadFile = File(...),
    return_image: bool = False,
    conf: float = 0.25,
):
    yolo.set_confidence(conf)

    data = await file.read()
    buf = np.frombuffer(data, np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    if bgr is None:
        return {"count": 0, "boxes": [], "distribution": {}}

    out = yolo.detect_image(bgr)

    if return_image and out.get("annotated") is not None:
        ok, png = cv2.imencode(".png", out["annotated"])
        if not ok:
            return {"error": "No se pudo codificar la imagen"}

        return StreamingResponse(
            BytesIO(png.tobytes()),
            media_type="image/png"
        )

    boxes_json = []
    boxes = out.get("boxes")
    names = out.get("names", {})

    if boxes is not None:
        for x1, y1, x2, y2, conf_v, cls in boxes:
            label = names.get(int(cls), str(int(cls)))
            boxes_json.append({
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "conf": float(conf_v),
                "cls": int(cls),
                "label": label
            })

    dist = count_by_class(boxes, names)
    return {"count": len(boxes_json), "boxes": boxes_json, "distribution": dist}

# =========================
# STREAM (bajo demanda)
# =========================
@app.get("/stream")
def stream(key: str = Security(api_key_header)):
    verify_api_key(key)
    def generate():
        while True:
            frame_event.wait(timeout=1.0)
            frame_event.clear()
            with lock:
                if latest_annotated is None:
                    continue
                frame = latest_annotated.copy()
            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                jpeg.tobytes() +
                b"\r\n"
            )
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")