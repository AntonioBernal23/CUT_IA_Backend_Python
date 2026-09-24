from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response, StreamingResponse
from io import BytesIO
from pathlib import Path
import numpy as np
import cv2

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

# URLs derivadas
SPRING_DETECTIONS_URL = f"{SPRING_URL}/detections"
SPRING_BEHAVIOR_URL   = f"{SPRING_URL}/behavior" 

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

# =========================
# DEDUPLICACIÓN POR CAMBIO DE ESTADO
# =========================
FRAME_SKIP = 5
CONFIRMATIONS_NEEDED = 3  # frames consecutivos para confirmar un cambio

_last_behavior: dict[int, str]         = {}  # {track_id: ultimo_label_confirmado}
_last_seen:     dict[int, float]        = {}  # {track_id: timestamp ultima vez visto}
_pending:       dict[int, tuple[str, int]] = {}  # {track_id: (label_candidato, contador)}
_state_lock = threading.Lock()

def should_send(track_id: int, label: str) -> bool:
    """Envía solo si el comportamiento cambió Y se confirmó N frames consecutivos."""
    now = time.time()
    with _state_lock:
        _last_seen[track_id] = now

        pending_label, count = _pending.get(track_id, (label, 0))

        if pending_label == label:
            count += 1
        else:
            pending_label = label
            count = 1

        _pending[track_id] = (pending_label, count)

        if count >= CONFIRMATIONS_NEEDED:
            ultimo = _last_behavior.get(track_id)
            if ultimo != label:
                _last_behavior[track_id] = label
                return True

        return False

def limpiar_tracks_viejos(max_age_segundos: int = 120):
    now = time.time()
    with _state_lock:
        expirados = [
            tid for tid, ts in _last_seen.items()
            if now - ts > max_age_segundos
        ]
        for tid in expirados:
            _last_behavior.pop(tid, None)
            _last_seen.pop(tid, None)
            _pending.pop(tid, None)


# =========================
# CAMERA WORKER
# =========================
def camera_worker():
    global latest_frame, latest_annotated, running

    print("Worker iniciado...")

    send_errors = 0
    MAX_ERRORS = 5
    frame_count = 0
    last_cleanup = time.time()
    last_stats_send = time.time()

    SPRING_BEHAVIOR_URL = f"{SPRING_URL}/behavior"

    while running:
        print("Conectando a cámara...")
        cap = cv2.VideoCapture(get_stream_source(STREAM_URL))

        if not cap.isOpened():
            print("No se pudo abrir el stream, reintentando en 2s...")
            time.sleep(2)
            continue

        print("Conectado a la cámara")

        while running:
            ret, frame = cap.read()

            if not ret or frame is None:
                print("Frame perdido / stream caído, reconectando...")
                break

            frame_count += 1

            # --- THROTTLE: actualiza stream pero salta YOLO ---
            if frame_count % FRAME_SKIP != 0:
                with lock:
                    latest_frame = frame.copy()
                    if latest_annotated is None:
                        latest_annotated = frame.copy()
                frame_event.set()
                time.sleep(0.01)
                continue

            # --- YOLO TRACKING ---
            out = yolo.detect_image(frame, return_crops=True)
            annotated = out.get("annotated", frame)

            with lock:
                latest_frame = frame.copy()
                latest_annotated = annotated.copy()
            frame_event.set()

            crops = out.get("crops", [])

            if not crops:
                print("Sin detecciones en este frame.")
                time.sleep(0.01)
                continue

            # --- ESTADÍSTICAS AGREGADAS ---
            if time.time() - last_stats_send >= SEND_INTERVAL:
                counts = {}
                for c in crops:
                    counts[c["label"]] = counts.get(c["label"], 0) + 1

                try:
                    r = requests.post(
                    SPRING_DETECTIONS_URL,
                    json={"counts": counts},
                    headers={"X-Internal-Key": INTERNAL_API_KEY},
                    timeout=2,
                    )
                    r.raise_for_status()
                    print(f"[STATS] {counts} → {r.status_code}")
                except requests.exceptions.RequestException as e:
                    print(f"[STATS ERROR] {e}")
                last_stats_send = time.time()

            # --- DEDUPLICACIÓN + ENVÍO ---
            for crop in crops:
                track_id = crop.get("track_id", -1)
                label    = crop["label"]

                if not should_send(track_id, label):
                    print(f"[SKIP] track_id={track_id} | {label} (sin cambio o pendiente confirmación)")
                    continue

                payload = {
                    "timestamp": time.time(),
                    "track_id":  track_id,
                    "label":     label,
                    "conf":      crop["conf"],
                    "image_b64": crop["image_b64"],
                }

                try:
                    response = requests.post(
                        SPRING_BEHAVIOR_URL,
                        json=payload,
                        headers={"X-Internal-Key": INTERNAL_API_KEY},
                        timeout=2
                    )
                    response.raise_for_status()
                    print(f"[CAMBIO] track_id={track_id} | {label} ({crop['conf']:.2f}) → {response.status_code}")
                    send_errors = 0

                except requests.exceptions.Timeout:
                    send_errors += 1
                    print(f"[TIMEOUT {send_errors}/{MAX_ERRORS}] Spring no respondió")

                except requests.exceptions.ConnectionError:
                    send_errors += 1
                    print(f"[CONN ERROR {send_errors}/{MAX_ERRORS}] No se pudo conectar a Spring")

                except Exception as e:
                    send_errors += 1
                    print(f"[ERROR {send_errors}/{MAX_ERRORS}] {e}")

                finally:
                    if send_errors >= MAX_ERRORS:
                        print("Demasiados errores consecutivos, deteniendo worker...")
                        cap.release()
                        running = False
                        return

            # --- LIMPIEZA PERIÓDICA DE TRACKS ---
            if time.time() - last_cleanup > 60:
                limpiar_tracks_viejos()
                last_cleanup = time.time()
                print("[CLEANUP] Tracks viejos eliminados")

            time.sleep(0.01)

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