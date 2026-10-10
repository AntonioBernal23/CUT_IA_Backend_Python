import os
import time
import logging
import asyncio
import threading
from pathlib import Path
from contextlib import asynccontextmanager

import cv2
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Security, HTTPException
from fastapi.security import APIKeyHeader
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from picamera2 import Picamera2

from app_core.yolo_runtime import YOLOModel

# Silenciar logs verbosos de Ultralytics
logging.getLogger("ultralytics").setLevel(logging.WARNING)
logger = logging.getLogger("main")

load_dotenv()

# Configuración
SPRING_URL = os.getenv("SPRING_URL", "http://localhost:8080")
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "")
SPRING_DETECTIONS_URL = f"{SPRING_URL}/detections"
SPRING_BEHAVIOR_URL = f"{SPRING_URL}/behavior"

FRAME_SKIP = 3
CONFIRMATIONS_NEEDED = 3
SEND_INTERVAL = 5

# Estado global
latest_annotated: bytes = b""
lock = threading.Lock()
frame_event = threading.Event()
running = True

# Tracking & Deduplicación
_last_behavior: dict[int, str] = {}
_last_seen: dict[int, float] = {}
_pending: dict[int, tuple[str, int]] = {}
_state_lock = threading.Lock()

yolo = YOLOModel(model_path="models/best.onnx", conf=0.25)
api_key_header = APIKeyHeader(name="X-Internal-Key", auto_error=False)


def verify_api_key(key: str = Security(api_key_header)):
    if key != INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="API Key inválida")


def should_send(track_id: int, label: str) -> bool:
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
            if _last_behavior.get(track_id) != label:
                _last_behavior[track_id] = label
                return True
        return False


def limpiar_tracks_viejos(max_age_segundos: int = 120):
    now = time.time()
    with _state_lock:
        expirados = [tid for tid, ts in _last_seen.items() if now - ts > max_age_segundos]
        for tid in expirados:
            _last_behavior.pop(tid, None)
            _last_seen.pop(tid, None)
            _pending.pop(tid, None)


def camera_worker():
    global latest_annotated, running
    
    print("Iniciando trabajador de cámara...")
    
    try:
        picam2 = Picamera2()
        config = picam2.create_video_configuration(main={"size": (640, 480), "format": "BGR888"})
        picam2.configure(config)
        picam2.start()
        time.sleep(1.0)
    except Exception as e:
        logger.error(f"Error crítico al inicializar Picamera2: {e}")
        return

    frame_count = 0
    last_cleanup = time.time()
    last_stats_send = time.time()

    # Cliente HTTP persistente para reutilizar conexiones TCP
    with httpx.Client(timeout=2.0, headers={"X-Internal-Key": INTERNAL_API_KEY}) as client:
        while running:
            try:
                frame = picam2.capture_array()
            except Exception as e:
                logger.error(f"Error capturando frame: {e}")
                time.sleep(0.1)
                continue

            frame_count += 1
            
            # Codificar y guardar rápido para stream público
            if frame_count % FRAME_SKIP != 0:
                _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                with lock:
                    latest_annotated = jpeg.tobytes()
                frame_event.set()
                continue

            # Inferencia YOLO
            out = yolo.detect_image(frame, return_crops=True)
            annotated_frame = out.get("annotated", frame)
            crops = out.get("crops", [])

            _, jpeg = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
            with lock:
                latest_annotated = jpeg.tobytes()
            frame_event.set()

            now = time.time()

            # Envío agrupado de estadísticas
            if now - last_stats_send >= SEND_INTERVAL and crops:
                counts = {}
                for c in crops:
                    counts[c["label"]] = counts.get(c["label"], 0) + 1
                try:
                    client.post(SPRING_DETECTIONS_URL, json={"counts": counts})
                except Exception as e:
                    logger.warning(f"[STATS ERROR] {e}")
                last_stats_send = now

            # Deduplicación y envío por comportamientos
            for crop in crops:
                track_id = crop.get("track_id", -1)
                label = crop["label"]

                if should_send(track_id, label):
                    payload = {
                        "timestamp": now,
                        "track_id": track_id,
                        "label": label,
                        "conf": crop["conf"],
                        "image_b64": crop["image_b64"],
                    }
                    try:
                        client.post(SPRING_BEHAVIOR_URL, json=payload)
                    except Exception as e:
                        logger.warning(f"[SEND ERROR] track_id={track_id} | {e}")

            if now - last_cleanup > 60:
                limpiar_tracks_viejos()
                last_cleanup = now

    try:
        picam2.stop()
        picam2.close()
    except Exception:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    thread = threading.Thread(target=camera_worker, daemon=True)
    thread.start()
    yield
    global running
    running = False


app = FastAPI(title="CUT IA Behavior API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/labels")
def labels():
    return {str(k): v for k, v in yolo.get_names().items()}


@app.get("/stream")
def stream(key: str = Security(api_key_header)):
    verify_api_key(key)

    def generate():
        while running:
            if frame_event.wait(timeout=1.0):
                frame_event.clear()
                with lock:
                    buf = latest_annotated
                if buf:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + buf + b"\r\n"
                    )

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")