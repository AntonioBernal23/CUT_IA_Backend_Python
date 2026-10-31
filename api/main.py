import os
import time
import queue
import random
import secrets
import logging
import threading
from collections import deque
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Security, HTTPException, Depends, Query
from fastapi.security import APIKeyHeader
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# Importación condicional: en PC o dentro de Docker no existe picamera2
try:
    from picamera2 import Picamera2
    HAS_PICAM2 = True
except ImportError:
    Picamera2 = None
    HAS_PICAM2 = False

from app_core.yolo_runtime import YOLOModel

load_dotenv()

# ──────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-7s | %(name)-9s | %(message)s",
    datefmt="%H:%M:%S",
)
for noisy in ("ultralytics", "httpx", "httpcore", "uvicorn.access"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

log = logging.getLogger("main")
log_cam = logging.getLogger("camara")
log_inf = logging.getLogger("inferenc")
log_spr = logging.getLogger("spring")
log_stm = logging.getLogger("stream")

# ──────────────────────────────────────────────────────────────
# Configuración
# ──────────────────────────────────────────────────────────────
SPRING_URL = os.getenv("SPRING_URL", "http://localhost:8080").rstrip("/")
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "")
SPRING_DETECTIONS_URL = f"{SPRING_URL}/detections"
SPRING_BEHAVIOR_URL = f"{SPRING_URL}/behavior"
# Ruta del ImageController de Spring (ajústala en el .env si la renombras)
SPRING_IMAGES_URL = os.getenv("SPRING_IMAGES_URL", f"{SPRING_URL}/api/dataset").rstrip("/")

MODEL_PATH = os.getenv("MODEL_PATH", "models/best.onnx")
MODEL_CONF = float(os.getenv("MODEL_CONF", "0.25"))

FRAME_W = int(os.getenv("FRAME_W", "640"))
FRAME_H = int(os.getenv("FRAME_H", "480"))

# Formato de Picamera2. Si los colores salen invertidos (rojo/azul), pon RGB888 en el .env
PICAM_FORMAT = os.getenv("PICAM_FORMAT", "BGR888")

# Máximo de inferencias por segundo (limita CPU y temperatura de la Pi)
INFER_FPS = float(os.getenv("INFER_FPS", "10"))
INFER_INTERVAL = 1.0 / max(INFER_FPS, 0.1)

CONFIRMATIONS_NEEDED = 3
SEND_INTERVAL = 5            # seg entre envíos de estadísticas
SPRING_QUEUE_MAX = 50        # cola de envíos pendientes
SPRING_TIMEOUT = 3.0
SPRING_UPLOAD_TIMEOUT = 10.0 # subir una imagen tarda más que un JSON
SPRING_PROBE_EVERY = 30      # seg: chequeo de conexión cuando no hay tráfico
STATS_LOG_EVERY = 60         # seg: resumen periódico en logs
JPEG_QUALITY_STREAM = 65

# Muestreo de imágenes crudas para el dataset (antes de que YOLO las procese)
SAMPLE_ENABLED = os.getenv("SAMPLE_ENABLED", "true").lower() == "true"
SAMPLE_MIN_S = float(os.getenv("SAMPLE_MIN_S", "30"))    # intervalo mínimo entre muestras
SAMPLE_MAX_S = float(os.getenv("SAMPLE_MAX_S", "120"))   # intervalo máximo entre muestras
JPEG_QUALITY_SAMPLE = int(os.getenv("JPEG_QUALITY_SAMPLE", "90"))

CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]

api_key_header = APIKeyHeader(name="X-Internal-Key", auto_error=False)


# ──────────────────────────────────────────────────────────────
# Seguridad
# ──────────────────────────────────────────────────────────────
def _keys_match(provided: Optional[str]) -> bool:
    if not INTERNAL_API_KEY or not provided:
        return False
    return secrets.compare_digest(provided.encode(), INTERNAL_API_KEY.encode())


def verify_stream_key(
    header_key: Optional[str] = Security(api_key_header),
    key: Optional[str] = Query(None, description="Alternativa al header para <img src=...>"),
):
    """Acepta la key por header X-Internal-Key o por ?key=. Si no hay key configurada, no exige auth."""
    if not INTERNAL_API_KEY:
        return
    if not (_keys_match(header_key) or _keys_match(key)):
        log_stm.warning("Acceso denegado a /stream (API key inválida o ausente)")
        raise HTTPException(status_code=401, detail="API Key inválida")


# ──────────────────────────────────────────────────────────────
# Cámara (hilo dedicado: siempre conserva solo el último frame)
# ──────────────────────────────────────────────────────────────
class CameraSource:
    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._seq = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.backend = "ninguno"
        self.ok = False
        self.captured = 0

    def start(self):
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                log_cam.warning("El hilo de cámara no terminó en 5 s")

    def latest(self, last_seq: int):
        """Devuelve (frame, seq) si hay un frame nuevo; si no, (None, last_seq)."""
        with self._lock:
            if self._frame is None or self._seq == last_seq:
                return None, last_seq
            return self._frame, self._seq

    def _publish(self, frame):
        with self._lock:
            self._frame = frame
            self._seq += 1
        self.captured += 1

    # ---- apertura de backends ----
    def _open_picam(self):
        if not HAS_PICAM2:
            log_cam.info("Picamera2 no disponible (PC o Docker), se usará OpenCV")
            return None
        try:
            cam = Picamera2()
            cfg = cam.create_video_configuration(main={"size": (FRAME_W, FRAME_H), "format": PICAM_FORMAT})
            cam.configure(cfg)
            cam.start()
            time.sleep(1.0)
            log_cam.info("✅ Cámara iniciada con Picamera2 (%dx%d, %s)", FRAME_W, FRAME_H, PICAM_FORMAT)
            return cam
        except Exception as e:
            log_cam.warning("No se pudo usar Picamera2: %s", e)
            return None

    def _open_cv(self):
        src = os.getenv("STREAM_URL", "0").strip()
        src = int(src) if src.isdigit() else src
        cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # evita procesar frames viejos
        if not cap.isOpened():
            log_cam.error("❌ No se pudo abrir OpenCV VideoCapture en la fuente %r", src)
            cap.release()
            return None
        log_cam.info("✅ Cámara iniciada con OpenCV VideoCapture (%r)", src)
        return cap

    def _run(self):
        log_cam.info("Iniciando hilo de cámara...")
        picam = None
        cap = None
        fails = 0

        while not self._stop.is_set():
            # (Re)abrir la fuente si hace falta
            if picam is None and cap is None:
                picam = self._open_picam()
                if picam is not None:
                    self.backend = "picamera2"
                else:
                    cap = self._open_cv()
                    self.backend = "opencv" if cap is not None else "ninguno"
                if picam is None and cap is None:
                    self.ok = False
                    log_cam.error("Sin cámara disponible. Reintentando en 5 s...")
                    self._stop.wait(5)
                    continue
                self.ok = True
                fails = 0

            # Captura
            frame = None
            try:
                if picam is not None:
                    frame = picam.capture_array()
                else:
                    ret, frame = cap.read()
                    if not ret:
                        frame = None
            except Exception as e:
                log_cam.error("Error capturando frame (%s): %s", self.backend, e)
                frame = None

            if frame is None:
                fails += 1
                if fails == 1:
                    log_cam.warning("Frame vacío o error de lectura (%s)", self.backend)
                if fails >= 30:
                    log_cam.error("30 lecturas fallidas seguidas: reiniciando la cámara")
                    self._release(picam, cap)
                    picam = cap = None
                    self.ok = False
                    self._stop.wait(2)
                else:
                    time.sleep(0.05)
                continue

            if fails:
                log_cam.info("Cámara recuperada tras %d fallos", fails)
                fails = 0
            self.ok = True
            self._publish(frame)

        self._release(picam, cap)
        self.ok = False
        log_cam.info("Cámara detenida y recursos liberados")

    @staticmethod
    def _release(picam, cap):
        if picam is not None:
            try:
                picam.stop()
                picam.close()
            except Exception as e:
                log_cam.warning("Error cerrando Picamera2: %s", e)
        if cap is not None:
            try:
                cap.release()
            except Exception as e:
                log_cam.warning("Error liberando VideoCapture: %s", e)


# ──────────────────────────────────────────────────────────────
# Cliente Spring (cola + hilo: la inferencia nunca espera a la red)
# ──────────────────────────────────────────────────────────────
class SpringClient:
    def __init__(self):
        self._q: queue.Queue = queue.Queue(maxsize=SPRING_QUEUE_MAX)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._throttle: dict[str, float] = {}
        self.connected: Optional[bool] = None   # None = aún no sabemos
        self.sent_ok = 0
        self.sent_fail = 0
        self.dropped = 0

    def start(self):
        self._thread = threading.Thread(target=self._run, name="spring", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def pending(self) -> int:
        return self._q.qsize()

    def submit(self, kind: str, url: str, payload: dict):
        try:
            self._q.put_nowait((kind, url, payload))
        except queue.Full:
            self.dropped += 1
            self._warn_throttled("cola_llena", 10,
                                 "Cola de envíos llena (%d): descartando %s. ¿Spring lento o caído? (descartados: %d)",
                                 SPRING_QUEUE_MAX, kind, self.dropped)

    # ---- helpers de log ----
    def _warn_throttled(self, key: str, every: float, msg: str, *args):
        now = time.monotonic()
        if now - self._throttle.get(key, 0) >= every:
            self._throttle[key] = now
            log_spr.warning(msg, *args)

    def _set_state(self, ok: bool, detail: str):
        if ok and self.connected is not True:
            if self.connected is False:
                log_spr.info("✅ Conexión con Spring RESTABLECIDA (%s) → %s", detail, SPRING_URL)
            else:
                log_spr.info("✅ Conectado a Spring (%s) → %s", detail, SPRING_URL)
        elif not ok:
            if self.connected is not False:
                log_spr.error("❌ No se pudo conectar a Spring → %s | %s", SPRING_URL, detail)
            else:
                self._warn_throttled("sigue_caido", 30, "Spring sigue sin responder → %s | %s", SPRING_URL, detail)
        self.connected = ok

    def _probe(self, client: httpx.Client):
        """Cualquier respuesta HTTP significa que el servidor es alcanzable."""
        try:
            r = client.get(SPRING_URL, timeout=3.0)
            self._set_state(True, f"HTTP {r.status_code}")
        except httpx.HTTPError as e:
            self._set_state(False, f"{type(e).__name__}: {e}")

    def _send(self, client: httpx.Client, kind: str, url: str, payload: dict):
        t0 = time.monotonic()
        try:
            if kind == "sample":
                # Imagen cruda para el dataset: multipart con el campo "files"
                r = client.post(
                    url,
                    files={"files": (payload["filename"], payload["jpeg"], "image/jpeg")},
                    timeout=SPRING_UPLOAD_TIMEOUT,
                )
            else:
                r = client.post(url, json=payload)
        except httpx.TransportError as e:
            self.sent_fail += 1
            self._set_state(False, f"{kind}: {type(e).__name__}: {e}")
            return
        except Exception as e:
            self.sent_fail += 1
            log_spr.exception("Error inesperado enviando %s: %s", kind, e)
            return

        ms = (time.monotonic() - t0) * 1000
        self._set_state(True, f"HTTP {r.status_code}")

        if r.is_success:
            self.sent_ok += 1
            if kind == "sample":
                try:
                    res = r.json()[0]
                    log_spr.info("Muestra enviada: %s → %s (%.0f ms)", res.get("fileName"), res.get("status"), ms)
                except Exception:
                    log_spr.debug("Muestra enviada (respuesta no interpretable)")
            else:
                log_spr.debug("%s enviado OK (%d, %.0f ms)", kind, r.status_code, ms)
        elif r.status_code in (401, 403):
            self.sent_fail += 1
            self._warn_throttled("auth", 30,
                                 "Spring rechazó el envío de %s (HTTP %d): revisa INTERNAL_API_KEY",
                                 kind, r.status_code)
        else:
            self.sent_fail += 1
            self._warn_throttled(f"http_{kind}_{r.status_code}", 10,
                                 "Spring respondió HTTP %d a %s (%.0f ms): %s",
                                 r.status_code, kind, ms, r.text[:200])

    def _run(self):
        # La key viaja en TODAS las peticiones (envíos, sondeo y subida de imágenes)
        headers = {"X-Internal-Key": INTERNAL_API_KEY} if INTERNAL_API_KEY else {}
        log_spr.info("Verificando conexión con Spring en %s ...", SPRING_URL)
        with httpx.Client(timeout=SPRING_TIMEOUT, headers=headers) as client:
            self._probe(client)
            while not self._stop.is_set():
                try:
                    kind, url, payload = self._q.get(timeout=SPRING_PROBE_EVERY)
                except queue.Empty:
                    self._probe(client)  # sin tráfico: verifica que Spring siga vivo
                    continue
                self._send(client, kind, url, payload)
        log_spr.info("Cliente Spring detenido")


# ──────────────────────────────────────────────────────────────
# Publicación del stream (multi-espectador, JPEG solo si hay alguien viendo)
# ──────────────────────────────────────────────────────────────
class StreamHub:
    def __init__(self):
        self._cond = threading.Condition()
        self._jpeg = b""
        self._seq = 0
        self._viewers = 0

    @property
    def viewers(self) -> int:
        return self._viewers

    def publish(self, jpeg: bytes):
        with self._cond:
            self._jpeg = jpeg
            self._seq += 1
            self._cond.notify_all()

    def add_viewer(self):
        with self._cond:
            self._viewers += 1
            n = self._viewers
        log_stm.info("Espectador conectado (activos: %d)", n)

    def remove_viewer(self):
        with self._cond:
            self._viewers = max(0, self._viewers - 1)
            n = self._viewers
            self._cond.notify_all()
        log_stm.info("Espectador desconectado (activos: %d)", n)

    def wait_next(self, last_seq: int, timeout: float = 1.0):
        with self._cond:
            self._cond.wait_for(lambda: self._seq != last_seq, timeout=timeout)
            if self._seq == last_seq:
                return None, last_seq
            return self._jpeg, self._seq


# ──────────────────────────────────────────────────────────────
# Estado global
# ──────────────────────────────────────────────────────────────
stop_event = threading.Event()
camera = CameraSource()
spring = SpringClient()
hub = StreamHub()
yolo: Optional[YOLOModel] = None

_last_behavior: dict[int, str] = {}
_last_seen: dict[int, float] = {}
_pending: dict[int, tuple[str, int]] = {}
_state_lock = threading.Lock()

_infer_ms: deque = deque(maxlen=30)
_infer_count = 0


# ──────────────────────────────────────────────────────────────
# Tracking & deduplicación
# ──────────────────────────────────────────────────────────────
def should_send(track_id: int, label: str) -> bool:
    now = time.time()
    with _state_lock:
        _last_seen[track_id] = now
        pending_label, count = _pending.get(track_id, (label, 0))

        if pending_label == label:
            count += 1
        else:
            pending_label, count = label, 1

        _pending[track_id] = (pending_label, count)

        if count >= CONFIRMATIONS_NEEDED and _last_behavior.get(track_id) != label:
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
    if expirados:
        log_inf.debug("Tracks expirados limpiados: %d", len(expirados))


# ──────────────────────────────────────────────────────────────
# Muestreo de imágenes crudas para el dataset
# ──────────────────────────────────────────────────────────────
def next_sample_time() -> float:
    """Momento de la próxima muestra: intervalo aleatorio entre MIN y MAX."""
    lo = min(SAMPLE_MIN_S, SAMPLE_MAX_S)
    hi = max(SAMPLE_MIN_S, SAMPLE_MAX_S)
    return time.time() + random.uniform(lo, hi)


def submit_sample(frame):
    """Codifica el frame crudo (sin anotaciones) y lo encola para Spring."""
    # No competir con detecciones/comportamientos si la cola ya está cargada
    if spring.pending() >= SPRING_QUEUE_MAX // 2:
        log_inf.debug("Muestra omitida: cola de Spring ocupada")
        return
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY_SAMPLE])
    if not ok:
        log_inf.warning("No se pudo codificar la muestra a JPEG")
        return
    spring.submit("sample", SPRING_IMAGES_URL, {
        "filename": f"cam_{int(time.time())}.jpg",
        "jpeg": jpeg.tobytes(),
    })
    log_inf.info("Muestra para dataset encolada (%.0f KB)", len(jpeg) / 1024)


# ──────────────────────────────────────────────────────────────
# Inferencia (hilo propio, ritmo limitado por INFER_FPS)
# ──────────────────────────────────────────────────────────────
def inference_worker():
    global _infer_count
    log_inf.info("Hilo de inferencia iniciado (máx %.1f FPS)", INFER_FPS)

    last_seq = 0
    last_infer = 0.0
    last_cleanup = time.time()
    last_stats_send = time.time()
    last_summary = time.time()
    no_frame_warned = False
    next_sample = next_sample_time()

    while not stop_event.is_set():
        # Respeta el ritmo máximo de inferencia
        wait = INFER_INTERVAL - (time.monotonic() - last_infer)
        if wait > 0:
            time.sleep(wait)

        frame, seq = camera.latest(last_seq)
        if frame is None:
            if not no_frame_warned and not camera.ok:
                log_inf.warning("Esperando frames de la cámara...")
                no_frame_warned = True
            time.sleep(0.01)
            continue
        no_frame_warned = False
        last_seq = seq
        last_infer = time.monotonic()

        # Muestra aleatoria para el dataset: frame crudo, ANTES de que YOLO lo procese
        if SAMPLE_ENABLED and time.time() >= next_sample:
            submit_sample(frame)
            next_sample = next_sample_time()

        try:
            t0 = time.monotonic()
            out = yolo.detect_image(frame, return_crops=True)
            _infer_ms.append((time.monotonic() - t0) * 1000)
            _infer_count += 1
        except Exception as e:
            log_inf.exception("Error en inferencia YOLO: %s", e)
            time.sleep(0.5)
            continue

        annotated = out.get("annotated", frame)
        crops = out.get("crops", [])

        # Solo codificamos JPEG si alguien está viendo el stream
        if hub.viewers > 0:
            ok, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY_STREAM])
            if ok:
                hub.publish(jpeg.tobytes())

        now = time.time()

        # Estadísticas agrupadas
        if now - last_stats_send >= SEND_INTERVAL and crops:
            counts: dict[str, int] = {}
            for c in crops:
                counts[c["label"]] = counts.get(c["label"], 0) + 1
            spring.submit("detections", SPRING_DETECTIONS_URL, {"counts": counts})
            log_inf.debug("Estadísticas encoladas: %s", counts)
            last_stats_send = now

        # Comportamientos confirmados
        for crop in crops:
            track_id = crop.get("track_id", -1)
            label = crop["label"]
            if should_send(track_id, label):
                log_inf.info("Comportamiento confirmado → track_id=%s label=%s conf=%.2f",
                             track_id, label, crop["conf"])
                spring.submit("behavior", SPRING_BEHAVIOR_URL, {
                    "timestamp": now,
                    "track_id": track_id,
                    "label": label,
                    "conf": crop["conf"],
                    "image_b64": crop["image_b64"],
                })

        if now - last_cleanup > 60:
            limpiar_tracks_viejos()
            last_cleanup = now

        # Resumen periódico
        if now - last_summary >= STATS_LOG_EVERY:
            avg = sum(_infer_ms) / len(_infer_ms) if _infer_ms else 0
            log_inf.info(
                "Resumen: inferencia %.0f ms (~%.1f FPS) | cámara=%s | Spring=%s ok=%d fallos=%d "
                "descartados=%d cola=%d | espectadores=%d",
                avg, (1000 / avg) if avg else 0,
                "OK" if camera.ok else "SIN SEÑAL",
                {True: "OK", False: "CAÍDO", None: "?"}[spring.connected],
                spring.sent_ok, spring.sent_fail, spring.dropped, spring.pending(),
                hub.viewers,
            )
            last_summary = now

    log_inf.info("Hilo de inferencia detenido")


# ──────────────────────────────────────────────────────────────
# FastAPI
# ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global yolo
    log.info("Arrancando CUT IA Behavior API")
    log.info("Spring: %s | modelo: %s (conf=%.2f) | INFER_FPS=%.1f", SPRING_URL, MODEL_PATH, MODEL_CONF, INFER_FPS)
    if SAMPLE_ENABLED:
        log.info("Muestreo de dataset ACTIVO: cada %.0f-%.0f s → %s", SAMPLE_MIN_S, SAMPLE_MAX_S, SPRING_IMAGES_URL)
    else:
        log.info("Muestreo de dataset desactivado (SAMPLE_ENABLED=false)")

    if not INTERNAL_API_KEY:
        log.warning("INTERNAL_API_KEY vacía: /stream queda SIN autenticación y Spring puede rechazar los envíos")
    if CORS_ORIGINS == ["*"]:
        log.warning("CORS abierto a cualquier origen (define CORS_ORIGINS en producción)")

    t0 = time.monotonic()
    yolo = YOLOModel(model_path=MODEL_PATH, conf=MODEL_CONF)
    log.info("Modelo YOLO cargado en %.1f s | clases: %s", time.monotonic() - t0, list(yolo.get_names().values()))

    spring.start()
    camera.start()
    inf_thread = threading.Thread(target=inference_worker, name="inference", daemon=True)
    inf_thread.start()

    yield

    log.info("Apagando servicio...")
    stop_event.set()
    inf_thread.join(timeout=5)
    camera.stop()
    spring.stop()
    log.info("Servicio detenido")


app = FastAPI(title="CUT IA Behavior API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    avg = sum(_infer_ms) / len(_infer_ms) if _infer_ms else 0
    return {
        "status": "ok" if camera.ok else "degraded",
        "camera": {"ok": camera.ok, "backend": camera.backend},
        "spring": {
            "connected": spring.connected,
            "sent_ok": spring.sent_ok,
            "sent_fail": spring.sent_fail,
            "dropped": spring.dropped,
            "pending": spring.pending(),
        },
        "inference_ms": round(avg, 1),
        "viewers": hub.viewers,
    }


@app.get("/labels")
def labels():
    return {str(k): v for k, v in yolo.get_names().items()}


@app.get("/stream", dependencies=[Depends(verify_stream_key)])
def stream():
    def generate():
        hub.add_viewer()
        last_seq = 0
        try:
            while not stop_event.is_set():
                buf, last_seq = hub.wait_next(last_seq, timeout=1.0)
                if buf:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf + b"\r\n")
        finally:
            hub.remove_viewer()

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")