from typing import Any, Dict, Optional
import os, logging, base64
import numpy as np
import cv2
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class YOLOModel:
    def __init__(self, model_path: str = "models/best_2.0.pt", device: str = "cpu", conf: float = 0.25):
        self.model_path = model_path
        self.device = device
        self.conf = float(conf)
        self._model: Optional[YOLO] = None
        self._names: Dict[int, str] = {}

    def _ensure_model(self) -> None:
        if self._model is None:
            if not os.path.exists(self.model_path):
                logger.error("Modelo no encontrado %s", self.model_path)
                raise FileNotFoundError(self.model_path)
            self._model = YOLO(self.model_path)
            try:
                if self.device and self.device != "cpu":
                    self._model.to(self.device)
            except Exception:
                pass
            try:
                self._names = getattr(self._model.model, "names", {}) or {}
            except Exception:
                self._names = {}

    def set_confidence(self, conf: float) -> None:
        self.conf = float(conf)

    def get_names(self) -> Dict[int, str]:
        self._ensure_model()
        return self._names

    def _crop_to_b64(self, image_bgr: np.ndarray, box: np.ndarray, padding: int = 10) -> str:
        """Recorta la detección del frame y la devuelve como base64 JPEG."""
        h, w = image_bgr.shape[:2]
        x1 = max(0, int(box[0]) - padding)
        y1 = max(0, int(box[1]) - padding)
        x2 = min(w, int(box[2]) + padding)
        y2 = min(h, int(box[3]) + padding)

        crop = image_bgr[y1:y2, x1:x2]
        _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf).decode("utf-8")

    def detect_image(self, image_bgr: np.ndarray, return_crops: bool = False) -> Dict[str, Any]:
        self._ensure_model()
    
        # ← CAMBIO CLAVE: track() en lugar de __call__()
        results = self._model.track(image_bgr, conf=self.conf, persist=True, tracker="bytetrack.yaml")
        r0 = results[0]

        boxes = None
        try:
            if hasattr(r0, "boxes") and r0.boxes is not None:
                boxes = r0.boxes.data.cpu().numpy()  # ahora incluye track_id en col[4] o [6]
        except Exception:
            boxes = None

        # Con tracking, data tiene 7 columnas: x1,y1,x2,y2,track_id,conf,cls
        # Sin detección tiene 6 (sin track_id), manejar ambos
        crops = []
        if return_crops and boxes is not None:
            names = getattr(r0, "names", {}) or self.get_names()
            for box in boxes:
                if len(box) == 7:
                    x1, y1, x2, y2, track_id, conf_v, cls = box
                    track_id = int(track_id)
                else:
                    x1, y1, x2, y2, conf_v, cls = box
                    track_id = -1  # sin ID asignado

                cls_id = int(cls)
                label = names.get(cls_id, str(cls_id))

                crops.append({
                    "track_id":   track_id,
                    "label":      label,
                    "cls":        cls_id,
                    "conf":       round(float(conf_v), 3),
                    "image_b64":  self._crop_to_b64(image_bgr, box),
                })

        annotated = None
        try:
            annotated = r0.plot()
        except Exception:
            pass

        names = getattr(r0, "names", {}) or self.get_names()
        return {"annotated": annotated, "boxes": boxes, "names": names, "crops": crops, "raw": r0}