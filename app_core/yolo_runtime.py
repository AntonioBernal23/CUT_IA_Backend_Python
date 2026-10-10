import base64
import logging
import os
from typing import Any, Dict, Optional
import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class YOLOModel:
    def __init__(self, model_path: str = "models/best.onnx", device: str = "cpu", conf: float = 0.25):
        self.model_path = model_path
        self.device = device
        self.conf = float(conf)
        self._model: Optional[YOLO] = None
        self._names: Dict[int, str] = {}

    def _ensure_model(self) -> None:
        if self._model is None:
            if not os.path.exists(self.model_path):
                logger.error("Modelo no encontrado en %s", self.model_path)
                raise FileNotFoundError(self.model_path)
            # Ultralytics carga ONNX directamente si el path termina en .onnx
            self._model = YOLO(self.model_path, task="detect")
            self._names = getattr(self._model, "names", {}) or {}

    def get_names(self) -> Dict[int, str]:
        self._ensure_model()
        return self._names

    @staticmethod
    def crop_to_b64(image_bgr: np.ndarray, box: np.ndarray, padding: int = 10) -> str:
        h, w = image_bgr.shape[:2]
        x1 = max(0, int(box[0]) - padding)
        y1 = max(0, int(box[1]) - padding)
        x2 = min(w, int(box[2]) + padding)
        y2 = min(h, int(box[3]) + padding)

        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return ""
        _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return base64.b64encode(buf).decode("utf-8")

    def detect_image(self, image_bgr: np.ndarray, return_crops: bool = False) -> Dict[str, Any]:
        self._ensure_model()
        results = self._model.track(
            image_bgr, 
            conf=self.conf, 
            persist=True, 
            tracker="bytetrack.yaml", 
            verbose=False
        )
        r0 = results[0]

        boxes = getattr(r0, "boxes", None)
        boxes_data = boxes.data.cpu().numpy() if boxes is not None else None

        crops = []
        if return_crops and boxes_data is not None:
            names = getattr(r0, "names", {}) or self._names
            for box in boxes_data:
                if len(box) == 7:
                    x1, y1, x2, y2, track_id, conf_v, cls = box
                    track_id = int(track_id)
                else:
                    x1, y1, x2, y2, conf_v, cls = box
                    track_id = -1

                cls_id = int(cls)
                crops.append({
                    "track_id": track_id,
                    "label": names.get(cls_id, str(cls_id)),
                    "cls": cls_id,
                    "conf": round(float(conf_v), 2),
                    "image_b64": self.crop_to_b64(image_bgr, box),
                })

        annotated = r0.plot() if boxes_data is not None else image_bgr
        return {"annotated": annotated, "crops": crops}