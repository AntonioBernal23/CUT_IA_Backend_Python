from typing import Any, Dict, Optional
import os, logging
import numpy as np
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

    def detect_image(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        self._ensure_model()
        results = self._model(image_bgr, conf=self.conf)
        r0 = results[0]

        # boxes
        boxes = None
        try:
            if hasattr(r0, "boxes") and r0.boxes is not None:
                boxes = r0.boxes.data.cpu().numpy()
        except Exception:
            boxes = None

        # anotada
        annotated = None
        try:
            annotated = r0.plot()  # devuelve BGR
        except Exception:
            annotated = None

        names = getattr(r0, "names", {}) or self.get_names()
        return {"annotated": annotated, "boxes": boxes, "names": names, "raw": r0}