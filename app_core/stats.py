from typing import Dict
import numpy as np

def count_by_class(boxes: np.ndarray, id2label: Dict[int, str]) -> Dict[str, int]:
    if boxes is None or boxes.size == 0:
        return {}
    cls_col = boxes[:, 5].astype(int)
    out: Dict[str, int] = {}
    for cid in cls_col:
        label = id2label.get(int(cid), str(int(cid)))
        out[label] = out.get(label, 0) + 1
    return out
