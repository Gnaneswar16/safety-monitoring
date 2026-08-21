from dataclasses import dataclass
from typing import List, Tuple
import os
from ultralytics import YOLO
import config

@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2

class PPEDetector:
    def __init__(self, model_path: str = config.MODEL_PATH,
                 person_conf: float = config.PERSON_CONF_THRESHOLD,
                 glasses_conf: float = config.GLASSES_CONF_THRESHOLD,
                 shoes_conf: float = config.SHOES_CONF_THRESHOLD):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model file not found at {model_path}")
        self.model = YOLO(model_path)
        self.person_conf = person_conf
        self.glasses_conf = glasses_conf
        self.shoes_conf = shoes_conf
        self.class_names = config.CLASS_NAMES

    def detect(self, frame) -> List[Detection]:
        """Runs YOLO model inference on frame and filters detections per class threshold."""
        min_conf = min(self.person_conf, self.glasses_conf, self.shoes_conf)
        results = self.model(frame, conf=min_conf, verbose=False)

        detections: List[Detection] = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            cls_id = int(box.cls[0].cpu().item())
            conf = float(box.conf[0].cpu().item())
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(float, xyxy)

            # Apply per-class confidence thresholding
            if cls_id == config.CLASS_PERSON and conf < self.person_conf:
                continue
            elif cls_id == config.CLASS_GLASSES and conf < self.glasses_conf:
                continue
            elif cls_id == config.CLASS_SHOES and conf < self.shoes_conf:
                continue

            cls_name = self.class_names.get(cls_id, f"class_{cls_id}")
            detections.append(Detection(
                class_id=cls_id,
                class_name=cls_name,
                confidence=conf,
                bbox=(x1, y1, x2, y2)
            ))

        return detections
