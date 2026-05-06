"""
Pedestrian Detection Engine

YOLOv8 singleton engine with two-pass detection strategy:
  Pass 1: full image at high resolution (catches most persons)
  Pass 2: sliding-window tiling fallback (catches small/edge persons)

All parameters are read from Settings at initialization time.
GPU can be toggled per-request via the `use_gpu` parameter, or globally
via DMS_PEDESTRIAN_DETECTION_USE_GPU=true.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    class_name: str
    confidence: float
    bbox: list[float]  # [x1, y1, x2, y2] in original image coords


class PedestrianEngine:
    """YOLOv8 pedestrian detector — singleton."""

    _instance: Optional["PedestrianEngine"] = None

    def __init__(self, settings) -> None:
        self._available = False

        try:
            from ultralytics import YOLO

            self._model = YOLO(settings.pedestrian_detection_model)
            self._target_class_ids = self._resolve_class_ids(["person"])

            # Default device from config
            self._default_use_gpu = settings.pedestrian_detection_use_gpu

            # Primary pass settings
            self._confidence = settings.pedestrian_detection_confidence
            self._iou = settings.pedestrian_detection_iou
            self._imgsz = settings.pedestrian_detection_imgsz

            # Tiling settings
            self._tile_enabled = settings.pedestrian_detection_tile_enabled
            self._tile_size = settings.pedestrian_detection_tile_size
            self._tile_overlap = settings.pedestrian_detection_tile_overlap
            self._tile_confidence = settings.pedestrian_detection_tile_confidence
            self._tile_nms_iou = 0.5

            self._available = True
            device_label = "GPU (CUDA)" if self._default_use_gpu else "CPU"
            logger.info(
                f"Pedestrian engine initialized "
                f"(model={settings.pedestrian_detection_model}, "
                f"default device={device_label})"
            )
        except Exception:
            logger.exception("Failed to initialize pedestrian engine")

    @classmethod
    def get_instance(cls, settings=None) -> "PedestrianEngine":
        if cls._instance is None:
            if settings is None:
                from ..config import get_settings

                settings = get_settings()
            cls._instance = cls(settings)
        return cls._instance

    @property
    def available(self) -> bool:
        return self._available

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_from_bytes(
        self, image_bytes: bytes, use_gpu: bool | None = None
    ) -> dict:
        """Detect pedestrians from raw image bytes.

        Args:
            image_bytes: Raw image file content.
            use_gpu: Override device for this call.
                     None = use config default,
                     True = force GPU, False = force CPU.

        Returns:
            {"status": "正常"|"异常", "detections": [...]}
        """
        if not self._available:
            return {"status": "正常", "detections": []}

        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return {"status": "正常", "detections": []}

        # Resolve device: per-call override > config default
        gpu = use_gpu if use_gpu is not None else self._default_use_gpu
        device = "cuda" if gpu else "cpu"

        detections = self._detect(img, device)
        status = "异常" if detections else "正常"

        return {
            "status": status,
            "detections": [
                {
                    "class_name": d.class_name,
                    "confidence": round(d.confidence, 3),
                    "bbox": [round(v, 1) for v in d.bbox],
                }
                for d in detections
            ],
        }

    # ------------------------------------------------------------------
    # Detection pipeline
    # ------------------------------------------------------------------

    def _detect(
        self, img: np.ndarray, device: str
    ) -> list[DetectionResult]:
        """Two-pass detection: full image, then tiling fallback."""
        # Pass 1: full image at primary resolution
        detections = self._run_yolo(
            img,
            conf=self._confidence,
            imgsz=self._imgsz,
            device=device,
        )
        if detections:
            return detections

        # Pass 2: sliding-window tiling
        if not self._tile_enabled:
            return []

        img_h, img_w = img.shape[:2]
        tiles = self._generate_tiles(img_h, img_w)
        all_detections: list[DetectionResult] = []

        for x1, y1, x2, y2 in tiles:
            tile = img[y1:y2, x1:x2]
            tile_dets = self._run_yolo(
                tile,
                conf=self._tile_confidence,
                imgsz=self._tile_size,
                device=device,
            )
            for d in tile_dets:
                all_detections.append(DetectionResult(
                    class_name=d.class_name,
                    confidence=d.confidence,
                    bbox=[
                        d.bbox[0] + x1,
                        d.bbox[1] + y1,
                        d.bbox[2] + x1,
                        d.bbox[3] + y1,
                    ],
                ))

        nms_detections = self._nms(all_detections, self._tile_nms_iou)

        # Filter tiling detections by safety gear color verification
        verified = [
            d for d in nms_detections if self._has_safety_gear(img, d.bbox)
        ]
        return verified

    def _run_yolo(
        self,
        img: np.ndarray,
        conf: float,
        imgsz: int,
        device: str,
    ) -> list[DetectionResult]:
        """Run YOLO inference and extract target-class detections."""
        results = self._model.predict(
            source=img,
            conf=conf,
            iou=self._iou,
            imgsz=imgsz,
            device=device,
            classes=self._target_class_ids,
            verbose=False,
        )
        detections = []
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                confidence = float(boxes.conf[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                detections.append(DetectionResult(
                    class_name=self._model.names[cls_id],
                    confidence=confidence,
                    bbox=[x1, y1, x2, y2],
                ))
        return detections

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_class_ids(self, target_names: list[str]) -> list[int]:
        """Map class names to model class IDs."""
        name_to_id = {name: idx for idx, name in self._model.names.items()}
        ids = []
        for name in target_names:
            if name in name_to_id:
                ids.append(name_to_id[name])
            else:
                logger.warning(f"Class '{name}' not found in model")
        return ids

    def _generate_tiles(
        self, img_h: int, img_w: int
    ) -> list[tuple[int, int, int, int]]:
        """Generate overlapping tile coordinates."""
        step = int(self._tile_size * (1 - self._tile_overlap))
        tiles = []
        for y in range(0, img_h, step):
            for x in range(0, img_w, step):
                x2 = min(x + self._tile_size, img_w)
                y2 = min(y + self._tile_size, img_h)
                x1 = max(0, x2 - self._tile_size)
                y1 = max(0, y2 - self._tile_size)
                tiles.append((x1, y1, x2, y2))
        return list(dict.fromkeys(tiles))

    @staticmethod
    def _has_safety_gear(
        img: np.ndarray, bbox: list[float], min_pixels: int = 100
    ) -> bool:
        """Check if the bounding box contains safety gear colors.

        Workers in freight carriage images wear orange hard hats and
        fluorescent vests.  We count pixels matching those HSV ranges
        inside the detection bbox.  This filters tiling false-positives
        (mechanical parts, shadows) that lack these colours.

        Args:
            img: Original BGR image.
            bbox: [x1, y1, x2, y2] in pixel coordinates.
            min_pixels: Minimum combined orange+fluorescent pixel count.

        Returns:
            True if enough safety-gear pixels are found.
        """
        h, w = img.shape[:2]
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(w, int(bbox[2]))
        y2 = min(h, int(bbox[3]))
        if x2 <= x1 or y2 <= y1:
            return False

        roi = img[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Orange hard hat: H 5-22, S > 100, V > 100
        mask_hat = cv2.inRange(hsv, (5, 100, 100), (22, 255, 255))
        # Fluorescent vest: H 25-85, S > 60, V > 80
        mask_vest = cv2.inRange(hsv, (25, 60, 80), (85, 255, 255))

        gear_pixels = int(cv2.countNonZero(mask_hat) + cv2.countNonZero(mask_vest))
        return gear_pixels >= min_pixels

    @staticmethod
    def _nms(
        detections: list[DetectionResult], iou_threshold: float
    ) -> list[DetectionResult]:
        """Non-maximum suppression."""
        if not detections:
            return []

        boxes = np.array([d.bbox for d in detections])
        scores = np.array([d.confidence for d in detections])

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            union = areas[i] + areas[order[1:]] - inter
            iou = inter / np.maximum(union, 1e-6)

            remaining = np.where(iou <= iou_threshold)[0]
            order = order[remaining + 1]

        return [detections[i] for i in keep]
