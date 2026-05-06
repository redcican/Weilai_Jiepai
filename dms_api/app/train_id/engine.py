"""
Train ID OCR Engine

Hybrid CnOCR engine wrapper with two detection models.
Handles engine initialization, singleton management, and raw OCR calls.
"""

import logging
from typing import List, Optional

import numpy as np

from .models import OCRBox

logger = logging.getLogger(__name__)


class TrainIDEngine:
    """Hybrid CnOCR engine for train ID recognition.

    Uses two detection models:
    - db_resnet18 (pytorch): superior digit accuracy
    - ch_PP-OCRv3_det (onnx): better vehicle type detection

    Singleton pattern — call get_instance() to reuse.
    """

    _instance: Optional["TrainIDEngine"] = None

    def __init__(self):
        self._available = False
        self.ocr_resnet = None
        self.ocr_ppocr = None

        try:
            from cnocr import CnOcr

            self.ocr_resnet = CnOcr(
                rec_model_name="densenet_lite_136-gru",
                det_model_name="db_resnet18",
                rec_model_backend="onnx",
                det_model_backend="pytorch",
            )
            self.ocr_ppocr = CnOcr(
                rec_model_name="densenet_lite_136-gru",
                det_model_name="ch_PP-OCRv3_det",
                rec_model_backend="onnx",
                det_model_backend="onnx",
            )
            self._available = True
            logger.info("Train ID OCR engines initialized (db_resnet18 + ch_PP-OCRv3_det)")
        except ImportError as e:
            logger.error(f"CnOCR or torch not installed: {e}")
        except Exception as e:
            logger.error(f"Train ID engine initialization failed: {e}")

    @classmethod
    def get_instance(cls) -> "TrainIDEngine":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def available(self) -> bool:
        return self._available

    def recognize(self, image: np.ndarray, engine: str = "resnet") -> List[OCRBox]:
        """Run OCR on a preprocessed image and return parsed boxes.

        Args:
            image: Preprocessed RGB/BGR numpy array
            engine: "resnet" for db_resnet18 or "ppocr" for ch_PP-OCRv3_det

        Returns:
            List of OCRBox with bounding box, text, and confidence
        """
        if not self._available:
            return []

        ocr = self.ocr_resnet if engine == "resnet" else self.ocr_ppocr
        try:
            result = ocr.ocr(image)
        except Exception as e:
            logger.warning(f"OCR recognition failed ({engine}): {e}")
            return []

        if not result:
            return []

        return self._parse_results(result)

    def _parse_results(self, result) -> List[OCRBox]:
        """Parse CnOCR results into OCRBox list."""
        boxes = []
        for item in result:
            text = item.get("text", "").strip()
            if not text:
                continue
            score = float(item.get("score", 0.0))
            position = item.get("position")
            if position is not None:
                try:
                    xs = [p[0] for p in position]
                    ys = [p[1] for p in position]
                    box = [int(min(xs)), int(min(ys)),
                           int(max(xs)), int(max(ys))]
                except (TypeError, ValueError, IndexError):
                    box = [0, len(boxes) * 30, 100, (len(boxes) + 1) * 30]
            else:
                box = [0, len(boxes) * 30, 100, (len(boxes) + 1) * 30]
            boxes.append(OCRBox(box=box, text=text, confidence=score))
        return boxes
