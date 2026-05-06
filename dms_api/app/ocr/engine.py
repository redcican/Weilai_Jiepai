"""
CnOCR Engine

OCR engine wrapper using CnOCR with ONNX backend.
"""

import logging
from typing import List, Optional, Union
from io import BytesIO

from .models import OCRBox
from .utils import correct_ocr_text, enhance_image_for_ocr, enhance_image_bytes_for_ocr

logger = logging.getLogger(__name__)


class CnOCREngine:
    """CnOCR Engine with ONNX backend for optimized inference."""

    _instance: Optional["CnOCREngine"] = None

    def __init__(self, enhance_image: bool = True):
        self._available = False
        self.ocr = None
        self.enhance_image = enhance_image

        try:
            from cnocr import CnOcr

            self.ocr = CnOcr(
                rec_model_name='densenet_lite_136-gru',
                det_model_name='ch_PP-OCRv3_det',
                rec_model_backend='onnx',
                det_model_backend='onnx',
            )
            self._available = True
            logger.info("CnOCR (ONNX) engine initialized successfully")
        except ImportError as e:
            logger.error(f"CnOCR not installed: {e}")
        except Exception as e:
            logger.error(f"CnOCR initialization failed: {e}")

    @classmethod
    def get_instance(cls, enhance_image: bool = True) -> "CnOCREngine":
        """Get singleton instance of the OCR engine."""
        if cls._instance is None:
            cls._instance = cls(enhance_image=enhance_image)
        return cls._instance

    @property
    def available(self) -> bool:
        return self._available

    def recognize_file(self, image_path: str) -> List[OCRBox]:
        """Execute OCR recognition on an image file."""
        if not self._available:
            return []

        try:
            if self.enhance_image:
                enhanced = enhance_image_for_ocr(image_path)
                if enhanced is not None:
                    result = self.ocr.ocr(enhanced)
                else:
                    result = self.ocr.ocr(image_path)
            else:
                result = self.ocr.ocr(image_path)
        except Exception as e:
            logger.warning(f"OCR recognition failed: {e}")
            return []

        return self._parse_results(result)

    def recognize_bytes(self, image_bytes: bytes) -> List[OCRBox]:
        """Execute OCR recognition on image bytes."""
        if not self._available:
            return []

        try:
            if self.enhance_image:
                enhanced = enhance_image_bytes_for_ocr(image_bytes)
                if enhanced is not None:
                    result = self.ocr.ocr(enhanced)
                else:
                    from PIL import Image
                    import numpy as np
                    img = Image.open(BytesIO(image_bytes))
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    result = self.ocr.ocr(np.array(img))
            else:
                from PIL import Image
                import numpy as np
                img = Image.open(BytesIO(image_bytes))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                result = self.ocr.ocr(np.array(img))
        except Exception as e:
            logger.warning(f"OCR recognition failed: {e}")
            return []

        return self._parse_results(result)

    def recognize(self, image: Union[str, bytes]) -> List[OCRBox]:
        """Execute OCR recognition on image (file path or bytes)."""
        if isinstance(image, str):
            return self.recognize_file(image)
        elif isinstance(image, bytes):
            return self.recognize_bytes(image)
        else:
            logger.error(f"Unsupported image type: {type(image)}")
            return []

    def _parse_results(self, result) -> List[OCRBox]:
        """Parse CnOCR results into OCRBox list."""
        if not result:
            return []

        results = []
        for item in result:
            text = item.get('text', '').strip()
            if not text:
                continue

            text = correct_ocr_text(text)
            score = item.get('score', 0.0)
            position = item.get('position', None)

            if position is not None:
                try:
                    x_coords = [p[0] for p in position]
                    y_coords = [p[1] for p in position]
                    box = [int(min(x_coords)), int(min(y_coords)),
                           int(max(x_coords)), int(max(y_coords))]
                except (TypeError, ValueError, IndexError):
                    box = [0, len(results) * 30, 100, (len(results) + 1) * 30]
            else:
                box = [0, len(results) * 30, 100, (len(results) + 1) * 30]

            results.append(OCRBox(box=box, text=text, confidence=float(score)))

        return results
