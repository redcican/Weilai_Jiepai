"""
PaddleOCR Video Engine

Supports both English and Chinese OCR models for different recognition tasks.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class PaddleOCREngine:
    """PaddleOCR engine for video frame recognition — singleton per (lang, use_gpu)."""

    _instances: dict = {}

    def __init__(self, use_gpu: bool = True, lang: str = "en"):
        self._available = False
        self.ocr = None
        self._use_gpu = False
        self._lang = lang

        try:
            import paddle
            from paddleocr import PaddleOCR

            gpu_available = (
                use_gpu
                and paddle.is_compiled_with_cuda()
                and paddle.device.cuda.device_count() > 0
            )

            if gpu_available:
                try:
                    self.ocr = PaddleOCR(
                        use_angle_cls=True,
                        lang=lang,
                        show_log=False,
                        use_gpu=True,
                    )
                    self._use_gpu = True
                    logger.info(f"PaddleOCR ({lang}) initialized on GPU")
                except Exception as e:
                    logger.warning(f"PaddleOCR ({lang}) GPU init failed: {e}, falling back to CPU")
                    gpu_available = False

            if not gpu_available:
                self.ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang=lang,
                    show_log=False,
                    use_gpu=False,
                )
                logger.info(f"PaddleOCR ({lang}) initialized on CPU")

            self._available = True

        except ImportError as e:
            logger.error(f"PaddleOCR not installed: {e}")
        except Exception as e:
            logger.error(f"PaddleOCR initialization failed: {e}")

    @classmethod
    def get_instance(cls, use_gpu: bool = True, lang: str = "en") -> "PaddleOCREngine":
        key = (lang, use_gpu)
        if key not in cls._instances:
            cls._instances[key] = cls(use_gpu=use_gpu, lang=lang)
        return cls._instances[key]

    @property
    def available(self) -> bool:
        return self._available

    @property
    def is_gpu(self) -> bool:
        return self._use_gpu

    def recognize(self, img_path: str):
        """Run OCR on an image file path."""
        if not self._available or self.ocr is None:
            return None
        try:
            return self.ocr.ocr(img_path, cls=True)
        except Exception as e:
            logger.warning(f"PaddleOCR ({self._lang}) recognition failed: {e}")
            return None
