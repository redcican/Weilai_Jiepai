"""
Train ID Service

Business logic for train identification recognition operations.
直接复用 train_id_ocr_paddle.py 的 PaddleOCRProcessor。
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# 把项目根目录加入路径，以便导入 train_id_ocr_paddle
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from train_id_ocr.train_id_ocr_paddle import PaddleOCRProcessor, ImageResult

from ..schemas.train_id import TrainIDData, TrainIDBatchItem

logger = logging.getLogger(__name__)


class TrainIDService:
    """
    Service for train ID recognition operations.

    直接复用 train_id_ocr_paddle.py 的 PaddleOCRProcessor，
    保证 API 和 CLI 使用同一份核心代码。
    """

    _ocr_processor: Optional[PaddleOCRProcessor] = None

    @classmethod
    def get_ocr_processor(cls) -> PaddleOCRProcessor:
        """Get singleton PaddleOCR processor instance (from train_id_ocr_paddle)."""
        if cls._ocr_processor is None:
            cls._ocr_processor = PaddleOCRProcessor()
        return cls._ocr_processor

    @property
    def available(self) -> bool:
        """Check if train ID engine is available."""
        return self.get_ocr_processor().ocr is not None

    # ------------------------------------------------------------------
    # Single image recognition
    # ------------------------------------------------------------------

    async def recognize_image(
        self,
        image_bytes: bytes,
        filename: str = "unknown",
    ) -> TrainIDData:
        """
        Recognize vehicle type and number from a single image.

        直接复用 train_id_ocr_paddle.py 的 PaddleOCRProcessor，
        支持空挡检测(type字段)。
        """
        processor = self.get_ocr_processor()

        if processor.ocr is None:
            logger.error("Train ID engine not available")
            return TrainIDData()

        logger.info(f"Processing train ID image: {filename}, size={len(image_bytes)} bytes")

        result: ImageResult = processor.process_bytes(image_bytes)

        vehicle_type = result.train_types[0][0] if result.train_types else ""
        vehicle_number = result.train_numbers[0][0] if result.train_numbers else ""

        # 计算平均置信度
        confs = []
        if result.train_types:
            confs.append(result.train_types[0][1])
        if result.train_numbers:
            confs.append(result.train_numbers[0][1])
        avg_conf = round(sum(confs) / len(confs), 4) if confs else 0.0

        logger.info(
            f"Train ID result: type='{vehicle_type}' "
            f"number='{vehicle_number}' "
            f"gap='{'########' if result.is_gap else ''}' "
            f"confidence={avg_conf:.3f}"
        )

        return TrainIDData(
            type="########" if result.is_gap else "",
            vehicleType=vehicle_type,
            vehicleNumber=vehicle_number,
            confidence=avg_conf,
        )

    # ------------------------------------------------------------------
    # Batch recognition
    # ------------------------------------------------------------------

    async def recognize_batch(
        self,
        images: list[tuple[bytes, str]],
    ) -> list[TrainIDBatchItem]:
        """
        Recognize vehicle info from multiple images.
        """
        results = []
        for image_bytes, filename in images:
            data = await self.recognize_image(image_bytes, filename)
            results.append(TrainIDBatchItem(
                filename=filename,
                type=data.type,
                vehicleType=data.vehicle_type,
                vehicleNumber=data.vehicle_number,
                confidence=data.confidence,
            ))
        return results


# Singleton instance
_train_id_service: Optional[TrainIDService] = None


def get_train_id_service_singleton() -> TrainIDService:
    """Get train ID service singleton instance."""
    global _train_id_service
    if _train_id_service is None:
        _train_id_service = TrainIDService()
    return _train_id_service
