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
from train_id_ocr.run_bottom_merge_ocr import FlatcarBottomProcessor

from ..schemas.train_id import TrainIDData, TrainIDBatchItem, FlatcarData

logger = logging.getLogger(__name__)


class TrainIDService:
    """
    Service for train ID recognition operations.

    直接复用 train_id_ocr_paddle.py 的 PaddleOCRProcessor，
    保证 API 和 CLI 使用同一份核心代码。
    """

    _ocr_processor: Optional[PaddleOCRProcessor] = None
    _flatcar_processor: Optional[FlatcarBottomProcessor] = None

    @classmethod
    def get_ocr_processor(cls) -> PaddleOCRProcessor:
        """Get singleton PaddleOCR processor instance (from train_id_ocr_paddle)."""
        if cls._ocr_processor is None:
            cls._ocr_processor = PaddleOCRProcessor()
        return cls._ocr_processor

    @classmethod
    def get_flatcar_processor(cls) -> FlatcarBottomProcessor:
        """Get singleton flatcar bottom processor instance (from run_bottom_merge_ocr)."""
        if cls._flatcar_processor is None:
            cls._flatcar_processor = FlatcarBottomProcessor()
        return cls._flatcar_processor

    @property
    def available(self) -> bool:
        """Check if train ID engine is available."""
        return self.get_ocr_processor().ocr is not None

    @property
    def flatcar_available(self) -> bool:
        """Check if flatcar engine is available."""
        return self.get_flatcar_processor().available

    # ------------------------------------------------------------------
    # Single image recognition
    # ------------------------------------------------------------------

    async def recognize_image(
        self,
        image_bytes: bytes,
        filename: str = "unknown",
    ) -> TrainIDData:
        """
        Recognize vehicle type and number from a JPEG/PNG image.

        直接复用 train_id_ocr_paddle.py 的 PaddleOCRProcessor，
        支持空挡检测(type字段)。
        """
        processor = self.get_ocr_processor()

        if processor.ocr is None:
            logger.error("Train ID engine not available")
            return TrainIDData()

        logger.info(f"Processing train ID image: {filename}, size={len(image_bytes)} bytes")

        result: ImageResult = processor.process_bytes(image_bytes)
        return self._build_train_id_data(result)

    async def recognize_raw_image(
        self,
        image_bytes: bytes,
        pixel_type: int,
        width: int,
        height: int,
        filename: str = "unknown",
    ) -> TrainIDData:
        """
        Recognize vehicle type and number from raw camera pixel data.

        Uses decode_raw_image() to convert Bayer/Mono raw data to BGR.
        """
        processor = self.get_ocr_processor()

        if processor.ocr is None:
            logger.error("Train ID engine not available")
            return TrainIDData()

        logger.info(
            f"Processing raw train ID image: {filename}, "
            f"pixel_type=0x{pixel_type:08X}, size={len(image_bytes)} bytes, "
            f"{width}x{height}"
        )

        result: ImageResult = processor.process_raw_bytes(image_bytes, pixel_type, width, height)
        return self._build_train_id_data(result)

    def _build_train_id_data(self, result: ImageResult) -> TrainIDData:
        """Build TrainIDData from ImageResult."""
        vehicle_type = result.train_types[0][0] if result.train_types else ""
        vehicle_number = result.train_numbers[0][0] if result.train_numbers else ""

        # 集装箱：取置信度最高的一个
        container = result.containers[0][0] if result.containers else ""
        container_conf = result.containers[0][1] if result.containers else 0.0

        # 计算平均置信度（车型+车号）
        confs = []
        if result.train_types:
            confs.append(result.train_types[0][1])
        if result.train_numbers:
            confs.append(result.train_numbers[0][1])
        avg_conf = round(sum(confs) / len(confs), 4) if confs else 0.0

        logger.info(
            f"Train ID result: type='{vehicle_type}' "
            f"number='{vehicle_number}' "
            f"container='{container}' "
            f"gap='{'########' if result.is_gap else ''}' "
            f"confidence={avg_conf:.3f}"
        )

        return TrainIDData(
            type="########" if result.is_gap else "",
            vehicleType=vehicle_type,
            vehicleNumber=vehicle_number,
            confidence=avg_conf,
            container=container,
            containerConfidence=round(container_conf, 4),
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
                container=data.container,
                containerConfidence=data.container_confidence,
            ))
        return results

    # ------------------------------------------------------------------
    # Flatcar single-image recognition (bottom region)
    # ------------------------------------------------------------------

    async def recognize_flatcar_image(
        self,
        image_bytes: bytes,
        filename: str = "unknown",
    ) -> FlatcarData:
        """
        Recognize flatcar type and number from a single image.

        使用 run_bottom_merge_ocr.py 的 FlatcarBottomProcessor，
        底部区域（75%-100% 高度）+ 暗光增强 + 同行框拼接。
        """
        processor = self.get_flatcar_processor()

        if not processor.available:
            logger.error("Flatcar engine not available")
            return FlatcarData()

        logger.info(f"Processing flatcar image: {filename}, size={len(image_bytes)} bytes")

        result = processor.process_bytes(image_bytes)

        logger.info(
            f"Flatcar result: type='{result.get('type', '')}' "
            f"vehicleType='{result['vehicleType']}' "
            f"number='{result['vehicleNumber']}' "
            f"confidence={result['confidence']:.3f}"
        )

        return FlatcarData(
            type=result.get("type", ""),
            vehicleType=result["vehicleType"],
            vehicleNumber=result["vehicleNumber"],
            confidence=result["confidence"],
        )


# Singleton instance
_train_id_service: Optional[TrainIDService] = None


def get_train_id_service_singleton() -> TrainIDService:
    """Get train ID service singleton instance."""
    global _train_id_service
    if _train_id_service is None:
        _train_id_service = TrainIDService()
    return _train_id_service
