"""
Train ID Service

Business logic for train identification recognition operations.
直接复用 train_id_ocr_paddle.py 的 PaddleOCRProcessor / PaddleOCRProcessPool。
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

# 把项目根目录加入路径，以便导入 train_id_ocr_paddle
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from train_id_ocr.train_id_ocr_paddle import (
    get_ocr_processor,
    PaddleOCRProcessor,
    ImageResult,
    FrameFilterConfig,
)
from train_id_ocr.run_bottom_merge_ocr import FlatcarBottomProcessor

from ..schemas.train_id import TrainIDData, TrainIDBatchItem, FlatcarData

logger = logging.getLogger(__name__)


class TrainIDService:
    """
    Service for train ID recognition operations.

    4摄像头并发场景：使用多进程 OCR 工作池（num_workers=4），
    每个工作进程有独立 PaddleOCR 实例，绕过 Python GIL 限制。
    """

    _ocr_processor: Optional[PaddleOCRProcessor] = None
    _flatcar_processor: Optional[FlatcarBottomProcessor] = None
    _thread_pool: Optional[asyncio.AbstractEventLoop] = None

    @classmethod
    def get_ocr_processor(cls):
        """Get OCR processor instance (multi-process pool for concurrent cameras)."""
        if cls._ocr_processor is None:
            # 自动决策：GPU可用→单进程GPU（最快），无GPU→4进程CPU（保底）
            # 4摄像头并发场景下，GPU单进程已能满足3fps，CPU才需要多进程兜底
            cls._ocr_processor = get_ocr_processor(
                frame_filter_config=FrameFilterConfig(
                    min_interval_sec=0.15,   # 150ms内重复帧跳过
                    cache_ttl_sec=3.0,
                    enable_cache=True,
                ),
            )
        return cls._ocr_processor

    @classmethod
    def get_flatcar_processor(cls) -> FlatcarBottomProcessor:
        """Get singleton flatcar bottom processor instance."""
        if cls._flatcar_processor is None:
            cls._flatcar_processor = FlatcarBottomProcessor()
        return cls._flatcar_processor

    @property
    def available(self) -> bool:
        """Check if train ID engine is available."""
        return self.get_ocr_processor().available

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
        pixel_type: int | None = None,
        width: int | None = None,
        height: int | None = None,
        cam_id: Optional[str] = None,
    ) -> TrainIDData:
        """
        Recognize vehicle type and number from an image.

        Supports both JPEG/PNG and raw camera pixel data (Bayer/Mono).
        Multi-process: OCR inference runs in a worker process pool,
        wrapped with run_in_executor to avoid blocking the event loop.
        """
        processor = self.get_ocr_processor()

        if not processor.available:
            logger.error("Train ID engine not available")
            return TrainIDData()

        loop = asyncio.get_event_loop()

        if pixel_type is not None and width is not None and height is not None:
            logger.info(
                f"Processing raw train ID image: {filename}, "
                f"pixel_type=0x{pixel_type:08X}, size={len(image_bytes)} bytes, "
                f"{width}x{height}"
            )
            # 异步包装：在线程池中执行同步的进程池调用
            result: ImageResult = await loop.run_in_executor(
                None,
                processor.process_raw_bytes,
                image_bytes, pixel_type, width, height,
                cam_id,
            )
        else:
            logger.info(f"Processing train ID image: {filename}, size={len(image_bytes)} bytes")
            result: ImageResult = await loop.run_in_executor(
                None,
                processor.process_bytes,
                image_bytes,
                cam_id,
            )

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
        cam_id: Optional[str] = None,
    ) -> list[TrainIDBatchItem]:
        """
        Recognize vehicle info from multiple images.
        使用 asyncio.gather 并行处理批量请求。
        """
        tasks = [
            self.recognize_image(image_bytes, filename, cam_id=cam_id)
            for image_bytes, filename in images
        ]
        datas = await asyncio.gather(*tasks)

        results = []
        for (image_bytes, filename), data in zip(images, datas):
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
        pixel_type: int | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> FlatcarData:
        """
        Recognize flatcar type and number from an image.
        """
        processor = self.get_flatcar_processor()

        if not processor.available:
            logger.error("Flatcar engine not available")
            return FlatcarData()

        loop = asyncio.get_event_loop()

        if pixel_type is not None and width is not None and height is not None:
            logger.info(
                f"Processing raw flatcar image: {filename}, "
                f"pixel_type=0x{pixel_type:08X}, size={len(image_bytes)} bytes, "
                f"{width}x{height}"
            )
            result = await loop.run_in_executor(
                None,
                processor.process_raw_bytes,
                image_bytes, pixel_type, width, height,
            )
        else:
            logger.info(f"Processing flatcar image: {filename}, size={len(image_bytes)} bytes")
            result = await loop.run_in_executor(
                None,
                processor.process_bytes,
                image_bytes,
            )

        gap_info = ""
        if result.get("gapFeatures"):
            gf = result["gapFeatures"]
            gap_info = (
                f" gap_score={result.get('gapScore', 0)}/8"
                f" mb={gf.get('mean_brightness', 0):.0f}"
                f" std={gf.get('std_brightness', 0):.0f}"
                f" edge={gf.get('edge_score', 0):.0f}"
                f" pv={gf.get('profile_variance', 0):.0f}"
                f" pm={gf.get('profile_min', 0):.0f}"
                f" asy={gf.get('asymmetry', 0):.2f}"
                f" cv={gf.get('col_variance', 0):.0f}"
                f" cr={gf.get('col_max', 0) - gf.get('col_min', 0):.0f}"
            )
        logger.info(
            f"Flatcar result: type='{result.get('type', '')}' "
            f"vehicleType='{result['vehicleType']}' "
            f"number='{result['vehicleNumber']}' "
            f"confidence={result['confidence']:.3f}{gap_info}"
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
