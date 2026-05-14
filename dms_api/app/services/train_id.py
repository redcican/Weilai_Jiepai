"""
Train ID Service

Business logic for train identification recognition operations.
"""

import logging
from typing import Optional

from ..train_id import TrainIDProcessor, TrainIDResult
from ..train_id import VideoTrainIDProcessor, VideoRecognitionResult
from ..train_id import FlatcarVideoProcessor, FlatcarRecognitionResult
from ..schemas.train_id import (
    TrainIDData,
    TrainIDBatchItem,
    VideoTrainIDData,
    FlatcarVideoData,
    FlatcarItem,
)

logger = logging.getLogger(__name__)


class TrainIDService:
    """
    Service for train ID recognition operations.

    Uses local CnOCR engine with hybrid detection models for images,
    and PaddleOCR for video-based recognition.
    Does not require DMS backend — all processing is local.
    """

    _processor: Optional[TrainIDProcessor] = None
    _video_processor: Optional[VideoTrainIDProcessor] = None
    _flatcar_processor: Optional[FlatcarVideoProcessor] = None

    @classmethod
    def get_processor(cls) -> TrainIDProcessor:
        """Get singleton processor instance."""
        if cls._processor is None:
            cls._processor = TrainIDProcessor()
        return cls._processor

    @classmethod
    def get_video_processor(cls) -> VideoTrainIDProcessor:
        """Get singleton video processor instance."""
        if cls._video_processor is None:
            cls._video_processor = VideoTrainIDProcessor()
        return cls._video_processor

    @classmethod
    def get_flatcar_processor(cls) -> FlatcarVideoProcessor:
        """Get singleton flatcar processor instance."""
        if cls._flatcar_processor is None:
            cls._flatcar_processor = FlatcarVideoProcessor()
        return cls._flatcar_processor

    @property
    def available(self) -> bool:
        """Check if train ID engine is available."""
        return self.get_processor().available

    @property
    def video_available(self) -> bool:
        """Check if video recognition engine is available."""
        return self.get_video_processor().available

    @property
    def flatcar_available(self) -> bool:
        """Check if flatcar recognition engine is available."""
        return self.get_flatcar_processor().available

    # ------------------------------------------------------------------
    # Image recognition (existing)
    # ------------------------------------------------------------------

    async def recognize_image(
        self,
        image_bytes: bytes,
        filename: str = "unknown",
    ) -> TrainIDData:
        """
        Recognize vehicle type and number from a station-entry camera image.
        """
        processor = self.get_processor()

        if not processor.available:
            logger.error("Train ID engine not available")
            return TrainIDData()

        logger.info(f"Processing train ID image: {filename}, size={len(image_bytes)} bytes")

        result = processor.process_bytes(image_bytes)

        logger.info(
            f"Train ID result: type='{result.vehicle_type}' "
            f"number='{result.vehicle_number}' confidence={result.confidence:.3f}"
        )

        return TrainIDData(
            vehicleType=result.vehicle_type,
            vehicleNumber=result.vehicle_number,
            confidence=round(result.confidence, 4),
        )

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
                vehicleType=data.vehicle_type,
                vehicleNumber=data.vehicle_number,
                confidence=data.confidence,
            ))
        return results

    # ------------------------------------------------------------------
    # Video recognition (container + train)
    # ------------------------------------------------------------------

    async def recognize_video(
        self,
        video_bytes: bytes,
        filename: str = "unknown",
        interval_sec: float = 0.5,
        gap_sec: float = 3.0,
    ) -> VideoTrainIDData:
        """
        Recognize container IDs and train IDs from a video.
        """
        processor = self.get_video_processor()

        if not processor.available:
            logger.error("Video train ID engine (PaddleOCR) not available")
            return VideoTrainIDData()

        logger.info(
            f"Processing video: {filename}, size={len(video_bytes)} bytes, "
            f"interval={interval_sec}s, gap={gap_sec}s"
        )

        result = processor.process_video_bytes(
            video_bytes=video_bytes,
            filename=filename,
            interval_sec=interval_sec,
            gap_sec=gap_sec,
        )

        logger.info(
            f"Video result: {result.container_count} containers, "
            f"{result.train_type_count} train types, "
            f"{result.train_number_count} train numbers, "
            f"{result.frames_processed} frames processed"
        )

        return VideoTrainIDData(
            containers=result.containers,
            trainTypes=result.train_types,
            trainNumbers=result.train_numbers,
            containerCount=result.container_count,
            trainTypeCount=result.train_type_count,
            trainNumberCount=result.train_number_count,
            framesProcessed=result.frames_processed,
            durationSec=result.duration_sec,
        )

    # ------------------------------------------------------------------
    # Flatcar (车板号) video recognition
    # ------------------------------------------------------------------

    async def recognize_flatcar_video(
        self,
        video_bytes: bytes,
        filename: str = "unknown",
        interval_sec: float = 0.05,
        gap_sec: float = 0.15,
    ) -> FlatcarVideoData:
        """
        Recognize flatcar (车板号) type and number from a video.

        Processes only the bottom 75%-100% region of each frame.
        Uses Chinese PaddleOCR with row-wise box merging.

        Args:
            video_bytes: Raw video file content
            filename: Original filename
            interval_sec: Frame extraction interval in seconds (default 0.05)
            gap_sec: Temporal aggregation gap in seconds (default 0.15)

        Returns:
            FlatcarVideoData with type+number pairs
        """
        processor = self.get_flatcar_processor()

        if not processor.available:
            logger.error("Flatcar recognition engine (PaddleOCR ch) not available")
            return FlatcarVideoData()

        logger.info(
            f"Processing flatcar video: {filename}, size={len(video_bytes)} bytes, "
            f"interval={interval_sec}s, gap={gap_sec}s"
        )

        result = processor.process_video_bytes(
            video_bytes=video_bytes,
            filename=filename,
            interval_sec=interval_sec,
            gap_sec=gap_sec,
        )

        logger.info(
            f"Flatcar result: {result.type_count} types, "
            f"{result.number_count} numbers, "
            f"{len(result.results)} merged entries, "
            f"{result.frames_processed} frames processed"
        )

        return FlatcarVideoData(
            results=[
                FlatcarItem(
                    type=r.get("type"),
                    number=r.get("number"),
                    frames=r.get("frames", 0),
                    avgConf=r.get("avg_conf", 0.0),
                )
                for r in result.results
            ],
            typeCount=result.type_count,
            numberCount=result.number_count,
            framesProcessed=result.frames_processed,
            durationSec=result.duration_sec,
        )


# Singleton instance
_train_id_service: Optional[TrainIDService] = None


def get_train_id_service_singleton() -> TrainIDService:
    """Get train ID service singleton instance."""
    global _train_id_service
    if _train_id_service is None:
        _train_id_service = TrainIDService()
    return _train_id_service
