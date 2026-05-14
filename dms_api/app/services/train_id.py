"""
Train ID Service

Business logic for train identification recognition operations.
"""

import logging
from typing import Optional

from ..train_id import TrainIDProcessor, TrainIDResult
from ..train_id import PaddleImageProcessor, FlatcarImageProcessor
from ..schemas.train_id import TrainIDData, TrainIDBatchItem, PaddleImageData, FlatcarImageData

logger = logging.getLogger(__name__)


class TrainIDService:
    """
    Service for train ID recognition operations.

    Uses local CnOCR engine with hybrid detection models for images,
    and PaddleOCR for single-image container + train ID recognition.
    Does not require DMS backend — all processing is local.
    """

    _processor: Optional[TrainIDProcessor] = None
    _paddle_processor: Optional[PaddleImageProcessor] = None
    _flatcar_processor: Optional[FlatcarImageProcessor] = None

    @classmethod
    def get_processor(cls) -> TrainIDProcessor:
        """Get singleton processor instance."""
        if cls._processor is None:
            cls._processor = TrainIDProcessor()
        return cls._processor

    @classmethod
    def get_paddle_processor(cls) -> PaddleImageProcessor:
        """Get singleton PaddleOCR image processor instance."""
        if cls._paddle_processor is None:
            cls._paddle_processor = PaddleImageProcessor()
        return cls._paddle_processor

    @classmethod
    def get_flatcar_processor(cls) -> FlatcarImageProcessor:
        """Get singleton flatcar image processor instance."""
        if cls._flatcar_processor is None:
            cls._flatcar_processor = FlatcarImageProcessor()
        return cls._flatcar_processor

    @property
    def available(self) -> bool:
        """Check if train ID engine is available."""
        return self.get_processor().available

    @property
    def paddle_available(self) -> bool:
        """Check if PaddleOCR image engine is available."""
        return self.get_paddle_processor().available

    @property
    def flatcar_available(self) -> bool:
        """Check if flatcar image engine is available."""
        return self.get_flatcar_processor().available

    # ------------------------------------------------------------------
    # Image recognition (CnOCR, existing)
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
    # PaddleOCR single-image recognition (container + train)
    # ------------------------------------------------------------------

    async def recognize_paddle_image(
        self,
        image_bytes: bytes,
        filename: str = "unknown",
    ) -> PaddleImageData:
        """
        Recognize container IDs and train IDs from a single image using PaddleOCR.

        Uses upper/lower split strategy:
        - Upper half: container IDs
        - Lower half: train vehicle types and numbers

        Args:
            image_bytes: Raw image file content
            filename: Original filename

        Returns:
            PaddleImageData with containers, train_types, train_numbers
        """
        processor = self.get_paddle_processor()

        if not processor.available:
            logger.error("PaddleOCR image engine not available")
            return PaddleImageData()

        logger.info(f"Processing PaddleOCR image: {filename}, size={len(image_bytes)} bytes")

        result = processor.process_bytes(image_bytes)

        logger.info(
            f"PaddleOCR result: {len(result['containers'])} containers, "
            f"{len(result['train_types'])} train types, "
            f"{len(result['train_numbers'])} train numbers"
        )

        return PaddleImageData(
            containers=result["containers"],
            trainTypes=result["train_types"],
            trainNumbers=result["train_numbers"],
        )

    # ------------------------------------------------------------------
    # Flatcar single-image recognition (bottom region)
    # ------------------------------------------------------------------

    async def recognize_flatcar_image(
        self,
        image_bytes: bytes,
        filename: str = "unknown",
    ) -> FlatcarImageData:
        """
        Recognize flatcar type and number from a single image using PaddleOCR (ch).

        Uses bottom region extraction (75%-100% height) with dark scene preprocessing
        and row-wise box merging for split digit strings.

        Args:
            image_bytes: Raw image file content
            filename: Original filename

        Returns:
            FlatcarImageData with types and numbers
        """
        processor = self.get_flatcar_processor()

        if not processor.available:
            logger.error("Flatcar image engine not available")
            return FlatcarImageData()

        logger.info(f"Processing flatcar image: {filename}, size={len(image_bytes)} bytes")

        result = processor.process_bytes(image_bytes)

        logger.info(
            f"Flatcar result: {len(result['types'])} types, "
            f"{len(result['numbers'])} numbers"
        )

        return FlatcarImageData(
            types=result["types"],
            numbers=result["numbers"],
        )


# Singleton instance
_train_id_service: Optional[TrainIDService] = None


def get_train_id_service_singleton() -> TrainIDService:
    """Get train ID service singleton instance."""
    global _train_id_service
    if _train_id_service is None:
        _train_id_service = TrainIDService()
    return _train_id_service
