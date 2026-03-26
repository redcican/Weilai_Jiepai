"""
OCR Service

Business logic for OCR operations using local CnOCR engine.
Returns raw table data (no column-name mapping).
"""

import logging
from typing import Dict, Any, Optional

from ..ocr import TableOCRProcessor
from ..ocr.models import OCRResult

logger = logging.getLogger(__name__)


class OCRService:
    """
    Service for OCR operations.

    Uses local CnOCR engine for table extraction from railway tickets.
    Returns raw OCRResult with table_type.
    """

    _processor: Optional[TableOCRProcessor] = None

    def __init__(self, enhance_image: bool = True):
        self.enhance_image = enhance_image

    @classmethod
    def get_processor(cls, enhance_image: bool = True) -> TableOCRProcessor:
        """Get singleton processor instance."""
        if cls._processor is None:
            cls._processor = TableOCRProcessor(enhance_image=enhance_image)
        return cls._processor

    @property
    def available(self) -> bool:
        """Check if OCR engine is available."""
        return self.get_processor(self.enhance_image).available

    async def parse_ticket_image(
        self,
        image_bytes: bytes,
        filename: str = "unknown",
    ) -> OCRResult:
        """
        Parse ticket image using local OCR.

        Returns OCRResult with table_type, metadata, and raw table_data.
        """
        processor = self.get_processor(self.enhance_image)

        if not processor.available:
            logger.error("OCR engine not available")
            return OCRResult(
                status="error",
                message="OCR engine not available",
            )

        logger.info(f"Processing ticket image: {filename}, size={len(image_bytes)} bytes")

        try:
            result = processor.process(image_bytes)

            if result.is_success:
                logger.info(
                    f"OCR successful: type={result.table_type}, "
                    f"{len(result.table_data)} rows, "
                    f"{len(result.metadata)} metadata items"
                )

            return result

        except Exception as e:
            logger.exception(f"OCR processing error: {e}")
            return OCRResult(
                status="error",
                message=str(e),
            )


# Singleton instance
_ocr_service: Optional[OCRService] = None


def get_ocr_service() -> OCRService:
    """Get OCR service singleton instance."""
    global _ocr_service
    if _ocr_service is None:
        _ocr_service = OCRService(enhance_image=True)
    return _ocr_service
