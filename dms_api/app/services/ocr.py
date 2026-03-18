"""
OCR Service

Business logic for OCR operations using local CnOCR engine.
"""

import logging
from typing import Dict, Any, Optional

from ..ocr import TableOCRProcessor, CnOCREngine
from ..schemas.ticket import TicketData

logger = logging.getLogger(__name__)


class OCRService:
    """
    Service for OCR operations.

    Uses local CnOCR engine for table extraction from railway tickets.
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
    ) -> Dict[str, Any]:
        """
        Parse ticket image using local OCR.

        Args:
            image_bytes: Image file content
            filename: Original filename (for logging)

        Returns:
            Dictionary with parsed ticket data
        """
        processor = self.get_processor(self.enhance_image)

        if not processor.available:
            logger.error("OCR engine not available")
            return {
                "success": False,
                "error": "OCR engine not available",
            }

        logger.info(f"Processing ticket image: {filename}, size={len(image_bytes)} bytes")

        try:
            result = processor.process(image_bytes)

            if not result.is_success:
                logger.warning(f"OCR processing failed: {result.message}")
                return {
                    "success": False,
                    "error": result.message,
                }

            # Map to ticket data format (list of dicts, one per row)
            ticket_list = self._map_ocr_to_ticket(result.metadata, result.table_data)

            logger.info(
                f"OCR successful: {len(result.table_data)} rows, "
                f"{len(result.metadata)} metadata items"
            )

            return {
                "success": True,
                "data": ticket_list,
                "raw_table": result.table_data,
                "metadata": result.metadata,
            }

        except Exception as e:
            logger.exception(f"OCR processing error: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def _map_ocr_to_ticket(
        self,
        metadata: Dict[str, str],
        table_data: list[list[str]],
    ) -> list[Dict[str, Any]]:
        """
        Map OCR results to a list of TicketData schema fields (one dict per row).

        Column mapping (based on typical 编组单 format):
        - 0: 序号 (seq)
        - 1: 车型 (trainType)
        - 2: 车号 (trainNo)
        - 3: 自重 (emptyCapacity)
        - 4: 换长 (changeLength)
        - 5: 载重 (loadCapacity)
        - 6: 集装箱1 (container1)
        - 7: 集装箱2 (container2)
        - 8: 发站 (startStation)
        - 9: 到站 (destStation)
        - 10: 品名 (carryType)
        - 11: 记事 (descr)
        """
        column_mapping = [
            ("seq", 0),
            ("trainType", 1),
            ("trainNo", 2),
            ("emptyCapacity", 3),
            ("changeLength", 4),
            ("loadCapacity", 5),
            ("container1", 6),
            ("container2", 7),
            ("startStation", 8),
            ("destStation", 9),
            ("carryType", 10),
            ("descr", 11),
        ]

        # Shared metadata fields
        meta_base: Dict[str, Any] = {}
        if metadata:
            if v := metadata.get("计划序号") or metadata.get("计划号"):
                meta_base["planNo"] = v
            if v := metadata.get("股道"):
                meta_base["stock"] = v
            if v := metadata.get("票据号") or metadata.get("票号"):
                meta_base["ticketNo"] = v

        tickets = []
        for row in table_data:
            ticket = dict(meta_base)
            for field_name, col_idx in column_mapping:
                if col_idx < len(row):
                    value = row[col_idx].strip() if row[col_idx] else None
                    if value:
                        ticket[field_name] = value
            tickets.append({k: v for k, v in ticket.items() if v})

        return tickets

    def create_ticket_data(self, parsed_data: Dict[str, Any]) -> Optional[list[TicketData]]:
        """
        Create a list of TicketData from parsed OCR result (one per table row).

        Args:
            parsed_data: Dictionary from parse_ticket_image

        Returns:
            List of TicketData instances, or None if parsing failed
        """
        if not parsed_data.get("success"):
            return None

        rows = parsed_data.get("data", [])
        if not rows:
            return None

        result = []
        for row_dict in rows:
            try:
                result.append(TicketData.model_validate(row_dict))
            except Exception as e:
                logger.warning(f"Failed to create TicketData for row: {e}")

        return result if result else None


# Singleton instance
_ocr_service: Optional[OCRService] = None


def get_ocr_service() -> OCRService:
    """Get OCR service singleton instance."""
    global _ocr_service
    if _ocr_service is None:
        _ocr_service = OCRService(enhance_image=True)
    return _ocr_service
