"""
Table OCR Processor

Extracts structured table data from railway ticket images.
Supports two table types:
  Type 1 — 站存车打印 (16-column dict output with column names)
  Type 2 — 集装箱编组单 (slash vehicle/number, container numbers)
"""

import logging
from typing import Optional, Union

from .models import OCRResult
from .engine import CnOCREngine
from .utils import (
    aggregate_to_rows,
    detect_table_type,
    extract_type2,
    extract_type1_columns,
)

logger = logging.getLogger(__name__)


class TableOCRProcessor:
    """
    Processor for extracting table data from railway ticket images.

    Supports both file paths and image bytes as input.
    Auto-detects table type and applies type-specific extraction.
    """

    def __init__(self, engine: Optional[CnOCREngine] = None, enhance_image: bool = True):
        self.engine = engine or CnOCREngine.get_instance(enhance_image=enhance_image)

    @property
    def available(self) -> bool:
        return self.engine.available

    def process(
        self,
        image: Union[str, bytes],
        tolerance: Optional[int] = None
    ) -> OCRResult:
        """
        Extract table data from an image.

        Auto-detects table type:
        - Type 1: returns list of 16-key dicts (column names as keys)
        - Type 2: returns list of raw arrays

        Returns:
            OCRResult with table_type, metadata, and table_data
        """
        if not self.engine.available:
            return OCRResult(
                status="error",
                message="OCR engine not available",
            )

        ocr_results = self.engine.recognize(image)

        if not ocr_results:
            return OCRResult(
                status="error",
                message="OCR recognition failed or no text detected",
            )

        table_type = detect_table_type(ocr_results)
        logger.info(f"Detected table type: {table_type}")

        # ── Type 2: 集装箱编组单 ──
        if table_type == 2:
            all_rows = aggregate_to_rows(ocr_results, tolerance)
            metadata, data_rows = extract_type2(all_rows)
            return OCRResult(
                status="success",
                message="识别成功",
                table_type=2,
                metadata=metadata,
                table_data=data_rows,
            )

        # ── Type 1: 站存车打印 (16-column dict) ──
        metadata, row_dicts = extract_type1_columns(ocr_results, tolerance)

        return OCRResult(
            status="success",
            message="识别成功",
            table_type=1,
            metadata=metadata,
            table_data=row_dicts,
        )
