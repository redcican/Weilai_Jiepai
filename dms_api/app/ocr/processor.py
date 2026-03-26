"""
Table OCR Processor

Extracts structured table data from railway ticket images.
Supports two table types:
  Type 1 — 站存车打印 (vehicle type/number in separate columns)
  Type 2 — 集装箱编组单 (slash vehicle/number, container numbers)
"""

import re
import logging
from typing import Optional, Union, List, Any

from .models import OCRResult
from .engine import CnOCREngine
from .utils import (
    aggregate_to_rows,
    is_metadata_item,
    is_header_row,
    is_page_footer,
    is_valid_data_row,
    is_potential_sequence_number,
    detect_sequence_column,
    detect_vehicle_type_column,
    detect_vehicle_id_column,
    normalize_row,
    extract_track_number_from_first_row,
    detect_table_type,
    extract_type2,
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

        Auto-detects table type and returns raw table data arrays
        (no column-name mapping).

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

        all_rows = aggregate_to_rows(ocr_results, tolerance)

        # ── Type 2: 集装箱编组单 ──
        if table_type == 2:
            metadata, data_rows = extract_type2(all_rows)
            return OCRResult(
                status="success",
                message="识别成功",
                table_type=2,
                metadata=metadata,
                table_data=data_rows,
            )

        # ── Type 1: 站存车打印 ──
        return self._extract_type1(all_rows)

    def _extract_type1(self, all_rows: List[List[str]]) -> OCRResult:
        """Type 1 extraction: 站存车打印 tables."""
        metadata = {}
        candidate_rows = []

        for row in all_rows:
            is_meta, key, value = is_metadata_item(row)
            if is_meta and key and value:
                metadata[key] = value
                continue

            if is_header_row(row):
                continue

            if is_page_footer(row):
                continue

            if len(row) >= 3:
                candidate_rows.append(row)

        if not candidate_rows:
            return OCRResult(
                status="success",
                message="识别成功",
                table_type=1,
                metadata=metadata,
                table_data=[],
            )

        # Auto-detect table structure
        seq_col = detect_sequence_column(candidate_rows)
        vehicle_type_col = detect_vehicle_type_column(candidate_rows)
        vehicle_id_col = detect_vehicle_id_column(candidate_rows)

        logger.debug(
            f"Detection: seq_col={seq_col}, "
            f"vehicle_type_col={vehicle_type_col}, "
            f"vehicle_id_col={vehicle_id_col}"
        )

        # Filter valid data rows
        data_rows = []
        has_sequence_numbers = seq_col >= 0

        if has_sequence_numbers:
            for row in candidate_rows:
                if is_valid_data_row(row):
                    if seq_col < len(row) and is_potential_sequence_number(row[seq_col]):
                        data_rows.append(row)

            # Extract track number
            track_num, data_rows = extract_track_number_from_first_row(data_rows, seq_col)
            if track_num:
                metadata["股道"] = track_num
                if seq_col > 0:
                    seq_col -= 1
                    if vehicle_type_col > 0:
                        vehicle_type_col -= 1

            logger.debug(f"Sequence column detected, keeping {len(data_rows)} rows")

        else:
            for row in candidate_rows:
                if is_valid_data_row(row):
                    has_vehicle = False
                    for i in range(min(3, len(row))):
                        val = row[i].strip()
                        if re.match(r'^[A-Za-z]+\d+', val):
                            has_vehicle = True
                            break
                        if re.match(r'^\d{6,8}$', val):
                            has_vehicle = True
                            break
                        if re.match(r'^\d{2,3}[A-Za-z]*$', val):
                            has_vehicle = True
                            break

                    if has_vehicle:
                        data_rows.append(row)

            logger.debug(f"No sequence column, using vehicle detection, keeping {len(data_rows)} rows")

        # Normalize data rows
        vehicle_col = vehicle_type_col if vehicle_type_col >= 0 else -1
        data_rows = [normalize_row(row, vehicle_col) for row in data_rows]

        # Auto-generate sequence numbers if not present
        if not has_sequence_numbers and data_rows:
            logger.debug("Auto-generating sequence numbers")
            data_rows = [[str(i + 1)] + row for i, row in enumerate(data_rows)]

        # Type 1: 股道 as first element of table_data
        table_data: List[Any] = []
        track_value = metadata.pop("股道", None)
        if track_value:
            table_data.append({"股道": track_value})
        table_data.extend(data_rows)

        return OCRResult(
            status="success",
            message="识别成功",
            table_type=1,
            metadata=metadata,
            table_data=table_data,
        )
