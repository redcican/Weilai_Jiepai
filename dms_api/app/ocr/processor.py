"""
Table OCR Processor

Extracts structured table data from railway ticket images.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Union

from .models import OCRBox, OCRResult
from .engine import CnOCREngine
from .utils import (
    aggregate_to_rows,
    is_metadata_item,
    is_header_row,
    is_valid_data_row,
    is_potential_sequence_number,
    detect_sequence_column,
    detect_vehicle_type_column,
    detect_vehicle_id_column,
    normalize_row,
    extract_track_number_from_first_row,
)

logger = logging.getLogger(__name__)


class TableOCRProcessor:
    """
    Processor for extracting table data from railway ticket images.

    Supports both file paths and image bytes as input.
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

        Args:
            image: Image file path or image bytes
            tolerance: Row aggregation tolerance (auto-adaptive if None)

        Returns:
            OCRResult with extracted metadata and table data
        """
        if not self.engine.available:
            return OCRResult(
                status="error",
                message="OCR engine not available",
            )

        # Perform OCR
        ocr_results = self.engine.recognize(image)

        if not ocr_results:
            return OCRResult(
                status="error",
                message="OCR recognition failed or no text detected",
            )

        # Aggregate results into rows
        all_rows = aggregate_to_rows(ocr_results, tolerance)

        # Separate metadata and candidate data rows
        metadata = {}
        candidate_rows = []

        for row in all_rows:
            is_meta, key, value = is_metadata_item(row)
            if is_meta and key and value:
                metadata[key] = value
                continue

            if is_header_row(row):
                continue

            if len(row) >= 3:
                candidate_rows.append(row)

        if not candidate_rows:
            return OCRResult(
                status="success",
                message="Recognition successful, no table data found",
                metadata=metadata,
                table_data=[],
            )

        # Auto-detect table structure
        seq_col = detect_sequence_column(candidate_rows)
        vehicle_type_col = detect_vehicle_type_column(candidate_rows)
        vehicle_id_col = detect_vehicle_id_column(candidate_rows)

        logger.debug(f"Detection: seq_col={seq_col}, vehicle_type_col={vehicle_type_col}, vehicle_id_col={vehicle_id_col}")

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

        return OCRResult(
            status="success",
            message="Recognition successful",
            metadata=metadata,
            table_data=data_rows,
        )

    def process_to_ticket_data(
        self,
        image: Union[str, bytes],
        tolerance: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process image and convert to ticket data format.

        Returns a dictionary matching the TicketData schema.
        """
        result = self.process(image, tolerance)

        if not result.is_success:
            return {
                "error": result.message,
                "status": result.status,
            }

        # Map OCR result to TicketData fields
        ticket_data = self._map_to_ticket_fields(result)
        return ticket_data

    def _map_to_ticket_fields(self, result: OCRResult) -> Dict[str, Any]:
        """
        Map OCR result to TicketData schema fields.

        Field mapping from table columns:
        - Column 0: 序号 (seq)
        - Column 1: 车型 (trainType)
        - Column 2: 车号 (trainNo)
        - Column 3: 自重 (emptyCapacity)
        - Column 4: 换长 (changeLength)
        - Column 5: 载重 (loadCapacity)
        - Column 6: 集装箱1 (container1)
        - Column 7: 集装箱2 (container2)
        - Column 8: 发站 (startStation)
        - Column 9: 到站 (destStation)
        - Column 10: 品名 (carryType)
        - Column 11: 记事 (descr)
        """
        metadata = result.metadata
        table_data = result.table_data

        # Start with metadata
        ticket = {
            "planNo": metadata.get("计划序号") or metadata.get("计划号"),
            "stock": metadata.get("股道"),
            "ticketNo": metadata.get("票据号") or metadata.get("票号"),
        }

        # If we have table data, extract the first row as primary data
        if table_data and len(table_data) > 0:
            first_row = table_data[0]

            # Map columns to fields based on position
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

            for field_name, col_idx in column_mapping:
                if col_idx < len(first_row):
                    value = first_row[col_idx].strip()
                    if value:
                        ticket[field_name] = value

        # Clean up None values
        ticket = {k: v for k, v in ticket.items() if v is not None}

        return ticket
