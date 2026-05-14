"""
Flatcar (车板号) Single Image Processor

Based on flatcar_processor.py.
Processes a single image's bottom region (75%-100%) to extract flatcar type and number.

Features:
  - Bottom ROI extraction (75%-100% height)
  - Dark scene preprocessing (CLAHE in LAB space)
  - Row-wise box merging for split digit strings
  - Flatcar-specific type correction
  - Single-frame candidate extraction (no temporal aggregation)
"""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .video_engine import PaddleOCREngine

logger = logging.getLogger(__name__)

# ============ Reuse flatcar constants ============
FLATCAR_TYPES = {
    "X70", "X6K", "X2K", "X2H", "X4K", "NX70", "NX17", "NX17B",
    "C70", "C70E", "C80",
}

FLATCAR_CORRECTION = {
    "X7O": "X70", "X7D": "X70", "X7B": "X70", "X70E": "X70",
    "70": "X70", "X": "X70",
    "C7O": "C70", "C7D": "C70", "C7B": "C70", "CO": "C70",
    "C70B": "C70", "C70H": "C70", "C7OE": "C70E",
}

BOTTOM_Y_START = 0.75
BOTTOM_Y_END = 1.0


# ============ Helper functions (mirrored from flatcar_processor.py) ============

def _fix_flatcar_type(text: str) -> Optional[str]:
    if text in FLATCAR_CORRECTION:
        return FLATCAR_CORRECTION[text]
    if text in FLATCAR_TYPES:
        return text
    for t in sorted(FLATCAR_TYPES, key=len, reverse=True):
        if text.startswith(t):
            return t
    if len(text) == 4:
        fixed = list(text)
        for i, c in enumerate(fixed):
            if c == "O" and i >= 1:
                fixed[i] = "0"
            if c == "I" and i >= 1:
                fixed[i] = "1"
            if c == "D" and i >= 1:
                fixed[i] = "0"
        result = "".join(fixed)
        if result in FLATCAR_TYPES:
            return result
    if len(text) >= 3 and text[0].isalpha():
        first = text[0]
        digits = "".join(c for c in text[1:] if c.isdigit())
        letters = "".join(c for c in text[1:] if c.isalpha())
        if len(digits) >= 2:
            candidate = f"{first}{digits[-2:]}{letters[:1]}"
            if candidate in FLATCAR_TYPES:
                return candidate
    return None


def _preprocess_for_dark(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


import re


def _merge_boxes_by_row(texts: List[Tuple[str, float, list]], y_tolerance: float = 50) -> List[Tuple[str, float, list]]:
    """Group boxes by row (Y coordinate), sort by X, merge text."""
    if not texts:
        return []

    def center_y(box):
        return sum(p[1] for p in box) / 4

    def center_x(box):
        return sum(p[0] for p in box) / 4

    sorted_texts = sorted(texts, key=lambda x: center_y(x[2]))
    rows = []
    current_row = [sorted_texts[0]]
    current_y = center_y(sorted_texts[0][2])

    for item in sorted_texts[1:]:
        cy = center_y(item[2])
        if abs(cy - current_y) <= y_tolerance:
            current_row.append(item)
        else:
            current_row.sort(key=lambda x: center_x(x[2]))
            merged_text = "".join(t for t, _, _ in current_row)
            avg_conf = sum(c for _, c, _ in current_row) / len(current_row)
            all_x = [p[0] for item in current_row for p in item[2]]
            all_y = [p[1] for item in current_row for p in item[2]]
            merged_box = [
                [min(all_x), min(all_y)],
                [max(all_x), min(all_y)],
                [max(all_x), max(all_y)],
                [min(all_x), max(all_y)],
            ]
            rows.append((merged_text, avg_conf, merged_box))
            current_row = [item]
            current_y = cy

    if current_row:
        current_row.sort(key=lambda x: center_x(x[2]))
        merged_text = "".join(t for t, _, _ in current_row)
        avg_conf = sum(c for _, c, _ in current_row) / len(current_row)
        all_x = [p[0] for item in current_row for p in item[2]]
        all_y = [p[1] for item in current_row for p in item[2]]
        merged_box = [
            [min(all_x), min(all_y)],
            [max(all_x), min(all_y)],
            [max(all_x), max(all_y)],
            [min(all_x), max(all_y)],
        ]
        rows.append((merged_text, avg_conf, merged_box))

    return rows


def _extract_candidates(merged_rows: List[Tuple[str, float, list]]) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """Extract flatcar type and number candidates from merged rows."""
    type_candidates = []
    num_candidates = []

    for merged_text, conf, box in merged_rows:
        upper = merged_text.upper().replace(" ", "").replace("-", "")

        # Remove railway emblem mis-recognition prefix
        for ft in sorted(FLATCAR_TYPES, key=len, reverse=True):
            if len(upper) > len(ft) + 1 and upper[1:].startswith(ft):
                upper = upper[1:]
                break

        # Extract type
        fixed_type = _fix_flatcar_type(upper)
        if fixed_type:
            type_candidates.append((fixed_type, conf))

        # Filter parameter text with -/. unless type is at start
        if "." in merged_text or "-" in merged_text:
            if not (fixed_type and upper.startswith(fixed_type)):
                continue

        # Extract number from text after type prefix
        text_for_num = upper
        if fixed_type and text_for_num.startswith(fixed_type):
            text_for_num = text_for_num[len(fixed_type):]

        for m in re.finditer(r"(\d{3,8})", text_for_num):
            num = m.group(1)
            if len(num) >= 3:
                num_candidates.append((num, conf))

    return type_candidates, num_candidates


# ============ Single-image processor ============

class FlatcarImageProcessor:
    """Processor for flatcar (车板号) single-image recognition from bottom region."""

    def __init__(self, engine: Optional[PaddleOCREngine] = None):
        # Use Chinese OCR model for flatcar recognition
        self.engine = engine or PaddleOCREngine.get_instance(lang="ch")

    @property
    def available(self) -> bool:
        return self.engine.available

    def process_bytes(self, image_bytes: bytes) -> dict:
        """Process raw image bytes and return flatcar recognition results.

        Returns:
            {
                "types": ["X70", ...],
                "numbers": ["1755648", ...],
            }
        """
        if not self.engine.available:
            logger.error("PaddleOCR (ch) engine not available")
            return {"types": [], "numbers": []}

        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("Failed to decode image bytes")
            return {"types": [], "numbers": []}

        return self._process_img(img)

    def _process_img(self, img: np.ndarray) -> dict:
        h, w = img.shape[:2]

        # Bottom ROI extraction (75%-100%)
        y1 = int(h * BOTTOM_Y_START)
        y2 = int(h * BOTTOM_Y_END)
        bottom_roi = img[y1:y2, :]

        # Dark scene preprocessing
        enhanced = _preprocess_for_dark(bottom_roi)

        # OCR
        ocr_result = self.engine.ocr(enhanced, cls=True)

        # Collect raw detection boxes
        raw_texts = []
        if ocr_result and ocr_result[0]:
            for line in ocr_result[0]:
                if not line:
                    continue
                raw_texts.append((line[1][0], line[1][1], line[0]))

        # Row-wise merge
        merged_rows = _merge_boxes_by_row(raw_texts, y_tolerance=50)

        # Extract candidates
        type_candidates, num_candidates = _extract_candidates(merged_rows)

        # Deduplicate and format
        types = list(dict.fromkeys([t for t, _ in type_candidates]))
        numbers = list(dict.fromkeys([n for n, _ in num_candidates]))

        logger.info(
            f"Flatcar image result: {len(types)} types, {len(numbers)} numbers"
        )

        return {
            "types": types,
            "numbers": numbers,
        }
