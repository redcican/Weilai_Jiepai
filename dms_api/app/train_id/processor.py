"""
Train ID Processor

Orchestrates the multi-pass OCR pipeline: preprocessing → engine → extraction → merging.
Analogous to ocr/processor.py (TableOCRProcessor) but for station-entry camera images.
"""

import re
import logging
from collections import Counter
from typing import List, Optional, Tuple

import numpy as np

from .models import OCRBox, TrainIDResult
from .engine import TrainIDEngine
from .utils import (
    decode_image_bytes,
    resize,
    preprocess_bilateral_clahe,
    preprocess_clahe,
    preprocess_gamma_clahe,
    extract_from_boxes,
)

logger = logging.getLogger(__name__)


class TrainIDProcessor:
    """Processor for extracting train IDs from station-entry camera images.

    Runs a 4-pass preprocessing pipeline with the hybrid engine
    and merges results via majority voting.
    """

    def __init__(self, engine: Optional[TrainIDEngine] = None):
        self.engine = engine or TrainIDEngine.get_instance()

    @property
    def available(self) -> bool:
        return self.engine.available

    def process(self, image: np.ndarray) -> TrainIDResult:
        """Run multi-pass OCR pipeline on a BGR image and return merged result.

        Passes:
        1. Bilateral+CLAHE (resnet) — best general quality, edge-preserving
        2. CLAHE (ppocr) — good baseline, complete type strings
        3. Gamma(2.0)+CLAHE (resnet) — reveals faint text
        4. Gamma(3.0)+CLAHE (resnet) — strongest brightening
        """
        if not self.engine.available:
            logger.error("Train ID engine not available")
            return TrainIDResult()

        img = resize(image)

        passes = [
            ("bilateral_clahe", lambda im: preprocess_bilateral_clahe(im), "resnet"),
            ("clahe_ppocr", lambda im: preprocess_clahe(im), "ppocr"),
            ("gamma2_clahe", lambda im: preprocess_gamma_clahe(im, gamma=2.0), "resnet"),
            ("gamma3_clahe", lambda im: preprocess_gamma_clahe(im, gamma=3.0), "resnet"),
        ]

        results: List[TrainIDResult] = []

        for label, preprocessor, det_engine in passes:
            enhanced = preprocessor(img.copy())
            boxes = self.engine.recognize(enhanced, engine=det_engine)
            tid = extract_from_boxes(boxes)
            results.append(tid)
            logger.debug(
                f"  {label}: type='{tid.vehicle_type}' num='{tid.vehicle_number}'"
            )

        if not any(r.vehicle_type or r.vehicle_number for r in results):
            return TrainIDResult()

        best_type = _pick_best_type(results)
        best_number = _pick_best_number(results)

        confs = [r.confidence for r in results if r.confidence > 0]
        avg_conf = sum(confs) / len(confs) if confs else 0.0

        return TrainIDResult(
            vehicle_type=best_type,
            vehicle_number=best_number,
            confidence=avg_conf,
        )

    def process_bytes(self, image_bytes: bytes) -> TrainIDResult:
        """Recognize train ID from raw image bytes."""
        img = decode_image_bytes(image_bytes)
        if img is None:
            logger.warning("Failed to decode image bytes")
            return TrainIDResult()
        return self.process(img)


# ---------------------------------------------------------------------------
# Multi-pass merging (module-level helpers)
# ---------------------------------------------------------------------------

def _pick_best_type(candidates: List[TrainIDResult]) -> str:
    """Choose best vehicle type via majority voting.

    On ties, prefer later passes (gamma-enhanced) which tend to
    produce better character contrast for letter/digit distinction.
    """
    types = [c.vehicle_type for c in candidates if c.vehicle_type]
    if not types:
        return ""

    counts = Counter(types)
    top_count = counts.most_common(1)[0][1]

    if top_count >= 2:
        for t in reversed(types):
            if counts[t] == top_count:
                return t

    # No majority — prefer last non-empty type (from gamma passes)
    return types[-1]


def _pick_best_number(candidates: List[TrainIDResult]) -> str:
    """Choose best vehicle number using superset-aware majority voting.

    Strategy:
    1. Find the most common digit string.
    2. Before returning majority, check if a superset exists (same digits
       plus trailing edge digits found by gamma passes) — prefer longer.
    3. If majority (count >= 2), use it.
    4. Fallback to first pass with superset check.
    """
    if not candidates:
        return ""

    digit_map: List[Tuple[str, str]] = []
    for c in candidates:
        digits = re.sub(r"\D", "", c.vehicle_number)
        if digits:
            digit_map.append((digits, c.vehicle_number))

    if not digit_map:
        return ""

    digit_counts = Counter(d for d, _ in digit_map)
    most_common_digits, count = digit_counts.most_common(1)[0]

    # Always check for superset of the most common digits first
    for digits, full in digit_map:
        if len(digits) > len(most_common_digits) and digits.startswith(most_common_digits):
            return full

    # Use majority if count >= 2
    if count >= 2:
        for digits, full in digit_map:
            if digits == most_common_digits:
                return full

    # Fallback: prefer first pass, with superset check
    primary = candidates[0].vehicle_number
    primary_digits = re.sub(r"\D", "", primary)

    if not primary_digits:
        return digit_map[0][1] if digit_map else ""

    for digits, full in digit_map[1:]:
        if len(digits) > len(primary_digits) and digits.startswith(primary_digits):
            return full

    return primary
