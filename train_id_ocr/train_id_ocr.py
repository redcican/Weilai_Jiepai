#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train ID OCR — Railway Car Identification Recognition

Extracts vehicle type (车种) and vehicle number (车号) from
station-entry camera images using CnOCR with multi-pass preprocessing.

Pipeline:
  1. Load image and resize (0.25 scale: 4096x3000 → 1024x750)
  2. Run 4 preprocessing variants (bilateral+CLAHE, CLAHE, gamma+CLAHE x2)
  3. Hybrid OCR (db_resnet18 for digits, ch_PP-OCRv3_det for types)
  4. Noise filtering → line grouping → type/number extraction
  5. Merge results via majority voting (type) and superset-aware voting (number)

Usage:
    python train_id_ocr.py /path/to/image/folder -o ./output
"""

import json
import re
import sys
import logging
import argparse
import time
from collections import Counter
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
from cnocr import CnOcr

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
RESIZE_SCALE = 0.25


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class OCRBox:
    """Single OCR detection."""
    box: List[int]      # [x_min, y_min, x_max, y_max]
    text: str
    confidence: float = 0.0


@dataclass
class TrainID:
    """Extracted train identification."""
    vehicle_type: str = ""        # e.g. C64K, C70E, C70
    vehicle_number: str = ""      # e.g. 49 31846
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Image loading and preprocessing
# ---------------------------------------------------------------------------

def _load_and_resize(image_path: str, scale: float = RESIZE_SCALE) -> Optional[np.ndarray]:
    """Load image from disk and resize by scale factor."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    nw, nh = int(w * scale), int(h * scale)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def _apply_clahe(img: np.ndarray, clip_limit: float = 3.0) -> np.ndarray:
    """Apply CLAHE on LAB L-channel and return RGB image."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def preprocess_bilateral_clahe(img: np.ndarray) -> np.ndarray:
    """Bilateral filter (edge-preserving) + CLAHE — best general quality."""
    filtered = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    return _apply_clahe(filtered)


def preprocess_clahe(img: np.ndarray) -> np.ndarray:
    """CLAHE only — good baseline, detects edge digits."""
    return _apply_clahe(img)


def preprocess_gamma_clahe(img: np.ndarray, gamma: float = 2.0) -> np.ndarray:
    """Gamma correction + CLAHE — reveals faint text in dark regions."""
    table = np.array(
        [((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]
    ).astype("uint8")
    bright = cv2.LUT(img, table)
    return _apply_clahe(bright)


# ---------------------------------------------------------------------------
# OCR engine
# ---------------------------------------------------------------------------

class TrainIDEngine:
    """Hybrid CnOCR engine using two detection models.

    - db_resnet18 (pytorch): superior digit accuracy, better edge detection
    - ch_PP-OCRv3_det (onnx): better at detecting complete vehicle type strings
    """

    def __init__(self):
        self.ocr_resnet = CnOcr(
            rec_model_name="densenet_lite_136-gru",
            det_model_name="db_resnet18",
            rec_model_backend="onnx",
            det_model_backend="pytorch",
        )
        self.ocr_ppocr = CnOcr(
            rec_model_name="densenet_lite_136-gru",
            det_model_name="ch_PP-OCRv3_det",
            rec_model_backend="onnx",
            det_model_backend="onnx",
        )
        logger.info("CnOCR engines initialized (db_resnet18 + ch_PP-OCRv3_det)")

    def recognize(self, image: np.ndarray, engine: str = "resnet") -> List[OCRBox]:
        """Run OCR and return parsed boxes."""
        ocr = self.ocr_resnet if engine == "resnet" else self.ocr_ppocr
        result = ocr.ocr(image)
        if not result:
            return []

        boxes = []
        for item in result:
            text = item.get("text", "").strip()
            if not text:
                continue
            score = float(item.get("score", 0.0))
            position = item.get("position")
            if position is not None:
                try:
                    xs = [p[0] for p in position]
                    ys = [p[1] for p in position]
                    box = [int(min(xs)), int(min(ys)),
                           int(max(xs)), int(max(ys))]
                except (TypeError, ValueError, IndexError):
                    box = [0, len(boxes) * 30, 100, (len(boxes) + 1) * 30]
            else:
                box = [0, len(boxes) * 30, 100, (len(boxes) + 1) * 30]
            boxes.append(OCRBox(box=box, text=text, confidence=score))
        return boxes


# ---------------------------------------------------------------------------
# Post-processing: general pattern-based OCR correction
# ---------------------------------------------------------------------------

# Character confusion tables — position-based disambiguation
_LETTER_TO_DIGIT = str.maketrans({
    "O": "0", "o": "0", "Q": "0", "D": "0",
    "I": "1", "l": "1", "i": "1",
    "S": "5", "s": "5",
    "A": "4", "a": "4",
    "G": "6", "g": "6",
    "T": "7",
    "B": "8", "b": "8",
    "Z": "2", "z": "2",
})

_DIGIT_TO_LETTER = str.maketrans({
    "0": "O",
    "1": "I",
    "4": "A",
    "5": "S",
    "6": "G",
    "7": "T",
    "8": "B",
})

# Noise patterns — artefacts from railway logo, background, etc.
_NOISE_RE = re.compile(
    r"^[TtGR®京上CEXN]$"         # single-char noise
    r"|^[（(（]\S*$"               # leading parens artefact
    r"|^On$|^G\d$|^CD$"           # logo artefacts observed empirically
)


def _is_noise(text: str, confidence: float) -> bool:
    """Return True if the text is likely OCR noise."""
    t = text.strip()
    if not t:
        return True
    if len(t) == 1 and confidence < 0.5:
        return True
    if len(t) <= 2 and confidence < 0.15:
        return True
    if _NOISE_RE.match(t):
        return True
    return False


def _fix_vehicle_type(text: str) -> str:
    """Correct OCR errors in vehicle type using position-based rules.

    Vehicle types follow: [A-Z]+[0-9]+[A-Z]* (e.g. C64K, C70E, NX70).
    Instead of hardcoded string substitutions, we segment the text and
    apply character-level confusion disambiguation by position.
    """
    t = text.strip()
    # Strip punctuation artefacts
    t = re.sub(r"^[/(（\[{]+", "", t)
    t = re.sub(r"[/）)\]}.]+$", "", t)
    t = t.upper()

    if not t:
        return ""

    # Remove trailing Q artefact (common OCR ghost on stenciled text)
    if len(t) > 2 and t.endswith("Q") and t[-2].isdigit():
        t = t[:-1]

    # Segment into letter_prefix + digit_middle + letter_suffix
    # Find the first digit-like character (digit or common letter→digit confusion)
    digit_like = set("0123456789OoQDIilSsAaGgTBbZz")
    first_digit_pos = None
    for i, c in enumerate(t):
        if i > 0 and (c.isdigit() or (c in digit_like and t[0].isalpha())):
            if c.isdigit():
                first_digit_pos = i
                break
            # Check if this looks like a digit-position char (preceded by letter)
            if i >= 1 and t[i - 1].isalpha() and not t[i - 1].isdigit():
                first_digit_pos = i
                break

    if first_digit_pos is None:
        return t

    # Find where suffix letters begin (after digit segment)
    # Must include all chars that appear in _LETTER_TO_DIGIT
    _digit_confusable = set("OoQDIilSsAaGgTBbZz")
    last_digit_pos = first_digit_pos
    for i in range(first_digit_pos, len(t)):
        c = t[i]
        if c.isdigit() or c in _digit_confusable:
            last_digit_pos = i
        else:
            break
    else:
        last_digit_pos = len(t) - 1

    prefix = t[:first_digit_pos]
    digit_seg = t[first_digit_pos:last_digit_pos + 1]
    suffix = t[last_digit_pos + 1:]

    # Fix prefix: digit-like chars → letters
    fixed_prefix = prefix.translate(_DIGIT_TO_LETTER)

    # Fix digit segment: letter-like chars → digits
    fixed_digits = digit_seg.translate(_LETTER_TO_DIGIT)
    # Remove any remaining non-digit characters in digit segment
    fixed_digits = re.sub(r"[^0-9]", "", fixed_digits)

    # Fix suffix: digit-like chars → letters
    fixed_suffix = suffix.translate(_DIGIT_TO_LETTER)

    result = fixed_prefix + fixed_digits + fixed_suffix
    return result


def _fix_vehicle_number(text: str) -> str:
    """Clean up a vehicle-number string using general character mapping."""
    t = text.strip()
    # Strip punctuation artefacts (slashes, dots, commas, etc.)
    t = re.sub(r"[/\\.,;:!?'\"()\[\]{}]", " ", t)
    # Single-pass character substitution for digit context
    char_map = str.maketrans({
        "O": "0", "o": "0", "Q": "0", "D": "0",
        "I": "1", "l": "1", "i": "1", "t": "1",
        "S": "5", "s": "5",
        "B": "8", "b": "8",
        "N": "",  "n": "",
    })
    t = t.translate(char_map)
    # Remove any remaining non-digit, non-space characters
    t = re.sub(r"[^\d\s]", "", t)
    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _is_vehicle_type_pattern(text: str) -> bool:
    """Check if text looks like a vehicle type (letter + digits + optional suffix).

    Rejects garbled numbers where all characters are digit-confusable
    (e.g. "Ot7O" → all chars map to digits → it's really "0170").
    """
    t = text.strip().upper()
    t = re.sub(r"^[/(（]+", "", t)
    if not re.match(r"^[A-Z]+\d+[A-Z]*Q?$", t):
        return False
    # Reject if converting all letter-like chars to digits gives all-numeric result
    fully_numeric = t.translate(_LETTER_TO_DIGIT)
    fully_numeric = re.sub(r"[^0-9]", "", fully_numeric)
    if len(fully_numeric) >= len(t):
        return False
    return True


# ---------------------------------------------------------------------------
# Line grouping and extraction
# ---------------------------------------------------------------------------

def _group_to_lines(
    boxes: List[OCRBox],
    tolerance: Optional[int] = None,
) -> List[List[OCRBox]]:
    """Group OCR boxes into horizontal lines by Y-centre."""
    if not boxes:
        return []

    if tolerance is None:
        heights = [b.box[3] - b.box[1] for b in boxes if b.box[3] > b.box[1]]
        if heights:
            heights.sort()
            tolerance = max(30, int(heights[len(heights) // 2] * 0.6))
        else:
            tolerance = 50

    sorted_boxes = sorted(boxes, key=lambda b: (b.box[1] + b.box[3]) / 2)
    lines: List[List[OCRBox]] = []
    cur_line: List[OCRBox] = []
    cur_y: Optional[float] = None

    for box in sorted_boxes:
        cy = (box.box[1] + box.box[3]) / 2
        if cur_y is None:
            cur_y = cy
            cur_line = [box]
        elif abs(cy - cur_y) <= tolerance:
            cur_line.append(box)
            cur_y = sum((b.box[1] + b.box[3]) / 2 for b in cur_line) / len(cur_line)
        else:
            cur_line.sort(key=lambda b: b.box[0])
            lines.append(cur_line)
            cur_line = [box]
            cur_y = cy

    if cur_line:
        cur_line.sort(key=lambda b: b.box[0])
        lines.append(cur_line)

    return lines


def _extract_from_boxes(boxes: List[OCRBox]) -> TrainID:
    """Extract vehicle type + number from OCR boxes."""
    clean = [b for b in boxes if not _is_noise(b.text, b.confidence)]
    if not clean:
        return TrainID()

    lines = _group_to_lines(clean)

    vehicle_type = ""
    number_parts: List[str] = []

    for line in lines:
        line_has_type = False
        line_number_parts: List[str] = []

        for box in line:
            t = box.text.strip()
            if _is_vehicle_type_pattern(t):
                candidate = _fix_vehicle_type(t)
                if not vehicle_type or len(candidate) > len(vehicle_type):
                    vehicle_type = candidate
                line_has_type = True
            elif re.search(r"\d", t):
                fixed = _fix_vehicle_number(t)
                if re.search(r"\d", fixed):
                    line_number_parts.append(fixed)

        if not line_has_type and line_number_parts:
            number_parts = line_number_parts

    vehicle_number = " ".join(number_parts)
    avg_conf = sum(b.confidence for b in clean) / len(clean) if clean else 0.0

    return TrainID(
        vehicle_type=vehicle_type,
        vehicle_number=vehicle_number,
        confidence=avg_conf,
    )


# ---------------------------------------------------------------------------
# Multi-pass merging
# ---------------------------------------------------------------------------

def _pick_best_type(candidates: List[TrainID]) -> str:
    """Choose best vehicle type via majority voting across passes.

    On ties, prefer later passes (gamma-enhanced) which tend to
    produce better character contrast for letter/digit distinction.
    """
    types = [c.vehicle_type for c in candidates if c.vehicle_type]
    if not types:
        return ""

    counts = Counter(types)
    top_count = counts.most_common(1)[0][1]

    if top_count >= 2:
        # Pick the tied type that appears latest (gamma passes are more reliable)
        for t in reversed(types):
            if counts[t] == top_count:
                return t

    # No majority — prefer last non-empty type (from gamma passes)
    return types[-1]


def _pick_best_number(candidates: List[TrainID]) -> str:
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
    # (catches trailing edge digits found by gamma passes)
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


# ---------------------------------------------------------------------------
# Multi-pass recognition pipeline
# ---------------------------------------------------------------------------

def recognize_train_id(
    image_path: str,
    engine: TrainIDEngine,
) -> TrainID:
    """Run multi-pass OCR with different preprocessing and merge results.

    Passes:
      1. Bilateral+CLAHE — best general quality, edge-preserving
      2. CLAHE — good baseline, detects edge digits
      3. Gamma(2.0)+CLAHE — reveals faint text
      4. Gamma(3.0)+CLAHE — strongest brightening for very faint text
    """
    # Load once, resize once
    img = _load_and_resize(image_path)
    if img is None:
        logger.warning(f"Cannot read image: {image_path}")
        return TrainID()

    # Each pass: (label, preprocessor, detection_engine)
    # - db_resnet18 ("resnet"): superior digit accuracy
    # - ch_PP-OCRv3_det ("ppocr"): better vehicle type detection
    passes = [
        ("bilateral_clahe", lambda im: preprocess_bilateral_clahe(im), "resnet"),
        ("clahe_ppocr", lambda im: preprocess_clahe(im), "ppocr"),
        ("gamma2_clahe", lambda im: preprocess_gamma_clahe(im, gamma=2.0), "resnet"),
        ("gamma3_clahe", lambda im: preprocess_gamma_clahe(im, gamma=3.0), "resnet"),
    ]

    results: List[TrainID] = []

    for label, preprocessor, det_engine in passes:
        enhanced = preprocessor(img.copy())
        boxes = engine.recognize(enhanced, engine=det_engine)
        tid = _extract_from_boxes(boxes)
        results.append(tid)
        logger.debug(
            f"  {label}: type='{tid.vehicle_type}' num='{tid.vehicle_number}'"
        )

    if not any(r.vehicle_type or r.vehicle_number for r in results):
        return TrainID()

    best_type = _pick_best_type(results)
    best_number = _pick_best_number(results)

    confs = [r.confidence for r in results if r.confidence > 0]
    avg_conf = sum(confs) / len(confs) if confs else 0.0

    return TrainID(
        vehicle_type=best_type,
        vehicle_number=best_number,
        confidence=avg_conf,
    )


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_folder(
    folder_path: str,
    output_dir: str,
    engine: TrainIDEngine,
) -> Dict[str, TrainID]:
    """Process all images in a folder and save per-image JSON results."""
    folder = Path(folder_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_files:
        logger.warning(f"No images found in {folder_path}")
        return {}

    logger.info(f"Processing {len(image_files)} images from {folder_path}")

    results: Dict[str, TrainID] = {}
    start = time.time()

    for img_path in image_files:
        t0 = time.time()
        tid = recognize_train_id(str(img_path), engine)
        elapsed = time.time() - t0

        results[img_path.name] = tid
        logger.info(
            f"  {img_path.name}: type='{tid.vehicle_type}' "
            f"number='{tid.vehicle_number}' ({elapsed:.2f}s)"
        )

        result_dict = {
            "file": img_path.name,
            "vehicle_type": tid.vehicle_type,
            "vehicle_number": tid.vehicle_number,
            "confidence": round(tid.confidence, 4),
        }
        json_path = out / f"{img_path.stem}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)

    total_time = time.time() - start
    logger.info(
        f"Done: {len(image_files)} images in {total_time:.1f}s "
        f"({total_time / len(image_files):.2f}s/image)"
    )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train ID OCR — extract vehicle type and number from station images"
    )
    parser.add_argument("folder", help="Input image folder")
    parser.add_argument("-o", "--output", default="./output", help="Output directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    folder = Path(args.folder)
    if not folder.is_dir():
        logger.error(f"Not a directory: {args.folder}")
        sys.exit(1)

    engine = TrainIDEngine()
    process_folder(str(folder), args.output, engine)


if __name__ == "__main__":
    main()
