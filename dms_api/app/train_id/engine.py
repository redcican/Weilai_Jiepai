"""
Train ID OCR Engine

Hybrid CnOCR engine with multi-pass preprocessing for railway car
identification recognition. Adapted from train_id_ocr/train_id_ocr.py
for in-process use (accepts numpy arrays or bytes instead of file paths).
"""

import re
import logging
from collections import Counter
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .models import OCRBox, TrainIDResult

logger = logging.getLogger(__name__)

RESIZE_SCALE = 0.25

# ---------------------------------------------------------------------------
# Character confusion tables — position-based disambiguation
# ---------------------------------------------------------------------------

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

_NOISE_RE = re.compile(
    r"^[TtGR\u00ae\u4eac\u4e0aCEXN]$"  # single-char noise
    r"|^[(\uff08\u300a]\S*$"              # leading parens artefact
    r"|^On$|^G\d$|^CD$"                   # logo artefacts
)


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def _decode_image_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
    """Decode image bytes to BGR numpy array."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def _resize(img: np.ndarray, scale: float = RESIZE_SCALE) -> np.ndarray:
    """Resize image by scale factor."""
    h, w = img.shape[:2]
    nw, nh = int(w * scale), int(h * scale)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def _apply_clahe(img: np.ndarray, clip_limit: float = 3.0) -> np.ndarray:
    """Apply CLAHE on LAB L-channel and return RGB image."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    lab = cv2.merge([l_ch, a_ch, b_ch])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _preprocess_bilateral_clahe(img: np.ndarray) -> np.ndarray:
    """Bilateral filter (edge-preserving) + CLAHE."""
    filtered = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    return _apply_clahe(filtered)


def _preprocess_clahe(img: np.ndarray) -> np.ndarray:
    """CLAHE only."""
    return _apply_clahe(img)


def _preprocess_gamma_clahe(img: np.ndarray, gamma: float = 2.0) -> np.ndarray:
    """Gamma correction + CLAHE — reveals faint text in dark regions."""
    table = np.array(
        [((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]
    ).astype("uint8")
    bright = cv2.LUT(img, table)
    return _apply_clahe(bright)


# ---------------------------------------------------------------------------
# Noise filtering
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# OCR correction
# ---------------------------------------------------------------------------

def _fix_vehicle_type(text: str) -> str:
    """Correct OCR errors in vehicle type using position-based rules.

    Vehicle types follow: [A-Z]+[0-9]+[A-Z]* (e.g. C64K, C70E, NX70).
    """
    t = text.strip()
    t = re.sub(r"^[/((\[{]+", "", t)
    t = re.sub(r"[/)\]}.]+$", "", t)
    t = t.upper()

    if not t:
        return ""

    if len(t) > 2 and t.endswith("Q") and t[-2].isdigit():
        t = t[:-1]

    digit_like = set("0123456789OoQDIilSsAaGgTBbZz")
    first_digit_pos = None
    for i, c in enumerate(t):
        if i > 0 and (c.isdigit() or (c in digit_like and t[0].isalpha())):
            if c.isdigit():
                first_digit_pos = i
                break
            if i >= 1 and t[i - 1].isalpha() and not t[i - 1].isdigit():
                first_digit_pos = i
                break

    if first_digit_pos is None:
        return t

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

    fixed_prefix = prefix.translate(_DIGIT_TO_LETTER)
    fixed_digits = digit_seg.translate(_LETTER_TO_DIGIT)
    fixed_digits = re.sub(r"[^0-9]", "", fixed_digits)
    fixed_suffix = suffix.translate(_DIGIT_TO_LETTER)

    return fixed_prefix + fixed_digits + fixed_suffix


def _fix_vehicle_number(text: str) -> str:
    """Clean up a vehicle-number string using general character mapping."""
    t = text.strip()
    t = re.sub(r"[/\\.,;:!?'\"()\[\]{}]", " ", t)
    char_map = str.maketrans({
        "O": "0", "o": "0", "Q": "0", "D": "0",
        "I": "1", "l": "1", "i": "1", "t": "1",
        "S": "5", "s": "5",
        "B": "8", "b": "8",
        "N": "", "n": "",
    })
    t = t.translate(char_map)
    t = re.sub(r"[^\d\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _is_vehicle_type_pattern(text: str) -> bool:
    """Check if text looks like a vehicle type (letter + digits + optional suffix).

    Rejects garbled numbers where all characters are digit-confusable.
    """
    t = text.strip().upper()
    t = re.sub(r"^[/(\uff08]+", "", t)
    if not re.match(r"^[A-Z]+\d+[A-Z]*Q?$", t):
        return False
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


def _extract_from_boxes(boxes: List[OCRBox]) -> TrainIDResult:
    """Extract vehicle type + number from OCR boxes."""
    clean = [b for b in boxes if not _is_noise(b.text, b.confidence)]
    if not clean:
        return TrainIDResult()

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

    return TrainIDResult(
        vehicle_type=vehicle_type,
        vehicle_number=vehicle_number,
        confidence=avg_conf,
    )


# ---------------------------------------------------------------------------
# Multi-pass merging
# ---------------------------------------------------------------------------

def _pick_best_type(candidates: List[TrainIDResult]) -> str:
    """Choose best vehicle type via majority voting.

    On ties, prefer later passes (gamma-enhanced).
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

    return types[-1]


def _pick_best_number(candidates: List[TrainIDResult]) -> str:
    """Choose best vehicle number using superset-aware majority voting."""
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

    for digits, full in digit_map:
        if len(digits) > len(most_common_digits) and digits.startswith(most_common_digits):
            return full

    if count >= 2:
        for digits, full in digit_map:
            if digits == most_common_digits:
                return full

    primary = candidates[0].vehicle_number
    primary_digits = re.sub(r"\D", "", primary)

    if not primary_digits:
        return digit_map[0][1] if digit_map else ""

    for digits, full in digit_map[1:]:
        if len(digits) > len(primary_digits) and digits.startswith(primary_digits):
            return full

    return primary


# ---------------------------------------------------------------------------
# Main engine class
# ---------------------------------------------------------------------------

class TrainIDEngine:
    """Hybrid CnOCR engine for train ID recognition.

    Uses two detection models:
    - db_resnet18 (pytorch): superior digit accuracy
    - ch_PP-OCRv3_det (onnx): better vehicle type detection

    Singleton pattern — call get_instance() to reuse.
    """

    _instance: Optional["TrainIDEngine"] = None

    def __init__(self):
        self._available = False
        self.ocr_resnet = None
        self.ocr_ppocr = None

        try:
            from cnocr import CnOcr

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
            self._available = True
            logger.info("Train ID OCR engines initialized (db_resnet18 + ch_PP-OCRv3_det)")
        except ImportError as e:
            logger.error(f"CnOCR or torch not installed: {e}")
        except Exception as e:
            logger.error(f"Train ID engine initialization failed: {e}")

    @classmethod
    def get_instance(cls) -> "TrainIDEngine":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def available(self) -> bool:
        return self._available

    def _run_ocr(self, image: np.ndarray, engine: str = "resnet") -> List[OCRBox]:
        """Run OCR on a preprocessed image."""
        ocr = self.ocr_resnet if engine == "resnet" else self.ocr_ppocr
        try:
            result = ocr.ocr(image)
        except Exception as e:
            logger.warning(f"OCR recognition failed ({engine}): {e}")
            return []

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

    def recognize(self, image: np.ndarray) -> TrainIDResult:
        """Run multi-pass OCR pipeline on a BGR image and return merged result.

        The image is resized and run through 4 preprocessing passes:
        1. Bilateral+CLAHE (resnet) — best general quality
        2. CLAHE (ppocr) — good baseline, complete type strings
        3. Gamma(2.0)+CLAHE (resnet) — reveals faint text
        4. Gamma(3.0)+CLAHE (resnet) — strongest brightening
        """
        if not self._available:
            logger.error("Train ID engine not available")
            return TrainIDResult()

        img = _resize(image)

        passes = [
            ("bilateral_clahe", lambda im: _preprocess_bilateral_clahe(im), "resnet"),
            ("clahe_ppocr", lambda im: _preprocess_clahe(im), "ppocr"),
            ("gamma2_clahe", lambda im: _preprocess_gamma_clahe(im, gamma=2.0), "resnet"),
            ("gamma3_clahe", lambda im: _preprocess_gamma_clahe(im, gamma=3.0), "resnet"),
        ]

        results: List[TrainIDResult] = []

        for label, preprocessor, det_engine in passes:
            enhanced = preprocessor(img.copy())
            boxes = self._run_ocr(enhanced, engine=det_engine)
            tid = _extract_from_boxes(boxes)
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

    def recognize_bytes(self, image_bytes: bytes) -> TrainIDResult:
        """Recognize train ID from raw image bytes."""
        img = _decode_image_bytes(image_bytes)
        if img is None:
            logger.warning("Failed to decode image bytes")
            return TrainIDResult()
        return self.recognize(img)
