"""
Train ID Utility Functions

Image preprocessing, OCR text correction, noise filtering,
and line grouping helpers for train ID recognition.
"""

import re
from typing import List, Optional

import cv2
import numpy as np

from .models import OCRBox, TrainIDResult

RESIZE_SCALE = 0.25

# ---------------------------------------------------------------------------
# Character confusion tables — position-based disambiguation
# ---------------------------------------------------------------------------

LETTER_TO_DIGIT = str.maketrans({
    "O": "0", "o": "0", "Q": "0", "D": "0",
    "I": "1", "l": "1", "i": "1",
    "S": "5", "s": "5",
    "A": "4", "a": "4",
    "G": "6", "g": "6",
    "T": "7",
    "B": "8", "b": "8",
    "Z": "2", "z": "2",
})

DIGIT_TO_LETTER = str.maketrans({
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

# Characters that could be digits in a digit context
_DIGIT_LIKE = set("0123456789OoQDIilSsAaGgTBbZz")

# Characters that are letters confused as digits
_DIGIT_CONFUSABLE = set("OoQDIilSsAaGgTBbZz")


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def decode_image_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
    """Decode image bytes to BGR numpy array."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def resize(img: np.ndarray, scale: float = RESIZE_SCALE) -> np.ndarray:
    """Resize image by scale factor."""
    h, w = img.shape[:2]
    nw, nh = int(w * scale), int(h * scale)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def apply_clahe(img: np.ndarray, clip_limit: float = 3.0) -> np.ndarray:
    """Apply CLAHE on LAB L-channel and return RGB image."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    lab = cv2.merge([l_ch, a_ch, b_ch])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def preprocess_bilateral_clahe(img: np.ndarray) -> np.ndarray:
    """Bilateral filter (edge-preserving) + CLAHE."""
    filtered = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    return apply_clahe(filtered)


def preprocess_clahe(img: np.ndarray) -> np.ndarray:
    """CLAHE only."""
    return apply_clahe(img)


def preprocess_gamma_clahe(img: np.ndarray, gamma: float = 2.0) -> np.ndarray:
    """Gamma correction + CLAHE — reveals faint text in dark regions."""
    table = np.array(
        [((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]
    ).astype("uint8")
    bright = cv2.LUT(img, table)
    return apply_clahe(bright)


# ---------------------------------------------------------------------------
# Noise filtering
# ---------------------------------------------------------------------------

def is_noise(text: str, confidence: float) -> bool:
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
# OCR text correction
# ---------------------------------------------------------------------------

def fix_vehicle_type(text: str) -> str:
    """Correct OCR errors in vehicle type using position-based rules.

    Vehicle types follow: [A-Z]+[0-9]+[A-Z]* (e.g. C64K, C70E, NX70).
    Segments text into letter_prefix + digit_middle + letter_suffix,
    then applies character-level confusion disambiguation by position.
    """
    t = text.strip()
    t = re.sub(r"^[/((\[{]+", "", t)
    t = re.sub(r"[/)\]}.]+$", "", t)
    t = t.upper()

    if not t:
        return ""

    # Remove trailing Q artefact (common OCR ghost on stenciled text)
    if len(t) > 2 and t.endswith("Q") and t[-2].isdigit():
        t = t[:-1]

    # Find the first digit-like character position
    first_digit_pos = None
    for i, c in enumerate(t):
        if i > 0 and (c.isdigit() or (c in _DIGIT_LIKE and t[0].isalpha())):
            if c.isdigit():
                first_digit_pos = i
                break
            if i >= 1 and t[i - 1].isalpha() and not t[i - 1].isdigit():
                first_digit_pos = i
                break

    if first_digit_pos is None:
        return t

    # Find where suffix letters begin (after digit segment)
    last_digit_pos = first_digit_pos
    for i in range(first_digit_pos, len(t)):
        c = t[i]
        if c.isdigit() or c in _DIGIT_CONFUSABLE:
            last_digit_pos = i
        else:
            break
    else:
        last_digit_pos = len(t) - 1

    prefix = t[:first_digit_pos]
    digit_seg = t[first_digit_pos:last_digit_pos + 1]
    suffix = t[last_digit_pos + 1:]

    # Fix each segment with appropriate confusion table
    fixed_prefix = prefix.translate(DIGIT_TO_LETTER)
    fixed_digits = digit_seg.translate(LETTER_TO_DIGIT)
    fixed_digits = re.sub(r"[^0-9]", "", fixed_digits)
    fixed_suffix = suffix.translate(DIGIT_TO_LETTER)

    return fixed_prefix + fixed_digits + fixed_suffix


def fix_vehicle_number(text: str) -> str:
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


def is_vehicle_type_pattern(text: str) -> bool:
    """Check if text looks like a vehicle type (letter + digits + optional suffix).

    Rejects garbled numbers where all characters are digit-confusable
    (e.g. "Ot7O" → all chars map to digits → it's really "0170").
    """
    t = text.strip().upper()
    t = re.sub(r"^[/(\uff08]+", "", t)
    if not re.match(r"^[A-Z]+\d+[A-Z]*Q?$", t):
        return False
    fully_numeric = t.translate(LETTER_TO_DIGIT)
    fully_numeric = re.sub(r"[^0-9]", "", fully_numeric)
    if len(fully_numeric) >= len(t):
        return False
    return True


# ---------------------------------------------------------------------------
# Line grouping and extraction
# ---------------------------------------------------------------------------

def group_to_lines(
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


def extract_from_boxes(boxes: List[OCRBox]) -> TrainIDResult:
    """Extract vehicle type + number from OCR boxes.

    Groups boxes into lines, identifies type patterns and number patterns,
    then returns the best match.
    """
    clean = [b for b in boxes if not is_noise(b.text, b.confidence)]
    if not clean:
        return TrainIDResult()

    lines = group_to_lines(clean)

    vehicle_type = ""
    number_parts: List[str] = []

    for line in lines:
        line_has_type = False
        line_number_parts: List[str] = []

        for box in line:
            t = box.text.strip()
            if is_vehicle_type_pattern(t):
                candidate = fix_vehicle_type(t)
                if not vehicle_type or len(candidate) > len(vehicle_type):
                    vehicle_type = candidate
                line_has_type = True
            elif re.search(r"\d", t):
                fixed = fix_vehicle_number(t)
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
