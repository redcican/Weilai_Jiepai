"""
PaddleOCR Single Image Processor

Based on train_id_ocr_video_paddle_v6.py.
Processes a single image to extract:
  - Upper half: container IDs
  - Lower half: train vehicle types and numbers

No video/temporal logic — pure single-frame processing.
"""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .video_engine import PaddleOCREngine

logger = logging.getLogger(__name__)


# ============ Core constants (mirrored from video_processor.py) ============

_common_prefixes = {
    "TBJU", "TBCU", "TRLU", "TGHU", "TCNU", "EMCU", "UESU",
    "HJCU", "BSIU", "APRU", "MEDU", "MSCW", "MRSU", "CSNU",
    "OOLU", "CMAU", "EGLV", "HLCU", "ONEU", "YMLU", "MAEU",
    "COSU", "UASC", "KLINE",
}

_prefix_correction = {
    "TEJU": "TBJU", "TRJU": "TBJU", "TPJU": "TBJU", "T8JU": "TBJU",
    "T3JU": "TBJU", "TCJU": "TBJU", "TBIU": "TBJU", "T0JU": "TBJU",
    "TB1U": "TBJU", "TBJ0": "TBJU", "T9JU": "TBJU",
    "IBJU": "TBJU", "I BJU": "TBJU",
    "IBCU": "TBCU", "TBCU": "TBCU", "1BCU": "TBCU",
}

_prefix_digit_fix = {
    "1": "T", "3": "B", "8": "B", "0": "O",
    "5": "S", "2": "Z", "7": "T", "4": "A",
    "6": "G", "9": "G",
}

_train_type_correction = {
    "C7OE": "C70E", "C7DE": "C70E", "C70B": "C70E", "C7BE": "C70E",
    "C7O": "C70", "C7D": "C70", "COE": "C70E",
    "C64R": "C64K", "C64H": "C64K", "C6AK": "C64K",
    "G70E": "C70E",
    "C70": "C70E",
}

_common_train_types = {
    "C70E", "C70", "C64K", "C64", "C62A", "C62",
    "P64", "P64K", "P70", "N17", "NX70",
}


# ============ Text utilities ============

import re


class _TextBox:
    def __init__(self, text: str, conf: float, center_x: float, center_y: float, width: float, height: float):
        self.text = text
        self.conf = conf
        self.center_x = center_x
        self.center_y = center_y
        self.width = width
        self.height = height


def _clean_digits(digits: str) -> str:
    mapping = str.maketrans({
        "i": "1", "I": "1", "l": "1", "L": "1",
        "o": "0", "O": "0", "Q": "0",
        "g": "9", "q": "9", "G": "6",
        "b": "6", "B": "8",
        "s": "5", "S": "5",
        "z": "2", "Z": "2",
        "a": "4", "A": "4",
    })
    return digits.translate(mapping)


def _fix_prefix_digits(prefix: str) -> Optional[str]:
    if len(prefix) != 4:
        return None
    digit_positions = [i for i, c in enumerate(prefix) if c.isdigit()]
    if len(digit_positions) == 0:
        return prefix
    if len(digit_positions) > 2:
        return None
    fixed = list(prefix)
    for pos in digit_positions:
        d = prefix[pos]
        if d in _prefix_digit_fix:
            fixed[pos] = _prefix_digit_fix[d]
        else:
            return None
    result = "".join(fixed)
    if result in _common_prefixes or result in _prefix_correction.values():
        return result
    for white in _common_prefixes:
        diff = sum(1 for a, b in zip(result, white) if a != b)
        if diff <= 1:
            return white
    return None


def _correct_prefix(prefix: str) -> str:
    if prefix in _prefix_correction:
        return _prefix_correction[prefix]
    if prefix in _common_prefixes:
        return prefix
    return prefix


def _extract_container_id(text: str) -> Optional[str]:
    text = text.upper().replace(" ", "").replace("-", "").replace(".", "")
    match = re.search(r"([A-Z]{4})(\d{6,7})", text)
    if match:
        prefix = match.group(1)
        digits = match.group(2)[:6]
        prefix = _correct_prefix(prefix)
        return f"{prefix}{digits}"
    loose_match = re.search(r"([A-Z0-9]{4})(\d{6,7})", text)
    if loose_match:
        prefix = loose_match.group(1)
        digits = loose_match.group(2)[:6]
        fixed_prefix = _fix_prefix_digits(prefix)
        if fixed_prefix:
            return f"{fixed_prefix}{digits}"
    for prefix_match in re.finditer(r"[A-Z0-9]{4}", text):
        prefix = prefix_match.group(0)
        prefix_pos = prefix_match.end()
        remaining = text[prefix_pos:]
        cleaned = _clean_digits(remaining)
        digit_match = re.search(r"(\d{6,7})", cleaned)
        if digit_match:
            digits = digit_match.group(1)[:6]
            fixed_prefix = _fix_prefix_digits(prefix)
            if fixed_prefix:
                fixed_prefix = _correct_prefix(fixed_prefix)
                return f"{fixed_prefix}{digits}"
            if prefix.isalpha():
                prefix = _correct_prefix(prefix)
                return f"{prefix}{digits}"
    return None


def _fix_train_type(text: str) -> Optional[str]:
    if text in _train_type_correction:
        return _train_type_correction[text]
    if text in _common_train_types:
        return text
    if text == "C":
        return "C70E"
    if text == "70E":
        return "C70E"
    if text == "70":
        return "C70"
    if len(text) == 4:
        fixed = list(text)
        for i, c in enumerate(fixed):
            if c == "O" and i >= 1:
                fixed[i] = "0"
            if c == "I" and i >= 1:
                fixed[i] = "1"
        result = "".join(fixed)
        if result in _common_train_types:
            return result
    if len(text) >= 4 and text[0].isalpha():
        first = text[0]
        digits = "".join(c for c in text[1:] if c.isdigit())
        letters = "".join(c for c in text[1:] if c.isalpha())
        if len(digits) >= 2:
            candidate = f"{first}{digits[-2:]}{letters[:1]}"
            if candidate in _common_train_types:
                return candidate
    return None


def _is_train_param(text: str) -> bool:
    t = text.lower()
    if any(c in t for c in ["t", "m³", "m3", "载", "重", "自", "容", "积", "换", "长", "定"]):
        return True
    if any(c in text for c in [".", "(", ")", "X", "×", "*"]):
        return True
    if all("\u4e00" <= c <= "\u9fff" for c in text):
        return True
    if any(c in text for c in ["#", "+", "-"]):
        return True
    if text.isalpha() and len(text) <= 3 and text.upper() not in ["C70", "C64", "P64", "P70", "N17"]:
        return True
    return False


def _extract_train_id(text: str) -> Optional[str]:
    text = text.upper().replace(" ", "").replace("-", "").replace(".", "")
    match = re.search(r"(\d{3,8})", text)
    if match:
        return match.group(1)
    for type_pattern in [r"([A-Z]\d{2,4}?[A-Z])", r"([A-Z]\d{2,4}?)"]:
        match = re.search(type_pattern, text)
        if match:
            fixed_type = _fix_train_type(match.group(1))
            if fixed_type:
                return fixed_type
    return None


# ============ Box parsing and merging ============

def _parse_ocr_boxes(result, img_height: int) -> Tuple[List[_TextBox], List[_TextBox]]:
    upper_boxes = []
    lower_boxes = []
    split_y = img_height * 0.55

    if result and result[0]:
        for line in result[0]:
            if not line:
                continue
            coords = line[0]
            text, conf = line[1]
            xs = [p[0] for p in coords]
            ys = [p[1] for p in coords]
            center_x = sum(xs) / 4
            center_y = sum(ys) / 4
            width = max(xs) - min(xs)
            height = max(ys) - min(ys)
            box = _TextBox(text, conf, center_x, center_y, width, height)

            if center_y < split_y:
                upper_boxes.append(box)
            else:
                if not _is_train_param(text):
                    lower_boxes.append(box)

    upper_boxes.sort(key=lambda b: b.center_x)
    lower_boxes.sort(key=lambda b: b.center_x)
    return upper_boxes, lower_boxes


def _is_y_close(b1: _TextBox, b2: _TextBox, threshold_ratio: float = 0.6) -> bool:
    y_threshold = max(b1.height, b2.height) * threshold_ratio
    return abs(b1.center_y - b2.center_y) < y_threshold


def _merge_boxes(boxes: List[_TextBox], extractor) -> List[str]:
    candidates = []
    n = len(boxes)

    for b in boxes:
        cid = extractor(b.text)
        if cid:
            candidates.append(cid)

    for i in range(n - 1):
        b1, b2 = boxes[i], boxes[i + 1]
        if not _is_y_close(b1, b2):
            continue
        merged_text = b1.text + b2.text
        cid = extractor(merged_text)
        if cid:
            candidates.append(cid)
        merged_text_space = b1.text + " " + b2.text
        cid2 = extractor(merged_text_space)
        if cid2 and cid2 not in candidates:
            candidates.append(cid2)

    for i in range(n - 2):
        b1, b2, b3 = boxes[i], boxes[i + 1], boxes[i + 2]
        if not (_is_y_close(b1, b2) and _is_y_close(b2, b3)):
            continue
        merged_text = b1.text + b2.text + b3.text
        cid = extractor(merged_text)
        if cid:
            candidates.append(cid)

    return list(dict.fromkeys(candidates))


def _merge_boxes_train(boxes: List[_TextBox]) -> List[str]:
    candidates = []

    type_boxes = []
    num_boxes = []
    for b in boxes:
        t = b.text.upper().replace(" ", "").replace("-", "").replace(".", "")
        if re.match(r"^[A-Z]\d{0,4}[A-Z]?$", t):
            fixed = _fix_train_type(t)
            if fixed:
                type_boxes.append((b, fixed))
        elif t.isdigit():
            num_boxes.append(b)

    for b, fixed in type_boxes:
        candidates.append(fixed)
    m = len(type_boxes)
    if m >= 2:
        from itertools import combinations
        for i, j in combinations(range(m), 2):
            b1, fixed1 = type_boxes[i]
            b2, fixed2 = type_boxes[j]
            merged = b1.text + b2.text
            fixed = _fix_train_type(merged)
            if not fixed:
                merged2 = b2.text + b1.text
                fixed = _fix_train_type(merged2)
            if fixed:
                candidates.append(fixed)

    num_boxes.sort(key=lambda b: b.center_x)
    n = len(num_boxes)
    for b in num_boxes:
        match = re.search(r"\d{3,8}", b.text)
        if match:
            candidates.append(match.group())
    for length in range(2, min(5, n + 1)):
        for i in range(n - length + 1):
            merged = "".join(num_boxes[j].text for j in range(i, i + length))
            match = re.search(r"\d{3,8}", merged)
            if match:
                candidates.append(match.group())

    return list(dict.fromkeys(candidates))


# ============ Top-level processor ============

class PaddleImageProcessor:
    """Single-image processor using PaddleOCR with upper/lower split."""

    def __init__(self, engine: Optional[PaddleOCREngine] = None):
        self.engine = engine or PaddleOCREngine.get_instance(lang="en")

    @property
    def available(self) -> bool:
        return self.engine.available

    def process_bytes(self, image_bytes: bytes) -> dict:
        """Process raw image bytes and return structured results.

        Returns:
            {
                "containers": ["TBJU123456", ...],
                "train_types": ["C70E", ...],
                "train_numbers": ["1755648", ...],
            }
        """
        if not self.engine.available:
            logger.error("PaddleOCR engine not available")
            return {"containers": [], "train_types": [], "train_numbers": []}

        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("Failed to decode image bytes")
            return {"containers": [], "train_types": [], "train_numbers": []}

        return self._process_img(img)

    def _process_img(self, img: np.ndarray) -> dict:
        h, w = img.shape[:2]

        result = self.engine.ocr(img, cls=True)

        upper_boxes, lower_boxes = _parse_ocr_boxes(result, h)

        container_ids = _merge_boxes(upper_boxes, _extract_container_id)
        train_ids = _merge_boxes_train(lower_boxes)

        train_types = [t for t in train_ids if re.match(r"^[A-Z]\d{2,4}[A-Z]?$", t)]
        train_numbers = [t for t in train_ids if t.isdigit()]

        return {
            "containers": container_ids,
            "train_types": train_types,
            "train_numbers": train_numbers,
        }
