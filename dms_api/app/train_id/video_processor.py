"""
Video Train ID & Container Recognition Processor

Based on train_id_ocr_video_paddle_v6.py
Processes video uploads to extract:
  - Container IDs (upper half of frame)
  - Train vehicle types & numbers (lower half of frame)

Adapted for FastAPI integration:
  - accepts raw video bytes
  - uses tempfile for frame extraction
  - returns structured results instead of CLI output
"""

import os
import re
import json
import logging
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from collections import Counter
from datetime import timedelta

import cv2
import numpy as np

from .video_engine import PaddleOCREngine

logger = logging.getLogger(__name__)

# Fix Intel OpenMP duplicate library conflict
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ============ Constants from V6 ============
PREFIX_CORRECTION = {
    "TEJU": "TBJU", "TRJU": "TBJU", "TPJU": "TBJU", "T8JU": "TBJU",
    "T3JU": "TBJU", "TCJU": "TBJU", "TBIU": "TBJU", "T0JU": "TBJU",
    "TB1U": "TBJU", "TBJ0": "TBJU", "T9JU": "TBJU",
    "IBJU": "TBJU", "I BJU": "TBJU",
    "IBCU": "TBCU", "TBCU": "TBCU", "1BCU": "TBCU",
}

PREFIX_DIGIT_FIX = {
    "1": "T", "3": "B", "8": "B", "0": "O",
    "5": "S", "2": "Z", "7": "T", "4": "A",
    "6": "G", "9": "G",
}

COMMON_PREFIXES = {
    "TBJU", "TBCU", "TRLU", "TGHU", "TCNU", "EMCU", "UESU",
    "HJCU", "BSIU", "APRU", "MEDU", "MSCW", "MRSU", "CSNU",
    "OOLU", "CMAU", "EGLV", "HLCU", "ONEU", "YMLU", "MAEU",
    "COSU", "UASC", "KLINE",
}

TRAIN_TYPE_CORRECTION = {
    "C7OE": "C70E", "C7DE": "C70E", "C70B": "C70E", "C7BE": "C70E",
    "C7O": "C70", "C7D": "C70", "COE": "C70E",
    "C64R": "C64K", "C64H": "C64K", "C6AK": "C64K",
    "G70E": "C70E",
    "C70": "C70E",
}

COMMON_TRAIN_TYPES = {
    "C70E", "C70", "C64K", "C64", "C62A", "C62",
    "P64", "P64K", "P70", "N17", "NX70",
}


@dataclass
class TextBox:
    text: str
    conf: float
    center_x: float
    center_y: float
    width: float
    height: float


@dataclass
class FrameResult:
    timestamp_sec: float
    texts: List[Tuple[str, float]] = field(default_factory=list)
    container_candidates: List[Tuple[str, float]] = field(default_factory=list)
    train_candidates: List[Tuple[str, float]] = field(default_factory=list)
    train_fragments: List[dict] = field(default_factory=list)


@dataclass
class VideoRecognitionResult:
    """Final structured result from video processing."""

    containers: List[str] = field(default_factory=list)
    train_types: List[str] = field(default_factory=list)
    train_numbers: List[str] = field(default_factory=list)
    container_count: int = 0
    train_type_count: int = 0
    train_number_count: int = 0
    frames_processed: int = 0
    duration_sec: float = 0.0


# ============ Helper functions ============
def _format_timestamp(sec: float) -> str:
    td = timedelta(seconds=sec)
    mm, ss = divmod(td.seconds, 60)
    return f"{mm:02d}:{ss:05.2f}"


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
        if d in PREFIX_DIGIT_FIX:
            fixed[pos] = PREFIX_DIGIT_FIX[d]
        else:
            return None
    result = "".join(fixed)
    if result in COMMON_PREFIXES or result in PREFIX_CORRECTION.values():
        return result
    for white in COMMON_PREFIXES:
        diff = sum(1 for a, b in zip(result, white) if a != b)
        if diff <= 1:
            return white
    return None


def _correct_prefix(prefix: str) -> str:
    if prefix in PREFIX_CORRECTION:
        return PREFIX_CORRECTION[prefix]
    if prefix in COMMON_PREFIXES:
        return prefix
    return prefix


# ============ Container extraction ============
def _extract_container_id(text: str, conf: float) -> Optional[Tuple[str, float]]:
    text = text.upper().replace(" ", "").replace("-", "").replace(".", "")
    match = re.search(r"([A-Z]{4})(\d{6,7})", text)
    if match:
        prefix = match.group(1)
        digits = match.group(2)[:6]
        prefix = _correct_prefix(prefix)
        return (f"{prefix}{digits}", conf)
    loose_match = re.search(r"([A-Z0-9]{4})(\d{6,7})", text)
    if loose_match:
        prefix = loose_match.group(1)
        digits = loose_match.group(2)[:6]
        fixed_prefix = _fix_prefix_digits(prefix)
        if fixed_prefix:
            return (f"{fixed_prefix}{digits}", conf)
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
                return (f"{fixed_prefix}{digits}", conf)
            if prefix.isalpha():
                prefix = _correct_prefix(prefix)
                return (f"{prefix}{digits}", conf)
    return None


# ============ Train ID extraction ============
def _fix_train_type(text: str) -> Optional[str]:
    if text in TRAIN_TYPE_CORRECTION:
        return TRAIN_TYPE_CORRECTION[text]
    if text in COMMON_TRAIN_TYPES:
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
        if result in COMMON_TRAIN_TYPES:
            return result
    if len(text) >= 4 and text[0].isalpha():
        first = text[0]
        digits = "".join(c for c in text[1:] if c.isdigit())
        letters = "".join(c for c in text[1:] if c.isalpha())
        if len(digits) >= 2:
            candidate = f"{first}{digits[-2:]}{letters[:1]}"
            if candidate in COMMON_TRAIN_TYPES:
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


def _extract_train_id(text: str, conf: float) -> Optional[Tuple[str, float]]:
    text = text.upper().replace(" ", "").replace("-", "").replace(".", "")
    match = re.search(r"(\d{3,8})", text)
    if match:
        return (match.group(1), conf)
    for type_pattern in [r"([A-Z]\d{2,4}?[A-Z])", r"([A-Z]\d{2,4}?)"]:
        match = re.search(type_pattern, text)
        if match:
            fixed_type = _fix_train_type(match.group(1))
            if fixed_type:
                return (fixed_type, conf)
    return None


# ============ Frame extraction ============
def _extract_frames(
    video_path: str,
    output_dir: str,
    interval_sec: float = 0.5,
    max_duration_sec: Optional[float] = None,
    start_time_sec: Optional[float] = None,
    end_time_sec: Optional[float] = None,
) -> Tuple[List[Tuple[float, str]], float, float]:
    """Extract frames from video to output_dir. Returns list of (timestamp, path)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    step = max(1, int(fps * interval_sec))
    cap.release()

    frames_dir = os.path.join(output_dir, "_frames")
    os.makedirs(frames_dir, exist_ok=True)

    # Clean old frames
    for f in os.listdir(frames_dir):
        if f.endswith(".jpg"):
            os.remove(os.path.join(frames_dir, f))

    cap = cv2.VideoCapture(video_path)
    frame_paths = []
    count = 0
    max_frames = int((max_duration_sec or duration) * fps) if max_duration_sec else total_frames
    start_frame = int(start_time_sec * fps) if start_time_sec else 0
    end_frame = int(end_time_sec * fps) if end_time_sec else total_frames

    while True:
        ret, frame = cap.read()
        if not ret or count >= max_frames or count >= end_frame:
            break
        if count >= start_frame and count % step == 0:
            ts = count / fps
            fname = f"frame_{ts:07.2f}.jpg"
            fpath = os.path.join(frames_dir, fname)
            cv2.imwrite(fpath, frame)
            frame_paths.append((ts, fpath))
        count += 1

    cap.release()
    logger.info(
        f"Extracted {len(frame_paths)} frames (fps={fps:.2f}, step={step}, "
        f"range={start_time_sec or 0:.1f}s~{end_time_sec or duration:.1f}s)"
    )
    return frame_paths, fps, duration


# ============ OCR box parsing ============
def _parse_ocr_boxes(result, img_height: int) -> Tuple[List[TextBox], List[TextBox]]:
    """Parse OCR results, split into upper/lower halves by Y coordinate."""
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
            box = TextBox(text, conf, center_x, center_y, width, height)

            if center_y < split_y:
                upper_boxes.append(box)
            else:
                if not _is_train_param(text):
                    lower_boxes.append(box)

    upper_boxes.sort(key=lambda b: b.center_x)
    lower_boxes.sort(key=lambda b: b.center_x)
    return upper_boxes, lower_boxes


# ============ Box merging ============
def _is_y_close(b1: TextBox, b2: TextBox, threshold_ratio: float = 0.6) -> bool:
    y_threshold = max(b1.height, b2.height) * threshold_ratio
    return abs(b1.center_y - b2.center_y) < y_threshold


def _merge_boxes(
    boxes: List[TextBox], extractor
) -> List[Tuple[str, float]]:
    """Multi-box merge: single + adjacent 2-box + adjacent 3-box."""
    candidates = []
    n = len(boxes)

    # Single box
    for b in boxes:
        cid = extractor(b.text, b.conf)
        if cid:
            candidates.append(cid)

    # Adjacent 2-box
    for i in range(n - 1):
        b1, b2 = boxes[i], boxes[i + 1]
        if not _is_y_close(b1, b2):
            continue
        merged_text = b1.text + b2.text
        merged_conf = (b1.conf + b2.conf) / 2
        cid = extractor(merged_text, merged_conf)
        if cid:
            candidates.append(cid)
        merged_text_space = b1.text + " " + b2.text
        cid2 = extractor(merged_text_space, merged_conf)
        if cid2 and cid2 not in candidates:
            candidates.append(cid2)

    # Adjacent 3-box
    for i in range(n - 2):
        b1, b2, b3 = boxes[i], boxes[i + 1], boxes[i + 2]
        if not (_is_y_close(b1, b2) and _is_y_close(b2, b3)):
            continue
        merged_text = b1.text + b2.text + b3.text
        merged_conf = (b1.conf + b2.conf + b3.conf) / 3
        cid = extractor(merged_text, merged_conf)
        if cid:
            candidates.append(cid)

    # Deduplicate, keep best confidence
    best = {}
    for cid, conf in candidates:
        if cid not in best or conf > best[cid]:
            best[cid] = conf
    return [(cid, conf) for cid, conf in best.items()]


def _merge_boxes_train(boxes: List[TextBox]) -> List[Tuple[str, float]]:
    """Train-specific merge: type boxes merge freely, digit boxes merge consecutively."""
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

    # Type boxes: single + any 2-box merge
    for b, fixed in type_boxes:
        candidates.append((fixed, b.conf))
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
                conf = (b1.conf + b2.conf) / 2
                candidates.append((fixed, conf))

    # Digit boxes: consecutive 2-4 box merge
    num_boxes.sort(key=lambda b: b.center_x)
    n = len(num_boxes)
    for b in num_boxes:
        match = re.search(r"\d{3,8}", b.text)
        if match:
            candidates.append((match.group(), b.conf))
    for length in range(2, min(5, n + 1)):
        for i in range(n - length + 1):
            merged = "".join(num_boxes[j].text for j in range(i, i + length))
            match = re.search(r"\d{3,8}", merged)
            if match:
                conf = sum(num_boxes[j].conf for j in range(i, i + length)) / length
                candidates.append((match.group(), conf))

    best = {}
    for cid, conf in candidates:
        if cid not in best or conf > best[cid]:
            best[cid] = conf
    return [(cid, conf) for cid, conf in best.items()]


# ============ Frame processing ============
def _process_frame(ocr_engine, img_path: str, ts: float) -> FrameResult:
    """Process a single frame: 1 OCR pass, split upper/lower."""
    img = cv2.imread(img_path)
    if img is None:
        return FrameResult(timestamp_sec=ts)

    h, w = img.shape[:2]
    result = ocr_engine.recognize(img_path)
    upper_boxes, lower_boxes = _parse_ocr_boxes(result, h)

    all_texts = [(b.text, b.conf) for b in upper_boxes + lower_boxes]
    container_ids = _merge_boxes(upper_boxes, _extract_container_id)
    train_ids = _merge_boxes_train(lower_boxes)

    # Collect fragments
    fragments = []
    type_boxes = []
    num_boxes = []
    for b in lower_boxes:
        t = b.text.upper().replace(" ", "").replace("-", "").replace(".", "")
        if not t:
            continue
        if re.match(r"^[A-Z]\d{0,4}[A-Z]?$", t):
            fixed = _fix_train_type(t)
            if fixed:
                type_boxes.append((b, fixed))
        elif t.isdigit():
            num_boxes.append((b, t))

    for b, fixed in type_boxes:
        fragments.append({"text": fixed, "conf": b.conf, "pos": "type", "timestamp_sec": ts})

    num_boxes.sort(key=lambda x: x[0].center_x)
    if len(num_boxes) >= 2:
        left_b, left_t = num_boxes[0]
        right_b, right_t = num_boxes[-1]
        merged_text = left_t + right_t
        merged_conf = (left_b.conf + right_b.conf) / 2
        fragments.append({"text": merged_text, "conf": merged_conf, "pos": "paired", "timestamp_sec": ts})
        if len(num_boxes) > 2:
            for b, t in num_boxes[1:-1]:
                fragments.append({"text": t, "conf": b.conf, "pos": "single", "timestamp_sec": ts})
    elif len(num_boxes) == 1:
        b, t = num_boxes[0]
        fragments.append({"text": t, "conf": b.conf, "pos": "single", "timestamp_sec": ts})

    return FrameResult(
        timestamp_sec=ts,
        texts=all_texts,
        container_candidates=container_ids,
        train_candidates=train_ids,
        train_fragments=fragments,
    )


# ============ Temporal aggregation ============
def _vote_best(fragments: List[dict]) -> Tuple[str, float]:
    """Vote: most frequent -> longest -> highest confidence."""
    if not fragments:
        return "", 0.0
    counts = Counter(f["text"] for f in fragments)
    best = None
    best_score = (-1, -1, -1.0)
    for text, count in counts.items():
        confs = [f["conf"] for f in fragments if f["text"] == text]
        avg_conf = sum(confs) / len(confs)
        score = (count, len(text), avg_conf)
        if score > best_score:
            best_score = score
            best = text
    confs = [f["conf"] for f in fragments if f["text"] == best]
    return best, sum(confs) / len(confs)


def _overlap_merge(a: str, b: str, min_overlap: int = 2) -> Optional[str]:
    for i in range(min(len(a), len(b)), min_overlap - 1, -1):
        if a[-i:] == b[:i]:
            return a + b[i:]
    return None


def _try_merge_singles(
    singles: List[dict], max_gap: float = 0.5
) -> Tuple[Optional[str], float]:
    if not singles:
        return None, 0.0

    sorted_singles = sorted(singles, key=lambda f: f.get("timestamp_sec", 0))
    texts = [f["text"] for f in sorted_singles]
    timestamps = [f.get("timestamp_sec", 0) for f in sorted_singles]

    best = texts[0]
    best_conf = sorted_singles[0]["conf"]
    used = set()

    for i in range(1, len(texts)):
        if i in used:
            continue
        if abs(timestamps[i] - timestamps[0]) > max_gap:
            continue
        merged = _overlap_merge(best, texts[i], min_overlap=2)
        if merged and len(merged) > len(best) and len(merged) <= 8:
            best = merged
            best_conf = (best_conf + sorted_singles[i]["conf"]) / 2
            used.add(i)

    if len(best) >= 5:
        return best, best_conf

    # Complementary merge: 3-digit + 4-digit = 7-digit
    threes = [(f, i) for i, f in enumerate(sorted_singles) if len(f["text"]) == 3 and i not in used]
    fours = [(f, i) for i, f in enumerate(sorted_singles) if len(f["text"]) == 4 and i not in used]
    for tf, ti in threes:
        for ff, fi in fours:
            if abs(tf.get("timestamp_sec", 0) - ff.get("timestamp_sec", 0)) <= max_gap:
                combined = tf["text"] + ff["text"]
                if len(combined) == 7:
                    conf = (tf["conf"] + ff["conf"]) / 2
                    return combined, conf

    # Any two fragments that sum to 7 digits
    for i, f1 in enumerate(sorted_singles):
        if i in used:
            continue
        for j, f2 in enumerate(sorted_singles):
            if j <= i or j in used:
                continue
            if abs(f1.get("timestamp_sec", 0) - f2.get("timestamp_sec", 0)) <= max_gap:
                combined = f1["text"] + f2["text"]
                if len(combined) == 7:
                    conf = (f1["conf"] + f2["conf"]) / 2
                    return combined, conf

    if len(best) >= 3:
        return best, best_conf
    return None, 0.0


def _assemble_train_fragments(
    frames: List[FrameResult], gap_sec: float = 3.0, complement_gap: float = 0.5
) -> Tuple[List[dict], List[dict]]:
    """Fragment-based temporal aggregation for train types and numbers."""
    sequences = []
    current_seq = None

    for frame in sorted(frames, key=lambda f: f.timestamp_sec):
        frags = frame.train_fragments
        if not frags:
            if current_seq:
                sequences.append(current_seq)
                current_seq = None
            continue

        if current_seq is None:
            current_seq = {
                "start_time": frame.timestamp_sec,
                "end_time": frame.timestamp_sec,
                "frames_count": 1,
                "types": [],
                "paired": [],
                "singles": [],
            }
            for f in frags:
                if f["pos"] == "type":
                    current_seq["types"].append(f)
                elif f["pos"] == "paired":
                    current_seq["paired"].append(f)
                else:
                    current_seq["singles"].append(f)
        elif frame.timestamp_sec - current_seq["end_time"] <= gap_sec:
            current_seq["end_time"] = frame.timestamp_sec
            current_seq["frames_count"] += 1
            for f in frags:
                if f["pos"] == "type":
                    current_seq["types"].append(f)
                elif f["pos"] == "paired":
                    current_seq["paired"].append(f)
                else:
                    current_seq["singles"].append(f)
        else:
            sequences.append(current_seq)
            current_seq = {
                "start_time": frame.timestamp_sec,
                "end_time": frame.timestamp_sec,
                "frames_count": 1,
                "types": [],
                "paired": [],
                "singles": [],
            }
            for f in frags:
                if f["pos"] == "type":
                    current_seq["types"].append(f)
                elif f["pos"] == "paired":
                    current_seq["paired"].append(f)
                else:
                    current_seq["singles"].append(f)

    if current_seq:
        sequences.append(current_seq)

    type_seqs = []
    num_seqs = []

    for seq in sequences:
        type_id, type_conf = _vote_best(seq["types"])

        num_id = None
        num_conf = 0.0

        if seq["paired"]:
            best_paired = max(seq["paired"], key=lambda f: (len(f["text"]), f["conf"]))
            num_id = best_paired["text"]
            num_conf = best_paired["conf"]

        if not num_id or len(num_id) < 7:
            candidates = []
            for f in seq["paired"]:
                if len(f["text"]) >= 2:
                    candidates.append(f)
            for f in seq["singles"]:
                candidates.append(f)

            if candidates:
                merged, merged_conf = _try_merge_singles(candidates, complement_gap)
                if merged and (not num_id or len(merged) > len(num_id)):
                    num_id = merged
                    num_conf = merged_conf

        if type_id:
            type_seqs.append(
                {
                    "id": type_id,
                    "start_time": _format_timestamp(seq["start_time"]),
                    "end_time": _format_timestamp(seq["end_time"]),
                    "frames_count": seq["frames_count"],
                    "avg_conf": round(type_conf, 3),
                }
            )

        if num_id and len(num_id) >= 3:
            num_seqs.append(
                {
                    "id": num_id,
                    "start_time": _format_timestamp(seq["start_time"]),
                    "end_time": _format_timestamp(seq["end_time"]),
                    "frames_count": seq["frames_count"],
                    "avg_conf": round(num_conf, 3),
                }
            )

    return type_seqs, num_seqs


def _temporal_dedup_containers(
    frames: List[FrameResult], gap_sec: float = 3.0
) -> List[dict]:
    """Temporal deduplication for container candidates."""
    corrected_candidates = []
    for frame in frames:
        cands = frame.container_candidates
        corrected_candidates.append(cands)

    sequences = []
    current_seq = None

    for i, frame in enumerate(frames):
        if not corrected_candidates[i]:
            if current_seq:
                sequences.append(current_seq)
                current_seq = None
            continue

        best_cid, best_conf = max(corrected_candidates[i], key=lambda x: x[1])

        if current_seq is None:
            current_seq = {
                "id": best_cid,
                "start_time": frame.timestamp_sec,
                "end_time": frame.timestamp_sec,
                "frames_count": 1,
                "raw_ids": [best_cid],
            }
        elif frame.timestamp_sec - current_seq["end_time"] <= gap_sec:
            current_seq["end_time"] = frame.timestamp_sec
            current_seq["frames_count"] += 1
            current_seq["raw_ids"].append(best_cid)
            current_seq["id"] = Counter(current_seq["raw_ids"]).most_common(1)[0][0]
        else:
            sequences.append(current_seq)
            current_seq = {
                "id": best_cid,
                "start_time": frame.timestamp_sec,
                "end_time": frame.timestamp_sec,
                "frames_count": 1,
                "raw_ids": [best_cid],
            }

    if current_seq:
        sequences.append(current_seq)

    for i, seq in enumerate(sequences, 1):
        seq["index"] = i
        seq["start_time"] = _format_timestamp(seq["start_time"])
        seq["end_time"] = _format_timestamp(seq["end_time"])

    return sequences


# ============ Top-level processor ============
class VideoTrainIDProcessor:
    """Processor for video-based train ID and container recognition."""

    def __init__(self, engine: Optional[PaddleOCREngine] = None):
        self.engine = engine or PaddleOCREngine.get_instance()

    @property
    def available(self) -> bool:
        return self.engine.available

    def process_video_bytes(
        self,
        video_bytes: bytes,
        filename: str = "unknown",
        interval_sec: float = 0.5,
        gap_sec: float = 3.0,
        max_duration_sec: Optional[float] = None,
        start_time_sec: Optional[float] = None,
        end_time_sec: Optional[float] = None,
    ) -> VideoRecognitionResult:
        """Process video bytes and return final recognition results.

        Args:
            video_bytes: Raw video file content
            filename: Original filename
            interval_sec: Frame extraction interval in seconds
            gap_sec: Temporal deduplication gap in seconds
            max_duration_sec: Maximum processing duration
            start_time_sec: Start time offset
            end_time_sec: End time offset

        Returns:
            VideoRecognitionResult with containers, train_types, train_numbers
        """
        if not self.engine.available:
            logger.error("PaddleOCR engine not available")
            return VideoRecognitionResult()

        with tempfile.TemporaryDirectory(prefix="dms_video_") as tmpdir:
            # Save video to temp file
            video_path = os.path.join(tmpdir, filename or "video.mp4")
            with open(video_path, "wb") as f:
                f.write(video_bytes)

            # Extract frames
            frame_list, fps, duration = _extract_frames(
                video_path,
                tmpdir,
                interval_sec=interval_sec,
                max_duration_sec=max_duration_sec,
                start_time_sec=start_time_sec,
                end_time_sec=end_time_sec,
            )

            if not frame_list:
                logger.warning("No frames extracted from video")
                return VideoRecognitionResult(duration_sec=duration)

            # Process each frame
            logger.info(f"Processing {len(frame_list)} frames...")
            results = []
            for idx, (ts, fpath) in enumerate(frame_list):
                result = _process_frame(self.engine, fpath, ts)
                results.append(result)
                if idx % 30 == 0 or idx == len(frame_list) - 1:
                    cids = [c for c, _ in result.container_candidates]
                    tids = [c for c, _ in result.train_candidates]
                    logger.debug(
                        f"  [{idx + 1}/{len(frame_list)}] containers={cids}, trains={tids}"
                    )

            # Temporal aggregation
            container_seqs = _temporal_dedup_containers(results, gap_sec)
            type_seqs, num_seqs = _assemble_train_fragments(results, gap_sec)

            # Extract final IDs
            containers = [s["id"] for s in container_seqs]
            train_types = [s["id"] for s in type_seqs]
            train_numbers = [s["id"] for s in num_seqs]

            logger.info(
                f"Video result: {len(containers)} containers, "
                f"{len(train_types)} train types, {len(train_numbers)} train numbers"
            )

            return VideoRecognitionResult(
                containers=containers,
                train_types=train_types,
                train_numbers=train_numbers,
                container_count=len(containers),
                train_type_count=len(train_types),
                train_number_count=len(train_numbers),
                frames_processed=len(frame_list),
                duration_sec=duration,
            )
