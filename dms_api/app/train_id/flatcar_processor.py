"""
Flatcar (车板号) Video Recognition Processor

Based on run_bottom_merge_ocr.py
Processes video bottom region (75%-100%) to extract flatcar type and number.
Features:
  - Bottom ROI extraction (75%-100% height)
  - Row-wise box merging for split digit strings
  - Flatcar-specific type correction
  - Number overlap merging across frames
  - Temporal aggregation with type-number pairing
"""

import os
import re
import json
import logging
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from collections import Counter, defaultdict
from datetime import timedelta

import cv2
import numpy as np

from .video_engine import PaddleOCREngine

logger = logging.getLogger(__name__)

# ============ Flatcar constants ============
BOTTOM_Y_START = 0.75
BOTTOM_Y_END = 1.0

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


# ============ Result dataclass ============
@dataclass
class FlatcarRecognitionResult:
    """Final structured result from flatcar video processing."""

    results: List[Dict] = field(default_factory=list)
    type_count: int = 0
    number_count: int = 0
    frames_processed: int = 0
    duration_sec: float = 0.0


# ============ Helper functions ============
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


def _format_ts(sec: float) -> str:
    td = timedelta(seconds=sec)
    return str(td)[:-3] if "." in str(td) else str(td) + ".000"


def _extract_frames(video_path: str, interval_sec: float) -> Tuple[List[Tuple[float, np.ndarray]], float]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = idx / fps
        if idx % max(1, int(fps * interval_sec)) == 0:
            frames.append((t, frame))
        idx += 1
    cap.release()
    return frames, fps


# ============ Row-wise box merging ============
def _merge_boxes_by_row(texts: List[Tuple[str, float, list]], y_tolerance: float = 50) -> List[Tuple[str, float, list, list]]:
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
            rows.append((merged_text, avg_conf, merged_box, current_row))
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
        rows.append((merged_text, avg_conf, merged_box, current_row))

    return rows


# ============ Candidate extraction ============
def _extract_candidates(merged_rows: List[Tuple[str, float, list, list]]) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """Extract flatcar type and number candidates from merged rows."""
    type_candidates = []
    num_candidates = []

    for merged_text, conf, box, raw_items in merged_rows:
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


# ============ Number overlap merging ============
def _merge_numbers_by_overlap(
    nums_with_conf: List[Tuple[str, float]], target_len: int = 7, min_overlap: int = 2
) -> Tuple[Optional[str], float]:
    """Merge number candidates via overlap拼接 to reach target length."""
    if len(nums_with_conf) < 2:
        return None, 0.0

    best_result = None
    best_conf = 0.0

    def try_merge(a, ca, b, cb):
        nonlocal best_result, best_conf
        max_ol = min(len(a), len(b))
        for ol in range(max_ol, min_overlap - 1, -1):
            if a[-ol:] == b[:ol]:
                merged = a + b[ol:]
                if len(merged) == target_len:
                    conf = (ca + cb) / 2
                    if conf > best_conf:
                        best_conf = conf
                        best_result = merged
                return merged, (ca + cb) / 2
        return None, 0.0

    # Pairwise merge
    for i in range(len(nums_with_conf)):
        for j in range(len(nums_with_conf)):
            if i == j:
                continue
            a, ca = nums_with_conf[i]
            b, cb = nums_with_conf[j]
            try_merge(a, ca, b, cb)
            try_merge(b, cb, a, ca)

    # Three-frame merge
    if best_result is None and len(nums_with_conf) >= 3:
        partials = []
        for i in range(len(nums_with_conf)):
            for j in range(len(nums_with_conf)):
                if i == j:
                    continue
                a, ca = nums_with_conf[i]
                b, cb = nums_with_conf[j]
                max_ol = min(len(a), len(b))
                for ol in range(max_ol, min_overlap - 1, -1):
                    if a[-ol:] == b[:ol]:
                        merged = a + b[ol:]
                        if len(merged) <= target_len:
                            partials.append((merged, (ca + cb) / 2))
                        break

        for p, pc in partials:
            for k in range(len(nums_with_conf)):
                c, cc = nums_with_conf[k]
                try_merge(p, pc, c, cc)
                try_merge(c, cc, p, pc)

    return best_result, best_conf


# ============ Top-level processor ============
class FlatcarVideoProcessor:
    """Processor for flatcar (车板号) video recognition from bottom region."""

    def __init__(self, engine: Optional[PaddleOCREngine] = None):
        # Use Chinese OCR model for flatcar recognition
        self.engine = engine or PaddleOCREngine.get_instance(lang="ch")

    @property
    def available(self) -> bool:
        return self.engine.available

    def process_video_bytes(
        self,
        video_bytes: bytes,
        filename: str = "unknown",
        interval_sec: float = 0.05,
        gap_sec: float = 0.15,
    ) -> FlatcarRecognitionResult:
        """Process video bytes and return flatcar recognition results.

        Args:
            video_bytes: Raw video file content
            filename: Original filename
            interval_sec: Frame extraction interval in seconds (default 0.05 for fast moving)
            gap_sec: Temporal aggregation gap in seconds (default 0.15)

        Returns:
            FlatcarRecognitionResult with type+number pairs
        """
        if not self.engine.available:
            logger.error("PaddleOCR (ch) engine not available")
            return FlatcarRecognitionResult()

        with tempfile.TemporaryDirectory(prefix="dms_flatcar_") as tmpdir:
            # Save video to temp file
            video_path = os.path.join(tmpdir, filename or "video.avi")
            with open(video_path, "wb") as f:
                f.write(video_bytes)

            # Extract frames
            frames, fps = _extract_frames(video_path, interval_sec)
            duration = len(frames) * interval_sec if frames else 0.0

            if not frames:
                logger.warning("No frames extracted from video")
                return FlatcarRecognitionResult(duration_sec=duration)

            logger.info(f"Processing {len(frames)} frames for flatcar recognition...")

            # Process each frame
            results = []
            for i, (t, frame) in enumerate(frames):
                h, w = frame.shape[:2]
                y1 = int(h * BOTTOM_Y_START)
                y2 = int(h * BOTTOM_Y_END)
                bottom_roi = frame[y1:y2, :]

                enhanced = _preprocess_for_dark(bottom_roi)

                # Save temp image for OCR
                tmp_img_path = os.path.join(tmpdir, f"frame_{i:04d}.jpg")
                cv2.imwrite(tmp_img_path, enhanced)
                ocr_result = self.engine.recognize(tmp_img_path)

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

                results.append({
                    "timestamp_sec": t,
                    "timestamp": _format_ts(t),
                    "type_candidates": type_candidates,
                    "num_candidates": num_candidates,
                })

                if i % 50 == 0 or i == len(frames) - 1:
                    logger.debug(
                        f"  [{i + 1}/{len(frames)}] types={len(type_candidates)} nums={len(num_candidates)}"
                    )

            # Temporal aggregation
            logger.info("Running temporal aggregation...")

            # Type sequences
            type_sequences = []
            current = None
            for r in results:
                if not r["type_candidates"]:
                    continue
                if current is None or r["timestamp_sec"] - current[-1]["timestamp_sec"] > gap_sec:
                    if current:
                        type_sequences.append(current)
                    current = [r]
                else:
                    current.append(r)
            if current:
                type_sequences.append(current)

            # Number sequences
            num_sequences = []
            current = None
            for r in results:
                has_digit = any(len(n) >= 3 for n, c in r["num_candidates"])
                if not has_digit:
                    continue
                if current is None or r["timestamp_sec"] - current[-1]["timestamp_sec"] > gap_sec:
                    if current:
                        num_sequences.append(current)
                    current = [r]
                else:
                    current.append(r)
            if current:
                num_sequences.append(current)

            # Build type seqs
            type_seqs = []
            for seq_idx, seq in enumerate(type_sequences, 1):
                types = [(t, c) for r in seq for t, c in r["type_candidates"]]
                cnt = Counter([t for t, _ in types])
                type_id = cnt.most_common(1)[0][0]
                type_conf = sum(c for t, c in types if t == type_id) / len([1 for t, _ in types if t == type_id])
                type_seqs.append({
                    "index": seq_idx,
                    "start_time": seq[0]["timestamp"],
                    "end_time": seq[-1]["timestamp"],
                    "frames_count": len(seq),
                    "avg_conf": round(type_conf, 4),
                    "id": type_id,
                    "start_sec": seq[0]["timestamp_sec"],
                    "end_sec": seq[-1]["timestamp_sec"],
                })

            # Build number seqs
            num_seqs = []
            for seq_idx, seq in enumerate(num_sequences, 1):
                nums = [(n, c) for r in seq for n, c in r["num_candidates"] if len(n) >= 3]
                num_id = None
                num_conf = 0.0
                by_len = defaultdict(list)
                for n, c in nums:
                    by_len[len(n)].append((n, c))
                if 7 in by_len:
                    candidates = [(n, c) for n, c in by_len[7] if c >= 0.95]
                    if not candidates:
                        candidates = by_len[7]
                    weighted = defaultdict(float)
                    for n, c in candidates:
                        weighted[n] += c
                    num_id = max(weighted.keys(), key=lambda k: weighted[k])
                    num_conf = weighted[num_id] / len(candidates)
                else:
                    num_id, num_conf = _merge_numbers_by_overlap(nums, target_len=7, min_overlap=2)
                    if num_id is None:
                        for target_len in [6, 5, 8, 4, 3]:
                            if target_len in by_len:
                                candidates = by_len[target_len]
                                cnt = Counter([n for n, _ in candidates])
                                num_id = cnt.most_common(1)[0][0]
                                num_conf = sum(c for n, c in candidates if n == num_id) / len([1 for n, _ in candidates if n == num_id])
                                break
                if num_id:
                    num_seqs.append({
                        "index": seq_idx,
                        "start_time": seq[0]["timestamp"],
                        "end_time": seq[-1]["timestamp"],
                        "frames_count": len(seq),
                        "avg_conf": round(num_conf, 4),
                        "id": num_id,
                        "start_sec": seq[0]["timestamp_sec"],
                        "end_sec": seq[-1]["timestamp_sec"],
                    })

            # Filter: only 7-digit numbers with conf >= 0.90
            filtered_nums = [s for s in num_seqs if len(s["id"]) == 7 and s["avg_conf"] >= 0.90]

            # Merge type and number
            merged_results = []
            used_num_indices = set()

            for t in type_seqs:
                t_start, t_end = t["start_sec"], t["end_sec"]
                best_match = None
                best_overlap = -1
                for n in filtered_nums:
                    n_start, n_end = n["start_sec"], n["end_sec"]
                    if max(t_start, n_start) <= min(t_end, n_end):
                        overlap = min(t_end, n_end) - max(t_start, n_start)
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_match = n

                entry = {
                    "type": t["id"],
                    "number": best_match["id"] if best_match else None,
                    "frames": best_match["frames_count"] if best_match else t["frames_count"],
                    "avg_conf": round(best_match["avg_conf"], 4) if best_match else round(t["avg_conf"], 4),
                }
                if best_match:
                    used_num_indices.add(best_match["index"])
                merged_results.append(entry)

            # Add unmatched numbers
            for n in filtered_nums:
                if n["index"] not in used_num_indices:
                    merged_results.append({
                        "type": None,
                        "number": n["id"],
                        "frames": n["frames_count"],
                        "avg_conf": round(n["avg_conf"], 4),
                    })

            logger.info(
                f"Flatcar result: {len(type_seqs)} types, {len(filtered_nums)} numbers, "
                f"{len(merged_results)} merged entries"
            )

            return FlatcarRecognitionResult(
                results=merged_results,
                type_count=len(type_seqs),
                number_count=len(filtered_nums),
                frames_processed=len(frames),
                duration_sec=duration,
            )
