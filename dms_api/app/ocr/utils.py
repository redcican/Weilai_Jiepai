"""
OCR Utility Functions

Helper functions for text correction, splitting, and row processing.
"""

import re
from typing import List, Tuple, Optional, Dict, Any

from .models import OCRBox


# ── Patterns for Type 2 (集装箱编组单) ──
_CONTAINER_RE = re.compile(r'[A-Z]{3,4}[A-Z0-9]\d{6,7}')
_SLASH_VEHICLE_RE = re.compile(r'[A-Za-z]\w*/\d{5,}')
_CHINESE_RE = re.compile(r'[\u4e00-\u9fff]')


def enhance_image_for_ocr(image_path: str):
    """
    Enhance image to improve OCR recognition.
    Increases contrast and sharpness.
    """
    try:
        from PIL import Image, ImageEnhance
        import numpy as np

        img = Image.open(image_path)

        if img.mode != 'RGB':
            img = img.convert('RGB')

        contrast_enhancer = ImageEnhance.Contrast(img)
        img = contrast_enhancer.enhance(1.3)

        sharpness_enhancer = ImageEnhance.Sharpness(img)
        img = sharpness_enhancer.enhance(1.5)

        return np.array(img)

    except Exception:
        return None


def enhance_image_bytes_for_ocr(image_bytes: bytes):
    """
    Enhance image bytes to improve OCR recognition.
    """
    try:
        from PIL import Image, ImageEnhance
        import numpy as np
        from io import BytesIO

        img = Image.open(BytesIO(image_bytes))

        if img.mode != 'RGB':
            img = img.convert('RGB')

        contrast_enhancer = ImageEnhance.Contrast(img)
        img = contrast_enhancer.enhance(1.3)

        sharpness_enhancer = ImageEnhance.Sharpness(img)
        img = sharpness_enhancer.enhance(1.5)

        return np.array(img)

    except Exception:
        return None


def correct_ocr_text(text: str) -> str:
    """
    Fix common OCR recognition errors.
    """
    if not text:
        return text

    def fix_vehicle_type(match):
        s = match.group(0).upper()
        result = []
        for i, c in enumerate(s):
            if c == 'O' and i > 0:
                if s[i-1].isdigit() or (i < len(s)-1 and (s[i+1].isdigit() or s[i+1] == 'E' or s[i+1] == 'H')):
                    result.append('0')
                    continue
            if c == 'E' and i > 0 and i < len(s)-1:
                if s[i-1].isdigit() and s[i+1] == 'E':
                    result.append('0')
                    continue
            result.append(c)
        return ''.join(result)

    text = re.sub(r'[A-Za-z]+[0-9OoEeHh]+[A-Za-z]*', fix_vehicle_type, text)

    if re.match(r'^[Cc]7$', text):
        text = 'C70'
    elif re.match(r'^[Cc]64$', text):
        text = 'C64K'

    corrections = {
        '敬二': '敞二',
        '散二': '敞二',
        'θ': '0',
        'Θ': '0',
        'о': '0',
        'О': '0',
    }

    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)

    return text


def split_merged_text(text: str) -> List[str]:
    """Split OCR merged text intelligently."""
    if not text:
        return []

    text = text.strip()
    if not text:
        return []

    result = []
    parts = re.split(r'[；;]', text)

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if re.search(r'[：:]', part):
            sub_parts = re.split(r'[：:]', part, maxsplit=1)
            for sp in sub_parts:
                sp = sp.strip()
                if sp:
                    result.append(sp)
            continue

        match = re.match(r'^(.+空\d+?)(.+箱.*)$', part)
        if match:
            result.append(match.group(1))
            result.extend(split_merged_text(match.group(2)))
            continue

        match = re.match(r'^(.+空\d+?)(部/局.*)$', part)
        if match:
            result.append(match.group(1))
            result.append(match.group(2))
            continue

        if len(part) > 4:
            match = re.match(r'^(.*部/局)([一-龥]{2,4})$', part)
            if match:
                result.append(match.group(1))
                result.append(match.group(2))
                continue

        result.append(part)

    return result if result else [text]


def split_merged_numbers(text: str) -> List[str]:
    """Recursively split incorrectly merged numbers."""
    if not text or not text.strip():
        return []

    text = text.strip()

    # Sequence number and vehicle type merged
    match = re.match(r'^(\d{1,3})([A-Za-z].*)$', text)
    if match:
        return [match.group(1), match.group(2)]

    # Pattern: digits + decimal + decimal + digits
    match = re.match(r'^(.+?)(\d{2}\.\d)(\d\.\d)(\d+)$', text)
    if match:
        prefix = match.group(1)
        first_decimal = match.group(2)
        second_decimal = match.group(3)
        suffix = match.group(4)

        result = split_merged_numbers(prefix)
        result.append(first_decimal)
        result.append(second_decimal)
        if suffix:
            result.append(suffix)
        return result

    # Pattern: digits + two decimals (no suffix)
    match = re.match(r'^(.+?)(\d{2}\.\d)(\d\.\d+)$', text)
    if match:
        prefix = match.group(1)
        first_decimal = match.group(2)
        second_decimal = match.group(3)

        result = split_merged_numbers(prefix)
        result.append(first_decimal)
        result.append(second_decimal)
        return result

    # Pattern: two decimals
    match = re.match(r'^(\d+\.\d)(\d\.\d+)$', text)
    if match:
        return [match.group(1), match.group(2)]

    # Pattern: 7-digit vehicle ID + decimal
    match = re.match(r'^(\d{7})(\d{2}\.\d+)$', text)
    if match:
        return [match.group(1), match.group(2)]

    # Pattern: 7-digit + 1 digit + decimal
    match = re.match(r'^(\d{7})(\d)(\d{2}\.\d+)$', text)
    if match:
        return [match.group(1), match.group(2), match.group(3)]

    # Pattern: 8-digit vehicle ID + decimal
    match = re.match(r'^(\d{8})(\d{2}\.\d+)$', text)
    if match:
        return [match.group(1), match.group(2)]

    # Pattern: 6-digit vehicle ID + decimal
    match = re.match(r'^(\d{6})(\d{2}\.\d+)$', text)
    if match:
        return [match.group(1), match.group(2)]

    # Pattern: single digit + decimal
    match = re.match(r'^(\d)(\d{2}\.\d+)$', text)
    if match:
        return [match.group(1), match.group(2)]

    # Handle space-separated
    if ' ' in text:
        parts = text.split()
        result = []
        for part in parts:
            result.extend(split_merged_numbers(part))
        return result

    return [text]


def aggregate_to_rows(ocr_results: List[OCRBox], tolerance: int = None) -> List[List[str]]:
    """Aggregate OCR results by row using center Y coordinate."""
    if not ocr_results:
        return []

    def center_y(box):
        return (box.box[1] + box.box[3]) / 2

    def box_height(box):
        return box.box[3] - box.box[1]

    # Adaptive tolerance based on median text height
    if tolerance is None:
        heights = [box_height(r) for r in ocr_results if box_height(r) > 0]
        if heights:
            heights.sort()
            median_height = heights[len(heights) // 2]
            tolerance = max(10, int(median_height * 0.6))
        else:
            tolerance = 20

    sorted_results = sorted(ocr_results, key=center_y)

    rows = []
    current_row = []
    current_y = None

    for result in sorted_results:
        y = center_y(result)

        if current_y is None:
            current_y = y
            current_row = [result]
        elif abs(y - current_y) <= tolerance:
            current_row.append(result)
            current_y = sum(center_y(r) for r in current_row) / len(current_row)
        else:
            current_row.sort(key=lambda r: r.box[0])
            row_texts = []
            for r in current_row:
                for text_part in split_merged_text(r.text):
                    row_texts.extend(split_merged_numbers(text_part))
            rows.append(row_texts)
            current_row = [result]
            current_y = y

    if current_row:
        current_row.sort(key=lambda r: r.box[0])
        row_texts = []
        for r in current_row:
            for text_part in split_merged_text(r.text):
                row_texts.extend(split_merged_numbers(text_part))
        rows.append(row_texts)

    return rows


def is_potential_sequence_number(text: str) -> bool:
    return bool(re.match(r'^\d{1,3}$', text.strip()))


def detect_sequence_column(rows: List[List[str]]) -> int:
    """Auto-detect sequence number column."""
    if not rows or len(rows) < 3:
        return -1

    max_cols_to_check = min(3, min(len(r) for r in rows if r))

    for col_idx in range(max_cols_to_check):
        values = []
        for row in rows:
            if col_idx < len(row):
                val = row[col_idx].strip()
                if re.match(r'^\d{1,3}$', val):
                    values.append(int(val))
                else:
                    values.append(None)

        valid_values = [v for v in values if v is not None]
        if len(valid_values) < 3:
            continue

        sorted_vals = sorted(valid_values)
        if sorted_vals[0] <= 10:
            consecutive = 0
            for i in range(len(valid_values) - 1):
                if valid_values[i] is not None and valid_values[i + 1] is not None:
                    diff = valid_values[i + 1] - valid_values[i]
                    if 0 <= diff <= 2:
                        consecutive += 1

            if consecutive >= len(valid_values) * 0.5:
                return col_idx

    return -1


def is_metadata_item(row: List[str]) -> Tuple[bool, Optional[str], Optional[str]]:
    """Check if row is a metadata item."""
    if len(row) == 1:
        text = row[0]
        match = re.match(r'^(.+)[：:](.+)$', text)
        if match:
            return True, match.group(1).strip(), match.group(2).strip()

    if len(row) == 2:
        key, value = row[0], row[1]
        if re.match(r'^[\u4e00-\u9fa5]+$', key) and len(key) <= 4:
            return True, key, value

    return False, None, None


def is_header_row(row: List[str]) -> bool:
    """Check if row is a header row."""
    if not row:
        return False
    header_keywords = {
        '车号', '到站', '品名', '记事', '发站', '票据', '自重', '换长', '载重',
        '序号', '站存车打印', '集装箱', '箱号', '油种', '蓬布', '属性', '收货人',
    }
    row_text = ''.join(row)
    return any(kw in row_text for kw in header_keywords)


def is_page_footer(row: List[str]) -> bool:
    """Check if row is a page footer (e.g. 第1页, 第2页)."""
    if not row:
        return False
    row_text = ''.join(row).strip()
    return bool(re.match(r'^第\d+页$', row_text))


def detect_table_type(ocr_results: List[OCRBox]) -> int:
    """
    Detect table type:
      Type 1 — 站存车打印 (~16 cols, vehicle type/number in separate columns)
      Type 2 — 集装箱编组单 (~5 cols, slash vehicle/number like C70E/1805776)

    Key signal: slash-vehicle patterns are unique to Type 2.
    Container patterns alone are unreliable (Type 1 may have cargo reference codes).
    """
    slash_vehicle_count = 0
    for box in ocr_results:
        slash_vehicle_count += len(_SLASH_VEHICLE_RE.findall(box.text))

    if slash_vehicle_count >= 2:
        return 2
    return 1


TYPE2_COLUMNS = ["序", "ID1", "ID2", "ID3", "地点"]


def _classify_type2_row(
    row: List[str], next_seq: int
) -> Tuple[Optional[Dict[str, str]], int]:
    """
    Pattern-based classification for a Type 2 data row.

    Assigns each value to a column by pattern rather than position:
    - digits (1-3 chars) at position 0 → 序
    - slash-vehicle (e.g., C70E/1721133) → ID1
    - container number (e.g., TBJU3216534) → ID2, then ID3
    - Chinese text (e.g., 漳平) → 地点
    - Unrecognized values → skipped as noise

    Returns (row_dict, updated_next_seq) or (None, updated_next_seq)
    for fragment rows that have no meaningful data.
    """
    if not row:
        return None, next_seq

    first = row[0].strip()
    has_real_seq = bool(re.match(r'^\d{1,3}$', first))

    if has_real_seq:
        seq = first
        data_vals = row[1:]
    elif _SLASH_VEHICLE_RE.search(first) or _CONTAINER_RE.search(first):
        seq = str(next_seq)
        data_vals = row
    else:
        seq = str(next_seq)
        data_vals = row[1:]

    id1 = ""
    containers: List[str] = []
    location = ""

    for val in data_vals:
        v = val.strip()
        if not v:
            continue
        if _SLASH_VEHICLE_RE.search(v):
            id1 = v
        elif _CONTAINER_RE.search(v):
            containers.append(v)
        elif _CHINESE_RE.search(v):
            location = v

    if not id1 and not containers and not location:
        return None, (int(seq) + 1 if has_real_seq else next_seq)

    result = {
        "序": seq,
        "ID1": id1,
        "ID2": containers[0] if len(containers) >= 1 else "",
        "ID3": containers[1] if len(containers) >= 2 else "",
        "地点": location,
    }

    return result, int(seq) + 1


def extract_type2(all_rows: List[List[str]]) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    """
    Type 2 (集装箱编组单) extraction:
    Filter headers/footers/metadata, keep data rows, then classify
    each value by pattern into the correct column.
    """
    metadata: Dict[str, Any] = {}
    raw_rows: List[List[str]] = []

    for row in all_rows:
        if not row:
            continue
        if is_header_row(row):
            continue
        if is_page_footer(row):
            continue

        is_meta, key, value = is_metadata_item(row)
        if is_meta and key and value:
            metadata[key] = value
            continue

        row_text = ''.join(row)
        has_container = bool(_CONTAINER_RE.search(row_text))
        has_slash = bool(_SLASH_VEHICLE_RE.search(row_text))
        has_seq = bool(row[0].strip()) and re.match(r'^\d{1,3}$', row[0].strip())

        if has_container or has_slash or has_seq:
            raw_rows.append(row)

    data_rows: List[Dict[str, str]] = []
    next_seq = 1

    for row in raw_rows:
        row_dict, next_seq = _classify_type2_row(row, next_seq)
        if row_dict is not None:
            data_rows.append(row_dict)

    return metadata, data_rows


def detect_vehicle_type_column(rows: List[List[str]]) -> int:
    """Auto-detect vehicle type column."""
    if not rows or len(rows) < 3:
        return -1

    max_cols = min(5, min(len(r) for r in rows if r))

    for col_idx in range(max_cols):
        vehicle_pattern_count = 0
        for row in rows:
            if col_idx < len(row):
                val = row[col_idx].strip()
                if re.match(r'^[A-Za-z]+\d+', val) or re.match(r'^[A-Za-z]\d+[A-Za-z]*$', val):
                    vehicle_pattern_count += 1
                elif re.match(r'^\d{2,3}[A-Za-z]*$', val):
                    vehicle_pattern_count += 1

        if vehicle_pattern_count >= len(rows) * 0.6:
            return col_idx

    return -1


def detect_vehicle_id_column(rows: List[List[str]]) -> int:
    """Auto-detect vehicle ID column (6-8 digits)."""
    if not rows or len(rows) < 3:
        return -1

    max_cols = min(5, min(len(r) for r in rows if r))

    for col_idx in range(max_cols):
        id_pattern_count = 0
        for row in rows:
            if col_idx < len(row):
                val = row[col_idx].strip()
                if re.match(r'^\d{6,8}$', val):
                    id_pattern_count += 1

        if id_pattern_count >= len(rows) * 0.6:
            return col_idx

    return -1


def is_valid_data_row(row: List[str], min_cols: int = 4) -> bool:
    """Check if row is a valid data row."""
    if not row or len(row) < min_cols:
        return False
    if is_header_row(row):
        return False

    content = ''.join(str(c).strip() for c in row)
    return len(content) > 5


def normalize_vehicle_type(cell: str) -> str:
    """Normalize vehicle type code."""
    cell = cell.strip()

    if re.match(r'^[A-Za-z]', cell):
        return cell

    if re.match(r'^(70|64|62)\w*$', cell):
        return 'C' + cell

    return cell


def normalize_row(row: List[str], vehicle_col: int = -1) -> List[str]:
    """Normalize data row."""
    if not row:
        return row

    result = []
    for i, cell in enumerate(row):
        cell = cell.strip()

        if vehicle_col >= 0 and i == vehicle_col:
            cell = normalize_vehicle_type(cell)
        elif vehicle_col < 0 and i <= 2:
            cell = normalize_vehicle_type(cell)

        result.append(cell)

    return result


def extract_track_number_from_first_row(
    rows: List[List[str]], seq_col: int
) -> Tuple[Optional[str], List[List[str]]]:
    """Extract track number from first row."""
    if not rows or len(rows) < 2:
        return None, rows

    first_row = rows[0]
    if not first_row or len(first_row) < 2:
        return None, rows

    first_val = first_row[0].strip()

    if not re.match(r'^\d{1,2}$', first_val):
        return None, rows

    first_num = int(first_val)

    # Pattern 1: First row has more columns than subsequent rows
    other_row_lens = [len(r) for r in rows[1:5] if r]
    if other_row_lens:
        avg_len = sum(other_row_lens) / len(other_row_lens)
        if len(first_row) > avg_len + 0.5:
            if len(first_row) > 1:
                second_val = first_row[1].strip()
                if re.match(r'^[12]$', second_val):
                    track = first_val
                    new_first_row = first_row[1:]
                    return track, [new_first_row] + rows[1:]

    # Pattern 2: First row format [large_num, 1, ...], subsequent [2, ...], [3, ...]
    if len(first_row) > 1:
        second_val = first_row[1].strip()
        if re.match(r'^[12]$', second_val):
            second_num = int(second_val)

            subsequent_firsts = []
            for row in rows[1:5]:
                if row and re.match(r'^\d{1,2}$', row[0].strip()):
                    subsequent_firsts.append(int(row[0].strip()))

            if subsequent_firsts:
                expected_next = second_num + 1
                if subsequent_firsts[0] == expected_next:
                    track = first_val
                    new_first_row = first_row[1:]
                    return track, [new_first_row] + rows[1:]

    # Pattern 3: First row's first number doesn't fit sequence
    seq_vals = []
    for row in rows[1:6]:
        if row and re.match(r'^\d{1,2}$', row[0].strip()):
            seq_vals.append(int(row[0].strip()))

    if len(seq_vals) >= 2:
        is_sequential = all(seq_vals[i+1] - seq_vals[i] in [1, 2] for i in range(len(seq_vals)-1))
        if is_sequential:
            expected_start = seq_vals[0] - 1
            if first_num > seq_vals[-1] or first_num < expected_start - 1:
                track = first_val
                if len(first_row) > 1 and re.match(r'^[12]$', first_row[1].strip()):
                    new_first_row = first_row[1:]
                    return track, [new_first_row] + rows[1:]

    return None, rows


# ── Type 1 column definitions (站存车打印, 16 columns) ──
TYPE1_COLUMNS = [
    "股道", "序", "车种", "油种", "车号", "自重",
    "换长", "载重", "到站", "品名", "记事", "发站",
    "篷布", "票据号", "属性", "收货人",
]

# Map header text fragments → column index (longest match first)
_HEADER_COL_MAP = {
    "收货人": 15, "票据号": 13, "站存车打印": -1,
    "股道": 0, "车种": 2, "油种": 3, "车号": 4,
    "自重": 5, "换长": 6, "载重": 7, "到站": 8,
    "品名": 9, "记事": 10, "发站": 11, "篷布": 12,
    "蓬布": 12, "属性": 14, "序": 1, "属": 14,
}


def aggregate_to_box_rows(
    ocr_results: List[OCRBox], tolerance: int = None
) -> List[List[OCRBox]]:
    """Group OCR boxes into rows by Y-coordinate, preserving box positions."""
    if not ocr_results:
        return []

    def center_y(box):
        return (box.box[1] + box.box[3]) / 2

    def box_height(box):
        return box.box[3] - box.box[1]

    if tolerance is None:
        heights = [box_height(r) for r in ocr_results if box_height(r) > 0]
        if heights:
            heights.sort()
            median_height = heights[len(heights) // 2]
            tolerance = max(10, int(median_height * 0.6))
        else:
            tolerance = 20

    sorted_results = sorted(ocr_results, key=center_y)
    rows: List[List[OCRBox]] = []
    current_row: List[OCRBox] = []
    current_y: Optional[float] = None

    for result in sorted_results:
        y = center_y(result)
        if current_y is None:
            current_y = y
            current_row = [result]
        elif abs(y - current_y) <= tolerance:
            current_row.append(result)
            current_y = sum(center_y(r) for r in current_row) / len(current_row)
        else:
            current_row.sort(key=lambda r: r.box[0])
            rows.append(current_row)
            current_row = [result]
            current_y = y

    if current_row:
        current_row.sort(key=lambda r: r.box[0])
        rows.append(current_row)

    return rows


def _is_header_box_row(box_row: List[OCRBox]) -> bool:
    """Check if a box row is a header row (primary: >=3 keywords)."""
    text = ''.join(b.text for b in box_row)
    header_kw = ['车号', '到站', '品名', '自重', '换长', '载重', '记事']
    return sum(1 for kw in header_kw if kw in text) >= 3


def _is_secondary_header_row(box_row: List[OCRBox]) -> bool:
    """Check if a row is a secondary header row (>=1 header keyword)."""
    text = ''.join(b.text for b in box_row)
    all_kw = ['股道', '序', '车种', '油种', '车号', '到站', '品名', '自重',
              '换长', '载重', '记事', '发站', '篷布', '票据号', '属性', '收货人']
    return sum(1 for kw in all_kw if kw in text) >= 1


def _parse_column_centers(header_boxes: List[OCRBox]) -> Dict[int, float]:
    """Extract column center x-positions from header boxes using proportional character mapping."""
    col_centers: Dict[int, float] = {}
    sorted_keys = sorted(_HEADER_COL_MAP.keys(), key=len, reverse=True)

    for box in header_boxes:
        raw_text = box.text.replace(' ', '')
        x_left, x_right = box.box[0], box.box[2]
        n_chars = len(raw_text)
        if n_chars == 0:
            continue
        char_w = (x_right - x_left) / n_chars

        pos = 0
        while pos < n_chars:
            matched = False
            for key in sorted_keys:
                klen = len(key)
                if pos + klen <= n_chars and raw_text[pos:pos + klen] == key:
                    col_idx = _HEADER_COL_MAP[key]
                    if col_idx >= 0:
                        center_x = x_left + (pos + klen / 2) * char_w
                        col_centers[col_idx] = center_x
                    pos += klen
                    matched = True
                    break
            if not matched:
                pos += 1

    return col_centers


def _build_column_boundaries(
    col_centers: Dict[int, float], image_width: int
) -> List[Tuple[float, float]]:
    """Convert column centers to (left, right) boundary ranges."""
    known = sorted(col_centers.items())
    if not known:
        return [(0, image_width)] * 16

    all_centers: Dict[int, float] = dict(known)

    spacings = []
    for i in range(len(known) - 1):
        idx_a, x_a = known[i]
        idx_b, x_b = known[i + 1]
        if idx_b > idx_a:
            spacings.append((x_b - x_a) / (idx_b - idx_a))
    avg_spacing = sum(spacings) / len(spacings) if spacings else 50.0

    for i in range(16):
        if i in all_centers:
            continue
        nearest_idx = min(all_centers.keys(), key=lambda k: abs(k - i))
        all_centers[i] = all_centers[nearest_idx] + (i - nearest_idx) * avg_spacing

    boundaries: List[Tuple[float, float]] = []
    for i in range(16):
        if i == 0:
            left = 0.0
        else:
            left = (all_centers[i - 1] + all_centers[i]) / 2
        if i == 15:
            right = float(image_width)
        else:
            right = (all_centers[i] + all_centers[i + 1]) / 2
        boundaries.append((left, right))

    return boundaries


def _find_spanning_columns(
    box: OCRBox, boundaries: List[Tuple[float, float]]
) -> Tuple[int, int]:
    """Find the column range a box spans."""
    bx_left, bx_right = box.box[0], box.box[2]
    start_col = 0
    end_col = 0
    for i, (bl, br) in enumerate(boundaries):
        mid = (bl + br) / 2
        if bx_left <= mid:
            start_col = i
            break
    for i in range(15, -1, -1):
        bl, br = boundaries[i]
        mid = (bl + br) / 2
        if bx_right >= mid:
            end_col = i
            break
    if end_col < start_col:
        end_col = start_col
    return start_col, end_col


def _box_row_to_dict(
    box_row: List[OCRBox],
    boundaries: List[Tuple[float, float]],
    track_number: str = "",
) -> Dict[str, str]:
    """Convert a row of OCRBox objects to a 16-column dict."""
    row = {col: "" for col in TYPE1_COLUMNS}
    if track_number:
        row["股道"] = track_number

    for box in box_row:
        start_col, end_col = _find_spanning_columns(box, boundaries)
        parts = box.text.split()

        # Extend span when text has more parts than detected columns
        # (OCR merges adjacent column values into one box)
        if parts and len(parts) > (end_col - start_col + 1):
            desired_end = start_col + len(parts) - 1
            if desired_end < 16:
                ext_left, _ = boundaries[desired_end]
                if box.box[2] > ext_left:
                    end_col = desired_end

        n_cols = end_col - start_col + 1

        if n_cols == 1:
            key = TYPE1_COLUMNS[start_col]
            row[key] = (row[key] + " " + box.text).strip()
        else:
            if not parts:
                continue

            if len(parts) >= n_cols:
                for j in range(n_cols):
                    key = TYPE1_COLUMNS[start_col + j]
                    if j < len(parts):
                        row[key] = (row[key] + " " + parts[j]).strip()
            elif len(parts) == 1:
                text = parts[0]
                m = re.match(r'^(\d{1,3})([A-Za-z].*)$', text)
                if m and start_col <= 1 and end_col >= 2:
                    row["序"] = m.group(1)
                    row["车种"] = normalize_vehicle_type(m.group(2))
                else:
                    mid_col = (start_col + end_col) // 2
                    row[TYPE1_COLUMNS[mid_col]] = text
            else:
                for j, part in enumerate(parts):
                    col_idx = start_col + j
                    if col_idx <= end_col:
                        row[TYPE1_COLUMNS[col_idx]] = part

    if row["车种"]:
        row["车种"] = normalize_vehicle_type(row["车种"])

    return row


def extract_type1_columns(
    ocr_results: List[OCRBox], tolerance: int = None
) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    """
    Type 1 (站存车打印) extraction with 16-column mapping.
    Returns (metadata, list_of_row_dicts).
    """
    import logging
    logger = logging.getLogger(__name__)

    box_rows = aggregate_to_box_rows(ocr_results, tolerance)
    if not box_rows:
        return {}, []

    image_width = max(b.box[2] for b in ocr_results)

    # Find primary header row and merge with adjacent secondary header rows
    header_indices: set = set()
    header_boxes: List[OCRBox] = []
    primary_idx = -1

    for i, brow in enumerate(box_rows):
        if _is_header_box_row(brow):
            primary_idx = i
            header_indices.add(i)
            header_boxes.extend(brow)
            break

    if primary_idx >= 0:
        for adj_idx in [primary_idx - 1, primary_idx + 1]:
            if 0 <= adj_idx < len(box_rows) and adj_idx not in header_indices:
                adj_row = box_rows[adj_idx]
                if _is_secondary_header_row(adj_row):
                    for box in adj_row:
                        box_text = box.text.replace(' ', '')
                        if any(kw in box_text for kw in _HEADER_COL_MAP):
                            header_boxes.append(box)
                    header_indices.add(adj_idx)

    last_header_idx = max(header_indices) if header_indices else -1

    if header_boxes:
        col_centers = _parse_column_centers(header_boxes)
        logger.debug(f"Header column centers: {col_centers}")
        boundaries = _build_column_boundaries(col_centers, image_width)
    else:
        logger.warning("Header row not found, using uniform boundaries")
        step = image_width / 16
        boundaries = [(i * step, (i + 1) * step) for i in range(16)]

    metadata: Dict[str, Any] = {}
    data_box_rows: List[List[OCRBox]] = []
    track_number = ""

    for i, brow in enumerate(box_rows):
        if i in header_indices:
            continue

        row_text = ''.join(b.text for b in brow)

        if re.match(r'^第\d+页$', row_text.strip()):
            continue
        if _is_header_box_row(brow):
            continue
        if last_header_idx >= 0 and i < last_header_idx:
            continue

        data_box_rows.append(brow)

    # Detect track number from first data row
    if data_box_rows:
        first_row_boxes = data_box_rows[0]
        if len(first_row_boxes) >= 2:
            leftmost = first_row_boxes[0]
            second = first_row_boxes[1]
            left_text = leftmost.text.strip()
            second_text = second.text.strip()

            if re.match(r'^\d{1,2}$', left_text):
                seq_match = re.match(r'^[12]\b', second_text)
                if seq_match:
                    track_number = left_text
                    data_box_rows[0] = first_row_boxes[1:]

    table_data: List[Dict[str, str]] = []
    for brow in data_box_rows:
        row_text = ''.join(b.text for b in brow)
        if not re.search(r'\d', row_text):
            continue

        row_dict = _box_row_to_dict(brow, boundaries, track_number)

        if row_dict["序"] or row_dict["车号"]:
            table_data.append(row_dict)

    return metadata, table_data
