"""
OCR Utility Functions

Helper functions for text correction, splitting, and row processing.
"""

import re
from typing import List, Tuple, Optional, Dict, Any

from .models import OCRBox


# ── Patterns for Type 2 (集装箱编组单) detection ──
_CONTAINER_RE = re.compile(r'[A-Z]{3,4}[A-Z0-9]\d{6,7}')
_SLASH_VEHICLE_RE = re.compile(r'[A-Za-z]\w*/\d{5,}')


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


def extract_type2(all_rows: List[List[str]]) -> Tuple[Dict[str, Any], List[List[str]]]:
    """
    Type 2 (集装箱编组单) extraction:
    Filter headers/footers/metadata, keep data rows, then post-process:
    - Cap rows at MAX_COLS (seq, vehicle, container1, container2, station)
    - Infer missing sequence numbers
    - Filter fragment rows
    """
    MAX_COLS = 5
    MIN_INFERRED = 4

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
            # Clean leading OCR noise (single letters like "J", "I")
            while len(row) > 1 and re.match(r'^[A-Za-z]$', row[0].strip()):
                row = row[1:]
            raw_rows.append(row)

    # Post-process: infer missing seq, cap length, filter fragments
    data_rows: List[List[str]] = []
    next_seq = 1

    for row in raw_rows:
        first = row[0].strip()
        has_seq = bool(re.match(r'^\d{1,3}$', first))

        if has_seq:
            next_seq = int(first) + 1
            data_rows.append(row[:MAX_COLS])
        else:
            row_out = [str(next_seq)] + row
            row_out = row_out[:MAX_COLS]
            if len(row_out) >= MIN_INFERRED:
                next_seq += 1
                data_rows.append(row_out)

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
