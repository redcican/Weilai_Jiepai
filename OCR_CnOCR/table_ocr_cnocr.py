#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
表格OCR提取工具 - CnOCR优化版
参考PaddleOCR实现，优化识别精度
"""

import json
import sys
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import time

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


@dataclass
class OCRBox:
    """OCR识别结果"""
    box: List[int]
    text: str
    confidence: float = 0.0


def enhance_image_for_ocr(image_path: str):
    """
    增强图像以提高OCR识别率
    通过提高对比度和锐度来帮助识别
    """
    try:
        from PIL import Image, ImageEnhance
        import numpy as np

        img = Image.open(image_path)

        if img.mode != 'RGB':
            img = img.convert('RGB')

        # 增强对比度
        contrast_enhancer = ImageEnhance.Contrast(img)
        img = contrast_enhancer.enhance(1.3)

        # 增强锐度
        sharpness_enhancer = ImageEnhance.Sharpness(img)
        img = sharpness_enhancer.enhance(1.5)

        return np.array(img)

    except Exception as e:
        logger.debug(f"图像增强失败: {e}")
        return None


def correct_ocr_text(text: str) -> str:
    """
    修正常见的OCR识别错误
    """
    if not text:
        return text

    # 车型中的 O/0/e 混淆修正和大小写修正
    def fix_vehicle_type(match):
        s = match.group(0).upper()  # 先转大写
        result = []
        for i, c in enumerate(s):
            # O/E -> 0 当在特定上下文时
            # 例如 C7OE -> C70E, C7EE -> C70E
            if c == 'O' and i > 0:
                if s[i-1].isdigit() or (i < len(s)-1 and (s[i+1].isdigit() or s[i+1] == 'E' or s[i+1] == 'H')):
                    result.append('0')
                    continue
            # 处理 C7EE -> C70E 的情况 (第一个E应该是0)
            if c == 'E' and i > 0 and i < len(s)-1:
                if s[i-1].isdigit() and s[i+1] == 'E':
                    result.append('0')
                    continue
            result.append(c)
        return ''.join(result)

    # 修正车型 (匹配 C70, NX70AF, X70 等)
    text = re.sub(r'[A-Za-z]+[0-9OoEeHh]+[A-Za-z]*', fix_vehicle_type, text)

    # 修正截断的车型
    # "C7" -> "C70" (只有C7没有后缀)
    if re.match(r'^[Cc]7$', text):
        text = 'C70'
    # "C64" -> "C64K" (C64系列常见为C64K)
    elif re.match(r'^[Cc]64$', text):
        text = 'C64K'

    # 修正常见的识别错误
    corrections = {
        '敬二': '敞二',
        '散二': '敞二',
        'θ': '0',  # 希腊字母theta误识别为0
        'Θ': '0',
        'о': '0',  # 西里尔字母o
        'О': '0',
    }

    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)

    return text


class CnOCREngine:
    """CnOCR引擎 - 优化版"""

    def __init__(self, enhance_image: bool = True):
        self._available = False
        self.ocr = None
        self.enhance_image = enhance_image

        try:
            from cnocr import CnOcr

            # 使用ONNX后端加速，选择更准确的模型
            self.ocr = CnOcr(
                rec_model_name='densenet_lite_136-gru',
                det_model_name='ch_PP-OCRv3_det',
                rec_model_backend='onnx',
                det_model_backend='onnx',
            )
            self._available = True
            logger.info("CnOCR (ONNX) 引擎初始化成功")
        except Exception as e:
            logger.error(f"CnOCR初始化失败: {e}")

    @property
    def available(self) -> bool:
        return self._available

    def recognize(self, image_path: str) -> List[OCRBox]:
        """执行OCR识别，支持图像增强"""
        if not self._available:
            return []

        try:
            # 图像增强
            if self.enhance_image:
                enhanced = enhance_image_for_ocr(image_path)
                if enhanced is not None:
                    result = self.ocr.ocr(enhanced)
                else:
                    result = self.ocr.ocr(image_path)
            else:
                result = self.ocr.ocr(image_path)
        except Exception as e:
            logger.warning(f"OCR识别失败: {e}")
            return []

        if not result:
            return []

        results = []
        for item in result:
            text = item.get('text', '').strip()
            if not text:
                continue

            # 应用OCR文本修正
            text = correct_ocr_text(text)

            score = item.get('score', 0.0)
            position = item.get('position', None)

            if position is not None:
                try:
                    x_coords = [p[0] for p in position]
                    y_coords = [p[1] for p in position]
                    box = [int(min(x_coords)), int(min(y_coords)),
                           int(max(x_coords)), int(max(y_coords))]
                except (TypeError, ValueError, IndexError):
                    box = [0, len(results) * 30, 100, (len(results) + 1) * 30]
            else:
                box = [0, len(results) * 30, 100, (len(results) + 1) * 30]

            results.append(OCRBox(box=box, text=text, confidence=float(score)))

        return results


def split_merged_text(text: str) -> List[str]:
    """智能分割OCR合并的文本"""
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

        # 模式: "XXX空X" + "XXX箱"
        match = re.match(r'^(.+空\d+?)(.+箱.*)$', part)
        if match:
            result.append(match.group(1))
            result.extend(split_merged_text(match.group(2)))
            continue

        # 模式: "XXX空X" 后跟 "部/局..."
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
    """递归分割被错误合并的数字"""
    if not text or not text.strip():
        return []

    text = text.strip()

    # 序列号和车型合并 (e.g., "21C70" -> ["21", "C70"])
    match = re.match(r'^(\d{1,3})([A-Za-z].*)$', text)
    if match:
        return [match.group(1), match.group(2)]

    # 模式: 任意数字开头 + 小数 + 小数 + 数字 (e.g., "166444222.71.35")
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

    # 模式: 任意数字开头 + 小数 + 小数 (无后缀)
    match = re.match(r'^(.+?)(\d{2}\.\d)(\d\.\d+)$', text)
    if match:
        prefix = match.group(1)
        first_decimal = match.group(2)
        second_decimal = match.group(3)

        result = split_merged_numbers(prefix)
        result.append(first_decimal)
        result.append(second_decimal)
        return result

    # 模式: 数字 + 小数 + 小数
    match = re.match(r'^(\d+\.\d)(\d\.\d+)$', text)
    if match:
        return [match.group(1), match.group(2)]

    # 模式: 7位车号+2位小数
    match = re.match(r'^(\d{7})(\d{2}\.\d+)$', text)
    if match:
        return [match.group(1), match.group(2)]

    # 模式: 7位车号+1位+2位小数
    match = re.match(r'^(\d{7})(\d)(\d{2}\.\d+)$', text)
    if match:
        return [match.group(1), match.group(2), match.group(3)]

    # 模式: 8位车号+2位小数
    match = re.match(r'^(\d{8})(\d{2}\.\d+)$', text)
    if match:
        return [match.group(1), match.group(2)]

    # 模式: 6位车号+2位小数
    match = re.match(r'^(\d{6})(\d{2}\.\d+)$', text)
    if match:
        return [match.group(1), match.group(2)]

    # 模式: 整数+小数合并
    match = re.match(r'^(\d)(\d{2}\.\d+)$', text)
    if match:
        return [match.group(1), match.group(2)]

    # 处理空格分隔
    if ' ' in text:
        parts = text.split()
        result = []
        for part in parts:
            result.extend(split_merged_numbers(part))
        return result

    return [text]


def aggregate_to_rows(ocr_results: List[OCRBox], tolerance: int = None) -> List[List[str]]:
    """将OCR结果按行聚合，使用中心Y坐标，自适应容差"""
    if not ocr_results:
        return []

    def center_y(box):
        return (box.box[1] + box.box[3]) / 2

    def box_height(box):
        return box.box[3] - box.box[1]

    # 自适应容差：使用文本高度的中位数的60%
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
            # 更新平均Y
            current_y = sum(center_y(r) for r in current_row) / len(current_row)
        else:
            # 保存当前行
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
    """自动检测序列号列"""
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
    """判断是否为元数据项"""
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
    """判断是否为表头行"""
    if not row:
        return False
    header_keywords = {
        '车号', '到站', '品名', '记事', '发站', '票据', '自重', '换长', '载重',
        '序号', '站存车打印', '集装箱', '箱号', '油种', '蓬布', '属性', '收货人',
    }
    row_text = ''.join(row)
    return any(kw in row_text for kw in header_keywords)


def is_page_footer(row: List[str]) -> bool:
    """判断是否为页脚（如 第1页、第2页）"""
    if not row:
        return False
    row_text = ''.join(row).strip()
    return bool(re.match(r'^第\d+页$', row_text))


# ── Patterns for Type 2 (集装箱编组单) ──
_CONTAINER_RE = re.compile(r'[A-Z]{3,4}[A-Z0-9]\d{6,7}')
_SLASH_VEHICLE_RE = re.compile(r'[A-Za-z]\w*/\d{5,}')
_CHINESE_RE = re.compile(r'[\u4e00-\u9fff]')

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


def aggregate_to_box_rows(ocr_results: List[OCRBox], tolerance: int = None) -> List[List[OCRBox]]:
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
    """Check if a box row is a header row (primary: ≥3 keywords)."""
    text = ''.join(b.text for b in box_row)
    header_kw = ['车号', '到站', '品名', '自重', '换长', '载重', '记事']
    return sum(1 for kw in header_kw if kw in text) >= 3


def _is_secondary_header_row(box_row: List[OCRBox]) -> bool:
    """Check if a row is a secondary header row (≥1 header keyword, typically left-side columns)."""
    text = ''.join(b.text for b in box_row)
    all_kw = ['股道', '序', '车种', '油种', '车号', '到站', '品名', '自重', '换长', '载重', '记事', '发站', '篷布', '票据号', '属性', '收货人']
    return sum(1 for kw in all_kw if kw in text) >= 1


def _parse_column_centers(header_boxes: List[OCRBox]) -> Dict[int, float]:
    """Extract column center x-positions from header boxes using proportional character mapping."""
    col_centers: Dict[int, float] = {}

    # Sort keys by length descending for greedy longest-match
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
                    if col_idx >= 0:  # skip non-column markers like 站存车打印
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
    # Fill gaps via interpolation
    known = sorted(col_centers.items())
    if not known:
        return [(0, image_width)] * 16

    all_centers: Dict[int, float] = dict(known)

    # Estimate average spacing from known columns
    spacings = []
    for i in range(len(known) - 1):
        idx_a, x_a = known[i]
        idx_b, x_b = known[i + 1]
        if idx_b > idx_a:
            spacings.append((x_b - x_a) / (idx_b - idx_a))
    avg_spacing = sum(spacings) / len(spacings) if spacings else 50.0

    # Fill missing columns by extrapolation from nearest known
    for i in range(16):
        if i in all_centers:
            continue
        # Find nearest known column
        nearest_idx = min(all_centers.keys(), key=lambda k: abs(k - i))
        all_centers[i] = all_centers[nearest_idx] + (i - nearest_idx) * avg_spacing

    # Build boundaries using midpoints
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
                # Verify the box physically reaches into the extended area
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

    # Normalize vehicle type
    if row["车种"]:
        row["车种"] = normalize_vehicle_type(row["车种"])

    return row


def _extract_type1_columns(
    ocr_results: List[OCRBox], tolerance: int = None
) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    """
    Type 1 (站存车打印) extraction with 16-column mapping.
    Returns (metadata, list_of_row_dicts).
    """
    box_rows = aggregate_to_box_rows(ocr_results, tolerance)
    if not box_rows:
        return {}, []

    # Find image width
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

    # Check adjacent rows for secondary header content (split headers)
    if primary_idx >= 0:
        for adj_idx in [primary_idx - 1, primary_idx + 1]:
            if 0 <= adj_idx < len(box_rows) and adj_idx not in header_indices:
                adj_row = box_rows[adj_idx]
                if _is_secondary_header_row(adj_row):
                    # Only merge boxes that contain header keywords
                    for box in adj_row:
                        box_text = box.text.replace(' ', '')
                        if any(kw in box_text for kw in _HEADER_COL_MAP):
                            header_boxes.append(box)
                    header_indices.add(adj_idx)

    last_header_idx = max(header_indices) if header_indices else -1

    # Parse column boundaries
    if header_boxes:
        col_centers = _parse_column_centers(header_boxes)
        logger.debug(f"Header column centers: {col_centers}")
        boundaries = _build_column_boundaries(col_centers, image_width)
    else:
        logger.warning("Header row not found, using uniform boundaries")
        step = image_width / 16
        boundaries = [(i * step, (i + 1) * step) for i in range(16)]

    # Separate metadata rows, track number, and data rows
    metadata: Dict[str, Any] = {}
    data_box_rows: List[List[OCRBox]] = []
    track_number = ""

    for i, brow in enumerate(box_rows):
        if i in header_indices:
            continue

        row_text = ''.join(b.text for b in brow)

        # Skip page footers
        if re.match(r'^第\d+页$', row_text.strip()):
            continue

        # Skip header-like rows
        if _is_header_box_row(brow):
            continue

        # Skip title/date lines above header
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

    # Convert each data row to dict
    table_data: List[Dict[str, str]] = []
    for brow in data_box_rows:
        row_text = ''.join(b.text for b in brow)
        if not re.search(r'\d', row_text):
            continue

        row_dict = _box_row_to_dict(brow, boundaries, track_number)

        if row_dict["序"] or row_dict["车号"]:
            table_data.append(row_dict)

    return metadata, table_data


def detect_table_type(ocr_results: List[OCRBox]) -> int:
    """
    判断表格类型:
      Type 1 — 站存车打印 (~16列, 车种/车号分开)
      Type 2 — 集装箱编组单 (~5列, 含集装箱号和斜杠车种/车号)

    核心依据: 斜杠车种/车号 (如 C70E/1805776) 是 Type 2 的唯一特征。
    集装箱号模式不可靠，因为 Type 1 中也有类似的货运参考号 (如 JHSX4535071)。
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
    按模式分类 Type 2 行中的每个值，而非按位置。

    分类规则:
    - 位置 0 的 1-3 位数字 → 序
    - 斜杠车号 (如 C70E/1721133) → ID1
    - 集装箱号 (如 TBJU3216534) → ID2，再 ID3
    - 中文文本 (如 漳平) → 地点
    - 其他无法识别的值 → 噪声跳过

    碎片行（无任何有效数据）返回 None。
    """
    if not row:
        return None, next_seq

    first = row[0].strip()
    has_real_seq = bool(re.match(r'^\d{1,3}$', first))

    if has_real_seq:
        seq = first
        data_vals = row[1:]
    elif _SLASH_VEHICLE_RE.search(first) or _CONTAINER_RE.search(first):
        # 位置 0 直接就是数据（序号缺失）
        seq = str(next_seq)
        data_vals = row
    else:
        # 位置 0 是噪声/OCR 损坏的序号（如 \\y, 寸, o）
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
        # 其他值视为噪声跳过

    # 碎片行：无任何有效数据 → 过滤
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


def _extract_type2(all_rows: List[List[str]]) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    """
    Type 2 (集装箱编组单) 提取:
    过滤表头/页脚/元数据，保留数据行，
    然后按模式分类每个值到对应列（而非按位置）。
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

        # 保留含集装箱号、斜杠车号、或以序号开头的行
        row_text = ''.join(row)
        has_container = bool(_CONTAINER_RE.search(row_text))
        has_slash = bool(_SLASH_VEHICLE_RE.search(row_text))
        has_seq = bool(row[0].strip()) and re.match(r'^\d{1,3}$', row[0].strip())

        if has_container or has_slash or has_seq:
            raw_rows.append(row)

    # ── 按模式分类 ──
    data_rows: List[Dict[str, str]] = []
    next_seq = 1

    for row in raw_rows:
        row_dict, next_seq = _classify_type2_row(row, next_seq)
        if row_dict is not None:
            data_rows.append(row_dict)

    return metadata, data_rows


def detect_vehicle_type_column(rows: List[List[str]]) -> int:
    """自动检测车型列"""
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
    """自动检测车辆编号列（6-8位数字）"""
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
    """判断是否为有效数据行"""
    if not row or len(row) < min_cols:
        return False
    if is_header_row(row):
        return False

    content = ''.join(str(c).strip() for c in row)
    return len(content) > 5


def normalize_vehicle_type(cell: str) -> str:
    """规范化车型代码"""
    cell = cell.strip()

    if re.match(r'^[A-Za-z]', cell):
        return cell

    if re.match(r'^(70|64|62)\w*$', cell):
        return 'C' + cell

    return cell


def normalize_row(row: List[str], vehicle_col: int = -1) -> List[str]:
    """规范化数据行"""
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


def extract_track_number_from_first_row(rows: List[List[str]], seq_col: int) -> Tuple[Optional[str], List[List[str]]]:
    """从第一行提取股道号"""
    if not rows or len(rows) < 2:
        return None, rows

    first_row = rows[0]
    if not first_row or len(first_row) < 2:
        return None, rows

    first_val = first_row[0].strip()

    if not re.match(r'^\d{1,2}$', first_val):
        return None, rows

    first_num = int(first_val)

    # 模式1：检查第一行是否比后续行多一列
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

    # 模式2：第一行格式 [大数字, 1, ...], 后续行 [2, ...], [3, ...]
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

    # 模式3：第一行的第一个数字明显不符合递增序列
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


def extract_table_data(image_path: str, engine: CnOCREngine, tolerance: int = None) -> Dict[str, Any]:
    """从单张图片提取表格数据，自动检测表格类型"""
    ocr_results = engine.recognize(str(image_path))

    if not ocr_results:
        return {
            "message": f"图片 {Path(image_path).name} OCR识别失败",
            "status": "error",
            "table_type": 0,
            "metadata": {},
            "table data": []
        }

    table_type = detect_table_type(ocr_results)
    logger.info(f"{Path(image_path).name}: 检测为类型 {table_type}")

    # 聚合为行
    all_rows = aggregate_to_rows(ocr_results, tolerance)

    # ── Type 2: 集装箱编组单 ──
    if table_type == 2:
        metadata, data_rows = _extract_type2(all_rows)
        return {
            "message": "识别成功",
            "status": "success",
            "table_type": 2,
            "metadata": metadata,
            "table data": data_rows
        }

    # ── Type 1: 站存车打印 (16-column dict extraction) ──
    metadata, row_dicts = _extract_type1_columns(ocr_results, tolerance)

    return {
        "message": "识别成功",
        "status": "success",
        "table_type": 1,
        "metadata": metadata,
        "table data": row_dicts
    }


def process_folder(folder_path: str, output_dir: str, recursive: bool = False, tolerance: int = None):
    """处理整个文件夹"""
    folder = Path(folder_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if recursive:
        image_files = [f for f in folder.rglob('*') if f.suffix.lower() in IMAGE_EXTENSIONS]
    else:
        image_files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]

    image_files = sorted(image_files)

    if not image_files:
        logger.warning(f"文件夹 {folder_path} 中没有找到图片")
        return [], 0

    logger.info(f"找到 {len(image_files)} 个图片文件")

    engine = CnOCREngine(enhance_image=False)  # 禁用图像增强以提升速度
    if not engine.available:
        logger.error("OCR引擎不可用")
        return [], 0

    results = []
    total_records = 0

    start_time = time.time()

    for img_path in image_files:
        try:
            img_start = time.time()
            result = extract_table_data(str(img_path), engine, tolerance)
            img_time = time.time() - img_start

            output_path = output_dir / f"{img_path.stem}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            record_count = len(result.get('table data', []))
            total_records += record_count
            meta_count = len(result.get('metadata', {}))
            logger.info(f"处理 {img_path.name}: {record_count} 条数据, {meta_count} 条元数据 ({img_time:.2f}s)")
            results.append(result)

        except Exception as e:
            logger.error(f"处理 {img_path} 失败: {e}")
            results.append({
                "message": f"处理失败: {e}",
                "status": "error",
                "metadata": {},
                "table data": []
            })

    total_time = time.time() - start_time
    logger.info(f"总耗时: {total_time:.2f}s, 平均每张: {total_time/len(image_files):.2f}s")

    return results, total_records


def main():
    import argparse

    parser = argparse.ArgumentParser(description='表格OCR提取工具 - CnOCR优化版')
    parser.add_argument('folder', help='输入图片文件夹路径')
    parser.add_argument('-o', '--output', default='./output', help='输出目录')
    parser.add_argument('-r', '--recursive', action='store_true', help='递归处理子文件夹')
    parser.add_argument('--tolerance', type=int, default=None, help='行聚合容差（默认自适应）')
    parser.add_argument('-v', '--verbose', action='store_true', help='详细输出')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        input_path = Path(args.folder)
        if not input_path.is_dir():
            raise ValueError(f"无效的文件夹路径: {args.folder}")

        results, total_records = process_folder(
            str(input_path),
            args.output,
            recursive=args.recursive,
            tolerance=args.tolerance
        )

        successful = sum(1 for r in results if r.get('status') == 'success')

        print(f"\n处理完成:")
        print(f"  - 文件数: {len(results)} 个 (成功: {successful})")
        print(f"  - 总记录: {total_records} 条")
        print(f"  - 输出目录: {args.output}/")

    except Exception as e:
        logger.error(f"错误: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
