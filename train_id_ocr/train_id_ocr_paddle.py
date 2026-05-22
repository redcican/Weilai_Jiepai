#!/usr/bin/env python3
"""
Train ID OCR — PaddleOCR 单图识别版 (修改版)

修改内容：
  1. lang='ch' 支持中文识别，同时过滤所有纯中文文本框
  2. 恢复置信度传播，合并/去重时保留最高置信度候选
  3. 车种纠错替换为 CnOCR 版的位置感知分段策略

Usage:
    python train_id_ocr_paddle.py image.jpg
    python train_id_ocr_paddle.py ./images/ -o ./output
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import re
import json
import argparse
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Union
from collections import Counter
from itertools import combinations
from pathlib import Path
from enum import Enum

import cv2
import numpy as np
from paddleocr import PaddleOCR


# ============ 集装箱前缀纠错映射表 ============
PREFIX_CORRECTION = {
    'TEJU': 'TBJU', 'TRJU': 'TBJU', 'TPJU': 'TBJU', 'T8JU': 'TBJU',
    'T3JU': 'TBJU', 'TBIU': 'TBJU', 'T0JU': 'TBJU',
    'TB1U': 'TBJU', 'TBJ0': 'TBJU', 'T9JU': 'TBJU',
    'IBJU': 'TBJU', 'I BJU': 'TBJU',
    'IBCU': 'TBCU', 'TBCU': 'TBCU', '1BCU': 'TBCU',
}

PREFIX_DIGIT_FIX = {
    '1': 'T', '3': 'B', '8': 'B', '0': 'O',
    '5': 'S', '2': 'Z', '7': 'T', '4': 'A',
    '6': 'G', '9': 'G',
}

COMMON_PREFIXES = {
    'TBJU', 'TBCU', 'TRLU', 'TGHU', 'TCNU', 'EMCU', 'UESU',
    'HJCU', 'BSIU', 'APRU', 'MEDU', 'MSCW', 'MRSU', 'CSNU',
    'OOLU', 'CMAU', 'EGLV', 'HLCU', 'ONEU', 'YMLU', 'MAEU',
    'COSU', 'UASC', 'KLINE'
}

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


# ============ 空挡检测器（不训练，纯 CV）============

class GapType(Enum):
    """空挡检测结果类型"""
    GAP = "gap"
    NORMAL = "normal"
    TRANSITION = "transition"


@dataclass
class GapResult:
    """空挡检测结果"""
    gap_type: GapType
    vert_edge_ratio: float
    center_std: float
    score: float


class GapDetector:
    """
    空挡检测器 - 严格多条件组合版 (方案B)

    核心原理：车厢连接处的空挡区域同时满足多个图像特征：
      1. 中间有竖直金属结构（梯子/栏杆）-> 竖直边缘密度高
      2. 背景有开口和灯光 -> CLAHE后低梯度占比适中（不太高）
      3. 灯光可见 -> 亮斑连通域存在
      4. 不是黄色/白色机车侧面 -> 平均亮度不太高
    """

    def __init__(
        self,
        roi_x: Tuple[float, float] = (0.45, 0.55),
        roi_y: Tuple[float, float] = (0.2, 0.9),
        # 方案A 参数（兼容旧版）
        edge_thresh: int = 50,
        gap_thresh: float = 0.018,
        normal_thresh: float = 0.005,
        # 方案B 严格组合参数
        strict_mode: bool = True,
        min_vert_ratio: float = 0.015,
        max_low_grad_ratio: float = 0.75,
        max_brightness: float = 140.0,
        min_bright_blob_ratio: float = 0.02,
    ):
        self.roi_x = roi_x
        self.roi_y = roi_y
        self.edge_thresh = edge_thresh
        self.gap_thresh = gap_thresh
        self.normal_thresh = normal_thresh
        self.strict_mode = strict_mode
        # 严格组合：避免将正常车厢误判为空挡
        self.min_vert_ratio = min_vert_ratio
        self.max_low_grad_ratio = max_low_grad_ratio
        self.max_brightness = max_brightness
        self.min_bright_blob_ratio = min_bright_blob_ratio

    def _extract_roi(self, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape
        x1 = int(w * self.roi_x[0])
        x2 = int(w * self.roi_x[1])
        y1 = int(h * self.roi_y[0])
        y2 = int(h * self.roi_y[1])
        return gray[y1:y2, x1:x2]

    def _compute_vert_edge_ratio(self, roi: np.ndarray) -> float:
        if roi.size == 0:
            return 0.0
        sobel_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
        mag = np.abs(sobel_x)
        strong_pixels = np.sum(mag > self.edge_thresh)
        return float(strong_pixels / mag.size)

    def _compute_std(self, roi: np.ndarray) -> float:
        return float(np.std(roi))

    def _compute_low_grad_ratio(self, roi: np.ndarray) -> float:
        """CLAHE后梯度<20的像素占比"""
        if roi.size == 0:
            return 1.0
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        roi_clahe = clahe.apply(roi)
        grad_x = cv2.Sobel(roi_clahe, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(roi_clahe, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        return float(np.sum(grad_mag < 20) / grad_mag.size)

    def _compute_bright_blob_ratio(self, roi: np.ndarray, thresh: int = 220) -> float:
        """亮斑最大连通域占比"""
        if roi.size == 0:
            return 0.0
        _, binary = cv2.threshold(roi, thresh, 255, cv2.THRESH_BINARY)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        max_area = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= 10 and area > max_area:
                max_area = area
        return float(max_area / roi.size) if roi.size > 0 else 0.0

    def detect(self, image: Union[np.ndarray, str, Path]) -> GapResult:
        if isinstance(image, (str, Path)):
            data = np.fromfile(str(image), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                return GapResult(GapType.TRANSITION, 0.0, 0.0, 0.0)
        else:
            img = image

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        roi = self._extract_roi(gray)
        vert_ratio = self._compute_vert_edge_ratio(roi)
        center_std = self._compute_std(roi)

        if self.strict_mode:
            # 方案B: 严格多条件组合
            low_grad_ratio = self._compute_low_grad_ratio(roi)
            bright_blob_ratio = self._compute_bright_blob_ratio(roi)
            brightness_mean = float(np.mean(roi))

            is_gap = (
                vert_ratio > self.min_vert_ratio and
                low_grad_ratio < self.max_low_grad_ratio and
                brightness_mean < self.max_brightness and
                bright_blob_ratio > self.min_bright_blob_ratio
            )

            gap_type = GapType.GAP if is_gap else GapType.NORMAL
            # 综合评分：基于 vert_ratio 的 sigmoid，仅用于调试参考
            score = 1.0 / (1.0 + np.exp(-200 * (vert_ratio - 0.01)))
            return GapResult(gap_type, vert_ratio, center_std, round(score, 4))
        else:
            # 方案A: 仅竖直边缘密度
            if vert_ratio >= self.gap_thresh:
                gap_type = GapType.GAP
            elif vert_ratio <= self.normal_thresh:
                gap_type = GapType.NORMAL
            else:
                gap_type = GapType.TRANSITION

            score = 1.0 / (1.0 + np.exp(-200 * (vert_ratio - 0.01)))
            return GapResult(gap_type, vert_ratio, center_std, round(score, 4))


# ============ CnOCR 版字符混淆表（车种纠错用）============
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


@dataclass
class OCRBox:
    """Single OCR detection (aligned with CnOCR version)."""
    box: List[int]      # [x_min, y_min, x_max, y_max]
    text: str
    confidence: float


@dataclass
class TextBox:
    text: str
    conf: float
    center_x: float
    center_y: float
    width: float
    height: float


@dataclass
class ImageResult:
    """单张图片的识别结果（带置信度）"""
    containers: List[Tuple[str, float]] = field(default_factory=list)
    train_types: List[Tuple[str, float]] = field(default_factory=list)
    train_numbers: List[Tuple[str, float]] = field(default_factory=list)
    is_gap: bool = False  # 标记是否为空挡帧


# ============ 工具函数 ============
def clean_digits(digits: str) -> str:
    mapping = str.maketrans({
        'i': '1', 'I': '1', 'l': '1', 'L': '1',
        'o': '0', 'O': '0', 'Q': '0',
        'g': '9', 'q': '9', 'G': '6',
        'b': '6', 'B': '8',
        's': '5', 'S': '5',
        'z': '2', 'Z': '2',
        'a': '4', 'A': '4',
    })
    return digits.translate(mapping)


def fix_prefix_digits(prefix: str) -> Optional[str]:
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
    result = ''.join(fixed)
    if result in COMMON_PREFIXES or result in PREFIX_CORRECTION.values():
        return result
    for white in COMMON_PREFIXES:
        diff = sum(1 for a, b in zip(result, white) if a != b)
        if diff <= 1:
            return white
    return None


def correct_prefix(prefix: str) -> str:
    if prefix in PREFIX_CORRECTION:
        return PREFIX_CORRECTION[prefix]
    if prefix in COMMON_PREFIXES:
        return prefix
    return prefix


# ============ 中文过滤 ============
def _is_pure_chinese(text: str) -> bool:
    """判断是否纯中文文本（应过滤）"""
    t = text.strip()
    return len(t) > 0 and all('\u4e00' <= c <= '\u9fff' for c in t)


# ============ 集装箱提取（恢复置信度）============
def extract_container_id(text: str, conf: float) -> Optional[Tuple[str, float]]:
    text = text.upper().replace(" ", "").replace("-", "").replace(".", "")
    match = re.search(r'([A-Z]{4})(\d{6,7})', text)
    if match:
        prefix = match.group(1)
        digits = match.group(2)[:6]
        prefix = correct_prefix(prefix)
        return (f"{prefix}{digits}", conf)
    loose_match = re.search(r'([A-Z0-9]{4})(\d{6,7})', text)
    if loose_match:
        prefix = loose_match.group(1)
        digits = loose_match.group(2)[:6]
        fixed_prefix = fix_prefix_digits(prefix)
        if fixed_prefix:
            return (f"{fixed_prefix}{digits}", conf)
    for prefix_match in re.finditer(r'[A-Z0-9]{4}', text):
        prefix = prefix_match.group(0)
        prefix_pos = prefix_match.end()
        remaining = text[prefix_pos:]
        cleaned = clean_digits(remaining)
        digit_match = re.search(r'(\d{6,7})', cleaned)
        if digit_match:
            digits = digit_match.group(1)[:6]
            fixed_prefix = fix_prefix_digits(prefix)
            if fixed_prefix:
                fixed_prefix = correct_prefix(fixed_prefix)
                return (f"{fixed_prefix}{digits}", conf)
            if prefix.isalpha():
                prefix = correct_prefix(prefix)
                return (f"{prefix}{digits}", conf)
    return None


# ============ CnOCR 版铁路货车提取（位置感知分段纠错）============
def _fix_vehicle_type(text: str) -> str:
    """CnOCR 版：位置感知分段纠错。
    
    将车种字符串分段为：字母前缀 + 数字中段 + 字母后缀，
    前缀中的数字-like字符→字母，数字段中的字母-like字符→数字。
    """
    t = text.strip()
    t = re.sub(r"^[/(（\[{]+", "", t)
    t = re.sub(r"[/）)\]}.]+", "", t)
    t = t.upper()

    if not t:
        return ""

    # Remove trailing Q artefact
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


def _is_vehicle_type_pattern(text: str) -> bool:
    """CnOCR 版：判断是否像车种（字母+数字+可选后缀）。
    
    Rejects garbled numbers where all characters are digit-confusable.
    """
    t = text.strip().upper()
    t = re.sub(r"^[/(（]+", "", t)
    if not re.match(r"^[A-Z]+\d+[A-Z]*Q?$", t):
        return False
    fully_numeric = t.translate(_LETTER_TO_DIGIT)
    fully_numeric = re.sub(r"[^0-9]", "", fully_numeric)
    if len(fully_numeric) >= len(t):
        return False
    return True


def _fix_vehicle_number(text: str) -> str:
    """CnOCR 版：数字清理。"""
    t = text.strip()
    t = re.sub(r"[/\\.,;:!?'\"()\[\]{}]", " ", t)
    char_map = str.maketrans({
        "O": "0", "o": "0", "Q": "0", "D": "0",
        "I": "1", "l": "1", "i": "1", "t": "1",
        "S": "5", "s": "5",
        "B": "8", "b": "8",
        "N": "",  "n": "",
    })
    t = t.translate(char_map)
    t = re.sub(r"[^\d\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def is_train_param(text: str) -> bool:
    """判断是否为货车参数框（载重/自重/容积等），应排除。"""
    t = text.lower()
    if any(c in t for c in ['t', 'm³', 'm3', '载', '重', '自', '容', '积', '换', '长', '定']):
        return True
    if any(c in text for c in ['.', '(', ')', 'X', '×', '*']):
        return True
    if _is_pure_chinese(text):
        return True
    if any(c in text for c in ['#', '+', '-']):
        return True
    if text.isalpha() and len(text) <= 3 and text.upper() not in ['C70', 'C64', 'P64', 'P70', 'N17']:
        return True
    return False


# ============ OCR 框解析（上下分区 + 中文过滤）============
def parse_ocr_boxes(ocr_boxes: List[OCRBox], img_height: int) -> Tuple[List[TextBox], List[TextBox]]:
    """解析 OCR 框列表，按 y 坐标分上半区(集装箱)/下半区(铁路货车)，过滤纯中文。"""
    upper_boxes = []
    lower_boxes = []
    split_y = img_height * 0.55

    for ob in ocr_boxes:
        # 过滤纯中文文本框
        if _is_pure_chinese(ob.text):
            continue

        x_min, y_min, x_max, y_max = ob.box
        box = TextBox(
            text=ob.text,
            conf=ob.confidence,
            center_x=(x_min + x_max) / 2,
            center_y=(y_min + y_max) / 2,
            width=x_max - x_min,
            height=y_max - y_min,
        )

        if box.center_y < split_y:
            upper_boxes.append(box)
        else:
            if not is_train_param(ob.text):
                lower_boxes.append(box)

    upper_boxes.sort(key=lambda b: b.center_x)
    lower_boxes.sort(key=lambda b: b.center_x)
    return upper_boxes, lower_boxes


# ============ CnOCR 版行分组（按 Y 坐标聚类）============
def _group_to_lines(
    boxes: List[TextBox],
    tolerance: Optional[int] = None,
) -> List[List[TextBox]]:
    """Group TextBoxes into horizontal lines by Y-centre."""
    if not boxes:
        return []

    if tolerance is None:
        heights = [b.height for b in boxes if b.height > 0]
        if heights:
            heights.sort()
            tolerance = max(30, int(heights[len(heights) // 2] * 0.6))
        else:
            tolerance = 50

    sorted_boxes = sorted(boxes, key=lambda b: b.center_y)
    lines: List[List[TextBox]] = []
    cur_line: List[TextBox] = []
    cur_y: Optional[float] = None

    for box in sorted_boxes:
        if cur_y is None:
            cur_y = box.center_y
            cur_line = [box]
        elif abs(box.center_y - cur_y) <= tolerance:
            cur_line.append(box)
            cur_y = sum(b.center_y for b in cur_line) / len(cur_line)
        else:
            cur_line.sort(key=lambda b: b.center_x)
            lines.append(cur_line)
            cur_line = [box]
            cur_y = box.center_y

    if cur_line:
        cur_line.sort(key=lambda b: b.center_x)
        lines.append(cur_line)

    return lines


# ============ 多框合并（恢复置信度）============
def is_y_close(b1: TextBox, b2: TextBox, threshold_ratio: float = 0.6) -> bool:
    y_threshold = max(b1.height, b2.height) * threshold_ratio
    return abs(b1.center_y - b2.center_y) < y_threshold


def merge_boxes(boxes: List[TextBox], extractor) -> List[Tuple[str, float]]:
    """多框合并：单框 + 相邻2框 + 相邻3框，保留并传播置信度。"""
    candidates: List[Tuple[str, float]] = []
    n = len(boxes)

    def try_match(text: str, conf: float) -> Optional[Tuple[str, float]]:
        return extractor(text, conf)

    # 单框
    for b in boxes:
        cid = try_match(b.text, b.conf)
        if cid:
            candidates.append(cid)

    # 相邻2框
    for i in range(n - 1):
        b1, b2 = boxes[i], boxes[i + 1]
        if not is_y_close(b1, b2):
            continue
        merged_conf = (b1.conf + b2.conf) / 2
        merged_text = b1.text + b2.text
        cid = try_match(merged_text, merged_conf)
        if cid:
            candidates.append(cid)
        merged_text_space = b1.text + " " + b2.text
        cid2 = try_match(merged_text_space, merged_conf)
        if cid2:
            candidates.append(cid2)

    # 相邻3框
    for i in range(n - 2):
        b1, b2, b3 = boxes[i], boxes[i + 1], boxes[i + 2]
        if not (is_y_close(b1, b2) and is_y_close(b2, b3)):
            continue
        merged_conf = (b1.conf + b2.conf + b3.conf) / 3
        merged_text = b1.text + b2.text + b3.text
        cid = try_match(merged_text, merged_conf)
        if cid:
            candidates.append(cid)

    # 去重：保留最高置信度（参考 CnOCR/V6 模式）
    best: Dict[str, float] = {}
    for cid, conf in candidates:
        if cid not in best or conf > best[cid]:
            best[cid] = conf
    return [(cid, conf) for cid, conf in best.items()]


def merge_boxes_train(boxes: List[TextBox]) -> List[Tuple[str, float]]:
    """铁路货车专用合并：先按行分组，每行内横向合并，数字类只保留最长结果。"""
    candidates: List[Tuple[str, float]] = []
    type_boxes: List[Tuple[TextBox, str]] = []
    num_boxes: List[Tuple[TextBox, str]] = []

    for b in boxes:
        t = b.text.upper().replace(" ", "").replace("-", "").replace(".", "")
        if not t:
            continue
        if _is_vehicle_type_pattern(t):
            fixed = _fix_vehicle_type(t)
            if fixed:
                type_boxes.append((b, fixed))
        elif re.search(r"\d", t):
            fixed_num = _fix_vehicle_number(t)
            if re.search(r"\d", fixed_num):
                num_boxes.append((b, fixed_num))

    # ========== 车种类：按行分组后合并 ==========
    if type_boxes:
        type_lines = _group_to_lines([b for b, _ in type_boxes])
        type_map = {id(b): fixed for b, fixed in type_boxes}

        for line in type_lines:
            m = len(line)
            for b in line:
                candidates.append((type_map[id(b)], b.conf))
            if m >= 2:
                for i, j in combinations(range(m), 2):
                    b1, b2 = line[i], line[j]
                    merged = b1.text + b2.text
                    fixed = _fix_vehicle_type(merged)
                    if not fixed:
                        merged2 = b2.text + b1.text
                        fixed = _fix_vehicle_type(merged2)
                    if fixed:
                        conf = (b1.conf + b2.conf) / 2
                        candidates.append((fixed, conf))

    # ========== 数字类：按行分组后合并，每行只保留最长结果 ==========
    if num_boxes:
        num_lines = _group_to_lines([b for b, _ in num_boxes])
        num_map = {id(b): fixed for b, fixed in num_boxes}

        for line in num_lines:
            n = len(line)
            line_candidates: List[Tuple[str, float]] = []

            # 单框
            for b in line:
                match = re.search(r'\d{3,8}', num_map[id(b)])
                if match:
                    line_candidates.append((match.group(), b.conf))

            # 同一行内连续多框合并（整行无上限）
            for length in range(2, n + 1):
                for i in range(n - length + 1):
                    merged_text = ''.join(num_map[id(line[j])] for j in range(i, i + length))
                    merged_conf = sum(line[j].conf for j in range(i, i + length)) / length
                    match = re.search(r'\d{3,8}', merged_text)
                    if match:
                        line_candidates.append((match.group(), merged_conf))

            # 只保留该行最长的结果（长度相同取置信度高的）
            if line_candidates:
                line_candidates.sort(key=lambda x: (len(x[0]), x[1]), reverse=True)
                longest_len = len(line_candidates[0][0])
                for cid, conf in line_candidates:
                    if len(cid) == longest_len:
                        candidates.append((cid, conf))
                        break

    # 去重：保留最高置信度
    best: Dict[str, float] = {}
    for cid, conf in candidates:
        if cid not in best or conf > best[cid]:
            best[cid] = conf
    return [(cid, conf) for cid, conf in best.items()]


# ============ 单图处理入口 ============
class PaddleOCRProcessor:
    """基于 PaddleOCR 的单图识别处理器（修改版）"""

    def __init__(self, use_gpu: bool = True):
        self.ocr = None
        self.gap_detector = GapDetector()

        # 优先尝试 GPU，失败则自动回退 CPU
        if use_gpu:
            try:
                self.ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False, use_gpu=True)
                print("INFO: PaddleOCR initialized on GPU (lang=ch)")
                return
            except Exception as e:
                print(f"WARNING: PaddleOCR GPU init failed: {e}, falling back to CPU")

        try:
            self.ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False, use_gpu=False)
            print("INFO: PaddleOCR initialized on CPU (lang=ch)")
        except Exception as e:
            print(f"ERROR: PaddleOCR init failed: {e}")

    def recognize(self, image: np.ndarray) -> List[OCRBox]:
        """Run OCR and return parsed boxes (aligned with CnOCR version)."""
        if self.ocr is None:
            return []

        result = self.ocr.ocr(image, cls=True)
        boxes: List[OCRBox] = []

        if result and result[0]:
            for line in result[0]:
                if not line:
                    continue
                coords = line[0]
                text, conf = line[1]
                xs = [p[0] for p in coords]
                ys = [p[1] for p in coords]
                box = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
                boxes.append(OCRBox(box=box, text=text, confidence=float(conf)))

        return boxes

    def process(self, image_path: str) -> ImageResult:
        """处理单张图片，返回带置信度的识别结果。"""
        if self.ocr is None:
            return ImageResult()

        # 使用 imdecode 避免中文路径问题
        img_array = np.fromfile(image_path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            print(f"ERROR: Cannot read image: {image_path}")
            return ImageResult()

        # ========== 空挡检测（只做标记，不跳过 OCR）==========
        gap_result = self.gap_detector.detect(img)
        is_gap = gap_result.gap_type == GapType.GAP
        if is_gap:
            print(f"  [空挡检测] 空挡帧 (vert={gap_result.vert_edge_ratio:.4f})")
        elif gap_result.gap_type == GapType.TRANSITION:
            print(f"  [空挡检测] 过渡帧 (vert={gap_result.vert_edge_ratio:.4f})")
        else:
            print(f"  [空挡检测] 正常帧 (vert={gap_result.vert_edge_ratio:.4f})")

        h, w = img.shape[:2]
        ocr_boxes = self.recognize(img)
        upper_boxes, lower_boxes = parse_ocr_boxes(ocr_boxes, h)

        # 上半区提取集装箱
        container_ids = merge_boxes(upper_boxes, extract_container_id)

        # 下半区提取铁路货车
        train_ids = merge_boxes_train(lower_boxes)

        # 分类车种和车号
        train_types: List[Tuple[str, float]] = []
        train_numbers: List[Tuple[str, float]] = []
        for tid, conf in train_ids:
            if _is_vehicle_type_pattern(tid):
                train_types.append((tid, conf))
            elif tid.isdigit():
                train_numbers.append((tid, conf))

        # ========== 无 OCR 二次修正 ==========
        # 用户确认：过渡帧可能同时拍到过道和边缘集装箱号，OCR修正反而多余。
        # 空挡检测纯靠 CV 特征，不过滤。

        return ImageResult(
            containers=container_ids,
            train_types=train_types,
            train_numbers=train_numbers,
            is_gap=is_gap,
        )

    def process_bytes(self, image_bytes: bytes) -> ImageResult:
        """Process JPEG/PNG image bytes and return structured results (for API use)."""
        if self.ocr is None:
            return ImageResult()

        img_array = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            print("ERROR: Cannot decode image bytes")
            return ImageResult()

        return self._process_img(img)

    def process_raw_bytes(
        self,
        image_bytes: bytes,
        pixel_type: int,
        width: int,
        height: int,
    ) -> ImageResult:
        """Process raw camera pixel bytes (Bayer/Mono) and return structured results.

        Uses decode_raw_image() to convert raw industrial camera data to BGR.
        """
        if self.ocr is None:
            return ImageResult()

        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dms_api"))
        from app.train_id.utils import decode_raw_image

        img = decode_raw_image(image_bytes, pixel_type, width, height)
        if img is None:
            print("ERROR: Cannot decode raw image bytes")
            return ImageResult()

        return self._process_img(img)

    def _process_img(self, img: np.ndarray) -> ImageResult:
        """Process a decoded BGR image and return structured results."""
        # ========== 空挡检测（只做标记，不跳过 OCR）==========
        gap_result = self.gap_detector.detect(img)
        is_gap = gap_result.gap_type == GapType.GAP
        if is_gap:
            print(f"  [空挡检测] 空挡帧 (vert={gap_result.vert_edge_ratio:.4f})")
        elif gap_result.gap_type == GapType.TRANSITION:
            print(f"  [空挡检测] 过渡帧 (vert={gap_result.vert_edge_ratio:.4f})")
        else:
            print(f"  [空挡检测] 正常帧 (vert={gap_result.vert_edge_ratio:.4f})")

        h, w = img.shape[:2]
        ocr_boxes = self.recognize(img)
        upper_boxes, lower_boxes = parse_ocr_boxes(ocr_boxes, h)

        # 上半区提取集装箱
        container_ids = merge_boxes(upper_boxes, extract_container_id)

        # 下半区提取铁路货车
        train_ids = merge_boxes_train(lower_boxes)

        # 分类车种和车号
        train_types: List[Tuple[str, float]] = []
        train_numbers: List[Tuple[str, float]] = []
        for tid, conf in train_ids:
            if _is_vehicle_type_pattern(tid):
                train_types.append((tid, conf))
            elif tid.isdigit():
                train_numbers.append((tid, conf))

        return ImageResult(
            containers=container_ids,
            train_types=train_types,
            train_numbers=train_numbers,
            is_gap=is_gap,
        )


# ============ CLI 入口 ============
def main():
    parser = argparse.ArgumentParser(description='铁路图片 OCR 识别 - PaddleOCR 修改版')
    parser.add_argument('input', help='输入图片或文件夹路径')
    parser.add_argument('-o', '--output', default='./output_paddle', help='输出目录')
    parser.add_argument('--cpu', action='store_true', help='强制使用 CPU')
    args = parser.parse_args()

    processor = PaddleOCRProcessor(use_gpu=not args.cpu)

    input_path = Path(args.input)
    os.makedirs(args.output, exist_ok=True)

    if input_path.is_file():
        image_paths = [input_path]
    else:
        image_paths = sorted([
            p for p in input_path.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ])

    all_results = {}
    for img_path in image_paths:
        print(f"\nProcessing: {img_path.name}")
        result = processor.process(str(img_path))

        # 向 CnOCR 版输出格式靠拢：每张图只保留最佳结果
        best_container = result.containers[0][0] if result.containers else ""
        best_type = result.train_types[0][0] if result.train_types else ""
        best_number = result.train_numbers[0][0] if result.train_numbers else ""

        confs = []
        if result.containers:
            confs.append(result.containers[0][1])
        if result.train_types:
            confs.append(result.train_types[0][1])
        if result.train_numbers:
            confs.append(result.train_numbers[0][1])
        avg_conf = round(sum(confs) / len(confs), 4) if confs else 0.0

        # 空挡帧：type 字段输出 ######## 作为标记
        result_dict = {
            "file": img_path.name,
            "type": "########" if result.is_gap else "",
            "container": best_container,
            "vehicle_type": best_type,
            "vehicle_number": best_number,
            "confidence": avg_conf,
        }
        all_results[img_path.name] = result_dict

        print(f"  集装箱: {best_container or 'N/A'}")
        print(f"  车种: {best_type or 'N/A'}")
        print(f"  车号: {best_number or 'N/A'}")

        # 每张图一个独立 JSON（和 CnOCR 版一致）
        json_path = os.path.join(args.output, f"{img_path.stem}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)

    # 汇总 JSON
    out_path = os.path.join(args.output, 'result.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n共处理 {len(image_paths)} 张图片")
    print(f"单图 JSON: {args.output}/{{name}}.json")
    print(f"汇总 JSON: {out_path}")


if __name__ == '__main__':
    main()
