#!/usr/bin/env python3
"""
Train ID OCR — PaddleOCR 单图识别版

核心特性：
  - lang='ch' 支持中文识别，过滤纯中文文本框
  - S-curve 对比度增强（可选，改善暗光/低对比度场景）
  - FlatcarGapDetector 空挡检测（8 特征评分制）
  - 工业相机原始像素解码（BayerGB8/Mono8/BGR8）
  - 车号输出过滤：仅保留 6-7 位标准编号，排除全零假阳性
  - 强制 CPU：现场 GPU 模式 PaddleOCR 输出乱码

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
from typing import List, Optional, Tuple, Dict
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from paddleocr import PaddleOCR

from flatcar_gap_detector import FlatcarGapDetector


# ============ 工业相机原始像素解码 ============
# 海康工业相机常用像素格式常量
PixelType_Gvsp_Mono8 = 0x01080001
PixelType_Gvsp_BayerGR8 = 0x01080002
PixelType_Gvsp_BayerRG8 = 0x01080009
PixelType_Gvsp_BayerGB8 = 0x0108000A
PixelType_Gvsp_BayerBG8 = 0x0108000B
PixelType_Gvsp_RGB8 = 0x02180014
PixelType_Gvsp_BGR8 = 0x02180015

_BAYER_CV_MAP = {
    PixelType_Gvsp_BayerGR8: cv2.COLOR_BayerGR2BGR,
    PixelType_Gvsp_BayerRG8: cv2.COLOR_BayerRG2BGR,
    PixelType_Gvsp_BayerGB8: cv2.COLOR_BayerGB2BGR,
    PixelType_Gvsp_BayerBG8: cv2.COLOR_BayerBG2BGR,
}

# 根据实际测试数据校准：海康 BayerGB8 在不同分辨率下需不同的 OpenCV Bayer 模式
RESOLUTION_BAYER_MAP = {
    (2448, 2048): cv2.COLOR_BayerRG2BGR,
    (4096, 3000): cv2.COLOR_BayerGR2BGR,
}


def decode_raw_image(image_bytes: bytes, pixel_type: int, width: int, height: int,
                     bayer_cv_code: Optional[int] = None) -> Optional[np.ndarray]:
    """Decode raw industrial camera pixel data to BGR numpy array.

    Supports Hikrobot/GigE Vision raw formats (Mono8, Bayer8, RGB8, BGR8).
    """
    if pixel_type == PixelType_Gvsp_Mono8:
        expected = width * height
        if len(image_bytes) != expected:
            print(f"WARNING: Mono8 size mismatch: expected {expected}, got {len(image_bytes)}")
        gray = np.frombuffer(image_bytes, dtype=np.uint8).reshape(height, width)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if pixel_type in _BAYER_CV_MAP:
        expected = width * height
        if len(image_bytes) != expected:
            print(f"WARNING: Bayer8 size mismatch: expected {expected}, got {len(image_bytes)}")
        bayer = np.frombuffer(image_bytes, dtype=np.uint8).reshape(height, width)

        if bayer_cv_code is not None:
            cv_code = bayer_cv_code
        elif pixel_type == PixelType_Gvsp_BayerGB8 and (width, height) in RESOLUTION_BAYER_MAP:
            cv_code = RESOLUTION_BAYER_MAP[(width, height)]
        else:
            cv_code = _BAYER_CV_MAP[pixel_type]

        return cv2.cvtColor(bayer, cv_code)

    if pixel_type == PixelType_Gvsp_BGR8:
        expected = width * height * 3
        if len(image_bytes) != expected:
            print(f"WARNING: BGR8 size mismatch: expected {expected}, got {len(image_bytes)}")
        return np.frombuffer(image_bytes, dtype=np.uint8).reshape(height, width, 3)

    if pixel_type == PixelType_Gvsp_RGB8:
        expected = width * height * 3
        if len(image_bytes) != expected:
            print(f"WARNING: RGB8 size mismatch: expected {expected}, got {len(image_bytes)}")
        rgb = np.frombuffer(image_bytes, dtype=np.uint8).reshape(height, width, 3)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    print(f"ERROR: Unsupported pixel type: 0x{pixel_type:08X}")
    return None


# ============ 图像预处理 ============
def enhance_contrast_s_curve(image: np.ndarray, steepness: float = 2.5) -> np.ndarray:
    """S-curve 对比度增强：暗部压低，亮部提亮。

    在 LAB 颜色空间对 L 通道做 tanh S-curve 映射，增强文字与背景的对比度。
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0].astype(np.float32)
    l_norm = (l - 128) / 128.0
    l_enh = np.tanh(l_norm * steepness)
    l_enh = ((l_enh + 1) / 2 * 255).astype(np.uint8)
    lab[:, :, 0] = l_enh
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def preprocess_otsu_fusion(image: np.ndarray, close_kernel: Tuple[int, int] = (2, 2),
                            alpha: float = 0.8, beta: float = 0.2) -> np.ndarray:
    """OTSU二值化 + 形态学闭运算 + 与原图融合（参考 kimi 预处理）。

    用于填充空心字、连接断线，同时保留原图灰度信息供 PaddleOCR 使用。
    Args:
        image: BGR 图像
        close_kernel: 闭运算核大小，默认 (2,2) 小核连接细断线
        alpha: 原图权重
        beta: 二值图权重
    Returns:
        融合后的 BGR 图像
    """
    if image is None or image.size == 0:
        return image
    img = image.copy()
    # 1. 快速降噪
    img = cv2.GaussianBlur(img, (3, 3), 0.5)
    # 2. CLAHE 亮度均衡
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    # 3. 对比度增强（gamma=0.8）
    gamma = 0.8
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
    img = cv2.LUT(img, table)
    img = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
    img = np.clip(img, 0, 255).astype(np.uint8)
    # 4. 边缘锐化（Unsharp Masking）
    gaussian = cv2.GaussianBlur(img, (0, 0), 3)
    img = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
    img = np.clip(img, 0, 255).astype(np.uint8)
    # 5. OTSU + 闭运算 + 融合（核心）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, close_kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    img = cv2.addWeighted(img, alpha, binary_color, beta, 0)
    return img


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


# ============ 中国铁路货车标准车型白名单 ============
# 基于《铁路货车车型编码规范》整理，用于过滤 OCR 误识（如 T0E/CH1N 等）
_CHINA_RAIL_VEHICLE_TYPES = {
    # --- 敞车 C 系列 ---
    'C50', 'C60', 'C61', 'C61Y',
    'C62', 'C62A', 'C62AK', 'C62AT', 'C62B', 'C62BK', 'C62BT', 'C62K', 'C62M', 'C62N', 'C62T',
    'C63', 'C63A',
    'C64', 'C64A', 'C64H', 'C64K', 'C64T',
    'C65',
    'C70', 'C70A', 'C70B', 'C70E', 'C70EH', 'C70H', 'C70BH',
    'C76', 'C76A', 'C76B', 'C76C', 'C76H',
    'C80', 'C80A', 'C80B', 'C80C', 'C80H',
    'C100', 'CF',
    # --- 棚车 P 系列 ---
    'P50', 'P60', 'P61',
    'P62', 'P62K', 'P62N', 'P62NK', 'P62NT', 'P62T',
    'P63',
    'P64', 'P64A', 'P64AK', 'P64AT', 'P64GK', 'P64GH', 'P64GT', 'P64K', 'P64T',
    'P65', 'P65A', 'P65S',
    'P66', 'P66H', 'P66K',
    'P70', 'P70H',
    'PB',
    # --- 平车 N 系列 ---
    'N17', 'N70',
    # --- 罐车 G 系列 ---
    'G60', 'G70', 'G70K', 'G70H',
    # --- 矿石车 K 系列 ---
    'K13', 'K18', 'K18F',
    # --- 集装箱车 X 系列 ---
    'X6B', 'X6BK', 'X6C', 'X6CK', 'X70',
    # --- 长大货物车 D 系列 ---
    'D22B', 'D26', 'D32', 'D38',
    # --- 冷藏车 B 系列 ---
    'B6', 'B22', 'B23',
    # --- 特种车 T 系列 ---
    'T11BK',
    # --- 汽车运输车 ---
    'SQ6', 'J5SQ', 'J6SQ',
}


# ============ CnOCR 版字符混淆表（车种纠错用）============
_LETTER_TO_DIGIT = str.maketrans({
    "O": "0", "o": "0", "Q": "0", "D": "0",
    "I": "1", "l": "1", "i": "1",
    "S": "5", "s": "5",
    "A": "4", "a": "4",
    "G": "6", "g": "6",
    "T": "7", "t": "7",
    "B": "8", "b": "8",
    "Z": "2", "z": "2",
    # 注意：E/e 不在全局映射中，避免 C70E→C706
    # CeAK 等特殊情况在 _fix_vehicle_type 中单独处理
})

_DIGIT_TO_LETTER = str.maketrans({
    "1": "I",
    "4": "A",
    "5": "S",
    "6": "G",
    "7": "T",
    "8": "B",
    # 注意：0 不映射为 C，避免 06197→C6197 等误修复
    # 070→C70 在 _fix_vehicle_type 中通过 re.match 特判处理
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
    新增：处理 070→C70、CeAK→C64K 等常见误识。
    """
    t = text.strip()
    # 清理更多干扰字符（冒号、分号等）
    t = re.sub(r"^[/(（\[{:;]+", "", t)
    t = re.sub(r"[/）)\]}.:;]+", "", t)
    t = t.upper()

    if not t:
        return ""

    # Remove trailing Q artefact
    if len(t) > 2 and t.endswith("Q") and t[-2].isdigit():
        t = t[:-1]

    # 特殊处理：0 开头的3位数字 → C 开头（如 070→C70, 064→C64）
    if re.match(r'^[0O][4567]\d[A-Z]?$', t):
        return 'C' + t[1:]
    
    # 特殊处理：670/664 等常见误识 → C70/C64
    # OCR 常把 C 识别为 6，把 C70 识别为 670
    if t in ('670', '664', '665', '662'):
        return 'C' + t[1:]

    # 特殊处理：T→7, O→0 误识（如 CTOE → C70E）
    # 空心字 7 和 T 形状极似，0 和 O 极似，且 T/O 不在数字段
    if t[0] in 'CPNG':
        candidate = t.translate(str.maketrans({'T': '7', 'O': '0'}))
        if candidate in _CHINA_RAIL_VEHICLE_TYPES:
            return candidate

    # 特殊处理：全字母车型误识（如 CeAK → C64K, C6AK → C64K）
    # 当首字母是 C/P/N/G 且剩余部分包含易混淆字母时
    if len(t) >= 4 and t[0] in 'CPNG' and not any(c.isdigit() for c in t):
        mapped = t.translate(str.maketrans({
            'E': '6', 'A': '4', 'O': '0', 'S': '5',
            'I': '1', 'B': '8', 'Z': '2', 'G': '6',
        }))
        is_valid, corrected = _is_vehicle_type_pattern(mapped)
        if is_valid:
            return corrected

    digit_like = set("0123456789OoQDIilSsAaGgTBbZzTt")
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

    _digit_confusable = set("OoQDIilSsAaGgTBbZzTt")
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


def _is_vehicle_type_pattern(text: str) -> Tuple[bool, str]:
    """判断是否像车种，并返回（是否通过, 修正后文本）。

    在原有格式校验基础上，新增：
      1. 白名单校验：仅允许中国铁路货车标准车型
      2. 近似匹配：如 C701E → C70E（删除多余字符）
      3. 尾部 Q 清理
    """
    t = text.strip().upper()
    t = re.sub(r"^[/(（]+", "", t)
    # 车型通常为 3~7 个字符（如 C70, C70E, C64K, C70EH）
    if len(t) > 7:
        return False, t
    if not re.match(r"^[A-Z]+\d+[A-Z]*Q?$", t):
        return False, t
    fully_numeric = t.translate(_LETTER_TO_DIGIT)
    fully_numeric = re.sub(r"[^0-9]", "", fully_numeric)
    if len(fully_numeric) >= len(t):
        return False, t

    # 白名单校验（去掉尾部可能残留的 Q）
    clean_t = t.rstrip('Q')
    if clean_t in _CHINA_RAIL_VEHICLE_TYPES:
        return True, clean_t

    # 近似匹配：尝试删除一个字符，看是否能命中白名单
    # 这修复 C701E→C70E、C64HK→C64K 等多字符误识
    for i in range(len(clean_t)):
        candidate = clean_t[:i] + clean_t[i + 1:]
        if candidate in _CHINA_RAIL_VEHICLE_TYPES and len(candidate) >= 3:
            return True, candidate

    return False, t


def _fix_vehicle_number(text: str) -> str:
    """CnOCR 版：数字清理。
    
    新增：彻底去除空格，避免分块车号中的空格导致匹配失败。
    """
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
    # 彻底去掉所有空格，使 "49 44030" → "4944030"
    t = t.replace(" ", "")
    return t


def is_train_param(text: str) -> bool:
    """判断是否为货车参数框（载重/自重/容积等），应排除。
    
    参考 kimi 代码扩展参数关键词，覆盖更多现场参数格式。
    新增：保护分块车号（如 49/44030）不被误过滤。
    新增：过滤车厢侧面"中国铁路/CHINA RAILWAY"大字误识。
    """
    t = text.lower()
    # 核心参数关键词（参考 kimi + 现场实测扩展）
    # 注意：'t' 已移除，避免误杀含 t 的车型（如 CTOE→C70E）
    # 吨数标记改用正则 \d+t 精确匹配
    param_keywords = [
        'm³', 'm3', '载', '重', '自', '容', '积', '换', '长', '定',
        '自重', '容积', '换长', '定检', '载重', 'mc', '均载', '集中',
        '吨', '米', '立方米', '制造', '日期', '厂', '段', '限', '超',
    ]
    if any(kw in t for kw in param_keywords):
        return True
    # 吨数标记：仅匹配数字+t 格式（如 70t、73.3t），不单独匹配 t
    if re.search(r'\d+t', t):
        return True
    
    # 过滤车厢侧面"中国铁路/CHINA RAILWAY"大字误识
    # 这些大字被OCR识别后会被误判为车型（如 CH1NANRAILNAY, R41LWAY）
    railway_keywords = ['china', 'railway', 'rail', '铁路', '中铁']
    if any(kw in t for kw in railway_keywords):
        return True
    
    # 明显的参数格式：小数、乘法格式（如 12.5×2.6, 73.3t）
    if re.search(r'\d+\.\d+', text):
        return True
    if re.search(r'\d+\s*[X×x]\s*\d+', text):
        return True
    
    # 车号保护规则：含≥5位连续数字的字符串，即使有 / 也认为是车号
    # 这处理了分块车号被 OCR 连在一起的情况（如 49/44030, 15/9104）
    digits_only = re.sub(r'[^\d]', '', text)
    if len(digits_only) >= 5:
        return False
    
    # 含数学运算符/单位的格式（如 20.7t, 12.5X2.0）
    # 注意：这个检查放在车号保护之后，所以 49/44030 已经被保护了
    if any(c in text for c in ['.', '(', ')', 'X', '×', '*', '\\']):
        return True
    if '/' in text:
        # 单独的 / 且数字不足5位，认为是参数（如 "61t" 不含 /，但 "12/3" 是参数格式）
        return True
    
    # 纯中文（如"中铁集""铁路"等）
    if _is_pure_chinese(text):
        return True
    # 特殊符号
    if any(c in text for c in ['#', '+', '-', '=', '%', '&', '$']):
        return True
    # 过短纯字母且非标准车种（如 "Mc" "HC" "DK"）
    if text.isalpha() and len(text) <= 3 and text.upper() not in [
        'C70', 'C64', 'C62', 'P64', 'P70', 'P62', 'N17', 'N70', 'G70', 'G60',
        'C70E', 'C64K', 'C64H', 'C62K', 'P64K', 'P62K'
    ]:
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
            # 增大 tolerance：从 0.6 倍高度 → 0.8 倍高度，减少跨行分块被分到不同行
            tolerance = max(40, int(heights[len(heights) // 2] * 0.8))
        else:
            tolerance = 60

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


def _box_iou(b1: TextBox, b2: TextBox) -> float:
    """计算两个框的 IoU（交并比）。"""
    # 转换为 [x1, y1, x2, y2] 格式
    x1_1 = b1.center_x - b1.width / 2
    y1_1 = b1.center_y - b1.height / 2
    x2_1 = b1.center_x + b1.width / 2
    y2_1 = b1.center_y + b1.height / 2
    x1_2 = b2.center_x - b2.width / 2
    y1_2 = b2.center_y - b2.height / 2
    x2_2 = b2.center_x + b2.width / 2
    y2_2 = b2.center_y + b2.height / 2
    
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h
    
    area1 = b1.width * b1.height
    area2 = b2.width * b2.height
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


def _dedup_overlapping_boxes(boxes: List[Tuple[TextBox, str]], iou_thresh: float = 0.3) -> List[Tuple[TextBox, str]]:
    """去重高度重叠的框，保留置信度高的。"""
    if not boxes:
        return []
    # 按置信度降序排序
    sorted_boxes = sorted(boxes, key=lambda x: x[0].conf, reverse=True)
    kept = []
    for b, fixed in sorted_boxes:
        overlap = False
        for kept_b, _ in kept:
            if _box_iou(b, kept_b) > iou_thresh:
                overlap = True
                break
        if not overlap:
            kept.append((b, fixed))
    return kept


def merge_boxes_train(boxes: List[TextBox]) -> List[Tuple[str, float]]:
    """铁路货车专用合并：先按行分组，每行内横向合并，数字类只保留最长结果。
    
    新增：
      1. 重叠框去重（IOU>0.3 时保留置信度高的）
      2. 跨行数字合并（上下两行都是数字，且 x 方向互补时合并）
    """
    candidates: List[Tuple[str, float]] = []
    type_boxes: List[Tuple[TextBox, str]] = []
    num_boxes: List[Tuple[TextBox, str]] = []

    for b in boxes:
        t = b.text.upper().replace(" ", "").replace("-", "").replace(".", "")
        if not t:
            continue
        # 先用 _fix_vehicle_type 预处理，再判断车型模式
        # 这处理了 070→C70 等需要先修正再判断的情况
        fixed_type = _fix_vehicle_type(t)
        is_valid_t, corrected_t = _is_vehicle_type_pattern(t)
        is_valid_fixed, corrected_fixed = _is_vehicle_type_pattern(fixed_type) if fixed_type else (False, fixed_type)

        if is_valid_t or is_valid_fixed:
            final_type = corrected_fixed if is_valid_fixed else corrected_t
            type_boxes.append((b, final_type))
        elif re.search(r"\d", t):
            fixed_num = _fix_vehicle_number(t)
            if re.search(r"\d", fixed_num):
                num_boxes.append((b, fixed_num))

    # 去重高度重叠的框
    type_boxes = _dedup_overlapping_boxes(type_boxes)
    num_boxes = _dedup_overlapping_boxes(num_boxes)

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
                        is_valid, corrected = _is_vehicle_type_pattern(fixed)
                        if is_valid:
                            conf = (b1.conf + b2.conf) / 2
                            candidates.append((corrected, conf))

    # ========== 数字类：按行分组后合并 ==========
    if num_boxes:
        num_lines = _group_to_lines([b for b, _ in num_boxes])
        num_map = {id(b): fixed for b, fixed in num_boxes}

        # --- 收集所有行候选 ---
        all_line_results: List[List[Tuple[str, float]]] = []
        
        for line in num_lines:
            n = len(line)
            line_candidates: List[Tuple[str, float]] = []

            # 单框
            for b in line:
                match = re.search(r'\d{2,8}', num_map[id(b)])
                if match:
                    line_candidates.append((match.group(), b.conf))

            # 同一行内连续多框合并
            for length in range(2, n + 1):
                for i in range(n - length + 1):
                    too_far = False
                    for j in range(i, i + length - 1):
                        b1, b2 = line[j], line[j + 1]
                        x_gap = b2.center_x - b1.center_x - (b1.width + b2.width) / 2
                        max_w = max(b1.width, b2.width)
                        if x_gap > 800:
                            too_far = True
                            break
                    if too_far:
                        continue
                    merged_text = ''.join(num_map[id(line[j])] for j in range(i, i + length))
                    merged_conf = sum(line[j].conf for j in range(i, i + length)) / length
                    match = re.search(r'\d{2,8}', merged_text)
                    if match:
                        line_candidates.append((match.group(), merged_conf))

            all_line_results.append(line_candidates)

        # --- 跨行合并：相邻行的数字框如果 x 方向互补，尝试合并 ---
        for i in range(len(num_lines) - 1):
            line1 = num_lines[i]
            line2 = num_lines[i + 1]
            # 取每行最右和最左的框
            if not line1 or not line2:
                continue
            rightmost_l1 = max(line1, key=lambda b: b.center_x)
            leftmost_l2 = min(line2, key=lambda b: b.center_x)
            # 检查 line2 是否整体在 line1 的右侧（互补而非重叠）
            l2_all_right = all(b.center_x > rightmost_l1.center_x - rightmost_l1.width for b in line2)
            if l2_all_right:
                # 合并 line1 整行 + line2 整行
                merged_line = sorted(line1 + line2, key=lambda b: b.center_x)
                merged_text = ''.join(num_map[id(b)] for b in merged_line)
                merged_conf = sum(b.conf for b in merged_line) / len(merged_line)
                match = re.search(r'\d{2,8}', merged_text)
                if match:
                    all_line_results[i].append((match.group(), merged_conf))

        # --- 每行只保留最长结果 ---
        for line_candidates in all_line_results:
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
    """基于 PaddleOCR 的单图识别处理器（修改版）
    
    支持多种预处理模式：
      - none: 原图直接识别（最快）
      - s_curve: S-curve 对比度增强（改善暗光场景）
      - otsu_fusion: OTSU+闭运算+原图融合（参考 kimi，填充空心字）
    """

    def __init__(self, use_gpu: bool = False, enhancement_mode: str = 'none'):
        self.ocr = None
        self.gap_detector = FlatcarGapDetector(min_gap_score=7, std_thresh=30.0, asymmetry_thresh=0.25)
        self.enhancement_mode = enhancement_mode

        # NOTE: 现场 GPU 模式 PaddleOCR 输出乱码，默认强制 CPU
        if use_gpu:
            try:
                self.ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False, use_gpu=True)
                print("INFO: PaddleOCR initialized on GPU (lang=ch)")
                return
            except Exception as e:
                print(f"WARNING: PaddleOCR GPU init failed: {e}, falling back to CPU")

        try:
            self.ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False, use_gpu=False)
            print(f"INFO: PaddleOCR initialized on CPU (lang=ch), enhancement={enhancement_mode}")
        except Exception as e:
            print(f"ERROR: PaddleOCR init failed: {e}")

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """根据配置选择预处理模式。"""
        if self.enhancement_mode == 'otsu_fusion':
            return preprocess_otsu_fusion(image)
        else:
            return image

    def recognize(self, image: np.ndarray) -> List[OCRBox]:
        """Run OCR and return parsed boxes (aligned with CnOCR version)."""
        if self.ocr is None:
            return []

        ocr_input = self._preprocess(image)
        result = self.ocr.ocr(ocr_input, cls=True)
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

        return self._process_img(img)

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

        img = decode_raw_image(image_bytes, pixel_type, width, height)
        if img is None:
            print("ERROR: Cannot decode raw image bytes")
            return ImageResult()

        return self._process_img(img)

    def _process_img(self, img: np.ndarray) -> ImageResult:
        """Process a decoded BGR image and return structured results."""
        is_gap, gap_conf, _ = self.gap_detector.detect(img)

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
            is_valid, corrected = _is_vehicle_type_pattern(tid)
            if is_valid:
                train_types.append((corrected, conf))
            elif tid.isdigit():
                train_numbers.append((tid, conf))

        # 过渡帧检测：如果检测到多个车型框（≥2），说明画面中有两节车厢
        # 此时只保留置信度最高的车型，并过滤掉明显是参数的数字结果
        if len(train_types) >= 2:
            train_types.sort(key=lambda x: x[1], reverse=True)
            best_type = train_types[0]
            # 过滤：只保留与最佳车型同侧（x坐标接近）的数字框
            # 简单策略：保留最长的车号（最有可能是真实车号）
            train_types = [best_type]
            if train_numbers:
                train_numbers.sort(key=lambda x: len(x[0]), reverse=True)
                # 过渡帧中，参数误识通常比真实车号短或长不一致
                # 保留 6-7 位中最长的
                train_numbers = [n for n in train_numbers if 6 <= len(n[0]) <= 7]
                if train_numbers:
                    train_numbers = [train_numbers[0]]

        # 输出后过滤：中国铁路货车编号为 6-7 位数字，排除全零假阳性
        train_numbers = [(n, c) for n, c in train_numbers
                         if 6 <= len(n) <= 7 and not re.fullmatch(r'0+', n)]

        return ImageResult(
            containers=container_ids,
            train_types=train_types,
            train_numbers=train_numbers,
            is_gap=is_gap,
        )


# ============ 全局单例（避免每次请求重复初始化 PaddleOCR）============
_global_processors: Dict[str, PaddleOCRProcessor] = {}


def get_ocr_processor(use_gpu: bool = False, enhancement_mode: str = 'none') -> PaddleOCRProcessor:
    """获取全局单例的 PaddleOCRProcessor。
    
    第一次调用会初始化模型（约 0.8s），后续调用直接返回已初始化的实例。
    支持多配置共存（如 CPU/GPU、none/otsu_fusion 等），用配置字符串作为 key。
    """
    key = f"gpu={use_gpu}_mode={enhancement_mode}"
    if key not in _global_processors:
        _global_processors[key] = PaddleOCRProcessor(
            use_gpu=use_gpu, enhancement_mode=enhancement_mode
        )
    return _global_processors[key]


# ============ CLI 入口 ============
def main():
    parser = argparse.ArgumentParser(description='铁路图片 OCR 识别 - PaddleOCR 修改版')
    parser.add_argument('input', help='输入图片或文件夹路径')
    parser.add_argument('-o', '--output', default='./output_paddle', help='输出目录')
    parser.add_argument('--gpu', action='store_true', help='尝试使用 GPU（现场不建议，可能输出乱码）')
    parser.add_argument('--enhancement', default='none',
                        choices=['none', 'otsu_fusion'],
                        help='图像预处理模式: none=原图, otsu_fusion=OTSU融合（参考kimi）')
    args = parser.parse_args()

    processor = get_ocr_processor(use_gpu=args.gpu, enhancement_mode=args.enhancement)

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
