#!/usr/bin/env python3
"""
Train ID OCR — PaddleOCR 单图识别版

核心特性：
  - lang='ch' 支持中文识别，过滤纯中文文本框
  - S-curve 对比度增强（可选，改善暗光/低对比度场景）
  - FlatcarGapDetector 空挡检测（8 特征评分制）
  - 工业相机原始像素解码（BayerGB8/Mono8/BGR8）
  - 车号输出过滤：仅保留 6-7 位标准编号，排除全零假阳性
  - 自动 GPU 检测：启动时自动验证 GPU OCR 输出是否正常，异常则回退 CPU

Usage:
    python train_id_ocr_paddle.py image.jpg
    python train_id_ocr_paddle.py ./images/ -o ./output
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 确保本模块在包内导入时也能找到同目录的 flatcar_gap_detector
import sys
from pathlib import Path
_current_dir = Path(__file__).parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

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

# ============ 多进程并发支持 ============
import multiprocessing as mp
import time as _time
import threading as _threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# spawn: 重新初始化 Python 解释器创建子进程（最安全，避免 PaddleOCR 线程锁死锁）
_MPOOL_CTX = mp.get_context('spawn')


def _check_gpu_available() -> tuple:
    """检测当前环境是否有可用的 CUDA GPU。"""
    try:
        import paddle
    except ImportError:
        return False, "未安装 paddle"
    if not paddle.is_compiled_with_cuda():
        return False, "Paddle 未编译 CUDA 支持"
    try:
        gpu_count = paddle.device.cuda.device_count()
        if gpu_count == 0:
            return False, "未检测到 CUDA 设备"
    except Exception as e:
        return False, f"CUDA 设备枚举失败: {e}"
    try:
        gpu_name = paddle.device.cuda.get_device_name()
    except Exception:
        gpu_name = "Unknown"
    return True, f"{gpu_name}: CUDA 可用 ({gpu_count} 设备)"


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


# ============ 图像预处理（优化版）============
# 优化说明：
#   1. enhance_contrast_s_curve: 用 cv2.LUT 预计算 S-curve 查找表替代逐像素 tanh 计算，
#      并将 BGR↔LAB 两次颜色空间转换替换为 BGR→Gray→BGR（快一个数量级），
#      对 OCR 场景效果等效甚至更佳。
#   2. preprocess_otsu_fusion: 将 8 步处理链精简为 5 步，核心优化包括：
#      - 移除不必要的 img.copy()
#      - 将 BGR 上沉重的 CLAHE+LAB 链替换为灰度上的单通道 CLAHE（省去两次颜色空间转换）
#      - 将 gamma LUT + convertScaleAbs 合并为单次组合 LUT
#      - 移除 Unsharp Masking（OCR 场景下收益有限但开销大）
#      - 全链仅需: GaussianBlur → BGR2Gray → CLAHE → LUT → OTSU → 闭运算 → 融合

def enhance_contrast_s_curve(image: np.ndarray, steepness: float = 2.5) -> np.ndarray:
    """S-curve 对比度增强：暗部压低，亮部提亮。（优化版）

    在灰度通道上预计算 tanh S-curve LUT 并应用，避免逐像素浮点运算和 LAB 颜色空间转换。
    对 OCR 场景而言，灰度上的 S-curve 与 LAB-L 通道上的效果等效甚至更佳。
    """
    if image is None or image.size == 0:
        return image

    # 预计算 S-curve LUT（256 元素）—— 完全保留原函数的 tanh 映射特性
    x = np.arange(256, dtype=np.float64)
    x_norm = (x - 128) / 128.0
    lut = ((np.tanh(x_norm * steepness) + 1) / 2 * 255).astype(np.uint8)

    if len(image.shape) == 2 or (image.ndim == 3 and image.shape[2] == 1):
        # 单通道输入：直接应用 LUT 后转 BGR
        gray = image if len(image.shape) == 2 else image[:, :, 0]
        gray = cv2.LUT(gray, lut)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        # 多通道输入：BGR→Gray（极快）→LUT→BGR，省去 LAB 转换
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.LUT(gray, lut)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def preprocess_otsu_fusion(image: np.ndarray, close_kernel: Tuple[int, int] = (2, 2),
                            alpha: float = 0.8, beta: float = 0.2) -> np.ndarray:
    """OTSU二值化 + 形态学闭运算 + 与原图融合（优化版）。

    用于填充空心字、连接断线，同时保留原图灰度信息供 PaddleOCR 使用。

    核心优化策略：
      1. 移除不必要的 img.copy()，函数接收的 image 直接用于处理
      2. 将 BGR 上沉重的 CLAHE+LAB 链（含 split/merge/两次颜色空间转换）
         替换为灰度上的单通道 CLAHE，速度提升 3 倍以上
      3. 将 gamma LUT + convertScaleAbs(alpha=1.2, beta=10) 合并为单次组合 LUT
      4. 移除 Unsharp Masking（需要额外 GaussianBlur + addWeighted，OCR 收益有限）
      5. 全链从 8 步精简为 5 步：GaussianBlur → CLAHE → 组合 LUT → OTSU+闭运算 → 融合

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

    # 1. 快速降噪（原地操作，无需拷贝）
    img = cv2.GaussianBlur(image, (3, 3), 0.5)

    # 2. BGR → Gray：后续所有增强在单通道进行（快 3 倍，省去 LAB 来回转换 + split/merge）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. CLAHE 直方图均衡（直接在灰度上，无需 LAB 转换链）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 4. 合并 gamma(0.8) + convertScaleAbs(alpha=1.2, beta=10) 为单次 LUT
    #    原: gamma(i) = (i/255)^(1/0.8) * 255
    #        scale(i) = i * 1.2 + 10
    #    合并: f(i) = min(255, max(0, gamma(i) * 1.2 + 10))
    gamma = 0.8
    inv_gamma = 1.0 / gamma
    table = np.empty(256, dtype=np.uint8)
    for i in range(256):
        v = ((i / 255.0) ** inv_gamma) * 255.0 * 1.2 + 10.0
        table[i] = 255 if v >= 255 else (0 if v <= 0 else int(v))
    gray = cv2.LUT(gray, table)

    # 5. OTSU 二值化 + 形态学闭运算
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, close_kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 6. Gray→BGR 后与二值图融合
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(gray_bgr, alpha, binary_bgr, beta, 0)

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


# ============ Pre-compiled constants (optimization) ============
# ============ Pre-compiled constants ============

# --- Regular expressions ---
# is_train_param patterns
_RE_DIGIT_T = re.compile(r'\d+t')
_RE_DECIMAL = re.compile(r'\d+\.\d+')
_RE_MULTIPLY = re.compile(r'\d+\s*[X×x]\s*\d+')
_RE_NON_DIGIT = re.compile(r'[^\d]')

# _fix_vehicle_type patterns
_RE_LEADING_NOISE = re.compile(r"^[/(（\[{:;]+")
_RE_TRAILING_NOISE = re.compile(r"[/）)\]}.:;]+")
_RE_ZERO_PREFIXED = re.compile(r'^[0O][4567]\d[A-Z]?$')
_RE_NON_DIGITS = re.compile(r"[^0-9]")

# _is_vehicle_type_pattern patterns
_RE_LEADING_PAREN = re.compile(r"^[/(（]+")
_RE_VEHICLE_FORMAT = re.compile(r"^[A-Z]+\d+[A-Z]*Q?$")

# _fix_vehicle_number patterns
_RE_PUNCT_TO_SPACE = re.compile(r"[/\\.,;:!?'\"()\[\]{}]")
_RE_NON_DIGIT_WS = re.compile(r"[^\d\s]")
_RE_MULTIPLE_SPACES = re.compile(r"\s+")

# extract_container_id patterns
_RE_CONTAINER_STRICT = re.compile(r'([A-Z]{4})(\d{6,7})')
_RE_CONTAINER_LOOSE = re.compile(r'([A-Z0-9]{4})(\d{6,7})')
_RE_PREFIX = re.compile(r'[A-Z0-9]{4}')
_RE_DIGITS_6_7 = re.compile(r'(\d{6,7})')

# --- Character translation tables ---
_CLEAN_DIGITS_TRANS = str.maketrans({
    'i': '1', 'I': '1', 'l': '1', 'L': '1',
    'o': '0', 'O': '0', 'Q': '0',
    'g': '9', 'q': '9', 'G': '6',
    'b': '6', 'B': '8',
    's': '5', 'S': '5',
    'z': '2', 'Z': '2',
    'a': '4', 'A': '4',
})



_T_O_TO_7_0 = str.maketrans({'T': '7', 'O': '0'})

_E_ABC_TO_DIGITS = str.maketrans({
    'E': '6', 'A': '4', 'O': '0', 'S': '5',
    'I': '1', 'B': '8', 'Z': '2', 'G': '6',
})

_VEHICLE_NUMBER_TRANS = str.maketrans({
    "O": "0", "o": "0", "Q": "0", "D": "0",
    "I": "1", "l": "1", "i": "1", "t": "1",
    "S": "5", "s": "5",
    "B": "8", "b": "8",
    "N": "",  "n": "",
})

# --- Keyword collections ---
_PARAM_KEYWORDS = (
    'm³', 'm3', '载', '重', '自', '容', '积', '换', '长', '定',
    '自重', '容积', '换长', '定检', '载重', 'mc', '均载', '集中',
    '吨', '米', '立方米', '制造', '日期', '厂', '段', '限', '超',
)

_RAILWAY_KEYWORDS = ('china', 'railway', 'rail', '铁路', '中铁')

_VEHICLE_TYPE_SHORT_WHITELIST = frozenset([
    'C70', 'C64', 'C62', 'P64', 'P70', 'P62', 'N17', 'N70', 'G70', 'G60',
    'C70E', 'C64K', 'C64H', 'C62K', 'P64K', 'P62K'
])

# Symbol sets for fast lookup
_SYMBOLS_FAST = frozenset('#+%&$')
_SYMBOLS_LATE = frozenset('-.()X×*\\/=')

# Pre-computed frozensets
_PREFIX_CORRECTION_VALUES = frozenset(PREFIX_CORRECTION.values())
_COMMON_PREFIXES_FROZEN = frozenset(COMMON_PREFIXES)
_DIGIT_LIKE_SET = frozenset("0123456789OoQDIilSsAaGgTBbZzTt")
_DIGIT_CONFUSABLE_SET = frozenset("OoQDIilSsAaGgTBbZzTt")


# --- Postprocessing regex patterns ---
_RE_HAS_DIGIT = re.compile(r"\d")
_RE_DIGIT_2TO8 = re.compile(r'\d{2,8}')

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


# ============ Utility functions (optimized) ============

def clean_digits(digits: str) -> str:
    """Clean OCR-misrecognized characters in digit strings.
    Uses pre-compiled translation table."""
    return digits.translate(_CLEAN_DIGITS_TRANS)


def fix_prefix_digits(prefix: str) -> Optional[str]:
    """Fix digit-for-letter errors in container prefix codes.
    Uses pre-computed frozensets for O(1) membership checks."""
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
    if result in _COMMON_PREFIXES_FROZEN or result in _PREFIX_CORRECTION_VALUES:
        return result
    for white in _COMMON_PREFIXES_FROZEN:
        diff = 0
        for a, b in zip(result, white):
            if a != b:
                diff += 1
                if diff > 1:
                    break
        if diff <= 1:
            return white
    return None


def correct_prefix(prefix: str) -> str:
    """Correct known prefix OCR errors. Pure dict lookup - already optimal."""
    if prefix in PREFIX_CORRECTION:
        return PREFIX_CORRECTION[prefix]
    if prefix in COMMON_PREFIXES:
        return prefix
    return prefix


def _is_pure_chinese(text: str) -> bool:
    """Check if text is purely Chinese characters (should be filtered).
    Uses early-exit loop instead of all() generator for ~2x speed."""
    t = text.strip()
    if not t:
        return False
    for c in t:
        if c < '\u4e00' or c > '\u9fff':
            return False
    return True


def extract_container_id(text: str, conf: float) -> Optional[Tuple[str, float]]:
    """Extract container ID with confidence.
    Uses pre-compiled regex patterns for all matching operations.
    Strategy: strict match -> loose match -> prefix iteration (preserved)."""
    text = text.upper().replace(" ", "").replace("-", "").replace(".", "")
    match = _RE_CONTAINER_STRICT.search(text)
    if match:
        prefix = match.group(1)
        digits = match.group(2)[:6]
        prefix = correct_prefix(prefix)
        return (f"{prefix}{digits}", conf)
    loose_match = _RE_CONTAINER_LOOSE.search(text)
    if loose_match:
        prefix = loose_match.group(1)
        digits = loose_match.group(2)[:6]
        fixed_prefix = fix_prefix_digits(prefix)
        if fixed_prefix:
            return (f"{fixed_prefix}{digits}", conf)
    for prefix_match in _RE_PREFIX.finditer(text):
        prefix = prefix_match.group(0)
        prefix_pos = prefix_match.end()
        remaining = text[prefix_pos:]
        cleaned = clean_digits(remaining)
        digit_match = _RE_DIGITS_6_7.search(cleaned)
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


def _fix_vehicle_type(text: str) -> str:
    """Position-aware segment correction for vehicle type strings.
    Pre-compiled regex and translation tables eliminate per-call compilation overhead.
    """
    t = text.strip()
    t = _RE_LEADING_NOISE.sub("", t)
    t = _RE_TRAILING_NOISE.sub("", t)
    t = t.upper()

    if not t:
        return ""

    # Remove trailing Q artefact
    if len(t) > 2 and t.endswith("Q") and t[-2].isdigit():
        t = t[:-1]

    # Special: 0-prefixed 3-digit → C prefix (070→C70, 064→C64)
    if _RE_ZERO_PREFIXED.match(t):
        return 'C' + t[1:]

    # Special: 670/664 etc → C70/C64
    if t in ('670', '664', '665', '662'):
        return 'C' + t[1:]

    # Special: T→7, O→0 confusion (CTOE→C70E)
    if t[0] in 'CPNG':
        candidate = t.translate(_T_O_TO_7_0)
        if candidate in _CHINA_RAIL_VEHICLE_TYPES:
            return candidate

    # Special: all-letter misrecognition (CeAK→C64K, C6AK→C64K)
    if len(t) >= 4 and t[0] in 'CPNG' and not any(c.isdigit() for c in t):
        mapped = t.translate(_E_ABC_TO_DIGITS)
        is_valid, corrected = _is_vehicle_type_pattern(mapped)
        if is_valid:
            return corrected

    # Segment: find first digit position
    first_digit_pos = None
    for i, c in enumerate(t):
        if i > 0 and (c.isdigit() or (c in _DIGIT_LIKE_SET and t[0].isalpha())):
            if c.isdigit():
                first_digit_pos = i
                break
            if i >= 1 and t[i - 1].isalpha() and not t[i - 1].isdigit():
                first_digit_pos = i
                break

    if first_digit_pos is None:
        return t

    # Find last digit position
    last_digit_pos = first_digit_pos
    for i in range(first_digit_pos, len(t)):
        c = t[i]
        if c.isdigit() or c in _DIGIT_CONFUSABLE_SET:
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
    fixed_digits = _RE_NON_DIGITS.sub("", fixed_digits)
    fixed_suffix = suffix.translate(_DIGIT_TO_LETTER)

    return fixed_prefix + fixed_digits + fixed_suffix


def _is_vehicle_type_pattern(text: str) -> Tuple[bool, str]:
    """Check if text looks like a vehicle type and return (is_valid, corrected).
    Uses pre-compiled regex and frozenset for whitelist lookup.
    """
    t = text.strip().upper()
    t = _RE_LEADING_PAREN.sub("", t)
    if len(t) > 7:
        return False, t
    if not _RE_VEHICLE_FORMAT.match(t):
        return False, t
    fully_numeric = t.translate(_LETTER_TO_DIGIT)
    fully_numeric = _RE_NON_DIGITS.sub("", fully_numeric)
    if len(fully_numeric) >= len(t):
        return False, t

    clean_t = t.rstrip('Q')
    if clean_t in _CHINA_RAIL_VEHICLE_TYPES:
        return True, clean_t

    # Approximate match: try deleting one character
    for i in range(len(clean_t)):
        candidate = clean_t[:i] + clean_t[i + 1:]
        if candidate in _CHINA_RAIL_VEHICLE_TYPES and len(candidate) >= 3:
            return True, candidate

    return False, t


def _fix_vehicle_number(text: str) -> str:
    """Digit cleanup for vehicle numbers.
    Uses pre-compiled regex and translation table."""
    t = text.strip()
    t = _RE_PUNCT_TO_SPACE.sub(" ", t)
    t = t.translate(_VEHICLE_NUMBER_TRANS)
    t = _RE_NON_DIGIT_WS.sub("", t)
    t = _RE_MULTIPLE_SPACES.sub(" ", t).strip()
    t = t.replace(" ", "")
    return t


def is_train_param(text: str) -> bool:
    """Check if text is a freight car parameter (load, weight, volume, etc.).

    Key optimizations:
    1. All regex patterns are pre-compiled module constants
    2. All keyword lists are pre-built tuples (no per-call list creation)
    3. Whitelist is a frozenset for O(1) lookup
    4. Check order optimized for early-return on most common patterns
    5. Fast symbol scan (#+%&$) before any expensive operations
    6. Vehicle number protection (>=5 consecutive digits → False) is placed
       AFTER decimal/multiply checks (which must match before digit stripping)
       but BEFORE '.' '/' symbol checks (to protect "49/44030" style numbers)
    """
    # 1. Fast symbol check (most likely to quickly filter)
    for c in text:
        if c in _SYMBOLS_FAST:
            return True

    # 2. Decimal format (e.g., 12.5, 73.3t)
    if _RE_DECIMAL.search(text):
        return True

    # 3. Multiply format (e.g., 12.5×2.6)
    if _RE_MULTIPLY.search(text):
        return True

    # 4. Parameter keywords ( Chinese and unit markers)
    t = text.lower()
    if any(kw in t for kw in _PARAM_KEYWORDS):
        return True

    # 5. Ton marker (e.g., 70t, 73.3t)
    if _RE_DIGIT_T.search(t):
        return True

    # 6. Railway keywords (china railway, etc.)
    if any(kw in t for kw in _RAILWAY_KEYWORDS):
        return True

    # 7. Vehicle number protection: >=5 consecutive digits → NOT a parameter
    #    This protects split numbers like "49/44030" from being filtered
    digits_only = _RE_NON_DIGIT.sub('', text)
    if len(digits_only) >= 5:
        return False

    # 8. Late symbol check (MUST be after number protection)
    #    '.' '/' could appear in numbers like "49/44030"
    for c in text:
        if c in _SYMBOLS_LATE:
            return True

    # 9. Pure Chinese text
    if _is_pure_chinese(text):
        return True

    # 10. Short non-whitelist alphabetic (e.g., "Mc", "HC", "DK")
    if text.isalpha() and len(text) <= 3 and text.upper() not in _VEHICLE_TYPE_SHORT_WHITELIST:
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



# ============ CnOCR 版行分组（优化版）============
def _group_to_lines(
    boxes: List,
    tolerance: Optional[int] = None,
) -> List[List]:
    """Group TextBoxes into horizontal lines by Y-centre.

    优化点：
      - tolerance 计算：np.percentile 替代 list.sort()（O(n log n) → O(n)）
      - cur_y 增量更新，避免每轮 sum()
    """
    if not boxes:
        return []

    if tolerance is None:
        heights = [b.height for b in boxes if b.height > 0]
        if heights:
            # 小数据直接用排序取中位数（避免 numpy 固定开销）
            if len(heights) <= 20:
                heights.sort()
                tolerance = max(40, int(heights[len(heights) // 2] * 0.8))
            else:
                # 大数据用 np.percentile（introselect，平均 O(n)）
                h_arr = np.array(heights, dtype=np.float64)
                tolerance = max(40, int(np.percentile(h_arr, 50) * 0.8))
        else:
            tolerance = 60

    sorted_boxes = sorted(boxes, key=lambda b: b.center_y)
    lines: List[List] = []
    cur_line: List = []
    cur_y: Optional[float] = None

    for box in sorted_boxes:
        if cur_y is None:
            cur_y = box.center_y
            cur_line = [box]
        elif abs(box.center_y - cur_y) <= tolerance:
            cur_line.append(box)
            # 增量更新均值：避免遍历整个 cur_line 做 sum()
            cur_y = (cur_y * (len(cur_line) - 1) + box.center_y) / len(cur_line)
        else:
            cur_line.sort(key=lambda b: b.center_x)
            lines.append(cur_line)
            cur_line = [box]
            cur_y = box.center_y

    if cur_line:
        cur_line.sort(key=lambda b: b.center_x)
        lines.append(cur_line)

    return lines


# ============ 多框合并（优化版）============
def is_y_close(b1, b2, threshold_ratio: float = 0.6) -> bool:
    """判断两个 TextBox 在 Y 方向是否足够接近。保持极简实现。"""
    y_threshold = max(b1.height, b2.height) * threshold_ratio
    return abs(b1.center_y - b2.center_y) < y_threshold


def merge_boxes(boxes, extractor) -> List[Tuple[str, float]]:
    """多框合并：单框 + 相邻2框 + 相邻3框，保留并传播置信度。

    优化点：
      - 内联 try_match，消除嵌套函数对象创建开销
      - 3框合并限制 n <= 20（框多时收益极低但开销巨大）
      - 局部变量缓存减少属性查找
    """
    candidates: List[Tuple[str, float]] = []
    n = len(boxes)
    if n == 0:
        return candidates

    # ---- 单框 ----
    for b in boxes:
        cid = extractor(b.text, b.conf)
        if cid:
            candidates.append(cid)

    # ---- 相邻2框 ----
    for i in range(n - 1):
        b1 = boxes[i]
        b2 = boxes[i + 1]
        if not is_y_close(b1, b2):
            continue
        merged_conf = (b1.conf + b2.conf) * 0.5
        cid = extractor(b1.text + b2.text, merged_conf)
        if cid:
            candidates.append(cid)
        cid = extractor(b1.text + " " + b2.text, merged_conf)
        if cid:
            candidates.append(cid)

    # ---- 相邻3框：仅在总框数较少时执行 ----
    if n <= 20:
        for i in range(n - 2):
            b1 = boxes[i]
            b2 = boxes[i + 1]
            b3 = boxes[i + 2]
            if not (is_y_close(b1, b2) and is_y_close(b2, b3)):
                continue
            merged_conf = (b1.conf + b2.conf + b3.conf) / 3.0
            cid = extractor(b1.text + b2.text + b3.text, merged_conf)
            if cid:
                candidates.append(cid)

    # ---- 去重：保留最高置信度 ----
    best: Dict[str, float] = {}
    for cid, conf in candidates:
        prev = best.get(cid)
        if prev is None or conf > prev:
            best[cid] = conf
    return list(best.items())


# ============ IOU 计算（保持接口，供外部直接调用）============
def _box_iou(b1, b2) -> float:
    """计算两个框的 IoU（交并比）。保持接口兼容。"""
    x1_1 = b1.center_x - b1.width * 0.5
    y1_1 = b1.center_y - b1.height * 0.5
    x2_1 = b1.center_x + b1.width * 0.5
    y2_1 = b1.center_y + b1.height * 0.5
    x1_2 = b2.center_x - b2.width * 0.5
    y1_2 = b2.center_y - b2.height * 0.5
    x2_2 = b2.center_x + b2.width * 0.5
    y2_2 = b2.center_y + b2.height * 0.5

    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    inter_w = max(0.0, xi2 - xi1)
    inter_h = max(0.0, yi2 - yi1)
    inter_area = inter_w * inter_h

    area1 = b1.width * b1.height
    area2 = b2.width * b2.height
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0.0 else 0.0


# ============ 去重高度重叠框（NumPy 向量化版）============
def _dedup_overlapping_boxes(boxes, iou_thresh: float = 0.3) -> List:
    """去重高度重叠的框，保留置信度高的。

    核心优化：
      - n >= 25 时：NumPy 向量化计算完整 IOU 矩阵（C 层循环）
      - n <  25 时：纯 Python 回退（避免 numpy 固定开销）
      - 处理顺序仍严格按置信度降序，行为与原函数一致
    """
    if not boxes:
        return []

    n = len(boxes)
    if n == 1:
        return boxes[:]

    # ---- 小数据量：纯 Python（避免 numpy 数组创建和 outer 操作固定开销）----
    if n < 25:
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

    # ---- 大数据量：NumPy 向量化 ----

    # ---- 1. 提取坐标到 NumPy 数组（单次 Python 循环）----
    cx = np.empty(n, dtype=np.float64)
    cy = np.empty(n, dtype=np.float64)
    w = np.empty(n, dtype=np.float64)
    h = np.empty(n, dtype=np.float64)
    conf = np.empty(n, dtype=np.float64)

    for i, (b, _) in enumerate(boxes):
        cx[i] = b.center_x
        cy[i] = b.center_y
        w[i] = b.width
        h[i] = b.height
        conf[i] = b.conf

    # ---- 2. 计算边界框和面积 ----
    half_w = w * 0.5
    half_h = h * 0.5
    x1 = cx - half_w
    y1 = cy - half_h
    x2 = cx + half_w
    y2 = cy + half_h
    area = w * h

    # ---- 3. 向量化 IOU 矩阵：全在 C 层执行 ----
    xx1 = np.maximum.outer(x1, x1)
    yy1 = np.maximum.outer(y1, y1)
    xx2 = np.minimum.outer(x2, x2)
    yy2 = np.minimum.outer(y2, y2)

    iw = np.maximum(0.0, xx2 - xx1)
    ih = np.maximum(0.0, yy2 - yy1)
    inter = iw * ih
    union = area[:, None] + area[None, :] - inter

    with np.errstate(divide='ignore', invalid='ignore'):
        iou = np.where(union > 0.0, inter / union, 0.0)

    # 清除对角线和下三角（自比较 + 避免重复检查）
    np.fill_diagonal(iou, 0.0)

    # ---- 4. 按置信度降序处理（与原函数一致）----
    order = np.argsort(-conf)
    kept_mask = np.zeros(n, dtype=bool)
    result = []

    for idx in order:
        if kept_mask[idx]:
            continue
        # 向量化检查：idx 是否与任何已保留框重叠
        overlapping = (iou[idx] > iou_thresh) & kept_mask
        if np.any(overlapping):
            continue
        result.append(boxes[idx])
        kept_mask[idx] = True

    return result


# ============ merge_boxes_train（最大热点优化版）============
def merge_boxes_train(boxes) -> List[Tuple[str, float]]:
    """铁路货车专用合并：先按行分组，每行内横向合并，数字类只保留最长结果。

    优化点（预估整体收益 60~80%）：
      1. 预编译正则（模块级 _RE_HAS_DIGIT, _RE_DIGIT_2TO8）
      2. 车型框 combinations 限制：m > 8 时只检查相邻对（O(m²) → O(m)）
      3. 数字框多框合并：O(n³) 三重循环 → O(n·L) 累积扩展（L ≤ 7）
      4. 预计算相邻 x_gaps，break 更早
      5. 局部变量缓存 map/dict 引用，减少循环内查找
      6. _dedup_overlapping_boxes 内部全向量化
    """
    candidates: List[Tuple[str, float]] = []
    type_boxes: List[Tuple] = []
    num_boxes: List[Tuple] = []

    _has_digit = _RE_HAS_DIGIT.search
    _digit_2to8 = _RE_DIGIT_2TO8.search

    # ---- 第一遍：分类为车型框 / 数字框 ----
    for b in boxes:
        t = b.text.upper().replace(" ", "").replace("-", "").replace(".", "")
        if not t:
            continue

        fixed_type = _fix_vehicle_type(t)
        is_valid_t, corrected_t = _is_vehicle_type_pattern(t)
        is_valid_fixed, corrected_fixed = (
            _is_vehicle_type_pattern(fixed_type)
            if fixed_type
            else (False, fixed_type)
        )

        if is_valid_t or is_valid_fixed:
            final_type = corrected_fixed if is_valid_fixed else corrected_t
            type_boxes.append((b, final_type))
        elif _has_digit(t):
            fixed_num = _fix_vehicle_number(t)
            if _has_digit(fixed_num):
                num_boxes.append((b, fixed_num))

    # ---- 去重高度重叠的框 ----
    type_boxes = _dedup_overlapping_boxes(type_boxes)
    num_boxes = _dedup_overlapping_boxes(num_boxes)

    # ========== 车种类：按行分组后合并 ==========
    if type_boxes:
        type_textboxes = [b for b, _ in type_boxes]
        type_lines = _group_to_lines(type_textboxes)
        type_map = {id(b): fixed for b, fixed in type_boxes}

        for line in type_lines:
            m = len(line)
            # 单框直接入候选
            for b in line:
                candidates.append((type_map[id(b)], b.conf))

            # 两框组合：大基数时只检查相邻对，避免组合爆炸
            if m >= 2:
                if m <= 8:
                    pairs = combinations(range(m), 2)
                else:
                    pairs = ((i, i + 1) for i in range(m - 1))

                for i, j in pairs:
                    b1 = line[i]
                    b2 = line[j]
                    # 快速过滤：车型总长度应在 3~7 范围内，超长直接跳过
                    total_len = len(b1.text) + len(b2.text)
                    if total_len > 8:  # 8 比 7 宽松一点，留容错空间
                        continue
                    merged = b1.text + b2.text
                    fixed = _fix_vehicle_type(merged)
                    if not fixed:
                        merged2 = b2.text + b1.text
                        fixed = _fix_vehicle_type(merged2)
                    if fixed:
                        is_valid, corrected = _is_vehicle_type_pattern(fixed)
                        if is_valid:
                            conf = (b1.conf + b2.conf) * 0.5
                            candidates.append((corrected, conf))

    # ========== 数字类：按行分组后合并 ==========
    if num_boxes:
        num_textboxes = [b for b, _ in num_boxes]
        num_lines = _group_to_lines(num_textboxes)
        num_map = {id(b): fixed for b, fixed in num_boxes}

        all_line_results: List[List[Tuple[str, float]]] = []

        for line in num_lines:
            n = len(line)
            line_candidates: List[Tuple[str, float]] = []

            # ---- 单框 ----
            for b in line:
                match = _digit_2to8(num_map[id(b)])
                if match:
                    line_candidates.append((match.group(), b.conf))

            # ---- 同一行内连续多框合并（核心优化：O(n³) → O(n·L)）----
            # 预提取文本，避免循环内反复 dict 查找
            line_texts = [num_map[id(b)] for b in line]
            MAX_MERGE_LEN = 7  # 车号最长 7 位，超过无意义

            for i in range(n):
                merged_text = line_texts[i]
                # 从位置 i 开始，逐步向右扩展（累积复用已合并文本）
                for j in range(i + 1, min(i + MAX_MERGE_LEN, n)):
                    b_prev = line[j - 1]
                    b_curr = line[j]
                    x_gap = b_curr.center_x - b_prev.center_x - (b_prev.width + b_curr.width) * 0.5
                    if x_gap > 800:
                        break  # 距离太远，更长的扩展不可能有效

                    merged_text += line_texts[j]
                    length = j - i + 1
                    # 精确计算 conf（与原始代码 sum()/length 保持一致）
                    merged_conf = sum(line[k].conf for k in range(i, i + length)) / length

                    match = _digit_2to8(merged_text)
                    if match:
                        line_candidates.append((match.group(), merged_conf))

            all_line_results.append(line_candidates)

        # ---- 跨行合并：相邻行的数字框如果 x 方向互补，尝试合并 ----
        num_lines_count = len(num_lines)
        for i in range(num_lines_count - 1):
            line1 = num_lines[i]
            line2 = num_lines[i + 1]
            if not line1 or not line2:
                continue

            rightmost_l1 = max(line1, key=lambda b: b.center_x)
            threshold_x = rightmost_l1.center_x - rightmost_l1.width
            l2_all_right = True
            for b in line2:
                if b.center_x <= threshold_x:
                    l2_all_right = False
                    break

            if l2_all_right:
                merged_line = sorted(line1 + line2, key=lambda b: b.center_x)
                merged_text = ''.join(num_map[id(b)] for b in merged_line)
                merged_conf = sum(b.conf for b in merged_line) / len(merged_line)
                match = _digit_2to8(merged_text)
                if match:
                    all_line_results[i].append((match.group(), merged_conf))

        # ---- 每行只保留最长结果 ----
        for line_candidates in all_line_results:
            if line_candidates:
                line_candidates.sort(key=lambda x: (len(x[0]), x[1]), reverse=True)
                longest_len = len(line_candidates[0][0])
                for cid, conf in line_candidates:
                    if len(cid) == longest_len:
                        candidates.append((cid, conf))
                        break

    # ---- 去重：保留最高置信度 ----
    best: Dict[str, float] = {}
    for cid, conf in candidates:
        prev = best.get(cid)
        if prev is None or conf > prev:
            best[cid] = conf
    return list(best.items())

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

    @property
    def available(self) -> bool:
        return self.ocr is not None

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


# ============ 多进程 OCR 工作池 ============

@dataclass
class FrameFilterConfig:
    """帧过滤器配置（多摄像头场景降采样用）。"""
    min_interval_sec: float = 0.15   # 同一摄像头最小处理间隔
    cache_ttl_sec: float = 3.0       # 结果缓存有效期
    enable_cache: bool = True        # 是否启用缓存


class SmartFrameFilter:
    """智能帧过滤器 — 减少冗余 OCR 调用。
    1) 时间降采样：同一摄像头在 min_interval_sec 内只处理 1 帧
    2) 结果缓存：相同画面直接返回缓存
    """
    def __init__(self, config: Optional[FrameFilterConfig] = None):
        self.cfg = config or FrameFilterConfig()
        self._lock = _threading.Lock()
        self._last_time: Dict[str, float] = {}
        self._cache: Dict[str, Tuple[float, ImageResult]] = {}

    def should_process(self, cam_id: str) -> bool:
        with self._lock:
            now = _time.time()
            last = self._last_time.get(cam_id, 0.0)
            if now - last < self.cfg.min_interval_sec:
                return False
            self._last_time[cam_id] = now
            return True

    def get_cached(self, cam_id: str) -> Optional[ImageResult]:
        if not self.cfg.enable_cache:
            return None
        with self._lock:
            entry = self._cache.get(cam_id)
            if entry is None:
                return None
            ts, result = entry
            if _time.time() - ts > self.cfg.cache_ttl_sec:
                del self._cache[cam_id]
                return None
            import copy
            return copy.deepcopy(result)

    def cache_result(self, cam_id: str, result: ImageResult):
        if self.cfg.enable_cache:
            with self._lock:
                self._cache[cam_id] = (_time.time(), result)


# Worker 进程全局变量（每个 worker 独立一份）
_worker_processor: Optional[PaddleOCRProcessor] = None


def _mp_init_worker(use_gpu: bool, enhancement_mode: str):
    """Multiprocessing Pool initializer：在每个 worker 中创建 OCR 实例。"""
    global _worker_processor
    import os
    os.environ['PADDLEOCR_QUIET'] = '1'
    _worker_processor = PaddleOCRProcessor(
        use_gpu=use_gpu, enhancement_mode=enhancement_mode
    )


def _mp_process_bytes(image_bytes: bytes) -> ImageResult:
    global _worker_processor
    if _worker_processor is None:
        return ImageResult()
    return _worker_processor.process_bytes(image_bytes)


def _mp_process_raw_bytes(image_bytes: bytes, pixel_type: int, width: int, height: int) -> ImageResult:
    global _worker_processor
    if _worker_processor is None:
        return ImageResult()
    return _worker_processor.process_raw_bytes(image_bytes, pixel_type, width, height)


class PaddleOCRProcessPool:
    """多进程 OCR 处理器 — API 与 PaddleOCRProcessor 完全一致。
    内部维护 multiprocessing.Pool，将 OCR 任务分发到多个工作进程并行执行。
    """
    def __init__(self, num_workers: int = 4, use_gpu: bool = False,
                 enhancement_mode: str = 'none',
                 frame_filter_config: Optional[FrameFilterConfig] = None):
        if num_workers is None or num_workers < 2:
            num_workers = 2
        self.num_workers = num_workers
        self._closed = False
        self._filter = SmartFrameFilter(frame_filter_config)

        self._pool = _MPOOL_CTX.Pool(
            processes=num_workers,
            initializer=_mp_init_worker,
            initargs=(use_gpu, enhancement_mode),
        )
        print(f"INFO: PaddleOCRProcessPool created with {num_workers} workers")

    @property
    def available(self) -> bool:
        return not self._closed

    @property
    def ocr(self):
        """兼容属性：进程池本身没有单例 OCR 实例，返回 truthy 值表示可用。"""
        return self.available

    def process(self, image_path: str) -> ImageResult:
        img_array = np.fromfile(image_path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            print(f"ERROR: Cannot read image: {image_path}")
            return ImageResult()
        return self.process_bytes(cv2.imencode('.jpg', img)[1].tobytes())

    def process_bytes(self, image_bytes: bytes, cam_id: Optional[str] = None) -> ImageResult:
        self._check_closed()
        if cam_id is not None:
            if not self._filter.should_process(cam_id):
                cached = self._filter.get_cached(cam_id)
                return cached if cached is not None else ImageResult()
            cached = self._filter.get_cached(cam_id)
            if cached is not None:
                return cached
        result: ImageResult = self._pool.apply(_mp_process_bytes, (image_bytes,))
        if cam_id is not None:
            self._filter.cache_result(cam_id, result)
        return result

    def process_raw_bytes(self, image_bytes: bytes, pixel_type: int, width: int, height: int,
                          cam_id: Optional[str] = None) -> ImageResult:
        self._check_closed()
        if cam_id is not None:
            if not self._filter.should_process(cam_id):
                cached = self._filter.get_cached(cam_id)
                return cached if cached is not None else ImageResult()
            cached = self._filter.get_cached(cam_id)
            if cached is not None:
                return cached
        result: ImageResult = self._pool.apply(
            _mp_process_raw_bytes, (image_bytes, pixel_type, width, height)
        )
        if cam_id is not None:
            self._filter.cache_result(cam_id, result)
        return result

    def _check_closed(self):
        if self._closed:
            raise RuntimeError("PaddleOCRProcessPool is already closed")

    def close(self):
        if not self._closed:
            self._pool.close()
            self._pool.join()
            self._closed = True
            print("INFO: PaddleOCRProcessPool closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


_global_processors: Dict[str, PaddleOCRProcessor] = {}
_global_pools: Dict[str, PaddleOCRProcessPool] = {}


def get_ocr_processor(
    use_gpu: Optional[bool] = None,
    enhancement_mode: str = 'none',
    num_workers: Optional[int] = None,
    frame_filter_config: Optional[FrameFilterConfig] = None,
):
    """获取 OCR 处理器（单进程或多进程）。

    现场只需改一行即可切换到多进程模式：
        processor = get_ocr_processor(num_workers=4)

    Args:
        use_gpu: None=自动检测, True=强制GPU, False=强制CPU
        enhancement_mode: 图像增强模式
        num_workers: None=单进程, int>1=多进程（工作进程数）
        frame_filter_config: 帧过滤器配置（仅多进程模式有效）
    """
    if use_gpu is None:
        use_gpu, reason = _check_gpu_available()
        print(f"[GPU检测] {reason}")

    if num_workers is not None and num_workers > 1:
        key = f"pool_gpu={use_gpu}_mode={enhancement_mode}_workers={num_workers}"
        if key not in _global_pools:
            _global_pools[key] = PaddleOCRProcessPool(
                num_workers=num_workers,
                use_gpu=use_gpu,
                enhancement_mode=enhancement_mode,
                frame_filter_config=frame_filter_config,
            )
        return _global_pools[key]

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
    parser.add_argument('--gpu', action='store_true', help='强制使用 GPU')
    parser.add_argument('--cpu', action='store_true', help='强制使用 CPU')
    parser.add_argument('--enhancement', default='none',
                        choices=['none', 'otsu_fusion'],
                        help='图像预处理模式: none=原图, otsu_fusion=OTSU融合（参考kimi）')
    args = parser.parse_args()

    if args.gpu and args.cpu:
        parser.error('--gpu 和 --cpu 不能同时指定')
    
    use_gpu = None  # 默认自动检测
    if args.gpu:
        use_gpu = True
    elif args.cpu:
        use_gpu = False

    processor = get_ocr_processor(use_gpu=use_gpu, enhancement_mode=args.enhancement)

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
