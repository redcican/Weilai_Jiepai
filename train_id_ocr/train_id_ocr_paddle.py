#!/usr/bin/env python3
"""
Train ID OCR — PaddleOCR 单图识别版 (基于 video_paddle_v6.py 改造)

从单张图片中识别：
  - 上半区：集装箱箱号
  - 下半区：铁路货车车种 + 车号

基于 train_id_ocr_video_paddle_v6.py，移除了视频/时序相关逻辑，
保留核心 OCR + 上下分区 + 多框合并能力。

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
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from paddleocr import PaddleOCR


# ============ 集装箱前缀纠错映射表 ============
PREFIX_CORRECTION = {
    'TEJU': 'TBJU', 'TRJU': 'TBJU', 'TPJU': 'TBJU', 'T8JU': 'TBJU',
    'T3JU': 'TBJU', 'TCJU': 'TBJU', 'TBIU': 'TBJU', 'T0JU': 'TBJU',
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

# ============ 铁路货车车种纠错 ============
TRAIN_TYPE_CORRECTION = {
    'C7OE': 'C70E', 'C7DE': 'C70E', 'C70B': 'C70E', 'C7BE': 'C70E',
    'C7O': 'C70', 'C7D': 'C70', 'COE': 'C70E',
    'C64R': 'C64K', 'C64H': 'C64K', 'C6AK': 'C64K',
    'G70E': 'C70E',
    'C70': 'C70E',
}

COMMON_TRAIN_TYPES = {
    'C70E', 'C70', 'C64K', 'C64', 'C62A', 'C62',
    'P64', 'P64K', 'P70', 'N17', 'NX70',
}

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


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
    """单张图片的识别结果"""
    containers: List[str] = field(default_factory=list)
    train_types: List[str] = field(default_factory=list)
    train_numbers: List[str] = field(default_factory=list)


# ============ 工具函数（与 v6 保持一致）============
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


# ============ 集装箱提取 ============
def extract_container_id(text: str) -> Optional[str]:
    text = text.upper().replace(" ", "").replace("-", "").replace(".", "")
    match = re.search(r'([A-Z]{4})(\d{6,7})', text)
    if match:
        prefix = match.group(1)
        digits = match.group(2)[:6]
        prefix = correct_prefix(prefix)
        return f"{prefix}{digits}"
    loose_match = re.search(r'([A-Z0-9]{4})(\d{6,7})', text)
    if loose_match:
        prefix = loose_match.group(1)
        digits = loose_match.group(2)[:6]
        fixed_prefix = fix_prefix_digits(prefix)
        if fixed_prefix:
            return f"{fixed_prefix}{digits}"
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
                return f"{fixed_prefix}{digits}"
            if prefix.isalpha():
                prefix = correct_prefix(prefix)
                return f"{prefix}{digits}"
    return None


# ============ 铁路货车提取 ============
def fix_train_type(text: str) -> Optional[str]:
    if text in TRAIN_TYPE_CORRECTION:
        return TRAIN_TYPE_CORRECTION[text]
    if text in COMMON_TRAIN_TYPES:
        return text
    if text == 'C':
        return 'C70E'
    if text == '70E':
        return 'C70E'
    if text == '70':
        return 'C70'
    if len(text) == 4:
        fixed = list(text)
        for i, c in enumerate(fixed):
            if c == 'O' and i >= 1:
                fixed[i] = '0'
            if c == 'I' and i >= 1:
                fixed[i] = '1'
        result = ''.join(fixed)
        if result in COMMON_TRAIN_TYPES:
            return result
    if len(text) >= 4 and text[0].isalpha():
        first = text[0]
        digits = ''.join(c for c in text[1:] if c.isdigit())
        letters = ''.join(c for c in text[1:] if c.isalpha())
        if len(digits) >= 2:
            candidate = f"{first}{digits[-2:]}{letters[:1]}"
            if candidate in COMMON_TRAIN_TYPES:
                return candidate
    return None


def is_train_param(text: str) -> bool:
    t = text.lower()
    if any(c in t for c in ['t', 'm³', 'm3', '载', '重', '自', '容', '积', '换', '长', '定']):
        return True
    if any(c in text for c in ['.', '(', ')', 'X', '×', '*']):
        return True
    if all('\u4e00' <= c <= '\u9fff' for c in text):
        return True
    if any(c in text for c in ['#', '+', '-']):
        return True
    if text.isalpha() and len(text) <= 3 and text.upper() not in ['C70', 'C64', 'P64', 'P70', 'N17']:
        return True
    return False


def extract_train_id(text: str) -> Optional[str]:
    text = text.upper().replace(" ", "").replace("-", "").replace(".", "")
    match = re.search(r'(\d{3,8})', text)
    if match:
        return match.group(1)
    for type_pattern in [r'([A-Z]\d{2,4}?[A-Z])', r'([A-Z]\d{2,4}?)']:
        match = re.search(type_pattern, text)
        if match:
            fixed_type = fix_train_type(match.group(1))
            if fixed_type:
                return fixed_type
    return None


# ============ OCR 框解析（上下分区）============
def parse_ocr_boxes(result, img_height: int) -> Tuple[List[TextBox], List[TextBox]]:
    """解析OCR结果，按y坐标分上半区(集装箱)/下半区(铁路货车)"""
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
                if not is_train_param(text):
                    lower_boxes.append(box)

    upper_boxes.sort(key=lambda b: b.center_x)
    lower_boxes.sort(key=lambda b: b.center_x)
    return upper_boxes, lower_boxes


# ============ 多框合并 ============
def is_y_close(b1: TextBox, b2: TextBox, threshold_ratio: float = 0.6) -> bool:
    y_threshold = max(b1.height, b2.height) * threshold_ratio
    return abs(b1.center_y - b2.center_y) < y_threshold


def merge_boxes(boxes: List[TextBox], extractor) -> List[str]:
    """多框合并：单框 + 相邻2框 + 相邻3框"""
    candidates = []
    n = len(boxes)

    # 单框
    for b in boxes:
        cid = extractor(b.text)
        if cid:
            candidates.append(cid)

    # 相邻2框
    for i in range(n - 1):
        b1, b2 = boxes[i], boxes[i + 1]
        if not is_y_close(b1, b2):
            continue
        merged_text = b1.text + b2.text
        cid = extractor(merged_text)
        if cid:
            candidates.append(cid)
        merged_text_space = b1.text + " " + b2.text
        cid2 = extractor(merged_text_space)
        if cid2 and cid2 not in candidates:
            candidates.append(cid2)

    # 相邻3框
    for i in range(n - 2):
        b1, b2, b3 = boxes[i], boxes[i + 1], boxes[i + 2]
        if not (is_y_close(b1, b2) and is_y_close(b2, b3)):
            continue
        merged_text = b1.text + b2.text + b3.text
        cid = extractor(merged_text)
        if cid:
            candidates.append(cid)

    # 去重
    return list(dict.fromkeys(candidates))


def merge_boxes_train(boxes: List[TextBox]) -> List[str]:
    """铁路货车专用合并"""
    candidates = []

    type_boxes = []
    num_boxes = []
    for b in boxes:
        t = b.text.upper().replace(" ", "").replace("-", "").replace(".", "")
        if re.match(r'^[A-Z]\d{0,4}[A-Z]?$', t):
            fixed = fix_train_type(t)
            if fixed:
                type_boxes.append((b, fixed))
        elif t.isdigit():
            num_boxes.append(b)

    # 车种类
    for b, fixed in type_boxes:
        candidates.append(fixed)
    m = len(type_boxes)
    if m >= 2:
        from itertools import combinations
        for i, j in combinations(range(m), 2):
            b1, fixed1 = type_boxes[i]
            b2, fixed2 = type_boxes[j]
            merged = b1.text + b2.text
            fixed = fix_train_type(merged)
            if not fixed:
                merged2 = b2.text + b1.text
                fixed = fix_train_type(merged2)
            if fixed:
                candidates.append(fixed)

    # 数字类
    num_boxes.sort(key=lambda b: b.center_x)
    n = len(num_boxes)
    for b in num_boxes:
        match = re.search(r'\d{3,8}', b.text)
        if match:
            candidates.append(match.group())
    for length in range(2, min(5, n + 1)):
        for i in range(n - length + 1):
            merged = ''.join(num_boxes[j].text for j in range(i, i + length))
            match = re.search(r'\d{3,8}', merged)
            if match:
                candidates.append(match.group())

    # 去重
    return list(dict.fromkeys(candidates))


# ============ 单图处理入口 ============
class PaddleOCRProcessor:
    """基于 PaddleOCR 的单图识别处理器"""

    def __init__(self, use_gpu: bool = True):
        self.ocr = None
        try:
            import paddle
            gpu_available = use_gpu and paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
            if gpu_available:
                try:
                    self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False, use_gpu=True)
                    print("INFO: PaddleOCR initialized on GPU")
                except Exception:
                    gpu_available = False
            if not gpu_available:
                self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False, use_gpu=False)
                print("INFO: PaddleOCR initialized on CPU")
        except Exception as e:
            print(f"ERROR: PaddleOCR init failed: {e}")

    def process(self, image_path: str) -> ImageResult:
        """处理单张图片，返回识别结果"""
        if self.ocr is None:
            return ImageResult()

        img = cv2.imread(image_path)
        if img is None:
            print(f"ERROR: Cannot read image: {image_path}")
            return ImageResult()

        h, w = img.shape[:2]
        result = self.ocr.ocr(img, cls=True)
        upper_boxes, lower_boxes = parse_ocr_boxes(result, h)

        # 上半区提取集装箱
        container_ids = merge_boxes(upper_boxes, extract_container_id)

        # 下半区提取铁路货车
        train_ids = merge_boxes_train(lower_boxes)

        # 分类车种和车号
        train_types = []
        train_numbers = []
        for tid in train_ids:
            if re.match(r'^[A-Z]\d{2,4}[A-Z]?$', tid):
                train_types.append(tid)
            elif tid.isdigit():
                train_numbers.append(tid)

        return ImageResult(
            containers=container_ids,
            train_types=train_types,
            train_numbers=train_numbers,
        )


# ============ CLI 入口 ============
def main():
    parser = argparse.ArgumentParser(description='铁路图片 OCR 识别 - PaddleOCR 单图版')
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

        print(f"  集装箱: {result.containers}")
        print(f"  车种: {result.train_types}")
        print(f"  车号: {result.train_numbers}")

        all_results[img_path.name] = {
            "containers": result.containers,
            "train_types": result.train_types,
            "train_numbers": result.train_numbers,
        }

    # 保存结果
    out_path = os.path.join(args.output, 'result.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")


if __name__ == '__main__':
    main()
