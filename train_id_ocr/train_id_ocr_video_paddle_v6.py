#!/usr/bin/env python3
"""
集装箱箱号 + 铁路货车车号 视频OCR识别 - PaddleOCR版本 V6
核心改进：
  1. 1帧1次OCR，按检测框y坐标分上半区(集装箱)/下半区(铁路货车)
  2. 集装箱：保留V5全部后处理（多框合并、末尾截断、前缀纠错、数字容错）
  3. 铁路货车：C70E/C64K等车种 + 5位车号，支持多框合并
"""

import os
# Fix Intel OpenMP duplicate library conflict (must be before numpy/paddle import)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import re
import json
import argparse
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from collections import Counter, defaultdict
from datetime import timedelta

import cv2
import paddle
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
    'G70E': 'C70E',  # G和C混淆
    'C70': 'C70E',   # 视频里实际都是C70E，E常被漏识
}

COMMON_TRAIN_TYPES = {
    'C70E', 'C70', 'C64K', 'C64', 'C62A', 'C62',
    'P64', 'P64K', 'P70', 'N17', 'NX70',
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
    train_fragments: List[dict] = field(default_factory=list)  # 铁路货车碎片，用于跨帧拼接


def format_timestamp(sec: float) -> str:
    td = timedelta(seconds=sec)
    mm, ss = divmod(td.seconds, 60)
    return f"{mm:02d}:{ss:05.2f}"


# ============ 集装箱提取（V5逻辑） ============
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


# ============ 铁路货车提取（新增） ============
def fix_train_type(text: str) -> Optional[str]:
    """修复车种型号，如 C7OE→C70E, C→C70E, 70E→C70E"""
    if text in TRAIN_TYPE_CORRECTION:
        return TRAIN_TYPE_CORRECTION[text]
    if text in COMMON_TRAIN_TYPES:
        return text
    # 单字母C → C70E（拆框碎片）
    if text == 'C':
        return 'C70E'
    # 70E → C70E（拆框碎片，补C）
    if text == '70E':
        return 'C70E'
    if text == '70':
        return 'C70'
    # C7OE → C70E (O→0, I→1)
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
    # C1770E → C70E：提取首尾字母+后2位数字（拆框时后2位更可靠）
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
    """判断是否为货车参数框（载重/自重/容积等），应排除"""
    t = text.lower()
    # 含单位或参数特征字符
    if any(c in t for c in ['t', 'm³', 'm3', '载', '重', '自', '容', '积', '换', '长', '定']):
        return True
    if any(c in text for c in ['.', '(', ')', 'X', '×', '*']):
        return True
    # 纯中文框
    if all('\u4e00' <= c <= '\u9fff' for c in text):
        return True
    # 特殊符号干扰
    if any(c in text for c in ['#', '+', '-']):
        return True
    # 纯字母但长度<=3且不是常见车种（RE, RK, AR, AK, ONILA等碎片）
    if text.isalpha() and len(text) <= 3 and text.upper() not in ['C70', 'C64', 'P64', 'P70', 'N17']:
        return True
    # 不再过滤1位数字，允许其参与合并（合并后长度>=5才输出）
    pass
    return False


def extract_train_id(text: str, conf: float) -> Optional[Tuple[str, float]]:
    """提取铁路货车信息，分开识别车种或车号，不拼接"""
    text = text.upper().replace(" ", "").replace("-", "").replace(".", "")
    
    # 模式1：车号 3-8位数字（保留碎片用于跨帧拼接）
    match = re.search(r'(\d{3,8})', text)
    if match:
        return (match.group(1), conf)
    
    # 模式2：车种（如 C70E, C70）
    for type_pattern in [r'([A-Z]\d{2,4}?[A-Z])', r'([A-Z]\d{2,4}?)']:
        match = re.search(type_pattern, text)
        if match:
            fixed_type = fix_train_type(match.group(1))
            if fixed_type:
                return (fixed_type, conf)
    
    return None


# ============ 帧处理 ============
def extract_frames(video_path: str, output_dir: str, max_duration_sec: Optional[float] = None,
                     interval_sec: float = 0.5, skip_extract: bool = False,
                     start_time_sec: Optional[float] = None, end_time_sec: Optional[float] = None) -> Tuple[List[Tuple[float, str]], float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    step = max(1, int(fps * interval_sec))
    cap.release()

    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "_frames")
    os.makedirs(frames_dir, exist_ok=True)

    if skip_extract:
        existing = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
        if existing:
            frame_paths = []
            for fname in existing:
                ts_match = re.search(r'frame_([\d.]+)\.jpg', fname)
                ts = float(ts_match.group(1)) if ts_match else 0.0
                if start_time_sec is not None and ts < start_time_sec:
                    continue
                if end_time_sec is not None and ts > end_time_sec:
                    continue
                frame_paths.append((ts, os.path.join(frames_dir, fname)))
            frame_paths.sort(key=lambda x: x[0])
            print(f"INFO: Reusing {len(frame_paths)} existing frames in time range (skipped extract)")
            return frame_paths, fps
        print("WARN: --skip-extract specified but no frames found, will extract.")

    for f in os.listdir(frames_dir):
        if f.endswith('.jpg'):
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
    print(f"INFO: Extracted {len(frame_paths)} frames (fps={fps:.2f}, step={step} frames, range={start_time_sec or 0:.1f}s~{end_time_sec or duration:.1f}s)")
    return frame_paths, fps


def parse_ocr_boxes(result, img_height: int) -> Tuple[List[TextBox], List[TextBox]]:
    """解析OCR结果，按y坐标分上半区/下半区，下半区过滤参数干扰"""
    upper_boxes = []
    lower_boxes = []
    split_y = img_height * 0.55  # 分界点在55%高度
    
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
                # 下半区：过滤货车参数干扰
                if not is_train_param(text):
                    lower_boxes.append(box)
    
    upper_boxes.sort(key=lambda b: b.center_x)
    lower_boxes.sort(key=lambda b: b.center_x)
    return upper_boxes, lower_boxes


def merge_boxes(boxes: List[TextBox], extractor) -> List[Tuple[str, float]]:
    """多框合并策略：单框 + 相邻2框 + 相邻3框（集装箱用，要求y相近）"""
    candidates = []
    n = len(boxes)
    
    def is_y_close(b1: TextBox, b2: TextBox, threshold_ratio: float = 0.6) -> bool:
        y_threshold = max(b1.height, b2.height) * threshold_ratio
        return abs(b1.center_y - b2.center_y) < y_threshold
    
    def try_match(text: str, conf: float) -> Optional[Tuple[str, float]]:
        return extractor(text, conf)
    
    # 单框匹配
    for b in boxes:
        cid = try_match(b.text, b.conf)
        if cid:
            candidates.append(cid)
    
    # 相邻2框合并
    for i in range(n - 1):
        b1, b2 = boxes[i], boxes[i + 1]
        if not is_y_close(b1, b2):
            continue
        merged_text = b1.text + b2.text
        merged_conf = (b1.conf + b2.conf) / 2
        cid = try_match(merged_text, merged_conf)
        if cid:
            candidates.append(cid)
        merged_text_space = b1.text + " " + b2.text
        cid2 = try_match(merged_text_space, merged_conf)
        if cid2 and cid2 not in candidates:
            candidates.append(cid2)
    
    # 相邻3框合并
    for i in range(n - 2):
        b1, b2, b3 = boxes[i], boxes[i + 1], boxes[i + 2]
        if not (is_y_close(b1, b2) and is_y_close(b2, b3)):
            continue
        merged_text = b1.text + b2.text + b3.text
        merged_conf = (b1.conf + b2.conf + b3.conf) / 3
        cid = try_match(merged_text, merged_conf)
        if cid:
            candidates.append(cid)
    
    # 去重
    best = {}
    for cid, conf in candidates:
        if cid not in best or conf > best[cid]:
            best[cid] = conf
    return [(cid, conf) for cid, conf in best.items()]


def merge_boxes_train(boxes: List[TextBox]) -> List[Tuple[str, float]]:
    """铁路货车专用合并：车种类任意合并，数字类只连续合并，避免字母数字混拼假阳性"""
    candidates = []
    
    # 分类
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
    
    # 车种类：单框 + 任意2框合并
    for b, fixed in type_boxes:
        candidates.append((fixed, b.conf))
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
                conf = (b1.conf + b2.conf) / 2
                candidates.append((fixed, conf))
    
    # 数字类：按x排序，连续2-3框合并（只匹配纯数字，避免混入字母）
    num_boxes.sort(key=lambda b: b.center_x)
    n = len(num_boxes)
    for b in num_boxes:
        match = re.search(r'\d{3,8}', b.text)
        if match:
            candidates.append((match.group(), b.conf))
    for length in range(2, min(5, n + 1)):
        for i in range(n - length + 1):
            merged = ''.join(num_boxes[j].text for j in range(i, i + length))
            match = re.search(r'\d{3,8}', merged)
            if match:
                conf = sum(num_boxes[j].conf for j in range(i, i + length)) / length
                candidates.append((match.group(), conf))
    
    # 去重
    best = {}
    for cid, conf in candidates:
        if cid not in best or conf > best[cid]:
            best[cid] = conf
    return [(cid, conf) for cid, conf in best.items()]


def process_frame(ocr_engine, img_path: str, ts: float) -> FrameResult:
    """处理单帧：1次OCR，分上半/下半分别提取"""
    img = cv2.imread(img_path)
    if img is None:
        return FrameResult(timestamp_sec=ts)
    
    h, w = img.shape[:2]
    result = ocr_engine.ocr(img, cls=True)
    upper_boxes, lower_boxes = parse_ocr_boxes(result, h)
    
    # 收集所有原始文本
    all_texts = [(b.text, b.conf) for b in upper_boxes + lower_boxes]
    
    # 上半区提取集装箱
    container_ids = merge_boxes(upper_boxes, extract_container_id)
    
    # 下半区提取铁路货车（专用合并：允许垂直分布）
    train_ids = merge_boxes_train(lower_boxes)
    
    # 收集铁路货车碎片（type / paired / single）
    fragments = []
    type_boxes = []
    num_boxes = []
    for b in lower_boxes:
        t = b.text.upper().replace(" ", "").replace("-", "").replace(".", "")
        if not t:
            continue
        # 车种类
        if re.match(r'^[A-Z]\d{0,4}[A-Z]?$', t):
            fixed = fix_train_type(t)
            if fixed:
                type_boxes.append((b, fixed))
        # 纯数字
        elif t.isdigit():
            num_boxes.append((b, t))
    
    # 车种类碎片
    for b, fixed in type_boxes:
        fragments.append({"text": fixed, "conf": b.conf, "pos": "type", "timestamp_sec": ts})
    
    # 数字类碎片：单帧内配对（paired）或单框（single）
    num_boxes.sort(key=lambda x: x[0].center_x)
    if len(num_boxes) >= 2:
        left_b, left_t = num_boxes[0]
        right_b, right_t = num_boxes[-1]
        merged_text = left_t + right_t
        merged_conf = (left_b.conf + right_b.conf) / 2
        fragments.append({"text": merged_text, "conf": merged_conf, "pos": "paired", "timestamp_sec": ts})
        # 中间框作为 single
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
        train_fragments=fragments
    )


# ============ 时序聚合（通用版） ============
def temporal_dedup(frames: List[FrameResult], gap_sec: float = 3.0,
                   vote_window_sec: float = 5.0,
                   extract_candidates_fn = lambda f: f.container_candidates) -> List[dict]:
    """
    通用时序聚合：对frames按extract_candidates_fn提取候选进行聚合
    """
    n = len(frames)
    
    # 前缀投票（只对集装箱有意义，货车传入空函数即可）
    corrected_candidates = []
    for i, frame in enumerate(frames):
        cands = extract_candidates_fn(frame)
        if not cands:
            corrected_candidates.append([])
            continue
        corrected_candidates.append(cands)
    
    # 时序聚合
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
                "confs": [best_conf],
            }
        elif frame.timestamp_sec - current_seq["end_time"] <= gap_sec:
            current_seq["end_time"] = frame.timestamp_sec
            current_seq["frames_count"] += 1
            current_seq["raw_ids"].append(best_cid)
            current_seq["confs"].append(best_conf)
            current_seq["id"] = Counter(current_seq["raw_ids"]).most_common(1)[0][0]
        else:
            sequences.append(current_seq)
            current_seq = {
                "id": best_cid,
                "start_time": frame.timestamp_sec,
                "end_time": frame.timestamp_sec,
                "frames_count": 1,
                "raw_ids": [best_cid],
                "confs": [best_conf],
            }
    
    if current_seq:
        sequences.append(current_seq)
    
    # 格式化输出
    for i, seq in enumerate(sequences, 1):
        seq["index"] = i
        seq["start_time"] = format_timestamp(seq["start_time"])
        seq["end_time"] = format_timestamp(seq["end_time"])
        seq["avg_conf"] = round(sum(seq["confs"]) / len(seq["confs"]), 3)
        del seq["confs"]
    
    return sequences


def merge_train_sequences(seqs: List[dict], gap_sec: float = 3.0) -> List[dict]:
    """合并时间重叠/相邻的车种和车号序列，如 C70E + 1755648 → C70E1755648"""
    if not seqs:
        return seqs
    
    def parse_time(t_str: str) -> float:
        mm, ss = t_str.split(':')
        return int(mm) * 60 + float(ss)
    
    def seq_type(sid: str) -> str:
        if re.match(r'^[A-Z]\d{2,4}[A-Z]?\d{6,8}$', sid):
            return 'complete'
        if re.match(r'^[A-Z]\d{2,4}[A-Z]?$', sid):
            return 'type_only'
        if re.match(r'^\d{6,8}$', sid):
            return 'num_only'
        return 'other'
    
    merged = [False] * len(seqs)
    result = []
    
    for i, seq in enumerate(seqs):
        if merged[i]:
            continue
        
        sid = seq["id"]
        stype = seq_type(sid)
        
        if stype == 'complete':
            result.append(seq)
            continue
        
        if stype not in ('type_only', 'num_only'):
            result.append(seq)
            continue
        
        start_sec = parse_time(seq["start_time"])
        end_sec = parse_time(seq["end_time"])
        
        best_partner = None
        best_partner_idx = -1
        
        for j, other in enumerate(seqs):
            if i == j or merged[j]:
                continue
            oid = other["id"]
            otype = seq_type(oid)
            # 类型互补才能合并
            if not ((stype == 'type_only' and otype == 'num_only') or (stype == 'num_only' and otype == 'type_only')):
                continue
            
            o_start = parse_time(other["start_time"])
            o_end = parse_time(other["end_time"])
            
            # 时间重叠或相邻（gap_sec内）
            if not (start_sec <= o_end + gap_sec and end_sec >= o_start - gap_sec):
                continue
            
            if best_partner is None or other["avg_conf"] > best_partner["avg_conf"]:
                best_partner = other
                best_partner_idx = j
        
        if best_partner:
            if stype == 'type_only':
                merged_id = sid + best_partner["id"]
            else:
                merged_id = best_partner["id"] + sid
            
            new_start = min(start_sec, parse_time(best_partner["start_time"]))
            new_end = max(end_sec, parse_time(best_partner["end_time"]))
            total_frames = seq["frames_count"] + best_partner["frames_count"]
            avg_conf = round((seq["avg_conf"] * seq["frames_count"] + best_partner["avg_conf"] * best_partner["frames_count"]) / total_frames, 3)
            
            merged_seq = {
                "index": len(result) + 1,
                "id": merged_id,
                "start_time": format_timestamp(new_start),
                "end_time": format_timestamp(new_end),
                "frames_count": total_frames,
                "avg_conf": avg_conf,
            }
            result.append(merged_seq)
            merged[i] = True
            merged[best_partner_idx] = True
        else:
            result.append(seq)
    
    # 重新编号
    for i, seq in enumerate(result, 1):
        seq["index"] = i
    return result


def assemble_train_fragments(frames: List[FrameResult], gap_sec: float = 3.0, complement_gap: float = 0.5) -> Tuple[List[dict], List[dict]]:
    """基于碎片的时序聚合：车种和车号分开输出，支持跨帧重叠拼接和互补拼接"""
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
    
    def vote_best(fragments: List[dict]) -> Tuple[str, float]:
        """投票：次数最多 → 长度最长 → 置信度最高"""
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
    
    def overlap_merge(a: str, b: str, min_overlap: int = 2) -> Optional[str]:
        for i in range(min(len(a), len(b)), min_overlap - 1, -1):
            if a[-i:] == b[:i]:
                return a + b[i:]
        return None
    
    def try_merge_singles(singles: List[dict], max_gap: float = 0.5) -> Tuple[Optional[str], float]:
        if not singles:
            return None, 0.0
        
        sorted_singles = sorted(singles, key=lambda f: f.get("timestamp_sec", 0))
        texts = [f["text"] for f in sorted_singles]
        timestamps = [f.get("timestamp_sec", 0) for f in sorted_singles]
        
        # 策略1：最大重叠拼接（链式，限制时间差）
        best = texts[0]
        best_conf = sorted_singles[0]["conf"]
        used = set()  # 只记录被额外消耗的碎片，起点(0)不占用
        
        for i in range(1, len(texts)):
            if i in used:
                continue
            if abs(timestamps[i] - timestamps[0]) > max_gap:
                continue
            merged = overlap_merge(best, texts[i], min_overlap=2)
            # 只有当合并后长度真正增加（不是伪重叠）才消耗碎片
            if merged and len(merged) > len(best) and len(merged) <= 8:
                best = merged
                best_conf = (best_conf + sorted_singles[i]["conf"]) / 2
                used.add(i)
        
        # 如果重叠拼接结果 >= 5位，优先返回
        if len(best) >= 5:
            return best, best_conf
        
        # 策略2：互补拼接（3位+4位=7位），时间差<max_gap
        threes = [(f, i) for i, f in enumerate(sorted_singles) if len(f["text"]) == 3 and i not in used]
        fours = [(f, i) for i, f in enumerate(sorted_singles) if len(f["text"]) == 4 and i not in used]
        for tf, ti in threes:
            for ff, fi in fours:
                if abs(tf.get("timestamp_sec", 0) - ff.get("timestamp_sec", 0)) <= max_gap:
                    combined = tf["text"] + ff["text"]
                    if len(combined) == 7:
                        conf = (tf["conf"] + ff["conf"]) / 2
                        return combined, conf
        
        # 策略3：任意两个碎片互补=7位
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
    
    type_seqs = []
    num_seqs = []
    
    for i, seq in enumerate(sequences, 1):
        type_id, type_conf = vote_best(seq["types"])
        
        num_id = None
        num_conf = 0.0
        
        # 优先用 paired（单帧内已配对）
        best_paired = None
        if seq["paired"]:
            best_paired = max(seq["paired"], key=lambda f: (len(f["text"]), f["conf"]))
            num_id = best_paired["text"]
            num_conf = best_paired["conf"]
        
        # 如果 paired 不够7位，尝试所有碎片拼接（包括paired之间）
        if not num_id or len(num_id) < 7:
            candidates = []
            # 所有长度>=2的paired碎片
            for f in seq["paired"]:
                if len(f["text"]) >= 2:
                    candidates.append(f)
            # 所有singles
            for f in seq["singles"]:
                candidates.append(f)
            
            if candidates:
                merged, merged_conf = try_merge_singles(candidates, complement_gap)
                if merged and (not num_id or len(merged) > len(num_id)):
                    num_id = merged
                    num_conf = merged_conf
        
        if type_id:
            type_seqs.append({
                "index": len(type_seqs) + 1,
                "id": type_id,
                "start_time": format_timestamp(seq["start_time"]),
                "end_time": format_timestamp(seq["end_time"]),
                "frames_count": seq["frames_count"],
                "avg_conf": round(type_conf, 3),
            })
        
        if num_id and len(num_id) >= 3:
            num_seqs.append({
                "index": len(num_seqs) + 1,
                "id": num_id,
                "start_time": format_timestamp(seq["start_time"]),
                "end_time": format_timestamp(seq["end_time"]),
                "frames_count": seq["frames_count"],
                "avg_conf": round(num_conf, 3),
            })
    
    return type_seqs, num_seqs


# ============ 主流程 ============
def process_video(video_path: str, output_dir: str, max_duration_sec: Optional[float] = None,
                  interval_sec: float = 0.5, gap_sec: float = 3.0, vote_window_sec: float = 5.0,
                  skip_extract: bool = False,
                  start_time_sec: Optional[float] = None, end_time_sec: Optional[float] = None,
                  use_gpu: bool = True):
    frame_list, fps = extract_frames(video_path, output_dir, max_duration_sec, interval_sec, skip_extract, start_time_sec, end_time_sec)

    # GPU 检测与自动回退
    gpu_available = False
    if use_gpu:
        if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            print(f"INFO: GPU detected — {paddle.device.cuda.device_count()} device(s), CUDA compiled: True")
            gpu_available = True
        else:
            print("WARN: GPU/CUDA not available, will use CPU")

    # 优先尝试 GPU，初始化失败则自动回退 CPU
    ocr_engine = None
    if gpu_available:
        try:
            print("INFO: Initializing PaddleOCR with GPU (lang=en)...")
            ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', show_log=False, use_gpu=True)
            print("INFO: PaddleOCR initialized on GPU.")
        except Exception as e:
            print(f"WARN: GPU initialization failed: {e}")
            print("WARN: Falling back to CPU...")
            gpu_available = False

    if ocr_engine is None:
        print("INFO: Initializing PaddleOCR with CPU (lang=en)...")
        ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', show_log=False, use_gpu=False)
        print("INFO: PaddleOCR initialized on CPU.")

    print(f"INFO: Processing {len(frame_list)} images with upper/lower split...")
    results = []
    for idx, (ts, fpath) in enumerate(frame_list):
        result = process_frame(ocr_engine, fpath, ts)
        results.append(result)

        if idx % 30 == 0 or idx == len(frame_list) - 1:
            cids = [c for c, _ in result.container_candidates]
            tids = [c for c, _ in result.train_candidates]
            print(f"  [{idx+1}/{len(frame_list)}] {os.path.basename(fpath)}: containers={cids}, trains={tids}")

    print(f"\nINFO: Running temporal dedup for containers...")
    container_seqs = temporal_dedup(results, gap_sec, vote_window_sec, lambda f: f.container_candidates)
    
    print(f"INFO: Assembling train fragments...")
    type_seqs, num_seqs = assemble_train_fragments(results, gap_sec)

    os.makedirs(output_dir, exist_ok=True)

    # 保存集装箱结果
    cpath = os.path.join(output_dir, "container_sequence.json")
    with open(cpath, 'w', encoding='utf-8') as f:
        json.dump(container_seqs, f, ensure_ascii=False, indent=2)
    print(f"\nINFO: Container sequences saved to {cpath}")

    # 保存铁路货车结果（车种和车号分开）
    tpath = os.path.join(output_dir, "train_type_sequence.json")
    with open(tpath, 'w', encoding='utf-8') as f:
        json.dump(type_seqs, f, ensure_ascii=False, indent=2)
    print(f"INFO: Train type sequences saved to {tpath}")
    
    npath = os.path.join(output_dir, "train_num_sequence.json")
    with open(npath, 'w', encoding='utf-8') as f:
        json.dump(num_seqs, f, ensure_ascii=False, indent=2)
    print(f"INFO: Train num sequences saved to {npath}")

    # 保存每帧详细结果
    frame_results = []
    for r in results:
        frame_results.append({
            "timestamp": format_timestamp(r.timestamp_sec),
            "timestamp_sec": r.timestamp_sec,
            "texts": [(t, round(c, 4)) for t, c in r.texts],
            "container_candidates": [c for c, _ in r.container_candidates],
            "train_candidates": [c for c, _ in r.train_candidates],
            "train_fragments": r.train_fragments,
        })
    fpath = os.path.join(output_dir, "frame_sequence.json")
    with open(fpath, 'w', encoding='utf-8') as f:
        json.dump(frame_results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"V6: 集装箱 {len(container_seqs)} 个, 铁路货车车种 {len(type_seqs)} 个, 车号 {len(num_seqs)} 个")
    print(f"上半区(集装箱): {len([f for f in results if f.container_candidates])} 帧有识别")
    print(f"下半区(货车): {len([f for f in results if f.train_candidates])} 帧有识别")
    print(f"{'='*60}")
    
    print(f"\n集装箱箱号段：")
    for s in container_seqs:
        print(f"  [{s['index']}] {s['id']} ({s['start_time']}~{s['end_time']}, {s['frames_count']}帧, conf={s['avg_conf']})")
    
    if type_seqs:
        print(f"\n铁路货车车种类：")
        for s in type_seqs:
            print(f"  [{s['index']}] {s['id']} ({s['start_time']}~{s['end_time']}, {s['frames_count']}帧, conf={s['avg_conf']})")
    
    if num_seqs:
        print(f"\n铁路货车车号：")
        for s in num_seqs:
            print(f"  [{s['index']}] {s['id']} ({s['start_time']}~{s['end_time']}, {s['frames_count']}帧, conf={s['avg_conf']})")
    
    if not type_seqs and not num_seqs:
        print(f"\n铁路货车：未识别到")


def main():
    parser = argparse.ArgumentParser(description="集装箱+铁路货车视频OCR - V6 (上下分区)")
    parser.add_argument("video", help="输入视频文件路径")
    parser.add_argument("-o", "--output", default="./output_video_paddle_v6", help="输出目录")
    parser.add_argument("-d", "--duration", type=float, default=None, help="最大处理时长(秒)")
    parser.add_argument("-i", "--interval", type=float, default=0.2, help="抽帧间隔(秒)")
    parser.add_argument("-g", "--gap", type=float, default=3.0, help="时序聚合间隔(秒)")
    parser.add_argument("-w", "--vote-window", type=float, default=5.0, help="前缀投票时间窗口(秒)")
    parser.add_argument("-s", "--skip-extract", action="store_true", help="跳过抽帧，复用已有帧")
    parser.add_argument("--start-time", type=float, default=None, help="开始时间(秒)，如 80")
    parser.add_argument("--end-time", type=float, default=None, help="结束时间(秒)，如 120")
    parser.add_argument("--cpu", action="store_true", help="强制使用CPU（默认优先尝试GPU）")
    args = parser.parse_args()

    process_video(args.video, args.output, args.duration, args.interval, args.gap, args.vote_window,
                  args.skip_extract, args.start_time, args.end_time, use_gpu=not args.cpu)


if __name__ == "__main__":
    main()
