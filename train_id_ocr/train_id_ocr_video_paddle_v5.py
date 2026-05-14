#!/usr/bin/env python3
"""
集装箱箱号视频OCR识别 - PaddleOCR版本 V5
核心改进：
  1. 同一帧内多框空间邻近合并（单框/相邻2框/相邻3框）
  2. 数字区字母清洗（i→1, o→0, g→9等）
  3. 末尾校验码截断（固定取前6位数字）
  4. 前缀硬编码映射（TEJU→TBJU等）
  5. 单帧也保留（取消min_frames限制，保留时间窗口去重）
"""

import os
import re
import json
import argparse
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from collections import Counter, defaultdict
from datetime import timedelta
from difflib import SequenceMatcher

import cv2
from paddleocr import PaddleOCR


# ============ 硬编码前缀纠错映射表 ============
PREFIX_CORRECTION = {
    'TEJU': 'TBJU', 'TRJU': 'TBJU', 'TPJU': 'TBJU', 'T8JU': 'TBJU',
    'T3JU': 'TBJU', 'TCJU': 'TBJU', 'TBIU': 'TBJU', 'T0JU': 'TBJU',
    'IBCU': 'TBCU', 'TBCU': 'TBCU', '1BCU': 'TBCU',
    'TB1U': 'TBJU', 'TBJ0': 'TBJU', 'T9JU': 'TBJU',
    'IBJU': 'TBJU', 'I BJU': 'TBJU',
    # 可以持续补充...
}

# 前缀位置数字→字母映射（波纹钢板上常见混淆）
PREFIX_DIGIT_FIX = {
    '1': 'T',   # 1→T 竖线
    '3': 'B',   # 3→B 上半部分
    '8': 'B',   # 8→B 整体相似
    '0': 'O',   # 0→O
    '5': 'S',   # 5→S
    '2': 'Z',   # 2→Z
    '7': 'T',   # 7→T 顶部横线
    '4': 'A',   # 4→A
    '6': 'G',   # 6→G
    '9': 'G',   # 9→G
}

# 常见集装箱前缀白名单（用于辅助判断）
COMMON_PREFIXES = {
    'TBJU', 'TBCU', 'TRLU', 'TGHU', 'TCNU', 'EMCU', 'UESU',
    'HJCU', 'BSIU', 'APRU', 'MEDU', 'MSCW', 'MRSU', 'CSNU',
    'OOLU', 'CMAU', 'EGLV', 'HLCU', 'ONEU', 'YMLU', 'MAEU',
    'COSU', 'HJCU', 'UASC', 'KLINE'
}


@dataclass
class TextBox:
    """OCR识别出的一个文本框"""
    text: str
    conf: float
    center_x: float
    center_y: float
    width: float
    height: float


@dataclass
class ContainerResult:
    timestamp_sec: float
    texts: List[Tuple[str, float]] = field(default_factory=list)
    container_candidates: List[Tuple[str, float]] = field(default_factory=list)


def format_timestamp(sec: float) -> str:
    td = timedelta(seconds=sec)
    mm, ss = divmod(td.seconds, 60)
    return f"{mm:02d}:{ss:05.2f}"


def clean_digits(digits: str) -> str:
    """
    数字区字母清洗：把易混淆字母替换为数字
    """
    mapping = str.maketrans({
        'i': '1', 'I': '1', 'l': '1', 'L': '1',
        'o': '0', 'O': '0', 'Q': '0',
        'g': '9', 'q': '9', 'G': '6',  # G在首位时像6
        'b': '6', 'B': '8',
        's': '5', 'S': '5',
        'z': '2', 'Z': '2',
        'a': '4', 'A': '4',
    })
    return digits.translate(mapping)


def fix_prefix_digits(prefix: str) -> Optional[str]:
    """
    前缀数字容错：前4位中有数字时，尝试替换为字母
    如果替换后的前缀在白名单或纠错表中，返回修正后的前缀
    最多允许2个数字
    """
    if len(prefix) != 4:
        return None
    
    digit_positions = [i for i, c in enumerate(prefix) if c.isdigit()]
    if len(digit_positions) == 0:
        return prefix  # 没有数字，无需处理
    if len(digit_positions) > 2:
        return None  # 超过2个数字，不可信
    
    # 将数字替换为对应字母
    fixed = list(prefix)
    for pos in digit_positions:
        d = prefix[pos]
        if d in PREFIX_DIGIT_FIX:
            fixed[pos] = PREFIX_DIGIT_FIX[d]
        else:
            return None  # 未知数字映射
    
    result = ''.join(fixed)
    
    # 校验：替换后的前缀是否在白名单或硬编码纠错表中
    if result in COMMON_PREFIXES or result in PREFIX_CORRECTION.values():
        return result
    
    # 也接受硬编码纠错映射的键（原始错误前缀）经过数字修复后的结果
    if result in PREFIX_CORRECTION:
        return PREFIX_CORRECTION[result]
    
    # 如果结果和某个白名单前缀只差1个字符，也接受
    for white in COMMON_PREFIXES:
        diff = sum(1 for a, b in zip(result, white) if a != b)
        if diff <= 1:
            return white
    
    return None  # 不信任，返回None


def correct_prefix(prefix: str) -> str:
    """前缀纠错：硬编码映射 + 数字容错 + 白名单匹配"""
    # 1. 先尝试硬编码映射
    if prefix in PREFIX_CORRECTION:
        return PREFIX_CORRECTION[prefix]
    # 2. 如果是纯字母且已经在白名单，直接返回
    if prefix in COMMON_PREFIXES:
        return prefix
    return prefix


def extract_container_id(text: str, conf: float) -> Optional[Tuple[str, float]]:
    """
    从OCR文本中提取集装箱箱号
    返回 (cid, conf)，已做末尾截断、前缀纠错、前缀数字容错
    """
    # 1. 基础清理
    text = text.upper().replace(" ", "").replace("-", "").replace(".", "")
    
    # 2. 严格匹配：4字母+6/7位数字
    match = re.search(r'([A-Z]{4})(\d{6,7})', text)
    if match:
        prefix = match.group(1)
        digits = match.group(2)[:6]
        prefix = correct_prefix(prefix)
        return (f"{prefix}{digits}", conf)
    
    # 3. 宽松匹配：前缀允许数字（[A-Z0-9]{4}）+ 6/7位数字
    loose_match = re.search(r'([A-Z0-9]{4})(\d{6,7})', text)
    if loose_match:
        prefix = loose_match.group(1)
        digits = loose_match.group(2)[:6]
        # 尝试修复前缀中的数字
        fixed_prefix = fix_prefix_digits(prefix)
        if fixed_prefix:
            return (f"{fixed_prefix}{digits}", conf)
    
    # 4. 如果没匹配到，尝试找前缀位置然后清洗数字区
    # 先找可能的前缀位置（4个连续字符，允许字母和数字）
    for prefix_match in re.finditer(r'[A-Z0-9]{4}', text):
        prefix = prefix_match.group(0)
        prefix_pos = prefix_match.end()
        remaining = text[prefix_pos:]
        # 清洗数字区的字母为数字
        cleaned = clean_digits(remaining)
        digit_match = re.search(r'(\d{6,7})', cleaned)
        if digit_match:
            digits = digit_match.group(1)[:6]
            # 先修复前缀数字
            fixed_prefix = fix_prefix_digits(prefix)
            if fixed_prefix:
                fixed_prefix = correct_prefix(fixed_prefix)
                return (f"{fixed_prefix}{digits}", conf)
            # 前缀已经是纯字母
            if prefix.isalpha():
                prefix = correct_prefix(prefix)
                return (f"{prefix}{digits}", conf)
    
    return None


def extract_frames(video_path: str, output_dir: str, max_duration_sec: Optional[float] = None, interval_sec: float = 0.5, skip_extract: bool = False) -> Tuple[List[Tuple[float, str]], float]:
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

    # 如果跳过抽帧且目录已有帧，直接复用
    if skip_extract:
        existing = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
        if existing:
            frame_paths = []
            for fname in existing:
                # 从文件名解析时间戳: frame_0066.55.jpg -> 66.55
                ts_match = re.search(r'frame_([\d.]+)\.jpg', fname)
                ts = float(ts_match.group(1)) if ts_match else 0.0
                frame_paths.append((ts, os.path.join(frames_dir, fname)))
            frame_paths.sort(key=lambda x: x[0])
            print(f"INFO: Reusing {len(frame_paths)} existing frames (skipped extract)")
            return frame_paths, fps
        print("WARN: --skip-extract specified but no frames found, will extract.")

    # 清空并重新抽帧
    for f in os.listdir(frames_dir):
        if f.endswith('.jpg'):
            os.remove(os.path.join(frames_dir, f))

    cap = cv2.VideoCapture(video_path)
    frame_paths = []
    count = 0
    max_frames = int((max_duration_sec or duration) * fps) if max_duration_sec else total_frames

    while True:
        ret, frame = cap.read()
        if not ret or count >= max_frames:
            break
        if count % step == 0:
            ts = count / fps
            fname = f"frame_{ts:07.2f}.jpg"
            fpath = os.path.join(frames_dir, fname)
            cv2.imwrite(fpath, frame)
            frame_paths.append((ts, fpath))
        count += 1

    cap.release()
    print(f"INFO: Extracted {len(frame_paths)} frames (fps={fps:.2f}, step={step} frames)")
    return frame_paths, fps


def parse_ocr_boxes(result) -> List[TextBox]:
    """将PaddleOCR结果解析为TextBox列表"""
    boxes = []
    if result and result[0]:
        for line in result[0]:
            if not line:
                continue
            # line 格式: [ [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], (text, conf) ]
            coords = line[0]
            text, conf = line[1]
            
            # 计算中心点和宽高
            xs = [p[0] for p in coords]
            ys = [p[1] for p in coords]
            center_x = sum(xs) / 4
            center_y = sum(ys) / 4
            width = max(xs) - min(xs)
            height = max(ys) - min(ys)
            
            boxes.append(TextBox(text, conf, center_x, center_y, width, height))
    
    # 按中心x坐标从左到右排序
    boxes.sort(key=lambda b: b.center_x)
    return boxes


def merge_boxes(boxes: List[TextBox]) -> List[Tuple[str, float]]:
    """
    多框合并策略：
    1. 每个单框尝试匹配
    2. 相邻2框合并尝试匹配
    3. 相邻3框合并尝试匹配
    只合并 y 坐标相近的框（避免跨行）
    """
    candidates = []
    n = len(boxes)
    
    def is_y_close(b1: TextBox, b2: TextBox, threshold_ratio: float = 0.6) -> bool:
        """判断两个框是否在垂直方向上接近"""
        y_threshold = max(b1.height, b2.height) * threshold_ratio
        return abs(b1.center_y - b2.center_y) < y_threshold
    
    def try_match(text: str, conf: float) -> Optional[Tuple[str, float]]:
        return extract_container_id(text, conf)
    
    # 1. 单框匹配
    for b in boxes:
        cid = try_match(b.text, b.conf)
        if cid:
            candidates.append(cid)
    
    # 2. 相邻2框合并
    for i in range(n - 1):
        b1, b2 = boxes[i], boxes[i + 1]
        if not is_y_close(b1, b2):
            continue
        # 文本拼接（不加空格，箱号是连续的）
        merged_text = b1.text + b2.text
        merged_conf = (b1.conf + b2.conf) / 2
        cid = try_match(merged_text, merged_conf)
        if cid:
            candidates.append(cid)
        # 也尝试加空格版本（有些框之间本来有空格）
        merged_text_space = b1.text + " " + b2.text
        cid2 = try_match(merged_text_space, merged_conf)
        if cid2 and cid2 not in candidates:
            candidates.append(cid2)
    
    # 3. 相邻3框合并
    for i in range(n - 2):
        b1, b2, b3 = boxes[i], boxes[i + 1], boxes[i + 2]
        if not (is_y_close(b1, b2) and is_y_close(b2, b3)):
            continue
        merged_text = b1.text + b2.text + b3.text
        merged_conf = (b1.conf + b2.conf + b3.conf) / 3
        cid = try_match(merged_text, merged_conf)
        if cid:
            candidates.append(cid)
    
    # 去重：同一帧内相同箱号只保留置信度最高的
    best = {}
    for cid, conf in candidates:
        if cid not in best or conf > best[cid]:
            best[cid] = conf
    
    return [(cid, conf) for cid, conf in best.items()]


def process_frame(ocr_engine, img_path: str, ts: float) -> ContainerResult:
    """处理单帧：全面遍历所有识别框进行匹配和合并"""
    img = cv2.imread(img_path)
    if img is None:
        return ContainerResult(timestamp_sec=ts)

    result = ocr_engine.ocr(img, cls=True)
    boxes = parse_ocr_boxes(result)
    
    # 收集所有原始文本
    texts = [(b.text, b.conf) for b in boxes]
    
    # 多框合并提取箱号
    container_ids = merge_boxes(boxes)
    
    return ContainerResult(timestamp_sec=ts, texts=texts, container_candidates=container_ids)


def prefix_similarity(p1: str, p2: str) -> float:
    if len(p1) != 4 or len(p2) != 4:
        return 0.0
    diff = sum(1 for a, b in zip(p1, p2) if a != b)
    if diff == 0:
        return 1.0
    if diff == 1:
        return 0.8
    if diff == 2:
        return 0.4
    return 0.0


def digits_similarity(d1: str, d2: str) -> float:
    if len(d1) != 6 or len(d2) != 6:
        return SequenceMatcher(None, d1, d2).ratio()
    diff = sum(1 for a, b in zip(d1, d2) if a != b)
    if diff == 0:
        return 1.0
    if diff == 1:
        return 0.85
    if diff == 2:
        return 0.6
    return 0.0


def is_same_container(cid1: str, cid2: str) -> bool:
    if len(cid1) != 10 or len(cid2) != 10:
        return False
    p1, d1 = cid1[:4], cid1[4:]
    p2, d2 = cid2[:4], cid2[4:]
    if d1 == d2 and prefix_similarity(p1, p2) >= 0.8:
        return True
    if prefix_similarity(p1, p2) == 1.0 and digits_similarity(d1, d2) >= 0.85:
        return True
    return False


def vote_prefix_in_window(candidates: List[Tuple[str, float]]) -> Dict[str, str]:
    """时间窗口内前缀投票"""
    if not candidates:
        return {}
    
    digit_groups = defaultdict(list)
    for cid, conf in candidates:
        digits = cid[4:]
        digit_groups[digits].append((cid, conf))
    
    correction_map = {}
    
    for digits, group in digit_groups.items():
        prefix_counter = Counter()
        prefix_conf = defaultdict(list)
        
        for cid, conf in group:
            prefix = cid[:4]
            prefix_counter[prefix] += 1
            prefix_conf[prefix].append(conf)
        
        if len(prefix_counter) == 1:
            for cid, conf in group:
                correction_map[cid] = cid
        else:
            best_prefix = prefix_counter.most_common(1)[0][0]
            avg_conf = {p: sum(prefix_conf[p])/len(prefix_conf[p]) for p in prefix_conf}
            print(f"    [Vote] digits={digits}: candidates={dict(prefix_counter)}, avg_conf={avg_conf}")
            print(f"    [Vote] -> choose prefix '{best_prefix}'")
            
            for cid, conf in group:
                corrected = best_prefix + digits
                correction_map[cid] = corrected
    
    return correction_map


def temporal_dedup_with_voting(frames: List[ContainerResult], gap_sec: float = 3.0, vote_window_sec: float = 5.0) -> List[dict]:
    """
    时序聚合 + 时间窗口前缀投票
    V5改动：取消min_frames限制，单帧匹配也保留
    """
    n = len(frames)
    corrected_candidates = []
    
    for i, frame in enumerate(frames):
        if not frame.container_candidates:
            corrected_candidates.append([])
            continue
        
        window_start = frame.timestamp_sec - vote_window_sec / 2
        window_end = frame.timestamp_sec + vote_window_sec / 2
        
        window_candidates = []
        for j in range(n):
            if window_start <= frames[j].timestamp_sec <= window_end:
                window_candidates.extend(frames[j].container_candidates)
        
        correction_map = vote_prefix_in_window(window_candidates)
        
        corrected = []
        for cid, conf in frame.container_candidates:
            new_cid = correction_map.get(cid, cid)
            corrected.append((new_cid, conf))
        
        corrected_candidates.append(corrected)
    
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
                "container_id": best_cid,
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
            current_seq["container_id"] = Counter(current_seq["raw_ids"]).most_common(1)[0][0]
        else:
            sequences.append(current_seq)
            current_seq = {
                "container_id": best_cid,
                "start_time": frame.timestamp_sec,
                "end_time": frame.timestamp_sec,
                "frames_count": 1,
                "raw_ids": [best_cid],
                "confs": [best_conf],
            }
    
    if current_seq:
        sequences.append(current_seq)
    
    # 合并数字相同但前缀不同的相邻序列
    merged = []
    for seq in sequences:
        if not merged:
            merged.append(seq)
            continue
        
        last = merged[-1]
        if seq["start_time"] - last["end_time"] <= gap_sec:
            d1, d2 = last["container_id"][4:], seq["container_id"][4:]
            p1, p2 = last["container_id"][:4], seq["container_id"][:4]
            if d1 == d2 and prefix_similarity(p1, p2) >= 0.8:
                last["end_time"] = seq["end_time"]
                last["frames_count"] += seq["frames_count"]
                last["raw_ids"].extend(seq["raw_ids"])
                last["confs"].extend(seq["confs"])
                last["container_id"] = Counter(last["raw_ids"]).most_common(1)[0][0]
                continue
        
        merged.append(seq)
    
    # V5: 不再过滤min_frames，所有序列都保留
    for i, seq in enumerate(merged, 1):
        seq["index"] = i
        seq["start_time"] = format_timestamp(seq["start_time"])
        seq["end_time"] = format_timestamp(seq["end_time"])
        seq["avg_conf"] = round(sum(seq["confs"]) / len(seq["confs"]), 3)
        del seq["confs"]
    
    return merged


def process_video(video_path: str, output_dir: str, max_duration_sec: Optional[float] = None,
                  interval_sec: float = 0.5, gap_sec: float = 3.0, vote_window_sec: float = 5.0,
                  skip_extract: bool = False):
    frame_list, fps = extract_frames(video_path, output_dir, max_duration_sec, interval_sec, skip_extract)

    print("INFO: Initializing PaddleOCR (lang=en)...")
    ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    print("INFO: PaddleOCR initialized.")

    print(f"INFO: Processing {len(frame_list)} images with multi-box merging...")
    results = []
    for idx, (ts, fpath) in enumerate(frame_list):
        result = process_frame(ocr_engine, fpath, ts)
        results.append(result)

        if idx % 30 == 0 or idx == len(frame_list) - 1:
            cids = [c for c, _ in result.container_candidates]
            raw_texts = [t for t, _ in result.texts]
            print(f"  [{idx+1}/{len(frame_list)}] {os.path.basename(fpath)}: raw={raw_texts} -> containers={cids}")

    print(f"\nINFO: Running temporal dedup with prefix voting (window={vote_window_sec}s, no min_frames filter)...")
    sequences = temporal_dedup_with_voting(results, gap_sec, vote_window_sec)

    os.makedirs(output_dir, exist_ok=True)

    container_path = os.path.join(output_dir, "container_sequence.json")
    with open(container_path, 'w', encoding='utf-8') as f:
        json.dump(sequences, f, ensure_ascii=False, indent=2)
    print(f"\nINFO: Container sequences saved to {container_path}")

    frame_results = []
    for r in results:
        frame_results.append({
            "timestamp": format_timestamp(r.timestamp_sec),
            "timestamp_sec": r.timestamp_sec,
            "texts": [(t, round(c, 4)) for t, c in r.texts],
            "container_candidates": [c for c, _ in r.container_candidates],
        })
    frame_path = os.path.join(output_dir, "frame_sequence.json")
    with open(frame_path, 'w', encoding='utf-8') as f:
        json.dump(frame_results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"V5配置: 抽帧间隔={interval_sec}s, 聚合间隔={gap_sec}s, 投票窗口={vote_window_sec}s")
    print(f"多框合并: 单框+相邻2框+相邻3框 | 末尾截断6位 | 数字字母清洗 | 前缀硬映射")
    print(f"识别到 {len(sequences)} 个集装箱箱号段（含单帧）：")
    for seq in sequences:
        print(f"  [{seq['index']}] {seq['container_id']} ({seq['start_time']} ~ {seq['end_time']}, {seq['frames_count']}帧, avg_conf={seq['avg_conf']})")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="集装箱箱号视频OCR - PaddleOCR V5 (多框合并+单帧保留)")
    parser.add_argument("video", help="输入视频文件路径")
    parser.add_argument("-o", "--output", default="./output_video_paddle_v5", help="输出目录")
    parser.add_argument("-d", "--duration", type=float, default=None, help="最大处理时长(秒)")
    parser.add_argument("-i", "--interval", type=float, default=0.5, help="抽帧间隔(秒)")
    parser.add_argument("-g", "--gap", type=float, default=3.0, help="时序聚合间隔(秒)")
    parser.add_argument("-w", "--vote-window", type=float, default=5.0, help="前缀投票时间窗口(秒)")
    parser.add_argument("-s", "--skip-extract", action="store_true", help="跳过抽帧，复用已有帧")
    args = parser.parse_args()

    process_video(args.video, args.output, args.duration, args.interval, args.gap, args.vote_window, args.skip_extract)


if __name__ == "__main__":
    main()
