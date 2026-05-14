#!/usr/bin/env python3
"""
底部区域 OCR + 同行框拼接：解决长数字串被拆成多个框的问题
"""

import os
import re
import sys
import cv2
import json
import numpy as np
from datetime import timedelta
from collections import Counter, defaultdict

from paddleocr import PaddleOCR

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.stdout.reconfigure(encoding='utf-8')

VIDEO_PATH = '车厢号识别/MV-CS050-60GC (DA6979473)（IP地址192.168.11.117）/MV-CS050-60GC (DA6979473)/Video_20260420103956155.avi'
OUTPUT_DIR = 'output_viz_bottom_merge'
INTERVAL_SEC = 0.05

BOTTOM_Y_START = 0.75
BOTTOM_Y_END = 1.0

FLATCAR_TYPES = {'X70', 'X6K', 'X2K', 'X2H', 'X4K', 'NX70', 'NX17', 'NX17B', 'C70', 'C70E', 'C80'}
FLATCAR_CORRECTION = {
    'X7O': 'X70', 'X7D': 'X70', 'X7B': 'X70', 'X70E': 'X70',
    '70': 'X70', 'X': 'X70',
    'C7O': 'C70', 'C7D': 'C70', 'C7B': 'C70', 'CO': 'C70',
    'C70B': 'C70', 'C70H': 'C70', 'C7OE': 'C70E',
}

def fix_flatcar_type(text):
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
            if c == 'O' and i >= 1: fixed[i] = '0'
            if c == 'I' and i >= 1: fixed[i] = '1'
            if c == 'D' and i >= 1: fixed[i] = '0'
        result = ''.join(fixed)
        if result in FLATCAR_TYPES: return result
    if len(text) >= 3 and text[0].isalpha():
        first = text[0]
        digits = ''.join(c for c in text[1:] if c.isdigit())
        letters = ''.join(c for c in text[1:] if c.isalpha())
        if len(digits) >= 2:
            candidate = f"{first}{digits[-2:]}{letters[:1]}"
            if candidate in FLATCAR_TYPES: return candidate
    return None

def preprocess_for_dark(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

def format_ts(sec):
    td = timedelta(seconds=sec)
    return str(td)[:-3] if '.' in str(td) else str(td) + '.000'

def extract_frames(video_path, interval_sec):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        t = idx / fps
        if idx % max(1, int(fps * interval_sec)) == 0:
            frames.append((t, frame))
        idx += 1
    cap.release()
    return frames, fps

def merge_boxes_by_row(texts, y_tolerance=40):
    """
    将检测框按行分组，同一行内按X坐标排序后拼接文本
    texts: list of (text, conf, box)
    """
    if not texts:
        return []

    # box格式: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    def center_y(box):
        return sum(p[1] for p in box) / 4
    def center_x(box):
        return sum(p[0] for p in box) / 4

    # 按Y坐标排序
    sorted_texts = sorted(texts, key=lambda x: center_y(x[2]))

    rows = []
    current_row = [sorted_texts[0]]
    current_y = center_y(sorted_texts[0][2])

    for item in sorted_texts[1:]:
        cy = center_y(item[2])
        if abs(cy - current_y) <= y_tolerance:
            current_row.append(item)
        else:
            # 按X坐标排序，拼接
            current_row.sort(key=lambda x: center_x(x[2]))
            merged_text = ''.join(t for t, _, _ in current_row)
            avg_conf = sum(c for _, c, _ in current_row) / len(current_row)
            # 用整行的包围框
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

    # 最后一行
    if current_row:
        current_row.sort(key=lambda x: center_x(x[2]))
        merged_text = ''.join(t for t, _, _ in current_row)
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

def extract_candidates(merged_rows):
    """从拼接后的行中提取车型和车号候选"""
    type_candidates = []
    num_candidates = []

    for merged_text, conf, box, raw_items in merged_rows:
        upper = merged_text.upper().replace(" ", "").replace("-", "")
        
        # 路徽误识别修正：如果第一个字母后面跟着已知车型，去掉该字母
        # （路徽常被识别为 G、Q、D 等单字母）
        for ft in sorted(FLATCAR_TYPES, key=len, reverse=True):
            if len(upper) > len(ft) + 1 and upper[1:].startswith(ft):
                upper = upper[1:]
                break

        # 提取车型
        fixed_type = fix_flatcar_type(upper)
        if fixed_type:
            type_candidates.append((fixed_type, conf))

        # 过滤包含 -/. 的参数文本，但如果已识别到车型且车型在文本开头，
        # 说明 -/. 可能只是车型和车号之间的分隔，不过滤
        if ('.' in merged_text or '-' in merged_text):
            if not (fixed_type and upper.startswith(fixed_type)):
                continue  # 无车型或车型不在开头，跳过整行

        # 构造用于数字提取的文本：去掉已识别的车型前缀，避免 X70+5240903 → 70524090 被过滤
        text_for_num = upper
        if fixed_type and text_for_num.startswith(fixed_type):
            text_for_num = text_for_num[len(fixed_type):]

        # 提取车号（3-8位数字），从去掉车型后的文本中提取
        for m in re.finditer(r'(\d{3,8})', text_for_num):
            num = m.group(1)
            # 纯日期格式等可通过其他规则过滤，不再使用705黑名单
            if len(num) >= 3:
                num_candidates.append((num, conf))

    return type_candidates, num_candidates

def merge_numbers_by_overlap(nums_with_conf, target_len=7, min_overlap=2):
    """
    从多帧数字候选中通过重叠拼接找出目标长度的数字。
    支持2-3帧拼接。
    nums_with_conf: list of (num_str, conf)
    返回 (num_id, num_conf) 或 (None, 0.0)
    """
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
    
    # 两两拼接
    for i in range(len(nums_with_conf)):
        for j in range(len(nums_with_conf)):
            if i == j:
                continue
            a, ca = nums_with_conf[i]
            b, cb = nums_with_conf[j]
            try_merge(a, ca, b, cb)
            try_merge(b, cb, a, ca)
    
    # 三帧拼接：用两两结果和第三帧继续拼
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

def draw_merged_results(img, merged_rows):
    viz = img.copy()
    h, w = viz.shape[:2]

    for merged_text, conf, box, raw_items in merged_rows:
        pts = np.array(box, np.int32).reshape((-1, 1, 2))

        # 判断这一行是什么类型
        upper = merged_text.upper().replace(" ", "").replace("-", "")
        fixed_type = fix_flatcar_type(upper)
        has_num = bool(re.search(r'\d{3,8}', upper))

        if fixed_type:
            color = (0, 255, 0)  # 绿色=车型
        elif has_num:
            color = (255, 0, 0)  # 蓝色=车号
        else:
            color = (0, 255, 255)  # 黄色=其他

        cv2.polylines(viz, [pts], True, color, 2)

        # 画原始小框（虚线效果用细线）
        for rt, rc, rb in raw_items:
            rpts = np.array(rb, np.int32).reshape((-1, 1, 2))
            cv2.polylines(viz, [rpts], True, (255, 255, 255), 1)

        label = f"{merged_text} ({conf:.2f})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top_y = min(p[1] for p in box)
        top_x = min(p[0] for p in box)
        bg_y1 = max(0, int(top_y) - th - 6)
        bg_x2 = min(w, int(top_x) + tw + 4)
        bg_y2 = int(top_y)
        cv2.rectangle(viz, (int(top_x), bg_y1), (bg_x2, bg_y2), color, -1)
        cv2.putText(viz, label, (int(top_x) + 2, bg_y2 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return viz

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    viz_dir = os.path.join(OUTPUT_DIR, "_viz")
    os.makedirs(viz_dir, exist_ok=True)

    print("[1/3] 抽帧...")
    frames, fps = extract_frames(VIDEO_PATH, INTERVAL_SEC)
    print(f"    共 {len(frames)} 帧")

    print("[2/3] 初始化 PaddleOCR (ch, CPU)...")
    ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False, use_gpu=False)
    print("    完成")

    print("[3/3] 底部区域 OCR + 同行拼接 + 可视化...")
    results = []

    for i, (t, frame) in enumerate(frames):
        h, w = frame.shape[:2]
        y1 = int(h * BOTTOM_Y_START)
        y2 = int(h * BOTTOM_Y_END)
        bottom_roi = frame[y1:y2, :]

        enhanced = preprocess_for_dark(bottom_roi)
        ocr_result = ocr.ocr(enhanced, cls=True)

        # 收集原始检测框
        raw_texts = []
        if ocr_result and ocr_result[0]:
            for line in ocr_result[0]:
                if not line: continue
                raw_texts.append((line[1][0], line[1][1], line[0]))

        # 同行拼接
        merged_rows = merge_boxes_by_row(raw_texts, y_tolerance=50)

        # 提取候选
        type_candidates, num_candidates = extract_candidates(merged_rows)

        # 画可视化
        viz = draw_merged_results(bottom_roi, merged_rows)

        # 保存
        out_path = os.path.join(viz_dir, f"viz_{t:07.2f}.jpg")
        cv2.imwrite(out_path, viz)

        # 记录
        text_details = []
        for mt, mc, mb, raw in merged_rows:
            adjusted_box = [[p[0], p[1] + y1] for p in mb]
            text_details.append({
                "merged_text": mt,
                "conf": round(mc, 4),
                "box": adjusted_box,
                "raw_boxes": len(raw),
            })

        results.append({
            "timestamp_sec": t,
            "timestamp": format_ts(t),
            "merged_rows": [(mt, round(mc, 4)) for mt, mc, _, _ in merged_rows],
            "type_candidates": type_candidates,
            "num_candidates": num_candidates,
        })

        if i % 50 == 0 or i == len(frames) - 1:
            preview = ', '.join([mt for mt, _, _, _ in merged_rows[:2]]) if merged_rows else '(none)'
            print(f"    [{i+1}/{len(frames)}] rows={len(merged_rows)} types={len(type_candidates)} nums={len(num_candidates)} | {preview}")

    # 时序聚合：车型和车号分别独立聚合
    print("[4/4] 时序聚合...")
    gap_sec = 0.15
    
    # 车型序列：只看有 type_candidates 的帧
    type_sequences = []
    current = None
    for r in results:
        if not r["type_candidates"]: continue
        if current is None or r["timestamp_sec"] - current[-1]["timestamp_sec"] > gap_sec:
            if current: type_sequences.append(current)
            current = [r]
        else:
            current.append(r)
    if current: type_sequences.append(current)
    
    # 车号序列：有3位及以上数字的帧都纳入（支持跨帧拼接）
    num_sequences = []
    current = None
    for r in results:
        has_digit = any(len(n) >= 3 for n, c in r["num_candidates"])
        if not has_digit: continue
        if current is None or r["timestamp_sec"] - current[-1]["timestamp_sec"] > gap_sec:
            if current: num_sequences.append(current)
            current = [r]
        else:
            current.append(r)
    if current: num_sequences.append(current)

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
            "start_sec": seq[0]["timestamp_sec"],
            "end_sec": seq[-1]["timestamp_sec"],
            "frames_count": len(seq),
            "avg_conf": round(type_conf, 4),
            "id": type_id,
        })

    num_seqs = []
    for seq_idx, seq in enumerate(num_sequences, 1):
        nums = [(n, c) for r in seq for n, c in r["num_candidates"] if len(n) >= 3]
        num_id = None; num_conf = 0.0
        by_len = defaultdict(list)
        for n, c in nums: by_len[len(n)].append((n, c))
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
            # 尝试跨帧重叠拼接
            num_id, num_conf = merge_numbers_by_overlap(nums, target_len=7, min_overlap=2)
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
                "start_sec": seq[0]["timestamp_sec"],
                "end_sec": seq[-1]["timestamp_sec"],
                "frames_count": len(seq),
                "avg_conf": round(num_conf, 4),
                "id": num_id,
            })

    # 过滤：只保留7位数字车号，置信度≥0.90（705黑名单已删除，依靠车型剥离机制）
    filtered_nums = [s for s in num_seqs 
                     if len(s['id']) == 7 
                     and s['avg_conf'] >= 0.90]

    # 合并车型和车号为单个JSON
    merged_results = []
    used_num_indices = set()
    
    for t in type_seqs:
        t_start, t_end = t['start_sec'], t['end_sec']
        best_match = None
        best_overlap = -1
        for n in filtered_nums:
            n_start, n_end = n['start_sec'], n['end_sec']
            # 单帧也算：只要时间窗口有交集（含边界接触）即可配对
            if max(t_start, n_start) <= min(t_end, n_end):
                overlap = min(t_end, n_end) - max(t_start, n_start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = n
        
        entry = {
            "index": len(merged_results) + 1,
            "type": t['id'],
            "num": best_match['id'] if best_match else None,
            "frames": best_match['frames_count'] if best_match else t['frames_count'],
            "avg_conf": round(best_match['avg_conf'], 4) if best_match else round(t['avg_conf'], 4),
        }
        if best_match:
            used_num_indices.add(best_match['index'])
        merged_results.append(entry)
    
    # 未配对的车号也加上
    for n in filtered_nums:
        if n['index'] not in used_num_indices:
            merged_results.append({
                "index": len(merged_results) + 1,
                "type": None,
                "num": n['id'],
                "frames": n['frames_count'],
                "avg_conf": round(n['avg_conf'], 4),
            })

    with open(os.path.join(OUTPUT_DIR, "sequence.json"), 'w', encoding='utf-8') as f:
        json.dump(merged_results, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUTPUT_DIR, "frame_sequence.json"), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "="*60)
    print(f"车型: {len(type_seqs)} 个, 车号(原始): {len(num_seqs)} 个, 车号(过滤后): {len(filtered_nums)} 个")
    print(f"合并结果: {len(merged_results)} 条")
    for s in merged_results:
        type_str = s['type'] if s['type'] else '(无车型)'
        num_str = s['num'] if s['num'] else '(无车号)'
        print(f"  [{s['index']}] {type_str} {num_str} conf={s['avg_conf']} frames={s['frames']}")
    print(f"可视化保存在: {viz_dir}/")
    print(f"合并JSON: {OUTPUT_DIR}/sequence.json")
    print("="*60)

if __name__ == "__main__":
    main()
