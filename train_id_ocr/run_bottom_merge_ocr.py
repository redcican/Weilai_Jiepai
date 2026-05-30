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

class FlatcarBottomProcessor:
    """平板车底部区域单图处理器（基于同行框拼接）。"""

    def __init__(self, use_gpu: bool = True):
        self.ocr = None
        self.gap_detector = None

        # 优先尝试 GPU，失败则自动回退 CPU
        if use_gpu:
            try:
                self.ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False, use_gpu=True)
                print("INFO: Flatcar PaddleOCR initialized on GPU (lang=ch)")
            except Exception as e:
                print(f"WARNING: Flatcar PaddleOCR GPU init failed: {e}, falling back to CPU")

        if self.ocr is None:
            try:
                self.ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False, use_gpu=False)
                print("INFO: Flatcar PaddleOCR initialized on CPU (lang=ch)")
            except Exception as e:
                print(f"ERROR: Flatcar PaddleOCR init failed: {e}")

        # 初始化空挡检测器
        try:
            from flatcar_gap_detector import FlatcarGapDetector
            self.gap_detector = FlatcarGapDetector()
            print("INFO: FlatcarGapDetector initialized")
        except Exception as e:
            print(f"WARNING: FlatcarGapDetector init failed: {e}")

    @property
    def available(self) -> bool:
        return self.ocr is not None

    def process_bytes(self, image_bytes: bytes) -> dict:
        """Process JPEG/PNG image bytes and return flatcar type + number.

        Returns:
            {
                "vehicleType": "X70",
                "vehicleNumber": "5240903",
                "confidence": 0.92,
            }
        """
        if self.ocr is None:
            return {"vehicleType": "", "vehicleNumber": "", "confidence": 0.0}

        img_array = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return {"vehicleType": "", "vehicleNumber": "", "confidence": 0.0}

        return self._process_img(img)

    def process_raw_bytes(
        self,
        image_bytes: bytes,
        pixel_type: int,
        width: int,
        height: int,
    ) -> dict:
        """Process raw camera pixel bytes (Bayer/Mono) and return flatcar results.

        Uses decode_raw_image() to convert raw industrial camera data to BGR.
        """
        if self.ocr is None:
            return {"vehicleType": "", "vehicleNumber": "", "confidence": 0.0}

        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dms_api'))
        from app.train_id.utils import decode_raw_image

        img = decode_raw_image(image_bytes, pixel_type, width, height)
        if img is None:
            print("ERROR: FlatcarBottomProcessor failed to decode raw image bytes")
            return {"vehicleType": "", "vehicleNumber": "", "confidence": 0.0}

        return self._process_img(img)

    def _process_img(self, img: np.ndarray) -> dict:
        """Process a decoded BGR image and return flatcar recognition results."""
        # ========== 空挡检测 ==========
        is_gap = False
        gap_score = 0
        gap_features = {}
        if self.gap_detector is not None:
            try:
                is_gap, gap_conf, gap_features = self.gap_detector.detect(img)
                gap_score = int(gap_conf * 8)
                feat_str = (
                    f"mb={gap_features.get('mean_brightness', 0):.0f} "
                    f"std={gap_features.get('std_brightness', 0):.0f} "
                    f"edge={gap_features.get('edge_score', 0):.0f} "
                    f"pv={gap_features.get('profile_variance', 0):.0f} "
                    f"pm={gap_features.get('profile_min', 0):.0f} "
                    f"asy={gap_features.get('asymmetry', 0):.2f} "
                    f"cv={gap_features.get('col_variance', 0):.0f} "
                    f"cr={gap_features.get('col_max', 0)-gap_features.get('col_min', 0):.0f}"
                )
                print(f"  [Flatcar空挡检测] {'空挡' if is_gap else '正常'} 得分={gap_score}/8 特征=[{feat_str}]")
            except Exception as e:
                print(f"  [Flatcar空挡检测] 检测异常: {e}")

        h, w = img.shape[:2]
        y1 = int(h * BOTTOM_Y_START)
        y2 = int(h * BOTTOM_Y_END)
        bottom_roi = img[y1:y2, :]

        enhanced = preprocess_for_dark(bottom_roi)
        ocr_result = self.ocr.ocr(enhanced, cls=True)

        # 收集原始检测框
        raw_texts = []
        if ocr_result and ocr_result[0]:
            for line in ocr_result[0]:
                if not line:
                    continue
                raw_texts.append((line[1][0], line[1][1], line[0]))

        # 同行拼接
        merged_rows = merge_boxes_by_row(raw_texts, y_tolerance=50)

        # 提取候选
        type_candidates, num_candidates = extract_candidates(merged_rows)

        # 选最佳车型
        vehicle_type = ""
        type_conf = 0.0
        if type_candidates:
            cnt = Counter([t for t, _ in type_candidates])
            vehicle_type = cnt.most_common(1)[0][0]
            type_conf = sum(c for t, c in type_candidates if t == vehicle_type) / len([1 for t, _ in type_candidates if t == vehicle_type])

        # 选最佳车号（优先7位，否则最长）
        vehicle_number = ""
        num_conf = 0.0
        if num_candidates:
            by_len = defaultdict(list)
            for n, c in num_candidates:
                by_len[len(n)].append((n, c))
            if 7 in by_len:
                candidates = by_len[7]
                weighted = defaultdict(float)
                for n, c in candidates:
                    weighted[n] += c
                vehicle_number = max(weighted.keys(), key=lambda k: weighted[k])
                num_conf = weighted[vehicle_number] / len(candidates)
            else:
                max_len = max(len(n) for n, _ in num_candidates)
                candidates = [(n, c) for n, c in num_candidates if len(n) == max_len]
                cnt = Counter([n for n, _ in candidates])
                vehicle_number = cnt.most_common(1)[0][0]
                num_conf = sum(c for n, c in candidates if n == vehicle_number) / len([1 for n, _ in candidates if n == vehicle_number])

        confs = []
        if vehicle_type:
            confs.append(type_conf)
        if vehicle_number:
            confs.append(num_conf)
        avg_conf = round(sum(confs) / len(confs), 4) if confs else 0.0

        return {
            "type": "########" if is_gap else "",
            "vehicleType": vehicle_type,
            "vehicleNumber": vehicle_number,
            "confidence": avg_conf,
            "gapScore": gap_score,
            "gapFeatures": {k: round(float(v), 2) for k, v in gap_features.items()} if gap_features else {},
        }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    viz_dir = os.path.join(OUTPUT_DIR, "_viz")
    os.makedirs(viz_dir, exist_ok=True)

    print("[1/3] 抽帧...")
    frames, fps = extract_frames(VIDEO_PATH, INTERVAL_SEC)
    print(f"    共 {len(frames)} 帧")

    print("[2/3] 初始化 FlatcarBottomProcessor...")
    processor = FlatcarBottomProcessor()
    print("    完成")

    print("[3/3] 底部区域 OCR + 同行拼接 + 可视化...")
    results = []

    for i, (t, frame) in enumerate(frames):
        import io
        _, buf = cv2.imencode('.jpg', frame)
        result = processor.process_bytes(buf.tobytes())

        results.append({
            "timestamp_sec": t,
            "timestamp": format_ts(t),
            "result": result,
        })

        if i % 50 == 0 or i == len(frames) - 1:
            print(f"    [{i+1}/{len(frames)}] type={result['vehicleType']} num={result['vehicleNumber']} conf={result['confidence']}")

    with open(os.path.join(OUTPUT_DIR, "frame_results.json"), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "="*60)
    print(f"共处理 {len(results)} 帧")
    print(f"结果保存在: {OUTPUT_DIR}/frame_results.json")
    print("="*60)


if __name__ == "__main__":
    main()
