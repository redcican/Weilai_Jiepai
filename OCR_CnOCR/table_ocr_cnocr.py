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
    header_keywords = {'车号', '到站', '品名', '记事', '发站', '票据', '自重', '换长', '载重', '序号', '站存车打印'}
    row_text = ''.join(row)
    return any(kw in row_text for kw in header_keywords)


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
    """从单张图片提取表格数据"""
    ocr_results = engine.recognize(str(image_path))

    if not ocr_results:
        return {
            "message": f"图片 {Path(image_path).name} OCR识别失败",
            "status": "error",
            "metadata": {},
            "table data": []
        }

    # 聚合为行
    all_rows = aggregate_to_rows(ocr_results, tolerance)

    # 分离元数据和候选数据行
    metadata = {}
    candidate_rows = []

    for row in all_rows:
        is_meta, key, value = is_metadata_item(row)
        if is_meta and key and value:
            metadata[key] = value
            continue

        if is_header_row(row):
            continue

        if len(row) >= 3:
            candidate_rows.append(row)

    if not candidate_rows:
        return {
            "message": "识别成功",
            "status": "success",
            "metadata": metadata,
            "table data": []
        }

    # 自动检测表格结构
    seq_col = detect_sequence_column(candidate_rows)
    vehicle_type_col = detect_vehicle_type_column(candidate_rows)
    vehicle_id_col = detect_vehicle_id_column(candidate_rows)

    logger.debug(f"检测结果: seq_col={seq_col}, vehicle_type_col={vehicle_type_col}, vehicle_id_col={vehicle_id_col}")

    # 过滤有效数据行
    data_rows = []
    has_sequence_numbers = seq_col >= 0

    if has_sequence_numbers:
        for row in candidate_rows:
            if is_valid_data_row(row):
                if seq_col < len(row) and is_potential_sequence_number(row[seq_col]):
                    data_rows.append(row)

        # 提取股道号
        track_num, data_rows = extract_track_number_from_first_row(data_rows, seq_col)
        if track_num:
            metadata["股道"] = track_num
            if seq_col > 0:
                seq_col -= 1
                if vehicle_type_col > 0:
                    vehicle_type_col -= 1

        logger.debug(f"检测到序列号列，保留 {len(data_rows)} 行")

    else:
        for row in candidate_rows:
            if is_valid_data_row(row):
                has_vehicle = False
                for i in range(min(3, len(row))):
                    val = row[i].strip()
                    if re.match(r'^[A-Za-z]+\d+', val):
                        has_vehicle = True
                        break
                    if re.match(r'^\d{6,8}$', val):
                        has_vehicle = True
                        break
                    if re.match(r'^\d{2,3}[A-Za-z]*$', val):
                        has_vehicle = True
                        break

                if has_vehicle:
                    data_rows.append(row)

        logger.debug(f"未检测到序列号列，使用车型检测，保留 {len(data_rows)} 行")

    # 规范化数据行
    vehicle_col = vehicle_type_col if vehicle_type_col >= 0 else -1
    data_rows = [normalize_row(row, vehicle_col) for row in data_rows]

    # 如果没有序列号，自动生成
    if not has_sequence_numbers and data_rows:
        logger.debug(f"自动生成序列号")
        data_rows = [[str(i + 1)] + row for i, row in enumerate(data_rows)]

    return {
        "message": "识别成功",
        "status": "success",
        "metadata": metadata,
        "table data": data_rows
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
