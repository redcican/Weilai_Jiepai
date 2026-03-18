#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
表格OCR提取工具 - 精简版
从文件夹读取图片，提取表格数据，输出JSON
"""

import json
import os
import sys
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# 支持的图片格式
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


@dataclass
class OCRBox:
    """OCR识别结果"""
    box: List[int]
    text: str
    confidence: float = 0.0


def enhance_image_for_ocr(image_path: str) -> Any:
    """
    增强图像以提高OCR识别率
    
    通过提高对比度和锐度来帮助识别难以检测的字符
    返回增强后的图像（PIL Image 或 numpy array）
    """
    try:
        from PIL import Image, ImageEnhance, ImageFilter
        import numpy as np
        
        img = Image.open(image_path)
        
        # 转换为RGB（如果是RGBA或其他格式）
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # 增强对比度（帮助识别浅色字符）
        contrast_enhancer = ImageEnhance.Contrast(img)
        img = contrast_enhancer.enhance(1.3)  # 轻微增强对比度
        
        # 增强锐度（帮助识别模糊字符）
        sharpness_enhancer = ImageEnhance.Sharpness(img)
        img = sharpness_enhancer.enhance(1.5)  # 增强锐度
        
        # 转换为numpy array供PaddleOCR使用
        return np.array(img)
        
    except Exception as e:
        logger.debug(f"图像增强失败，使用原图: {e}")
        return None


class PaddleOCREngine:
    """PaddleOCR引擎 - 自动GPU/CPU选择"""
    
    def __init__(self, lang='ch', enhance_image: bool = False):
        self.lang = lang
        self._available = False
        self.ocr = None
        self.enhance_image = enhance_image  # 是否进行图像增强
        
        try:
            # 自动选择GPU或CPU
            import paddle
            if paddle.device.is_compiled_with_cuda():
                paddle.set_device('gpu')
                logger.info("使用GPU加速")
            else:
                paddle.set_device('cpu')
                logger.info("使用CPU")
            
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(use_textline_orientation=True, lang=lang)
            self._available = True
        except Exception as e:
            logger.warning(f"PaddleOCR初始化失败: {e}")
    
    @property
    def available(self) -> bool:
        return self._available
    
    def recognize(self, image_path: str) -> List[OCRBox]:
        """执行OCR识别，可选图像增强"""
        if not self._available:
            return []
        
        try:
            # 尝试图像增强
            input_image = image_path
            if self.enhance_image:
                enhanced = enhance_image_for_ocr(image_path)
                if enhanced is not None:
                    input_image = enhanced
            
            result = self.ocr.predict(input_image)
        except Exception as e:
            logger.warning(f"OCR识别失败: {e}")
            return []
        
        if not result or not isinstance(result, list) or len(result) == 0:
            return []
        
        results = []
        data = result[0]
        
        # PP-OCRv5 返回格式: {'rec_texts': [...], 'rec_scores': [...], 'dt_polys': [...]}
        if hasattr(data, 'get') or isinstance(data, dict):
            texts = data.get('rec_texts', [])
            scores = data.get('rec_scores', [])
            polys = data.get('dt_polys', [])
            
            for i, text in enumerate(texts):
                if not text or not str(text).strip():
                    continue
                
                conf = float(scores[i]) if i < len(scores) else 0.0
                
                # 转换polygon为边界框
                try:
                    if i < len(polys) and polys[i] is not None:
                        poly = polys[i]
                        x_coords = [float(p[0]) for p in poly]
                        y_coords = [float(p[1]) for p in poly]
                        box = [int(min(x_coords)), int(min(y_coords)),
                               int(max(x_coords)), int(max(y_coords))]
                    else:
                        box = [0, i * 30, 100, (i + 1) * 30]
                except (TypeError, ValueError, IndexError):
                    box = [0, i * 30, 100, (i + 1) * 30]
                
                results.append(OCRBox(box=box, text=str(text).strip(), confidence=conf))
        
        return results


def split_merged_text(text: str) -> List[str]:
    """
    智能分割OCR合并的文本
    
    处理情况:
    1. 中文分号"；"分隔的文本
    2. 冒号"："或":"后跟的位置信息
    3. 特定模式识别（如数字+汉字组合后跟汉字）
    
    例如: "二空2顶箱；部/局：马尾" -> ["二空2", "顶箱", "部/局", "马尾"]
    """
    if not text:
        return []
    
    text = text.strip()
    if not text:
        return []
    
    result = []
    
    # 步骤1: 先用中文分号分割
    parts = re.split(r'[；;]', text)
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # 步骤2: 处理冒号分隔的情况（如 "部/局：马尾"）
        if re.search(r'[：:]', part):
            sub_parts = re.split(r'[：:]', part, maxsplit=1)
            for sp in sub_parts:
                sp = sp.strip()
                if sp:
                    result.append(sp)
            continue
        
        # 步骤3: 检测合并的列模式
        # 模式: "XXX空X" + "XXX箱" (如 "二空2顶箱" -> "二空2", "顶箱")
        # 或 "敞二空2敞顶箱" -> "敞二空2", "敞顶箱"
        match = re.match(r'^(.+空\d+?)(.+箱.*)$', part)
        if match:
            result.append(match.group(1))
            # 递归处理剩余部分
            remaining = match.group(2)
            result.extend(split_merged_text(remaining))
            continue
        
        # 模式: "XXX空X" 后跟 "部/局..." (如 "通四空1部/局令133580")
        match = re.match(r'^(.+空\d+?)(部/局.*)$', part)
        if match:
            result.append(match.group(1))
            result.append(match.group(2))
            continue
        
        # 模式: 检测文本末尾可能是位置名（2-4个汉字）
        # 如 "部/局马尾" -> "部/局", "马尾"
        if len(part) > 4:
            # 检查是否有 "部/局" 后跟汉字（无分隔符）
            match = re.match(r'^(.*部/局)([一-龥]{2,4})$', part)
            if match:
                result.append(match.group(1))
                result.append(match.group(2))
                continue
        
        result.append(part)
    
    return result if result else [text]


def split_merged_numbers(text: str) -> List[str]:
    """
    分割被错误合并的数字
    如 "22.71.35" -> ["22.7", "1.3", "5"]
    如 "21C70" -> ["21", "C70"]
    """
    # 先处理序列号和车型合并的情况，如 "21C70" -> ["21", "C70"]
    match = re.match(r'^(\d{1,3})([A-Za-z].*)$', text)
    if match:
        return [match.group(1), match.group(2)]
    
    # 先按空格分割
    parts = text.split()
    result = []
    
    for part in parts:
        # 检查是否是合并的小数模式: 如 22.71.35 (三个数字用.连接)
        match = re.match(r'^(\d+\.\d)(\d\.\d+)(\d+)$', part)
        if match:
            result.extend([match.group(1), match.group(2), match.group(3)])
            continue
        
        # 模式: 22.71.3 -> 22.7, 1.3
        match = re.match(r'^(\d+\.\d)(\d\.\d+)$', part)
        if match:
            result.extend([match.group(1), match.group(2)])
            continue
        
        # 模式: 数字.数字.数字 (如 23.9 1.35 被合并为 23.91.35)
        match = re.match(r'^(\d+\.\d+)(\d\.\d+)$', part)
        if match and len(match.group(1)) <= 4:
            result.extend([match.group(1), match.group(2)])
            continue
        
        result.append(part)
    
    return result


def aggregate_to_rows(ocr_results: List[OCRBox], tolerance: int = None) -> List[List[str]]:
    """
    将OCR结果按行聚合
    使用自适应容差：基于文本高度自动计算合适的容差
    """
    if not ocr_results:
        return []
    
    def center_y(box):
        return (box.box[1] + box.box[3]) / 2
    
    def box_height(box):
        return box.box[3] - box.box[1]
    
    # 自适应容差：使用文本高度的中位数的一半
    if tolerance is None:
        heights = [box_height(r) for r in ocr_results if box_height(r) > 0]
        if heights:
            heights.sort()
            median_height = heights[len(heights) // 2]
            tolerance = max(10, int(median_height * 0.6))  # 高度的60%，最小10
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
            # 使用加权平均更新当前行的Y坐标
            current_y = sum(center_y(r) for r in current_row) / len(current_row)
        else:
            current_row.sort(key=lambda r: r.box[0])
            row_texts = []
            for r in current_row:
                # 先分割合并的文本列，再分割合并的数字
                for text_part in split_merged_text(r.text):
                    row_texts.extend(split_merged_numbers(text_part))
            rows.append(row_texts)
            current_row = [result]
            current_y = y
    
    if current_row:
        current_row.sort(key=lambda r: r.box[0])
        row_texts = []
        for r in current_row:
            # 先分割合并的文本列，再分割合并的数字
            for text_part in split_merged_text(r.text):
                row_texts.extend(split_merged_numbers(text_part))
        rows.append(row_texts)
    
    return rows


def is_potential_sequence_number(text: str) -> bool:
    """判断单个值是否可能是序列号"""
    text = text.strip()
    if not re.match(r'^\d{1,3}$', text):
        return False
    return True


def detect_sequence_column(rows: List[List[str]]) -> int:
    """
    自动检测哪一列是序列号列
    通过检查是否存在递增的数字序列来判断
    
    Returns:
        序列号列的索引，如果没有找到返回 -1
    """
    if not rows or len(rows) < 3:
        return -1
    
    # 检查前几列是否可能是序列号
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
        
        # 检查是否存在递增序列
        valid_values = [v for v in values if v is not None]
        if len(valid_values) < 3:
            continue
        
        # 检查是否大致递增（允许一些OCR错误）
        sorted_vals = sorted(valid_values)
        # 检查是否从小数字开始（1-10左右）且递增
        if sorted_vals[0] <= 10:
            # 计算有多少是连续的
            consecutive = 0
            for i in range(len(valid_values) - 1):
                if valid_values[i] is not None and valid_values[i + 1] is not None:
                    diff = valid_values[i + 1] - valid_values[i]
                    if 0 <= diff <= 2:  # 允许小的跳跃
                        consecutive += 1
            
            # 如果大部分是连续的，认为是序列号列
            if consecutive >= len(valid_values) * 0.5:
                return col_idx
    
    return -1


def is_metadata_item(row: List[str]) -> Tuple[bool, str, str]:
    """
    判断是否为元数据项（如 "股道：4"）
    返回 (是否元数据, key, value)
    """
    if len(row) == 1:
        text = row[0]
        # 匹配 "key：value" 或 "key:value" 格式
        match = re.match(r'^(.+)[：:](.+)$', text)
        if match:
            return True, match.group(1).strip(), match.group(2).strip()
    
    if len(row) == 2:
        # 检查是否是 key-value 对
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
    """
    自动检测哪一列是车型列
    车型列特征：包含字母+数字组合，如 C70, NX70AF, C64K
    
    Returns:
        车型列的索引，如果没有找到返回 -1
    """
    if not rows or len(rows) < 3:
        return -1
    
    max_cols = min(5, min(len(r) for r in rows if r))
    
    for col_idx in range(max_cols):
        vehicle_pattern_count = 0
        for row in rows:
            if col_idx < len(row):
                val = row[col_idx].strip()
                # 车型模式：字母+数字，或纯数字（2-3位）+可能的后缀
                if re.match(r'^[A-Za-z]+\d+', val) or re.match(r'^[A-Za-z]\d+[A-Za-z]*$', val):
                    vehicle_pattern_count += 1
                elif re.match(r'^\d{2,3}[A-Za-z]*$', val):  # 如 70, 64K
                    vehicle_pattern_count += 1
        
        # 如果大部分行匹配车型模式
        if vehicle_pattern_count >= len(rows) * 0.6:
            return col_idx
    
    return -1


def detect_vehicle_id_column(rows: List[List[str]]) -> int:
    """
    自动检测哪一列是车辆编号列
    车辆编号特征：6-8位数字
    
    Returns:
        车辆编号列的索引，如果没有找到返回 -1
    """
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
    """
    判断是否为有效数据行
    1. 至少有 min_cols 个字段
    2. 不是表头行
    3. 有实际内容
    """
    if not row or len(row) < min_cols:
        return False
    if is_header_row(row):
        return False
    
    # 有足够的实际内容
    content = ''.join(str(c).strip() for c in row)
    return len(content) > 5


def normalize_vehicle_type(cell: str) -> str:
    """
    规范化车型代码
    自动推断前缀：70 -> C70, 64K -> C64K
    """
    cell = cell.strip()
    
    # 已有字母前缀，不处理
    if re.match(r'^[A-Za-z]', cell):
        return cell
    
    # 纯数字开头，尝试添加C前缀（常见于C70系列）
    if re.match(r'^(70|64|62)\w*$', cell):
        return 'C' + cell
    
    return cell


def normalize_row(row: List[str], vehicle_col: int = -1) -> List[str]:
    """
    规范化数据行
    """
    if not row:
        return row
    
    result = []
    for i, cell in enumerate(row):
        cell = cell.strip()
        
        # 如果知道车型列位置，对该列进行规范化
        if vehicle_col >= 0 and i == vehicle_col:
            cell = normalize_vehicle_type(cell)
        # 如果不知道，对前几列尝试规范化
        elif vehicle_col < 0 and i <= 2:
            cell = normalize_vehicle_type(cell)
        
        result.append(cell)
    
    return result


def extract_track_number_from_first_row(rows: List[List[str]], seq_col: int) -> Tuple[Optional[str], List[List[str]]]:
    """
    从第一行提取股道号
    
    检测模式：
    1. 第一行比其他行多一列，且首列是数字（股道号）
    2. 第一行格式: [股道, 1, 数据...], 其他行格式: [2, 数据...]
    
    Returns:
        (股道号或None, 处理后的行列表)
    """
    if not rows or len(rows) < 2:
        return None, rows
    
    first_row = rows[0]
    if not first_row or len(first_row) < 2:
        return None, rows
    
    first_val = first_row[0].strip()
    
    # 检查第一个值是否是数字
    if not re.match(r'^\d{1,2}$', first_val):
        return None, rows
    
    first_num = int(first_val)
    
    # 模式1：检查第一行是否比后续行多一列
    other_row_lens = [len(r) for r in rows[1:5] if r]
    if other_row_lens:
        avg_len = sum(other_row_lens) / len(other_row_lens)
        if len(first_row) > avg_len + 0.5:  # 第一行明显更长
            # 检查第二个值是否像序列号 "1"
            if len(first_row) > 1:
                second_val = first_row[1].strip()
                if re.match(r'^[12]$', second_val):
                    track = first_val
                    new_first_row = first_row[1:]
                    return track, [new_first_row] + rows[1:]
    
    # 模式2：第一行格式 [大数字, 1, ...], 后续行 [2, ...], [3, ...]
    if len(first_row) > 1:
        second_val = first_row[1].strip()
        if re.match(r'^[12]$', second_val):  # 第二个值是 1 或 2
            second_num = int(second_val)
            
            # 检查后续行是否从序号2/3开始
            subsequent_firsts = []
            for row in rows[1:5]:
                if row and re.match(r'^\d{1,2}$', row[0].strip()):
                    subsequent_firsts.append(int(row[0].strip()))
            
            if subsequent_firsts:
                # 后续行的第一个数字应该是递增的序号
                expected_next = second_num + 1
                if subsequent_firsts[0] == expected_next:
                    # 第一行的第一个值是股道号
                    track = first_val
                    new_first_row = first_row[1:]
                    return track, [new_first_row] + rows[1:]
    
    # 模式3：第一行的第一个数字明显不符合递增序列
    seq_vals = []
    for row in rows[1:6]:
        if row and re.match(r'^\d{1,2}$', row[0].strip()):
            seq_vals.append(int(row[0].strip()))
    
    if len(seq_vals) >= 2:
        # 检查后续行是否形成递增序列
        is_sequential = all(seq_vals[i+1] - seq_vals[i] in [1, 2] for i in range(len(seq_vals)-1))
        if is_sequential:
            expected_start = seq_vals[0] - 1  # 期望的序列起始值
            # 如果第一行的第一个数大于序列，可能是股道号
            if first_num > seq_vals[-1] or first_num < expected_start - 1:
                track = first_val
                if len(first_row) > 1 and re.match(r'^[12]$', first_row[1].strip()):
                    new_first_row = first_row[1:]
                    return track, [new_first_row] + rows[1:]
    
    return None, rows


def fix_duplicate_cells(rows: List[List[str]]) -> List[List[str]]:
    """
    修复重复的单元格（如 "空空" -> "空"）
    
    处理行聚合时可能出现的串行问题
    """
    fixed_rows = []
    for row in rows:
        fixed_row = []
        for cell in row:
            # 检测重复字符模式（如 "空空"、"药村药村"）
            if len(cell) >= 2:
                half = len(cell) // 2
                if len(cell) % 2 == 0 and cell[:half] == cell[half:]:
                    # 重复模式，取一半
                    cell = cell[:half]
            fixed_row.append(cell)
        fixed_rows.append(fixed_row)
    return fixed_rows


def extract_table_data(image_path: str, engine: PaddleOCREngine, tolerance: int = None) -> Dict[str, Any]:
    """
    从单张图片提取表格数据
    
    自动检测表格结构：
    1. 检测序列号列（递增的数字序列）
    2. 如果没有序列号，自动生成
    3. 智能提取元数据（如股道号）
    4. 修复串行导致的重复单元格
    
    返回格式:
    {
        "message": "识别成功",
        "status": "success",
        "metadata": {"股道": "4", ...},
        "table data": [["1", "C70", "1664442", ...], ...]
    }
    """
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
        # 检查是否为元数据项
        is_meta, key, value = is_metadata_item(row)
        if is_meta:
            metadata[key] = value
            continue
        
        # 排除表头
        if is_header_row(row):
            continue
        
        # 保留可能的数据行
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
        # 有序列号列
        for row in candidate_rows:
            if is_valid_data_row(row):
                if seq_col < len(row) and is_potential_sequence_number(row[seq_col]):
                    data_rows.append(row)
        
        # 提取股道号
        track_num, data_rows = extract_track_number_from_first_row(data_rows, seq_col)
        if track_num:
            metadata["股道"] = track_num
            # 更新列索引
            if seq_col > 0:
                seq_col -= 1
                if vehicle_type_col > 0:
                    vehicle_type_col -= 1
        
        logger.debug(f"检测到序列号列，保留 {len(data_rows)} 行")
        
    else:
        # 无序列号列，使用车型/车辆编号检测
        for row in candidate_rows:
            if is_valid_data_row(row):
                has_vehicle = False
                for i in range(min(3, len(row))):
                    val = row[i].strip()
                    # 检查是否像车型
                    if re.match(r'^[A-Za-z]+\d+', val):
                        has_vehicle = True
                        break
                    # 检查是否像车辆编号
                    if re.match(r'^\d{6,8}$', val):
                        has_vehicle = True
                        break
                    # 检查是否是常见车型数字格式
                    if re.match(r'^\d{2,3}[A-Za-z]*$', val):
                        has_vehicle = True
                        break
                
                if has_vehicle:
                    data_rows.append(row)
        
        logger.debug(f"未检测到序列号列，使用车型检测，保留 {len(data_rows)} 行")
    
    # 规范化数据行（推断车型前缀）
    vehicle_col = vehicle_type_col if vehicle_type_col >= 0 else -1
    data_rows = [normalize_row(row, vehicle_col) for row in data_rows]
    
    # 如果没有序列号，自动生成
    if not has_sequence_numbers and data_rows:
        logger.debug(f"自动生成序列号")
        data_rows = [[str(i + 1)] + row for i, row in enumerate(data_rows)]
    
    # 修复串行导致的重复单元格（如 "空空" -> "空"）
    data_rows = fix_duplicate_cells(data_rows)
    
    return {
        "message": "识别成功",
        "status": "success",
        "metadata": metadata,
        "table data": data_rows
    }


def process_folder(folder_path: str, output_dir: str, recursive: bool = False, tolerance: int = None):
    """处理整个文件夹，每张图片生成一个JSON"""
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
    
    engine = PaddleOCREngine()
    if not engine.available:
        logger.error("OCR引擎不可用")
        return [], 0
    
    results = []
    total_records = 0
    
    for img_path in image_files:
        try:
            result = extract_table_data(str(img_path), engine, tolerance)
            
            output_path = output_dir / f"{img_path.stem}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            record_count = len(result.get('table data', []))
            total_records += record_count
            meta_count = len(result.get('metadata', {}))
            logger.info(f"处理 {img_path.name}: {record_count} 条数据, {meta_count} 条元数据")
            results.append(result)
            
        except Exception as e:
            logger.error(f"处理 {img_path} 失败: {e}")
            results.append({
                "message": f"处理失败: {e}",
                "status": "error",
                "metadata": {},
                "table data": []
            })
    
    return results, total_records


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='表格OCR提取工具')
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
