#!/usr/bin/env python3
"""
批量处理列车图片（PaddleOCR 车种/车号识别）

输出格式：
  {
    "file": "xxx.jpg",
    "vehicleType": "C70E",
    "vehicleNumber": "1755648",
    "confidence": 0.9234
  }

不含空挡标记（type），只保留车种/车号/置信度。
"""

import os
import sys
import json
import argparse
from pathlib import Path

# 确保能导入同级目录的 train_id_ocr_paddle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_id_ocr_paddle import PaddleOCRProcessor, IMAGE_EXTENSIONS


def extract_best_result(result) -> dict:
    """从 ImageResult 中提取最佳车种、车号、置信度。"""
    # 最佳车种（取置信度最高的第一个）
    vehicle_type = ""
    if result.train_types:
        # train_types 是 List[Tuple[str, float]]，已按置信度排序
        vehicle_type = result.train_types[0][0]

    # 最佳车号（取最长的数字串，长度相同取置信度高的）
    vehicle_number = ""
    if result.train_numbers:
        # 按长度降序，再按置信度降序
        sorted_nums = sorted(
            result.train_numbers,
            key=lambda x: (len(x[0]), x[1]),
            reverse=True,
        )
        vehicle_number = sorted_nums[0][0]

    # 平均置信度（车种 + 车号）
    confs = []
    if result.train_types:
        confs.append(result.train_types[0][1])
    if result.train_numbers:
        confs.append(result.train_numbers[0][1])
    avg_conf = round(sum(confs) / len(confs), 4) if confs else 0.0

    return {
        "vehicleType": vehicle_type,
        "vehicleNumber": vehicle_number,
        "confidence": avg_conf,
    }


def main():
    parser = argparse.ArgumentParser(description="批量列车图片 OCR 识别")
    parser.add_argument("input", help="输入图片或文件夹路径")
    parser.add_argument(
        "-o", "--output", default="./output_batch", help="输出目录"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 收集图片路径
    if input_path.is_file():
        image_paths = [input_path]
    else:
        image_paths = sorted([
            p for p in input_path.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ])

    if not image_paths:
        print(f"未找到图片: {args.input}")
        return

    print(f"共找到 {len(image_paths)} 张图片")
    print("=" * 60)

    # 初始化处理器（只初始化一次，避免重复加载模型）
    processor = PaddleOCRProcessor()

    all_results = {}

    for i, img_path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}] {img_path.name}")

        result = processor.process(str(img_path))
        best = extract_best_result(result)

        result_dict = {
            "file": img_path.name,
            **best,
        }
        all_results[img_path.name] = result_dict

        print(f"  车种: {best['vehicleType'] or 'N/A'}")
        print(f"  车号: {best['vehicleNumber'] or 'N/A'}")
        print(f"  置信度: {best['confidence']}")

        # 每张图一个独立 JSON
        json_path = output_dir / f"{img_path.stem}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)

    # 汇总 JSON
    result_path = output_dir / "result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"处理完成: {len(image_paths)} 张图片")
    print(f"单图 JSON: {output_dir}/{{name}}.json")
    print(f"汇总 JSON: {result_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
