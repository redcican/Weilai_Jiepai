#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate Train ID OCR results against ground truth.

Reads per-image JSON files produced by train_id_ocr.py and compares them
to a ground truth file.  Prints a summary table and writes detailed
results to evaluation.json.

Usage:
    python evaluate.py -p ./output -g ../进站_OCR/ground_truth.text
    python evaluate.py -p ./output -g ../进站_OCR/ground_truth.text -o evaluation.json
"""

import json
import re
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any


# ---------------------------------------------------------------------------
# Ground truth parsing
# ---------------------------------------------------------------------------

def parse_ground_truth(gt_path: str) -> Dict[str, dict]:
    """
    Parse ground truth file.

    Expected format (repeating blocks):
        1.bmp:
        C64K
        49  31846

    Returns:
        {filename: {"vehicle_type": ..., "vehicle_number": ...}}
    """
    gt: Dict[str, dict] = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        match = re.match(r"^(\d+\.bmp):\s*$", line)
        if match:
            filename = match.group(1)
            vtype = lines[i + 1].strip() if i + 1 < len(lines) else ""
            vnum = lines[i + 2].strip() if i + 2 < len(lines) else ""
            gt[filename] = {"vehicle_type": vtype, "vehicle_number": vnum}
            i += 3
        else:
            i += 1
    return gt


# ---------------------------------------------------------------------------
# Load predictions from JSON files
# ---------------------------------------------------------------------------

def load_predictions(pred_dir: str) -> Dict[str, dict]:
    """
    Load per-image prediction JSON files from a directory.

    Each file is expected to contain:
        {"file": "1.bmp", "vehicle_type": "C64K", "vehicle_number": "49 31846", ...}

    Returns:
        {filename: {"vehicle_type": ..., "vehicle_number": ...}}
    """
    preds: Dict[str, dict] = {}
    pred_path = Path(pred_dir)

    for json_file in sorted(pred_path.glob("*.json")):
        if json_file.name == "evaluation.json":
            continue
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        filename = data.get("file", json_file.stem + ".bmp")
        preds[filename] = {
            "vehicle_type": data.get("vehicle_type", ""),
            "vehicle_number": data.get("vehicle_number", ""),
        }
    return preds


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Normalize for comparison: uppercase, collapse whitespace."""
    return re.sub(r"\s+", " ", text.strip().upper())


def _digits_only(text: str) -> str:
    """Extract only digit characters."""
    return re.sub(r"\D", "", text)


def _levenshtein(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if not s2:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        cur_row = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            cur_row.append(min(
                cur_row[j] + 1,
                prev_row[j + 1] + 1,
                prev_row[j] + cost,
            ))
        prev_row = cur_row
    return prev_row[-1]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    predictions: Dict[str, dict],
    ground_truth: Dict[str, dict],
) -> Dict[str, Any]:
    """
    Compare OCR predictions against ground truth.

    Returns:
        {"summary": {...}, "details": [...]}
    """
    details: List[dict] = []
    type_exact = 0
    num_exact = 0
    num_digit_exact = 0
    total = 0
    total_type_edit = 0
    total_num_edit = 0

    for filename in sorted(ground_truth.keys()):
        gt = ground_truth[filename]
        pred = predictions.get(filename, {"vehicle_type": "", "vehicle_number": ""})

        gt_type = _normalize(gt["vehicle_type"])
        pred_type = _normalize(pred["vehicle_type"])
        gt_num = _normalize(gt["vehicle_number"])
        pred_num = _normalize(pred["vehicle_number"])

        type_match = gt_type == pred_type
        num_match = gt_num == pred_num
        digit_match = _digits_only(gt_num) == _digits_only(pred_num)

        type_edit = _levenshtein(gt_type, pred_type)
        num_edit = _levenshtein(_digits_only(gt_num), _digits_only(pred_num))

        if type_match:
            type_exact += 1
        if num_match:
            num_exact += 1
        if digit_match:
            num_digit_exact += 1
        total += 1
        total_type_edit += type_edit
        total_num_edit += num_edit

        status = "OK" if type_match and digit_match else "MISMATCH"
        details.append({
            "file": filename,
            "status": status,
            "gt_type": gt["vehicle_type"],
            "pred_type": pred["vehicle_type"],
            "type_match": type_match,
            "gt_number": gt["vehicle_number"],
            "pred_number": pred["vehicle_number"],
            "number_match": num_match,
            "digit_match": digit_match,
            "type_edit_dist": type_edit,
            "number_edit_dist": num_edit,
        })

    summary = {
        "total_images": total,
        "type_exact_match": type_exact,
        "type_accuracy": round(type_exact / total, 4) if total else 0,
        "number_exact_match": num_exact,
        "number_accuracy": round(num_exact / total, 4) if total else 0,
        "number_digit_exact_match": num_digit_exact,
        "number_digit_accuracy": round(num_digit_exact / total, 4) if total else 0,
        "avg_type_edit_distance": round(total_type_edit / total, 4) if total else 0,
        "avg_number_edit_distance": round(total_num_edit / total, 4) if total else 0,
    }

    return {"summary": summary, "details": details}


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_report(result: Dict[str, Any]) -> None:
    """Print evaluation summary and per-image details to stdout."""
    s = result["summary"]

    print(f"\n{'=' * 60}")
    print(f"Evaluation Summary ({s['total_images']} images)")
    print(f"{'=' * 60}")
    print(f"  Vehicle Type   — exact match: {s['type_exact_match']}/{s['total_images']} "
          f"({s['type_accuracy']:.1%})")
    print(f"  Vehicle Number — exact match: {s['number_exact_match']}/{s['total_images']} "
          f"({s['number_accuracy']:.1%})")
    print(f"  Vehicle Number — digit match: {s['number_digit_exact_match']}/{s['total_images']} "
          f"({s['number_digit_accuracy']:.1%})")
    print(f"  Avg edit distance — type: {s['avg_type_edit_distance']:.2f}, "
          f"number: {s['avg_number_edit_distance']:.2f}")

    print(f"\n{'─' * 70}")
    print(f"{'File':<10} {'Status':<10} {'GT Type':<8} {'Pred Type':<10} "
          f"{'GT Number':<14} {'Pred Number':<14}")
    print(f"{'─' * 70}")
    for d in result["details"]:
        mark = "OK" if d["status"] == "OK" else "MISS"
        print(
            f"{d['file']:<10} {mark:<10} {d['gt_type']:<8} {d['pred_type']:<10} "
            f"{d['gt_number']:<14} {d['pred_number']:<14}"
        )
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Train ID OCR results against ground truth"
    )
    parser.add_argument(
        "-p", "--predictions", required=True,
        help="Directory containing per-image JSON results from train_id_ocr.py",
    )
    parser.add_argument(
        "-g", "--ground-truth", required=True,
        help="Path to ground truth file (e.g. ground_truth.text)",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Path to write evaluation JSON (default: <predictions>/evaluation.json)",
    )
    args = parser.parse_args()

    pred_dir = Path(args.predictions)
    if not pred_dir.is_dir():
        print(f"Error: not a directory: {args.predictions}", file=sys.stderr)
        sys.exit(1)

    gt_path = Path(args.ground_truth)
    if not gt_path.is_file():
        print(f"Error: file not found: {args.ground_truth}", file=sys.stderr)
        sys.exit(1)

    ground_truth = parse_ground_truth(str(gt_path))
    predictions = load_predictions(str(pred_dir))

    if not ground_truth:
        print("Error: no entries parsed from ground truth file", file=sys.stderr)
        sys.exit(1)

    missing = set(ground_truth.keys()) - set(predictions.keys())
    if missing:
        print(f"Warning: {len(missing)} ground truth files have no predictions: "
              f"{sorted(missing)}", file=sys.stderr)

    result = evaluate(predictions, ground_truth)
    print_report(result)

    output_path = args.output or str(pred_dir / "evaluation.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Evaluation saved to {output_path}")


if __name__ == "__main__":
    main()
