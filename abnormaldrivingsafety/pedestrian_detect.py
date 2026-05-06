"""
Railway Driving Safety — Pedestrian Anomaly Detection
======================================================
Detects pedestrians in freight carriage top-down camera images using YOLOv8.

If any person is detected → abnormal (异常)
No person detected      → normal (正常)

For large images with small targets, a sliding-window tiling strategy is used
to improve detection of small/partially-visible persons. All parameters are
loaded from config.json.

Usage:
    python pedestrian_detect.py ../abnormaldrivingsafety_figures
    python pedestrian_detect.py ../abnormaldrivingsafety_figures --debug
    python pedestrian_detect.py ../abnormaldrivingsafety_figures --json
    python pedestrian_detect.py image.bmp
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


CONFIG_FILE = Path(__file__).parent / "config.json"


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: list[float]  # [x1, y1, x2, y2] in original image coords

    def to_dict(self) -> dict:
        return {
            "class": self.class_name,
            "confidence": round(self.confidence, 3),
            "bbox": [round(v, 1) for v in self.bbox],
        }


def load_config(config_path: Path | None = None) -> dict:
    path = config_path or CONFIG_FILE
    if not path.exists():
        print(f"Error: config not found at {path}", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def load_model(config: dict) -> YOLO:
    return YOLO(config["model"])


def get_target_class_ids(model: YOLO, target_names: list[str]) -> list[int]:
    name_to_id = {name: idx for idx, name in model.names.items()}
    ids = []
    for name in target_names:
        if name in name_to_id:
            ids.append(name_to_id[name])
        else:
            print(f"Warning: class '{name}' not in model. "
                  f"Available: {list(name_to_id.keys())[:10]}...", file=sys.stderr)
    return ids


# ---------------------------------------------------------------------------
# Sliding-window tiling
# ---------------------------------------------------------------------------

def generate_tiles(
    img_h: int, img_w: int, tile_size: int, overlap_ratio: float
) -> list[tuple[int, int, int, int]]:
    """Generate (x1, y1, x2, y2) tile coordinates with overlap."""
    step = int(tile_size * (1 - overlap_ratio))
    tiles = []
    for y in range(0, img_h, step):
        for x in range(0, img_w, step):
            x2 = min(x + tile_size, img_w)
            y2 = min(y + tile_size, img_h)
            x1 = max(0, x2 - tile_size)
            y1 = max(0, y2 - tile_size)
            tiles.append((x1, y1, x2, y2))
    # Deduplicate (edge tiles may repeat)
    return list(dict.fromkeys(tiles))


def nms(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    """Non-maximum suppression across detections."""
    if not detections:
        return []

    boxes = np.array([d.bbox for d in detections])
    scores = np.array([d.confidence for d in detections])

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-6)

        remaining = np.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]

    return [detections[i] for i in keep]


def extract_boxes(
    result, model: YOLO, offset_x: int = 0, offset_y: int = 0
) -> list[Detection]:
    """Extract Detection objects from a YOLO result, applying coordinate offset."""
    detections = []
    boxes = result.boxes
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        conf = float(boxes.conf[i].item())
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        detections.append(Detection(
            class_name=model.names[cls_id],
            confidence=conf,
            bbox=[x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y],
        ))
    return detections


def has_safety_gear(
    img: np.ndarray, bbox: list[float], min_pixels: int = 100
) -> bool:
    """Check if the bounding box contains safety gear colors (hard hat / vest).

    Workers wear orange hard hats and fluorescent vests.  Counting pixels in
    those HSV ranges filters tiling false-positives (mechanical parts, shadows).
    """
    h, w = img.shape[:2]
    x1 = max(0, int(bbox[0]))
    y1 = max(0, int(bbox[1]))
    x2 = min(w, int(bbox[2]))
    y2 = min(h, int(bbox[3]))
    if x2 <= x1 or y2 <= y1:
        return False

    roi = img[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Orange hard hat: H 5-22, S > 100, V > 100
    mask_hat = cv2.inRange(hsv, (5, 100, 100), (22, 255, 255))
    # Fluorescent vest: H 25-85, S > 60, V > 80
    mask_vest = cv2.inRange(hsv, (25, 60, 80), (85, 255, 255))

    gear_pixels = int(cv2.countNonZero(mask_hat) + cv2.countNonZero(mask_vest))
    return gear_pixels >= min_pixels


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def detect_pedestrians(
    image_path: str,
    model: YOLO,
    config: dict,
    target_class_ids: list[int],
    debug: bool = False,
) -> dict:
    """Run pedestrian detection on a single image.

    Strategy:
      1. Full-image detection at primary_imgsz
      2. If no target found and tiling is enabled, run sliding-window tiling
      3. Merge results with NMS
    """
    img = cv2.imread(image_path)
    if img is None:
        return {"status": "error", "reason": f"cannot read: {image_path}", "detections": []}

    img_h, img_w = img.shape[:2]
    all_detections: list[Detection] = []

    # --- Pass 1: full image ---
    results = model.predict(
        source=img,
        conf=config["confidence_threshold"],
        iou=config["iou_threshold"],
        imgsz=config["primary_imgsz"],
        device=config["device"],
        classes=target_class_ids,
        verbose=False,
    )
    for r in results:
        all_detections.extend(extract_boxes(r, model))

    # --- Pass 2: tiling (if enabled and no detection yet) ---
    tiling_cfg = config.get("tiling", {})
    if not all_detections and tiling_cfg.get("enabled", False):
        tile_size = tiling_cfg["tile_size"]
        overlap = tiling_cfg["overlap_ratio"]
        tile_conf = tiling_cfg["confidence_threshold"]
        tile_imgsz = tiling_cfg["imgsz"]
        nms_iou = tiling_cfg["nms_iou_threshold"]

        tiles = generate_tiles(img_h, img_w, tile_size, overlap)

        for x1, y1, x2, y2 in tiles:
            tile = img[y1:y2, x1:x2]
            results = model.predict(
                source=tile,
                conf=tile_conf,
                iou=config["iou_threshold"],
                imgsz=tile_imgsz,
                device=config["device"],
                classes=target_class_ids,
                verbose=False,
            )
            for r in results:
                all_detections.extend(extract_boxes(r, model, offset_x=x1, offset_y=y1))

        all_detections = nms(all_detections, nms_iou)
        # Filter tiling detections by safety gear color verification
        all_detections = [d for d in all_detections if has_safety_gear(img, d.bbox)]

    status = "abnormal" if all_detections else "normal"

    result_dict = {
        "status": status,
        "detections": [d.to_dict() for d in all_detections],
    }

    if debug:
        dbg = _draw_debug(img, all_detections, status)
        result_dict["debug_img"] = dbg

    return result_dict


def _draw_debug(
    img: np.ndarray, detections: list[Detection], status: str
) -> np.ndarray:
    dbg = img.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        color = (0, 0, 255)
        cv2.rectangle(dbg, (x1, y1), (x2, y2), color, 3)
        label = f"{det.class_name} {det.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(dbg, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
        cv2.putText(dbg, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    h_img = dbg.shape[0]
    status_color = (0, 0, 255) if status == "abnormal" else (0, 200, 0)
    cv2.putText(dbg, f"STATUS: {status.upper()}", (10, h_img - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
    return dbg


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def load_ground_truth(folder: Path) -> dict[str, str]:
    """Load from 文本标识.txt. '行人' in label → abnormal."""
    gt = {}
    gt_file = folder / "文本标识.txt"
    if not gt_file.exists():
        return gt

    for line in gt_file.read_text(encoding="utf-8").strip().splitlines():
        line = line.strip()
        if not line:
            continue
        sep = "：" if "：" in line else (":" if ":" in line else None)
        if not sep:
            continue
        fname_part, label_part = line.split(sep, 1)
        stem = fname_part.strip().replace("png", "").replace("bmp", "")
        gt[stem] = "abnormal" if "行人" in label_part else "normal"

    return gt


def match_ground_truth(image_name: str, gt: dict[str, str]) -> str | None:
    stem = Path(image_name).stem
    if stem in gt:
        return gt[stem]
    return gt.get(stem.lstrip("0") or "0")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_folder(
    folder: Path,
    model: YOLO,
    config: dict,
    target_class_ids: list[int],
    debug: bool = False,
) -> dict:
    gt = load_ground_truth(folder)
    extensions = config["image_extensions"]

    images = []
    for ext in extensions:
        images.extend(folder.glob(f"*{ext}"))
    images = sorted(images)

    if not images:
        return {"folder": str(folder), "error": "no images found", "results": []}

    results = []
    correct, total = 0, 0

    debug_dir = None
    if debug:
        debug_dir = Path(__file__).parent / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        res = detect_pedestrians(
            str(img_path), model, config, target_class_ids, debug=debug
        )
        predicted = res["status"]
        expected = match_ground_truth(img_path.name, gt)
        n_persons = len(res["detections"])

        if expected is not None:
            total += 1
            match = predicted == expected
            if match:
                correct += 1
            status = "OK" if match else "FAIL"
        else:
            status = "NO_GT"

        results.append({
            "file": img_path.name,
            "predicted": predicted,
            "expected": expected,
            "status": status,
            "persons_detected": n_persons,
            "detections": res["detections"],
        })

        if debug and "debug_img" in res:
            cv2.imwrite(str(debug_dir / img_path.stem) + ".jpg", res["debug_img"])

    accuracy = correct / total if total > 0 else 0.0
    return {
        "folder": folder.name,
        "accuracy": round(accuracy, 3),
        "correct": correct,
        "total": total,
        "results": results,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pedestrian anomaly detection for railway driving safety"
    )
    parser.add_argument(
        "paths", nargs="*",
        help="Image file(s) or folder(s) to process.",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help=f"Config JSON path (default: {CONFIG_FILE})",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Save annotated debug images to ./debug/",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output JSON instead of table",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    model = load_model(config)
    target_class_ids = get_target_class_ids(model, config["target_classes"])

    if not target_class_ids:
        print("Error: no valid target classes found.", file=sys.stderr)
        sys.exit(1)

    targets: list[tuple[str, Path]] = []
    if args.paths:
        for p in args.paths:
            pp = Path(p)
            if pp.is_file():
                targets.append(("file", pp))
            elif pp.is_dir():
                targets.append(("dir", pp))
            else:
                print(f"Warning: {p} not found", file=sys.stderr)
    else:
        parent = Path(__file__).parent.parent
        for d in sorted(parent.iterdir()):
            if (d.is_dir()
                    and "abnormaldrivingsafety" in d.name.lower()
                    and d.name != "abnormaldrivingsafety"):
                targets.append(("dir", d))

    if not targets:
        print("No images or folders found. Provide a path argument.", file=sys.stderr)
        sys.exit(1)

    total_correct, total_tested = 0, 0

    for kind, path in targets:
        if kind == "file":
            res = detect_pedestrians(
                str(path), model, config, target_class_ids, debug=args.debug
            )
            if args.debug and "debug_img" in res:
                debug_dir = Path(__file__).parent / "debug"
                debug_dir.mkdir(exist_ok=True)
                cv2.imwrite(str(debug_dir / path.stem) + ".jpg", res.pop("debug_img"))

            n = len(res["detections"])
            print(f"  {path.name}: {res['status']} ({n} person(s) detected)")
            for det in res["detections"]:
                print(f"    - {det['class']} {det['confidence']:.0%} bbox={det['bbox']}")
        else:
            report = evaluate_folder(
                path, model, config, target_class_ids, debug=args.debug
            )
            total_correct += report["correct"]
            total_tested += report["total"]

            if args.json:
                print(json.dumps(report, ensure_ascii=False, indent=2))
                continue

            print(f"\n{'=' * 75}")
            print(f"  {report['folder']}")
            print(f"{'=' * 75}")
            print(f"  {'Image':<12} {'Expected':<12} {'Predicted':<12} "
                  f"{'Status':<6} {'Persons':>7}")
            print(f"  {'-' * 68}")

            for r in report["results"]:
                exp = r["expected"] or "?"
                print(
                    f"  {r['file']:<12} {exp:<12} {r['predicted']:<12} "
                    f"{r['status']:<6} {r['persons_detected']:>7}"
                )
            print(f"  {'-' * 68}")
            print(
                f"  Accuracy: {report['correct']}/{report['total']}"
                f" ({report['accuracy']:.0%})"
            )

    if total_tested > 0:
        acc = total_correct / total_tested
        print(f"\n{'=' * 75}")
        print(f"  OVERALL: {total_correct}/{total_tested} ({acc:.0%})")
        print(f"{'=' * 75}")


if __name__ == "__main__":
    main()
