"""
Railway Signal Light Color Detection
=====================================
Pure OpenCV + NumPy — no ML models. Detects blue / red / white signal lights
from fixed surveillance camera images.

Strategy: per-camera ROI + HSV color analysis.
Fixed cameras have consistent signal positions — analyzing only the signal
region eliminates false positives from sky, trains, equipment, etc.

ROI config is stored in signal_config.json. Use --calibrate to auto-detect
signal positions for new camera folders.

Usage:
    python signal_detect.py                          # all subfolders
    python signal_detect.py 拨车机前侧信号灯识别     # specific folder
    python signal_detect.py --calibrate              # auto-detect ROIs
    python signal_detect.py --debug                  # save annotated images
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import json
import sys

CONFIG_FILE = "signal_config.json"

# ---------------------------------------------------------------------------
# Default ROI config (calibrated from the three known cameras)
# Format: [x1, y1, x2, y2] — pixel coordinates in 1280x720 images
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "拨车机前侧信号灯识别": {
        "roi": [710, 255, 840, 370],
        "description": "Front-side signal at track switch point",
    },
    "拨车机后侧信号灯": {
        "roi": [340, 55, 870, 420],
        "description": "Rear-side signal — large ROI due to variable position",
    },
    "装车楼出口信号灯": {
        "roi": [520, 330, 630, 400],
        "signal_center": [573, 370],
        "description": "Loading exit signal between tracks",
    },
}


def load_config() -> dict:
    """Load camera config from file, falling back to defaults."""
    config_path = Path(CONFIG_FILE)
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return dict(DEFAULT_CONFIG)


def save_config(config: dict):
    """Save camera config to file."""
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def detect_signal_color(
    image_path: str,
    roi: list[int] | None = None,
    signal_center: list[int] | None = None,
    debug: bool = False,
) -> dict:
    """Detect signal light color from a railway camera image.

    Args:
        image_path: Path to the image file.
        roi: [x1, y1, x2, y2] region of interest. None = full image.
        debug: If True, include annotated debug image in result.

    Returns:
        dict with keys: color, confidence, scores, (optional) debug_img
    """
    img = cv2.imread(image_path)
    if img is None:
        return {"color": "unknown", "confidence": 0.0, "reason": "cannot read image"}

    h_img, w_img = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Grayscale check (IR night mode — no color information)
    if np.max(sat) < 5:
        result = {
            "color": "unknown",
            "confidence": 0.0,
            "reason": "grayscale/IR — no color info",
        }
        if debug:
            dbg = img.copy()
            cv2.putText(dbg, "GRAYSCALE - no color", (10, h_img - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            result["debug_img"] = dbg
        return result

    # Build ROI mask
    mask_roi = np.zeros((h_img, w_img), dtype=bool)
    if roi:
        x1, y1, x2, y2 = roi
        # Scale ROI if image dimensions differ from 1280x720
        sx, sy = w_img / 1280, h_img / 720
        x1, x2 = int(x1 * sx), int(x2 * sx)
        y1, y2 = int(y1 * sy), int(y2 * sy)
        mask_roi[y1:y2, x1:x2] = True
    else:
        # Fallback: exclude timestamp (top 12%) and text overlay (bottom 8%)
        mask_roi[int(h_img * 0.12) : int(h_img * 0.92), :] = True

    # --- Priority-based detection ---
    # Signal lights coexist with blue-painted equipment in the ROI.
    # Priority: red (if present) > blue LED (strict) > white (fallback).
    # Blue equipment is always present; only a true blue LED (V > 200) counts.

    # Red LED (two sub-ranges for different LED types):
    #   Cool red/magenta: H ≥ 155 (most railway signals)
    #   Warm red/orange:  H ≤ 18, S > 70 (strict S to reject soil/rust)
    cool_red = (hue >= 155) & (sat > 40) & (val > 60) & mask_roi
    warm_red = (hue <= 18) & (sat > 70) & (val > 90) & mask_roi
    red_mask = cool_red | warm_red
    red_blobs = _find_blobs(red_mask, max_area=2000)
    red_score = sum(b["area"] for b in red_blobs)

    # Blue LED (strict V > 200 to separate glowing LED from painted surface)
    blue_mask = (hue >= 85) & (hue <= 125) & (sat > 60) & (val > 200) & mask_roi
    blue_blobs = _find_blobs(blue_mask, max_area=2000)
    blue_score = sum(b["area"] for b in blue_blobs)

    # White: very bright + desaturated
    white_mask = (sat < 50) & (val > 200) & mask_roi
    white_blobs = _find_blobs(white_mask, max_area=800)
    white_score = sum(b["area"] for b in white_blobs) * 0.03

    scores = {"blue": blue_score, "red": red_score, "white": white_score}

    # Priority logic: red > strict-blue > white
    if red_score >= 10:
        color = "red"
    elif blue_score >= 10:
        color = "blue"
    elif white_score > 0:
        color = "white"
    else:
        # Relaxed white fallback
        wm2 = (sat < 60) & (val > 180) & mask_roi
        wb2 = _find_blobs(wm2, max_area=1000)
        ws2 = sum(b["area"] for b in wb2) * 0.02
        if ws2 > 0:
            color = "white"
            scores["white"] = ws2
        else:
            color = "unknown"

    # --- Signal-center R-B disambiguation (nighttime red vs white) ---
    # When signal_center is configured and priority logic picked "red",
    # check the actual LED center: white LEDs have B > R (R-B < -5),
    # while red LEDs have R ≈ B (R-B ≈ 0).
    if signal_center and color == "red":
        scx, scy = signal_center
        sx, sy = w_img / 1280, h_img / 720
        scx, scy = int(scx * sx), int(scy * sy)
        r = 3  # 7x7 patch
        py1, py2 = max(0, scy - r), min(h_img, scy + r + 1)
        px1, px2 = max(0, scx - r), min(w_img, scx + r + 1)
        patch_v = val[py1:py2, px1:px2]
        patch_s = sat[py1:py2, px1:px2]
        peak_v_center = np.max(patch_v)
        mean_s_center = np.mean(patch_s.astype(float))
        if peak_v_center > 200 and mean_s_center < 50:
            patch_bgr = img[py1:py2, px1:px2]
            r_mean = np.mean(patch_bgr[:, :, 2].astype(float))
            b_mean = np.mean(patch_bgr[:, :, 0].astype(float))
            if (r_mean - b_mean) < -5:
                color = "white"

    total_score = sum(scores.values())
    confidence = scores[color] / total_score if total_score > 0 and color != "unknown" else 0.0

    result = {
        "color": color,
        "confidence": round(confidence, 3),
        "scores": {k: round(v, 1) for k, v in scores.items()},
    }

    if debug:
        dbg = img.copy()
        # Draw ROI rectangle
        if roi:
            sx, sy = w_img / 1280, h_img / 720
            rx1, ry1 = int(roi[0] * sx), int(roi[1] * sy)
            rx2, ry2 = int(roi[2] * sx), int(roi[3] * sy)
            cv2.rectangle(dbg, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)
        _draw_blobs(dbg, blue_blobs, (255, 0, 0), "B")
        _draw_blobs(dbg, red_blobs, (0, 0, 255), "R")
        _draw_blobs(dbg, white_blobs, (200, 200, 200), "W")
        cv2.putText(
            dbg,
            f"{color} ({confidence:.0%})",
            (10, h_img - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        result["debug_img"] = dbg

    return result


def _find_blobs(
    mask: np.ndarray, min_area: int = 3, max_area: int = 3000
) -> list[dict]:
    """Find connected-component blobs within area range."""
    mask_u8 = mask.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask_u8)

    blobs = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            cx, cy = centroids[i]
            blobs.append({"area": area, "cx": cx, "cy": cy})
    return blobs


def _draw_blobs(img: np.ndarray, blobs: list[dict], color: tuple, tag: str):
    """Draw blob markers on debug image."""
    for b in blobs:
        cx, cy = int(b["cx"]), int(b["cy"])
        r = max(5, int(np.sqrt(b["area"] / np.pi)))
        cv2.circle(img, (cx, cy), r + 3, color, 2)
        cv2.putText(
            img, f"{tag}{b['area']}", (cx + r + 4, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
        )


# ---------------------------------------------------------------------------
# Calibration — auto-detect signal ROI for a folder
# ---------------------------------------------------------------------------

def calibrate_folder(folder: Path) -> dict | None:
    """Auto-detect signal light ROI for a folder of fixed-camera images.

    Finds the most consistent high-saturation bright spot across images.
    Returns ROI dict or None if detection fails.
    """
    images = sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg"))
    if not images:
        return None

    h_img, w_img = 720, 1280
    # Accumulate per-pixel S*V across images
    sv_accum = np.zeros((h_img, w_img), dtype=float)
    count = 0

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        if img.shape[:2] != (h_img, w_img):
            img = cv2.resize(img, (w_img, h_img))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if np.max(hsv[:, :, 1]) < 10:
            continue  # skip grayscale
        s_f = hsv[:, :, 1].astype(float)
        v_f = hsv[:, :, 2].astype(float)
        sv_accum += (s_f * v_f) / 65025.0
        count += 1

    if count == 0:
        return None

    sv_mean = sv_accum / count
    # Exclude margins
    sv_mean[:int(h_img * 0.12), :] = 0
    sv_mean[int(h_img * 0.92):, :] = 0
    sv_mean[:, :int(w_img * 0.05)] = 0
    sv_mean[:, int(w_img * 0.95):] = 0

    sv_smooth = cv2.GaussianBlur(sv_mean, (31, 31), 0)

    # Find top peaks
    peaks = []
    temp = sv_smooth.copy()
    for _ in range(10):
        py, px = np.unravel_index(temp.argmax(), temp.shape)
        val = temp[py, px]
        if val < 0.05:
            break
        peaks.append((px, py, val))
        y1, y2 = max(0, py - 50), min(h_img, py + 50)
        x1, x2 = max(0, px - 50), min(w_img, px + 50)
        temp[y1:y2, x1:x2] = 0

    if not peaks:
        return None

    # Use the top peak as center, with generous margin
    cx, cy, _ = peaks[0]
    margin = 60
    roi = [
        max(0, cx - margin),
        max(0, cy - margin),
        min(w_img, cx + margin),
        min(h_img, cy + margin),
    ]

    return {
        "roi": roi,
        "center": [cx, cy],
        "description": f"Auto-detected from {count} images",
    }


# ---------------------------------------------------------------------------
# Ground truth & evaluation
# ---------------------------------------------------------------------------

def load_ground_truth(folder: Path) -> dict[str, str]:
    """Load ground truth from ground_truth.* files in folder."""
    gt = {}
    for p in folder.iterdir():
        if p.name.startswith("ground_truth"):
            for line in p.read_text().strip().splitlines():
                line = line.strip()
                if ":" in line:
                    fname, label = line.rsplit(":", 1)
                    gt[fname.strip()] = label.strip().lower()
    return gt


def evaluate_folder(folder: Path, config: dict, debug: bool = False) -> dict:
    """Evaluate all images in a folder against ground truth."""
    gt = load_ground_truth(folder)
    images = sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg"))

    cam_cfg = config.get(folder.name, {})
    roi = cam_cfg.get("roi")
    signal_center = cam_cfg.get("signal_center")

    results = []
    correct, total, skipped = 0, 0, 0

    debug_dir = None
    if debug:
        debug_dir = Path("debug") / folder.name
        debug_dir.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        res = detect_signal_color(str(img_path), roi=roi, signal_center=signal_center, debug=debug)
        expected = gt.get(img_path.name)
        predicted = res["color"]

        if predicted == "unknown":
            status = "SKIP"
            skipped += 1
        elif expected is None:
            status = "NO_GT"
        else:
            total += 1
            if predicted == expected:
                correct += 1
                status = "OK"
            else:
                status = "FAIL"

        results.append({
            "file": img_path.name,
            "predicted": predicted,
            "expected": expected,
            "status": status,
            "confidence": res["confidence"],
            "scores": res.get("scores"),
        })

        if debug and "debug_img" in res:
            cv2.imwrite(str(debug_dir / img_path.name), res["debug_img"])

    accuracy = correct / total if total > 0 else 0.0
    return {
        "folder": folder.name,
        "accuracy": round(accuracy, 3),
        "correct": correct,
        "total": total,
        "skipped": skipped,
        "roi": roi,
        "results": results,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Railway signal light color detection (blue/red/white)"
    )
    parser.add_argument(
        "paths", nargs="*",
        help="Image file(s) or folder(s). Default: all subfolders.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Save annotated debug images to ./debug/",
    )
    parser.add_argument(
        "--calibrate", action="store_true",
        help="Auto-detect signal ROIs and save to config",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output JSON instead of table",
    )
    args = parser.parse_args()

    config = load_config()

    # --- Calibration mode ---
    if args.calibrate:
        folders = _discover_folders(args.paths)
        for folder in folders:
            print(f"Calibrating: {folder.name} ...")
            result = calibrate_folder(folder)
            if result:
                config[folder.name] = result
                print(f"  ROI = {result['roi']}  center = {result.get('center')}")
            else:
                print("  Failed — no colored images found")
        save_config(config)
        print(f"\nConfig saved to {CONFIG_FILE}")
        return

    # --- Detection mode ---
    targets = []
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
        for d in sorted(Path(".").iterdir()):
            if (d.is_dir() and not d.name.startswith(".")
                    and d.name != "debug"
                    and (list(d.glob("*.png")) or list(d.glob("*.jpg")))):
                targets.append(("dir", d))

    if not targets:
        print("No images or folders found.", file=sys.stderr)
        sys.exit(1)

    total_correct, total_tested, total_skipped = 0, 0, 0

    for kind, path in targets:
        if kind == "file":
            # Determine ROI from parent folder
            cam_cfg = config.get(path.parent.name, {})
            roi = cam_cfg.get("roi")
            sc = cam_cfg.get("signal_center")
            res = detect_signal_color(str(path), roi=roi, signal_center=sc, debug=args.debug)
            if args.debug and "debug_img" in res:
                Path("debug").mkdir(exist_ok=True)
                cv2.imwrite(f"debug/{path.name}", res.pop("debug_img", None))
            print(f"  {path.name}: {res['color']} ({res['confidence']:.0%})")
        else:
            report = evaluate_folder(path, config, debug=args.debug)
            total_correct += report["correct"]
            total_tested += report["total"]
            total_skipped += report["skipped"]

            if args.json:
                print(json.dumps(report, ensure_ascii=False, indent=2))
                continue

            roi_str = f"ROI={report['roi']}" if report["roi"] else "no ROI"
            print(f"\n{'=' * 65}")
            print(f"  {report['folder']}  ({roi_str})")
            print(f"{'=' * 65}")
            print(f"  {'Image':<12} {'Expected':<10} {'Predicted':<10} {'Status':<6} {'Conf':>6}")
            print(f"  {'-' * 58}")
            for r in report["results"]:
                exp = r["expected"] or "?"
                print(
                    f"  {r['file']:<12} {exp:<10} {r['predicted']:<10} "
                    f"{r['status']:<6} {r['confidence']:>5.0%}"
                )
            print(f"  {'-' * 58}")
            print(
                f"  Accuracy: {report['correct']}/{report['total']}"
                f" ({report['accuracy']:.0%})"
                f"  |  Skipped: {report['skipped']}"
            )

    if total_tested > 0:
        acc = total_correct / total_tested
        print(f"\n{'=' * 65}")
        print(f"  OVERALL: {total_correct}/{total_tested} ({acc:.0%})"
              f"  |  Skipped: {total_skipped}")
        print(f"{'=' * 65}")


def _discover_folders(paths: list[str]) -> list[Path]:
    """Discover image folders from CLI paths or CWD."""
    if paths:
        return [Path(p) for p in paths if Path(p).is_dir()]
    return [
        d for d in sorted(Path(".").iterdir())
        if d.is_dir() and not d.name.startswith(".")
        and d.name != "debug"
        and (list(d.glob("*.png")) or list(d.glob("*.jpg")))
    ]


if __name__ == "__main__":
    main()
