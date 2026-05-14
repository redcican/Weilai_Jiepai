#!/usr/bin/env python3
"""用 V6 代码批量处理平板车视频"""

import os
import glob
import subprocess
import sys

VIDEO_DIR = "D:/工作/Weilai_Jiepai/train_id_ocr/车厢号识别/MV-CH120-60GC (DA7755688)(IP地址192.168.11.118)/MV-CH120-60GC (DA7755688)"
OUTPUT_BASE = "D:/工作/Weilai_Jiepai/train_id_ocr/output_v6_flatcar_batch"


def main():
    videos = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.avi")))
    print(f"Found {len(videos)} videos")

    os.makedirs(OUTPUT_BASE, exist_ok=True)

    for i, video_path in enumerate(videos, 1):
        name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(OUTPUT_BASE, name)

        print(f"\n{'='*60}")
        print(f"[{i}/{len(videos)}] Processing: {name}")
        print(f"Output: {output_dir}")
        print(f"{'='*60}")

        cmd = [
            sys.executable,
            "train_id_ocr_video_paddle_v6.py",
            video_path,
            "-o", output_dir,
            "--cpu",
        ]
        result = subprocess.run(cmd, capture_output=False, text=True)

        if result.returncode != 0:
            print(f"ERROR: {name} failed with code {result.returncode}")

    print(f"\n{'='*60}")
    print(f"All done. Results in: {OUTPUT_BASE}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
