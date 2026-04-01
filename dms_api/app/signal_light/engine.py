"""
Signal Light Detection Engine

HSV color analysis for railway signal light recognition.
Pure OpenCV + NumPy — no ML models, no hardcoded camera config.

Two modes:
- With ROI: caller provides region coordinates → high accuracy
- Without ROI: auto-detect via S*V blob ranking → best effort
"""

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

COLOR_MAP = {
    "red": "红色",
    "blue": "蓝色",
    "white": "白色",
    "unknown": "未知",
}


class SignalLightEngine:
    """Signal light color detector — singleton, pure OpenCV."""

    _instance: Optional["SignalLightEngine"] = None

    def __init__(self):
        self._available = True
        logger.info("Signal light engine initialized (OpenCV + NumPy)")

    @classmethod
    def get_instance(cls) -> "SignalLightEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def available(self) -> bool:
        return self._available

    def detect_from_bytes(
        self,
        image_bytes: bytes,
        roi: list[int] | None = None,
    ) -> str:
        """Detect signal light color from raw image bytes.

        Args:
            image_bytes: Raw image file content (JPEG/PNG/BMP).
            roi: Optional [x1, y1, x2, y2] in 1280x720 coordinates.

        Returns:
            Chinese color string: 红色, 白色, 蓝色, or 未知.
        """
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return "未知"

        if roi:
            color_en = self._detect_with_roi(img, roi)
        else:
            color_en = self._detect_auto(img)
        return COLOR_MAP.get(color_en, "未知")

    # ------------------------------------------------------------------
    # ROI mode: precise detection within a known region
    # ------------------------------------------------------------------

    def _detect_with_roi(self, img: np.ndarray, roi: list[int]) -> str:
        h_img, w_img = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        if np.max(sat) < 5:
            return "unknown"

        # Scale ROI from 1280x720 reference
        sx, sy = w_img / 1280, h_img / 720
        x1, y1, x2, y2 = roi
        x1, x2 = int(x1 * sx), int(x2 * sx)
        y1, y2 = int(y1 * sy), int(y2 * sy)

        mask = np.zeros((h_img, w_img), dtype=bool)
        mask[y1:y2, x1:x2] = True

        cool_red = (hue >= 155) & (sat > 40) & (val > 60) & mask
        warm_red = (hue <= 18) & (sat > 70) & (val > 90) & mask
        red_score = self._blob_score(cool_red | warm_red)

        blue_mask = (hue >= 85) & (hue <= 125) & (sat > 60) & (val > 200) & mask
        blue_score = self._blob_score(blue_mask)

        white_mask = (sat < 50) & (val > 200) & mask
        white_score = self._blob_score(white_mask, max_area=800) * 0.03

        if red_score >= 10:
            color = "red"
        elif blue_score >= 10:
            color = "blue"
        elif white_score > 0:
            color = "white"
        else:
            wm2 = (sat < 60) & (val > 180) & mask
            ws2 = self._blob_score(wm2, max_area=1000) * 0.02
            color = "white" if ws2 > 0 else "unknown"

        # Auto red/white disambiguation: if classified red, check if the
        # brightest pixel in ROI is actually an overexposed white LED
        if color == "red":
            roi_v = val[y1:y2, x1:x2]
            py, px = np.unravel_index(np.argmax(roi_v), roi_v.shape)
            if roi_v[py, px] > 200:
                r = 3
                sy1 = max(0, py - r)
                sy2 = min(roi_v.shape[0], py + r + 1)
                sx1 = max(0, px - r)
                sx2 = min(roi_v.shape[1], px + r + 1)
                patch_s = float(np.mean(sat[y1 + sy1 : y1 + sy2, x1 + sx1 : x1 + sx2].astype(float)))
                if patch_s < 55:
                    patch = img[y1 + sy1 : y1 + sy2, x1 + sx1 : x1 + sx2]
                    r_mean = float(np.mean(patch[:, :, 2].astype(float)))
                    b_mean = float(np.mean(patch[:, :, 0].astype(float)))
                    if (r_mean - b_mean) < -5:
                        color = "white"

        return color

    # ------------------------------------------------------------------
    # Auto mode: find signal light by S*V blob ranking
    # ------------------------------------------------------------------

    def _detect_auto(self, img: np.ndarray) -> str:
        h_img, w_img = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        if np.max(sat) < 5:
            return "unknown"

        # Exclude timestamp margins
        margin = np.zeros((h_img, w_img), dtype=bool)
        margin[int(h_img * 0.08) : int(h_img * 0.95), :] = True

        # Find bright pixels → candidate blobs
        bright = (val > 150) & margin
        bright_u8 = bright.astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bright_u8 = cv2.morphologyEx(bright_u8, cv2.MORPH_OPEN, kernel)

        max_area = int(500 * (h_img * w_img) / (1280 * 720))
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            bright_u8
        )

        blobs = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 5 or area > max_area:
                continue
            bm = labels == i
            mean_s = float(np.mean(sat[bm].astype(float)))
            mean_v = float(np.mean(val[bm].astype(float)))
            mean_h = float(np.mean(hue[bm].astype(float)))
            sv = mean_s * mean_v

            if mean_s > 40:
                if mean_h >= 155 or mean_h <= 18:
                    color = "red"
                elif 85 <= mean_h <= 125:
                    color = "blue"
                else:
                    color = "other"
            else:
                color = "white"

            blobs.append({"sv": sv, "color": color, "mean_v": mean_v})

        if not blobs:
            return "unknown"

        # Sort by S*V — LED-like (bright + saturated) blobs rank first
        blobs.sort(key=lambda b: b["sv"], reverse=True)

        # First colored (non-white, non-other) blob wins
        for b in blobs[:10]:
            if b["color"] in ("red", "blue"):
                return b["color"]

        # No colored signal → check for white
        if blobs[0]["color"] == "white" and blobs[0]["mean_v"] > 200:
            return "white"

        return "unknown"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _blob_score(
        mask: np.ndarray,
        min_area: int = 3,
        max_area: int = 2000,
    ) -> int:
        """Sum of connected-component blob areas within size range."""
        mask_u8 = mask.astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_u8)
        total = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if min_area <= area <= max_area:
                total += area
        return total
