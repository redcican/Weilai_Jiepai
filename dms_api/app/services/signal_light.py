"""
Signal Light Service

Business logic for signal light color detection.
Standalone — no DMS backend dependency.
"""

import logging
from typing import Optional

import cv2
import numpy as np

from ..schemas.signal_light import SignalLightItem
from ..signal_light.signal_detect import (
    detect_signal_color_from_frame,
    load_config,
)

logger = logging.getLogger(__name__)

COLOR_MAP = {
    "red": "红色",
    "white": "白色",
    "blue": "蓝色",
    "unknown": "未知",
}


class SignalLightService:
    """Service for signal light color detection operations."""

    @property
    def available(self) -> bool:
        return True

    async def detect_batch(
        self,
        images: list[tuple[bytes, str]],
        camera_type: str | None = None,
    ) -> list[SignalLightItem]:
        """Detect signal light color from multiple images.

        Args:
            images: List of (image_bytes, filename) tuples.
            camera_type: Camera identifier for ROI lookup (front_signal / rear_signal / exit_signal).

        Returns:
            List of SignalLightItem with filename, Chinese color, confidence and scores.
        """
        config = load_config()
        cam_cfg = config.get(camera_type, {}) if camera_type else {}
        roi = cam_cfg.get("roi")
        signal_center = cam_cfg.get("signal_center")

        results = []
        for image_bytes, filename in images:
            img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                logger.warning(f"Failed to decode image: {filename}")
                results.append(
                    SignalLightItem(filename=filename, color="未知", confidence=0.0, scores={})
                )
                continue

            result = detect_signal_color_from_frame(img, roi, signal_center)
            color_en = result.get("color", "unknown")
            color_cn = COLOR_MAP.get(color_en, "未知")

            logger.info(f"Signal light: {filename} → {color_cn}")
            results.append(
                SignalLightItem(
                    filename=filename,
                    color=color_cn,
                    confidence=result.get("confidence", 0.0),
                    scores=result.get("scores", {}),
                )
            )
        return results


_signal_light_service: Optional[SignalLightService] = None


def get_signal_light_service_singleton() -> SignalLightService:
    global _signal_light_service
    if _signal_light_service is None:
        _signal_light_service = SignalLightService()
    return _signal_light_service
