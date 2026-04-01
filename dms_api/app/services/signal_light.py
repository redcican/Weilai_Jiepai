"""
Signal Light Service

Business logic for signal light color detection.
Standalone — no DMS backend dependency.
"""

import logging
from typing import Optional

from ..signal_light import SignalLightEngine
from ..schemas.signal_light import SignalLightItem

logger = logging.getLogger(__name__)


class SignalLightService:
    """Service for signal light color detection operations."""

    _engine: Optional[SignalLightEngine] = None

    @classmethod
    def get_engine(cls) -> SignalLightEngine:
        if cls._engine is None:
            cls._engine = SignalLightEngine.get_instance()
        return cls._engine

    @property
    def available(self) -> bool:
        return self.get_engine().available

    async def detect_batch(
        self,
        images: list[tuple[bytes, str]],
        roi: list[int] | None = None,
    ) -> list[SignalLightItem]:
        """Detect signal light color from multiple images.

        Args:
            images: List of (image_bytes, filename) tuples.
            roi: Optional [x1, y1, x2, y2] in 1280x720 coordinates.

        Returns:
            List of SignalLightItem with filename and Chinese color.
        """
        engine = self.get_engine()
        results = []
        for image_bytes, filename in images:
            color = engine.detect_from_bytes(image_bytes, roi=roi)
            logger.info(f"Signal light: {filename} → {color}")
            results.append(SignalLightItem(filename=filename, color=color))
        return results


_signal_light_service: Optional[SignalLightService] = None


def get_signal_light_service_singleton() -> SignalLightService:
    global _signal_light_service
    if _signal_light_service is None:
        _signal_light_service = SignalLightService()
    return _signal_light_service
