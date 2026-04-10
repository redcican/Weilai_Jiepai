"""
Pedestrian Detection Service

Business logic for pedestrian anomaly detection.
Standalone — no DMS backend dependency.
"""

import logging
from typing import Optional

from ..pedestrian import PedestrianEngine
from ..schemas.pedestrian import PedestrianItem, PedestrianDetection

logger = logging.getLogger(__name__)


class PedestrianService:
    """Service for pedestrian detection operations."""

    _engine: Optional[PedestrianEngine] = None

    @classmethod
    def get_engine(cls) -> PedestrianEngine:
        if cls._engine is None:
            cls._engine = PedestrianEngine.get_instance()
        return cls._engine

    @property
    def available(self) -> bool:
        return self.get_engine().available

    async def detect_batch(
        self,
        images: list[tuple[bytes, str]],
        use_gpu: bool | None = None,
    ) -> list[PedestrianItem]:
        """Detect pedestrians in multiple images.

        Args:
            images: List of (image_bytes, filename) tuples.
            use_gpu: Override device. None=config default, True=GPU, False=CPU.

        Returns:
            List of PedestrianItem with status and detection details.
        """
        engine = self.get_engine()
        results = []
        for image_bytes, filename in images:
            result = engine.detect_from_bytes(image_bytes, use_gpu=use_gpu)
            detections = [
                PedestrianDetection(
                    class_name=d["class_name"],
                    confidence=d["confidence"],
                    bbox=d["bbox"],
                )
                for d in result["detections"]
            ]
            item = PedestrianItem(
                filename=filename,
                status=result["status"],
                pedestrian_count=len(detections),
                detections=detections,
            )
            logger.info(
                f"Pedestrian: {filename} → {result['status']} "
                f"({len(detections)} detected)"
            )
            results.append(item)
        return results


_pedestrian_service: Optional[PedestrianService] = None


def get_pedestrian_service_singleton() -> PedestrianService:
    global _pedestrian_service
    if _pedestrian_service is None:
        _pedestrian_service = PedestrianService()
    return _pedestrian_service
