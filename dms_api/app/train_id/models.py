"""
Train ID Data Models

Data classes for train identification recognition results.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class OCRBox:
    """Single OCR detection."""

    box: List[int]  # [x_min, y_min, x_max, y_max]
    text: str
    confidence: float = 0.0


@dataclass
class TrainIDResult:
    """Extracted train identification."""

    vehicle_type: str = ""  # e.g. C64K, C70E, C70
    vehicle_number: str = ""  # e.g. 49 31846
    confidence: float = 0.0

    @property
    def is_empty(self) -> bool:
        return not self.vehicle_type and not self.vehicle_number
