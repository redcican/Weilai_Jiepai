"""
OCR Data Models

Data classes for OCR results.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class OCRBox:
    """Single OCR recognition result."""
    box: List[int]
    text: str
    confidence: float = 0.0


@dataclass
class OCRResult:
    """Complete OCR extraction result."""
    status: str
    message: str
    metadata: Dict[str, str] = field(default_factory=dict)
    table_data: List[List[str]] = field(default_factory=list)

    @property
    def is_success(self) -> bool:
        return self.status == "success"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "message": self.message,
            "metadata": self.metadata,
            "table_data": self.table_data,
        }
