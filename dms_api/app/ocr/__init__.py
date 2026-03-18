"""
OCR Module for Table Extraction

CnOCR-based table OCR extraction for railway tickets (编组单).
"""

from .models import OCRBox, OCRResult
from .engine import CnOCREngine
from .processor import TableOCRProcessor

__all__ = [
    "OCRBox",
    "OCRResult",
    "CnOCREngine",
    "TableOCRProcessor",
]
