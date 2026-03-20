"""
Train ID OCR Module

Railway car identification recognition from station-entry camera images.
Extracts vehicle type (车种) and vehicle number (车号) using CnOCR
with multi-pass preprocessing and hybrid detection models.
"""

from .models import OCRBox, TrainIDResult
from .engine import TrainIDEngine
from .processor import TrainIDProcessor

__all__ = [
    "OCRBox",
    "TrainIDResult",
    "TrainIDEngine",
    "TrainIDProcessor",
]
