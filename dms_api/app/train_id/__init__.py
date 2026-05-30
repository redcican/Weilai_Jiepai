"""
Train ID OCR Module

Railway car identification recognition from station-entry camera images.
Extracts vehicle type (车种) and vehicle number (车号) using CnOCR
with multi-pass preprocessing and hybrid detection models.

Also supports PaddleOCR-based single-image recognition for
container IDs, train IDs, and flatcar IDs.
"""

from .models import OCRBox, TrainIDResult
from .engine import TrainIDEngine
from .processor import TrainIDProcessor
from .video_engine import PaddleOCREngine
from .paddle_image_processor import PaddleImageProcessor
from .flatcar_image_processor import FlatcarImageProcessor

__all__ = [
    "OCRBox",
    "TrainIDResult",
    "TrainIDEngine",
    "TrainIDProcessor",
    "PaddleOCREngine",
    "PaddleImageProcessor",
    "FlatcarImageProcessor",
]
