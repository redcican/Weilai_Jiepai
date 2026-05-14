"""
Train ID OCR Module

Railway car identification recognition from station-entry camera images.
Extracts vehicle type (车种) and vehicle number (车号) using CnOCR
with multi-pass preprocessing and hybrid detection models.

Also supports video-based recognition using PaddleOCR for
container IDs, train IDs, and flatcar IDs with temporal aggregation.
"""

from .models import OCRBox, TrainIDResult
from .engine import TrainIDEngine
from .processor import TrainIDProcessor
from .video_engine import PaddleOCREngine
from .video_processor import VideoTrainIDProcessor, VideoRecognitionResult
from .flatcar_processor import FlatcarVideoProcessor, FlatcarRecognitionResult

__all__ = [
    "OCRBox",
    "TrainIDResult",
    "TrainIDEngine",
    "TrainIDProcessor",
    "PaddleOCREngine",
    "VideoTrainIDProcessor",
    "VideoRecognitionResult",
    "FlatcarVideoProcessor",
    "FlatcarRecognitionResult",
]
