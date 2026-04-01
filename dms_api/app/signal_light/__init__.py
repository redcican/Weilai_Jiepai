"""
Signal Light Detection Module

Pure OpenCV + NumPy color detection for railway signal lights.
Detects blue / red / white from fixed surveillance camera images
using per-camera ROI and HSV color analysis.
"""

from .engine import SignalLightEngine

__all__ = ["SignalLightEngine"]
