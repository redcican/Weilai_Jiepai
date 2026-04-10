"""
Pedestrian Detection Module

YOLOv8-based pedestrian detection for railway driving safety.
Detects persons in freight carriage top-down camera images using
a two-pass strategy: full-image + sliding-window tiling.
"""

from .engine import PedestrianEngine

__all__ = ["PedestrianEngine"]
