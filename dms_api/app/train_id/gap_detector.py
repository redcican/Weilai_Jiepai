"""
空挡检测器（纯 CV，不训练）

基于车厢连接处图像特征：
  1. 中间竖直金属结构 -> 竖直边缘密度高
  2. 背景开口和灯光 -> CLAHE 后低梯度占比适中
  3. 灯光可见 -> 亮斑连通域存在
  4. 非黄色/白色机车侧面 -> 平均亮度不太高
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np


class GapType(Enum):
    """空挡检测结果类型"""
    GAP = "gap"
    NORMAL = "normal"
    TRANSITION = "transition"


@dataclass
class GapResult:
    """空挡检测结果"""
    gap_type: GapType
    vert_edge_ratio: float
    center_std: float
    score: float


class GapDetector:
    """
    空挡检测器 - 严格多条件组合版
    """

    def __init__(
        self,
        roi_x: Tuple[float, float] = (0.45, 0.55),
        roi_y: Tuple[float, float] = (0.2, 0.9),
        edge_thresh: int = 50,
        gap_thresh: float = 0.018,
        normal_thresh: float = 0.005,
        strict_mode: bool = True,
        min_vert_ratio: float = 0.015,
        max_low_grad_ratio: float = 0.75,
        max_brightness: float = 140.0,
        min_bright_blob_ratio: float = 0.02,
    ):
        self.roi_x = roi_x
        self.roi_y = roi_y
        self.edge_thresh = edge_thresh
        self.gap_thresh = gap_thresh
        self.normal_thresh = normal_thresh
        self.strict_mode = strict_mode
        self.min_vert_ratio = min_vert_ratio
        self.max_low_grad_ratio = max_low_grad_ratio
        self.max_brightness = max_brightness
        self.min_bright_blob_ratio = min_bright_blob_ratio

    def _extract_roi(self, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape
        x1 = int(w * self.roi_x[0])
        x2 = int(w * self.roi_x[1])
        y1 = int(h * self.roi_y[0])
        y2 = int(h * self.roi_y[1])
        return gray[y1:y2, x1:x2]

    def _compute_vert_edge_ratio(self, roi: np.ndarray) -> float:
        if roi.size == 0:
            return 0.0
        sobel_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
        mag = np.abs(sobel_x)
        strong_pixels = np.sum(mag > self.edge_thresh)
        return float(strong_pixels / mag.size)

    def _compute_std(self, roi: np.ndarray) -> float:
        return float(np.std(roi))

    def _compute_low_grad_ratio(self, roi: np.ndarray) -> float:
        """CLAHE 后梯度 < 20 的像素占比"""
        if roi.size == 0:
            return 1.0
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        roi_clahe = clahe.apply(roi)
        grad_x = cv2.Sobel(roi_clahe, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(roi_clahe, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        return float(np.sum(grad_mag < 20) / grad_mag.size)

    def _compute_bright_blob_ratio(self, roi: np.ndarray, thresh: int = 220) -> float:
        """亮斑最大连通域占比"""
        if roi.size == 0:
            return 0.0
        _, binary = cv2.threshold(roi, thresh, 255, cv2.THRESH_BINARY)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        max_area = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= 10 and area > max_area:
                max_area = area
        return float(max_area / roi.size) if roi.size > 0 else 0.0

    def detect(self, image: Union[np.ndarray, str, Path]) -> GapResult:
        if isinstance(image, (str, Path)):
            data = np.fromfile(str(image), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                return GapResult(GapType.TRANSITION, 0.0, 0.0, 0.0)
        else:
            img = image

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        roi = self._extract_roi(gray)
        vert_ratio = self._compute_vert_edge_ratio(roi)
        center_std = self._compute_std(roi)

        if self.strict_mode:
            low_grad_ratio = self._compute_low_grad_ratio(roi)
            bright_blob_ratio = self._compute_bright_blob_ratio(roi)
            brightness_mean = float(np.mean(roi))

            is_gap = (
                vert_ratio > self.min_vert_ratio and
                low_grad_ratio < self.max_low_grad_ratio and
                brightness_mean < self.max_brightness and
                bright_blob_ratio > self.min_bright_blob_ratio
            )

            gap_type = GapType.GAP if is_gap else GapType.NORMAL
            score = 1.0 / (1.0 + np.exp(-200 * (vert_ratio - 0.01)))
            return GapResult(gap_type, vert_ratio, center_std, round(score, 4))
        else:
            if vert_ratio >= self.gap_thresh:
                gap_type = GapType.GAP
            elif vert_ratio <= self.normal_thresh:
                gap_type = GapType.NORMAL
            else:
                gap_type = GapType.TRANSITION

            score = 1.0 / (1.0 + np.exp(-200 * (vert_ratio - 0.01)))
            return GapResult(gap_type, vert_ratio, center_std, round(score, 4))
