"""
板车空挡检测器 - 生产级代码 (v2)
传统 CV 方案，零依赖（仅需 OpenCV + NumPy）

v2 改进：增加"左右不对称性"特征，解决暗光空挡漏检问题
"""

import cv2
import numpy as np
import os
import glob


class FlatcarGapDetector:
    """
    板车空挡检测器
    基于中央 ROI 的亮度、边缘、方差、左右不对称性特征进行二分类判断。
    """

    def __init__(
        self,
        roi_x_ratio=(0.30, 0.70),
        roi_y_ratio=(0.20, 0.80),
        brightness_thresh=55.0,
        std_thresh=35.0,
        edge_thresh=25.0,
        profile_var_thresh=20.0,
        profile_min_thresh=25.0,
        asymmetry_thresh=0.45,
        min_gap_score=7,
        # 新增：车型模式切换
        vehicle_type='standard',  # 'standard' 或 'flatcar'
    ):
        # 根据车型自动切换阈值
        if vehicle_type == 'flatcar':
            # 板车专用参数（基于1227帧分布估算）
            brightness_thresh = 45.0
            std_thresh = 25.0
            edge_thresh = 12.0
            profile_var_thresh = 15.0
            profile_min_thresh = 20.0
            asymmetry_thresh = 0.20
            min_gap_score = 5
            self._vehicle_type = 'flatcar'
        else:
            self._vehicle_type = 'standard'
        self.roi_x_ratio = roi_x_ratio
        self.roi_y_ratio = roi_y_ratio
        self.brightness_thresh = brightness_thresh
        self.std_thresh = std_thresh
        self.edge_thresh = edge_thresh
        self.profile_var_thresh = profile_var_thresh
        self.profile_min_thresh = profile_min_thresh
        self.asymmetry_thresh = asymmetry_thresh
        self.min_gap_score = min_gap_score

    def _imread(self, path):
        img_array = np.fromfile(path, np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    def _extract_features(self, img):
        h, w = img.shape[:2]
        x1, x2 = int(w * self.roi_x_ratio[0]), int(w * self.roi_x_ratio[1])
        y1, y2 = int(h * self.roi_y_ratio[0]), int(h * self.roi_y_ratio[1])
        roi = img[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_h, roi_w = gray.shape

        # 1. 基础特征
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        edge_score = np.mean(cv2.convertScaleAbs(sobel_y)) + np.mean(cv2.convertScaleAbs(sobel_x))

        mean_brightness = float(np.mean(gray))
        std_brightness = float(np.std(gray))
        horizontal_profile = np.mean(gray, axis=1)
        profile_variance = float(np.std(horizontal_profile))
        profile_min = float(np.min(horizontal_profile))

        # 2. 左右不对称性（核心改进）
        # 把 ROI 分成左半和右半，计算差异度
        left_half = gray[:, : roi_w // 2]
        right_half = gray[:, roi_w // 2 :]

        # 方法A：均值差异（归一化）
        left_mean = np.mean(left_half)
        right_mean = np.mean(right_half)
        mean_diff = abs(left_mean - right_mean) / 255.0

        # 方法B：标准差差异
        left_std = np.std(left_half)
        right_std = np.std(right_half)
        std_diff = abs(left_std - right_std) / 255.0

        # 方法C：直方图相关性（1=完全相同，0=完全不同）
        hist_left = cv2.calcHist([left_half], [0], None, [32], [0, 256])
        hist_right = cv2.calcHist([right_half], [0], None, [32], [0, 256])
        hist_left = cv2.normalize(hist_left, hist_left).flatten()
        hist_right = cv2.normalize(hist_right, hist_right).flatten()
        hist_correl = cv2.compareHist(hist_left, hist_right, cv2.HISTCMP_CORREL)
        hist_diff = 1.0 - max(0.0, hist_correl)  # 转成差异度

        # 综合不对称性分数（0~1，越大越不对称）
        asymmetry = max(mean_diff, std_diff, hist_diff)

        # 3. 列方向（垂直方向）亮度方差 —— 空挡核心特征
        # 空挡：中央有缝隙暗带+金属反光，每列亮度差异大
        # 无空挡：连续波纹板，每列亮度接近
        col_profile = np.mean(gray, axis=0)
        col_variance = float(np.std(col_profile))
        col_max = float(np.max(col_profile))
        col_min = float(np.min(col_profile))

        return {
            "mean_brightness": mean_brightness,
            "std_brightness": std_brightness,
            "edge_score": edge_score,
            "profile_variance": profile_variance,
            "profile_min": profile_min,
            "asymmetry": asymmetry,
            "mean_diff": mean_diff,
            "std_diff": std_diff,
            "hist_diff": hist_diff,
            "col_variance": col_variance,
            "col_max": col_max,
            "col_min": col_min,
        }

    def detect(self, image):
        if isinstance(image, str):
            img = self._imread(image)
            if img is None:
                raise ValueError(f"无法读取图像: {image}")
        else:
            img = image

        features = self._extract_features(img)

        # v3 新增：中心车钩反光检测（基于全图）
        # 纯车钩空挡的核心特征：中间窄带有强光反射
        gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray_full.shape
        narrow_cx1, narrow_cx2 = int(w * 0.45), int(w * 0.55)
        narrow_band = gray_full[:, narrow_cx1:narrow_cx2]
        narrow_mean = np.mean(narrow_band)
        full_mean = np.mean(gray_full)
        narrow_bright = np.sum(narrow_band > 200) / narrow_band.size
        wide_cx1, wide_cx2 = int(w * 0.35), int(w * 0.65)
        wide_band = gray_full[:, wide_cx1:wide_cx2]
        col_var = np.var(np.mean(wide_band, axis=0))

        coupler_score = 0.0
        if full_mean > 0:
            brightness_ratio = narrow_mean / full_mean
            if brightness_ratio > 1.55 and 0.015 <= narrow_bright <= 0.08 and col_var < 130:
                coupler_score = 1.0
            elif brightness_ratio > 1.54 and narrow_bright < 0.04 and col_var < 130:
                coupler_score = 0.6

        features["coupler_score"] = coupler_score
        features["brightness_ratio"] = narrow_mean / full_mean if full_mean > 0 else 0.0

        # 8个主特征评分
        score = 0
        if features["mean_brightness"] > self.brightness_thresh:
            score += 1
        if features["std_brightness"] > self.std_thresh:
            score += 1
        if features["edge_score"] > self.edge_thresh:
            score += 1
        if features["profile_variance"] > self.profile_var_thresh:
            score += 1
        if features["profile_min"] > self.profile_min_thresh:
            score += 1
        if features["asymmetry"] > self.asymmetry_thresh:
            score += 1
        if features["col_variance"] > 8.0:
            score += 1
        col_range = features["col_max"] - features["col_min"]
        if col_range > 48.0:
            score += 1

        # v3: 车钩反光作为辅助判定
        # 如果主评分接近阈值但车钩特征强，额外加分
        if coupler_score >= 1.0:
            score += 1
        elif coupler_score >= 0.6 and score >= 5:
            score += 1

        is_gap = score >= self.min_gap_score
        confidence = score / 9.0  # 满分9分（8主+1车钩）
        return is_gap, confidence, features

    def visualize(self, image, is_gap, confidence, save_path=None):
        img = self._imread(image) if isinstance(image, str) else image.copy()
        h, w = img.shape[:2]
        x1, x2 = int(w * self.roi_x_ratio[0]), int(w * self.roi_x_ratio[1])
        y1, y2 = int(h * self.roi_y_ratio[0]), int(h * self.roi_y_ratio[1])
        color = (0, 0, 255) if is_gap else (0, 255, 0)
        label = f"GAP ({confidence:.2f})" if is_gap else f"NO GAP ({confidence:.2f})"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, label, (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        if save_path:
            ret, buf = cv2.imencode(".jpg", img)
            if ret:
                with open(save_path, "wb") as f:
                    f.write(buf)
        return img

    def detect_video_frames(self, frames_dir, output_dir=None, sample_interval=1):
        frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
        results = []
        for i, frame_path in enumerate(frame_files):
            if i % sample_interval != 0:
                continue
            is_gap, confidence, features = self.detect(frame_path)
            frame_name = os.path.basename(frame_path)
            results.append({
                "frame": frame_name,
                "is_gap": is_gap,
                "confidence": confidence,
                "features": features,
            })
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                vis_path = os.path.join(output_dir, f"vis_{frame_name}")
                self.visualize(frame_path, is_gap, confidence, save_path=vis_path)
        return results

    def calibrate(self, gap_samples, no_gap_samples):
        gap_features = [self._extract_features(self._imread(p)) for p in gap_samples]
        no_gap_features = [self._extract_features(self._imread(p)) for p in no_gap_samples]
        print(f"校准: 空挡{len(gap_features)}张, 无空挡{len(no_gap_features)}张")
        keys = ["mean_brightness", "std_brightness", "edge_score", "profile_variance", "profile_min", "asymmetry", "col_variance"]
        for key in keys:
            g_vals = [f[key] for f in gap_features]
            n_vals = [f[key] for f in no_gap_features]
            print(f"{key}: 空挡[{min(g_vals):.2f},{max(g_vals):.2f}] vs 无空挡[{min(n_vals):.2f},{max(n_vals):.2f}]")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="板车空挡检测")
    parser.add_argument("--frames", required=True, help="帧图片目录")
    parser.add_argument("--output", default=None, help="可视化输出目录")
    parser.add_argument("--interval", type=int, default=1, help="采样间隔")
    args = parser.parse_args()

    detector = FlatcarGapDetector()
    results = detector.detect_video_frames(
        args.frames, output_dir=args.output, sample_interval=args.interval
    )
    gap_count = sum(1 for r in results if r["is_gap"])
    print(f"\n检测完成: 共{len(results)}帧, 空挡{gap_count}帧, 无空挡{len(results)-gap_count}帧")
