from __future__ import annotations
from typing import Literal
import cv2
import numpy as np


class ImageProcessor:
    """Pure OpenCV operations (no GUI)."""

    @staticmethod
    def _require(img: np.ndarray) -> None:
        if img is None or not isinstance(img, np.ndarray):
            raise ValueError("No image loaded.")

    @staticmethod
    def grayscale(img_bgr: np.ndarray) -> np.ndarray:
        ImageProcessor._require(img_bgr)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def blur(img_bgr: np.ndarray, intensity: int) -> np.ndarray:
        ImageProcessor._require(img_bgr)
        if intensity < 1:
            intensity = 1
        if intensity > 31:
            intensity = 31
        # kernel must be odd
        k = intensity if intensity % 2 == 1 else intensity + 1
        return cv2.GaussianBlur(img_bgr, (k, k), 0)

    @staticmethod
    def edge_detect(img_bgr: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
        ImageProcessor._require(img_bgr)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, low, high)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def adjust_brightness(img_bgr: np.ndarray, value: int) -> np.ndarray:
        ImageProcessor._require(img_bgr)
        if value < -100:
            value = -100
        if value > 100:
            value = 100
        # beta shifts brightness
        return cv2.convertScaleAbs(img_bgr, alpha=1.0, beta=value)

    @staticmethod
    def adjust_contrast(img_bgr: np.ndarray, value: int) -> np.ndarray:
        ImageProcessor._require(img_bgr)
        if value < -100:
            value = -100
        if value > 100:
            value = 100
        # Map [-100..100] -> alpha [0.5..1.5]
        alpha = 1.0 + (value / 200.0)
        return cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=0)

    @staticmethod
    def rotate(img_bgr: np.ndarray, degrees: Literal[90, 180, 270]) -> np.ndarray:
        ImageProcessor._require(img_bgr)
        if degrees == 90:
            return cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
        if degrees == 180:
            return cv2.rotate(img_bgr, cv2.ROTATE_180)
        if degrees == 270:
            return cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        raise ValueError("Rotation must be 90, 180, or 270 degrees.")

    @staticmethod
    def flip(img_bgr: np.ndarray, mode: Literal["horizontal", "vertical"]) -> np.ndarray:
        ImageProcessor._require(img_bgr)
        if mode == "horizontal":
            return cv2.flip(img_bgr, 1)
        if mode == "vertical":
            return cv2.flip(img_bgr, 0)
        raise ValueError("Flip mode must be 'horizontal' or 'vertical'.")

    @staticmethod
    def resize_percent(img_bgr: np.ndarray, scale_percent: int) -> np.ndarray:
        ImageProcessor._require(img_bgr)
        if scale_percent < 10:
            scale_percent = 10
        if scale_percent > 200:
            scale_percent = 200
        h, w = img_bgr.shape[:2]
        new_w = max(1, int(w * scale_percent / 100.0))
        new_h = max(1, int(h * scale_percent / 100.0))
        return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
