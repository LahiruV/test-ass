from __future__ import annotations
from typing import Literal
import cv2
import numpy as np


class ImageProcessor:
    """Pure OpenCV operations"""

    @staticmethod
    def _require(img: np.ndarray) -> None:
        """
        Validate that an image is present and correctly typed.

        Args:
            img: Expected OpenCV image.

        Raises:
            ValueError: If `img` is missing or not a numpy array.
        """
        if img is None or not isinstance(img, np.ndarray):
            raise ValueError("No image loaded.")

    @staticmethod
    def grayscale(img_bgr: np.ndarray) -> np.ndarray:
        """
        Convert an image to grayscale and return as 3-channel BGR.

        Keeping BGR output simplifies GUI rendering and avoids branching
        in code that assumes 3 channels.
        """
        ImageProcessor._require(img_bgr)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def blur(img_bgr: np.ndarray, intensity: int) -> np.ndarray:
        """
        Apply Gaussian blur using an intensity-based kernel size.

        Args:
            img_bgr: Source BGR image.
            intensity: Intended kernel size.

        Returns:
            Blurred BGR image.
        """
        ImageProcessor._require(img_bgr)
        if intensity < 1:
            intensity = 1
        if intensity > 31:
            intensity = 31
            
        # GaussianBlur requires an odd kernel size; bump to next odd if needed.
        k = intensity if intensity % 2 == 1 else intensity + 1
        return cv2.GaussianBlur(img_bgr, (k, k), 0)

    @staticmethod
    def edge_detect(img_bgr: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
        """
        Perform Canny edge detection and return as 3 channel BGR.

        Args:
            img_bgr: Source BGR image.
            low: Lower threshold for hysteresis.
            high: Upper threshold for hysteresis.

        Returns:
            Edge map converted back to BGR for consistent display.
        """
        ImageProcessor._require(img_bgr)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, low, high)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def adjust_brightness(img_bgr: np.ndarray, value: int) -> np.ndarray:
        """
        Adjust brightness by shifting pixel intensities.

        Args:
            img_bgr: Source BGR image.
            value: Brightness shift (clamped to -100..100).

        Returns:
            Brightness-adjusted BGR image.
        """
        ImageProcessor._require(img_bgr)
        if value < -100:
            value = -100
        if value > 100:
            value = 100
        # beta shifts brightness
        return cv2.convertScaleAbs(img_bgr, alpha=1.0, beta=value)

    @staticmethod
    def adjust_contrast(img_bgr: np.ndarray, value: int) -> np.ndarray:
        """
        Adjust contrast by scaling pixel intensities.

        Args:
            img_bgr: Source BGR image.
            value: Contrast adjustment (clamped to -100..100).

        Returns:
            Contrast-adjusted BGR image.
        """
        ImageProcessor._require(img_bgr)
        if value < -100:
            value = -100
        if value > 100:
            value = 100
        # Map [-100..100] -> alpha [0.5..1.5] for a simple, intuitive contrast scale.
        alpha = 1.0 + (value / 200.0)
        return cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=0)

    @staticmethod
    def rotate(img_bgr: np.ndarray, degrees: Literal[90, 180, 270]) -> np.ndarray:
        """
        Rotate the image by fixed right angle increments.

        Args:
            img_bgr: Source BGR image.
            degrees: One of 90, 180, 270.

        Returns:
            Rotated BGR image.

        Raises:
            ValueError: If degrees is not 90/180/270.
        """
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
        """
        Flip an image horizontally or vertically.

        Args:
            img_bgr: Source BGR image.
            mode: "horizontal" or "vertical".

        Returns:
            Flipped BGR image.

        Raises:
            ValueError: If mode is invalid.
        """
        ImageProcessor._require(img_bgr)
        if mode == "horizontal":
            return cv2.flip(img_bgr, 1)
        if mode == "vertical":
            return cv2.flip(img_bgr, 0)
        raise ValueError("Flip mode must be 'horizontal' or 'vertical'.")

    @staticmethod
    def resize_percent(img_bgr: np.ndarray, scale_percent: int) -> np.ndarray:
        """
        Resize the image by a percentage scale.

        Args:
            img_bgr: Source BGR image.
            scale_percent: Percent scale (10 to 200).

        Returns:
            Resized BGR image.
        """
        ImageProcessor._require(img_bgr)
        if scale_percent < 10:
            scale_percent = 10
        if scale_percent > 200:
            scale_percent = 200
        h, w = img_bgr.shape[:2]
        new_w = max(1, int(w * scale_percent / 100.0))
        new_h = max(1, int(h * scale_percent / 100.0))
        # INTER_AREA is generally best for downscaling; fine for modest upscales too.
        return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
