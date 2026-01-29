from __future__ import annotations
from typing import Optional, Tuple
import os
import cv2
import numpy as np


class ImageModel:
    """
    Holds image data and file state.
    Encapsulation: internal fields are private and accessed through methods.
    """
    SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

    def __init__(self):
        self._original_bgr: Optional[np.ndarray] = None
        self._current_bgr: Optional[np.ndarray] = None
        self._file_path: Optional[str] = None

    def has_image(self) -> bool:
        return self._current_bgr is not None

    def file_path(self) -> Optional[str]:
        return self._file_path

    def filename(self) -> str:
        return os.path.basename(self._file_path) if self._file_path else "Untitled"

    def current(self) -> Optional[np.ndarray]:
        return None if self._current_bgr is None else self._current_bgr.copy()

    def set_current(self, img_bgr: np.ndarray) -> None:
        if img_bgr is None or not isinstance(img_bgr, np.ndarray):
            raise ValueError("Invalid image data.")
        self._current_bgr = img_bgr.copy()

    def load(self, path: str) -> None:
        if not path:
            raise ValueError("No file path provided.")
        ext = os.path.splitext(path)[1].lower()
        if ext not in self.SUPPORTED_EXTS:
            raise ValueError(f"Unsupported format: {ext}. Use JPG/PNG/BMP.")

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not read image file. File may be corrupted or unsupported.")

        self._file_path = path
        self._original_bgr = img.copy()
        self._current_bgr = img.copy()

    def reset_to_original(self) -> None:
        if self._original_bgr is None:
            raise ValueError("No original image to reset to.")
        self._current_bgr = self._original_bgr.copy()

    def set_file_path(self, path: str) -> None:
        self._file_path = path

    def get_dimensions(self) -> Tuple[int, int]:
        if self._current_bgr is None:
            return (0, 0)
        h, w = self._current_bgr.shape[:2]
        return (w, h)
