from __future__ import annotations
from typing import Optional, Tuple
import os
import cv2
import numpy as np


class ImageModel:
    """
    Stores and manages image state for the editor.

    Load images from disk
    Preserve original image for reset
    Provide safe access to the current working image
    Images are stored internally in BGR format (OpenCV default)
    All getters return copies to preserve encapsulation
    """

    SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

    def __init__(self):
        """
        Initialise an empty image model with no loaded image.
        """
        self._original_bgr: Optional[np.ndarray] = None
        self._current_bgr: Optional[np.ndarray] = None
        self._file_path: Optional[str] = None

    def has_image(self) -> bool:
        """
        Check whether an image is currently loaded.
        """
        return self._current_bgr is not None

    def file_path(self) -> Optional[str]:
        """
        Return the full file path of the loaded image, if any.
        """
        return self._file_path

    def filename(self) -> str:
        """
        Return the image filename for display purposes.
        """
        return os.path.basename(self._file_path) if self._file_path else "Untitled"

    def current(self) -> Optional[np.ndarray]:
        """
        Get a copy of the current working image.

        A copy is returned to prevent accidental modification
        of internal state by other components.
        """
        return None if self._current_bgr is None else self._current_bgr.copy()

    def set_current(self, img_bgr: np.ndarray) -> None:
        """
        Replace the current working image.

        Raises:
            ValueError: If the provided image is invalid.
        """
        if img_bgr is None or not isinstance(img_bgr, np.ndarray):
            raise ValueError("Invalid image data.")
        self._current_bgr = img_bgr.copy()

    def load(self, path: str) -> None:
        """
        Load an image from disk and initialise editor state.

        The loaded image is stored as both original and current versions

        Raises:
            ValueError: If the file path, format, or image data is invalid.
        """
        if not path:
            raise ValueError("No file path provided.")

        ext = os.path.splitext(path)[1].lower()
        if ext not in self.SUPPORTED_EXTS:
            raise ValueError(f"Unsupported format: {ext}. Use JPG/PNG/BMP.")

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not read image file.")
        # Keep original immutable , current mutable 
        self._file_path = path
        self._original_bgr = img.copy()
        self._current_bgr = img.copy()

    def reset_to_original(self) -> None:
        """
        Restore the image to its original loaded state.
        """
        if self._original_bgr is None:
            raise ValueError("No original image to reset to.")
        self._current_bgr = self._original_bgr.copy()

    def set_file_path(self, path: str) -> None:
        """
        Update the stored file path.
        """
        self._file_path = path

    def get_dimensions(self) -> Tuple[int, int]:
        """
        Return the current image dimensions.
        """
        if self._current_bgr is None:
            return (0, 0)

        h, w = self._current_bgr.shape[:2]
        return (w, h)
