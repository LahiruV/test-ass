from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import numpy as np


@dataclass
class EditorState:
    """
    Immutable snapshot of editor state for undo and redo.

    Stores both image data and UI values so visual controls
    can be restored exactly when navigating history.
    """
    image_bgr: np.ndarray
    file_path: Optional[str]
    ui: Dict[str, Any]  


class HistoryManager:
    """
    Manages undo and redo stacks for the image editor.

    undo: past states
    redo: states undone and available to restore
    """
    def __init__(self, max_states: int = 30):
        """
        Initialise history manager.

        Args:
            max_states: Maximum number of undo states to retain.
        """
        self.max_states = max_states
        self._undo: List[EditorState] = []
        self._redo: List[EditorState] = []

    def clear(self) -> None:
        """Clear all undo and redo history."""
        self._undo.clear()
        self._redo.clear()

    def can_undo(self) -> bool:
        """Return True if an undo operation is possible."""
        return len(self._undo) > 0

    def can_redo(self) -> bool:
        """Return True if a redo operation is possible."""
        return len(self._redo) > 0

    def push(self, state: EditorState) -> None:
        """
        Push a new editor state onto the undo stack.

        Pushing a new state invalidates the redo stack.
        """
        self._undo.append(state)
        # Enforce maximum history size by discarding the oldest state.
        if len(self._undo) > self.max_states:
            self._undo.pop(0)
        self._redo.clear()

    def undo(self, current_state: EditorState) -> Optional[EditorState]:
        """
        Revert to the previous editor state.

        Args:
            current_state: The current state before undo.

        Returns:
            The previous state, or None if undo is not possible.
        """
        if not self.can_undo():
            return None
        prev = self._undo.pop()
        self._redo.append(current_state)
        return prev

    def redo(self, current_state: EditorState) -> Optional[EditorState]:
        """
        Reapply a previously undone editor state.

        Args:
            current_state: The current state before redo.

        Returns:
            The next state, or None if redo is not possible.
        """
        if not self.can_redo():
            return None
        nxt = self._redo.pop()
        self._undo.append(current_state)
        return nxt
