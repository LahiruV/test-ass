from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import numpy as np


@dataclass
class EditorState:
    """A snapshot of editor state for Undo/Redo."""
    image_bgr: np.ndarray
    file_path: Optional[str]
    ui: Dict[str, Any]  # slider + UI values (blur, brightness, contrast, scale, etc.)


class HistoryManager:
    """Manages undo/redo history stacks."""
    def __init__(self, max_states: int = 30):
        self.max_states = max_states
        self._undo: List[EditorState] = []
        self._redo: List[EditorState] = []

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()

    def can_undo(self) -> bool:
        return len(self._undo) > 0

    def can_redo(self) -> bool:
        return len(self._redo) > 0

    def push(self, state: EditorState) -> None:
        """Push a new state and clear redo stack."""
        self._undo.append(state)
        if len(self._undo) > self.max_states:
            self._undo.pop(0)
        self._redo.clear()

    def undo(self, current_state: EditorState) -> Optional[EditorState]:
        """Move one step back. Returns previous state or None."""
        if not self.can_undo():
            return None
        prev = self._undo.pop()
        self._redo.append(current_state)
        return prev

    def redo(self, current_state: EditorState) -> Optional[EditorState]:
        """Move one step forward. Returns next state or None."""
        if not self.can_redo():
            return None
        nxt = self._redo.pop()
        self._undo.append(current_state)
        return nxt
