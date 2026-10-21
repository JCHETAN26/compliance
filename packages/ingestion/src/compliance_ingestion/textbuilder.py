"""Accumulates canonical document text together with a per-character box index."""

from __future__ import annotations

from dataclasses import dataclass, field

# (page, x0, y0, x1, y1); None for synthetic separators and characters without geometry.
Box = tuple[int, float, float, float, float]
CharBox = Box | None

BLOCK_SEPARATOR = "\n\n"
LINE_SEPARATOR = "\n"


@dataclass
class TextBuilder:
    parts: list[str] = field(default_factory=list)
    boxes: list[CharBox] = field(default_factory=list)
    length: int = 0
    _tail: str = ""

    def append(self, text: str, box: CharBox = None) -> None:
        if not text:
            return
        self.parts.append(text)
        self.boxes.extend([box] * len(text))
        self.length += len(text)
        self._tail = (self._tail + text)[-2:]

    def append_char(self, char: str, box: CharBox) -> None:
        self.parts.append(char)
        self.boxes.append(box)
        self.length += 1
        self._tail = (self._tail + char)[-2:]

    def separate(self, separator: str) -> None:
        """Make the text end with ``separator`` (no-op at the start of the document)."""
        if self.length == 0 or self._tail.endswith(separator):
            return
        if separator == BLOCK_SEPARATOR and self._tail.endswith(LINE_SEPARATOR):
            self.append(LINE_SEPARATOR)
            return
        self.append(separator)

    def text(self) -> str:
        return "".join(self.parts)
