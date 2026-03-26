import logging
from typing import Optional

logger = logging.getLogger(__name__)

class LineChunker:
    def __init__(
        self,
        size: int,
    ):
        self._size = size
        self._lines: list[str] = []
        self._partial = ""
        self._drained = False

    def next(self) -> Optional[str]:
        if not self._lines:
            return None

        chunk = ""
        while self._lines:
            line = self._lines[0]
            if not chunk:
                # First line - must take at least part of it
                if len(line) <= self._size:
                    chunk = line
                    self._lines.pop(0)
                else:
                    # Line too long, take max size and trim
                    chunk = line[:self._size]
                    self._lines[0] = line[self._size:]
                    return chunk
            elif len(chunk) + len(line) <= self._size:
                # Can fit this line in current chunk
                chunk += line
                self._lines.pop(0)
            else:
                # Can't fit, return what we have
                break

        return chunk if chunk else None

    def append(self, text: Optional[str]):
        if text is None:
            self._drained = True
            if self._partial:
                self._lines.append(self._partial)
                self._partial = ""
            return
        if self._drained:
            raise ValueError("Cannot append after drain")

        text = self._partial + text
        self._partial = ""

        lines = text.splitlines(keepends=True)

        if not lines:
            return

        if lines and not lines[-1].endswith(('\n', '\r')):
            self._partial = lines.pop()

        self._lines.extend(lines)

        while len(self._partial) > self._size:
            self._lines.append(self._partial[:self._size])
            self._partial = self._partial[self._size:]