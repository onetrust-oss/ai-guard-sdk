import pytest

from ai_guard.client.chunker import LineChunker


class TestLineChunker:
    def test_empty_buffer_returns_none(self):
        chunker = LineChunker(size=100)
        assert chunker.next() is None

    def test_no_line_break_under_size_returns_none(self):
        chunker = LineChunker(size=100)
        chunker.append("hello world")
        assert chunker.next() is None

    def test_single_line_with_newline(self):
        chunker = LineChunker(size=100)
        chunker.append("hello world\n")
        assert chunker.next() == "hello world\n"
        assert chunker.next() is None

    def test_single_line_with_crlf(self):
        chunker = LineChunker(size=100)
        chunker.append("hello world\r\n")
        assert chunker.next() == "hello world\r\n"
        assert chunker.next() is None

    def test_single_line_with_cr(self):
        chunker = LineChunker(size=100)
        chunker.append("hello world\r")
        assert chunker.next() == "hello world\r"
        assert chunker.next() is None

    def test_multiple_lines_under_size(self):
        chunker = LineChunker(size=100)
        chunker.append("line1\nline2\nline3\n")
        assert chunker.next() == "line1\nline2\nline3\n"
        assert chunker.next() is None

    def test_chunk_at_line_break_when_over_size(self):
        chunker = LineChunker(size=10)
        chunker.append("hello\nworld\n")
        assert chunker.next() == "hello\n"
        assert chunker.next() == "world\n"
        assert chunker.next() is None

    def test_break_at_max_size_when_no_line_break(self):
        chunker = LineChunker(size=5)
        chunker.append("helloworld\n")
        assert chunker.next() == "hello"
        assert chunker.next() == "world"
        assert chunker.next() == "\n"
        assert chunker.next() is None

    def test_preserves_line_breaks_in_output(self):
        chunker = LineChunker(size=20)
        chunker.append("line1\r\nline2\nline3\r")
        chunk = chunker.next()
        assert "\r\n" in chunk or "\n" in chunk or "\r" in chunk

    def test_mixed_line_breaks(self):
        chunker = LineChunker(size=10)
        chunker.append("a\r\nb\nc\r")
        assert chunker.next() == "a\r\nb\nc\r"

    def test_append_multiple_times(self):
        chunker = LineChunker(size=100)
        chunker.append("hello ")
        chunker.append("world\n")
        assert chunker.next() == "hello world\n"

    def test_incremental_append_and_next(self):
        chunker = LineChunker(size=10)
        chunker.append("hello")
        assert chunker.next() is None
        chunker.append("\nworld\n")
        assert chunker.next() == "hello\n"
        assert chunker.next() == "world\n"

    def test_long_line_without_linebreak_waits(self):
        # Without a line break and under/at size, next() waits for more data
        chunker = LineChunker(size=5)
        chunker.append("abcdefghij")
        assert chunker.next() == "abcde"
        # Remaining "fghij" is exactly at size but no line break - waits for more
        assert chunker.next() is None
        # Add more data with line break - buffer becomes "fghij\n" (6 chars > 5)
        # No line break in first 5 chars, so breaks at max size
        chunker.append("\n")
        assert chunker.next() == "fghij"
        assert chunker.next() == "\n"
        assert chunker.next() is None

    def test_exact_size_with_line_break(self):
        chunker = LineChunker(size=6)
        chunker.append("hello\nworld\n")
        assert chunker.next() == "hello\n"
        assert chunker.next() == "world\n"

    def test_crlf_not_split(self):
        chunker = LineChunker(size=7)
        chunker.append("hello\r\nworld\r\n")
        assert chunker.next() == "hello\r\n"
        assert chunker.next() == "world\r\n"

    def test_size_property(self):
        chunker = LineChunker(size=42)
        assert chunker._size == 42

    def test_drain_flushes_remaining(self):
        chunker = LineChunker(size=100)
        chunker.append("no newline here")
        assert chunker.next() is None
        chunker.append(None)  # drain
        assert chunker.next() == "no newline here"
        assert chunker.next() is None

    def test_drain_with_empty_pending(self):
        chunker = LineChunker(size=100)
        chunker.append("line\n")
        assert chunker.next() == "line\n"
        chunker.append(None)  # drain with nothing pending
        assert chunker.next() is None

    def test_append_after_drain_raises(self):
        chunker = LineChunker(size=100)
        chunker.append("hello")
        chunker.append(None)
        with pytest.raises(ValueError, match="Cannot append after drain"):
            chunker.append("more data")

    def test_drain_partial_line_at_size_boundary(self):
        chunker = LineChunker(size=5)
        chunker.append("abcde")
        assert chunker.next() is None  # waiting for line break
        chunker.append(None)  # drain
        assert chunker.next() == "abcde"
        assert chunker.next() is None
