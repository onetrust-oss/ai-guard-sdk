import logging
import queue as _queue
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock, Thread

from ai_guard.api import (
    ClassificationRequest,
    ClassificationRequestMetadata,
    ClassificationResponse,
    ClassifierDescription,
)
from ai_guard.client.chunker import LineChunker
from ai_guard.client.client import AIGuardClient
from ai_guard.redact import ClassificationRedactor, Redaction, RedactKind

logger = logging.getLogger(__name__)


@dataclass
class ClassificationStreamResult:
    """A single result yielded by :class:`ClassificationStream`.

    Attributes:
        text: The chunk text, redacted if a redactor was provided.
        response: Classification response for this chunk.
        redaction: Redaction result, or ``None`` when no redactor is attached.
    """

    text: str
    response: ClassificationResponse
    redaction: Redaction | None


class ClassificationStream:
    """Streaming classification over an iterable of text chunks.

    Buffers incoming text into line-aware chunks, classifies them
    concurrently, and yields :class:`ClassificationStreamResult` objects
    in input order.  An optional :class:`ClassificationRedactor` applies
    redaction inline.

    Args:
        classifier_description: Which classifiers to use.
        client: :class:`AIGuardClient` used to send classification requests.
        input: Any iterable of strings (generator, list, file, etc.).
        context: Request context; must include ``"actor"`` (``"user"`` or
            ``"agent"``).
        redactor: Optional redactor for inline redaction of results.
        chunk_size: Max characters per classification chunk.
        max_workers: Thread pool size for concurrent classification.
    """

    def __init__(
        self,
        classifier_description: ClassifierDescription,
        client: AIGuardClient,
        input: Iterable[str],
        context: dict[str, str] | None = None,
        redactor: ClassificationRedactor | None = None,
        chunk_size: int | None = 100,
        max_workers: int | None = 4,
    ):
        self._context = context
        self._classifier_description = classifier_description
        self._client = client
        self._input = input
        self._redactor = redactor
        self._chunker = LineChunker(size=chunk_size)
        self._stopped = False
        self._task = 0
        self._result: dict[int, ClassificationStreamResult] = {}
        self._chunks: dict[int, str] = {}
        self._futures: dict[int, Future] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._next_to_write = 0
        self._lock = Lock()
        self._char_offset = 0
        self._offsets: dict[int, int] = {}
        self._output_queue: _queue.Queue[ClassificationStreamResult | None] = (
            _queue.Queue()
        )

    def __iter__(self) -> Iterator[ClassificationStreamResult]:
        def _consume():
            try:
                for text in self._input:
                    if self._stopped:
                        break
                    self._append(text)
                self._finalize()
            finally:
                self._output_queue.put(None)

        thread = Thread(target=_consume)
        thread.start()

        while (result := self._output_queue.get()) is not None:
            yield result

        thread.join()

    def _classify_chunk(
        self,
        task: int,
        chunk: str,
        offset: int,
        metadata: ClassificationRequestMetadata | None,
    ) -> tuple[int, str, int, ClassificationResponse, Redaction | None]:
        req = ClassificationRequest(
            context=self._context,
            classifier_description=self._classifier_description,
            text=chunk,
            metadata=metadata,
        )
        response = self._client.classify(req)
        redacted = None
        if self._redactor is not None:
            redacted = self._redactor.redact(chunk, response)
            if any(action.kind == RedactKind.BLOCK for action in redacted.actions):
                self._stopped = True
        return task, chunk, offset, response, redacted

    def _on_complete(
        self,
        task: int,
        chunk: str,
        offset: int,
        response: ClassificationResponse,
        redacted: Redaction | None,
    ):
        for match in response.matches:
            match.start += offset
            match.end += offset
        with self._lock:
            text = redacted.text if redacted is not None else chunk
            result = ClassificationStreamResult(
                text=text, response=response, redaction=redacted
            )
            self._result[task] = result
            self._chunks[task] = chunk
            self._try_write()

    def _try_write(self):
        while self._next_to_write in self._result:
            result = self._result[self._next_to_write]
            self._output_queue.put(result)
            self._next_to_write += 1

    def _submit_chunk(
        self, chunk: str, metadata: ClassificationRequestMetadata | None
    ) -> None:
        task = self._task
        offset = self._char_offset
        self._task += 1
        self._char_offset += len(chunk)
        self._offsets[task] = offset
        future = self._executor.submit(
            self._classify_chunk, task, chunk, offset, metadata
        )
        future.add_done_callback(lambda f: self._handle_future_done(f))
        self._futures[task] = future

    def _handle_future_done(self, future: Future):
        task, chunk, offset, response, redacted = future.result()
        self._on_complete(task, chunk, offset, response, redacted)

    def _append(
        self, text: str, metadata: ClassificationRequestMetadata | None = None
    ) -> bool:
        if self._stopped:
            return False
        self._chunker.append(text)
        while chunk := self._chunker.next():
            if self._stopped:
                return False
            self._submit_chunk(chunk, metadata)
        return not self._stopped

    def _finalize(self, metadata: ClassificationRequestMetadata | None = None) -> None:
        self._chunker.append(None)
        while chunk := self._chunker.next():
            if self._stopped:
                break
            self._submit_chunk(chunk, metadata)

        for task_id in sorted(self._futures.keys()):
            if self._stopped:
                break
            self._futures[task_id].result()

        with self._lock:
            self._try_write()

        self._executor.shutdown(wait=False)
