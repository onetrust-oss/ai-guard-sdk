import logging
import queue as _queue
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
from threading import Lock, Thread
from typing import Optional, List, Dict, Tuple, Iterable, Iterator

from ai_guard.client.client import AIGuardClient
from ai_guard.api import ClassificationResponse, ClassificationRequest, \
    ClassificationRequestMetadata, ClassifierDescription
from ai_guard.client.chunker import LineChunker
from ai_guard.redact import ClassificationRedactor, Redaction, RedactKind

logger = logging.getLogger(__name__)


@dataclass
class ClassificationStreamResult:
    text: str
    response: ClassificationResponse
    redaction: Optional[Redaction]


class ClassificationStream:
    def __init__(
            self,
            classifier_description: ClassifierDescription,
            client: AIGuardClient,
            input: Iterable[str],
            context: Optional[Dict[str, str]] = None,
            redactor: Optional[ClassificationRedactor] = None,
            chunk_size: Optional[int] = 100,
            max_workers: Optional[int] = 4,
    ):
        self._context = context
        self._classifier_description = classifier_description
        self._client = client
        self._input = input
        self._redactor = redactor
        self._chunker = LineChunker(size=chunk_size)
        self._stopped = False
        self._task = 0
        self._result: Dict[int, ClassificationStreamResult] = {}
        self._chunks: Dict[int, str] = {}
        self._futures: Dict[int, Future] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._next_to_write = 0
        self._lock = Lock()
        self._char_offset = 0
        self._offsets: Dict[int, int] = {}
        self._output_queue: _queue.Queue[Optional[ClassificationStreamResult]] = _queue.Queue()

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
            metadata: Optional[ClassificationRequestMetadata],
    ) -> Tuple[int, str, int, ClassificationResponse, Optional[Redaction]]:
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

    def _on_complete(self, task: int, chunk: str, offset: int, response: ClassificationResponse,
                     redacted: Optional[Redaction]):
        for match in response.matches:
            match.start += offset
            match.end += offset
        with self._lock:
            if redacted is not None:
                text = redacted.text
            else:
                text = chunk
            result = ClassificationStreamResult(text=text, response=response, redaction=redacted)
            self._result[task] = result
            self._chunks[task] = chunk
            self._try_write()

    def _try_write(self):
        while self._next_to_write in self._result:
            result = self._result[self._next_to_write]
            self._output_queue.put(result)
            self._next_to_write += 1

    def _submit_chunk(self, chunk: str, metadata: Optional[ClassificationRequestMetadata]) -> None:
        task = self._task
        offset = self._char_offset
        self._task += 1
        self._char_offset += len(chunk)
        self._offsets[task] = offset
        future = self._executor.submit(self._classify_chunk, task, chunk, offset, metadata)
        future.add_done_callback(lambda f: self._handle_future_done(f))
        self._futures[task] = future

    def _handle_future_done(self, future: Future):
        task, chunk, offset, response, redacted = future.result()
        self._on_complete(task, chunk, offset, response, redacted)

    def _append(self, text: str, metadata: Optional[ClassificationRequestMetadata] = None) -> bool:
        if self._stopped:
            return False
        self._chunker.append(text)
        while chunk := self._chunker.next():
            if self._stopped:
                return False
            self._submit_chunk(chunk, metadata)
        return not self._stopped

    def _finalize(self, metadata: Optional[ClassificationRequestMetadata] = None) -> None:
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
