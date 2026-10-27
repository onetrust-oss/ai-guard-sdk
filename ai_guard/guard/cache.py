import threading
import time
from dataclasses import dataclass


@dataclass
class _Entry:
    guardrails: list  # empty list is an authoritative "no guardrails" result
    resolved_at: float


class GuardrailCache:
    """Sticky, thread-safe, TTL cache with per-key single-flight locks.

    Entries are keyed by ``(agent_id, account, region)``. The cache is only
    ever overwritten by a successful response (:meth:`store`); it never evicts
    on failure, so once an agent has a good list a later resolver outage keeps
    serving the last-known-good list.
    """

    def __init__(self, ttl_seconds: int):
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._entries: dict[tuple, _Entry] = {}
        self._key_locks: dict[tuple, threading.Lock] = {}

    def lookup(self, key: tuple) -> tuple[_Entry | None, str]:
        with self._lock:
            e = self._entries.get(key)
        if e is None:
            return None, "miss"
        fresh = (time.monotonic() - e.resolved_at) < self._ttl
        return e, ("fresh" if fresh else "stale")

    def store(self, key: tuple, guardrails: list) -> None:
        with self._lock:  # overwrite only on success; never evict on failure
            self._entries[key] = _Entry(guardrails, time.monotonic())

    def key_lock(self, key: tuple) -> threading.Lock:
        with self._lock:
            return self._key_locks.setdefault(key, threading.Lock())
