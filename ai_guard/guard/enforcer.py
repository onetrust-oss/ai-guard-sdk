import logging
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from ..constants import ACTION_INTERVENED, ACTION_NONE, ALLOW
from .bedrock import apply_guardrail_chunked
from .cache import GuardrailCache
from .exceptions import AIGuardConfigError, GuardrailResolutionError
from .resolver import resolve
from .result import GuardResult

if TYPE_CHECKING:
    from ai_guard.client.client import AIGuardClient

log = logging.getLogger("ai_guard.guard")


class GuardrailEnforcer:
    """Resolve (worker node, sticky-cached, single-flight) + enforce
    (``apply_guardrail`` per guardrail, composed) for the active agent.

    Owner- and agent-aware: it reuses the client's existing active-agent
    resolution, so it works unchanged for single- and multi-agent clients.
    PII-safe: no guardrail input/output/masked text is ever logged.
    """

    def __init__(self, client: "AIGuardClient"):
        self._c = client
        self._cache = GuardrailCache(client.guardrail_cache_ttl)
        self._account_cache: str | None = None
        self._brt = None  # lazily-built bedrock-runtime client
        self._sts_client = None
        # Reentrant: _account() -> _sts() both take this lock; also guards the
        # background-refresh bookkeeping below.
        self._lock = threading.RLock()
        self._last_attempt: dict[tuple, float] = {}  # key -> last resolve attempt time
        self._refreshing: set[tuple] = set()
        self._pool = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="ai-guard-refresh"
        )
        # Parallel apply_guardrail calls for chunked texts (chunk order is
        # preserved in the concatenated result).
        self._call_pool = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="ai-guard-call"
        )
        self._closed = False

    # ---- public entry points ----
    def check_input(self, text: str) -> GuardResult:
        return self._check("INPUT", text)

    def check_output(self, text: str) -> GuardResult:
        return self._check("OUTPUT", text)

    def check_output_stream(
        self, tokens: Iterable[str], emit: Callable[[str], None]
    ) -> GuardResult:
        """Guard a streaming output.

        ``tokens``: iterator of output text pieces. ``emit``: callback for
        cleared (safe) text. Buffers to ``guardrail_stream_buffer_size``,
        checks per buffer, and stops on a block. Caveat: already-emitted
        buffers cannot be recalled.
        """
        buf: list[str] = []
        size = 0
        for tok in tokens:
            buf.append(tok)
            size += len(tok)
            if size >= self._c.guardrail_stream_buffer_size:
                res = self.check_output("".join(buf))
                if res.blocked:
                    return res
                emit(res.text)
                buf, size = [], 0
        res = (
            self.check_output("".join(buf))
            if buf
            else GuardResult(False, "", ACTION_NONE)
        )
        if not res.blocked and buf:
            emit(res.text)
        return res

    def close(self) -> None:
        """Shut the background-refresh pool down. Idempotent.

        The SDK creates one enforcer per client, so hosts rarely call this
        directly (the pool's threads are daemons, so it is not required for a
        clean process exit).
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._pool.shutdown(wait=False, cancel_futures=True)
        self._call_pool.shutdown(wait=False, cancel_futures=True)

    # ---- core ----
    def _check(self, source: str, text: str) -> GuardResult:
        agent = self._c._resolve_agent()  # existing ResolvedAgent (.agent_id)
        region = self._require_region()
        try:
            account = self._account()
        except AIGuardConfigError:
            raise  # misconfiguration (e.g. boto3 missing) must fail fast
        except Exception as e:  # STS failure -> can't form the key: resolution failure
            log.warning("account resolution (STS) failed: %s", e)
            return self._on_resolve_failure(text, "account could not be resolved (STS)")
        key = (agent.agent_id, account, region)
        guardrails = self._get_guardrails(key)
        if guardrails is None:  # resolution failed with nothing cached to serve
            return self._on_resolve_failure(text, "guardrails could not be resolved")
        # authoritative NO_GUARDRAILS (or empty list) -> pass through
        if not guardrails:
            return self._passthrough(text)
        return self._enforce(guardrails, source, text)

    def _get_guardrails(self, key) -> list | None:
        """Return the agent's guardrail list (possibly empty), or ``None`` if
        resolution failed with nothing cached to fall back on.

        Nothing here blocks except one bounded first attempt: on a cold miss the
        first caller resolves inline (single-flight, ``guardrail_resolve_timeout``);
        concurrent callers wait on that one attempt. If it fails, later misses do
        not resolve inline -- they fire a throttled async retry (at most one per
        ``guardrail_retry_interval``) and return None, so the agent path never
        blocks again and self-heals when the resolver recovers.
        """
        entry, freshness = self._cache.lookup(key)
        if entry and freshness == "fresh":
            return entry.guardrails
        if entry and freshness == "stale":
            self._refresh_async(key)  # serve stale, refresh in background
            return entry.guardrails
        # single-flight: concurrent cold callers wait here
        with self._cache.key_lock(key):
            entry, _ = self._cache.lookup(key)
            if entry:  # another caller resolved while we waited
                return entry.guardrails
            now = time.monotonic()
            last = self._last_attempt.get(key)
            if last is None:
                # First cold attempt for this agent: one bounded inline resolve.
                self._last_attempt[key] = now
                try:
                    return self._resolve_and_cache(key)
                except GuardrailResolutionError as e:
                    log.warning(
                        "guardrail resolve failed for agent=%s account=%s "
                        "region=%s: %s",
                        key[0],
                        key[1],
                        key[2],
                        e,
                    )
                    return None
            # Already attempted and still uncached: throttle async retries; never
            # block the agent path again.
            if now - last >= self._c.guardrail_retry_interval:
                self._last_attempt[key] = now
                self._refresh_async(key)
            return None

    def _enforce(self, guardrails: list, source: str, text: str) -> GuardResult:
        current = text
        all_assessments: list = []  # accumulated for logging/debug (incl. masks)
        for g in guardrails:
            try:
                blocked, current, assessments, reason = apply_guardrail_chunked(
                    self._bedrock(),
                    g,
                    source,
                    current,
                    cap=self._c.guardrail_chunk_size,
                    executor=self._call_pool,
                )
            except AIGuardConfigError:
                raise  # misconfiguration (e.g. boto3 missing) must fail fast
            except Exception as e:  # apply_guardrail call error
                log.warning(
                    "apply_guardrail error on guardrail=%s: %s", g.get("guardrailId"), e
                )
                if self._c.on_error == ALLOW:
                    continue
                return self._blocked(
                    current,
                    f"guardrail check errored: {e}",
                    guardrail_id=g.get("guardrailId"),
                    assessments=all_assessments,
                )
            all_assessments += assessments
            if blocked:
                return self._blocked(
                    current,
                    reason,
                    guardrail_id=g["guardrailId"],
                    assessments=all_assessments,
                )
        action = ACTION_INTERVENED if current != text else ACTION_NONE
        return GuardResult(False, current, action, assessments=all_assessments)

    # ---- result factories ----
    @staticmethod
    def _passthrough(text: str) -> GuardResult:
        return GuardResult(False, text, ACTION_NONE)

    @staticmethod
    def _blocked(
        text: str,
        reason: str | None,
        *,
        guardrail_id: str | None = None,
        assessments: list | None = None,
    ) -> GuardResult:
        return GuardResult(
            True, text, ACTION_INTERVENED, reason, assessments or [], guardrail_id
        )

    def _on_resolve_failure(self, text: str, reason: str) -> GuardResult:
        # No guardrails resolvable and nothing cached: policy decides.
        if self._c.on_resolve_failure == ALLOW:
            return self._passthrough(text)
        return self._blocked(text, reason)

    # ---- helpers ----
    def _resolve_and_cache(self, key) -> list:
        """Resolve from the worker node and cache the list on success.

        Transient failures (5xx / 429 / timeout / transport) get a short
        bounded inline retry so a brief blip does not fail-close the calling
        turn; terminal failures and exhausted retries raise
        :class:`GuardrailResolutionError` (nothing cached).
        """
        max_attempts = self._c.guardrail_resolve_max_attempts
        backoff = self._c.guardrail_resolve_retry_backoff
        for attempt in range(max_attempts):
            try:
                guardrails = resolve(
                    self._c._session,
                    self._c.base_url,
                    self._c.token,
                    *key,
                    timeout=self._c.guardrail_resolve_timeout,
                )[1]
                self._cache.store(key, guardrails)
                return guardrails
            except GuardrailResolutionError as e:
                if not e.transient or attempt == max_attempts - 1:
                    raise
                time.sleep(backoff * (2**attempt))

    def _require_region(self) -> str:
        if not self._c.region:
            raise AIGuardConfigError(
                "region is required for guardrail enforcement (set it from config)"
            )
        return self._c.region

    def _account(self) -> str:
        if self._c.account:
            return self._c.account
        with self._lock:  # single STS call per process, thread-safe
            if self._account_cache is None:
                self._account_cache = self._sts().get_caller_identity()["Account"]
            return self._account_cache

    def _boto_session(self):
        # The SDK builds its own bedrock-runtime + STS clients from a default
        # boto3 session, so credentials, region and endpoint all come from the
        # standard AWS chain (env vars, shared config/profile, instance/task or
        # AgentCore execution role). Custom endpoints (e.g. PrivateLink) and
        # retry tuning are configured the standard AWS way, not via the SDK.
        try:
            import boto3
        except ImportError as e:
            raise AIGuardConfigError(
                "guardrail enforcement requires boto3; install the extra: "
                "pip install 'onetrust-ai-guard-sdk[bedrock]'"
            ) from e
        return boto3.Session()

    def _bedrock(self):
        with self._lock:  # build the runtime client at most once
            if self._brt is None:
                self._brt = self._boto_session().client(
                    "bedrock-runtime", region_name=self._c.region
                )
            return self._brt

    def _sts(self):
        # Callers already hold self._lock (see _account); RLock makes this safe.
        with self._lock:
            if self._sts_client is None:
                self._sts_client = self._boto_session().client(
                    "sts", region_name=self._c.region
                )
            return self._sts_client

    def _refresh_async(self, key) -> None:
        with self._lock:  # single-flight: one background refresh per key at a time
            if self._closed or key in self._refreshing:
                return
            self._refreshing.add(key)

        def _do():
            try:
                self._resolve_and_cache(key)  # overrides only on success
            except GuardrailResolutionError as e:
                log.warning(
                    "background refresh failed for agent=%s (keeping "
                    "last-known-good): %s",
                    key[0],
                    e,
                )
            finally:
                with self._lock:
                    self._refreshing.discard(key)

        try:
            self._pool.submit(_do)
        except RuntimeError:  # pool already shut down (post-close)
            with self._lock:
                self._refreshing.discard(key)
