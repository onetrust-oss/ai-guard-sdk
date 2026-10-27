import base64
import functools
import hashlib
import inspect
import logging
import ssl
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

import requests
import urllib3
from cryptography import x509
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from requests.adapters import HTTPAdapter

from ai_guard import otel
from ai_guard.agent import (
    AgentActivation,
    AgentConfig,
    build_agent_registry,
    resolve_active_agent,
)
from ai_guard.api import (
    AIPlatform,
    ClassificationRequest,
    ClassificationResponse,
    MetricsEvent,
)
from ai_guard.constants import ALLOW, BLOCK

if TYPE_CHECKING:
    from ai_guard.guard import GuardResult

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class _PinningAdapter(HTTPAdapter):
    def __init__(self, ssl_context: ssl.SSLContext, **kwargs):
        self._ssl_context = ssl_context
        super().__init__(**kwargs)

    def init_poolmanager(self, *args, **kwargs):
        kwargs["ssl_context"] = self._ssl_context
        return super().init_poolmanager(*args, **kwargs)


class AIGuardClient:
    """Client for the AI Guard classification and metrics API.

    Sends classification requests to an AI Guard server and returns match
    results.  Optionally records metrics events.

    Args:
        base_url: AI Guard server URL (e.g. ``"https://guard.example.com"``).
        token: Bearer token for authentication.
        agent_id: Identifier for the calling agent, used whenever no agent
            is activated (see *agents* below). Required for single-agent
            use; optional for clients that only ever use
            :meth:`activate`.
        platform: AI platform originating requests. Optional; a
            deployment-level property shared by every agent in this
            process (it does not vary with which agent is activated). Only
            sent to the server if provided.
        session: Optional pre-configured :class:`requests.Session`.  When
            provided, *pin_sha256* is ignored.
        timeout: HTTP request timeout in seconds.
        pin_sha256: Base64-encoded SHA-256 of the server's SPKI for
            certificate pinning.  Mutually exclusive with *session*.
        agents: Optional additional agents for multi-agent use, as
            :class:`~ai_guard.agent.AgentConfig` instances or equivalent
            mappings with ``name`` and ``agent_id`` keys. Activate one with
            :meth:`activate`, or with :meth:`agent` when the scope is a
            whole function. *agent_id* above remains the default identity
            used whenever no agent is activated -- single-agent usage is
            unaffected.

    Raises:
        ValueError: *agents* contains a malformed, incomplete, or
            duplicate-named entry.
    """

    def __init__(
        self,
        base_url: str,
        token: str,
        agent_id: str | None = None,
        platform: AIPlatform | None = None,
        session: requests.Session | None = None,
        timeout: float = 30.0,
        pin_sha256: str | None = None,
        agents: Sequence[AgentConfig | Mapping[str, Any]] | None = None,
        region: str | None = None,
        account: str | None = None,
        on_error: str = BLOCK,
        on_resolve_failure: str = BLOCK,
        guardrail_cache_ttl: int = 86400,
        guardrail_retry_interval: int = 60,
        guardrail_resolve_timeout: float = 3.0,
        guardrail_resolve_max_attempts: int = 3,
        guardrail_resolve_retry_backoff: float = 0.25,
        guardrail_chunk_size: int = 100_000,
        guardrail_stream_buffer_size: int = 25_000,
    ):
        cleaned = base_url.strip().rstrip("/") if base_url else ""
        if not cleaned:
            raise ValueError("base_url is required and must be a non-empty string")
        self._base_url = cleaned
        self._token = token
        self._agent_id = agent_id
        self._platform = platform
        self._timeout = timeout
        self._agents = build_agent_registry(agents)

        # Guardrail-enforcement config (see SDK README "Configuration").
        if on_error not in (BLOCK, ALLOW):
            raise ValueError(f"on_error must be 'block' or 'allow', got {on_error!r}")
        if on_resolve_failure not in (BLOCK, ALLOW):
            raise ValueError(
                "on_resolve_failure must be 'block' or 'allow', got "
                f"{on_resolve_failure!r}"
            )
        if guardrail_cache_ttl < 0:
            raise ValueError(
                f"guardrail_cache_ttl must be >= 0, got {guardrail_cache_ttl!r}"
            )
        if guardrail_retry_interval <= 0:
            raise ValueError(
                "guardrail_retry_interval must be > 0, got "
                f"{guardrail_retry_interval!r}"
            )
        if guardrail_resolve_timeout <= 0:
            raise ValueError(
                "guardrail_resolve_timeout must be > 0, got "
                f"{guardrail_resolve_timeout!r}"
            )
        if guardrail_resolve_max_attempts < 1:
            raise ValueError(
                "guardrail_resolve_max_attempts must be >= 1, got "
                f"{guardrail_resolve_max_attempts!r}"
            )
        if guardrail_resolve_retry_backoff < 0:
            raise ValueError(
                "guardrail_resolve_retry_backoff must be >= 0, got "
                f"{guardrail_resolve_retry_backoff!r}"
            )
        if guardrail_chunk_size <= 0:
            raise ValueError(
                f"guardrail_chunk_size must be > 0, got {guardrail_chunk_size!r}"
            )
        if guardrail_stream_buffer_size <= 0:
            raise ValueError(
                "guardrail_stream_buffer_size must be > 0, got "
                f"{guardrail_stream_buffer_size!r}"
            )
        self.region = region
        self.account = account
        self.on_error = on_error
        self.on_resolve_failure = on_resolve_failure
        self.guardrail_cache_ttl = guardrail_cache_ttl
        self.guardrail_retry_interval = guardrail_retry_interval
        self.guardrail_resolve_timeout = guardrail_resolve_timeout
        self.guardrail_resolve_max_attempts = guardrail_resolve_max_attempts
        self.guardrail_resolve_retry_backoff = guardrail_resolve_retry_backoff
        self.guardrail_chunk_size = guardrail_chunk_size
        self.guardrail_stream_buffer_size = guardrail_stream_buffer_size
        self._guard: Any | None = None
        self._guard_lock = threading.Lock()

        if session is not None:
            self._session = session
        else:
            self._session = requests.Session()
            if pin_sha256 is not None:
                self._apply_public_key_pin(pin_sha256)

    @property
    def base_url(self) -> str:
        """OneTrust worker-node URL (normalized, trailing slash stripped)."""
        return self._base_url

    @property
    def token(self) -> str:
        """OneTrust API key, sent as ``Authorization: Bearer``."""
        return self._token

    def _apply_public_key_pin(self, expected_key_sha256_b64: str) -> None:
        try:
            expected_digest = base64.b64decode(expected_key_sha256_b64, validate=True)
        except Exception as e:
            raise ValueError(f"pin_sha256 is not valid base64: {e}") from e

        if len(expected_digest) != 32:
            raise ValueError(
                f"pin_sha256 must be a base64-encoded SHA-256 digest (32 bytes), "
                f"got {len(expected_digest)}"
            )

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        _original_wrap_socket = ctx.wrap_socket

        def _pinning_wrap_socket(sock, *args, **kwargs):
            ssl_sock = _original_wrap_socket(sock, *args, **kwargs)
            peer_der = ssl_sock.getpeercert(binary_form=True)
            if peer_der is None:
                ssl_sock.close()
                raise ssl.SSLCertVerificationError("No peer certificate presented")
            spki_der = (
                x509.load_der_x509_certificate(peer_der)
                .public_key()
                .public_bytes(Encoding.DER, PublicFormat.SubjectPublicKeyInfo)
            )
            if hashlib.sha256(spki_der).digest() != expected_digest:
                ssl_sock.close()
                raise ssl.SSLCertVerificationError(
                    "Server public key does not match trusted key"
                )
            return ssl_sock

        ctx.wrap_socket = _pinning_wrap_socket

        self._session.verify = False
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        self._session.mount("https://", _PinningAdapter(ssl_context=ctx))

    def _lookup_agent(self, name: str) -> AgentConfig:
        try:
            return self._agents[name]
        except KeyError:
            raise ValueError(
                f"Unknown agent {name!r}; registered agents: {sorted(self._agents)}"
            ) from None

    def activate(self, name: str) -> AgentActivation:
        """Activate a registered agent for the duration of a ``with`` block.

        While active, :meth:`classify` and :meth:`metric` use this agent's
        *agent_id* instead of the client's default, and OTel spans created
        in the current context receive the ``onetrust.reference.id``
        attribute. *platform* (if configured) is unaffected by activation --
        it is a deployment-level property shared by every agent in this
        process. Activation is context-local (safe across nested scopes,
        threads, and asyncio tasks) and is restored automatically on exit,
        including on exception. Emits a one-time warning (see
        :func:`ai_guard.otel.warn_if_no_tracer_provider`) if OTel is
        installed but no `TracerProvider` is configured, since in that case
        no span-level stamping can ever take effect.

        Args:
            name: Key from the *agents* passed to the constructor.

        Raises:
            ValueError: *name* was not registered.
        """
        agent = self._lookup_agent(name)
        otel.warn_if_no_tracer_provider()
        return AgentActivation(self, agent)

    def agent(self, name: str) -> Callable[[F], F]:
        """Decorator form of :meth:`activate`, for whole-function scopes.

        ``@client.agent("support")`` is exactly equivalent to wrapping the
        entire function body in ``with client.activate("support"):`` -- the
        same `contextvars`-backed activation, the same nesting, restore-on-
        exception, and per-thread/per-task isolation semantics::

            @client.agent("support")
            def handle(message):
                return client.classify(request)   # uses support's agent_id

        Use :meth:`activate` instead when the agent scope is *not* the whole
        function -- a mid-function handoff to another agent, a loop over
        several agents, or a name chosen at runtime. The two compose freely:
        a ``with client.activate(...)`` block inside a decorated function
        nests and restores exactly as it would anywhere else.

        Plain functions, coroutine functions, generator functions, and async
        generator functions are all supported. For the two generator forms
        the agent is active for the duration of *iteration*, not merely for
        the call that creates the generator object -- so a decorated
        generator that yields between `classify()` calls still resolves to
        the right agent on every one of them.

        Each call gets its own activation, so a single decorated function is
        safe to call concurrently from many threads or asyncio tasks.

        Like :meth:`activate`, this does not follow work handed to a raw
        `Thread` or `ThreadPoolExecutor`; use
        :func:`~ai_guard.agent.capture_active_agent` and
        :func:`~ai_guard.agent.bind_captured_agent` there.

        Args:
            name: Key from the *agents* passed to the constructor. Validated
                immediately, at decoration time, so a misspelled name fails
                at import rather than on the first call to the decorated
                function.

        Raises:
            ValueError: *name* was not registered.
        """
        self._lookup_agent(name)

        def decorate(fn: F) -> F:
            # A fresh AgentActivation per call: the contextvars Token held by
            # an activation is single-use and must never be shared across
            # calls, threads, or tasks.
            if inspect.isasyncgenfunction(fn):

                @functools.wraps(fn)
                async def wrapper(*args: Any, **kwargs: Any) -> Any:
                    with self.activate(name):
                        async for item in fn(*args, **kwargs):
                            yield item

            elif inspect.iscoroutinefunction(fn):

                @functools.wraps(fn)
                async def wrapper(*args: Any, **kwargs: Any) -> Any:
                    with self.activate(name):
                        return await fn(*args, **kwargs)

            elif inspect.isgeneratorfunction(fn):

                @functools.wraps(fn)
                def wrapper(*args: Any, **kwargs: Any) -> Any:
                    with self.activate(name):
                        yield from fn(*args, **kwargs)

            else:

                @functools.wraps(fn)
                def wrapper(*args: Any, **kwargs: Any) -> Any:
                    with self.activate(name):
                        return fn(*args, **kwargs)

            return wrapper  # type: ignore[return-value]

        return decorate

    def _resolve_agent(self):
        return resolve_active_agent(self, self._agent_id, self._platform)

    def _enforcer(self):
        # Double-checked: concurrent first check_* calls must not race two
        # GuardrailEnforcer instances (each owns thread pools).
        if self._guard is None:
            with self._guard_lock:
                if self._guard is None:
                    from ai_guard.guard.enforcer import GuardrailEnforcer

                    self._guard = GuardrailEnforcer(self)
        return self._guard

    def check_input(self, text: str) -> "GuardResult":
        """Enforce the active agent's guardrails on model *input* text.

        Resolves the active agent's guardrails (worker node, sticky-cached)
        and runs one Bedrock ``apply_guardrail`` call per guardrail with
        ``source="INPUT"``. Returns a :class:`~ai_guard.guard.GuardResult`;
        always forward ``result.text`` (it may be masked), never the original.
        """
        return self._enforcer().check_input(text)

    def check_output(self, text: str) -> "GuardResult":
        """Enforce the active agent's guardrails on model *output* text
        (``source="OUTPUT"``). See :meth:`check_input`."""
        return self._enforcer().check_output(text)

    def check_output_stream(
        self, tokens: "Iterable[str]", emit: "Callable[[str], None]"
    ) -> "GuardResult":
        """Streaming variant of :meth:`check_output` for agents that stream.

        Buffers ``tokens`` to the ApplyGuardrail cap, checks each buffer, and
        stops on a block. ``emit`` receives cleared (safe) text. Already-emitted
        buffers cannot be recalled once a later buffer blocks."""
        return self._enforcer().check_output_stream(tokens, emit)

    def classify(self, request: ClassificationRequest) -> ClassificationResponse:
        """Classify text and return matches.

        Sends a ``POST /classifications/v1`` request.  The active agent's
        *agent_id* (the client's default *agent_id* when no agent is
        activated -- see :meth:`activate`) and, if configured, *platform*
        are injected into the request context sent to the server. The
        caller-owned *request* object (including ``request.context``) is
        never modified.

        Raises:
            ValueError: 400 response.
            PermissionError: 401 response.
            RuntimeError: Any other non-200 response, or no agent is
                activated and the client has no default *agent_id*.
        """
        agent = self._resolve_agent()
        otel.stamp_reference_id(agent.reference_id)

        url = f"{self._base_url}/classifications/v1"
        payload = request.to_dict()
        context = dict(payload.get("context") or {})
        context["agent_id"] = agent.agent_id
        if agent.platform is not None:
            context["platform"] = agent.platform
        payload["context"] = context
        logger.debug("Classification request payload: %s", payload)

        resp = self._session.post(
            url,
            json=payload,
            headers={
                "accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._token}",
            },
            timeout=self._timeout,
        )

        if resp.status_code == 200:
            data = resp.json()
            return ClassificationResponse.from_dict(data)

        content_type = resp.headers.get("Content-Type", "")
        is_json = "application/json" in content_type or content_type.endswith("+json")

        message = None
        if is_json:
            body = resp.json()
            message = body.get("message")

        msg = message or f"HTTP {resp.status_code} returned by AI Guard"
        if resp.status_code == 400:
            logger.error("Classification request error: %s", msg)
            raise ValueError(msg)
        elif resp.status_code == 401:
            logger.error("Classification request authorization error: %s", msg)
            raise PermissionError(msg)
        elif resp.status_code >= 500:
            logger.error("Classification server error: %s", msg)
            raise RuntimeError(msg)
        else:
            logger.error(
                "Unexpected response from AI Guard (%s): %s",
                resp.status_code,
                msg,
            )
            raise RuntimeError(msg)

    def metric(self, event: MetricsEvent) -> None:
        """Record a metrics event.

        Sends a ``POST /metric`` request.  The active agent's *agent_id*
        (the client's default *agent_id* when no agent is activated -- see
        :meth:`activate`) and, if configured, *platform* are injected into
        the attributes sent to the server. The caller-owned *event* object
        (including ``event.attributes``) is never modified.

        Raises:
            ValueError: 400 response.
            PermissionError: 401 response.
            RuntimeError: Any other non-200 response, or no agent is
                activated and the client has no default *agent_id*.
        """
        agent = self._resolve_agent()
        otel.stamp_reference_id(agent.reference_id)

        url = f"{self._base_url}/metric"
        attributes = dict(event.attributes)
        attributes["agent_id"] = agent.agent_id
        if agent.platform is not None:
            attributes["platform"] = agent.platform
        payload = {"attributes": attributes, "meter": event.meter.to_dict()}
        logger.debug("Metrics event payload: %s", payload)

        resp = self._session.post(
            url,
            json=payload,
            headers={
                "accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._token}",
            },
            timeout=self._timeout,
        )

        if resp.status_code == 200:
            return

        content_type = resp.headers.get("Content-Type", "")
        is_json = "application/json" in content_type or content_type.endswith("+json")

        message = None
        if is_json:
            body = resp.json()
            message = body.get("message")

        msg = message or f"HTTP {resp.status_code} returned by AI Guard"
        if resp.status_code == 400:
            logger.error("Metrics request error: %s", msg)
            raise ValueError(msg)
        elif resp.status_code == 401:
            logger.error("Metrics request authorization error: %s", msg)
            raise PermissionError(msg)
        else:
            logger.error(
                "Unexpected response from AI Guard (%s): %s",
                resp.status_code,
                msg,
            )
            raise RuntimeError(msg)
