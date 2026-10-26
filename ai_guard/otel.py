"""Optional OpenTelemetry span stamping for multi-agent activation.

The SDK never creates, configures, or owns a `TracerProvider`, span, or
exporter -- it only annotates whatever span is already current in the OTel
context, and only while inside an `AIGuardClient.activate()` scope. Outside
an activation scope (i.e. all existing single-agent usage), nothing here is
invoked at all.

Two independent, separately-optional pieces:

- `stamp_reference_id` (used internally by `classify()`/`metric()`) depends
  only on `opentelemetry-api` (the `onetrust-ai-guard-sdk[otel]` extra) and
  stamps whatever span is current at the moment it's called.
- `OneTrustSpanProcessor` depends on `opentelemetry-sdk` (the
  `onetrust-ai-guard-sdk[otel-sdk]` extra) and stamps *every* span created
  while an agent is active, for callers who register it on their own
  `TracerProvider`. It's provided so callers don't have to hand-write this
  themselves; the SDK still never constructs a `TracerProvider` or registers
  anything on your behalf.

When the corresponding package is not installed, each piece degrades
gracefully (a no-op for `stamp_reference_id`; a clear `ImportError` only if
`OneTrustSpanProcessor` is actually instantiated) so the rest of the SDK
behaves identically with or without either installed.

`AIGuardClient.activate()` also calls `warn_if_no_tracer_provider`, which
emits a one-time warning (never a hard error, never a behavior change) if
`opentelemetry-api` is installed but no real `TracerProvider` is configured
-- see its docstring for why that specific condition, and not "no
`OneTrustSpanProcessor` registered", is what's checked.
"""

import warnings

from ai_guard.agent import get_active_agent_id

REFERENCE_ID_ATTRIBUTE = "onetrust.reference.id"

try:
    from opentelemetry import trace as _trace

    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised via monkeypatched flag
    _trace = None  # type: ignore[assignment]
    _OTEL_AVAILABLE = False

try:
    from opentelemetry.sdk.trace import SpanProcessor as _SpanProcessor

    _OTEL_SDK_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised via monkeypatched flag
    _SpanProcessor = object  # type: ignore[assignment,misc]
    _OTEL_SDK_AVAILABLE = False


_warned_no_tracer_provider = False


def warn_if_no_tracer_provider() -> None:
    """Warn, at most once per process, if `activate()` is used without a
    real OpenTelemetry `TracerProvider` configured.

    No-ops entirely if `opentelemetry-api` isn't installed -- OTel was never
    part of this deployment's story, so there's nothing to warn about. If it
    *is* installed but `trace.get_tracer_provider()` still returns the
    default no-op proxy (no SDK `TracerProvider` has been set anywhere in
    the process), then `stamp_reference_id` and any registered
    `OneTrustSpanProcessor` can never take effect -- there is no recording
    span to stamp -- so this is a strong enough signal to warn on rather
    than fail silently.

    Deliberately does *not* try to detect whether a `OneTrustSpanProcessor`
    is registered on an existing provider: that would require reaching into
    SDK-private internals, and a customer legitimately using only the
    lighter-weight `stamp_reference_id` path (no `OneTrustSpanProcessor` at
    all) would trip a false positive.
    """
    global _warned_no_tracer_provider
    if _warned_no_tracer_provider or not _OTEL_AVAILABLE:
        return
    provider = _trace.get_tracer_provider()
    if hasattr(provider, "add_span_processor"):
        return
    _warned_no_tracer_provider = True
    warnings.warn(
        "AIGuardClient.activate() was called, but no OpenTelemetry "
        "TracerProvider is configured in this process -- "
        "onetrust.reference.id will not be stamped on any spans. If you "
        "want OTel span tagging, call trace.set_tracer_provider(...) "
        "(and register OneTrustSpanProcessor on it, if you want every "
        "span tagged rather than just classify()/metric() call sites) "
        "before activate() is used. See the README's OTel section for "
        "details.",
        stacklevel=3,
    )


def stamp_reference_id(reference_id: str | None) -> None:
    """Set `onetrust.reference.id` on the current span, if any.

    No-ops when *reference_id* is ``None`` (no agent activated),
    `opentelemetry-api` isn't installed, or there is no recording span in
    the current OTel context.
    """
    if reference_id is None or not _OTEL_AVAILABLE:
        return
    span = _trace.get_current_span()
    if span is not None and span.is_recording():
        span.set_attribute(REFERENCE_ID_ATTRIBUTE, reference_id)


class OneTrustSpanProcessor(_SpanProcessor):
    """Stamps `onetrust.reference.id` on every span while an agent is active.

    You must register an instance on your own `TracerProvider` yourself --
    this is never done automatically::

        from ai_guard import OneTrustSpanProcessor
        provider.add_span_processor(OneTrustSpanProcessor())

    Constructing this class does nothing by itself. The SDK never creates,
    discovers, or mutates a `TracerProvider` on your behalf (the same
    "no global provider mutation" rule that applies everywhere else in this
    SDK), so if you never call `add_span_processor(...)`, spans are created
    without `onetrust.reference.id` -- silently, with no error or warning.

    Unlike `classify()`/`metric()`'s own stamping (which only touches the
    one span current at the moment they're called), this stamps every span
    created for the duration of an `AIGuardClient.activate()` scope --
    register it *before* entering `activate()` so spans opened right at the
    start of the scope are included too (it reads whichever agent is active
    at each span's *creation* time, via `ai_guard.get_active_agent_id()`,
    independent of any specific client instance).

    Requires `opentelemetry-sdk` (the `onetrust-ai-guard-sdk[otel-sdk]`
    extra) -- raises `ImportError` on construction if it isn't installed,
    with an actionable message, rather than failing on import or silently
    doing nothing.
    """

    def __init__(self) -> None:
        if not _OTEL_SDK_AVAILABLE:
            raise ImportError(
                "OneTrustSpanProcessor requires opentelemetry-sdk. Install "
                'with `pip install "onetrust-ai-guard-sdk[otel-sdk]"`.'
            )
        super().__init__()

    def on_start(self, span, parent_context=None) -> None:
        agent_id = get_active_agent_id()
        if agent_id is not None:
            span.set_attribute(REFERENCE_ID_ATTRIBUTE, agent_id)

    def on_end(self, span) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
