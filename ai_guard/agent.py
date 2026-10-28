import contextvars
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ai_guard.api import AIPlatform

if TYPE_CHECKING:
    from ai_guard.client.client import AIGuardClient


@dataclass(frozen=True)
class AgentConfig:
    """Registration for one logical agent in a multi-agent process.

    Args:
        name: Local activation key passed to :meth:`AIGuardClient.activate`.
            Never sent to the AI Guard service.
        agent_id: Identifier sent to the AI Guard service as ``agent_id``
            and used as the OTel ``onetrust.reference.id`` value while this
            agent is active.
    """

    name: str
    agent_id: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("AgentConfig.name must be a non-empty string")
        if not self.agent_id:
            raise ValueError("AgentConfig.agent_id must be a non-empty string")

    @staticmethod
    def from_mapping(data: Mapping[str, Any]) -> "AgentConfig":
        """Build an :class:`AgentConfig` from a plain mapping (e.g. a dict
        loaded from the customer's own YAML/JSON config)."""
        if not isinstance(data, Mapping):
            raise ValueError(
                "agent config entry must be an AgentConfig or a mapping, got "
                f"{type(data).__name__}"
            )
        missing = [k for k in ("name", "agent_id") if k not in data]
        if missing:
            raise ValueError(
                f"agent config entry missing required field(s): {', '.join(missing)}"
            )
        return AgentConfig(name=data["name"], agent_id=data["agent_id"])


def build_agent_registry(
    agents: Sequence["AgentConfig | Mapping[str, Any]"] | None,
) -> dict[str, AgentConfig]:
    """Validate and index agent registrations by name.

    Raises:
        ValueError: On an unsupported container type, a malformed or
            incomplete entry, or a duplicate agent name.
    """
    if agents is None:
        return {}
    if isinstance(agents, str | bytes | Mapping):
        raise ValueError(
            "agents must be a sequence of AgentConfig or mapping entries, got "
            f"{type(agents).__name__}"
        )

    registry: dict[str, AgentConfig] = {}
    for entry in agents:
        cfg = (
            entry if isinstance(entry, AgentConfig) else AgentConfig.from_mapping(entry)
        )
        if cfg.name in registry:
            raise ValueError(f"duplicate agent name: {cfg.name!r}")
        registry[cfg.name] = cfg
    return registry


@dataclass(frozen=True)
class ResolvedAgent:
    """The agent identity resolved for a single classify()/metric() call."""

    agent_id: str
    platform: AIPlatform | None
    reference_id: str | None


@dataclass(frozen=True)
class _Activation:
    owner: "AIGuardClient"
    agent: AgentConfig


_active_agent: contextvars.ContextVar[_Activation | None] = contextvars.ContextVar(
    "ai_guard_active_agent", default=None
)


class AgentActivation:
    """Context manager returned by :meth:`AIGuardClient.activate`.

    Backed by a single `contextvars.ContextVar`, which gives independent,
    automatically-restored state per thread and per asyncio Task -- nesting,
    exceptions, and cancellation are handled by the context manager protocol
    and by `contextvars` itself, not by any mutable client-level state.
    """

    __slots__ = ("_owner", "_agent", "_token")

    def __init__(self, owner: "AIGuardClient", agent: AgentConfig):
        self._owner = owner
        self._agent = agent
        self._token: contextvars.Token[_Activation | None] | None = None

    def __enter__(self) -> AgentConfig:
        self._token = _active_agent.set(
            _Activation(owner=self._owner, agent=self._agent)
        )
        return self._agent

    def __exit__(self, *exc_info: object) -> None:
        assert self._token is not None
        _active_agent.reset(self._token)


def capture_active_agent() -> "_Activation | None":
    """Snapshot whatever agent activation is current on this thread/task.

    A raw `Thread` or a `ThreadPoolExecutor` worker does not automatically
    inherit the calling thread's contextvars state. Callers that hand work
    to such a thread (e.g. `ClassificationStream`) capture the snapshot on
    the originating thread and re-bind it with `bind_captured_agent` at the
    start of the worker's execution.
    """
    return _active_agent.get()


def bind_captured_agent(slot: "_Activation | None") -> None:
    """Bind a snapshot from `capture_active_agent` onto the calling thread.

    Intended to be called once at the start of a new thread's/worker's
    entry point, before any `resolve_active_agent` call on that thread.
    """
    _active_agent.set(slot)


def get_active_agent_id() -> str | None:
    """Return the ``agent_id`` of whichever agent is currently activated, or
    ``None`` if none is.

    Unlike :func:`resolve_active_agent`, this ignores which client performed
    the activation and never falls back to a client's default *agent_id* --
    it answers "is *any* agent activated right now, and if so which one,"
    for host-owned OTel instrumentation that wants to act on every span
    while an agent is active (e.g. a custom ``SpanProcessor.on_start()``),
    rather than only at the moment :meth:`AIGuardClient.classify`/
    :meth:`~AIGuardClient.metric` are called -- which is all the SDK itself
    stamps automatically (see :mod:`ai_guard.otel`).
    """
    slot = _active_agent.get()
    return slot.agent.agent_id if slot is not None else None


def resolve_active_agent(
    owner: "AIGuardClient",
    default_agent_id: str | None,
    platform: AIPlatform | None,
) -> ResolvedAgent:
    """Resolve the agent identity to use for a classify()/metric() call.

    *platform* is a deployment-level property shared by every agent
    co-located in this client/process (set once at construction, if at
    all) -- it does not vary with which agent is activated.

    Only agents activated on *this* client instance are honored -- an
    activation on a different (possibly reused) client is ignored so
    concurrently-used clients never observe each other's active agent.

    Raises:
        RuntimeError: No agent is activated and the client was constructed
            without a default *agent_id*.
    """
    slot = _active_agent.get()
    agent_id: str | None
    reference_id: str | None
    if slot is not None and slot.owner is owner:
        agent_id = slot.agent.agent_id
        reference_id = slot.agent.agent_id
    else:
        agent_id = default_agent_id
        reference_id = None

    if agent_id is None:
        raise RuntimeError(
            "No agent activated and AIGuardClient was constructed without a "
            "default agent_id. Use client.activate(name), or pass agent_id "
            "to AIGuardClient() for single-agent use."
        )

    return ResolvedAgent(
        agent_id=agent_id, platform=platform, reference_id=reference_id
    )
