from dataclasses import dataclass, field

from ..constants import ACTION_NONE


@dataclass
class GuardResult:
    """Outcome of a ``check_input`` / ``check_output`` call.

    Attributes:
        blocked: ``True`` iff some assessment action == ``"BLOCKED"``.
        text: Safe text to forward -- masked when a policy anonymized,
            otherwise the original text. Callers MUST forward this, not the
            original input.
        action: ``ACTION_NONE`` | ``ACTION_INTERVENED``.
        action_reason: Reason string from the response ``actionReason`` when
            blocked; otherwise ``None``.
        assessments: Raw per-guardrail assessments (for logging/debug).
        guardrail_id: The guardrail that blocked, if any.
    """

    blocked: bool
    text: str
    action: str = ACTION_NONE
    action_reason: str | None = None
    assessments: list = field(default_factory=list)
    guardrail_id: str | None = None
