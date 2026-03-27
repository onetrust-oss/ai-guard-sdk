import json
from dataclasses import dataclass
from enum import Enum

from ai_guard.api import ClassificationResponse


class RedactKind(Enum):
    """Action to take when a classifier matches.

    - ``NONE`` -- pass through unchanged.
    - ``REDACT`` -- replace each matched character with the redactor character.
    - ``BLOCK`` -- block the entire text (returns empty string immediately).
    """

    NONE = 1
    REDACT = 2
    BLOCK = 3


@dataclass
class RedactAction:
    """Maps a classifier code to a :class:`RedactKind` action."""

    kind: RedactKind
    classifier: str


@dataclass
class Redaction:
    """Result of a redaction pass.

    Attributes:
        actions: Actions that were applied.
        text: The redacted text (empty string when blocked).
    """

    actions: list[RedactAction]
    text: str

    def __repr__(self) -> str:
        return json.dumps(
            {
                "actions": [
                    {"kind": action.kind.name, "classifier": action.classifier}
                    for action in self.actions
                ]
            }
        )


@dataclass
class RedactPolicy:
    """Policy controlling how classified matches are redacted.

    Attributes:
        actions: Per-classifier redaction actions.
        default: Action applied to classifiers not listed in *actions*.
        redactor: Single character used to replace each redacted character.
    """

    actions: list[RedactAction]
    default: RedactKind
    redactor: str

    def __post_init__(self):
        if len(self.redactor) != 1:
            raise ValueError(
                f"redactor must be exactly 1 character, got {len(self.redactor)}"
            )


class ClassificationRedactor:
    """Redacts text based on classification results and a :class:`RedactPolicy`.

    Block takes priority -- if any match triggers ``BLOCK``, the entire text
    is blocked regardless of other actions.
    """

    def __init__(
        self,
        policy: RedactPolicy,
    ):
        self._policy = policy

    def redact(self, text: str, classification: ClassificationResponse) -> Redaction:
        """Apply the policy to *text* using the given classification result."""
        applied_actions: list[RedactAction] = []

        result_chars = list(text)

        for match in classification.matches:
            action = None
            for policy_action in self._policy.actions:
                if policy_action.classifier == match.classifier:
                    action = policy_action
                    break

            if action is None:
                action = RedactAction(
                    kind=self._policy.default, classifier=match.classifier
                )

            if action.kind == RedactKind.BLOCK:
                return Redaction(actions=[action], text="")
            elif action.kind == RedactKind.REDACT:
                if match.start < len(result_chars) and match.end <= len(result_chars):
                    for i in range(match.start, match.end):
                        result_chars[i] = self._policy.redactor

                if action not in applied_actions:
                    applied_actions.append(action)
            # RedactKind.NONE - do nothing

        result_text = "".join(result_chars)

        return Redaction(actions=applied_actions, text=result_text)
