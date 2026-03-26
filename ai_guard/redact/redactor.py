from dataclasses import dataclass
from typing import List
from enum import Enum
import json

from ai_guard.api import ClassificationResponse


class RedactKind(Enum):
    NONE = 1
    REDACT = 2
    BLOCK = 3


@dataclass
class RedactAction:
    kind: RedactKind
    classifier: str


@dataclass
class Redaction:
    actions: List[RedactAction]
    text: str

    def __repr__(self) -> str:
        return json.dumps({
            "actions": [
                {"kind": action.kind.name, "classifier": action.classifier}
                for action in self.actions
            ]
        })


@dataclass
class RedactPolicy:
    actions: List[RedactAction]
    default: RedactKind
    redactor: str

    def __post_init__(self):
        if len(self.redactor) != 1:
            raise ValueError(f"redactor must be exactly 1 character, got {len(self.redactor)}")


class ClassificationRedactor:
    def __init__(
            self,
            policy: RedactPolicy,
    ):
        self._policy = policy

    def redact(self, text: str, classification: ClassificationResponse) -> Redaction:
        applied_actions: List[RedactAction] = []

        result_chars = list(text)

        for match in classification.matches:
            action = None
            for policy_action in self._policy.actions:
                if policy_action.classifier == match.classifier:
                    action = policy_action
                    break

            if action is None:
                action = RedactAction(kind=self._policy.default, classifier=match.classifier)

            if action.kind == RedactKind.BLOCK:
                return Redaction(actions=[action], text="")
            elif action.kind == RedactKind.REDACT:
                if match.start < len(result_chars) and match.end <= len(result_chars):
                    for i in range(match.start, match.end):
                        result_chars[i] = self._policy.redactor

                if action not in applied_actions:
                    applied_actions.append(action)
            # RedactKind.NONE - do nothing

        result_text = ''.join(result_chars)

        return Redaction(actions=applied_actions, text=result_text)
