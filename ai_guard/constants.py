"""Public failure-policy constants for ``AIGuardClient``.

``on_error`` / ``on_resolve_failure`` accept exactly these values.
"""

BLOCK = "block"
ALLOW = "allow"

# ``GuardResult.action`` values.
ACTION_NONE = "NONE"
ACTION_INTERVENED = "GUARDRAIL_INTERVENED"
