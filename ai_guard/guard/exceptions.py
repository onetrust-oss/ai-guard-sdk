class AIGuardConfigError(ValueError):
    """Raised when required config is missing (e.g. region).

    Fail fast at init / first check.
    """


class GuardrailResolutionError(Exception):
    """Raised when resolving an agent's guardrails from the worker node fails.

    ``transient`` distinguishes retryable failures (5xx / timeout / transport)
    from terminal ones (401/403/404, malformed envelope).
    """

    def __init__(self, message: str, *, transient: bool):
        super().__init__(message)
        self.transient = transient
