import logging

import requests

from .exceptions import GuardrailResolutionError

log = logging.getLogger("ai_guard.guard")

_TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504}


def resolve(
    session: requests.Session,
    base_url: str,
    token: str,
    agent_id: str,
    account: str,
    region: str,
    timeout: float = 3.0,
) -> tuple[str, list]:
    """Resolve an agent's guardrails from the OneTrust worker node.

    Returns ``(status, guardrails)`` where ``status`` is ``"OK"`` or
    ``"NO_GUARDRAILS"``. Raises :class:`GuardrailResolutionError` on failure.
    PII-safe: logs keys/status only, never guardrail input/output text.
    """
    try:
        r = session.post(
            f"{base_url}/v1/agents/{agent_id}/guardrails",
            json={"account": account, "region": region, "platform": "AMAZON_BEDROCK"},
            headers={"Authorization": f"Bearer {token}", "accept": "application/json"},
            timeout=timeout,
        )
    except requests.RequestException as e:
        raise GuardrailResolutionError(
            f"resolve transport error: {e}", transient=True
        ) from e
    if r.status_code in _TRANSIENT_STATUS:
        raise GuardrailResolutionError(f"resolve HTTP {r.status_code}", transient=True)
    if r.status_code >= 400:
        raise GuardrailResolutionError(f"resolve HTTP {r.status_code}", transient=False)
    try:
        body = r.json()
    except ValueError as e:  # malformed/empty JSON on a 2xx -> terminal failure
        raise GuardrailResolutionError(
            "resolve response was not valid JSON", transient=False
        ) from e
    if not isinstance(body, dict):
        raise GuardrailResolutionError(
            "resolve response was not a JSON object", transient=False
        )
    status = body.get("status")
    if status not in ("OK", "NO_GUARDRAILS"):
        raise GuardrailResolutionError(f"unexpected status {status!r}", transient=False)
    return status, body.get("guardrails", []) or []
