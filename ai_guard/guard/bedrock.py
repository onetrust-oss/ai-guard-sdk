from collections.abc import Iterator

from ..client.chunker import LineChunker
from ..constants import ACTION_INTERVENED

_ACTION_BLOCKED = "BLOCKED"

_CHECK_CAP = 100_000  # default max chars per ApplyGuardrail request
# Measured limits: ~14MB per request overall, but per-policy caps are lower --
# denied topic policy (Classic tier) rejects input over 106 text units
# (106K chars, summed across a request's content blocks). 100K stays under
# the strictest observed per-policy cap.
_BLOCK_CAP = 25_000  # content-block granularity packed inside one request


def line_chunks(text: str, max_chars: int) -> Iterator[str]:
    """Yield ``text`` in chunks no longer than ``max_chars``.

    Wraps the SDK's stateful, single-use :class:`ai_guard.client.chunker.LineChunker`
    (``LineChunker(size)`` + ``append`` / ``next`` / drain) with a fresh instance
    per call, and guarantees any single over-long line is split so no chunk
    exceeds ``max_chars``.
    """
    chunker = LineChunker(size=max_chars)
    chunker.append(text)
    chunker.append(None)  # drain any trailing partial line
    while True:
        chunk = chunker.next()
        if chunk is None:
            break
        yield chunk


def _has_blocked(assessments: list) -> bool:
    for a in assessments:
        sip = a.get("sensitiveInformationPolicy", {})
        if any(
            x.get("action") == _ACTION_BLOCKED
            for coll in (sip.get("piiEntities", []), sip.get("regexes", []))
            for x in coll
        ):
            return True
        for pol, items in (
            ("contentPolicy", "filters"),
            ("topicPolicy", "topics"),
            ("contextualGroundingPolicy", "filters"),
        ):
            if any(
                x.get("action") == _ACTION_BLOCKED
                for x in a.get(pol, {}).get(items, [])
            ):
                return True
        wp = a.get("wordPolicy", {})
        if any(
            x.get("action") == _ACTION_BLOCKED
            for coll in (wp.get("customWords", []), wp.get("managedWordLists", []))
            for x in coll
        ):
            return True
    return False


def apply_one(client, guardrail: dict, source: str, text: str):
    """One ``apply_guardrail`` call -> (blocked, safe_text, assessments, reason)."""
    blocked, parts, assessments, reason = _apply_blocks(
        client, guardrail, source, [text]
    )
    return blocked, "".join(parts), assessments, reason


def _apply_blocks(client, guardrail: dict, source: str, blocks: list):
    """One ``apply_guardrail`` call over multiple content blocks.

    Blocks are evaluated independently in one request; ``outputs`` is
    position-aligned with ``content`` and carries each block's masked text
    (verified live), so per-block safe text concatenates in order.
    Returns (blocked, [safe_text_per_block], assessments, reason).
    """
    resp = client.apply_guardrail(
        guardrailIdentifier=guardrail["guardrailId"],  # bare id (same-account model)
        guardrailVersion=guardrail["version"],  # numbered published version, not DRAFT
        source=source,  # "INPUT" | "OUTPUT"
        content=[{"text": {"text": b}} for b in blocks],
        # outputScope defaults to "INTERVENTIONS" (keep responses small)
    )
    assessments = resp.get("assessments", [])
    outputs = resp.get("outputs") or []
    # Masked text when ANONYMIZED. A NONE response may carry no outputs --
    # the original blocks are safe to forward. But once the guardrail
    # intervened, a missing/malformed output must fail closed: echoing the
    # original block could forward unmasked content.
    if resp.get("action") == ACTION_INTERVENED and (
        len(outputs) < len(blocks)
        or any(not isinstance(o, dict) or "text" not in o for o in outputs)
    ):
        return True, [], assessments, "guardrail response missing masked output"
    safe = [
        outputs[i].get("text", b) if i < len(outputs) else b
        for i, b in enumerate(blocks)
    ]
    return _has_blocked(assessments), safe, assessments, resp.get("actionReason")


def _pack_blocks(blocks: list, budget: int) -> Iterator[list]:
    """Group consecutive blocks into requests of at most ``budget`` chars."""
    group, size = [], 0
    for b in blocks:
        if group and size + len(b) > budget:
            yield group
            group, size = [], 0
        group.append(b)
        size += len(b)
    if group:
        yield group


def apply_guardrail_chunked(
    client,
    guardrail: dict,
    source: str,
    text: str,
    cap: int = _CHECK_CAP,
    executor=None,
):
    """Split ``text`` into content blocks packed into as few
    ``apply_guardrail`` calls as the per-request char budget ``cap`` allows.

    Blocks are ``_BLOCK_CAP``-sized, line-aligned slices; consecutive blocks
    are sent as one request's content-block array while their combined size
    stays <= ``cap`` (the per-request unit budget — keep it under the
    strictest policy cap in play, e.g. denied-topic Classic = 106K chars).

    Requests run sequentially and stop after the first that blocks (later
    requests are never sent, saving quota); with an ``executor`` they run
    concurrently. Blocked if any request blocks; masked outputs concatenate
    in block order.
    """
    if len(text) <= cap:
        return apply_one(client, guardrail, source, text)
    blocks = list(line_chunks(text, min(_BLOCK_CAP, cap)))
    groups = list(_pack_blocks(blocks, cap))
    if executor is not None and len(groups) > 1:
        return _apply_parallel(client, guardrail, source, groups, executor)
    blocked, parts, assessments, reason = False, [], [], None
    for group in groups:
        b, safe, asmt, r = _apply_blocks(client, guardrail, source, group)
        blocked, reason = blocked or b, reason or r
        assessments += asmt
        parts += safe
        if b:
            break
    return blocked, "".join(parts), assessments, reason


def _apply_parallel(client, guardrail, source, groups, executor):
    """Concurrent packed-block requests; results concatenated in block order."""
    futures = [
        executor.submit(_apply_blocks, client, guardrail, source, group)
        for group in groups
    ]
    blocked, reason = False, None
    parts: list[str] = []
    assessments: list = []
    for fut in futures:  # iteration order == request order
        b, safe, asmt, r = fut.result()
        blocked, reason = blocked or b, reason or r
        parts += safe
        assessments += asmt
    return blocked, "".join(parts), assessments, reason
