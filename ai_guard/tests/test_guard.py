"""Unit tests for the Bedrock-Guardrails enforcement layer (impl-plan §2.10).

No network / AWS: the worker-node resolver is stubbed via a fake
``requests.Session`` and Bedrock ``apply_guardrail`` via a fake client.
"""

import threading
import time
from unittest.mock import Mock

import pytest
import requests

from ai_guard import AgentConfig, AIGuardClient
from ai_guard.guard import GuardrailEnforcer, GuardResult
from ai_guard.guard.bedrock import (
    _has_blocked,
    apply_guardrail_chunked,
    apply_one,
    line_chunks,
)
from ai_guard.guard.cache import GuardrailCache
from ai_guard.guard.exceptions import AIGuardConfigError, GuardrailResolutionError
from ai_guard.guard.resolver import resolve

# --------------------------------------------------------------------------
# fakes / helpers
# --------------------------------------------------------------------------


class _FakeResp:
    def __init__(self, status_code=200, body=None, json_error=False):
        self.status_code = status_code
        self._body = body if body is not None else {}
        self._json_error = json_error

    def json(self):
        if self._json_error:
            raise ValueError("No JSON object could be decoded")
        return self._body


def _ok_body(guardrails):
    return {"status": "OK", "guardrails": guardrails}


def _none_resp(text=None):
    return {"action": "NONE", "assessments": [], "outputs": []}


def _mask_resp(masked):
    return {
        "action": "GUARDRAIL_INTERVENED",
        "assessments": [
            {"sensitiveInformationPolicy": {"piiEntities": [{"action": "ANONYMIZED"}]}}
        ],
        "outputs": [{"text": masked}],
    }


def _block_resp(text, reason="blocked by content filter"):
    return {
        "action": "GUARDRAIL_INTERVENED",
        "assessments": [{"contentPolicy": {"filters": [{"action": "BLOCKED"}]}}],
        "outputs": [{"text": text}],
        "actionReason": reason,
    }


class _FakeBedrock:
    """Records apply_guardrail calls; returns queued responses (or by rule)."""

    def __init__(self, responses=None, rule=None):
        self._responses = list(responses or [])
        self._rule = rule
        self.calls = []

    def apply_guardrail(self, **kwargs):
        self.calls.append(kwargs)
        if self._rule is not None:
            return self._rule(kwargs)
        return self._responses.pop(0)


def _make_client(
    *,
    session_post=None,
    bedrock=None,
    agent_id="agent-1",
    agents=None,
    account="123456789012",
    region="us-east-1",
    **kw,
):
    session = Mock(spec=requests.Session)
    if session_post is not None:
        session.post = session_post
    client = AIGuardClient(
        base_url="https://worker.local",
        token="test-token",
        agent_id=agent_id,
        agents=agents,
        session=session,
        region=region,
        account=account,
        **kw,
    )
    if bedrock is not None:
        # Private test seam: pre-seed the enforcer's bedrock-runtime client so
        # no real boto3/AWS client is built. Mirrors how hosts get a real client
        # from the default AWS chain, without a public injection param.
        client._enforcer()._brt = bedrock
    return client


# --------------------------------------------------------------------------
# response interpretation: block vs mask vs none
# --------------------------------------------------------------------------


def test_has_blocked_across_policies():
    assert _has_blocked([{"contentPolicy": {"filters": [{"action": "BLOCKED"}]}}])
    assert _has_blocked([{"topicPolicy": {"topics": [{"action": "BLOCKED"}]}}])
    assert _has_blocked(
        [{"sensitiveInformationPolicy": {"regexes": [{"action": "BLOCKED"}]}}]
    )
    assert _has_blocked([{"wordPolicy": {"customWords": [{"action": "BLOCKED"}]}}])
    assert not _has_blocked(
        [{"sensitiveInformationPolicy": {"piiEntities": [{"action": "ANONYMIZED"}]}}]
    )
    assert not _has_blocked([])


def test_apply_one_none_mask_block():
    bd = _FakeBedrock([_none_resp()])
    blocked, safe, _, reason = apply_one(
        bd, {"guardrailId": "g", "version": "1"}, "INPUT", "hi"
    )
    assert (blocked, safe, reason) == (False, "hi", None)

    bd = _FakeBedrock([_mask_resp("m***ed")])
    blocked, safe, _, _ = apply_one(
        bd, {"guardrailId": "g", "version": "1"}, "INPUT", "orig"
    )
    assert blocked is False and safe == "m***ed"

    bd = _FakeBedrock([_block_resp("orig", "nope")])
    blocked, safe, _, reason = apply_one(
        bd, {"guardrailId": "g", "version": "1"}, "OUTPUT", "orig"
    )
    assert blocked is True and reason == "nope"


def test_check_input_none_passthrough():
    bd = _FakeBedrock([_none_resp()])
    post = Mock(
        return_value=_FakeResp(200, _ok_body([{"guardrailId": "g1", "version": "1"}]))
    )
    client = _make_client(session_post=post, bedrock=bd)
    res = client.check_input("benign")
    assert res.blocked is False
    assert res.text == "benign"
    assert res.action == "NONE"


def test_check_input_mask_forwards_masked_text():
    bd = _FakeBedrock([_mask_resp("my email is ***")])
    post = Mock(
        return_value=_FakeResp(200, _ok_body([{"guardrailId": "g1", "version": "1"}]))
    )
    client = _make_client(session_post=post, bedrock=bd)
    res = client.check_input("my email is a@b.com")
    assert res.blocked is False
    assert res.text == "my email is ***"
    assert res.action == "GUARDRAIL_INTERVENED"
    # the mask assessment is surfaced for logging/debug, not dropped
    assert res.assessments


def test_check_output_block():
    bd = _FakeBedrock([_block_resp("bad", "hate speech")])
    post = Mock(
        return_value=_FakeResp(200, _ok_body([{"guardrailId": "g1", "version": "1"}]))
    )
    client = _make_client(session_post=post, bedrock=bd)
    res = client.check_output("bad")
    assert res.blocked is True
    assert res.guardrail_id == "g1"
    assert res.action_reason == "hate speech"
    assert bd.calls[0]["source"] == "OUTPUT"


# --------------------------------------------------------------------------
# multi-guardrail composition
# --------------------------------------------------------------------------


def test_compose_masks_thread_through():
    # g1 masks "orig" -> "mask1"; g2 must receive "mask1" and masks -> "mask2"
    bd = _FakeBedrock([_mask_resp("mask1"), _mask_resp("mask2")])
    post = Mock(
        return_value=_FakeResp(
            200,
            _ok_body(
                [
                    {"guardrailId": "g1", "version": "1"},
                    {"guardrailId": "g2", "version": "1"},
                ]
            ),
        )
    )
    client = _make_client(session_post=post, bedrock=bd)
    res = client.check_input("orig")
    assert res.blocked is False
    assert res.text == "mask2"
    # g2 received g1's masked output, not the original
    assert bd.calls[1]["content"][0]["text"]["text"] == "mask1"


def test_compose_block_short_circuits():
    bd = _FakeBedrock([_block_resp("x", "denied topic")])  # only one response queued
    post = Mock(
        return_value=_FakeResp(
            200,
            _ok_body(
                [
                    {"guardrailId": "g1", "version": "1"},
                    {"guardrailId": "g2", "version": "1"},
                ]
            ),
        )
    )
    client = _make_client(session_post=post, bedrock=bd)
    res = client.check_input("x")
    assert res.blocked is True
    assert res.guardrail_id == "g1"
    assert len(bd.calls) == 1  # never called g2


# --------------------------------------------------------------------------
# chunking
# --------------------------------------------------------------------------


def test_line_chunks_splits_long_line():
    cap = 10_000
    text = "a" * (cap * 2 + 5)  # single over-long line, no newlines
    chunks = list(line_chunks(text, max_chars=cap))
    assert all(len(c) <= cap for c in chunks)
    assert "".join(chunks) == text


def test_apply_guardrail_chunked_evaluates_all_chunks():
    text = ("x" * 100 + "\n") * 400  # ~40k chars
    bd = _FakeBedrock(rule=lambda kw: _none_resp())
    blocked, safe, _, _ = apply_guardrail_chunked(
        bd, {"guardrailId": "g", "version": "1"}, "INPUT", text, cap=10_000
    )
    assert blocked is False
    assert safe == text
    assert len(bd.calls) >= 2  # chunked into multiple calls


def test_apply_guardrail_chunked_blocks_on_any_chunk():
    text = "y" * (1_000 + 10)
    calls = {"n": 0}

    def rule(kw):
        calls["n"] += 1
        return (
            _block_resp(kw["content"][0]["text"]["text"])
            if calls["n"] == 1
            else _none_resp()
        )

    bd = _FakeBedrock(rule=rule)
    blocked, _, _, _ = apply_guardrail_chunked(
        bd, {"guardrailId": "g", "version": "1"}, "INPUT", text, cap=1_000
    )
    assert blocked is True
    assert calls["n"] == 1  # sequential path short-circuits after the block


def test_apply_guardrail_chunked_parallel_preserves_order_and_blocks():
    from concurrent.futures import ThreadPoolExecutor

    text = ("x" * 900 + "\n") * 4  # 4 blocks at cap=1000, 1 block per request

    def rule(kw):
        ts = [b["text"]["text"] for b in kw["content"]]
        if any("BAD" in t for t in ts):
            return _block_resp(ts[0])
        return {
            "action": "NONE",
            "assessments": [],
            "outputs": [{"text": t[:3]} for t in ts],
        }

    bd = _FakeBedrock(rule=rule)
    with ThreadPoolExecutor(max_workers=4) as pool:
        blocked, safe, _, _ = apply_guardrail_chunked(
            bd,
            {"guardrailId": "g", "version": "1"},
            "INPUT",
            text[:2_000] + "BAD" + text[2_003:],  # marker lands in a later block
            cap=1_000,
            executor=pool,
        )
    assert blocked is True
    # parallel path checks every request (no short-circuit), output stays ordered
    assert len(bd.calls) == 4
    assert safe.startswith("xxx")  # block-ordered concat, first block masked to [:3]


def test_apply_guardrail_chunked_packs_blocks_into_fewer_calls(monkeypatch):
    # block granularity below the request budget -> several blocks per call
    import ai_guard.guard.bedrock as bedrock

    monkeypatch.setattr(bedrock, "_BLOCK_CAP", 500)
    text = ("x" * 400 + "\n") * 6  # ~2.4k -> 6 blocks of ~401 chars
    bd = _FakeBedrock(rule=lambda kw: _none_resp())
    blocked, safe, _, _ = apply_guardrail_chunked(
        bd, {"guardrailId": "g", "version": "1"}, "INPUT", text, cap=1_000
    )
    assert blocked is False
    assert safe == text
    # 6 blocks packed 2-per-request at a 1k budget -> 3 calls, not 6
    assert len(bd.calls) == 3
    assert [len(c["content"]) for c in bd.calls] == [2, 2, 2]


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------


def test_cache_hit_stale_miss():
    c = GuardrailCache(ttl_seconds=1000)
    assert c.lookup(("a",)) == (None, "miss")
    c.store(("a",), [{"guardrailId": "g"}])
    entry, fresh = c.lookup(("a",))
    assert fresh == "fresh" and entry.guardrails == [{"guardrailId": "g"}]

    c2 = GuardrailCache(ttl_seconds=0)
    c2.store(("a",), [])
    time.sleep(0.001)
    _, freshness = c2.lookup(("a",))
    assert freshness == "stale"


def test_cache_single_flight_one_resolve():
    barrier_calls = []
    lock = threading.Lock()

    def slow_post(*args, **kwargs):
        with lock:
            barrier_calls.append(1)
        time.sleep(0.05)
        return _FakeResp(200, _ok_body([{"guardrailId": "g1", "version": "1"}]))

    bd = _FakeBedrock(rule=lambda kw: _none_resp())
    client = _make_client(session_post=Mock(side_effect=slow_post), bedrock=bd)

    def worker():
        client.check_input("hi")

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(barrier_calls) == 1  # single-flight: only one resolve


def test_refresh_failure_keeps_last_known_good():
    enforcer = _make_client(
        session_post=Mock(return_value=_FakeResp(200, _ok_body([]))),
        bedrock=_FakeBedrock(),
    )._enforcer()
    key = ("agent-1", "123456789012", "us-east-1")
    enforcer._cache.store(key, [{"guardrailId": "g1", "version": "1"}])
    # background refresh raises -> must not evict
    enforcer._c._session.post = Mock(side_effect=requests.ConnectionError("down"))
    enforcer._refresh_async(key)
    time.sleep(0.05)
    entry, _ = enforcer._cache.lookup(key)
    assert entry.guardrails == [{"guardrailId": "g1", "version": "1"}]


# --------------------------------------------------------------------------
# resolver: status handling + transient classification
# --------------------------------------------------------------------------


def test_resolve_ok_and_no_guardrails():
    sess = Mock(spec=requests.Session)
    sess.post = Mock(
        return_value=_FakeResp(200, _ok_body([{"guardrailId": "g", "version": "2"}]))
    )
    status, gr = resolve(sess, "https://w", "t", "a", "123", "us-east-1")
    assert status == "OK" and gr == [{"guardrailId": "g", "version": "2"}]

    sess.post = Mock(
        return_value=_FakeResp(200, {"status": "NO_GUARDRAILS", "guardrails": []})
    )
    status, gr = resolve(sess, "https://w", "t", "a", "123", "us-east-1")
    assert status == "NO_GUARDRAILS" and gr == []


def test_resolve_transient_vs_terminal():
    sess = Mock(spec=requests.Session)
    sess.post = Mock(return_value=_FakeResp(503))
    with pytest.raises(GuardrailResolutionError) as ei:
        resolve(sess, "https://w", "t", "a", "123", "us-east-1")
    assert ei.value.transient is True

    sess.post = Mock(return_value=_FakeResp(404))
    with pytest.raises(GuardrailResolutionError) as ei:
        resolve(sess, "https://w", "t", "a", "123", "us-east-1")
    assert ei.value.transient is False

    sess.post = Mock(side_effect=requests.Timeout("t"))
    with pytest.raises(GuardrailResolutionError) as ei:
        resolve(sess, "https://w", "t", "a", "123", "us-east-1")
    assert ei.value.transient is True


# --------------------------------------------------------------------------
# status pass-through / failure policies
# --------------------------------------------------------------------------


def test_no_guardrails_passthrough():
    post = Mock(
        return_value=_FakeResp(200, {"status": "NO_GUARDRAILS", "guardrails": []})
    )
    client = _make_client(session_post=post, bedrock=_FakeBedrock())
    res = client.check_input("anything")
    assert res.blocked is False and res.action == "NONE"


def test_on_resolve_failure_block_vs_allow():
    post = Mock(return_value=_FakeResp(404))
    client = _make_client(
        session_post=post, bedrock=_FakeBedrock(), on_resolve_failure="block"
    )
    res = client.check_input("x")
    assert res.blocked is True

    post2 = Mock(return_value=_FakeResp(404))
    client2 = _make_client(
        session_post=post2, bedrock=_FakeBedrock(), on_resolve_failure="allow"
    )
    res2 = client2.check_input("x")
    assert res2.blocked is False and res2.action == "NONE"


def test_on_error_block_vs_allow():
    bd = _FakeBedrock(rule=lambda kw: (_ for _ in ()).throw(RuntimeError("boom")))
    post = Mock(
        return_value=_FakeResp(200, _ok_body([{"guardrailId": "g1", "version": "1"}]))
    )
    client = _make_client(session_post=post, bedrock=bd, on_error="block")
    res = client.check_input("x")
    assert res.blocked is True

    bd2 = _FakeBedrock(rule=lambda kw: (_ for _ in ()).throw(RuntimeError("boom")))
    post2 = Mock(
        return_value=_FakeResp(200, _ok_body([{"guardrailId": "g1", "version": "1"}]))
    )
    client2 = _make_client(session_post=post2, bedrock=bd2, on_error="allow")
    res2 = client2.check_input("x")
    assert res2.blocked is False  # continued ungoverned


def test_missing_region_fail_fast():
    client = _make_client(session_post=Mock(), bedrock=_FakeBedrock(), region=None)
    with pytest.raises(AIGuardConfigError):
        client.check_input("x")


def test_sts_failure_is_resolution_failure():
    client = _make_client(
        session_post=Mock(),
        bedrock=_FakeBedrock(),
        account=None,  # force STS
        on_resolve_failure="block",
    )
    enf = client._enforcer()
    sts = Mock()
    sts.get_caller_identity.side_effect = RuntimeError("no creds")
    enf._sts = lambda: sts  # private test seam: force STS failure
    res = client.check_input("x")
    assert res.blocked is True
    assert "STS" in res.action_reason


# --------------------------------------------------------------------------
# multi-agent: different agents resolve different lists
# --------------------------------------------------------------------------


def test_multi_agent_resolves_per_agent():
    def post(url, **kwargs):
        if "aaaa0001" in url:
            return _FakeResp(
                200, _ok_body([{"guardrailId": "g-research", "version": "1"}])
            )
        return _FakeResp(200, {"status": "NO_GUARDRAILS", "guardrails": []})

    bd = _FakeBedrock(rule=lambda kw: _mask_resp("masked"))
    client = _make_client(
        session_post=Mock(side_effect=post),
        bedrock=bd,
        agent_id=None,
        agents=[
            AgentConfig(name="researcher", agent_id="3f2a0c9e-aaaa0001"),
            AgentConfig(name="writer", agent_id="9b7c5d21-aaaa0002"),
        ],
    )
    with client.activate("researcher"):
        r = client.check_input("hello")
    assert r.text == "masked"  # researcher has a guardrail -> masked

    with client.activate("writer"):
        w = client.check_input("hello")
    assert w.blocked is False and w.action == "NONE"  # writer NO_GUARDRAILS


def test_streaming_stops_on_block():
    # first buffer masks, second buffer blocks -> stream stops on block
    def rule(kw):
        t = kw["content"][0]["text"]["text"]
        return _block_resp(t) if "BAD" in t else _mask_resp(t)

    bd = _FakeBedrock(rule=rule)
    post = Mock(
        return_value=_FakeResp(200, _ok_body([{"guardrailId": "g1", "version": "1"}]))
    )
    client = _make_client(session_post=post, bedrock=bd)
    emitted = []
    buf_cap = client.guardrail_stream_buffer_size
    tokens = ["a" * buf_cap, "BAD" + "b" * buf_cap]
    res = client.check_output_stream(iter(tokens), emitted.append)
    assert res.blocked is True
    assert len(emitted) == 1  # only the first (cleared) buffer was emitted


def test_intervened_response_missing_outputs_fails_closed():
    # guardrail intervened but returned no masked output for a block ->
    # fail closed instead of echoing the original (possibly unmasked) text
    resp = {
        "action": "GUARDRAIL_INTERVENED",
        "assessments": [
            {"sensitiveInformationPolicy": {"piiEntities": [{"action": "ANONYMIZED"}]}}
        ],
        "outputs": [],  # malformed: intervened but nothing to safely forward
    }
    bd = _FakeBedrock([resp])
    post = Mock(
        return_value=_FakeResp(200, _ok_body([{"guardrailId": "g1", "version": "1"}]))
    )
    client = _make_client(session_post=post, bedrock=bd)
    res = client.check_output("email me at a@b.com")
    assert res.blocked is True


def test_resolve_malformed_body_is_terminal():
    sess = Mock(spec=requests.Session)
    sess.post = Mock(return_value=_FakeResp(200, json_error=True))
    with pytest.raises(GuardrailResolutionError) as ei:
        resolve(sess, "https://w", "t", "a", "123", "us-east-1")
    assert ei.value.transient is False


def test_malformed_body_fails_closed_not_crash():
    # A 200 with an unparseable body must fail closed (block), never crash the turn.
    post = Mock(return_value=_FakeResp(200, json_error=True))
    client = _make_client(
        session_post=post, bedrock=_FakeBedrock(), on_resolve_failure="block"
    )
    res = client.check_input("x")
    assert res.blocked is True


def test_config_error_in_enforce_propagates_even_with_on_error_allow():
    # A misconfiguration surfaced during enforcement must fail fast, not be
    # swallowed by on_error="allow".
    from ai_guard.guard.exceptions import AIGuardConfigError as _CfgErr

    def raise_cfg(**kw):
        raise _CfgErr("boto3 missing")

    bd = _FakeBedrock()
    bd.apply_guardrail = raise_cfg
    post = Mock(
        return_value=_FakeResp(200, _ok_body([{"guardrailId": "g1", "version": "1"}]))
    )
    client = _make_client(session_post=post, bedrock=bd, on_error="allow")
    with pytest.raises(AIGuardConfigError):
        client.check_input("x")


def test_refresh_is_single_flight_per_key():
    # Many stale checks must not spawn duplicate background refreshes.
    started = []
    release = threading.Event()

    def slow_post(*args, **kwargs):
        started.append(1)
        release.wait(0.5)
        return _FakeResp(200, _ok_body([{"guardrailId": "g1", "version": "1"}]))

    client = _make_client(
        session_post=Mock(side_effect=slow_post),
        bedrock=_FakeBedrock(rule=lambda kw: _none_resp()),
        guardrail_cache_ttl=0,  # everything is immediately stale
    )
    enf = client._enforcer()
    key = ("agent-1", "123456789012", "us-east-1")
    enf._cache.store(key, [{"guardrailId": "g1", "version": "1"}])
    # Fire several stale reads concurrently; only one refresh should be in flight.
    for _ in range(6):
        enf._refresh_async(key)
    time.sleep(0.05)
    assert sum(started) == 1
    release.set()
    enf.close()


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"on_error": "deny"}, "on_error must be 'block' or 'allow'"),
        (
            {"on_resolve_failure": "nope"},
            "on_resolve_failure must be 'block' or 'allow'",
        ),
        ({"guardrail_retry_interval": 0}, "guardrail_retry_interval must be > 0"),
        ({"guardrail_resolve_timeout": 0}, "guardrail_resolve_timeout must be > 0"),
        (
            {"guardrail_resolve_max_attempts": 0},
            "guardrail_resolve_max_attempts must be >= 1",
        ),
        (
            {"guardrail_resolve_retry_backoff": -0.1},
            "guardrail_resolve_retry_backoff must be >= 0",
        ),
        ({"guardrail_cache_ttl": -1}, "guardrail_cache_ttl must be >= 0"),
        ({"guardrail_chunk_size": 0}, "guardrail_chunk_size must be > 0"),
        (
            {"guardrail_stream_buffer_size": 0},
            "guardrail_stream_buffer_size must be > 0",
        ),
    ],
)
def test_invalid_enforcement_config_fails_fast(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _make_client(session_post=Mock(), bedrock=_FakeBedrock(), **kwargs)


@pytest.mark.parametrize("bad", [None, "", "   ", "/", " // "])
def test_missing_base_url_fails_fast(bad):
    with pytest.raises(ValueError, match="base_url is required"):
        AIGuardClient(base_url=bad, token="t", agent_id="a")


def test_check_input_retries_transient_resolve():
    calls = {"n": 0}

    def flaky(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeResp(503)  # transient
        return _FakeResp(200, _ok_body([{"guardrailId": "g1", "version": "1"}]))

    bd = _FakeBedrock([_none_resp()])
    client = _make_client(session_post=flaky, bedrock=bd)
    res = client.check_input("hi")
    assert res.blocked is False
    assert calls["n"] == 2  # one transient failure, one successful retry


def test_check_input_resolve_max_attempts_disables_retry():
    calls = {"n": 0}

    def always_flaky(*a, **kw):
        calls["n"] += 1
        return _FakeResp(503)

    bd = _FakeBedrock([_none_resp()])
    client = _make_client(
        session_post=always_flaky,
        bedrock=bd,
        guardrail_resolve_max_attempts=1,
        on_resolve_failure="allow",
    )
    res = client.check_input("hi")
    assert res.blocked is False
    assert calls["n"] == 1  # no retry


def test_enforcer_direct_construction():
    # GuardrailEnforcer binds to a client and exposes the public entry points
    client = _make_client(
        session_post=Mock(return_value=_FakeResp(200, _ok_body([]))),
        bedrock=_FakeBedrock(),
    )
    enf = GuardrailEnforcer(client)
    assert isinstance(enf.check_input("x"), GuardResult)
