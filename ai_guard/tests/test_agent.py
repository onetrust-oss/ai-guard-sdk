import asyncio
import inspect
import threading
import warnings
from unittest.mock import Mock, patch

import pytest
import requests

import ai_guard.otel as otel_module
from ai_guard import AgentConfig, AIGuardClient, get_active_agent_id
from ai_guard.agent import build_agent_registry
from ai_guard.api import (
    AIPlatform,
    ClassificationRequest,
    ClassifierDescriptionProfile,
    MetricsEvent,
    MetricsEventMeter,
)


def _mock_response(status_code=200, json_body=None):
    resp = Mock()
    resp.status_code = status_code
    resp.headers = {"Content-Type": "application/json"}
    resp.json.return_value = (
        json_body
        if json_body is not None
        else {
            "context": None,
            "matches": [],
        }
    )
    return resp


def _make_client(
    agent_id="default-agent",
    platform=AIPlatform.AMAZON_BEDROCK,
    agents=None,
    side_effect=None,
):
    mock_session = Mock()
    if side_effect is not None:
        mock_session.post.side_effect = side_effect
    else:
        mock_session.post.return_value = _mock_response()
    client = AIGuardClient(
        "http://svc",
        token="token",
        agent_id=agent_id,
        platform=platform,
        session=mock_session,
        agents=agents,
    )
    return client, mock_session


def _sent_context(mock_session, call_index=0):
    payload = mock_session.post.call_args_list[call_index].kwargs["json"]
    return payload["context"]


def _sent_attributes(mock_session, call_index=0):
    payload = mock_session.post.call_args_list[call_index].kwargs["json"]
    return payload["attributes"]


def _classify_req(context=None):
    return ClassificationRequest(
        context=context,
        classifier_description=ClassifierDescriptionProfile(uuid="u", version=1),
        text="hello",
    )


def _metric_event(attributes=None):
    return MetricsEvent(
        attributes=attributes if attributes is not None else {},
        meter=MetricsEventMeter(name="ai_guard.agent", value="1"),
    )


# --------------------------------------------------------------------------
# Registration: valid + malformed/incomplete/duplicate/unsupported manifests
# --------------------------------------------------------------------------


class TestAgentRegistration:
    def test_none_agents_yields_empty_registry(self):
        assert build_agent_registry(None) == {}

    def test_valid_agentconfig_instances(self):
        registry = build_agent_registry(
            [
                AgentConfig(name="a", agent_id="id-a"),
                AgentConfig(name="b", agent_id="id-b"),
            ]
        )
        assert set(registry) == {"a", "b"}
        assert registry["a"].agent_id == "id-a"

    def test_valid_mapping_entries(self):
        registry = build_agent_registry([{"name": "a", "agent_id": "id-a"}])
        assert registry["a"].agent_id == "id-a"

    def test_missing_field_raises(self):
        with pytest.raises(ValueError, match="missing required field"):
            build_agent_registry([{"name": "a"}])

    def test_incomplete_empty_mapping_raises(self):
        with pytest.raises(ValueError, match="missing required field"):
            build_agent_registry([{}])

    def test_duplicate_name_raises(self):
        with pytest.raises(ValueError, match="duplicate agent name"):
            build_agent_registry(
                [
                    {"name": "a", "agent_id": "id-1"},
                    {"name": "a", "agent_id": "id-2"},
                ]
            )

    def test_unsupported_container_type_raises(self):
        with pytest.raises(ValueError, match="agents must be a sequence"):
            build_agent_registry("agents.yaml")

    def test_malformed_entry_type_raises(self):
        with pytest.raises(ValueError, match="must be an AgentConfig or a mapping"):
            build_agent_registry([42])

    def test_agentconfig_rejects_empty_name(self):
        with pytest.raises(ValueError, match="name must be a non-empty string"):
            AgentConfig(name="", agent_id="id")

    def test_agentconfig_rejects_empty_agent_id(self):
        with pytest.raises(ValueError, match="agent_id must be a non-empty string"):
            AgentConfig(name="a", agent_id="")

    def test_client_construction_rejects_bad_manifest(self):
        with pytest.raises(ValueError):
            _make_client(agents=[{"name": "a"}])


# --------------------------------------------------------------------------
# Backward compatibility: existing single-agent construction/usage unchanged
# --------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_construction_without_agents_param_unaffected(self):
        client, mock_session = _make_client()
        client.classify(_classify_req())
        assert _sent_context(mock_session) == {
            "agent_id": "default-agent",
            "platform": AIPlatform.AMAZON_BEDROCK,
        }

    def test_classify_without_activation_uses_constructor_default(self):
        client, mock_session = _make_client(
            agents=[AgentConfig(name="other", agent_id="other-id")]
        )
        client.classify(_classify_req())
        assert _sent_context(mock_session)["agent_id"] == "default-agent"

    def test_metric_without_activation_uses_constructor_default(self):
        client, mock_session = _make_client()
        client.metric(_metric_event())
        assert _sent_attributes(mock_session)["agent_id"] == "default-agent"


# --------------------------------------------------------------------------
# platform and agent_id are both optional on AIGuardClient
# --------------------------------------------------------------------------


class TestOptionalConstructorFields:
    def test_platform_omitted_when_not_configured(self):
        client, mock_session = _make_client(platform=None)
        client.classify(_classify_req())
        assert _sent_context(mock_session) == {"agent_id": "default-agent"}

    def test_platform_shared_across_all_agents_when_configured(self):
        client, mock_session = _make_client(
            platform=AIPlatform.AZURE_FOUNDRY,
            agents=[AgentConfig(name="support", agent_id="acme/support")],
        )
        with client.activate("support"):
            client.classify(_classify_req())
        assert _sent_context(mock_session) == {
            "agent_id": "acme/support",
            "platform": AIPlatform.AZURE_FOUNDRY,
        }

    def test_no_default_agent_id_and_no_activation_raises(self):
        client, _ = _make_client(
            agent_id=None,
            platform=None,
            agents=[AgentConfig(name="support", agent_id="acme/support")],
        )
        with pytest.raises(RuntimeError, match="No agent activated"):
            client.classify(_classify_req())

    def test_no_default_agent_id_but_activated_works(self):
        client, mock_session = _make_client(
            agent_id=None,
            platform=None,
            agents=[AgentConfig(name="support", agent_id="acme/support")],
        )
        with client.activate("support"):
            client.classify(_classify_req())
        assert _sent_context(mock_session) == {"agent_id": "acme/support"}

    def test_no_default_agent_id_metric_also_raises(self):
        client, _ = _make_client(agent_id=None, platform=None)
        with pytest.raises(RuntimeError, match="No agent activated"):
            client.metric(_metric_event())


# --------------------------------------------------------------------------
# Activation: valid activation, unknown agent, nesting, restoration
# --------------------------------------------------------------------------


class TestActivation:
    def _client(self):
        return _make_client(
            agents=[
                AgentConfig(name="support", agent_id="acme/support"),
                AgentConfig(name="research", agent_id="acme/research"),
            ]
        )

    def test_activation_switches_identity(self):
        client, mock_session = self._client()
        with client.activate("support"):
            client.classify(_classify_req())
        assert _sent_context(mock_session) == {
            "agent_id": "acme/support",
            "platform": AIPlatform.AMAZON_BEDROCK,
        }

    def test_unknown_agent_raises_value_error(self):
        client, _ = self._client()
        with pytest.raises(ValueError, match="Unknown agent"):
            client.activate("does-not-exist")

    def test_nested_activation_restores_outer(self):
        client, mock_session = self._client()
        with client.activate("support"):
            with client.activate("research"):
                client.classify(_classify_req())
            client.classify(_classify_req())
        assert _sent_context(mock_session, 0)["agent_id"] == "acme/research"
        assert _sent_context(mock_session, 1)["agent_id"] == "acme/support"

    def test_restored_to_default_after_activation_exits(self):
        client, mock_session = self._client()
        with client.activate("support"):
            pass
        client.classify(_classify_req())
        assert _sent_context(mock_session)["agent_id"] == "default-agent"

    def test_restored_after_exception(self):
        client, mock_session = self._client()
        with pytest.raises(RuntimeError), client.activate("support"):
            raise RuntimeError("boom")
        client.classify(_classify_req())
        assert _sent_context(mock_session)["agent_id"] == "default-agent"

    def test_restored_after_timeout_error(self):
        def side_effect(*args, **kwargs):
            raise requests.exceptions.Timeout("timed out")

        client, mock_session = self._client()
        mock_session.post.side_effect = side_effect
        with pytest.raises(requests.exceptions.Timeout), client.activate("support"):
            client.classify(_classify_req())

        mock_session.post.side_effect = None
        mock_session.post.return_value = _mock_response()
        client.classify(_classify_req())
        assert _sent_context(mock_session, 1)["agent_id"] == "default-agent"

    def test_two_clients_do_not_cross_contaminate(self):
        client_a, session_a = self._client()
        client_b, session_b = _make_client()

        with client_a.activate("support"):
            client_b.classify(_classify_req())
        assert _sent_context(session_b)["agent_id"] == "default-agent"

    def test_reusing_request_object_across_agents_no_leakage(self):
        client, mock_session = self._client()
        shared_request = _classify_req()

        with client.activate("support"):
            client.classify(shared_request)
        with client.activate("research"):
            client.classify(shared_request)

        assert _sent_context(mock_session, 0)["agent_id"] == "acme/support"
        assert _sent_context(mock_session, 1)["agent_id"] == "acme/research"
        assert shared_request.context is None

    def test_reusing_metrics_event_across_agents_no_leakage(self):
        client, mock_session = self._client()
        shared_event = _metric_event()

        with client.activate("support"):
            client.metric(shared_event)
        with client.activate("research"):
            client.metric(shared_event)

        assert _sent_attributes(mock_session, 0)["agent_id"] == "acme/support"
        assert _sent_attributes(mock_session, 1)["agent_id"] == "acme/research"
        assert shared_event.attributes == {}


# --------------------------------------------------------------------------
# Decorator activation: @client.agent(name) as shorthand for wrapping a whole
# function body in `with client.activate(name):`
# --------------------------------------------------------------------------


class TestAgentDecorator:
    def _client(self):
        return _make_client(
            agents=[
                AgentConfig(name="support", agent_id="acme/support"),
                AgentConfig(name="research", agent_id="acme/research"),
            ]
        )

    # -- basics ------------------------------------------------------------

    def test_sync_function_uses_decorated_agent(self):
        client, mock_session = self._client()

        @client.agent("support")
        def handler():
            client.classify(_classify_req())

        handler()
        assert _sent_context(mock_session) == {
            "agent_id": "acme/support",
            "platform": AIPlatform.AMAZON_BEDROCK,
        }

    def test_restored_to_default_after_return(self):
        client, mock_session = self._client()

        @client.agent("support")
        def handler():
            client.classify(_classify_req())

        handler()
        client.classify(_classify_req())
        assert _sent_context(mock_session, 0)["agent_id"] == "acme/support"
        assert _sent_context(mock_session, 1)["agent_id"] == "default-agent"

    def test_restored_after_exception(self):
        client, mock_session = self._client()

        @client.agent("support")
        def handler():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            handler()
        assert get_active_agent_id() is None
        client.classify(_classify_req())
        assert _sent_context(mock_session)["agent_id"] == "default-agent"

    def test_args_kwargs_and_return_value_pass_through(self):
        client, _ = self._client()

        @client.agent("support")
        def handler(a, b, *, c=3):
            return (a, b, c, get_active_agent_id())

        assert handler(1, 2, c=9) == (1, 2, 9, "acme/support")

    def test_unknown_agent_raises_at_decoration_time(self):
        client, _ = self._client()
        with pytest.raises(ValueError, match="Unknown agent"):
            client.agent("does-not-exist")

    def test_same_function_reusable_across_many_calls(self):
        client, mock_session = self._client()

        @client.agent("support")
        def handler():
            client.classify(_classify_req())

        for _ in range(5):
            handler()
        assert [_sent_context(mock_session, i)["agent_id"] for i in range(5)] == [
            "acme/support"
        ] * 5

    def test_decorator_on_instance_method(self):
        client, _ = self._client()

        class Service:
            @client.agent("research")
            def run(self, value):
                return (value, get_active_agent_id())

        assert Service().run(7) == (7, "acme/research")

    # -- introspection: frameworks inspect what they are handed -------------

    def test_preserves_name_doc_and_signature(self):
        client, _ = self._client()

        @client.agent("support")
        def handler(a: int, b: str = "x") -> None:
            """Docstring."""

        assert handler.__name__ == "handler"
        assert handler.__doc__ == "Docstring."
        assert str(inspect.signature(handler)) == "(a: int, b: str = 'x') -> None"

    def test_preserves_function_kind_flags(self):
        client, _ = self._client()

        @client.agent("support")
        def sync_fn():
            pass

        @client.agent("support")
        async def async_fn():
            pass

        @client.agent("support")
        def gen_fn():
            yield 1

        @client.agent("support")
        async def agen_fn():
            yield 1

        assert not inspect.iscoroutinefunction(sync_fn)
        assert inspect.iscoroutinefunction(async_fn)
        assert inspect.isgeneratorfunction(gen_fn)
        assert inspect.isasyncgenfunction(agen_fn)

    # -- coroutine functions -----------------------------------------------

    def test_async_function_active_for_whole_await(self):
        client, mock_session = self._client()

        @client.agent("support")
        async def handler():
            await asyncio.sleep(0)
            client.classify(_classify_req())
            return get_active_agent_id()

        assert asyncio.run(handler()) == "acme/support"
        assert _sent_context(mock_session)["agent_id"] == "acme/support"

    def test_async_function_restores_after_exception(self):
        client, _ = self._client()

        @client.agent("support")
        async def handler():
            raise RuntimeError("boom")

        async def main():
            with pytest.raises(RuntimeError, match="boom"):
                await handler()
            return get_active_agent_id()

        assert asyncio.run(main()) is None

    # -- generator functions: active during *iteration*, not just the call --

    def test_sync_generator_active_during_iteration(self):
        client, mock_session = self._client()

        @client.agent("support")
        def handler(items):
            for item in items:
                client.classify(_classify_req())
                yield (item, get_active_agent_id())

        # Create the generator first; it must not activate anything yet.
        gen = handler(["a", "b"])
        assert get_active_agent_id() is None

        # Activation happens on iteration, and covers every yield.
        assert list(gen) == [("a", "acme/support"), ("b", "acme/support")]
        assert get_active_agent_id() is None
        assert _sent_context(mock_session, 0)["agent_id"] == "acme/support"
        assert _sent_context(mock_session, 1)["agent_id"] == "acme/support"

    def test_sync_generator_forwards_send(self):
        client, _ = self._client()

        @client.agent("support")
        def handler():
            received = yield get_active_agent_id()
            yield f"{received}:{get_active_agent_id()}"

        gen = handler()
        assert next(gen) == "acme/support"
        assert gen.send("hello") == "hello:acme/support"

    def test_sync_generator_abandoned_midway_restores(self):
        client, _ = self._client()

        @client.agent("support")
        def handler():
            yield get_active_agent_id()
            yield get_active_agent_id()

        gen = handler()
        assert next(gen) == "acme/support"
        gen.close()  # abandoned before exhaustion
        assert get_active_agent_id() is None

    def test_async_generator_active_during_iteration(self):
        client, mock_session = self._client()

        @client.agent("support")
        async def handler(items):
            for item in items:
                await asyncio.sleep(0)
                client.classify(_classify_req())
                yield (item, get_active_agent_id())

        async def main():
            agen = handler(["a", "b"])
            assert get_active_agent_id() is None
            return [item async for item in agen]

        assert asyncio.run(main()) == [
            ("a", "acme/support"),
            ("b", "acme/support"),
        ]
        assert _sent_context(mock_session, 0)["agent_id"] == "acme/support"
        assert _sent_context(mock_session, 1)["agent_id"] == "acme/support"

    # -- composition with activate() and with other decorated functions ----

    def test_with_activate_nests_inside_decorated_function(self):
        client, mock_session = self._client()

        @client.agent("support")
        def handler():
            client.classify(_classify_req())  # support
            with client.activate("research"):
                client.classify(_classify_req())  # research
            client.classify(_classify_req())  # support restored

        handler()
        assert [_sent_context(mock_session, i)["agent_id"] for i in range(3)] == [
            "acme/support",
            "acme/research",
            "acme/support",
        ]

    def test_decorated_function_calling_decorated_function(self):
        client, mock_session = self._client()

        @client.agent("research")
        def inner():
            client.classify(_classify_req())

        @client.agent("support")
        def outer():
            inner()
            client.classify(_classify_req())

        outer()
        assert _sent_context(mock_session, 0)["agent_id"] == "acme/research"
        assert _sent_context(mock_session, 1)["agent_id"] == "acme/support"

    def test_decorated_function_inside_activate_block(self):
        client, mock_session = self._client()

        @client.agent("research")
        def handler():
            client.classify(_classify_req())

        with client.activate("support"):
            handler()
            client.classify(_classify_req())

        assert _sent_context(mock_session, 0)["agent_id"] == "acme/research"
        assert _sent_context(mock_session, 1)["agent_id"] == "acme/support"

    def test_two_clients_do_not_cross_contaminate(self):
        client_a, _ = self._client()
        client_b, session_b = _make_client()

        @client_a.agent("support")
        def handler():
            client_b.classify(_classify_req())

        handler()
        assert _sent_context(session_b)["agent_id"] == "default-agent"


# --------------------------------------------------------------------------
# Decorator concurrency: one decorated function object, many simultaneous
# callers. Regression guard for sharing a single AgentActivation (and hence a
# single single-use contextvars Token) across calls.
# --------------------------------------------------------------------------


class TestAgentDecoratorConcurrency:
    def _client(self):
        return _make_client(
            agents=[
                AgentConfig(name="support", agent_id="acme/support"),
                AgentConfig(name="research", agent_id="acme/research"),
            ]
        )

    def test_same_decorated_function_across_overlapping_threads(self):
        client, _ = self._client()
        thread_count = 8
        barrier = threading.Barrier(thread_count)
        seen = {}
        errors = []

        @client.agent("support")
        def handler(index):
            barrier.wait()  # force all calls to overlap inside the activation
            return get_active_agent_id()

        def run(index):
            try:
                seen[index] = handler(index)
            except Exception as exc:  # noqa: BLE001 - surfaced by the assert
                errors.append(exc)

        threads = [threading.Thread(target=run, args=(i,)) for i in range(thread_count)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert errors == []
        assert seen == {i: "acme/support" for i in range(thread_count)}
        assert get_active_agent_id() is None

    def test_different_decorated_functions_across_overlapping_threads(self):
        client, _ = self._client()
        barrier = threading.Barrier(2)
        seen = {}

        @client.agent("support")
        def support_handler():
            barrier.wait()
            seen["support"] = get_active_agent_id()

        @client.agent("research")
        def research_handler():
            barrier.wait()
            seen["research"] = get_active_agent_id()

        t1 = threading.Thread(target=support_handler)
        t2 = threading.Thread(target=research_handler)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert seen == {"support": "acme/support", "research": "acme/research"}

    def test_same_decorated_function_across_interleaved_asyncio_tasks(self):
        client, _ = self._client()

        @client.agent("support")
        async def handler(index):
            await asyncio.sleep(0)
            before = get_active_agent_id()
            await asyncio.sleep(0)
            return (index, before, get_active_agent_id())

        async def main():
            return await asyncio.gather(*(handler(i) for i in range(8)))

        results = asyncio.run(main())
        assert results == [(i, "acme/support", "acme/support") for i in range(8)]
        assert get_active_agent_id() is None

    def test_decorated_async_task_cancellation_does_not_leak(self):
        client, mock_session = self._client()

        @client.agent("support")
        async def handler():
            await asyncio.sleep(10)

        async def main():
            task = asyncio.ensure_future(handler())
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            client.classify(_classify_req())

        asyncio.run(main())
        assert _sent_context(mock_session)["agent_id"] == "default-agent"


# --------------------------------------------------------------------------
# Decorator + OTel: identical stamping and warning behavior to activate()
# --------------------------------------------------------------------------


class TestAgentDecoratorOtel:
    def _client(self):
        return _make_client(
            agents=[AgentConfig(name="support", agent_id="acme/support")]
        )

    def test_stamps_reference_id_inside_decorated_function(self):
        client, _ = self._client()
        span = Mock()
        span.is_recording.return_value = True

        @client.agent("support")
        def handler():
            client.classify(_classify_req())

        with patch.object(otel_module, "_trace") as mock_trace:
            mock_trace.get_current_span.return_value = span
            handler()

        span.set_attribute.assert_called_once_with(
            "onetrust.reference.id", "acme/support"
        )

    def test_span_processor_sees_agent_inside_decorated_function(self):
        client, _ = self._client()
        processor = otel_module.OneTrustSpanProcessor()
        span = Mock()
        span.is_recording.return_value = True

        @client.agent("support")
        def handler():
            processor.on_start(span)

        handler()
        span.set_attribute.assert_called_once_with(
            "onetrust.reference.id", "acme/support"
        )

    def test_no_tracer_provider_warning_fires_at_call_not_decoration(self):
        client, _ = self._client()
        otel_module._warned_no_tracer_provider = False
        with patch.object(otel_module, "_trace") as mock_trace:
            mock_trace.get_tracer_provider.return_value = object()

            # Decoration alone must not warn -- a TracerProvider is normally
            # configured in main(), long after module import time.
            with warnings.catch_warnings():
                warnings.simplefilter("error")

                @client.agent("support")
                def handler():
                    pass

            with pytest.warns(UserWarning, match="no OpenTelemetry TracerProvider"):
                handler()


# --------------------------------------------------------------------------
# get_active_agent_id(): for host-owned instrumentation (e.g. a custom
# SpanProcessor) that wants to act on every span while an agent is active,
# not just at classify()/metric() call sites.
# --------------------------------------------------------------------------


class TestGetActiveAgentId:
    def test_none_outside_activation(self):
        assert get_active_agent_id() is None

    def test_returns_active_agent_id(self):
        client, _ = _make_client(
            agents=[AgentConfig(name="support", agent_id="acme/support")]
        )
        with client.activate("support"):
            assert get_active_agent_id() == "acme/support"
        assert get_active_agent_id() is None

    def test_reflects_nested_activation(self):
        client, _ = _make_client(
            agents=[
                AgentConfig(name="support", agent_id="acme/support"),
                AgentConfig(name="research", agent_id="acme/research"),
            ]
        )
        with client.activate("support"):
            with client.activate("research"):
                assert get_active_agent_id() == "acme/research"
            assert get_active_agent_id() == "acme/support"


# --------------------------------------------------------------------------
# Caller-owned objects are never mutated
# --------------------------------------------------------------------------


class TestNoMutation:
    def test_classification_request_context_not_mutated(self):
        client, _ = _make_client()
        original_context = {"actor": "user"}
        request = _classify_req(context=original_context)
        client.classify(request)
        assert request.context is original_context
        assert original_context == {"actor": "user"}

    def test_none_context_stays_none(self):
        client, _ = _make_client()
        request = _classify_req(context=None)
        client.classify(request)
        assert request.context is None

    def test_metrics_event_attributes_not_mutated(self):
        client, _ = _make_client()
        original_attributes = {"key": "value"}
        event = _metric_event(attributes=original_attributes)
        client.metric(event)
        assert event.attributes is original_attributes
        assert original_attributes == {"key": "value"}


# --------------------------------------------------------------------------
# Concurrency: threads and asyncio tasks
# --------------------------------------------------------------------------


class TestConcurrency:
    def test_two_threads_different_agents_no_cross_contamination(self):
        client, mock_session = _make_client(
            agents=[
                AgentConfig(name="support", agent_id="acme/support"),
                AgentConfig(name="research", agent_id="acme/research"),
            ]
        )
        lock = threading.Lock()
        seen = {}
        barrier = threading.Barrier(2)

        def run(name):
            with client.activate(name):
                barrier.wait()  # force overlap between the two threads
                with lock:
                    client.classify(_classify_req())
                    seen[name] = _sent_context(
                        mock_session, len(mock_session.post.call_args_list) - 1
                    )["agent_id"]

        t1 = threading.Thread(target=run, args=("support",))
        t2 = threading.Thread(target=run, args=("research",))
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert seen == {"support": "acme/support", "research": "acme/research"}

    def test_two_asyncio_tasks_different_agents_no_cross_contamination(self):
        client, mock_session = _make_client(
            agents=[
                AgentConfig(name="support", agent_id="acme/support"),
                AgentConfig(name="research", agent_id="acme/research"),
            ]
        )
        results = {}

        async def run(name):
            with client.activate(name):
                await asyncio.sleep(0)  # yield control so tasks interleave
                client.classify(_classify_req())
                idx = len(mock_session.post.call_args_list) - 1
                results[name] = _sent_context(mock_session, idx)["agent_id"]

        async def main():
            await asyncio.gather(run("support"), run("research"))

        asyncio.run(main())

        assert results == {"support": "acme/support", "research": "acme/research"}

    def test_asyncio_task_cancellation_does_not_leak_activation(self):
        client, mock_session = _make_client(
            agents=[AgentConfig(name="support", agent_id="acme/support")]
        )

        async def cancels_itself():
            with client.activate("support"):
                await asyncio.sleep(10)

        async def main():
            task = asyncio.ensure_future(cancels_itself())
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            # Main task's own context was never activated; must still resolve
            # to the client's default agent.
            client.classify(_classify_req())

        asyncio.run(main())
        assert _sent_context(mock_session)["agent_id"] == "default-agent"


# --------------------------------------------------------------------------
# OTEL: onetrust.reference.id span stamping
# --------------------------------------------------------------------------


class TestOtelStamping:
    def _client(self):
        return _make_client(
            agents=[
                AgentConfig(name="support", agent_id="acme/support"),
                AgentConfig(name="research", agent_id="acme/research"),
            ]
        )

    def _mock_span(self):
        span = Mock()
        span.is_recording.return_value = True
        return span

    def test_no_stamp_outside_activation_scope(self):
        client, _ = self._client()
        span = self._mock_span()
        with patch.object(otel_module, "_trace") as mock_trace:
            mock_trace.get_current_span.return_value = span
            client.classify(_classify_req())
        span.set_attribute.assert_not_called()

    def test_stamps_reference_id_while_activated(self):
        client, _ = self._client()
        span = self._mock_span()
        with patch.object(otel_module, "_trace") as mock_trace:
            mock_trace.get_current_span.return_value = span
            with client.activate("support"):
                client.classify(_classify_req())
        span.set_attribute.assert_called_once_with(
            "onetrust.reference.id", "acme/support"
        )

    def test_different_activation_scopes_produce_different_reference_ids(self):
        client, _ = self._client()
        span = self._mock_span()
        with patch.object(otel_module, "_trace") as mock_trace:
            mock_trace.get_current_span.return_value = span
            with client.activate("support"):
                client.classify(_classify_req())
            with client.activate("research"):
                client.classify(_classify_req())
        span.set_attribute.assert_any_call("onetrust.reference.id", "acme/support")
        span.set_attribute.assert_any_call("onetrust.reference.id", "acme/research")
        assert span.set_attribute.call_count == 2

    def test_nested_scopes_stamp_correct_id_and_restore(self):
        client, _ = self._client()
        span = self._mock_span()
        with patch.object(otel_module, "_trace") as mock_trace:
            mock_trace.get_current_span.return_value = span
            with client.activate("support"):
                with client.activate("research"):
                    client.classify(_classify_req())  # inner: research
                client.classify(_classify_req())  # outer restored: support
        assert span.set_attribute.call_args_list[0].args == (
            "onetrust.reference.id",
            "acme/research",
        )
        assert span.set_attribute.call_args_list[1].args == (
            "onetrust.reference.id",
            "acme/support",
        )

    def test_non_recording_span_not_stamped(self):
        client, _ = self._client()
        span = self._mock_span()
        span.is_recording.return_value = False
        with patch.object(otel_module, "_trace") as mock_trace:
            mock_trace.get_current_span.return_value = span
            with client.activate("support"):
                client.classify(_classify_req())
        span.set_attribute.assert_not_called()

    def test_otel_unavailable_is_a_safe_no_op(self):
        client, mock_session = self._client()
        with (
            patch.object(otel_module, "_OTEL_AVAILABLE", False),
            client.activate("support"),
        ):
            client.classify(_classify_req())
        assert _sent_context(mock_session)["agent_id"] == "acme/support"

    def test_metric_also_stamps_reference_id(self):
        client, _ = self._client()
        span = self._mock_span()
        with patch.object(otel_module, "_trace") as mock_trace:
            mock_trace.get_current_span.return_value = span
            with client.activate("support"):
                client.metric(_metric_event())
        span.set_attribute.assert_called_once_with(
            "onetrust.reference.id", "acme/support"
        )


# --------------------------------------------------------------------------
# warn_if_no_tracer_provider: one-time warning when activate() can't
# possibly stamp anything because no real TracerProvider is configured.
# --------------------------------------------------------------------------


class TestWarnIfNoTracerProvider:
    def _client(self):
        agents = [AgentConfig(name="support", agent_id="acme/support")]
        return _make_client(agents=agents)

    def setup_method(self):
        otel_module._warned_no_tracer_provider = False

    def test_warns_when_otel_available_but_no_real_provider(self):
        client = self._client()[0]
        with patch.object(otel_module, "_trace") as mock_trace:
            mock_trace.get_tracer_provider.return_value = object()
            with (
                pytest.warns(UserWarning, match="no OpenTelemetry TracerProvider"),
                client.activate("support"),
            ):
                pass

    def test_no_warning_when_otel_unavailable(self):
        client = self._client()[0]
        with (
            patch.object(otel_module, "_OTEL_AVAILABLE", False),
            client.activate("support"),
        ):
            pass

    def test_no_warning_when_real_provider_configured(self):
        client = self._client()[0]
        with patch.object(otel_module, "_trace") as mock_trace:
            mock_trace.get_tracer_provider.return_value = Mock()
            with client.activate("support"):
                pass

    def test_warns_only_once_per_process(self):
        client = self._client()[0]
        with patch.object(otel_module, "_trace") as mock_trace:
            mock_trace.get_tracer_provider.return_value = object()
            with pytest.warns(UserWarning), client.activate("support"):
                pass
            with client.activate("support"):
                pass


# --------------------------------------------------------------------------
# OneTrustSpanProcessor: stamps every span while an agent is active, not just
# the one current at a classify()/metric() call site.
# --------------------------------------------------------------------------


class TestOneTrustSpanProcessor:
    def _client(self):
        return _make_client(
            agents=[
                AgentConfig(name="support", agent_id="acme/support"),
                AgentConfig(name="research", agent_id="acme/research"),
            ]
        )

    def _mock_span(self):
        span = Mock()
        span.is_recording.return_value = True
        return span

    def test_requires_opentelemetry_sdk(self):
        with (
            patch.object(otel_module, "_OTEL_SDK_AVAILABLE", False),
            pytest.raises(ImportError, match="opentelemetry-sdk"),
        ):
            otel_module.OneTrustSpanProcessor()

    def test_stamps_span_while_active(self):
        client, _ = self._client()
        processor = otel_module.OneTrustSpanProcessor()
        span = self._mock_span()
        with client.activate("support"):
            processor.on_start(span)
        span.set_attribute.assert_called_once_with(
            "onetrust.reference.id", "acme/support"
        )

    def test_does_not_stamp_outside_activation(self):
        processor = otel_module.OneTrustSpanProcessor()
        span = self._mock_span()
        processor.on_start(span)
        span.set_attribute.assert_not_called()

    def test_reflects_nested_activation_at_span_creation_time(self):
        client, _ = self._client()
        processor = otel_module.OneTrustSpanProcessor()
        outer_span, inner_span = self._mock_span(), self._mock_span()
        with client.activate("support"):
            processor.on_start(outer_span)
            with client.activate("research"):
                processor.on_start(inner_span)
        outer_span.set_attribute.assert_called_once_with(
            "onetrust.reference.id", "acme/support"
        )
        inner_span.set_attribute.assert_called_once_with(
            "onetrust.reference.id", "acme/research"
        )

    def test_on_end_shutdown_force_flush_are_safe_no_ops(self):
        processor = otel_module.OneTrustSpanProcessor()
        processor.on_end(self._mock_span())
        processor.shutdown()
        assert processor.force_flush() is True


# --------------------------------------------------------------------------
# Streaming: correct active identity across the internal thread pool
# --------------------------------------------------------------------------


class TestStreamingAgentIdentity:
    def test_stream_uses_activated_agent_across_worker_threads(self):
        from ai_guard import ClassificationStream

        contexts = []

        def side_effect(*args, **kwargs):
            contexts.append(kwargs["json"]["context"])
            return _mock_response()

        client, _ = _make_client(
            agents=[AgentConfig(name="support", agent_id="acme/support")],
            side_effect=side_effect,
        )

        with client.activate("support"):
            stream = ClassificationStream(
                classifier_description=ClassifierDescriptionProfile(
                    uuid="u", version=1
                ),
                client=client,
                input=["line one\n", "line two\n", "line three\n"],
                context={"actor": "user"},
                chunk_size=50,
                max_workers=4,
            )
            list(stream)

        assert contexts, "expected at least one classify() call from the stream"
        assert all(c["agent_id"] == "acme/support" for c in contexts)

    def test_stream_falls_back_to_default_outside_activation(self):
        from ai_guard import ClassificationStream

        contexts = []

        def side_effect(*args, **kwargs):
            contexts.append(kwargs["json"]["context"])
            return _mock_response()

        client, _ = _make_client(side_effect=side_effect)

        stream = ClassificationStream(
            classifier_description=ClassifierDescriptionProfile(uuid="u", version=1),
            client=client,
            input=["only line\n"],
            context={"actor": "user"},
        )
        list(stream)

        assert contexts
        assert all(c["agent_id"] == "default-agent" for c in contexts)
