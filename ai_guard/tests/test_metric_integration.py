import secrets
import unittest

from ai_guard import AIGuardClient
from ai_guard.api import (
    AIPlatform,
    MetricsEvent,
    MetricsEventMeter,
)
from ai_guard.tests.test_server import AIGuardTestServices, AuthMode, ClassifierClientType

POLL_TIMEOUT = 15.0


def random_agent_id() -> str:
    return secrets.token_hex(16)


def find_metric(metrics: list[dict], meter_name: str, agent_id: str) -> dict | None:
    for event in metrics:
        meter = event.get("meter", {})
        attrs = event.get("attributes", {})
        if meter.get("name") == meter_name and attrs.get("agent_id") == agent_id:
            return event
    return None


def poll_for_metric(service, meter_name: str, agent_id: str,
                    timeout: float = POLL_TIMEOUT) -> dict:
    import time
    deadline = time.monotonic() + timeout
    delay = 0.1
    while True:
        metrics = service.poll_metrics(min_count=1, timeout=max(0.1, deadline - time.monotonic()))
        found = find_metric(metrics, meter_name, agent_id)
        if found is not None:
            return found
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"timed out waiting for metric {meter_name} agent_id={agent_id} "
                f"after {timeout}s"
            )
        time.sleep(delay)
        delay = min(delay * 2, 2.0)


class TestMetricIntegration(unittest.TestCase):
    server: AIGuardTestServices

    @classmethod
    def setUpClass(cls):
        cls.server = AIGuardTestServices(
            auth=AuthMode.ONETRUST,
            classifier_client_type=ClassifierClientType.CLIENT,
        )
        cls.server.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()

    def _client(self, agent_id: str = "integration-test") -> AIGuardClient:
        return AIGuardClient(
            self.server.url,
            token=self.server.classification_token,
            agent_id=agent_id,
            platform=AIPlatform.AMAZON_BEDROCK,
            session=self.server.http_session(),
        )

    def test_metric_agent_event(self):
        agent_id = random_agent_id()
        client = self._client(agent_id=agent_id)
        expected = secrets.randbelow(9900) + 100

        event = MetricsEvent(
            attributes={},
            meter=MetricsEventMeter(name="ai_guard.agent", value=str(expected)),
        )
        client.metric(event)

        metric = poll_for_metric(
            self.server.classification_service, "ai_guard.agent", agent_id,
        )
        self.assertEqual(metric["meter"]["value"], str(expected))
        self.assertEqual(metric["attributes"]["platform"], "AMAZON_BEDROCK")

    def test_metric_user_event(self):
        agent_id = random_agent_id()
        client = self._client(agent_id=agent_id)

        event = MetricsEvent(
            attributes={"new_session": "true"},
            meter=MetricsEventMeter(name="ai_guard.user", value="1"),
        )
        client.metric(event)

        metric = poll_for_metric(
            self.server.classification_service, "ai_guard.user", agent_id,
        )
        self.assertEqual(metric["meter"]["value"], "1")
        self.assertEqual(metric["attributes"]["platform"], "AMAZON_BEDROCK")
        self.assertEqual(metric["attributes"]["new_session"], "true")

    def test_metric_redact_event(self):
        agent_id = random_agent_id()
        client = self._client(agent_id=agent_id)

        event = MetricsEvent(
            attributes={"action": "block", "actor": "user"},
            meter=MetricsEventMeter(name="ai_guard.redact", value="1"),
        )
        client.metric(event)

        metric = poll_for_metric(
            self.server.classification_service, "ai_guard.redact", agent_id,
        )
        self.assertEqual(metric["meter"]["value"], "1")
        self.assertEqual(metric["attributes"]["platform"], "AMAZON_BEDROCK")
        self.assertEqual(metric["attributes"]["action"], "block")

    def test_metric_invalid_meter_name_returns_bad_request(self):
        client = self._client()

        event = MetricsEvent(
            attributes={},
            meter=MetricsEventMeter(name="made.up.meter", value="1"),
        )

        with self.assertRaises(ValueError):
            client.metric(event)


if __name__ == "__main__":
    unittest.main()
