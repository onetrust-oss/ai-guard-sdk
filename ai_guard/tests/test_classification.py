import base64
import json
import unittest
from unittest.mock import Mock

from ai_guard import AIGuardClient
from ai_guard.api import (
    AIPlatform,
    ClassificationRequest,
    ClassificationResponse,
    ClassificationResponseMatch,
    ClassifierDescriptionProfile,
    ClassifierDescriptionCodes,
    ClassifierDescriptionCode,
    ClassifierDescriptionJson,
    ClassificationRequestMetadata,
    MetricsEvent,
    MetricsEventMeter,
)


class TestAIGuardClient(unittest.TestCase):
    @staticmethod
    def _make_client_with_mock_session(
            status_code=200,
            json_body=None,
            content_type="application/json",
            timeout=7.5,
    ):
        mock_resp = Mock()
        mock_resp.status_code = status_code
        mock_resp.headers = {"Content-Type": content_type}
        if json_body is None:
            json_body = {"context": None, "matches": []}
        mock_resp.json.return_value = json_body

        mock_session = Mock()
        mock_session.post.return_value = mock_resp

        client = AIGuardClient(
            "http://svc", session=mock_session, timeout=timeout,
            token="token", agent_id="test-agent",
            platform=AIPlatform.AMAZON_BEDROCK,
        )
        return client, mock_session, mock_resp

    def test_classify_success_returns_response(self):
        body = {
            "context": {"customerId": "123456"},
            "matches": [
                {
                    "start": 6,
                    "end": 18,
                    "confidence": 100,
                    "text": "321-507-0525",
                    "classifier": "US_PHONE_NUMBER",
                }
            ],
        }
        client, mock_session, _ = self._make_client_with_mock_session(
            status_code=200, json_body=body
        )

        req = ClassificationRequest(
            context=None,
            classifier_description=ClassifierDescriptionProfile(uuid="u", version=1),
            text="sample",
        )

        resp = client.classify(req)
        self.assertIsInstance(resp, ClassificationResponse)
        self.assertEqual(resp.context, {"customerId": "123456"})
        self.assertEqual(len(resp.matches), 1)
        m = resp.matches[0]
        self.assertIsInstance(m, ClassificationResponseMatch)
        self.assertEqual(
            (m.start, m.end, m.confidence, m.text, m.classifier),
            (6, 18, 100, "321-507-0525", "US_PHONE_NUMBER"),
        )

        called_url = (
                mock_session.post.call_args.kwargs.get("url")
                or mock_session.post.call_args.args[0]
        )
        self.assertEqual(called_url, "http://svc/classifications/v1")
        payload = mock_session.post.call_args.kwargs["json"]
        self.assertEqual(payload["context"], {"agent_id": "test-agent", "platform": AIPlatform.AMAZON_BEDROCK})
        self.assertEqual(payload["text"], "sample")
        self.assertEqual(payload["structured"], False)
        self.assertEqual(payload["classifierDescription"]["type"], "profile")

    def test_classify_400_raises_value_error_with_message(self):
        body = {"message": "bad request"}
        client, _, _ = self._make_client_with_mock_session(
            status_code=400, json_body=body
        )

        req = ClassificationRequest(
            context=None,
            classifier_description=ClassifierDescriptionProfile(uuid="u", version=1),
            text="x",
        )
        with self.assertRaises(ValueError) as ctx:
            client.classify(req)
        self.assertIn("bad request", str(ctx.exception))

    def test_classify_401_raises_value_error_with_message(self):
        body = {"message": "unauthorized"}
        client, _, _ = self._make_client_with_mock_session(
            status_code=401, json_body=body
        )

        req = ClassificationRequest(
            context=None,
            classifier_description=ClassifierDescriptionProfile(uuid="u", version=1),
            text="x",
        )
        with self.assertRaises(PermissionError) as ctx:
            client.classify(req)
        self.assertIn("unauthorized", str(ctx.exception))

    def test_classify_500_raises_runtime_error(self):
        client, _, _ = self._make_client_with_mock_session(
            status_code=500, json_body={"message": "boom"}
        )
        req = ClassificationRequest(
            context=None,
            classifier_description=ClassifierDescriptionProfile(uuid="u", version=1),
            text="x",
        )
        with self.assertRaises(RuntimeError) as ctx:
            client.classify(req)
        self.assertIn("boom", str(ctx.exception))

    def test_classify_unexpected_status_raises_runtime_error(self):
        client, _, _ = self._make_client_with_mock_session(
            status_code=418, json_body={"message": "teapot"}
        )
        req = ClassificationRequest(
            context=None,
            classifier_description=ClassifierDescriptionProfile(uuid="u", version=1),
            text="x",
        )
        with self.assertRaises(RuntimeError) as ctx:
            client.classify(req)
        self.assertIn("teapot", str(ctx.exception))

    def test_payload_variants_for_classifier_description(self):
        # Codes variant
        codes = ClassifierDescriptionCodes(
            codes=[
                ClassifierDescriptionCode(code="C1", version=1),
                ClassifierDescriptionCode(code="C2", version=2),
            ]
        )
        req_codes = ClassificationRequest(
            context=None, classifier_description=codes, text="x"
        )
        payload_codes = req_codes.to_dict()
        self.assertEqual(payload_codes["classifierDescription"]["type"], "codes")
        self.assertEqual(len(payload_codes["classifierDescription"]["codes"]), 2)

        # Json variant
        json_desc = ClassifierDescriptionJson(
            classifiers=[{"name": "A"}, {"name": "B"}]
        )
        req_json = ClassificationRequest(
            context=None, classifier_description=json_desc, text="x"
        )
        payload_json = req_json.to_dict()
        self.assertEqual(payload_json["classifierDescription"]["type"], "json")
        self.assertEqual(len(payload_json["classifierDescription"]["classifiers"]), 2)

    def test_metadata_serialization(self):
        md = ClassificationRequestMetadata(
            objectName="obj",
            parentObjectName="parent",
            xPath="/p/x",
            fileExtension="txt",
        )
        req = ClassificationRequest(
            context=None,
            classifier_description=ClassifierDescriptionProfile(uuid="u", version=1),
            text="x",
            metadata=md,
        )
        payload = req.to_dict()
        self.assertIn("metadata", payload)
        self.assertEqual(
            payload["metadata"],
            {
                "objectName": "obj",
                "parentObjectName": "parent",
                "xPath": "/p/x",
                "fileExtension": "txt",
            },
        )


class TestMetricClient(unittest.TestCase):
    @staticmethod
    def _make_client_with_mock_session(
            status_code=200,
            json_body=None,
            content_type="application/json",
    ):
        mock_resp = Mock()
        mock_resp.status_code = status_code
        mock_resp.headers = {"Content-Type": content_type}
        if json_body is not None:
            mock_resp.json.return_value = json_body

        mock_session = Mock()
        mock_session.post.return_value = mock_resp

        client = AIGuardClient(
            "http://svc", session=mock_session, timeout=7.5,
            token="token", agent_id="test-agent",
            platform=AIPlatform.AMAZON_BEDROCK,
        )
        return client, mock_session, mock_resp

    @staticmethod
    def _make_event():
        return MetricsEvent(
            attributes={"key": "value"},
            meter=MetricsEventMeter(name="ai_guard.agent", value="1.0"),
        )

    def test_metric_success(self):
        client, mock_session, _ = self._make_client_with_mock_session(status_code=200)
        event = self._make_event()

        client.metric(event)

        called_url = (
            mock_session.post.call_args.kwargs.get("url")
            or mock_session.post.call_args.args[0]
        )
        self.assertEqual(called_url, "http://svc/metric")
        payload = mock_session.post.call_args.kwargs["json"]
        self.assertEqual(payload["attributes"], {"key": "value", "agent_id": "test-agent", "platform": AIPlatform.AMAZON_BEDROCK})
        self.assertEqual(payload["meter"]["name"], "ai_guard.agent")
        self.assertEqual(payload["meter"]["value"], "1.0")

    def test_metric_400_raises_value_error(self):
        client, _, _ = self._make_client_with_mock_session(
            status_code=400, json_body={"message": "metrics not permitted"}
        )
        with self.assertRaises(ValueError) as ctx:
            client.metric(self._make_event())
        self.assertIn("metrics not permitted", str(ctx.exception))

    def test_metric_401_raises_permission_error(self):
        client, _, _ = self._make_client_with_mock_session(
            status_code=401, json_body={"message": "unauthorized"}
        )
        with self.assertRaises(PermissionError) as ctx:
            client.metric(self._make_event())
        self.assertIn("unauthorized", str(ctx.exception))

    def test_metric_unexpected_status_raises_runtime_error(self):
        client, _, _ = self._make_client_with_mock_session(
            status_code=500, json_body={"message": "internal error"}
        )
        with self.assertRaises(RuntimeError) as ctx:
            client.metric(self._make_event())
        self.assertIn("internal error", str(ctx.exception))


class TestMetricsEventSerialization(unittest.TestCase):
    def test_to_dict(self):
        event = MetricsEvent(
            attributes={"env": "prod", "service": "guard"},
            meter=MetricsEventMeter(name="ai_guard.agent", value="1.0"),
        )
        d = event.to_dict()
        self.assertEqual(d, {
            "attributes": {"env": "prod", "service": "guard"},
            "meter": {"name": "ai_guard.agent", "value": "1.0"},
        })

    def test_to_dict_empty_attributes(self):
        event = MetricsEvent(
            attributes={},
            meter=MetricsEventMeter(name="counter", value="42"),
        )
        d = event.to_dict()
        self.assertEqual(d["attributes"], {})
        self.assertEqual(d["meter"], {"name": "counter", "value": "42"})


class TestClassificationResponse(unittest.TestCase):

    def test_repr_with_multiple_matches(self):
        response = ClassificationResponse.from_dict({
            "context": None,
            "matches": [
                {
                    "start": 0,
                    "end": 10,
                    "confidence": 95,
                    "classifier": "US_PHONE_NUMBER"
                },
                {
                    "start": 20,
                    "end": 30,
                    "confidence": 100,
                    "classifier": "US_SSN"
                },
                {
                    "start": 40,
                    "end": 55,
                    "confidence": 85,
                    "classifier": "EMAIL_ADDRESS"
                }
            ]
        })

        result = json.loads(f"{response}")
        expected = {
            "context": None,
            "matches": [
                {
                    "start": 0,
                    "end": 10,
                    "confidence": 95,
                    "classifier": "US_PHONE_NUMBER"
                },
                {
                    "start": 20,
                    "end": 30,
                    "confidence": 100,
                    "classifier": "US_SSN"
                },
                {
                    "start": 40,
                    "end": 55,
                    "confidence": 85,
                    "classifier": "EMAIL_ADDRESS"
                }
            ]
        }
        self.assertEqual(result, expected)

    def test_repr_with_no_matches(self):
        response = ClassificationResponse.from_dict({
            "context": None,
            "matches": []
        })

        result = json.loads(f"{response}")
        expected = {
            "context": None,
            "matches": []
        }
        self.assertEqual(result, expected)

    def test_repr_with_single_match(self):
        response = ClassificationResponse.from_dict({
            "context": None,
            "matches": [
                {
                    "start": 5,
                    "end": 17,
                    "confidence": 100,
                    "classifier": "US_PHONE_NUMBER"
                }
            ]
        })

        result = json.loads(f"{response}")
        expected = {
            "context": None,
            "matches": [
                {
                    "start": 5,
                    "end": 17,
                    "confidence": 100,
                    "classifier": "US_PHONE_NUMBER"
                }
            ]
        }
        self.assertEqual(result, expected)


class TestPinSha256Validation(unittest.TestCase):

    def _make_client(self, pin_sha256: str):
        return AIGuardClient(
            "https://svc", token="token", agent_id="test-agent",
            platform=AIPlatform.AMAZON_BEDROCK,
            pin_sha256=pin_sha256,
        )

    def test_valid_pin_accepted(self):
        valid_pin = base64.b64encode(b"\x00" * 32).decode()
        client = self._make_client(valid_pin)
        self.assertIsNotNone(client)

    def test_empty_string_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            self._make_client("")
        self.assertIn("0", str(ctx.exception))

    def test_invalid_base64_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            self._make_client("!!!not-base64!!!")
        self.assertIn("not valid base64", str(ctx.exception))

    def test_wrong_length_short_rejected(self):
        short_pin = base64.b64encode(b"\x00" * 16).decode()
        with self.assertRaises(ValueError) as ctx:
            self._make_client(short_pin)
        self.assertIn("32 bytes", str(ctx.exception))
        self.assertIn("16", str(ctx.exception))

    def test_wrong_length_long_rejected(self):
        long_pin = base64.b64encode(b"\x00" * 64).decode()
        with self.assertRaises(ValueError) as ctx:
            self._make_client(long_pin)
        self.assertIn("32 bytes", str(ctx.exception))
        self.assertIn("64", str(ctx.exception))

    def test_whitespace_only_rejected(self):
        with self.assertRaises(ValueError):
            self._make_client("   ")

    def test_none_skips_pinning(self):
        client = AIGuardClient(
            "https://svc", token="token", agent_id="test-agent",
            platform=AIPlatform.AMAZON_BEDROCK,
        )
        self.assertFalse(client._session.verify is False)


if __name__ == "__main__":
    unittest.main()
