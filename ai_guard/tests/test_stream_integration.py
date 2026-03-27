import unittest

from ai_guard import AIGuardClient, ClassificationStream
from ai_guard.api import AIPlatform, ClassifierDescriptionProfile
from ai_guard.redact import (
    ClassificationRedactor,
    RedactAction,
    RedactKind,
    RedactPolicy,
)
from ai_guard.tests.test_server import (
    AIGuardTestServices,
    AuthMode,
    ClassifierClientType,
)


class TestClassificationStreamIntegration(unittest.TestCase):
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

    def _client(self) -> AIGuardClient:
        return AIGuardClient(
            self.server.url,
            token=self.server.classification_token,
            agent_id="integration-test",
            platform=AIPlatform.AMAZON_BEDROCK,
            session=self.server.http_session(),
        )

    @staticmethod
    def _phone_handler(request):
        text = request.get("text", "")
        phone = "321-507-0525"
        idx = text.find(phone)
        if idx == -1:
            return []
        return [
            {
                "start": idx,
                "end": idx + len(phone),
                "confidence": 100,
                "text": phone,
                "classifier": "US_PHONE_NUMBER",
            }
        ]

    def test_stream_multiple_chunks_with_redaction(self):
        client = self._client()
        self.server.set_classification_handler(self._phone_handler)
        policy = RedactPolicy(
            actions=[
                RedactAction(kind=RedactKind.REDACT, classifier="US_PHONE_NUMBER")
            ],
            default=RedactKind.NONE,
            redactor="*",
        )
        redactor = ClassificationRedactor(policy)

        stream = ClassificationStream(
            input=["First line here\n", "Phone: 321-507-0525\n", "Third line\n"],
            context={"actor": "user"},
            classifier_description=ClassifierDescriptionProfile(
                uuid="7dbf380f-0af8-4276-acb0-85413db2dbff",
                version=1,
            ),
            client=client,
            redactor=redactor,
            chunk_size=50,
        )

        results = list(stream)
        output = "".join(r.text for r in results)

        self.assertGreaterEqual(len(results), 2)
        self.assertIn("************", output)
        self.assertNotIn("321-507-0525", output)
        self.assertIn("First line here\n", output)
        self.assertIn("Third line\n", output)

        has_phone_match = any(
            any(m.classifier == "US_PHONE_NUMBER" for m in result.response.matches)
            for result in results
        )
        self.assertTrue(has_phone_match)

    def test_stream_no_redactor(self):
        client = self._client()
        self.server.set_classification_handler(self._phone_handler)

        stream = ClassificationStream(
            input=["My number is 321-507-0525\n"],
            context={"actor": "user"},
            classifier_description=ClassifierDescriptionProfile(
                uuid="7dbf380f-0af8-4276-acb0-85413db2dbff",
                version=1,
            ),
            client=client,
            chunk_size=100,
        )

        results = list(stream)

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertIsNone(result.redaction)
        self.assertEqual(result.text, "My number is 321-507-0525\n")
        self.assertEqual(len(result.response.matches), 1)


if __name__ == "__main__":
    unittest.main()
