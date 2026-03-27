import unittest

from ai_guard import AIGuardClient
from ai_guard.api import (
    AIPlatform,
    ClassificationRequest,
    ClassificationRequestMetadata,
    ClassifierDescriptionCodes,
    ClassifierDescriptionDefault,
    ClassifierDescriptionProfile,
)
from ai_guard.tests.test_server import AIGuardTestServices, AuthMode


class TestAIGuardClientIntegration(unittest.TestCase):
    server: AIGuardTestServices

    @classmethod
    def setUpClass(cls):
        cls.server = AIGuardTestServices(
            auth=AuthMode.ONETRUST,
        )
        cls.server.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()

    def _client(self, token: str | None = None) -> AIGuardClient:
        return AIGuardClient(
            self.server.url,
            token=token or self.server.classification_token,
            agent_id="integration-test",
            platform=AIPlatform.AMAZON_BEDROCK,
            session=self.server.http_session(),
        )

    def test_classify_with_profile(self):
        client = self._client()
        self.server.set_classification_response(
            [
                {
                    "start": 6,
                    "end": 18,
                    "confidence": 100,
                    "text": "321-507-0525",
                    "classifier": "US_PHONE_NUMBER",
                },
            ]
        )

        request = ClassificationRequest(
            context={"actor": "user", "agent_id": "123456"},
            classifier_description=ClassifierDescriptionProfile(
                uuid="7dbf380f-0af8-4276-acb0-85413db2dbff",
                version=1,
            ),
            structured=False,
            metadata=ClassificationRequestMetadata(
                object_name="object",
                parent_object_name="parent",
                x_path="path",
                file_extension="extension",
            ),
            text="phone 321-507-0525 number",
        )

        resp = client.classify(request)

        self.assertEqual(
            resp.context,
            {
                "actor": "user",
                "agent_id": "integration-test",
                "platform": "AMAZON_BEDROCK",
            },
        )
        self.assertEqual(len(resp.matches), 1)

        first = resp.matches[0]
        self.assertEqual(
            (first.start, first.end, first.confidence, first.text, first.classifier),
            (6, 18, 100, "321-507-0525", "US_PHONE_NUMBER"),
        )

    def test_classify_with_profile_explicit_session(self):
        client = AIGuardClient(
            self.server.url,
            token=self.server.classification_token,
            agent_id="integration-test",
            platform=AIPlatform.AMAZON_BEDROCK,
            session=self.server.http_session(),
        )
        self.server.set_classification_response(
            [
                {
                    "start": 6,
                    "end": 18,
                    "confidence": 100,
                    "text": "321-507-0525",
                    "classifier": "US_PHONE_NUMBER",
                },
            ]
        )

        request = ClassificationRequest(
            context={"actor": "user", "agent_id": "123456"},
            classifier_description=ClassifierDescriptionProfile(
                uuid="7dbf380f-0af8-4276-acb0-85413db2dbff",
                version=1,
            ),
            structured=False,
            metadata=ClassificationRequestMetadata(
                object_name="object",
                parent_object_name="parent",
                x_path="path",
                file_extension="extension",
            ),
            text="phone 321-507-0525 number",
        )

        resp = client.classify(request)

        self.assertEqual(
            resp.context,
            {
                "actor": "user",
                "agent_id": "integration-test",
                "platform": "AMAZON_BEDROCK",
            },
        )
        self.assertEqual(len(resp.matches), 1)

        first = resp.matches[0]
        self.assertEqual(
            (first.start, first.end, first.confidence, first.text, first.classifier),
            (6, 18, 100, "321-507-0525", "US_PHONE_NUMBER"),
        )

    def test_classify_with_default_profile(self):
        client = self._client()
        self.server.set_classification_response(
            [
                {
                    "start": 6,
                    "end": 18,
                    "confidence": 100,
                    "text": "321-507-0525",
                    "classifier": "US_PHONE_NUMBER",
                },
            ]
        )

        request = ClassificationRequest(
            context={"actor": "user", "agent_id": "123456"},
            classifier_description=ClassifierDescriptionDefault(),
            structured=False,
            metadata=ClassificationRequestMetadata(
                object_name="object",
                parent_object_name="parent",
                x_path="path",
                file_extension="extension",
            ),
            text="phone 321-507-0525 number",
        )

        resp = client.classify(request)

        self.assertEqual(
            resp.context,
            {
                "actor": "user",
                "agent_id": "integration-test",
                "platform": "AMAZON_BEDROCK",
            },
        )
        self.assertEqual(len(resp.matches), 1)

        first = resp.matches[0]
        self.assertEqual(
            (first.start, first.end, first.confidence, first.text, first.classifier),
            (6, 18, 100, "321-507-0525", "US_PHONE_NUMBER"),
        )

    def test_classify_empty_text(self):
        client = self._client()
        self.server.set_classification_response([])

        request = ClassificationRequest(
            context={"actor": "user"},
            classifier_description=ClassifierDescriptionProfile(
                uuid="7dbf380f-0af8-4276-acb0-85413db2dbff",
                version=1,
            ),
            structured=False,
            metadata=ClassificationRequestMetadata(
                object_name="object",
                parent_object_name="parent",
                x_path="path",
                file_extension="extension",
            ),
            text="",
        )

        resp = client.classify(request)
        self.assertEqual(
            resp.context,
            {
                "actor": "user",
                "agent_id": "integration-test",
                "platform": "AMAZON_BEDROCK",
            },
        )
        self.assertEqual(len(resp.matches), 0)

    def test_classify_invalid_profile_uuid_returns_400(self):
        client = self._client()
        self.server.set_classification_error(400, "Profile not found: not-a-valid-uuid")

        invalid_uuid = "not-a-valid-uuid"

        request = ClassificationRequest(
            context={"actor": "user"},
            classifier_description=ClassifierDescriptionProfile(
                uuid=invalid_uuid,
                version=1,
            ),
            structured=False,
            metadata=ClassificationRequestMetadata(
                object_name="object",
                parent_object_name="parent",
                x_path="path",
                file_extension="extension",
            ),
            text="phone 321-507-0525 number",
        )

        with self.assertRaises(ValueError):
            client.classify(request)

    def test_classify_invalid_token_returns_401(self):
        client = self._client(token="bad-token")

        request = ClassificationRequest(
            context={"actor": "user"},
            classifier_description=ClassifierDescriptionProfile(
                uuid="7dbf380f-0af8-4276-acb0-85413db2dbff",
                version=1,
            ),
            structured=False,
            metadata=None,
            text="phone 321-507-0525 number",
        )

        with self.assertRaises(PermissionError):
            client.classify(request)

    def test_classify_empty_codes(self):
        client = self._client()
        self.server.set_classification_response([])

        request = ClassificationRequest(
            context={"actor": "user"},
            classifier_description=ClassifierDescriptionCodes(codes=[]),
            structured=False,
            metadata=None,
            text="some text",
        )

        resp = client.classify(request)
        self.assertEqual(
            resp.context,
            {
                "actor": "user",
                "agent_id": "integration-test",
                "platform": "AMAZON_BEDROCK",
            },
        )
        self.assertEqual(len(resp.matches), 0)


class TestAIGuardClientNoTLSIntegration(unittest.TestCase):
    server: AIGuardTestServices

    @classmethod
    def setUpClass(cls):
        cls.server = AIGuardTestServices(
            tls=False,
            auth=AuthMode.ONETRUST,
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
        )

    def test_no_tls_health(self):
        client = self._client()
        resp = client._session.get(f"{self.server.url}/health", timeout=5)
        self.assertEqual(resp.status_code, 200)

    def test_no_tls_classify_with_profile(self):
        client = self._client()
        self.server.set_classification_response(
            [
                {
                    "start": 6,
                    "end": 18,
                    "confidence": 100,
                    "text": "321-507-0525",
                    "classifier": "US_PHONE_NUMBER",
                },
            ]
        )

        request = ClassificationRequest(
            context={"actor": "user"},
            classifier_description=ClassifierDescriptionProfile(
                uuid="7dbf380f-0af8-4276-acb0-85413db2dbff",
                version=1,
            ),
            structured=False,
            metadata=None,
            text="phone 321-507-0525 number",
        )

        resp = client.classify(request)

        self.assertEqual(len(resp.matches), 1)
        first = resp.matches[0]
        self.assertEqual(first.text, "321-507-0525")
        self.assertEqual(first.classifier, "US_PHONE_NUMBER")


if __name__ == "__main__":
    unittest.main()
