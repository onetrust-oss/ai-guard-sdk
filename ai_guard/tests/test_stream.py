import time
from unittest.mock import Mock

import pytest

from ai_guard import AIGuardClient, ClassificationStream
from ai_guard.api import (
    AIPlatform,
    ClassificationResponse,
    ClassificationResponseMatch,
    ClassifierDescriptionProfile,
)
from ai_guard.redact import (
    ClassificationRedactor,
    RedactAction,
    RedactKind,
    RedactPolicy,
)


def _response_to_dict(resp):
    return {
        "context": resp.context,
        "matches": [
            {
                "start": m.start,
                "end": m.end,
                "confidence": m.confidence,
                "text": m.text,
                "classifier": m.classifier,
            }
            for m in resp.matches
        ],
    }


def _make_client_with_mock_session(responses=None, delays=None):
    call_count = {"n": 0}

    def post_side_effect(*args, **kwargs):
        idx = call_count["n"]
        call_count["n"] += 1
        if delays and idx < len(delays):
            time.sleep(delays[idx])

        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "application/json"}

        if responses and idx < len(responses):
            mock_resp.json.return_value = _response_to_dict(responses[idx])
        else:
            payload = kwargs.get("json", {})
            mock_resp.json.return_value = {
                "context": payload.get("context"),
                "matches": [],
            }

        return mock_resp

    mock_session = Mock()
    mock_session.post.side_effect = post_side_effect

    client = AIGuardClient(
        "http://svc", token="token", agent_id="test-agent",
        platform=AIPlatform.AMAZON_BEDROCK,
        session=mock_session,
    )
    return client, mock_session


@pytest.fixture
def classifier_description():
    return ClassifierDescriptionProfile(uuid="test-uuid", version=1)


@pytest.fixture
def client():
    client, _ = _make_client_with_mock_session()
    return client


class TestClassificationStreamBasic:
    def test_single_line(self, classifier_description, client):
        stream = ClassificationStream(
            input=["Hello world\n"],
            classifier_description=classifier_description,
            client=client,
            chunk_size=100,
        )

        results = list(stream)

        assert len(results) == 1
        assert results[0].text == "Hello world\n"
        assert results[0].response.context == {"agent_id": "test-agent", "platform": AIPlatform.AMAZON_BEDROCK}
        assert results[0].response.matches == []
        assert results[0].redaction is None

    def test_multiple_lines(self, classifier_description):
        responses = [
            ClassificationResponse(
                context=None,
                matches=[
                    ClassificationResponseMatch(start=0, end=4, confidence=100, text="Line", classifier="word")
                ],
            ),
            ClassificationResponse(
                context=None,
                matches=[
                    ClassificationResponseMatch(start=0, end=4, confidence=100, text="Line", classifier="word")
                ],
            ),
            ClassificationResponse(
                context=None,
                matches=[
                    ClassificationResponseMatch(start=0, end=4, confidence=100, text="Line", classifier="word")
                ],
            ),
        ]
        client, _ = _make_client_with_mock_session(responses)
        stream = ClassificationStream(
            input=["Line 1\n", "Line 2\n", "Line 3\n"],
            classifier_description=classifier_description,
            client=client,
            chunk_size=100,
        )

        results = list(stream)

        assert "".join(r.text for r in results) == "Line 1\nLine 2\nLine 3\n"
        assert len(results) == 3
        assert results[0].response.matches[0].start == 0
        assert results[0].response.matches[0].end == 4
        assert results[1].response.matches[0].start == 7
        assert results[1].response.matches[0].end == 11
        assert results[2].response.matches[0].start == 14
        assert results[2].response.matches[0].end == 18

    def test_partial_line_flushed(self, classifier_description, client):
        stream = ClassificationStream(
            input=["No newline at end"],
            classifier_description=classifier_description,
            client=client,
            chunk_size=100,
        )

        results = list(stream)

        assert len(results) == 1
        assert results[0].text == "No newline at end"
        assert results[0].redaction is None

    def test_chunking_large_input(self, classifier_description):
        client, mock_session = _make_client_with_mock_session()
        stream = ClassificationStream(
            input=["Short line\n", "Another short\n", "Third line here\n"],
            classifier_description=classifier_description,
            client=client,
            chunk_size=20,
        )

        results = list(stream)

        assert "".join(r.text for r in results) == "Short line\nAnother short\nThird line here\n"
        assert mock_session.post.call_count >= 2
        assert len(results) == mock_session.post.call_count


class TestClassificationStreamWithRedaction:
    def test_redaction_applied(self, classifier_description):
        responses = [
            ClassificationResponse(
                context=None,
                matches=[
                    ClassificationResponseMatch(
                        start=7,
                        end=11,
                        confidence=100,
                        text="1234",
                        classifier="ssn",
                    )
                ],
            )
        ]
        client, _ = _make_client_with_mock_session(responses)
        policy = RedactPolicy(
            actions=[RedactAction(kind=RedactKind.REDACT, classifier="ssn")],
            default=RedactKind.NONE,
            redactor="*",
        )
        redactor = ClassificationRedactor(policy)

        stream = ClassificationStream(
            input=["My SSN 1234\n"],
            classifier_description=classifier_description,
            client=client,
            redactor=redactor,
            chunk_size=100,
        )

        results = list(stream)

        assert len(results) == 1
        result = results[0]
        assert result.text == "My SSN ****\n"
        assert len(result.response.matches) == 1
        assert result.response.matches[0].classifier == "ssn"
        assert result.redaction is not None
        assert result.redaction.text == "My SSN ****\n"
        assert len(result.redaction.actions) == 1
        assert result.redaction.actions[0].kind == RedactKind.REDACT
        assert result.redaction.actions[0].classifier == "ssn"

    def test_block_stops_stream(self, classifier_description):
        responses = [
            ClassificationResponse(
                context=None,
                matches=[
                    ClassificationResponseMatch(
                        start=0,
                        end=10,
                        confidence=100,
                        text="blocked",
                        classifier="pii",
                    )
                ],
            ),
            ClassificationResponse(
                context=None,
                matches=[],
            ),
        ]
        client, _ = _make_client_with_mock_session(responses)
        policy = RedactPolicy(
            actions=[RedactAction(kind=RedactKind.BLOCK, classifier="pii")],
            default=RedactKind.NONE,
            redactor="*",
        )
        redactor = ClassificationRedactor(policy)

        stream = ClassificationStream(
            input=["blocked text here\n"],
            classifier_description=classifier_description,
            client=client,
            redactor=redactor,
            chunk_size=50,
            max_workers=1,
        )

        results = list(stream)

        assert len(results) >= 1
        blocked = [r for r in results if r.redaction and any(a.kind == RedactKind.BLOCK for a in r.redaction.actions)]
        assert len(blocked) == 1
        assert blocked[0].text == ""

    def test_results_sorted_by_task(self, classifier_description, client):
        stream = ClassificationStream(
            input=["A\nB\nC\nD\nE\n"],
            classifier_description=classifier_description,
            client=client,
            chunk_size=3,
            max_workers=4,
        )

        results = list(stream)

        assert len(results) == 5
        for result in results:
            assert result.response.context == {"agent_id": "test-agent", "platform": AIPlatform.AMAZON_BEDROCK}
            assert result.response.matches == []
            assert result.redaction is None


class TestClassificationStreamOutOfOrder:
    def test_write_order_preserved_when_completion_out_of_order(self, classifier_description):
        delays = [0.1, 0.0, 0.0]
        client, _ = _make_client_with_mock_session(delays=delays)
        stream = ClassificationStream(
            input=["First\n", "Second\n", "Third\n"],
            classifier_description=classifier_description,
            client=client,
            chunk_size=10,
            max_workers=4,
        )

        results = list(stream)

        assert "".join(r.text for r in results) == "First\nSecond\nThird\n"
        assert len(results) == 3

    def test_write_order_with_reversed_completion(self, classifier_description):
        delays = [0.15, 0.1, 0.05, 0.0]
        client, _ = _make_client_with_mock_session(delays=delays)
        stream = ClassificationStream(
            input=["AAA\n", "BBB\n", "CCC\n", "DDD\n"],
            classifier_description=classifier_description,
            client=client,
            chunk_size=5,
            max_workers=4,
        )

        results = list(stream)

        assert "".join(r.text for r in results) == "AAA\nBBB\nCCC\nDDD\n"
        assert len(results) == 4


class TestClassificationStreamStopped:
    def test_block_stops_input_consumption(self, classifier_description):
        responses = [
            ClassificationResponse(
                context=None,
                matches=[
                    ClassificationResponseMatch(
                        start=0,
                        end=5,
                        confidence=100,
                        text="block",
                        classifier="blocked",
                    )
                ],
            )
        ]
        client, _ = _make_client_with_mock_session(responses)
        policy = RedactPolicy(
            actions=[RedactAction(kind=RedactKind.BLOCK, classifier="blocked")],
            default=RedactKind.NONE,
            redactor="*",
        )
        redactor = ClassificationRedactor(policy)

        stream = ClassificationStream(
            input=["block\n", "more text\n", "even more\n"],
            classifier_description=classifier_description,
            client=client,
            redactor=redactor,
            chunk_size=100,
            max_workers=1,
        )

        results = list(stream)

        assert len(results) >= 1
        blocked = [r for r in results if r.redaction and any(a.kind == RedactKind.BLOCK for a in r.redaction.actions)]
        assert len(blocked) == 1
