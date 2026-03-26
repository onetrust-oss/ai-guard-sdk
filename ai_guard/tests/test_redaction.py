import unittest
import json

from ai_guard.api import ClassificationResponse
from ai_guard.redact import (
    ClassificationRedactor,
    RedactPolicy,
    RedactAction,
    RedactKind,
    Redaction,
)


class TestClassificationRedactor(unittest.TestCase):

    def test_redaction(self):
        policy = RedactPolicy(
            actions=[
                RedactAction(
                    kind=RedactKind.REDACT,
                    classifier="US_PHONE_NUMBER")
            ],
            default=RedactKind.REDACT,
            redactor=' '
        )

        redactor = ClassificationRedactor(policy=policy)

        classification = ClassificationResponse.from_dict({
            "context": None,
            "matches": [
                {
                    "start": 6,
                    "end": 18,
                    "confidence": 100,
                    "text": "321-507-0525",
                    "classifier": "US_PHONE_NUMBER",
                }
            ],
        })

        redact = redactor.redact(text="phone 321-507-0525 number", classification=classification)

        self.assertEqual(redact.text, "phone              number")
        self.assertEqual(redact.actions, [RedactAction(kind=RedactKind.REDACT, classifier="US_PHONE_NUMBER")])
        self.assertTrue(all(action.kind != RedactKind.NONE for action in redact.actions))

    def test_block_behavior_wins_over_redact(self):
        policy = RedactPolicy(
            actions=[
                RedactAction(kind=RedactKind.REDACT, classifier="US_PHONE_NUMBER"),
                RedactAction(kind=RedactKind.BLOCK, classifier="US_STREET_ADDRESS")
            ],
            default=RedactKind.REDACT,
            redactor=' '
        )

        redactor = ClassificationRedactor(policy=policy)

        classification = ClassificationResponse.from_dict({
            "context": None,
            "matches": [
                {
                    "start": 6,
                    "end": 18,
                    "confidence": 100,
                    "text": "321-507-0525",
                    "classifier": "US_PHONE_NUMBER",
                },
                {
                    "start": 24,
                    "end": 44,
                    "confidence": 100,
                    "text": "123 Main St, CA 90210",
                    "classifier": "US_STREET_ADDRESS",
                }
            ],
        })

        redact = redactor.redact(
            text="phone 321-507-0525 addr 123 Main St, CA 90210",
            classification=classification
        )

        self.assertEqual(redact.text, "")
        self.assertEqual(redact.actions, [RedactAction(kind=RedactKind.BLOCK, classifier="US_STREET_ADDRESS")])
        self.assertTrue(all(action.kind != RedactKind.NONE for action in redact.actions))

    def test_multiple_matches_same_classifier(self):
        policy = RedactPolicy(
            actions=[
                RedactAction(kind=RedactKind.REDACT, classifier="US_PHONE_NUMBER")
            ],
            default=RedactKind.NONE,
            redactor=' '
        )

        redactor = ClassificationRedactor(policy=policy)

        classification = ClassificationResponse.from_dict({
            "context": None,
            "matches": [
                {
                    "start": 6,
                    "end": 18,
                    "confidence": 100,
                    "text": "321-507-0525",
                    "classifier": "US_PHONE_NUMBER"
                },
                {
                    "start": 27,
                    "end": 39,
                    "confidence": 100,
                    "text": "321-507-0522",
                    "classifier": "US_PHONE_NUMBER"
                }
            ]
        })

        redact = redactor.redact(
            text="phone 321-507-0525 or call 321-507-0522 today",
            classification=classification
        )

        self.assertEqual(redact.text, "phone              or call              today")
        self.assertEqual(redact.actions, [RedactAction(kind=RedactKind.REDACT, classifier="US_PHONE_NUMBER")])
        self.assertTrue(all(action.kind != RedactKind.NONE for action in redact.actions))

    def test_multiple_matches_different_classifiers(self):
        policy = RedactPolicy(
            actions=[
                RedactAction(kind=RedactKind.REDACT, classifier="US_PHONE_NUMBER"),
                RedactAction(kind=RedactKind.REDACT, classifier="US_STREET_ADDRESS")
            ],
            default=RedactKind.NONE,
            redactor=' '
        )

        redactor = ClassificationRedactor(policy=policy)

        classification = ClassificationResponse.from_dict({
            "context": None,
            "matches": [
                {
                    "start": 5,
                    "end": 17,
                    "confidence": 100,
                    "text": "321-507-0525",
                    "classifier": "US_PHONE_NUMBER",
                },
                {
                    "start": 21,
                    "end": 32,
                    "confidence": 100,
                    "text": "123 Main St",
                    "classifier": "US_STREET_ADDRESS",
                },
                {
                    "start": 36,
                    "end": 48,
                    "confidence": 100,
                    "text": "555-123-4567",
                    "classifier": "US_PHONE_NUMBER",
                }
            ],
        })

        redact = redactor.redact(
            text="call 321-507-0525 at 123 Main St or 555-123-4567",
            classification=classification
        )

        self.assertEqual(redact.text, "call              at             or             ")
        self.assertIn(RedactAction(kind=RedactKind.REDACT, classifier="US_PHONE_NUMBER"), redact.actions)
        self.assertIn(RedactAction(kind=RedactKind.REDACT, classifier="US_STREET_ADDRESS"), redact.actions)
        self.assertEqual(len(redact.actions), 2)
        self.assertTrue(all(action.kind != RedactKind.NONE for action in redact.actions))

    def test_default_action_none_with_multiple_matches(self):
        policy = RedactPolicy(
            actions=[
                RedactAction(kind=RedactKind.REDACT, classifier="US_PHONE_NUMBER")
            ],
            default=RedactKind.NONE,
            redactor=' '
        )

        redactor = ClassificationRedactor(policy=policy)

        classification = ClassificationResponse.from_dict({
            "context": None,
            "matches": [
                {
                    "start": 6,
                    "end": 18,
                    "confidence": 100,
                    "text": "321-507-0525",
                    "classifier": "US_PHONE_NUMBER",
                },
                {
                    "start": 23,
                    "end": 38,
                    "confidence": 100,
                    "text": "john@email.com",
                    "classifier": "EMAIL_ADDRESS",
                },
                {
                    "start": 48,
                    "end": 59,
                    "confidence": 100,
                    "text": "123-45-6789",
                    "classifier": "US_SSN",
                }
            ],
        })

        redact = redactor.redact(
            text="phone 321-507-0525 and john@email.com contact 123-45-6789",
            classification=classification
        )

        self.assertEqual(redact.text, "phone              and john@email.com contact 123-45-6789")
        self.assertEqual(redact.actions, [RedactAction(kind=RedactKind.REDACT, classifier="US_PHONE_NUMBER")])
        self.assertTrue(all(action.kind != RedactKind.NONE for action in redact.actions))

    def test_default_action_redact_with_multiple_matches(self):
        policy = RedactPolicy(
            actions=[
                RedactAction(kind=RedactKind.BLOCK, classifier="US_SSN")
            ],
            default=RedactKind.REDACT,
            redactor=' '
        )

        redactor = ClassificationRedactor(policy=policy)

        classification = ClassificationResponse.from_dict({
            "context": None,
            "matches": [
                {
                    "start": 6,
                    "end": 18,
                    "confidence": 100,
                    "text": "321-507-0525",
                    "classifier": "US_PHONE_NUMBER",
                },
                {
                    "start": 23,
                    "end": 37,
                    "confidence": 100,
                    "text": "john@email.com",
                    "classifier": "EMAIL_ADDRESS",
                }
            ],
        })

        redact = redactor.redact(
            text="phone 321-507-0525 and john@email.com contact",
            classification=classification
        )

        self.assertEqual(redact.text, "phone              and                contact")
        self.assertIn(RedactAction(kind=RedactKind.REDACT, classifier="US_PHONE_NUMBER"), redact.actions)
        self.assertIn(RedactAction(kind=RedactKind.REDACT, classifier="EMAIL_ADDRESS"), redact.actions)
        self.assertEqual(len(redact.actions), 2)
        self.assertTrue(all(action.kind != RedactKind.NONE for action in redact.actions))

    def test_default_action_block_with_multiple_matches(self):
        policy = RedactPolicy(
            actions=[
                RedactAction(kind=RedactKind.REDACT, classifier="US_PHONE_NUMBER")
            ],
            default=RedactKind.BLOCK,
            redactor=' '
        )

        redactor = ClassificationRedactor(policy=policy)

        classification = ClassificationResponse.from_dict({
            "context": None,
            "matches": [
                {
                    "start": 6,
                    "end": 18,
                    "confidence": 100,
                    "text": "321-507-0525",
                    "classifier": "US_PHONE_NUMBER",
                },
                {
                    "start": 23,
                    "end": 37,
                    "confidence": 100,
                    "text": "john@email.com",
                    "classifier": "EMAIL_ADDRESS",
                }
            ],
        })

        redact = redactor.redact(
            text="phone 321-507-0525 and john@email.com",
            classification=classification
        )

        self.assertEqual(redact.text, "")
        self.assertEqual(redact.actions, [RedactAction(kind=RedactKind.BLOCK, classifier="EMAIL_ADDRESS")])
        self.assertTrue(all(action.kind != RedactKind.NONE for action in redact.actions))


class TestRedaction(unittest.TestCase):

    def test_repr_with_multiple_actions(self):
        redaction = Redaction(
            actions=[
                RedactAction(kind=RedactKind.REDACT, classifier="US_PHONE_NUMBER"),
                RedactAction(kind=RedactKind.REDACT, classifier="US_SSN"),
                RedactAction(kind=RedactKind.BLOCK, classifier="US_STREET_ADDRESS")
            ],
            text="some text"
        )

        result = json.loads(f"{redaction}")
        expected = {
            "actions": [
                {"kind": "REDACT", "classifier": "US_PHONE_NUMBER"},
                {"kind": "REDACT", "classifier": "US_SSN"},
                {"kind": "BLOCK", "classifier": "US_STREET_ADDRESS"}
            ]
        }
        self.assertEqual(result, expected)

    def test_repr_with_single_action(self):
        redaction = Redaction(
            actions=[
                RedactAction(kind=RedactKind.REDACT, classifier="EMAIL_ADDRESS")
            ],
            text="text"
        )

        result = json.loads(f"{redaction}")
        expected = {
            "actions": [
                {"kind": "REDACT", "classifier": "EMAIL_ADDRESS"}
            ]
        }
        self.assertEqual(result, expected)

    def test_repr_with_no_actions(self):
        redaction = Redaction(
            actions=[],
            text="unchanged text"
        )

        result = json.loads(f"{redaction}")
        expected = {
            "actions": []
        }
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
