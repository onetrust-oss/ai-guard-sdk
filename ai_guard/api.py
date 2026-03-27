import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Optional


class AIPlatform(StrEnum):
    """AI platform originating a classification or metrics request."""

    AMAZON_BEDROCK = "AMAZON_BEDROCK"
    AMAZON_SAGEMAKER = "AMAZON_SAGEMAKER"
    AZURE_FOUNDRY = "AZURE_FOUNDRY"
    DATABRICKS = "DATABRICKS"
    GCP_VERTEX = "GCP_VERTEX"


@dataclass
class ClassificationResponse:
    """Response from a classification request containing matched classifiers."""

    matches: list["ClassificationResponseMatch"]
    context: dict[str, str] | None = field(default=None)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ClassificationResponse":
        return ClassificationResponse(
            context=data.get("context"),
            matches=[
                ClassificationResponseMatch.from_dict(m)
                for m in data.get("matches", [])
            ],
        )

    def __repr__(self) -> str:
        return json.dumps(
            {
                "context": self.context,
                "matches": [
                    {
                        "start": m.start,
                        "end": m.end,
                        "confidence": m.confidence,
                        "classifier": m.classifier,
                    }
                    for m in self.matches
                ],
            }
        )


@dataclass
class ClassificationRequest:
    """Request body for the ``POST /classifications/v1`` endpoint."""

    classifier_description: "ClassifierDescription"
    context: dict[str, str] | None = field(default=None)
    structured: bool = False
    metadata: Optional["ClassificationRequestMetadata"] = field(default=None)
    text: str | None = field(default=None)

    def to_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "context": self.context,
            "classifierDescription": self.classifier_description.to_dict(),
            "structured": self.structured,
        }
        if self.metadata is not None:
            body["metadata"] = self.metadata.to_dict()
        if self.text is not None:
            body["text"] = self.text
        return body


@dataclass
class ClassificationRequestMetadata:
    """Optional metadata describing the source object being classified."""

    object_name: str | None = None
    parent_object_name: str | None = None
    x_path: str | None = None
    file_extension: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.object_name is not None:
            out["objectName"] = self.object_name
        if self.parent_object_name is not None:
            out["parentObjectName"] = self.parent_object_name
        if self.x_path is not None:
            out["xPath"] = self.x_path
        if self.file_extension is not None:
            out["fileExtension"] = self.file_extension
        return out


class ClassifierDescription:
    """Base class for specifying which classifiers to use in a request."""

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError


@dataclass
class ClassifierDescriptionProfile(ClassifierDescription):
    """Select classifiers by profile UUID and version."""

    uuid: str
    version: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "profile",
            "uuid": self.uuid,
            "version": self.version,
        }


class ClassifierDescriptionDefault(ClassifierDescription):
    """Use the built-in default classifier profile."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "profile",
            "uuid": "7dbf380f-0af8-4276-acb0-85413db2dbff",
            "version": 1,
        }


@dataclass
class ClassifierDescriptionCode:
    """A single classifier identified by code and version."""

    code: str
    version: int

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "version": self.version}


@dataclass
class ClassifierDescriptionCodes(ClassifierDescription):
    """Select classifiers by a list of classifier codes."""

    codes: list[ClassifierDescriptionCode]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "codes",
            "codes": [c.to_dict() for c in self.codes],
        }


@dataclass
class ClassifierDescriptionJson(ClassifierDescription):
    """Select classifiers using inline JSON definitions."""

    classifiers: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "json",
            "classifiers": self.classifiers,
        }


@dataclass
class MetricsEventMeter:
    """Meter name and value within a metrics event."""

    name: str
    value: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "value": self.value}


@dataclass
class MetricsEvent:
    """Request body for the ``POST /metric`` endpoint."""

    attributes: dict[str, str]
    meter: MetricsEventMeter

    def to_dict(self) -> dict[str, Any]:
        return {
            "attributes": self.attributes,
            "meter": self.meter.to_dict(),
        }


@dataclass
class ClassificationResponseMatch:
    """A single classifier match with position, confidence, and matched text."""

    start: int
    end: int
    confidence: int
    text: str
    classifier: str

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ClassificationResponseMatch":
        return ClassificationResponseMatch(
            start=int(data.get("start", 0)),
            end=int(data.get("end", 0)),
            confidence=int(data.get("confidence", 0)),
            text=str(data.get("text", "")),
            classifier=str(data.get("classifier", "")),
        )
