import json
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional


class AIPlatform(str, Enum):
    AMAZON_BEDROCK = "AMAZON_BEDROCK"
    AMAZON_SAGEMAKER = "AMAZON_SAGEMAKER"
    AZURE_FOUNDRY = "AZURE_FOUNDRY"
    DATABRICKS = "DATABRICKS"
    GCP_VERTEX = "GCP_VERTEX"


@dataclass
class ClassificationResponse:
    matches: List["ClassificationResponseMatch"]
    context: Optional[Dict[str, str]] = field(default=None)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ClassificationResponse":
        return ClassificationResponse(
            context=data.get("context", None),
            matches=[
                ClassificationResponseMatch.from_dict(m)
                for m in data.get("matches", [])
            ],
        )

    def __repr__(self) -> str:
        return json.dumps({
            "context": self.context,
            "matches": [
                {
                    "start": m.start,
                    "end": m.end,
                    "confidence": m.confidence,
                    "classifier": m.classifier
                }
                for m in self.matches
            ]
        })


@dataclass
class ClassificationRequest:
    classifier_description: "ClassifierDescription"
    context: Optional[Dict[str, str]] = field(default=None)
    structured: bool = False
    metadata: Optional["ClassificationRequestMetadata"] = field(default=None)
    text: Optional[str] = field(default=None)

    def to_dict(self) -> Dict[str, Any]:
        body: Dict[str, Any] = {
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
    objectName: Optional[str] = None
    parentObjectName: Optional[str] = None
    xPath: Optional[str] = None
    fileExtension: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.objectName is not None:
            out["objectName"] = self.objectName
        if self.parentObjectName is not None:
            out["parentObjectName"] = self.parentObjectName
        if self.xPath is not None:
            out["xPath"] = self.xPath
        if self.fileExtension is not None:
            out["fileExtension"] = self.fileExtension
        return out


class ClassifierDescription:
    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError


@dataclass
class ClassifierDescriptionProfile(ClassifierDescription):
    uuid: str
    version: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "profile",
            "uuid": self.uuid,
            "version": self.version,
        }


class ClassifierDescriptionDefault(ClassifierDescription):
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "profile",
            "uuid": "7dbf380f-0af8-4276-acb0-85413db2dbff",
            "version": 1,
        }


@dataclass
class ClassifierDescriptionCode:
    code: str
    version: int

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "version": self.version}


@dataclass
class ClassifierDescriptionCodes(ClassifierDescription):
    codes: List[ClassifierDescriptionCode]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "codes",
            "codes": [c.to_dict() for c in self.codes],
        }


@dataclass
class ClassifierDescriptionJson(ClassifierDescription):
    classifiers: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "json",
            "classifiers": self.classifiers,
        }


@dataclass
class MetricsEventMeter:
    name: str
    value: str

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "value": self.value}


@dataclass
class MetricsEvent:
    attributes: Dict[str, str]
    meter: MetricsEventMeter

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attributes": self.attributes,
            "meter": self.meter.to_dict(),
        }


@dataclass
class ClassificationResponseMatch:
    start: int
    end: int
    confidence: int
    text: str
    classifier: str

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ClassificationResponseMatch":
        return ClassificationResponseMatch(
            start=int(data.get("start", 0)),
            end=int(data.get("end", 0)),
            confidence=int(data.get("confidence", 0)),
            text=str(data.get("text", "")),
            classifier=str(data.get("classifier", "")),
        )
