# AI Guard SDK

AI Guard brings real-time sensitive data protection to your GenAI applications. Powered by the OneTrust Data Discovery Classification System with 300+ built-in classifiers, AI Guard detects PII, credentials and other sensitive patterns in both user prompts and LLM responses. Use this SDK to redact, block, or monitor content before it ever leaves your environment. 

This SDK is the Python client for the **AI Guard Service** which is deployed in your infrastructure. The service handles all classification and metrics processing, while the SDK provides the interface for integrating AI Guard into your GenAI agent Python runtime. 

## Installation

Requires **Python 3.13+**.

```bash
pip install onetrust-ai-guard-sdk
```

## Quick Start

### 1. Initialize the Client

```python
import os
from ai_guard import AIGuardClient
from ai_guard.api import AIPlatform

client = AIGuardClient(
    os.environ["AI_GUARD_URL"],           # e.g. https://ai-guard.example.com:4443
    token=os.environ["AI_GUARD_TOKEN"],   # OneTrust API key with Data Discovery scope
    agent_id="my-agent",
    platform=AIPlatform.AMAZON_BEDROCK,
)
```

### 2. Classify Text

```python
from ai_guard.api import ClassificationRequest, ClassifierDescriptionDefault

response = client.classify(ClassificationRequest(
    context={"actor": "user"},
    classifier_description=ClassifierDescriptionDefault(),
    text="Call me at 321-507-0525",
))

for match in response.matches:
    print(f"{match.classifier}: '{match.text}' at [{match.start}:{match.end}] (confidence: {match.confidence})")
# US_PHONE_NUMBER: '321-507-0525' at [6:18] (confidence: 100)
```

### 3. Redact Sensitive Data

```python
from ai_guard.redact import ClassificationRedactor, RedactPolicy, RedactAction, RedactKind

policy = RedactPolicy(
    actions=[RedactAction(kind=RedactKind.REDACT, classifier="US_PHONE_NUMBER")],
    default=RedactKind.NONE,
    redactor="*",
)

redactor = ClassificationRedactor(policy=policy)
result = redactor.redact(text="Call me at 321-507-0525", classification=response)

print(result.text)     # "Call me at ************"
print(result.actions)  # [RedactAction(kind=REDACT, classifier="US_PHONE_NUMBER")]
```

## Client Parameters

| Parameter    | Type               | Required | Description                                                                            |
|--------------|--------------------|----------|----------------------------------------------------------------------------------------|
| `url`        | `str`              | Yes      | Base URL of your AI Guard service                                                      |
| `token`      | `str`              | Yes      | OneTrust API key with Data Discovery scope                                             |
| `agent_id`   | `str`              | Yes      | Unique identifier for your AI agent or application                                     |
| `platform`   | `AIPlatform`       | Yes      | AI platform your application uses (see below)                                          |
| `pin_sha256` | `str`              | No       | Certificate pin for TLS verification (see [Certificate Pinning](#certificate-pinning)) |
| `session`    | `requests.Session` | No       | Custom session for advanced TLS configuration                                          |

### Supported Platforms

| Platform               | Value                         |
|------------------------|-------------------------------|
| Amazon Bedrock         | `AIPlatform.AMAZON_BEDROCK`   |
| Amazon SageMaker       | `AIPlatform.AMAZON_SAGEMAKER` |
| Azure AI Foundry       | `AIPlatform.AZURE_FOUNDRY`    |
| Databricks             | `AIPlatform.DATABRICKS`       |
| Google Cloud Vertex AI | `AIPlatform.GCP_VERTEX`       |

## Classification

The `classify()` method sends text to the AI Guard service and returns matches with position offsets, confidence scores, and classifier identifiers.

```python
from ai_guard.api import ClassificationRequest, ClassifierDescriptionDefault

request = ClassificationRequest(
    context={"actor": "user"},           # "user" for prompts, "agent" for LLM responses
    classifier_description=ClassifierDescriptionDefault(),
    text="phone 321-507-0525 number",
)

response = client.classify(request)

for match in response.matches:
    print(f"{match.classifier}: '{match.text}' at [{match.start}:{match.end}] (confidence: {match.confidence})")
```

### Classifier Descriptions

Four ways to specify which classifiers to use:

```python
from ai_guard.api import (
    ClassifierDescriptionDefault,
    ClassifierDescriptionProfile,
    ClassifierDescriptionCodes,
    ClassifierDescriptionCode,
    ClassifierDescriptionJson,
)

# Default — uses the built-in classifier profile, no configuration needed
ClassifierDescriptionDefault()

# Profile — select a specific profile by UUID and version
ClassifierDescriptionProfile(uuid="7dbf380f-0af8-4276-acb0-85413db2dbff", version=1)

# Codes — target individual classifiers by code
ClassifierDescriptionCodes(codes=[
    ClassifierDescriptionCode(code="C1", version=1),
    ClassifierDescriptionCode(code="C2", version=2),
])

# JSON — provide inline classifier definitions
ClassifierDescriptionJson(classifiers=[{"name": "A"}, {"name": "B"}])
```

## Redaction

Apply redaction or blocking policies to classification results using `ClassificationRedactor`.

```python
from ai_guard.redact import ClassificationRedactor, RedactPolicy, RedactAction, RedactKind

policy = RedactPolicy(
    actions=[
        RedactAction(kind=RedactKind.REDACT, classifier="US_PHONE_NUMBER"),
        RedactAction(kind=RedactKind.BLOCK, classifier="US_SSN"),
    ],
    default=RedactKind.NONE,   # pass through classifiers not listed above
    redactor="*",              # character used to replace each redacted character
)

redactor = ClassificationRedactor(policy=policy)
result = redactor.redact(text=original_text, classification=response)
```

### Redaction Kinds

| Kind                | Behavior                                                              |
|---------------------|-----------------------------------------------------------------------|
| `RedactKind.NONE`   | Text passes through unchanged                                         |
| `RedactKind.REDACT` | Each character of the match is replaced with the `redactor` character |
| `RedactKind.BLOCK`  | The **entire text** is blocked — returns an empty string immediately  |

> **Note:** Block takes priority. If any match triggers a `BLOCK` action, the entire text is blocked regardless of other actions.

## Streaming Classification

Process text incrementally with concurrent classification using `ClassificationStream`. Accepts any iterable of strings — a generator, list, file, or LLM streaming response.

```python
from ai_guard import ClassificationStream
from ai_guard.api import ClassifierDescriptionDefault
from ai_guard.redact import ClassificationRedactor, RedactPolicy, RedactAction, RedactKind

policy = RedactPolicy(
    actions=[RedactAction(kind=RedactKind.REDACT, classifier="US_PHONE_NUMBER")],
    default=RedactKind.NONE,
    redactor="*",
)
redactor = ClassificationRedactor(policy)

source = ["First line here\n", "Phone: 321-507-0525\n", "Third line\n"]

stream = ClassificationStream(
    input=source,
    classifier_description=ClassifierDescriptionDefault(),
    client=client,
    context={"actor": "agent"},
    redactor=redactor,       # optional — omit for classification-only
    chunk_size=50,           # max characters per chunk (default: 100)
    max_workers=4,           # thread pool size (default: 4)
)

for result in stream:
    if result.redaction and any(a.kind == RedactKind.BLOCK for a in result.redaction.actions):
        print("Blocked!")
        break
    print(result.text, end="")
# First line here
# Phone: ************
# Third line
```

Each result yields `.text`, `.response` (raw classification), and `.redaction` (if a redactor is attached).

## Metrics

Send observability events to AI Guard for compliance monitoring in OneTrust AI Governance. The `agent_id` and `platform` are injected automatically.

```python
from ai_guard.api import MetricsEvent, MetricsEventMeter

# Record an LLM agent response time (milliseconds)
client.metric(MetricsEvent(
    attributes={},
    meter=MetricsEventMeter(name="ai_guard.agent", value="1234"),
))

# Record a user session
client.metric(MetricsEvent(
    attributes={"new_session": "true"},
    meter=MetricsEventMeter(name="ai_guard.user", value="1"),
))

# Record a redaction event
client.metric(MetricsEvent(
    attributes={"action": "redact", "actor": "user"},
    meter=MetricsEventMeter(name="ai_guard.redact", value="1"),
))
```

### Available Meters

| Meter                     | Type      | Description                                            |
|---------------------------|-----------|--------------------------------------------------------|
| `ai_guard.agent`          | Histogram | LLM agent response time in milliseconds                |
| `ai_guard.user`           | Counter   | User interaction / session event                       |
| `ai_guard.redact`         | Counter   | Redaction or block event                               |
| `ai_guard.classification` | Counter   | Classifier match count (auto-generated by the service) |

## Certificate Pinning

When your AI Guard service uses self-signed or internally-signed TLS certificates, use certificate pinning to verify the server's identity:

```python
client = AIGuardClient(
    "https://ai-guard.example.com:4443",
    token="your-api-key",
    agent_id="my-agent",
    platform=AIPlatform.AMAZON_BEDROCK,
    pin_sha256="x48Lk2iu3R3nAhSiz07bExGHTusDRjHqBx9ArK3cFGE=",
)
```

Extract the pin from a server certificate:

```bash
openssl x509 -in server.crt -pubkey -noout \
  | openssl pkey -pubin -outform DER \
  | openssl dgst -sha256 -binary \
  | base64
```

The pin is validated at construction time — invalid base64 or a digest that isn't exactly 32 bytes raises `ValueError` immediately.

**Pinning behavior:**

- CA chain verification is bypassed; the connection is secured by the pinned key alone
- Hostname verification is skipped; the URL can use an IP address or `localhost`
- Certificate rotation is transparent as long as the same key pair is reused

> **Note:** `session` and `pin_sha256` are mutually exclusive. If you supply your own `requests.Session`, `pin_sha256` is ignored. Use a custom session when you need standard CA-based verification (e.g., with a corporate CA bundle).

## Error Handling

Both `classify()` and `metric()` raise specific exceptions based on the HTTP response:

| HTTP Status | Exception         | Description                            |
|-------------|-------------------|----------------------------------------|
| 400         | `ValueError`      | Invalid request or metrics not enabled |
| 401         | `PermissionError` | Invalid or missing API key             |
| 500+        | `RuntimeError`    | Server error                           |
| Other       | `RuntimeError`    | Unexpected error                       |

```python
try:
    response = client.classify(request)
except ValueError as e:
    print(f"Bad request: {e}")
except PermissionError as e:
    print(f"Authentication failed: {e}")
except RuntimeError as e:
    print(f"Service error: {e}")
```

## Documentation

For complete documentation — including getting started guides, API reference, deployment, configuration, observability, and troubleshooting — visit the **OneTrust Developer Portal**:

**[https://developer.onetrust.com/onetrust/docs/ai-guard](https://developer.onetrust.com/onetrust/docs/ai-guard)**

## License

Copyright © OneTrust LLC. All rights reserved.

 