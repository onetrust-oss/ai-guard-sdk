# AI Guard SDK

Python SDK for AI Guard.

## Installation

Requires Python 3.13+.

### PyPI

**TODO: publish to PyPI and update this**

```bash
pip install ai-guard-sdk
```

### From source

```bash
git clone <repo-url>
cd ai-guard
python -m venv .venv
source .venv/bin/activate
pip install .
```

## Usage

### Classification

```python
from ai_guard import AIGuardClient
from ai_guard.api import (
    AIPlatform,
    ClassificationRequest,
    ClassifierDescriptionDefault,
)

client = AIGuardClient(
    "http://localhost:8888",
    token="your-token",
    agent_id="my-agent",
    platform=AIPlatform.AMAZON_BEDROCK,
)

request = ClassificationRequest(
    context={"actor": "user"},
    classifier_description=ClassifierDescriptionDefault(),
    text="phone 321-507-0525 number",
)

response = client.classify(request)

for match in response.matches:
    print(f"{match.classifier}: '{match.text}' at [{match.start}:{match.end}] (confidence: {match.confidence})")
# US_PHONE_NUMBER: '321-507-0525' at [6:18] (confidence: 100)
```

### Certificate Pinning

Pin the server's public key to verify its identity without the overhead of
managing internal DNS and CA-signed certificates for private IP addresses.
Pass `pin_sha256` — a base64-encoded SHA-256 hash of the server
certificate's Subject Public Key Info (SPKI), the same format used by curl's
`--pinnedpubkey` and OkHttp's `CertificatePinner`.

After the TLS handshake completes, the SDK extracts the server certificate's
public key, SHA-256 hashes it, and compares the result against the pinned
digest. If they don't match the connection is closed before any HTTP data is
sent. Because the pin is derived from the public key, not the certificate
itself, it survives certificate rotation as long as the same key pair is reused.

`session` and `pin_sha256` are **mutually exclusive**. If you supply your own
`requests.Session`, `pin_sha256` is ignored — the SDK assumes you have already
configured TLS on that session. Pinning is only applied when the SDK creates
its own session.

```python
from ai_guard import AIGuardClient
from ai_guard.api import AIPlatform

client = AIGuardClient(
    "https://guard.example.com",
    token="your-token",
    agent_id="my-agent",
    platform=AIPlatform.AMAZON_BEDROCK,
    pin_sha256="x48Lk2iu3R3nAhSiz07bExGHTusDRjHqBx9ArK3cFGE=",
)
```

Extract the pin from a certificate with openssl:

```bash
openssl x509 -in server.crt -pubkey -noout \
  | openssl pkey -pubin -outform DER \
  | openssl dgst -sha256 -binary \
  | base64
```

The pin is validated eagerly at construction time — invalid base64 or a digest
that isn't exactly 32 bytes raises `ValueError` immediately.

#### How pinning affects TLS verification

When `pin_sha256` is provided (and no `session` is given) the client disables
CA chain verification (`verify=False`) and replaces it with direct public key
comparison. This means:

- **Corporate CA bundles are bypassed.** The connection is secured by the
  pinned key alone, not by any certificate authority.
- **Hostname verification is skipped.** The server's certificate Common Name
  and Subject Alternative Names are not checked. Identity is established by
  the public key digest, so the URL can use an IP address or `localhost`
  without a matching certificate.
- **Certificate rotation is transparent.** As long as the new certificate
  uses the same key pair, the pin remains valid. If the key pair changes,
  you must update `pin_sha256` or the connection will be refused.

If you need standard CA-based verification (e.g. with a corporate CA bundle),
pass a pre-configured `requests.Session` with `verify` set to your CA bundle
path. Any `pin_sha256` value will be ignored when a `session` is provided.

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

# Default profile — uses the built-in classifier profile, no configuration needed
default = ClassifierDescriptionDefault()

# Profile-based (by UUID) — select a specific profile and version
profile = ClassifierDescriptionProfile(uuid="7dbf380f-0af8-4276-acb0-85413db2dbff", version=1)

# Code-based (by classifier codes)
codes = ClassifierDescriptionCodes(
    codes=[
        ClassifierDescriptionCode(code="C1", version=1),
        ClassifierDescriptionCode(code="C2", version=2),
    ]
)

# JSON-based (inline classifier definitions)
json_desc = ClassifierDescriptionJson(
    classifiers=[{"name": "A"}, {"name": "B"}]
)
```

### Metrics

Send metrics events to AI Guard for observability. See [METRICS.md](../METRICS.md) for meter definitions, server configuration, and the full observability pipeline.

```python
from ai_guard import AIGuardClient
from ai_guard.api import AIPlatform, MetricsEvent, MetricsEventMeter

client = AIGuardClient("http://localhost:8888", token="your-token", agent_id="my-agent", platform=AIPlatform.AMAZON_BEDROCK)

# Record an LLM response time (agent_id and platform are injected automatically by the client)
client.metric(MetricsEvent(
    attributes={},
    meter=MetricsEventMeter(name="ai_guard.agent", value="1.234"),
))

# Record a user interaction
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

Raises `ValueError` on `400`, `PermissionError` on `401`, and `RuntimeError` on other errors.

### Redaction

Redact sensitive data from text based on classification results.

```python
from ai_guard import AIGuardClient
from ai_guard.api import AIPlatform, ClassificationRequest, ClassifierDescriptionProfile
from ai_guard.redact import ClassificationRedactor, RedactPolicy, RedactAction, RedactKind

# Classify first (agent_id and platform are injected into context automatically)
client = AIGuardClient("http://localhost:8888", token="your-token", agent_id="my-agent", platform=AIPlatform.AMAZON_BEDROCK)
request = ClassificationRequest(
    context={"actor": "user"},
    classifier_description=ClassifierDescriptionProfile(
        uuid="7dbf380f-0af8-4276-acb0-85413db2dbff",
        version=1,
    ),
    text="phone 321-507-0525 number",
)
response = client.classify(request)

# Define a redaction policy
policy = RedactPolicy(
    actions=[
        RedactAction(kind=RedactKind.REDACT, classifier="US_PHONE_NUMBER"),
    ],
    default=RedactKind.NONE,  # leave unmatched classifiers unchanged
    redactor=" ",  # single character used to replace each redacted character
)

redactor = ClassificationRedactor(policy=policy)
result = redactor.redact(text="phone 321-507-0525 number", classification=response)

print(result.text)  # "phone              number"
print(result.actions)  # [RedactAction(kind=REDACT, classifier="US_PHONE_NUMBER")]
```

#### Redaction Kinds

- `RedactKind.NONE` - No action, text passes through unchanged.
- `RedactKind.REDACT` - Replace each character of the match with the `redactor` character.
- `RedactKind.BLOCK` - Block the entire text. Returns empty string immediately.

The `default` field on `RedactPolicy` sets the action for classifiers not explicitly listed in `actions`. For example, `default=RedactKind.REDACT` will redact all matched classifiers unless overridden by a specific action.

Block takes priority -- if any match triggers a `BLOCK` action, the entire text is blocked regardless of other actions.

### Streaming Classification

Process text incrementally with concurrent classification using `ClassificationStream`. Pass an iterable of text chunks as `input` and iterate the stream to get `ClassificationStreamResult` objects:

```python
from ai_guard import AIGuardClient, ClassificationStream
from ai_guard.api import AIPlatform, ClassifierDescriptionProfile
from ai_guard.redact import ClassificationRedactor, RedactPolicy, RedactAction, RedactKind

client = AIGuardClient("http://localhost:8888", token="your-token", agent_id="my-agent", platform=AIPlatform.AMAZON_BEDROCK)

# Optional: attach a redactor for inline redaction
policy = RedactPolicy(
    actions=[RedactAction(kind=RedactKind.REDACT, classifier="US_PHONE_NUMBER")],
    default=RedactKind.NONE,
    redactor="*",
)
redactor = ClassificationRedactor(policy)

# Any iterable of strings — a generator, list, file, boto3 stream, etc.
source = ["First line here\n", "Phone: 321-507-0525\n", "Third line\n"]

stream = ClassificationStream(
    input=source,
    classifier_description=ClassifierDescriptionProfile(
        uuid="7dbf380f-0af8-4276-acb0-85413db2dbff",
        version=1,
    ),
    client=client,
    context={"actor": "agent"},  # required: "user" or "agent"
    redactor=redactor,  # optional, omit for classification-only
    chunk_size=50,  # max characters per chunk (default: 100)
    max_workers=4,  # thread pool size (default: 4)
)

# Each result has .text, .response, and .redaction
for result in stream:
    if result.redaction and any(a.kind == RedactKind.BLOCK for a in result.redaction.actions):
        print("Blocked!")
        break
    print(result.text, end="")
# First line here
# Phone: ************
# Third line
```

When a `BLOCK` action is triggered, the result is still yielded so you can handle it inline.

## Error Handling

Both `classify()` and `metric()` raise specific exceptions based on HTTP status codes:

- **400** - `ValueError` with the service's error message
- **401** - `PermissionError` with the service's error message
- **500+** - `RuntimeError` with the service's error message
- **Other** - `RuntimeError` with the service's error message

## Development

Install from source with dev dependencies (pytest, coverage):

```bash
pip install ".[dev]"
```

Run unit tests:

```bash
pytest
```

### Integration tests

Integration tests launch real AI Guard instances and require the
test framework Docker services to be running. These services provide InfluxDB,
an OpenTelemetry Collector, and a credentials server that provisions API tokens.
See [`testing/README.md`](../testing/README.md) for full details on the test
framework.

**Start the services** before running integration tests. On a workstation, copy
`workstation.env` to `.env` at the project root (or source it directly) so
`TestingConfig` can find the required environment variables, then bring up the
Docker Compose stack:

```bash
cp workstation.env .env
docker compose -f testing/docker-compose.yaml --env-file testing/ci.env up -d
```

In CI, source `testing/ci.env` into the shell instead -- the containers address
each other by Docker network name rather than `localhost`:

```bash
source testing/ci.env
docker compose -f testing/docker-compose.yaml up -d
```

**Run the integration tests** once the services are healthy:

```bash
.venv/bin/python -m pytest ai_guard/tests/test_classification_integration.py ai_guard/tests/test_stream_integration.py -v
```

Each test class starts its own classification server on a unique port, so test
files run in parallel without conflicts. The test servers shut down automatically
when the test class tears down.

