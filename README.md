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

Register one or more agents up front, then activate whichever agent is running in the current scope:

```python
import os
from ai_guard import AgentConfig, AIGuardClient

client = AIGuardClient(
    os.environ["AI_GUARD_URL"],           # e.g. https://ai-guard.example.com:4443
    token=os.environ["AI_GUARD_TOKEN"],   # OneTrust API key with Data Discovery scope
    agents=[
        # agent_id: OneTrust registered agent identifier (UUID). Maps to the agent declared in OneTrust governance/intake
        AgentConfig(name="support-assistant", agent_id="312b5d93-b2c8-48fc-95a4-f08e9e667be4"),
        AgentConfig(name="research-crew", agent_id="4c3eb3ab-4b78-4d95-8d1d-50ca4da57e3f"),
    ],
)

with client.activate("support-assistant"):
    client.classify(request)   # uses support-assistant

with client.activate("research-crew"):
    client.classify(request)   # uses research-crew
```

`agents` entries can also be plain mappings (e.g. loaded from your own YAML/JSON config) with `name` and `agent_id` keys. Activating an unregistered name raises `ValueError`; malformed, incomplete, or duplicate entries in `agents` raise `ValueError` at construction time.

**Activation is context-local**, backed by `contextvars` — not a mutable attribute on the client. It nests correctly, restores the outer agent automatically (including when an exception is raised), and is isolated per thread and per asyncio task, so two threads or two asyncio tasks activating different agents never interfere with each other. `with client.activate(...):` is a regular (sync) context manager and works fine inside `async def` functions. `ClassificationStream` also respects whichever agent is active when it's constructed, including across its internal thread pool.

When the agent scope is an entire function — which is typically the case for a framework node, a tool handler, or a route — `@client.agent(name)` is the same thing without the extra indentation level. See [Decorator Activation](#decorator-activation) below.

**A single agent per process is still fully supported** — pass `agent_id` directly and skip `agents`/`activate()` entirely. This is exactly today's behavior for existing integrations:

```python
client = AIGuardClient(
    os.environ["AI_GUARD_URL"],
    token=os.environ["AI_GUARD_TOKEN"],
    agent_id="my-agent",
)
```

If neither a default `agent_id` nor an active agent is available when `classify()`/`metric()` is called, a `RuntimeError` is raised.

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

## Decorator Activation

`@client.agent(name)` is the decorator form of `activate()`. It is **exactly equivalent** to wrapping the whole function body in `with client.activate(name):` — same `contextvars`-backed activation, same nesting, same restore-on-exception, same per-thread and per-asyncio-task isolation. It exists because in practice an agent boundary is usually one whole function, and the `with` block then costs an indentation level on every handler:

```python
# equivalent
@client.agent("support-assistant")
def handle(message):
    return client.classify(request)

def handle(message):
    with client.activate("support-assistant"):
        return client.classify(request)
```

Register agents on the client as usual, then decorate:

```python
import os
from ai_guard import AgentConfig, AIGuardClient
from ai_guard.api import ClassificationRequest, ClassifierDescriptionDefault

client = AIGuardClient(
    os.environ["AI_GUARD_URL"],
    token=os.environ["AI_GUARD_TOKEN"],
    agents=[
        AgentConfig(name="support-assistant", agent_id="312b5d93-b2c8-48fc-95a4-f08e9e667be4"),
        AgentConfig(name="research-crew", agent_id="4c3eb3ab-4b78-4d95-8d1d-50ca4da57e3f"),
    ],
)

@client.agent("support-assistant")
def handle_support(user_text):
    # every classify()/metric() here is attributed to support-assistant, and
    # every span created here carries onetrust.reference.id for that agent
    return client.classify(ClassificationRequest(
        context={"actor": "user"},
        classifier_description=ClassifierDescriptionDefault(),
        text=user_text,
    ))

@client.agent("research-crew")
def handle_research(user_text):
    return client.classify(...)
```

### Supported function kinds

All four are handled; for the two generator forms the agent is active for the duration of **iteration**, not merely for the call that creates the generator — so a streaming handler that yields between `classify()` calls still resolves to the right agent on every one of them.

```python
@client.agent("support-assistant")
async def handle_async(user_text):          # coroutine function
    return await run_llm(user_text)

@client.agent("support-assistant")
def stream(chunks):                         # generator function
    for chunk in chunks:
        client.metric(...)                  # correct agent_id on every chunk
        yield chunk

@client.agent("support-assistant")
async def stream_async(chunks):             # async generator function
    async for chunk in chunks:
        client.metric(...)
        yield chunk
```

`functools.wraps` is applied, so `__name__`, `__doc__`, the signature, and the coroutine/generator function flags are preserved — frameworks that introspect what you hand them (FastAPI, LangChain tool schemas, `inspect.iscoroutinefunction` checks) see the original function's shape.

### Composing with `activate()`

Use `activate()` — not the decorator — when the agent scope is *not* the whole function: a mid-function handoff, a loop over several agents, or a name chosen at runtime. The two nest and restore correctly in any combination:

```python
@client.agent("support-assistant")
def handle(message):
    result = client.classify(...)                 # support-assistant
    if result.matches:
        with client.activate("research-crew"):
            escalate(message)                     # research-crew
    return client.classify(...)                   # support-assistant restored
```

### Behavior notes

- **A misspelled name fails at import, not at first call.** `@client.agent("supprt")` raises `ValueError` at decoration time, so the client must already be constructed (with its `agents` registered) when the decorated module is imported. This is usually what you want for a name typo; it does mean the decorator binds one specific client instance, so if you need to swap clients per test or per tenant, prefer `activate()` with the client passed in as an argument.
- **Each call gets its own activation**, so a single decorated function is safe to call concurrently from many threads or asyncio tasks.
- **Thread handoff still needs an explicit capture.** Like `activate()`, the decorator does not follow work submitted to a raw `Thread` or `ThreadPoolExecutor` — a decorated function is not a blanket guarantee over its whole call tree:

  ```python
  from ai_guard.agent import bind_captured_agent, capture_active_agent

  @client.agent("support-assistant")
  def fan_out(items):
      slot = capture_active_agent()          # on the originating thread
      with ThreadPoolExecutor() as pool:
          return list(pool.map(lambda i: _work(slot, i), items))

  def _work(slot, item):
      bind_captured_agent(slot)              # first thing in the worker
      return client.classify(...)
  ```

  (`ClassificationStream` already does this internally, so it needs no special handling.)
- **`.asend()` values are not forwarded into decorated async generators.** The sync generator form does forward `.send()`/`.throw()` normally; if you need a send-based protocol on an *async* generator, use `with client.activate(...)` inside it instead.
- The one-time "no `TracerProvider` configured" warning ([below](#warning-when-no-tracerprovider-exists-at-all)) fires on the first **call**, not at decoration time — a provider set up in `main()` isn't configured yet at import.

## Client Parameters

| Parameter    | Type                | Required | Description                                                                             |
|--------------|---------------------|----------|------------------------------------------------------------------------------------------|
| `base_url`   | `str`               | Yes      | Base URL of your AI Guard service                                                       |
| `token`      | `str`               | Yes      | OneTrust API key with Data Discovery scope                                              |
| `agents`     | `list[AgentConfig]` | No       | Agents to register for `activate()` / `@agent()` (see [Initialize the Client](#1-initialize-the-client)) |
| `agent_id`   | `str`               | No       | Default agent identifier, used whenever no agent is activated                          |
| `platform`   | `AIPlatform`        | No       | AI platform originating requests (e.g. `AIPlatform.AMAZON_BEDROCK`), shared by every agent in this process |
| `pin_sha256` | `str`               | No       | Certificate pin for TLS verification (see [Certificate Pinning](#certificate-pinning))  |
| `session`    | `requests.Session`  | No       | Custom session for advanced TLS configuration                                           |

At least one of `agents` or `agent_id` must be usable at call time — either register agents and `activate()` one, or pass `agent_id` for single-agent use. `platform` is optional; when set it applies to every request regardless of which agent is active, and is only included in a request's payload when configured.

## Guardrail Enforcement (Amazon Bedrock)

The SDK can enforce Amazon Bedrock Guardrails per agent. You configure the client with a
`region` (and, optionally, an `account`); the SDK resolves each agent's guardrails from the
OneTrust service (lazily on first use, sticky-cached), and runs Bedrock `ApplyGuardrail`
on request and response text. You never write boto3 guardrail code yourself.

Install the extra:

```bash
pip install "onetrust-ai-guard-sdk[bedrock]"
```

**AWS requirements.** The SDK builds its own `bedrock-runtime` and STS clients from `region` and the
ambient AWS credential chain (environment, instance/task role, or container role), so the runtime
identity needs the `bedrock:ApplyGuardrail` permission for the guardrails it enforces. The AWS
account is resolved automatically via STS unless you pass `account` explicitly.

Credentials, endpoints and retries are configured the **standard AWS way** — no SDK knobs:

- **Credentials / profile / role** — the default chain (`AWS_PROFILE`, env vars, instance/task or
  execution role).
- **Custom or PrivateLink endpoints** — per-service endpoint config, e.g.
  `AWS_ENDPOINT_URL_BEDROCK_RUNTIME` and `AWS_ENDPOINT_URL_STS` (or `endpoint_url` in your AWS config
  profile). Passing `account` explicitly skips the STS call entirely.
- **Retries** — `AWS_MAX_ATTEMPTS` / `AWS_RETRY_MODE`.

Wrap each agent's model call:

```python
guard = AIGuardClient(
    base_url=os.environ["AI_GUARD_URL"],   # e.g. https://ai-guard.example.com:4443
    token=os.environ["AI_GUARD_TOKEN"],    # OneTrust API key with Data Discovery scope
    region=os.environ["AWS_REGION"],       # required for enforcement
    agents=[AgentConfig(name="researcher", agent_id="…uuid…")],
)

with guard.activate("researcher"):
    r = guard.check_input(user_msg)
    if r.blocked:
        return "I can't help with that."   # r.action_reason explains why
    out = call_model(r.text)               # forward r.text — it may be MASKED
    r = guard.check_output(out)
    return "I can't help with that." if r.blocked else r.text
```

`check_input(text)` / `check_output(text)` return a `GuardResult`:

| Field | Meaning |
|---|---|
| `blocked` | `True` iff some assessment action was `BLOCKED`. Stop the turn. |
| `text` | Safe text to forward — masked when a PII/anonymize policy fired, else the original. **Always forward this, never the original.** |
| `action` | `ACTION_NONE` or `ACTION_INTERVENED` (`from ai_guard import ACTION_NONE, ACTION_INTERVENED`). |
| `action_reason` | Reason string when blocked. |
| `assessments` | Raw per-guardrail assessments (debug/logging). |
| `guardrail_id` | The guardrail that blocked, if any. |

Streaming agents can opt into `check_output_stream(tokens, emit)`, which buffers output to
`guardrail_stream_buffer_size` (default 25K chars), checks each buffer, and stops on a block
(already-emitted buffers cannot be recalled).

### Configuration

These constructor parameters control enforcement. All are optional; `platform` stays `None`.

| Property | Type | Default | Required? | Purpose / tuning notes |
|---|---|---|---|---|
| `base_url` | str | — | yes | AI Guard service URL (e.g. `https://ai-guard.example.com:4443`), the same one used for classification and metrics. The service resolves guardrails from OneTrust on the SDK's behalf. |
| `token` | str | — | yes | OneTrust API key; sent as `Authorization: Bearer`. From secret/config, never hardcoded. |
| `agent_id` | str | `None` | single-agent | The OneTrust agent UUID for single-agent use. |
| `agents` | list[AgentConfig] | `None` | multi-agent | Registry of `AgentConfig(name, agent_id)`; use with `activate(name)` / `@agent(name)`. |
| `platform` | AIPlatform\|None | `None` | no | **Leave `None`.** Changing it alters existing `classify`/`metric` payloads. Enforcement sets its platform internally. |
| `region` | str | `None` | yes (enforcement) | AWS region for regional guardrails + the bedrock-runtime client. **From env config, not a literal.** Missing → fail fast. |
| `account` | str | `None` | no | 12-digit AWS account string. Omit → auto-resolved via STS. Supply to override/validate, or to skip the STS call. |
| `on_error` | `BLOCK`\|`ALLOW` | `BLOCK` | no | On an `apply_guardrail` **call** error mid-turn: `BLOCK` (fail-closed) stops the turn; `ALLOW` continues ungoverned. |
| `on_resolve_failure` | `BLOCK`\|`ALLOW` | `BLOCK` | no | On a resolve failure **with no cached list**: `BLOCK` blocks turns until resolved; `ALLOW` runs ungoverned meanwhile. |
| `guardrail_resolve_timeout` | float (s) | `3.0` | no | Timeout for the one bounded inline resolve on a cold cache miss (concurrent callers wait on it). Must be `> 0`. |
| `guardrail_resolve_max_attempts` | int | `3` | no | Attempts for the inline resolve on transient failures (5xx / 429 / timeout / transport). Terminal errors are never retried. Must be `>= 1`. |
| `guardrail_resolve_retry_backoff` | float (s) | `0.25` | no | Base delay for the inline retry backoff — sleeps `backoff * 2**attempt` between attempts. Must be `>= 0`. |
| `guardrail_retry_interval` | int (s) | `60` | no | After a failed first resolve, minimum spacing between throttled async self-heal retries; the agent path never blocks again. Must be `> 0`. |
| `guardrail_cache_ttl` | int (s) | `86400` (24h) | no | Sticky cache lifetime. On expiry, background refresh; **never evicts on failure** (last-known-good). |
| `guardrail_chunk_size` | int (chars) | `100000` | no | Per-request char budget for `ApplyGuardrail` in `check_input`/`check_output`. Larger texts are split into ≤25K line-aligned content blocks packed into requests up to this budget; blocks are evaluated independently in one call and requests run in parallel. Sized under the strictest observed per-policy cap (denied-topic Classic tier = 106 text units/request, summed across blocks); the raw request ceiling is ~14MB but per-policy caps are lower. Must be `> 0`. |
| `guardrail_stream_buffer_size` | int (chars) | `25000` | no | Emission buffer for `check_output_stream`. Keep modest — already-emitted buffers cannot be recalled on a late block. Must be `> 0`. |

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

Five ways to specify which classifiers to use:

```python
from ai_guard.api import (
    ClassifierDescriptionDefault,
    ClassifierDescriptionProfile,
    ClassifierDescriptionProfileName,
    ClassifierDescriptionCodes,
    ClassifierDescriptionCode,
    ClassifierDescriptionJson,
)

# Default — uses the built-in classifier profile, no configuration needed
ClassifierDescriptionDefault()

# Profile — select a specific profile by UUID and version
ClassifierDescriptionProfile(uuid="7dbf380f-0af8-4276-acb0-85413db2dbff", version=1)

# Profile name — select a profile by name (resolved server-side)
ClassifierDescriptionProfileName(name="My Profile")

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

## OpenTelemetry Span Attribution

`onetrust.reference.id` is the attribute OneTrust uses to identify **which agent** produced a given span — it's how multi-agent telemetry gets attributed back to `acme/support-assistant` vs. `acme/research-crew` rather than all blurring into one undifferentiated stream. The SDK provides two ways to set it, and most multi-agent setups want both:

| Mechanism | What it stamps | Setup required |
|---|---|---|
| `classify()` / `metric()` | The one span current at the exact moment either is called | Just install the `otel` extra — nothing to register |
| `OneTrustSpanProcessor` | **Every** span created while an agent is active (LLM calls, tool calls, anything your own instrumentation creates) | Install `otel-sdk` **and** register it on your `TracerProvider` *before* any `activate()` call — see steps below |

### `classify()` / `metric()` (automatic, no registration)

```bash
pip install "onetrust-ai-guard-sdk[otel]"
```

While inside a `client.activate(name)` scope — or the body of an `@client.agent(name)`-decorated function, which is the same scope — `classify()` and `metric()` set `onetrust.reference.id` on the current span via `opentelemetry.trace.get_current_span()`. The SDK never creates a `TracerProvider`, `Resource`, `SpanProcessor`, or exporter — it only annotates whatever span your own instrumentation already has current, at the moment `classify()`/`metric()` is actually called. Outside an `activate()` scope (including all single-agent usage), no OTel calls are made at all. If `opentelemetry-api` isn't installed, this is a no-op — nothing else in the SDK is affected.

### `OneTrustSpanProcessor` (stamps every span — you must register it)

This is the piece that gets `onetrust.reference.id` onto your actual agent/LLM spans, not just whatever's current when you happen to call `classify()`. **It does nothing until you add it to your `TracerProvider` yourself** — the SDK never touches your OTel setup on its own, so this is a step you have to take, not something that happens automatically:

1. Install the `otel-sdk` extra (pulls in `opentelemetry-sdk`, and `-api` transitively):

   ```bash
   pip install "onetrust-ai-guard-sdk[otel-sdk]"
   ```

2. Register `OneTrustSpanProcessor` on your `TracerProvider`, alongside whatever exporter(s) you already use — do this once, at startup, **before** any `client.activate(...)` call:

   ```python
   from opentelemetry import trace
   from opentelemetry.sdk.trace import TracerProvider
   from opentelemetry.sdk.trace.export import BatchSpanProcessor
   from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
   from ai_guard import OneTrustSpanProcessor

   provider = TracerProvider()
   provider.add_span_processor(OneTrustSpanProcessor())                     # stamps onetrust.reference.id
   provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))   # your own exporter, unaffected
   trace.set_tracer_provider(provider)
   ```

3. Use `client.activate(name)` (or `@client.agent(name)`) as normal from here on — every span created for the rest of that scope now carries `onetrust.reference.id` set to that agent's `agent_id`, with no further code changes at your call sites.

**Register it early — before the first `activate()` call.** `OneTrustSpanProcessor` reads whichever agent is active at each span's *creation* time, so a span opened before `activate()` is entered won't get the attribute even if `activate()` is still in effect when that span later ends — see the demo script (`test_otel_agent/multi_agent_demo/`) for a worked example, including where the ordering matters.

If you skip step 2 entirely, spans are created without `onetrust.reference.id` — silently, no error or warning, since the SDK has no way to know you meant to register a processor at all (it could be intentional if you only want the lighter-weight `classify()`/`metric()` stamping above). The one exception: if you also never configured a `TracerProvider` at all, `activate()` warns you once — see [below](#warning-when-no-tracerprovider-exists-at-all). But if a `TracerProvider` already exists (e.g. you already export traces elsewhere) and you just forget to add this specific processor to it, that's the one case the SDK genuinely cannot detect — worth double-checking step 2 by hand if you're already running OTel for other purposes. Constructing `OneTrustSpanProcessor()` without `opentelemetry-sdk` installed raises a clear `ImportError` telling you what to install, rather than failing on import or silently doing nothing.

### Warning when no `TracerProvider` exists at all

Separately from the processor-registration check above (which the SDK can't safely detect — see why below), `client.activate(...)` emits a one-time `UserWarning` per process if `opentelemetry-api` is installed but **no real `TracerProvider` has been configured anywhere** — i.e. `opentelemetry.trace.get_tracer_provider()` still returns the default no-op proxy. In that state, neither `classify()`/`metric()` stamping nor `OneTrustSpanProcessor` can ever take effect, since there's no recording span to stamp, so this is a strong enough signal to be worth surfacing:

```
UserWarning: AIGuardClient.activate() was called, but no OpenTelemetry TracerProvider is configured in this
process -- onetrust.reference.id will not be stamped on any spans. If you want OTel span tagging, call
trace.set_tracer_provider(...) (and register OneTrustSpanProcessor on it, if you want every span tagged
rather than just classify()/metric() call sites) before activate() is used. See the README's OTel section
for details.
```

This warning does **not** fire just because `OneTrustSpanProcessor` specifically isn't registered on an otherwise-valid `TracerProvider` — detecting that would mean reaching into OTel SDK-private internals, and would false-positive on anyone intentionally using only the lighter-weight `classify()`/`metric()` stamping. It only fires on the stronger, unambiguous signal that no provider exists at all. If `opentelemetry-api` isn't installed, this check is a no-op like everything else OTel-related in this SDK.

## Metrics

Send observability events to AI Guard for compliance monitoring in OneTrust AI Governance. The active agent's `agent_id` (and `platform`, if configured) are injected automatically.

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

 