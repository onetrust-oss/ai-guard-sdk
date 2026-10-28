---
name: onetrust-ai-guardrails
description: >
  Integrate the OneTrust AI Guard SDK and wire OneTrust-governed Amazon Bedrock guardrails
  into a Bedrock AgentCore agent. Use when a developer is building or editing a Bedrock agent
  and wants governance guardrails on model input and output. The skill **always selects a
  OneTrust project first (mandatory)** — projects, then agents — then searches the named agents;
  if one doesn't exist it creates it (name, description, autonomy level) **always linked to that
  project** via the onetrust MCP. Project selection and the project→agent link are never
  skipped, even when the request is only about an agent. It then confirms the AWS region,
  determines the framework and model-invocation site(s) for check placement, and wires per-agent
  guardrail checks through the SDK. SCOPE: this skill only handles OneTrust SDK integration and
  guardrail wiring — not designing or building the agent itself (prompts, tools, orchestration,
  business logic). Ask only what's needed to integrate the SDK; never invent agent or guardrail
  IDs.
---

# OneTrust AI Guardrails — SDK integration skill

**Goal:** integrate the **OneTrust AI Guard SDK** so every model-invoking agent has **input and output guardrail checks**, each keyed to its **OneTrust agentId**. The agent's model must be invoked through **Amazon Bedrock** (the SDK enforces Amazon Bedrock Guardrails).

## Scope — read first
This skill does **one thing: OneTrust guardrail SDK integration.** It is not an agent-building assistant.
- **In scope:** selecting or creating the OneTrust agent record; showing the OneTrust policies applied to it; confirming the model runs on Amazon Bedrock; the AWS region; the **framework and model-invocation site(s)** — enough to know *where and how* to place the checks; adding the SDK dependency and configuration; inserting the guardrail checks.
- **Out of scope:** *which* framework to adopt and how to design or build the agent — its purpose, prompts, tools, orchestration, model choice, or business logic, and any code beyond the guardrail wiring. Those belong to the developer.
- **"Framework" has two senses — keep them apart:** *what framework the code uses and where the model is called* is **in scope** (it decides where the checks go); *how to build the agent in that framework* is **out of scope**.
- Only ask a question if the answer is needed to place a guardrail check. The skill needs the **framework + model-invocation site** — get them by inspecting existing code, or by asking the developer/coding agent (or defer per Step 5). Don't guess, and don't drift into how to build the agent.

**What the developer writes vs. what the SDK does:** the developer configures the SDK with the OneTrust **agentId** and an AWS **region**, and calls the SDK's check methods around the model call. The SDK resolves that agent's guardrails from OneTrust and applies them via the Amazon Bedrock Guardrails `ApplyGuardrail` API. **The developer never writes guardrail or boto3 code.**

## Prerequisites
- The **`onetrust` MCP** is configured (provides `search_ai_governance_projects`, `search_ai_governance_aiagents`, `get_ai_governance_attributes`, `get_ai_governance_link_types`, `create_ai_governance_aiagents`, and `get_ai_governance_policies`).
- The **OneTrust AI Guard SDK** with the Bedrock extra is installable: `pip install "onetrust-ai-guard-sdk[bedrock]"` (imports as `ai_guard`).
- A **OneTrust API key** and **base URL** are available — or will be set (see Step 6) — via the app's config/secrets, never hardcoded.
- The agent runs on **Amazon Bedrock** with ambient AWS credentials and a known **region**.
- The agent **may or may not exist yet** — if it isn't in the chosen project, the skill creates it.

## Non-negotiable rules
- **Project first — always. Never skip Step 1.** Select a OneTrust **project** before you search, pick, create, or wire any agent — even when the request is only "create/wire an agent." No agent step runs until a project **`id` (projectId)** and its **`orgId`** are recorded. Every agent you create is **linked to that project**; creating an agent without a project link is invalid — if you cannot form the link, stop and resolve the project first rather than proceeding.
- **Stay in scope.** Only ask questions needed to integrate the SDK / wire guardrails; defer agent design and build to the developer.
- **Never author the agent.** Do not create a runnable agent, an example entrypoint, a starter model call, hardcoded model IDs, or a bespoke wrapper abstraction (e.g. a `guarded_model_call(...)` you invented). Only **insert `check_input`/`check_output` into the developer's real model-invocation site**, or hand them the check **snippets to place** there. **If no call site exists yet, do not scaffold one** — hand the build back to the developer / coding agent and defer wiring (Step 5B).
- **Fit the codebase, don't impose a shape.** Adapt to the repo's existing structure, config, naming, error handling, and framework conventions. Enterprises structure agents their own way — never introduce POC scaffolding or a parallel structure to make wiring "work."
- **Use the OneTrust agentId** (the entity `id` from `search_ai_governance_aiagents` or `create_ai_governance_aiagents`) for every agent. **Never invent, rename, or guess an agentId or a guardrailId.**
- **Wire both `check_input` and `check_output`** for **every** model-invoking agent.
- **Confirm, don't infer, the integration facts — framework, topology (single-agent / one-class-per-agent / workflow), region, where the model is invoked.** These decide where and how many checks go. Confirm them with the developer even when the code hints at an answer; a sparse or empty repo is not evidence of "single-agent." Detection is a hint, not a decision. (Don't ask how to *build* the agent — only what's needed to place checks.)
- **The developer references only the SDK** for guardrails. Don't put raw `apply_guardrail` calls or guardrail IDs in the developer's code — the SDK owns that.
- **Don't assume the framework exposes a reusable Bedrock client.** By default the SDK creates its own from the region and ambient AWS credentials; only inject one for a custom endpoint or specific credentials.
- **Verify before you write.** Confirm the SDK's method names against its README before generating code; if a name differs, use the SDK's actual name and tell the developer.
- **Amazon Bedrock is required.** If the model isn't invoked through Amazon Bedrock, stop — this skill only enforces Amazon Bedrock Guardrails.

> **Version check — first.** This skill needs a OneTrust AI Guard SDK that provides the `[bedrock]` extra and input/output guardrail methods, plus an `onetrust` MCP exposing `search_ai_governance_projects`, `search_ai_governance_aiagents`, `get_ai_governance_attributes`, `get_ai_governance_link_types`, and `create_ai_governance_aiagents`. If any of these is missing, stop and tell the developer to upgrade. `get_ai_governance_policies` is used only for the final policy summary (Step 9) — if it's missing, skip that summary and say so; don't block the integration. Older MCP versions expose a single `get_ai_governance_aiagents_schema` (attributes + link types) and an `entityName` search parameter — if that's what the installed MCP offers, use those instead. If a tool or guardrail method has a different name than shown here, use the installed name.

---

## Step 1 — Pick the project (mandatory, always first)
**Do this before anything agent-related.** Even if the developer only asked to create or wire an agent, start here — do not search, create, or wire an agent until this step is complete.
1. Call **`search_ai_governance_projects`** — with no `criteria` to list projects, or `criteria: {"criteria": {"fullText": "<name>"}}` to filter by name (a substring match). Results are `{totalCount, entities: [{id, name, description, orgId}]}`, **10 per page by default** — if `totalCount` is larger than what you received, page through (`page`, `size`) so you don't present a truncated list.
2. Show each project as **name · id** (add `description` if present); ask which project the developer is building in. **Always ask for explicit confirmation — never auto-select.** Even if only **one** project is returned, or one **matches the name** the developer mentioned, do not assume it; present it and get an explicit **yes** ("Use project **`<name>`** (`<id>`)?") before recording it. If none fits, have the developer pick or create one in OneTrust — do not invent a project or proceed without one.
3. Record the project's **`id`** (the projectId) **and its `orgId`** — the `orgId` is required as `organizationId` when creating an agent in Step 3, and the projectId is required for the mandatory project→agent link.

> **Gate — do not pass until this is true:** a project **`id`** and its **`orgId`** are recorded **after the developer explicitly confirmed the project** (no auto-selection, even for a single or name-matching project). Steps 2–3 (search/create agent) must not run before this. If you catch yourself about to search or create an agent without a developer-confirmed projectId, stop and return here.

## Step 2 — Pick the agent(s)
**Precondition: Step 1 is complete** — a projectId + orgId are recorded. If not, go back to Step 1 first; do not start here.

The MCP searches agents **globally by name — there is no per-project agent listing**. This is a lookup limitation only; it does **not** mean projects are optional. Do not present an unfiltered agent list as "the agents in this project," and do not treat the global search as a reason to skip project selection.
1. Ask the developer for the **name(s)** of the agent(s) they are building (for multi-agent apps, expect several).
2. Call **`search_ai_governance_aiagents`** for each, with `criteria: {"criteria": {"fullText": "<name>"}}`. `fullText` is a **substring** match, so results can include other agents whose names merely contain the term (e.g. searching "Support" also returns "Customer Support Agent"). Page through if `totalCount` exceeds the returned page.
3. If an agent's name **exactly** matches (case-insensitive) → record its **`id` exactly as returned** (this is the agentId) and its **`orgId`**. Search only returns records the user can access — including agents in child orgs — so an `orgId` different from the project's is fine to use. If several exact matches come back, or only partial matches, show them and ask the developer which one they mean (or whether it's new) — never pick on their behalf. Found → skip Step 3, go to Step 4.
4. If not found ("it's new") → Step 3.

## Step 3 — Create the agent (only if it doesn't exist yet)
**Precondition: a projectId + orgId from Step 1 are recorded.** Creating an agent without them — and without the project link in point 6 — is invalid; do not call `create_ai_governance_aiagents` until you have both.

Discover what the tenant allows first — always pass **`entityTypeName: "aiagents"`**. Use only what these tools return; never invent attribute names, option values, or link-type names.
1. **Attributes:** call **`get_ai_governance_attributes(entityTypeName="aiagents")`** (no `attributeNames`). It returns attribute **display names** grouped by type under `attributes`, plus `required` and `systemManaged` (read-only). Confirm **Description** and **Autonomy Level** are present and not in `systemManaged`. Required attributes that are also in `systemManaged` are set by OneTrust — ignore them, and ignore **Name** (passed as the top-level `name`). If any **other** required attribute is creatable (listed under `attributes`, not `systemManaged`), stop: this skill can't set it (the MCP doesn't expose its internal fieldName) — ask the developer to create the agent in OneTrust instead, then return to Step 2 to look it up.
2. **Autonomy options:** call **`get_ai_governance_attributes(entityTypeName="aiagents", attributeNames=["Autonomy Level"])`** — `attributeOptions["Autonomy Level"]` holds the allowed labels.
3. **Project link type:** call **`get_ai_governance_link_types(entityTypeName="aiagents")`** — it returns `linkTypes: [{name, targetEntityTypes}]`. Pick the link type whose `targetEntityTypes` includes **`projects`**. If several qualify, ask the developer which to use; if none does, stop and tell the developer — never create an unlinked agent.
4. Ask the developer for exactly:
   - **Name** — the agent's name.
   - **Description** — a short description of the agent.
   - **Autonomy level** — have the developer choose from the labels returned in point 2. Use the exact label; never invent one.
5. **Confirm before creating** — show the exact record, **including the project it will be linked to**, and get an explicit **yes**: *"Create agent **`<name>`** in project **`<projectName>`** (`<projectId>`) with autonomy level **`<level>`** and the description "…"? After you create it, you can edit or delete it only in OneTrust."* Don't call `create_ai_governance_aiagents` without confirmation. Get it right the first time: the MCP has **no update or delete tool**, so a wrong record can only be fixed or removed by the developer in the OneTrust UI.
6. Call **`create_ai_governance_aiagents`** with:
   - `name` — the agent name as **plain text, exactly as confirmed**. Never HTML/XML-escape or URL-encode it: `&` stays `&` (not `&amp;`), and likewise `<`, `>`, `"`, `'` (not `&lt;`, `&gt;`, `&quot;`, `&#39;`). The same applies to `description`. Before calling, check the values: if any contains an entity such as `&amp;`, `&lt;`, or `&#39;`, replace it with the literal character first.
   - `organizationId` — the **project's `orgId`** recorded in Step 1
   - `attributes` — `{"description": "<desc>", "autonomyLevel": "<label from point 2>"}`. Keys are the **internal fieldNames** `description` and `autonomyLevel` — **not** the display names ("Description", "Autonomy Level") that `get_ai_governance_attributes` returns; the value is the exact option label. Set **only these two** — don't set any other attribute, even if the schema lists it or a policy asks for it; the developer fills in anything else in OneTrust.
   - `links` — **mandatory, never omit:** `[{"linkTypeName": "<link type name from point 3>", "targetEntityId": "<project id from Step 1>", "targetEntityType": "projects"}]` — `targetEntityType` is exactly the plural `projects`.
7. Record the returned **`id` exactly** — that is the agentId. The response is `{id, name, number, entityTypeName, organizationId}` and doesn't echo links; links are validated before creation and created together with the agent, so a successful response means the project link is in place. If the call errors (e.g. link resolution failed), nothing was created — fix the input and retry; **never retry without the link**.
8. Tell the developer the agent is created and may not have guardrails configured yet; wiring can proceed now, and enforcement begins automatically once guardrails are set up for it in OneTrust.
9. Repeat Steps 2–3 for any additional new agents (points 1–3 only need to run once per session).

## Step 4 — Confirm Amazon Bedrock, then get the region
1. **Confirm the model is invoked through Amazon Bedrock.** Infer from the code (boto3 `bedrock-runtime` calls; framework model classes like `ChatBedrock`, Strands `BedrockModel`, LlamaIndex `BedrockConverse`; Bedrock model IDs such as `anthropic.claude-*`), then confirm with the developer. If it's another provider, stop — this skill doesn't apply.
2. **Get the AWS region** (e.g. `us-east-1`) — required. Source it from environment config, not a code literal; reuse the repo's existing region if it reads one from config.
3. Don't ask for the AWS account — the SDK resolves it at runtime.

## Step 5 — Determine the framework and where to wire (two modes)
Placement depends on the framework and where the model is invoked, so establish both — but only to place checks, never to advise on building the agent. Handle the two cases explicitly.

### 5A. Existing agent code → inspect first
- **Inspect the code** to detect the **framework** (raw boto3, LangChain, Strands, LlamaIndex, CrewAI, a workflow graph, …) and **enumerate every model-invocation site**.
- **Map the topology:** a single call site, a class/module per agent, or nodes in a workflow — this is how many places you wire.
- For each site, note which variables hold the **input** and **output** text.
- **Adapt to the existing structure; do not refactor the agent.** If the structure makes clean placement hard (model calls scattered, no clear input/output boundary), **flag it and propose the minimal change** — don't restructure the agent yourself.
- Confirm your findings with the developer before editing.

### 5B. New / greenfield (no model-invocation site yet)
There is nothing to wire into yet — so **do not build the agent and do not scaffold a standalone agent, entrypoint, or wrapper module to fill the gap.** Instead:
- **Confirm the plan** with the developer / coding agent: intended **framework** and **topology** (single-agent / one-class-per-agent / workflow). If they're undecided, that's their design call — don't decide for them.
- **Hand the build back.** The developer / coding agent builds the agent in their own enterprise structure and conventions. State plainly: *"Build your agent as you normally would. Once the model call exists, I'll add the OneTrust guardrail checks there."*
- **Capture** `{name, agentId, region, framework?, topology?}` so wiring is precise later.
- Optionally provide the **guardrail check snippet(s)** (input + output) for the developer to place at their call site — clearly marked as snippets to insert into *their* code, **not a runnable agent and not a required wrapper**.

**Optional injection (either mode):** a boto3 `Session`/Bedrock client only if the team needs a custom endpoint or specific credentials — otherwise the SDK builds its own.

Don't ask what the agent does, how it's prompted, or how it's orchestrated — only the framework, topology, and model-invocation location needed to place checks.

## Step 6 — Add the dependency and OneTrust config
1. **Dependency:** add `onetrust-ai-guard-sdk[bedrock]` to requirements/pyproject (imports as `ai_guard`).
2. **Check for existing OneTrust config.** Look for a OneTrust **base URL** and **API key** in the app's config/secrets (env vars, config file, secret manager):
   - **If present** → reuse them; don't duplicate or rename.
   - **If not present (first-time integration)** → add config keys (for example `ONETRUST_BASE_URL`, `ONETRUST_API_KEY`) read from env/secrets — **never hardcode a value** — and **tell the developer to set them to their real OneTrust base URL and API key**, and where to set them. Note clearly that guardrail resolution won't work until they're set.
3. **Region:** use the region from Step 4 (from config).

## Step 7 — Wire the guardrail checks

**Discover the SDK's real surface first — this skill is not the API catalog.** The skill can't list every init/wiring form your installed SDK version supports, and they change over time. Before writing, learn the actual surface from the **installed** package and its README, e.g.:
```bash
python -c "import ai_guard, inspect; \
print(inspect.signature(ai_guard.AIGuardClient.__init__)); \
print([n for n in dir(ai_guard.AIGuardClient) if not n.startswith('_')])"
```
(and `help(ai_guard.AIGuardClient)` / the SDK README for the documented forms — constructor variants, the `activate()` context manager, the `@agent()` decorator, single- vs multi-agent, any config/env loader). **Use the forms you find there; the snippets below are orientation, not the full menu.**

### 7a. Configure the SDK once
**Use the configuration form the installed SDK documents — don't assume the shape below is the only one.** The SDK supports more than one way to initialize (for example: single-agent vs. a multi-agent registry; agents as objects vs. mapping/config entries; and any config-file or environment-based initialization the SDK provides). **Check the SDK's README, pick the form that fits the app, and match the developer's existing config conventions.** Always source `base_url`/`token`/`region` from config/secrets (Step 6), never hardcoded.

One illustrative form (multi-agent registry):
```python
from ai_guard import AIGuardClient, AgentConfig

guard = AIGuardClient(
    base_url=config.ONETRUST_BASE_URL,   # from config/secret (Step 6) — never hardcoded
    token=config.ONETRUST_API_KEY,       # from config/secret (Step 6)
    region=config.AWS_REGION,            # from Step 4 — from environment config
    agents=[                              # one entry per agent from Step 2 / Step 3
        AgentConfig(name="<local-name>", agent_id="<ONETRUST_AGENT_ID>"),
        # ... one per agent ...
    ],
    # The SDK builds its own Bedrock client and resolves guardrails from OneTrust internally.
    # Failure behavior defaults to fail-closed (block); see the SDK config reference to tune.
)
```
For a **single-agent** app, the SDK typically takes the agent id directly instead of a registry (e.g. `agent_id="<ONETRUST_AGENT_ID>"`) — confirm the exact parameter in the README. Keep the SDK's defaults unless the developer asks to change them; point them to the SDK config reference rather than guessing values.

### 7b. Wrap each agent's model call — both checks, scoped to that agent
Insert the checks **inline at the agent's real model-invocation site, following the repo's own structure and conventions** — do not create a separate wrapper function, a helper module, or an example agent unless the codebase already uses that pattern. Use `with guard.activate("<name>"):` — or the equivalent `@guard.agent("<name>")` decorator on the agent's function, if the SDK exposes it (confirm via the discovery step above) — so the checks resolve to that agent's guardrails. A check can **block** or **mask** — always forward `res.text` (the safe/masked text), never the original:
```python
with guard.activate("<local-name>"):
    res = guard.check_input(input_text)
    if res.blocked:
        ...                      # stop / return your blocked response (res.action_reason explains why)
        return
    safe_input = res.text        # may be masked — forward this

    output_text = <existing model call>(safe_input)   # the agent's own model invocation (developer's code)

    res = guard.check_output(output_text)
    if res.blocked:
        ...
        return
    safe_output = res.text       # masked if a policy fired; return/emit this
```
- **Blocked vs. masked:** `res.blocked` is true only for a real block. Masking (e.g. PII redaction) is not a block — `blocked=False` with the safe text in `res.text`.
- **Repeat for every model-invoking agent**, each scoped to the right agent — inside its own `activate(name)` in multi-agent setups, or simply under the client's `agent_id` in single-agent mode.
- **Placement:** raw boto3 → `check_input` before `invoke_model`/`converse`, `check_output` after. Framework → use its pre/post-model hook or wrap the node's model call.
- **Cross-thread workflows:** carry the active agent with `capture_active_agent` / `bind_captured_agent`.
- **Streaming output:** if the agent streams, use the SDK's streaming output guard; otherwise `check_output`. (Already-streamed chunks can't be recalled — a late block only stops the remaining output.)

**Failure behavior to explain to the developer:** if no guardrails can be resolved for an agent, the default is fail-closed (turns are blocked until they resolve); set the failure policy to allow if the team prefers availability. Once resolved, guardrails are cached, so a brief OneTrust outage keeps enforcing the last known guardrails rather than dropping protection.

## Step 8 — Verify
- [ ] A **project was selected first** (Step 1) and **explicitly confirmed by the developer** — not auto-selected — with projectId + orgId recorded before any agent step.
- [ ] Every **created** agent is **linked to that project** — created with the Step 3 `links` entry (`targetEntityType: "projects"`) and the create call succeeded.
- [ ] Created agents set **only** `description` and `autonomyLevel` (internal fieldNames), with an autonomy label returned by `get_ai_governance_attributes` — no other attributes, no invented keys or values.
- [ ] Created names and descriptions are plain text — no HTML entities (`&amp;`, `&lt;`, `&#39;`, …) or URL encoding.
- [ ] Every agentId is a **real value from `search_ai_governance_aiagents` or `create_ai_governance_aiagents`** — none invented.
- [ ] **Region** is set from config and confirmed.
- [ ] **Every** model-invoking agent has **both** `check_input` and `check_output`, each scoped to the correct agent — inside its own `activate(name)` (multi-agent) or under the client's `agent_id` (single-agent).
- [ ] `onetrust-ai-guard-sdk[bedrock]` is a dependency.
- [ ] OneTrust `base_url` + `token` are wired from config/secrets (and, if newly added, the developer has been told to set the real values).

## Step 9 — Show the policies applied to each agent
So the developer knows what governance their agents are under, list each wired agent's OneTrust policies. Read-only — this never changes anything.
1. For each agent from Steps 2–3, call **`get_ai_governance_policies(entityTypeName="aiagents", entityId="<agentId>")`**. It returns `{totalCount, policies: [{policyName, policyDescription, evaluationStatus, …}]}`, 20 per call by default — if `totalCount` is larger, pass a bigger `pageSize` or note that more exist.
2. **Format the results as one markdown table** covering every agent — one row per policy. It's shown once, under the **Policies applied** heading of the Step 10 report. Never put policy results in prose, numbered lists, or bullets mixed with other information.

   | Agent | Policy | Status | Description |
   |---|---|---|---|
   | `<local-name>` | `<policyName>` | `<evaluationStatus>` | `<policyDescription>` |

   - **Status** is `evaluationStatus` exactly as returned (e.g. `COMPLIANT`, `UNRESOLVED`, `SUPPRESSED`).
   - **Description** is `policyDescription`; show `—` when it's empty. Keep each cell on one line (replace line breaks with spaces).
   - Group rows by agent. Within each agent, list `UNRESOLVED` first, then the other statuses.
   - Above the table, add one plain-text line (no italics) per agent with its counts, e.g. Customer Support: 3 policies (1 COMPLIANT, 2 UNRESOLVED).
   - An agent with **no policies** (`totalCount` 0 — typical for a newly created agent) gets a single row: `<local-name>` | — | — | No OneTrust policies applied yet.
3. **Report only.** Don't diagnose or try to resolve `UNRESOLVED` policies — don't set attributes, call other tools, or add explanations beyond the policy's own description. If the developer wants to act on them, point them to the agent in OneTrust.
4. If the policy list can't be retrieved, don't fail the integration — show the matching message in place of the table and continue to Step 10:
   - **Authorization error** (the tool needs a policy-view permission that's separate from the other tools): "Access required to view policies. Contact a site admin for more information."
   - **Any other cause**: "We couldn't load the policy list due to `<reason>`. Try again later."

## Step 10 — Tell the developer what the integration added
Finish by listing everything the integration added or now requires, with real values filled in. Use the headings below, in this order — keep it scannable, with no narrative paragraphs:

**Dependency**
- `onetrust-ai-guard-sdk[bedrock]` (imports as `ai_guard`).

**Config the app now reads** (mark each new vs. reused; call out anything the developer must set)
- <"`ONETRUST_BASE_URL`: new. Set it to your OneTrust base URL." | "`ONETRUST_BASE_URL`: reused from existing config.">
- <"`ONETRUST_API_KEY`: new. Set it to your OneTrust API key." | "`ONETRUST_API_KEY`: reused from existing config.">
- `AWS_REGION`: `<region>` (from config).

**OneTrust project**
- `<projectName>` (`<projectId>`) — the project all created agents are linked to.

**OneTrust agents wired**
- <"`<local-name>` (agent ID `<agentId>`): newly created in `<projectName>`. Guardrails may not be configured yet." | "`<local-name>` (agent ID `<agentId>`): existing.">

**Policies applied** — the Step 9 counts line(s) and table, exactly as specified there:

`<local-name>`: `<n>` policies (`<counts by status>`).

| Agent | Policy | Status | Description |
|---|---|---|---|
| `<local-name>` | `<policyName>` | `<evaluationStatus>` | `<policyDescription or —>` |

<or, if the list couldn't be retrieved, the matching Step 9 message: "Access required to view policies. Contact a site admin for more information." | "We couldn't load the policy list due to `<reason>`. Try again later.">

**Code added**
- One `AIGuardClient`, configured with the base URL, API key, and region.
- `check_input` + `check_output` around the model call for each agent listed — placed for **`<framework>`** (or, if the call site doesn't exist yet, a labeled placeholder to finalize once it does).

**Action required** (include only the items that apply; omit this heading if none do)
- Only if `ONETRUST_BASE_URL` or `ONETRUST_API_KEY` was newly added: "Set `ONETRUST_BASE_URL` and `ONETRUST_API_KEY` to your real values before you run the agent. Until you do, guardrails can't be resolved, and by default the agent blocks every request."
- Only if an agent was newly created: "Configure guardrails for each newly created agent in OneTrust. Enforcement starts once they're set up."

Then hand back — this skill owns only the OneTrust guardrail integration; the rest of the agent's development is the developer's.

## Reference
- **Source of truth for SDK usage** — the installed OneTrust AI Guard SDK's **README / usage reference** (and runtime introspection, Step 7). It, not this skill, defines the supported initialization and wiring forms; prefer it whenever they differ.
- OneTrust AI Guard SDK primitives (as documented there): `AIGuardClient`, `AgentConfig`, `activate` / `@agent`, `check_input` / `check_output`.
- Amazon Bedrock Guardrails `ApplyGuardrail` API: https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-use-independent-api.html
