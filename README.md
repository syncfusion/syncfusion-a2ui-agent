# syncfusion-a2ui-agent

> A thin, platform-agnostic orchestration ADK for building **Syncfusion A2UI agents**, built on top of the open-source [A2A Python SDK](https://github.com/a2aproject/a2a-python) (the official Python implementation of the [Agent2Agent (A2A) protocol](https://a2a-protocol.org/)).

The ADK ships **zero hardcoded component validation** and **zero built-in data-source connectors**. UI knowledge is routed through **Syncfusion Platform Skills and MCPs**; business data lives behind **customer MCPs**. Everything goes through MCPs.

> **Why you need a Skill or Platform MCP.** A generic LLM does not know the Syncfusion component API by default. Without verified UI knowledge, the model can hallucinate component names, wrong prop values, missing imports, or the wrong setup pattern — and the resulting A2UI envelope fails validation or renders as broken UI. A **Skill** or **Platform MCP** gives the model the verified component grammar (exact props, lifecycle, imports, code patterns) so the first response is correct. You only need **one of the two** — pick whichever fits your workflow:
>
> 1. **Syncfusion Agent Skills** (preferred — lighter weight, no server process) — [Skills](https://www.syncfusion.com/explore/agent-skills/). Download the relevant skill, drop it into your skills directory, and call `agent.enable_skills(...)`.
> 2. **Syncfusion AI Coding Assistants (MCPs)** — [MCPs](https://www.syncfusion.com/explore/ai-coding-assistants/). Register with `agent.add_mcp(..., kind="platform")`.
>
> Both channels deliver the same verified Syncfusion component grammar. The bundled starter catalog in the default system prompt is a fallback, not a substitute — for accurate results, point the agent at official Syncfusion resources.

The ADK's default system prompt includes a **starter catalog** of common Syncfusion component names and prop conventions so that the agent produces useful outputs on first use, even without a Platform MCP registered. This starter catalog is purely a prompt-level default — it imposes **no enforcement** at the ADK layer, and a more current or more specialised catalog from a Platform MCP automatically overrides it via `set_catalog_reference()` / `set_catalog_reference_from_file()`.

---

## Table of contents

- [Why this ADK](#why-this-adk)
- [How it works](#how-it-works)
- [Platform MCPs](#platform-mcps)
- [Skills](#skills)
- [Build your own A2UI agent](#build-your-own-a2ui-agent)
- [Design contract (`set_design`)](#design-contract-set_design)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Getting started](#getting-started)
- [Examples](#examples)
- [FAQ](#faq)

---

## Why this ADK

Building a chat-driven UI agent usually means hard-coding component names, props, and rendering rules for a specific platform into your system prompt. That breaks the moment you add a new platform, a new component, or a new AI provider.

`syncfusion-a2ui-agent` inverts the problem:

| Concern | Where it lives |
| --- | --- |
| UI grammar & component schema | A **Syncfusion Platform MCP** you register at runtime, plus local **Skills** (`SKILL.md` files) you enable at runtime |
| Business data | Customer MCPs you register at runtime |
| AI provider choice | Pluggable registry — Gemini / OpenAI / Azure OpenAI / Claude / Ollama / custom. All providers are supported; pick the one that fits your stack. |
| A2UI envelope validation | The ADK, generic over any catalog |
| Renderer (React, Blazor, …) | Separate package outside this repo |

Zero hardcoded platforms. Zero hardcoded providers. Zero hardcoded data-source connectors. Zero runtime enforcement of component names — only the **default system prompt** has a starter catalog of common Syncfusion components, which a more current Platform MCP catalog automatically supersedes.

---

## How it works

For every user request, the ADK runs the following flow:

1. Loads a fixed Syncfusion system prompt (UI rules).
2. Merges it with your business prompt (`extend_system_prompt()`).
3. Pulls **business data** on demand from customer MCP servers you registered.
4. Pulls **UI knowledge** on demand from whichever Platform MCP(s) you registered (any renderer: React, Blazor, Angular, Vue, MAUI, WPF, WinForms, in-house, …) and from any local **Skills** you enabled (filesystem-discovered `SKILL.md` files).
5. Sends the merged context to your AI provider.
6. Validates the model's output as A2UI JSON, retrying on failure (bounded).
7. Hands the validated JSON to your renderer of choice.

---

## Platform MCPs

A **Platform MCP** is any MCP that teaches the model the component grammar of one renderer (React, Blazor, Angular, Vue, MAUI, WPF, WinForms, in-house, or third-party). The ADK ships **no static list** of Platform MCPs — you bring your own by calling `agent.add_mcp(name=..., command=..., kind="platform")`.

> **Browse the official Syncfusion MCPs.** The Syncfusion-published AI Coding Assistants (each a Platform MCP) are listed at [MCPs](https://www.syncfusion.com/explore/ai-coding-assistants/). Pick the one for your renderer, follow its install page, then register it via `agent.add_mcp(..., kind="platform")` as shown below.

The `kind="platform"` argument is a **tag only**; it is never enforced. Every MCP is treated the same way at runtime: tools are discovered via the standard MCP `tools/list` and dispatched via `tools/call`. The tag exists so the orchestrator and prompt builder can recognise UI-knowledge servers when they need to.

| Renderer | Published by | Package / command |
| --- | --- | --- |
| React | Syncfusion | `["npx", "-y", "@syncfusion/react-mcp@latest"]` |
| Anything else | You / a vendor | Whatever the renderer ships |

The bundled [`examples/generic_demo_agent.py`](examples/generic_demo_agent.py) and [`examples/flight_booking_agent.py`](examples/flight_booking_agent.py) register the Syncfusion React Platform MCP behind a `SYNCFUSION_REACT_MCP_AVAILABLE=1` env flag — copy that pattern and swap in your renderer's command.

```python
# Register any Platform MCP
agent.add_mcp(
    name="react-platform-mcp",
    command=["npx", "-y", "@syncfusion/react-mcp@latest"],
    kind="platform",
    env={
        "Syncfusion_API_Key": "Your Syncfusion API Key",
        # "MY_MCP_TOKEN": "...",
        # "MY_MCP_REGION": "us-east-1",
        # "MY_MCP_LOG_LEVEL": "info",
    },
)
agent.warm_catalog()  # spawn-once, reuse-on-every-request
```

---

## Skills

**Skills** are local `SKILL.md` documents that teach the model exactly how to render specific components (exact API props, code patterns, rules to follow). They are filesystem-discovered, vendored with your agent, and consulted on-demand by the model via a synthetic `read_skill(name=...)` tool.

> **Browse the official Syncfusion Skills.** The Syncfusion-published Agent Skills are listed at [skills](https://www.syncfusion.com/explore/agent-skills/). Download the skill that matches your renderer / framework, drop the resulting `SKILL.md` into your skills directory, then call `agent.enable_skills(...)`.

A skill directory looks like this:

```
my-skills/
├── forms-inputs/
│   └── SKILL.md       # --- front-matter: name, description ---
├── data-grid/
│   └── SKILL.md
└── charts/
    └── SKILL.md
```

Each `SKILL.md` starts with a small YAML front-matter (parsed by stdlib `re` — no PyYAML dependency) and contains a markdown body with the component knowledge:

```markdown
---
name: data-grid
description: "Syncfusion EJ2 Grid — sortable, filterable, pageable data table."
---

# EJ2 Grid

Use `type: "Grid"` in your A2UI envelope. The required props are:
- `dataSource` — list of records
- `columns` — list of `{ field, headerText }`
...
```

You enable skills with one call:

```python
# Point the agent at a directory of skill sub-folders, each with a SKILL.md
agent.enable_skills("path/to/my-skills")
```

At the next request, the agent:

1. Injects the **list of available skill descriptions** into the system prompt (so the model knows what skills exist).
2. Adds a synthetic `read_skill(name=...)` tool to the tool list.
3. Lets the model call `read_skill(name="data-grid")` whenever it needs the full knowledge document.
4. The agent reads the file and returns its content to the model — all in one AI session.

| Question | Skills | Platform MCPs |
| --- | --- | --- |
| Where do they live? | Vendored with your agent (filesystem) | Spawned at runtime (separate process, stdio or HTTP) |
| Best for | Bundling a team's component knowledge without a server | A live, queryable component catalog and full tool surface |
| Can be used without the other? | Yes — skills work on their own | Yes — Platform MCPs work on their own |

The `examples/generic_demo_agent.py` and `examples/flight_booking_agent.py` agents load skills from `SKILLS_PATH` (defaults to `./skills/` next to the script). Drop your own `SKILL.md` files there to extend them.

```python
# Enable a custom skills directory
agent.enable_skills("path/to/my-skills")
```

---

## Build your own A2UI agent

The whole ADK exists so you can write a working A2UI agent in a few lines. Here is a **complete, runnable application** — the smallest meaningful one. Drop it into any new project.

### `my_agent.py`

```python
"""
Minimal A2UI agent — single file, ~25 lines, no extras.

Run:
    pip install "git+https://github.com/syncfusion/syncfusion-a2ui-agent.git"
    set AZURE_API_KEY=...
    set AZURE_API_BASE=https://<your-resource>.openai.azure.com/
    python my_agent.py "show me a sales dashboard"
"""
import asyncio
import os
import sys

from syncfusion_a2ui_agent import SyncfusionAgent


class MyAgent(SyncfusionAgent):
    """Your business prompt goes here."""

    def extend_system_prompt(self) -> str:
        return (
            "You are a UI assistant for an internal HR dashboard. "
            "Generate production-quality Syncfusion components only."
        )


async def main(prompt: str) -> None:
    agent = MyAgent(
        model="azure_openai",
        api_key=os.environ["AZURE_API_KEY"],
        azure_endpoint=os.environ["AZURE_API_BASE"],
        azure_deployment="gpt-4o",  # or whatever your Azure deployment is named
        api_version="2025-01-01-preview",
    )
    envelope = await agent.handle(prompt)   # -> dict, A2UI v0.9
    print(envelope)                          # ship to your renderer


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "show me a dashboard"))
```

### What that gives you

- **Zero Syncfusion knowledge in your code** — the model knows the A2UI grammar; the ADK enforces it via validation.
- **Pluggable provider** — swap `"azure_openai"` for `"gemini"`, `"openai"`, `"claude"`, `"ollama"`, or a custom subclass.
- **Validated output** — `agent.handle_request()` retries up to N times until the response is a valid A2UI envelope.
- **Production-ready** — when you're ready to deploy, call `agent.serve(host=..., port=...)` instead of `handle_request()` to expose an A2A HTTP server.

> **Provider choice.** The ADK works with all six built-in providers (Azure OpenAI, OpenAI, Gemini, Claude, Ollama, and `CustomProvider` for DeepSeek and others) — all are available in this beta release and are exercised by the test suite. There is no single "recommended" provider. In practice, GPT-class models (Azure OpenAI / OpenAI) tend to produce the fewest validation retries on real prompts — especially for nested data, schemas, and tool-calling — but the ADK is designed to work well with any of them.

### Going further

| Need | Add |
| --- | --- |
| Your own business data | `agent.set_data_source_from_file("data.json")` or `agent.add_mcp(name="crm", command=[...])` |
| An A2UI design authored by an A2UI Composer tool | `agent.set_design("designs/dashboard.json")` or `agent.set_design("designs/")` (one method, any input shape — see below) |
| UI knowledge for any renderer (React, Blazor, Angular, Vue, MAUI, WPF, WinForms, in-house, …) | `agent.add_mcp(name="my-renderer", command=[...], kind="platform")` — see the [Platform MCPs](#platform-mcps) section |
| Local skills / routing rules | `agent.enable_skills("path/to/skills")` |
| Multiple tools the LLM can call | `agent.add_mcp(name="...", url="https://...")` |

### Design contract (`set_design`)

When you have a pre-authored A2UI envelope (e.g. copied from an A2UI Composer) and want the agent to echo that structure verbatim — only filling in data values from the user's prompt — call **`agent.set_design(...)`**. One method accepts every reasonable input shape:

```python
agent.set_design(design_dict)                       # one design (parsed dict / list)
agent.set_design('{"version":"v0.9", ...}')         # one design (JSON string)
agent.set_design("designs/finance.json")            # one design (file path)
agent.set_design("designs/")                        # every *.json in a directory → multi-page catalog
agent.set_design([                                  # list of any of the above
    "designs/dashboard.json",
    {"version": "v0.9", "createSurface": {...}},
])
```

When you pass a directory or a list, the agent concatenates every design into a single markdown catalog under a **"UI Surface Design Contract"** template so the LLM can see every page at once and pick the one that matches the user's intent. The agent is told to:

1. Echo the chosen design's structure verbatim (same component ids, same children, same dataModel paths, same event names).
2. Vary only the **data values** derived from the user's prompt.
3. Generate a **new unique** `surfaceId` on every response, since the renderer's `MessageProcessor` rejects reused ids with *"Surface already exists"*.

---

## Prerequisites

- **Python** ≥ 3.10
- **pip** ≥ 21 (for the `pyproject.toml` build system)
- A valid **AI provider** API key, e.g.:
  - **Azure OpenAI** (`AZURE_API_KEY` + `AZURE_API_BASE` + deployment)
  - **OpenAI** (`OPENAI_API_KEY`)
  - Google Gemini (`GOOGLE_API_KEY`)
  - Anthropic Claude (`ANTHROPIC_API_KEY`)
  - Ollama (local — no key, just a running server)
  - DeepSeek and any other OpenAI-compatible endpoint — via a `CustomProvider` subclass (see [Adding a custom provider](#adding-a-custom-provider-deepseek-in-house-openai-compatible))

  > **Which provider should I pick?** All six built-in providers (and any `CustomProvider` you register) are available in this beta release and are exercised by the test suite. Pick the one that fits your stack — there is no single "recommended" provider. In practice, GPT-class models (Azure OpenAI / OpenAI) tend to produce the fewest validation retries on real prompts, but the ADK is designed to work well with any of them. Choose based on your constraints: latency, cost, data-residency, an in-house model, etc.
- *(Optional)* **Node.js ≥ 18** and `npx` on `PATH` if you want to register a Platform MCP that ships as an `npx` package (Syncfusion publishes `@syncfusion/react-mcp@latest` for the React renderer; other vendors publish their own).
- *(Optional)* A **Syncfusion account API key** ([get one here](https://www.syncfusion.com/account/api-key)) passed as the `Syncfusion_API_Key` env var, only when consuming a Syncfusion-published Platform MCP. This is **not a product license** — it's the per-account key Syncfusion issues from your account dashboard. Third-party or in-house Platform MCPs do not require it.

> The ADK is renderer-agnostic and ships no hardcoded list of Platform MCPs or Skills — see [Platform MCPs](#platform-mcps) and [Skills](#skills).

---

## Installation

Install the **ADK directly from GitHub** in your own project:

```bash
pip install "git+https://github.com/syncfusion/syncfusion-a2ui-agent.git"
```

Install the **dev extras** (pytest, ruff) — only needed if you're running the test suite:

```bash
pip install "git+https://github.com/syncfusion/syncfusion-a2ui-agent.git#egg=syncfusion-a2ui-agent[dev]"
```

Install the **ADK from a local source checkout** (when contributing to the ADK itself):

```bash
git clone https://github.com/syncfusion/syncfusion-a2ui-agent.git
cd syncfusion-a2ui-agent
pip install -e ".[dev]"
```

The ADK uses the `src/` layout, so after `pip install -e .` the importable package lives at `src/syncfusion_a2ui_agent/`. The example package, by contrast, is installed separately (see [Examples](#examples)).

---

## Configuration

The ADK is a **library**, not an app — it does not ship a `.env` template of its own. You pass provider credentials programmatically (see [Build your own A2UI agent](#build-your-own-a2ui-agent)).

The **examples** do ship an env template; see [Examples](#examples) below.

| Variable | Used by | When |
| --- | --- | --- |
| `AZURE_API_KEY` | `AzureOpenAIProvider` | When using Azure |
| `AZURE_API_BASE` | `AzureOpenAIProvider` | When using Azure |
| `AZURE_API_VERSION` | `AzureOpenAIProvider` | Defaults to `2025-01-01-preview` |
| `GOOGLE_API_KEY` | `GeminiProvider` | When using Gemini |
| `OPENAI_API_KEY` | `OpenAIProvider` | When using OpenAI |
| `ANTHROPIC_API_KEY` | `ClaudeProvider` | When using Claude |
| `Syncfusion_API_Key` | Syncfusion Platform MCPs (e.g. `@syncfusion/react-mcp@latest`) | Only when consuming a Syncfusion-published Platform MCP. Get the value from your [Syncfusion account API keys page](https://www.syncfusion.com/account/api-key) — this is an account-level API key, **not** a product license. |
| `SKILLS_PATH` | Skill router (in the example agents) | When using local skills |

---

## Getting started

### 1. Minimal agent (Gemini + any Platform MCP)

```python
# pip install "git+https://github.com/syncfusion/syncfusion-a2ui-agent.git"
from syncfusion_a2ui_agent import SyncfusionAgent

agent = SyncfusionAgent(
    model="gemini",                # gemini | openai | azure_openai | claude | ollama | custom subclass
    api_key="YOUR_GOOGLE_API_KEY",
)

# Register a Platform MCP for the renderer you target. The ADK does
# not ship a list of "known" platforms — bring your own. Below is the
# official Syncfusion React Platform MCP; replace `name` and `command`
# with whatever your renderer publishes (Blazor, Angular, Vue, MAUI,
# WPF, WinForms, in-house, …). See the "Platform MCPs" section below.
agent.add_mcp(
    name="my-renderer-mcp",        # any name; used to route tool calls
    command=["npx", "-y", "@your-org/your-renderer-mcp"],
    kind="platform",               # tag only — never enforced
    env={
        "Syncfusion_API_Key": "Your Syncfusion API Key",
        # "MY_MCP_TOKEN": "...",
        # "MY_MCP_REGION": "us-east-1",
        # "MY_MCP_LOG_LEVEL": "info",
    },
)

# Warm the MCP tool catalog at boot so the first request doesn't pay a spawn cost
agent.warm_catalog()

# Subclass to inject your own business prompt
class MyAgent(SyncfusionAgent):
    def extend_system_prompt(self) -> str:
        return "You are an enterprise UI assistant for an HR dashboard."

# Start the A2A server
my_agent = MyAgent(model="gemini", api_key="YOUR_GOOGLE_API_KEY")
my_agent.serve(host="0.0.0.0", port=8080)
```

### 2. Register business data sources (all via MCP)

```python
# Customer MCP — the only way to expose business data
agent.add_mcp(name="CRM",          command=["python", "crm_mcp.py"])
agent.add_mcp(name="OrdersMCP",    command=["node", "orders_server.js"])
agent.add_mcp(name="InventoryMCP", url="https://mcp.company.com/inventory")

# Or inject a static data set directly (e.g. for demos / tests)
agent.set_data_source_from_file("path/to/your/data.json")
```

---

## Examples

The [`examples/`](examples/) directory is a **fully independent package** called `syncfusion-a2ui-examples`. It has its own `pyproject.toml`, its own `.env.example`.

| File | What it shows |
| --- | --- |
| [`generic_demo_agent.py`](examples/generic_demo_agent.py) | Contoso Dynamics enterprise agent on Azure OpenAI, with `set_data_source_from_file()` and the Syncfusion React Platform MCP. |
| [`flight_booking_agent.py`](examples/flight_booking_agent.py) | SkyWave Airlines two-stage booking workflow (search → form → summary). |
| [`demo_examples.json`](examples/demo_examples.json) | Sample data set (KPIs, sales reps, regional sales, inventory, calendar events) used by `generic_demo_agent.py`. |
| [`react_catalog.json`](examples/react_catalog.json) | A2UI v0.9 catalog reference for the Syncfusion React renderer (used by `set_catalog_reference_from_file()`). |
| [`send_message.json`](examples/send_message.json) | JSON-RPC `message/send` payload you can POST to the A2A server. |
| [`test-page.html`](examples/test-page.html) | A static page to drive the A2A server from a browser. |

### Run an example

```bash
cd examples
python -m venv .venv
.venv\Scripts\activate          # or: source .venv/bin/activate
pip install -e .                # installs the examples package
cp .env.example .env            # fill in your keys — .env is git-ignored
python generic_demo_agent.py "show a sales dashboard"   # single-shot
python flight_booking_agent.py --serve                   # A2A server on :10005
```
---

## Adding a custom provider (DeepSeek, in-house, OpenAI-compatible)

Any provider that speaks the OpenAI HTTP API — **DeepSeek**, Together, Anyscale, your own in-house endpoint, etc. — plugs in via `CustomProvider`. You only need to implement `generate()`; the ADK owns retry, timeout, validation, and streaming.

```python
# pip install "git+https://github.com/syncfusion/syncfusion-a2ui-agent.git"
from syncfusion_a2ui_agent import SyncfusionAgent
from syncfusion_a2ui_agent.providers import CustomProvider, register_provider


class DeepSeekProvider(CustomProvider):
    name = "deepseek"

    async def generate(self, messages, tools=None, context=None):
        # Hit https://api.deepseek.com/v1/chat/completions (OpenAI-compatible)
        # and return a ModelResponse with `text` and `tool_calls`.
        ...


# Register once, anywhere before constructing the agent.
register_provider("deepseek", DeepSeekProvider)

agent = SyncfusionAgent(model="deepseek", api_key="YOUR_DEEPSEEK_API_KEY")
```

`CustomProvider` is the same mechanism customers use to bring their own private LLM. The ADK treats it as a first-class provider — same validation, same retry budget, same streaming.

> Any OpenAI-compatible provider (DeepSeek, Together, Anyscale, an in-house endpoint, etc.) plugs in the same way. Pick the provider that fits your constraints — cost, data-residency, an in-house model, etc. In practice, GPT-class models (Azure OpenAI / OpenAI) tend to produce the fewest validation retries on real prompts, but the ADK is designed to work well with any of them.

---

## FAQ

### Q1. Is this ADK published on PyPI?
No — for now it is installed directly from this GitHub repository (`pip install "git+https://github.com/syncfusion/syncfusion-a2ui-agent.git"`). A PyPI release will follow once the public API stabilises.

### Q2. Which AI provider should I use?
There is no single "recommended" provider — pick the one that fits your stack. All six built-in providers (Azure OpenAI, OpenAI, Gemini, Claude, Ollama, and `CustomProvider` for DeepSeek and in-house models) are available in this beta release and are exercised by the test suite. In practice, GPT-class models (Azure OpenAI / OpenAI) tend to produce the fewest validation retries on real prompts, but the ADK is designed to work well with any of them. Choose based on your constraints: cost, data-residency, an in-house model, etc.

### Q3. Do I need a Syncfusion license to use this ADK?
Not for the ADK itself — it is renderer-agnostic. You only need a **Syncfusion account API key** (passed as the `Syncfusion_API_Key` env var) when you consume a **Syncfusion-published** Platform MCP (e.g. `@syncfusion/react-mcp@latest`). Get the key from [Syncfusion account API key](https://www.syncfusion.com/account/api-key). This is a per-account API key from your Syncfusion account dashboard — **it is not a product license**. Third-party or in-house Platform MCPs do not require a Syncfusion account or API key.  

### Q4. Do I need a Platform MCP?
A Platform MCP gives you a more current, more specialised catalog and a full tool surface for the renderer you target. Without one, the agent falls back to the bundled starter catalog. **Skills** are the other channel: local `SKILL.md` files you vendored with your agent give the model exact component knowledge without needing a server. See the [Skills](#skills) section.

### Q5. Do I really need a Syncfusion Skill or Platform MCP? Can't a generic LLM figure it out?
A generic LLM does not know the Syncfusion component API by default. Without verified UI knowledge, the model can hallucinate component names, prop values, imports, or setup patterns — the resulting A2UI envelope will fail validation or render as broken UI. **Skills** and **Platform MCPs** give the model the verified component grammar (exact props, lifecycle, imports, code patterns) so the first response is correct. The bundled starter catalog in the default system prompt is a fallback for quick experiments, not a substitute for an official Skill or Platform MCP. Browse the official Syncfusion resources at [Skills](https://www.syncfusion.com/explore/agent-skills/) and [MCPs](https://www.syncfusion.com/explore/ai-coding-assistants/).

### Q6. Why does the agent sometimes retry?
The ADK validates every model response as A2UI JSON (jsonschema + retry). When the model produces a near-miss envelope (a missing field, a slightly-wrong shape), the ADK feeds the error back to the model and asks it to fix it. After a bounded number of attempts the ADK raises the last validation error. Pick a stronger model (e.g. GPT-4o) to reduce retries.

### Q7. Is the ADK renderer-agnostic?
Yes. The ADK produces **A2UI v0.9 envelopes** — a JSON description of a UI. The renderer (React, Blazor, Angular, Vue, MAUI, WPF, WinForms, in-house, …) is a separate concern and lives in its own package. The ADK ships no renderer.

### Q8. How do I run the test suite?
`pip install -e ".[dev]" && pytest` from the repo root. The test suite is hermetic — it uses a `FakeProvider` so no real network calls or API keys are required. See `tests/` for the full list.

### Q9. How do I add a new AI provider (DeepSeek, in-house, OpenAI-compatible)?
Subclass `CustomProvider` from `syncfusion_a2ui_agent.providers`, implement `async def generate(...)` to return a `ModelResponse`, then call `register_provider("your-name", YourProviderClass)`. The ADK treats it as a first-class provider — same validation, retry, and streaming. Full example in [Adding a custom provider](#adding-a-custom-provider-deepseek-in-house-openai-compatible).

### Q10. Where is the source for the Syncfusion React renderer?
Out of this repo. The Syncfusion React Platform MCP (`@syncfusion/react-mcp@latest`) is published on npm and is brought in at runtime via `npx`. The A2UI v0.9 component catalog it exposes lives at `examples/react_catalog.json` in this repo (used by `set_catalog_reference_from_file()`).

### Q11. Can I use this with a non-Syncfusion UI library?
Yes — that is the whole point. Register any third-party or in-house Platform MCP (`add_mcp(..., kind="platform")`) and the agent will use whatever component grammar that MCP exposes. The default starter catalog only takes over when no Platform MCP is registered.

### Q12. How do I add a local Skill?
Create a sub-folder under your skills directory (e.g. `my-skills/data-grid/`) and put a `SKILL.md` inside it. The file starts with a small YAML front-matter — `name:` and `description:` — followed by a markdown body with the component knowledge. Then call `agent.enable_skills("path/to/my-skills")`. The model reads each `SKILL.md` on demand via a synthetic `read_skill(name=...)` tool. Full example in the [Skills](#skills) section.

### Q13. How do I lock the UI structure to a pre-authored A2UI design?
Call `agent.set_design(...)` with one of: a parsed design (`dict` / `list`), a JSON string, a file path, a directory of designs, or a list of any of these. The design is embedded into the system prompt as a **"UI Surface Design Contract"** — the LLM is told to echo the structure verbatim and only vary data values. Pass a directory (or a list) to enable multi-page mode: every design becomes a page in a catalog and the LLM picks the right one per request.

### Q14. Where do I report bugs or request features?
Open an issue at [syncfusion-a2ui-agent-issues](https://github.com/syncfusion/syncfusion-a2ui-agent/issues).

## Support and feedback

* For any other queries, reach our [Syncfusion support team](https://www.syncfusion.com/support/directtrac/incidents/newincident) or post the queries through the [Community forums](https://www.syncfusion.com/forums) and submit a feature request or a bug through our [Feedback portal](https://www.syncfusion.com/feedback).
* To renew the subscription, click [renew](https://www.syncfusion.com/sales/products) or contact our sales team at salessupport@syncfusion.com | Toll Free: 1-888-9 DOTNET.

## About Syncfusion

Founded in 2001 and headquartered in Research Triangle Park, N.C., Syncfusion has more than 29,000 customers and more than 1 million users, including large financial institutions, Fortune 500 companies, and global IT consultancies.

Today we provide 1,800+ controls and frameworks for web ([ASP.NET Core](https://www.syncfusion.com/aspnet-core-ui-controls), [ASP.NET MVC](https://www.syncfusion.com/aspnet-mvc-ui-controls), [ASP.NET WebForms](https://www.syncfusion.com/jquery/aspnet-web-forms-ui-controls), [JavaScript](https://www.syncfusion.com/javascript-ui-controls), [Angular](https://www.syncfusion.com/angular-ui-components), [React](https://www.syncfusion.com/react-ui-components), [Vue](https://www.syncfusion.com/vue-ui-components), and [Blazor](https://www.syncfusion.com/blazor-components), mobile ([Xamarin](https://www.syncfusion.com/xamarin-ui-controls), [Flutter](https://www.syncfusion.com/flutter-widgets), [UWP](https://www.syncfusion.com/uwp-ui-controls), and [JavaScript](https://www.syncfusion.com/javascript-ui-controls)), and desktop development ([WinForms](https://www.syncfusion.com/winforms-ui-controls), [WPF](https://www.syncfusion.com/wpf-ui-controls), and [UWP](https://www.syncfusion.com/uwp-ui-controls) and [WinUI](https://www.syncfusion.com/winui-controls))). We provide ready-to deploy enterprise software for dashboards, reports, data integration, and big data processing. Many customers have saved millions in licensing fees by deploying our software.
