"""Syncfusion A2UI enterprise demo agent using Azure OpenAI.

Part of the standalone `syncfusion-a2ui-examples` package. Loads credentials
from `.env` in this script's own directory (NOT from the SDK's repo root)
and starts a Contoso Dynamics enterprise UI agent. Grounded on
demo_examples.json — all data (employees, sales, inventory, calendar events,
etc.) comes from that file.

Run:
    python examples/generic_demo_agent.py            # interactive REPL
    python examples/generic_demo_agent.py "prompt"   # single-shot
    python examples/generic_demo_agent.py --serve    # A2A server on :10004

Provider note:
    This demo uses **Azure OpenAI** by default. The ADK also supports
    OpenAI, Gemini, Claude, Ollama, and `CustomProvider` subclasses
    (e.g. for DeepSeek). For best A2UI envelope quality and the fewest
    validation retries, we recommend **Azure OpenAI** or **OpenAI** —
    GPT-class models produce the cleanest output on real prompts.
    To switch providers, swap the `AzureOpenAIProvider(...)` call
    below for the equivalent provider (see README §"Configuration").
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys

# Best-effort .env loader scoped to THIS directory (the examples package).
# This script does NOT touch the SDK's repo root or its .env.
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except Exception:
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(_env_path):
        with open(_env_path, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line or _line.startswith("#") or "=" not in _line:
                    continue
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

from syncfusion_a2ui_agent import SyncfusionAgent
from syncfusion_a2ui_agent.providers.azure_openai_provider import AzureOpenAIProvider


class ContosoAgent(SyncfusionAgent):
    """Enterprise UI agent for Contoso Dynamics demo data."""

    def extend_system_prompt(self) -> str:
        return (
            "You are an enterprise UI assistant for Contoso Dynamics. "
            "Generate production-quality Syncfusion component layouts for business use cases "
            "such as dashboards, data grids, reports, forms, schedulers, and workflows. "
            "Always use the values from the Application Data Source when populating any data — "
            "employee names, sales figures, inventory records, calendar events, and so on. "
            "Keep layouts clean and professional. "
            "For grids, always include realistic column definitions and populate dataSource "
            "from the data source. "
            "For charts, use the monthlyRevenue or productRevenue data. "
            "For schedulers, use calendarEvents. "
            "For KPI cards, derive values from the provided data rather than inventing numbers."
        )


def build_agent() -> ContosoAgent:
    """Construct the agent from environment variables."""
    api_key = os.environ.get("AZURE_API_KEY")
    endpoint = os.environ.get("AZURE_API_BASE")
    api_version = os.environ.get("AZURE_API_VERSION", "2025-01-01-preview")
    deployment = os.environ.get("MODEL_NAME", "azure/gpt-5.4")
    # MODEL_NAME may be a litellm-style "azure/<deployment>" — strip the prefix.
    if "/" in deployment:
        deployment = deployment.split("/", 1)[1]

    if not (api_key and endpoint):
        raise SystemExit(
            "Missing AZURE_API_KEY / AZURE_API_BASE. Set them in .env or your shell."
        )

    provider = AzureOpenAIProvider(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
        azure_deployment=deployment,
    )

    agent = ContosoAgent(model=provider)

    # Surface agent-level logs (MCP tool catalog, tool calls, retries).
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    # ── Application data source ────────────────────────────────────────────
    # Load demo_examples.json so the LLM uses Contoso Dynamics data (employees,
    # sales reps, regional sales, inventory, calendar events, etc.) for every
    # response instead of fabricating values.  The file lives next to this
    # script; fall back gracefully if it's missing.
    _demo_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_examples.json")
    if os.path.exists(_demo_data):
        agent.set_data_source_from_file(_demo_data)
        logging.getLogger("syncfusion_a2ui_agent").info(
            "Loaded application data source from %s", _demo_data
        )
    else:
        logging.getLogger("syncfusion_a2ui_agent").warning(
            "demo_examples.json not found at %s — agent will generate its own data",
            _demo_data,
        )

    # ── Local skills (higher priority than MCP) ───────────────────────────
    # SKILLS_PATH in .env points to the directory that contains individual
    # skill sub-directories (each with a SKILL.md file).  Skills are always
    # tried FIRST; only if no skill matches does the request fall back to MCP.
    #
    # You can also pass an explicit path: agent.enable_skills("/your/skills")
    skills_path = os.environ.get("SKILLS_PATH", "")
    if skills_path:
        registry = agent.enable_skills(skills_path)
        skills = registry.all_skills()
        log = logging.getLogger("syncfusion_a2ui_agent")
        log.info(
            "Loaded %d skill(s) from %s: %s",
            len(skills),
            skills_path,
            [s.name for s in skills],
        )
    else:
        logging.getLogger("syncfusion_a2ui_agent").warning(
            "SKILLS_PATH not set — skill routing disabled. "
            "Add SKILLS_PATH=<path> to your .env to enable it."
        )

    # ── Syncfusion React Platform MCP (fallback) ──────────────────────────
    # Optionally register the Syncfusion React Platform MCP.
    if os.environ.get("SYNCFUSION_REACT_MCP_AVAILABLE", "0") == "1":
        # Pin the MCP package version so `npx` re-resolves are skipped
        # on every cold start. Override via SYNCFUSION_REACT_MCP_VERSION
        # to upgrade.
        mcp_version = os.environ.get("SYNCFUSION_REACT_MCP_VERSION", "")
        mcp_cmd = ["npx", "-y", "@syncfusion/react-mcp@latest"]
        if mcp_version:
            mcp_cmd.append(f"@{mcp_version}")
        agent.add_mcp(
            name="syncfusion-react",
            command=mcp_cmd,
            kind="platform",
            timeout=60.0,
        )

    return agent


async def single_shot(agent: ContosoAgent, prompt: str) -> None:
    print(f"\n[user] {prompt}\n", flush=True)
    payload = await agent.handle_request(prompt)
    print("[a2ui]", json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


async def repl(agent: ContosoAgent) -> None:
    print("Contoso Dynamics A2UI Agent — interactive mode (type 'exit' to quit).", flush=True)
    while True:
        try:
            prompt = await asyncio.to_thread(input, "\n[you] ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        prompt = prompt.strip()
        if not prompt or prompt.lower() in ("exit", "quit"):
            return
        try:
            await single_shot(agent, prompt)
        except Exception as exc:  # pragma: no cover - REPL UX
            print(f"[error] {exc}", flush=True)


def serve(agent: ContosoAgent, host: str, port: int, warmup: bool = False) -> None:
    """Boot the A2A server (blocking).

    The agent has a warmup path on the server, but it is currently
    **disabled by default** because the mcp SDK's stdio_client raises
    a known cosmetic ``RuntimeError("Attempted to exit cancel scope
    in a different task than it was entered in")`` during teardown
    when an mcp session is left dangling across the warmup request
    and the server's start. The deferred-warmup path inside
    ``agent.handle_request`` still fires the spawn in the background
    on the first real request and the catalog is populated by request
    #2.
    """
    if warmup:
        print(
            "[a2a] --warmup is currently a no-op due to a known mcp "
            "SDK teardown bug; the first request will still pay a "
            "small deferred-warmup tail.",
            flush=True,
        )
    print(f"[a2a] starting server on http://{host}:{port}", flush=True)
    agent.serve(host=host, port=port, warmup=False)


def _print_skills(agent: ContosoAgent) -> None:
    """Print every discovered skill and whether the router is active."""
    router = agent._skill_router
    if router is None:
        print("Skills: DISABLED (SKILLS_PATH not set in .env)")
        return
    skills = router._registry.all_skills()
    if not skills:
        print("Skills: ENABLED but no skills found — check SKILLS_PATH")
        return
    print(f"Skills: ENABLED — {len(skills)} skill(s) discovered\n")
    print(f"{'#':<4} {'Name':<45} Description")
    print("-" * 100)
    for i, s in enumerate(skills, 1):
        desc = s.description[:60] + "…" if len(s.description) > 60 else s.description
        print(f"{i:<4} {s.name:<45} {desc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?", help="Single prompt to send. Omit for REPL.")
    parser.add_argument(
        "--serve", action="store_true", help="Run the A2A server instead of the REPL."
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Pre-warm the MCP catalog before binding the port (recommended with --serve).",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging."
    )
    parser.add_argument(
        "--list-skills", action="store_true", help="List available skills and exit."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10004)
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    agent = build_agent()

    if args.list_skills:
        _print_skills(agent)
        return

    if args.serve:
        serve(agent, args.host, args.port, warmup=args.warmup)
        return

    if args.prompt:
        asyncio.run(single_shot(agent, args.prompt))
    else:
        asyncio.run(repl(agent))


if __name__ == "__main__":
    main()
