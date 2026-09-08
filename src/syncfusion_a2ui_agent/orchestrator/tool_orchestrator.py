"""Tool orchestrator.

Decides which MCP tools are needed for a given user request, executes them,
and merges the results into a single payload the AI provider can consume.

The orchestrator does NOT call the AI provider directly (that's `agent.py`).
It also does NOT validate A2UI output (that's the validator's job).

Selection policy in v1 (per spec §5 step 3):
  - UI knowledge needs  -> all MCP servers tagged kind="platform"
  - Business knowledge  -> all customer MCP servers.
For v1 the orchestrator fetches the full set of UI-knowledge tools and
business-knowledge tools in parallel and merges the results; finer-grained
selection (LLM-driven tool picking) is a follow-up enhancement.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ..mcp.mcp_manager import MCPManager, MCPServerConfig
from ..prompts.prompt_builder import ToolResult


@dataclass
class OrchestratorPlan:
    """The set of tools the orchestrator decided to invoke for a request.

    v1 runs the union of platform MCPs and all business MCPs in parallel.
    """

    mcp_servers: list[MCPServerConfig] = field(default_factory=list)


@dataclass
class OrchestratorResult:
    """Merged, prompt-friendly results ready to feed the AI provider."""

    results: list[ToolResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class ToolOrchestrator:
    """Selects and executes MCP tools for a given user request."""

    DEFAULT_AI_TIMEOUT = 120.0

    def __init__(self, mcp_manager: MCPManager) -> None:
        self.mcp = mcp_manager

    # ------------------------------------------------------------------ planning
    def plan(
        self,
        user_message: str,
        conversation_history: Iterable[Any] | None = None,
    ) -> OrchestratorPlan:
        """v1 plan: pull all UI-knowledge MCPs and (best-effort) business MCPs.

        The orchestrator does NOT inspect the message content to decide which
        tools to call — it surfaces the entire available tool set to the
        provider via the `tools` parameter and lets the model pick. For v1
        we also kick off a refresh of platform-MCP tool catalogs so the
        provider can see what is available.
        """
        return OrchestratorPlan(
            mcp_servers=self.mcp.list_servers(),  # refresh + invoke all in v1
        )

    # ------------------------------------------------------------------ execution
    async def execute(
        self,
        plan: OrchestratorPlan,
        mcp_tool_hints: dict[str, list[str]] | None = None,
    ) -> OrchestratorResult:
        """Run the planned MCP tools in parallel and merge their results.

        `mcp_tool_hints` is an optional map server_name -> [tool_names] to
        invoke on that server. When omitted, the orchestrator refreshes the
        tool catalog and invokes the first tool on each server (best-effort).
        For v1 this keeps the data flow explicit and easy to reason about;
        tighter tool selection can be layered on top later.
        """
        tasks: list[asyncio.Task] = []

        # --- MCP ---
        for cfg in plan.mcp_servers:
            if not cfg.command:
                continue
            server_name = cfg.name
            tools_to_call = (mcp_tool_hints or {}).get(server_name)
            if tools_to_call:
                for tool in tools_to_call:
                    tasks.append(
                        asyncio.create_task(
                            self._safe_mcp_call(server_name, tool, {}),
                            name=f"mcp:{server_name}:{tool}",
                        )
                    )
            else:
                # Try to refresh the catalog and call the first tool.
                tasks.append(
                    asyncio.create_task(
                        self._safe_mcp_refresh_and_first(server_name),
                        name=f"mcp-refresh:{server_name}",
                    )
                )

        results = await asyncio.gather(*tasks, return_exceptions=False)
        merged = OrchestratorResult()
        for r in results:
            if isinstance(r, ToolResult):
                merged.results.append(r)
            elif isinstance(r, list):
                merged.results.extend(r)
            elif isinstance(r, str):
                merged.errors.append(r)
        return merged

    # ------------------------------------------------------------------ helpers
    async def _safe_mcp_call(self, server: str, tool: str, args: dict[str, Any]) -> ToolResult:
        try:
            res = await self.mcp.call_tool(server, tool, args)
            return ToolResult(source=f"mcp:{server}:{tool}", payload=res)
        except Exception as exc:
            return ToolResult(source=f"mcp:{server}:{tool}", payload=f"<error: {exc}>")

    async def _safe_mcp_refresh_and_first(self, server: str) -> list[ToolResult]:
        try:
            tools = await self.mcp.refresh_tools(server)
        except Exception as exc:
            return [ToolResult(source=f"mcp-refresh:{server}", payload=f"<error: {exc}>")]
        if not tools:
            return [ToolResult(source=f"mcp-refresh:{server}", payload="(no tools)")]
        first = tools[0]
        return [await self._safe_mcp_call(server, first["name"], {})]
