"""Generic MCP server lifecycle — works for ANY MCP (platform or customer).

The manager keeps **one long-lived session per server** so subsequent
`tools/list` and `call_tool` requests don't have to spawn the server
process every time. The first refresh pays the cold-start cost; subsequent
refreshes are O(network RTT).

The manager does not know or care whether a server is a Syncfusion Platform
MCP. It simply exposes:
  - register(name, command | url, kind="customer")   # any server
  - remove(name)                                     # unplug at any time
  - list_tools()                                     # cached union, no network
  - call_tool(server, tool, args)                    # delegate to that server
  - refresh_tools(force=False)                       # discovery; cached by default

`kind` is an OPTIONAL tag ("platform" or "customer", default "customer")
used by the orchestrator/prompt builder to recognise UI-knowledge MCPs.
It NEVER restricts which MCPs can be registered.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import uuid
from dataclasses import dataclass, field
from typing import Any

try:  # pragma: no cover - optional SDK dep
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except Exception:  # pragma: no cover
    ClientSession = None  # type: ignore[assignment, misc]
    StdioServerParameters = None  # type: ignore[assignment, misc]
    stdio_client = None  # type: ignore[assignment]


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server.

    Either `command` (stdio transport) or `url` (HTTP/SSE transport) must be
    provided. `kind` is an optional tag used by routing/prompt logic.
    """

    name: str
    kind: str = "customer"  # "platform" | "customer" — tag only, never enforced
    command: list[str] | None = None
    url: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    timeout: float = 30.0


@dataclass
class MCPToolCall:
    """A single tool invocation against a registered MCP server."""

    server: str
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class _ServerState:
    """Cached session + process handle for one MCP server."""

    cfg: MCPServerConfig
    session: Any | None = None  # mcp.ClientSession once alive
    cm: Any | None = None  # stdio_client async context manager
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class MCPManager:
    """Manages registration, discovery, and invocation of MCP servers."""

    def __init__(self) -> None:
        self._servers: dict[str, MCPServerConfig] = {}
        self._tools: dict[str, dict[str, Any]] = {}
        self._state: dict[str, _ServerState] = {}
        self._sessions: dict[str, Any] = {}  # legacy alias for tests
        # Whether a background warmup task has already been kicked off in
        # this manager's lifetime. Used by the agent to decide between
        # "spawn synchronously" and "fire-and-forget warm_start" on the
        # first user-facing request after boot.
        self._warm_started: bool = False
        # Set by ``agent.warm_catalog()`` when called from sync code at boot
        # (no running event loop). The agent will fire the actual warmup
        # the first time it gets a request on a real loop.
        self._deferred_warmup: bool = False

    # ------------------------------------------------------------------ registration
    def add_mcp(
        self,
        name: str,
        command: list[str] | str | None = None,
        url: str | None = None,
        kind: str = "customer",
        env: dict[str, str] | None = None,
        timeout: float = 30.0,
    ) -> MCPServerConfig:
        if not command and not url:
            raise ValueError(f"MCP server '{name}' requires `command` or `url`.")
        if name in self._servers:
            raise ValueError(f"An MCP server named '{name}' is already registered.")
        cmd_list: list[str] | None = None
        if isinstance(command, str):
            cmd_list = shlex.split(command)
        elif isinstance(command, list):
            cmd_list = command
        cfg = MCPServerConfig(
            name=name,
            kind=kind,
            command=cmd_list,
            url=url,
            env=dict(env or {}),
            timeout=timeout,
        )
        self._servers[name] = cfg
        self._state[name] = _ServerState(cfg=cfg)
        return cfg

    def remove_mcp(self, name: str) -> None:
        cfg = self._servers.pop(name, None)
        if cfg is None:
            return
        prefix = f"{name}__"
        for k in [k for k in self._tools if k.startswith(prefix)]:
            self._tools.pop(k, None)
        state = self._state.pop(name, None)
        self._sessions.pop(name, None)
        if state is not None and state.cm is not None:
            cm = state.cm
            # Best-effort teardown.
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():

                    async def _close():
                        try:
                            await cm.__aexit__(None, None, None)
                        except Exception:
                            pass

                    loop.create_task(_close())
            except RuntimeError:
                pass

    # ------------------------------------------------------------------ discovery
    def list_servers(self) -> list[MCPServerConfig]:
        return list(self._servers.values())

    def list_tools(self, server: str | None = None) -> list[dict[str, Any]]:
        """Return cached tool descriptors (no network). Call `refresh_tools()`
        first if you need up-to-date schemas from a newly-registered server.
        """
        if server:
            return [self._tools[k] for k in self._tools if k.startswith(f"{server}__")]
        return list(self._tools.values())

    def has_cached_tools(self, server: str | None = None) -> bool:
        """True if at least one tool descriptor is cached.

        Used to skip a no-op MCP refresh — if we've already learned the
        tool catalog, we can hand the cached descriptors straight to the
        provider without paying the npx round-trip cost.
        """
        if server:
            return any(k.startswith(f"{server}__") for k in self._tools)
        return bool(self._tools)

    def warm_start(self) -> asyncio.Task:
        """Spawn a background task that warms the tool cache without awaiting.

        Returns the asyncio.Task so the caller can optionally ``await`` it
        at startup. Importantly, this does NOT block the calling request:
        per-server spawns happen concurrently and the parent task is the
        background coroutine itself, not the request handler.
        """

        async def _run() -> None:
            try:
                await self.refresh_tools(force=False)
            except Exception:
                pass

        return asyncio.create_task(_run())

    async def refresh_tools(
        self,
        server: str | None = None,
        force: bool = False,
    ) -> list[dict[str, Any]]:
        """Refresh the cached tool catalog from the given server(s).

        By default this is a no-op when the cache for that server is already
        populated (so per-request calls are cheap after the first discovery).
        Pass ``force=True`` to re-list tools from the server regardless.
        """
        if ClientSession is None or stdio_client is None or StdioServerParameters is None:
            raise RuntimeError(
                "The `mcp` Python SDK is not installed. `pip install mcp` to enable "
                "live MCP tool discovery. You can still call `add_mcp` and `call_tool` "
                "in unit tests via a custom transport."
            )
        servers = [self._servers[server]] if server else list(self._servers.values())
        for cfg in servers:
            if not cfg.command:
                continue
            if not force and self._has_cached_tools(cfg.name):
                continue  # cache hit — skip the spawn
            try:
                tools = await asyncio.wait_for(
                    self._list_tools_live(cfg),
                    timeout=cfg.timeout,
                )
            except Exception as exc:
                import logging as _log

                _log.getLogger("syncfusion_a2ui_agent").debug(
                    "MCP refresh_tools failed for %s: %s",
                    cfg.name,
                    exc,
                )
                continue
            prefix = f"{cfg.name}__"
            for k in [k for k in self._tools if k.startswith(prefix)]:
                self._tools.pop(k, None)
            for t in tools:
                fq = f"{cfg.name}__{t['name']}"
                self._tools[fq] = {
                    "name": t["name"],
                    "server": cfg.name,
                    "kind": cfg.kind,
                    "description": t.get("description", ""),
                    "inputSchema": t.get("inputSchema", {}),
                }
        return self.list_tools(server=server)

    def _has_cached_tools(self, server: str) -> bool:
        prefix = f"{server}__"
        return any(k.startswith(prefix) for k in self._tools)

    async def _list_tools_live(self, cfg: MCPServerConfig) -> list[dict[str, Any]]:
        session = await self._ensure_session(cfg)
        resp = await session.list_tools()
        tools = []
        for t in resp.tools:
            tools.append(
                {
                    "name": t.name,
                    "description": getattr(t, "description", "") or "",
                    "inputSchema": getattr(t, "inputSchema", {}) or {},
                }
            )
        return tools

    async def _ensure_session(self, cfg: MCPServerConfig) -> Any:
        """Return a live MCP session for this server, creating one if needed.

        The session and its stdio process are kept alive on the manager
        after the first call, so the next caller reuses it instead of
        paying the spawn cost again.
        """
        if ClientSession is None or stdio_client is None or StdioServerParameters is None:
            raise RuntimeError("The `mcp` Python SDK is not installed. `pip install mcp`.")
        state = self._state.get(cfg.name) or _ServerState(cfg=cfg)
        self._state[cfg.name] = state

        async with state.lock:
            if state.session is not None:
                # Probe the session — if the underlying process died, fall
                # through to a fresh start.
                try:
                    return state.session
                except Exception:
                    state.session = None

            if not cfg.command:
                raise ValueError("cfg.command cannot be empty")
            params = StdioServerParameters(
                command=cfg.command[0],
                args=cfg.command[1:],
                env={**os.environ, **cfg.env},
            )
            cm = stdio_client(params)
            try:
                read, write = await cm.__aenter__()
            except Exception:
                state.cm = None
                raise
            session = ClientSession(read, write)
            try:
                await session.__aenter__()
                await session.initialize()
            except Exception:
                try:
                    await cm.__aexit__(None, None, None)
                except Exception:
                    pass
                state.cm = None
                state.session = None
                raise
            state.cm = cm
            state.session = session
            self._sessions[cfg.name] = session
            return session

    # ------------------------------------------------------------------ invocation
    async def call_tool(
        self,
        server: str,
        tool: str,
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        """Invoke a tool on a registered MCP server.

        Re-uses the long-lived session if it's still alive; otherwise
        reconnects once.
        """
        if server not in self._servers:
            raise KeyError(f"MCP server '{server}' is not registered.")
        cfg = self._servers[server]
        if not cfg.command:
            raise NotImplementedError(
                "HTTP/SSE transport for MCP servers is not implemented in v1. "
                "Use stdio transport (command=...) or subclass MCPManager."
            )
        if ClientSession is None or stdio_client is None or StdioServerParameters is None:
            raise RuntimeError("The `mcp` Python SDK is not installed. `pip install mcp`.")

        async def _invoke() -> Any:
            session = await self._ensure_session(cfg)
            result = await session.call_tool(tool, arguments or {})
            content = getattr(result, "content", None)
            if content is not None and not isinstance(content, (str, bytes)):
                try:
                    return [c.model_dump() for c in content]
                except Exception:
                    return str(content)
            return content

        return await asyncio.wait_for(_invoke(), timeout=cfg.timeout)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def call_id() -> str:
        return uuid.uuid4().hex

    def register_tool_descriptor(
        self,
        server: str,
        name: str,
        description: str = "",
        input_schema: dict[str, Any] | None = None,
        kind: str = "customer",
    ) -> None:
        """Statically register a tool descriptor without connecting (useful for tests)."""
        fq = f"{server}__{name}"
        self._tools[fq] = {
            "name": name,
            "server": server,
            "kind": kind,
            "description": description,
            "inputSchema": input_schema or {},
        }

    async def call_cached(
        self,
        server: str,
        tool: str,
        result: Any,
    ) -> Any:
        """Return a pre-computed result for a (server, tool) pair (test helper)."""
        fq = f"{server}__{tool}"
        if fq not in self._tools:
            raise KeyError(f"Tool '{tool}' on server '{server}' is not registered.")
        return result

    # ------------------------------------------------------------------ platform knowledge
    # NOTE: There is intentionally NO custom `fetch_platform_knowledge` /
    # `fetch_components` / `render_*` helper here. The MCP protocol's
    # purpose is to advertise its own tools via `tools/list` and let the
    # model call them via the provider's native function-calling — with
    # no SDK-side whitelist, alias map, or intent extraction step in the
    # middle. See Agent.handle_request for the tool-calling loop.
