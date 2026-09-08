"""A2A server wiring: transport construction, agent card, uvicorn
lifecycle, and the mcp SDK teardown-exception suppression.

The class-level :meth:`SyncfusionAgent.serve` is a thin delegator
to :func:`serve` here.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from .a2a.agent_card import default_agent_card
from .a2a.transport import A2ATransport, start_server
from .agent_exceptions import is_mcp_teardown_exception

logger = logging.getLogger("syncfusion_a2ui_agent")


def _resolve_allowed_origins(
    allowed_origins: list[str] | None,
) -> list[str]:
    """Pick the CORS origin list for the A2A transport.

    Resolution order:

    1. ``allowed_origins`` if the caller passed a non-empty list.
    2. The :envvar:`SYNCFUSION_A2A_CORS_ORIGINS` env var
       (comma-separated). Useful for deployment-time configuration
       without changing the call site.
    3. The default ``["*"]`` (permissive) for local development.
       Production deployments should always pass an explicit list.
    """
    if allowed_origins:
        return allowed_origins
    env_value = os.environ.get("SYNCFUSION_A2A_CORS_ORIGINS", "")
    parsed = [o.strip() for o in env_value.split(",") if o.strip()]
    if parsed:
        return parsed
    return ["*"]


def _streaming_enabled_from_env() -> bool:
    """Read the ``A2UI_STREAMING`` env var (default: enabled)."""
    return os.environ.get("A2UI_STREAMING", "1") != "0"


async def _serve_async(
    agent: Any,
    host: str,
    port: int,
    warmup: bool,
    allowed_origins: list[str] | None,
) -> None:
    """Async body of :func:`serve`. Bound to the agent's event loop.

    With ``warmup=True``, runs a no-op request inside the same
    event loop as the server before binding the port. This
    populates the MCP tool catalog without tearing down the mcp
    SDK's stdio context in a separate loop (which trips an
    anyio cancel-scope bug — see :mod:`agent_exceptions`).
    """
    if warmup:
        logger.info("Pre-warming MCP catalog before binding %s:%d", host, port)
        try:
            await agent.handle_request("__warmup__")
        except Exception as exc:
            logger.warning("Warmup request failed: %s", exc)

    # Streaming is now the default for `message/stream` requests.
    # The dual-mode executor picks the streaming path per-request by
    # inspecting the JSON-RPC method name on the call context (see
    # ``a2a/transport.py``), so a single A2ATransport instance serves
    # both `message/send` (buffered, single artifact) and
    # `message/stream` (per-op SSE) without reconfiguration.
    streaming_enabled = _streaming_enabled_from_env()
    transport = A2ATransport(
        agent.handle_request,
        allowed_origins=_resolve_allowed_origins(allowed_origins),
        stream_handler=agent.handle_request_stream,
        streaming=streaming_enabled,
    )
    if streaming_enabled:
        logger.info(
            "A2UI streaming enabled — message/stream requests will "
            "fan out envelope ops per-event over SSE."
        )
    else:
        logger.info(
            "A2UI streaming disabled (A2UI_STREAMING=0) — "
            "message/stream will fall back to the buffered "
            "message/send behaviour."
        )
    agent_card = default_agent_card(
        name=agent.__class__.__name__,
        url=f"http://{host}:{port}/",
    )
    app = transport.build_app(agent_card=agent_card)
    await start_server(app, host=host, port=port)


def serve(
    agent: Any,
    host: str = "0.0.0.0",
    port: int = 8080,
    warmup: bool = False,
    allowed_origins: list[str] | None = None,
) -> None:
    """Boot the Google A2A SDK server and block.

    Parameters
    ----------
    agent:
        The :class:`SyncfusionAgent` (or subclass) to serve.
    host:
        Network interface to bind. Defaults to ``"0.0.0.0"`` so the
        server is reachable from other hosts; use ``"127.0.0.1"``
        for local-only deployments.
    port:
        TCP port to bind. Defaults to ``8080``; the bundled
        examples use ``10004`` / ``10005`` / ``10006`` to avoid
        clashing with the port-8080 dev servers on the same host.
    warmup:
        When ``True``, runs a no-op request on the same event loop
        as the server *before* binding the port. This populates the
        MCP tool catalog so the first user-facing request doesn't
        pay the spawn cost. Currently disabled by default because
        the mcp SDK's stdio_client raises a known cosmetic
        RuntimeError during teardown when a session is left dangling
        across the warmup request — see
        :func:`agent_exceptions.is_mcp_teardown_exception` for the
        suppression contract.
    allowed_origins:
        Explicit CORS origin list. When ``None`` the helper falls
        back to the :envvar:`SYNCFUSION_A2A_CORS_ORIGINS` env var
        (comma-separated) and finally to ``["*"]`` for dev. The
        CORS middleware is configured with ``allow_credentials=False``
        so the wildcard default does not violate the browser spec.

    Notes
    -----
    The known mcp SDK teardown bug is suppressed inside this
    function's ``except`` block. The walker
    (:func:`is_mcp_teardown_exception`) distinguishes the bug from
    real user-code errors so unrelated ``RuntimeError``s are *not*
    swallowed. ``BaseException`` subclasses (``CancelledError``,
    ``KeyboardInterrupt``, ``SystemExit``) bypass the catch
    entirely, so Ctrl+C cleanly stops the server.
    """
    try:
        asyncio.run(
            _serve_async(
                agent=agent,
                host=host,
                port=port,
                warmup=warmup,
                allowed_origins=allowed_origins,
            )
        )
    except Exception as exc:
        # Known cosmetic bug in the mcp SDK's stdio_client: when an
        # mcp session is GC'd across task boundaries during teardown,
        # anyio raises RuntimeError("Attempted to exit cancel scope
        # in a different task than it was entered in") wrapped in an
        # ExceptionGroup. The session was used successfully during
        # the warmup request — the exception is only raised during
        # teardown and is safe to suppress.
        if not is_mcp_teardown_exception(exc):
            raise
        logger.debug("Suppressed known mcp SDK teardown exception: %r", exc)


__all__ = ["serve", "default_agent_card", "is_mcp_teardown_exception"]
