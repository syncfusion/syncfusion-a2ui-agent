"""A2A transport glue.

This module wires the agent's `handle_request` method into the Google A2A
Python SDK. The actual SDK is OPTIONAL — if it's not installed we expose a
graceful fallback that lets the agent run in-process for tests, and a clear
error if a customer tries to call `serve()` without it.

Compatible with `a2a-sdk` 0.3.x — the constructor shape is:

    A2AStarletteApplication(
        agent_card=...,
        http_handler=DefaultRequestHandler(agent_executor=..., task_store=...),
    )

The transport is intentionally thin — it does not contain orchestration or
business logic, only lifecycle/session/streaming/task-execution concerns
(spec §4).
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger("syncfusion_a2ui_agent.a2a")

# A `Handler` is the agent's `handle_request` coroutine.
Handler = Callable[[str, list[dict[str, Any]] | None], Awaitable[dict[str, Any]]]


try:  # pragma: no cover - import-time only
    from a2a.server.agent_execution import AgentExecutor, RequestContext
    from a2a.server.apps import A2AStarletteApplication
    from a2a.server.events import EventQueue
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
    from a2a.types import (
        DataPart,
        Message,
        Part,
        Role,
        TaskState,
        TextPart,
    )

    _A2A_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    AgentExecutor = None  # type: ignore
    RequestContext = None  # type: ignore
    A2AStarletteApplication = None  # type: ignore
    DefaultRequestHandler = None  # type: ignore
    InMemoryTaskStore = None  # type: ignore
    TaskUpdater = None  # type: ignore
    EventQueue = None  # type: ignore
    Part = None  # type: ignore
    TextPart = None  # type: ignore
    DataPart = None  # type: ignore
    TaskState = None  # type: ignore
    Role = None  # type: ignore
    Message = None  # type: ignore
    _A2A_AVAILABLE = False
    logger.debug("a2a-sdk not importable: %s", exc)


class A2ATransport:
    """Thin wrapper around the Google A2A SDK AgentExecutor.

    Customers normally do not interact with this directly — `SyncfusionAgent.serve()`
    instantiates and runs it.
    """

    def __init__(
        self,
        handler: Handler,
        allowed_origins: list[str] | None = None,
        stream_handler: Callable[..., Any] | None = None,
        streaming: bool = False,
    ) -> None:
        if not _A2A_AVAILABLE:
            raise RuntimeError(
                "The `a2a-sdk` package is not installed. `pip install a2a-sdk` to serve "
                "the agent over A2A. You can still call `agent.handle_request()` directly."
            )
        self._handler = handler
        self._stream_handler = stream_handler
        self._streaming_enabled = streaming and stream_handler is not None
        # One executor for both message/send and message/stream — the
        # A2A framework calls the SAME agent_executor.execute() for
        # both methods; only the framework's post-processing differs
        # (blocking single result vs SSE). We branch INSIDE execute()
        # on the actual per-request method so message/send keeps its
        # original single-artifact behaviour even when streaming is
        # enabled for message/stream requests.
        self._executor = _SyncfusionDualModeExecutor(
            handler, stream_handler if self._streaming_enabled else None
        )
        self._task_store = InMemoryTaskStore()
        # Default: allow common local dev origins. Pass `allowed_origins`
        # explicitly to the A2ATransport constructor for production
        # deployments; ``SyncfusionAgent.serve()`` forwards its own
        # `allowed_origins` kwarg through to this constructor.
        self.allowed_origins = (
            allowed_origins
            if allowed_origins is not None
            else [
                "http://127.0.0.1:5007",
                "http://localhost:5007",
                "http://127.0.0.1:5006",
                "http://localhost:5006",
                "http://127.0.0.1:5005",
                "http://localhost:5005",
            ]
        )

    @property
    def executor(self) -> Any:
        return self._executor

    def build_app(
        self,
        agent_card: Any,
        agent_card_url: str = "/.well-known/agent-card.json",
        rpc_url: str = "/",
    ) -> Any:
        """Build the Starlette ASGI app from the executor + agent card.

        Wires the executor through a DefaultRequestHandler, hands it to
        A2AStarletteApplication as `http_handler`, and calls `.build()` to
        produce a real Starlette app (the A2AStarletteApplication itself is
        not directly ASGI-callable in a2a-sdk 0.3.x).
        """
        if not _A2A_AVAILABLE:
            raise RuntimeError("a2a-sdk is not installed; cannot build A2A app.")
        http_handler = DefaultRequestHandler(
            agent_executor=self._executor,
            task_store=self._task_store,
        )
        app = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=http_handler,
        )
        starlette_app = app.build(agent_card_url=agent_card_url, rpc_url=rpc_url)
        return _wrap_cors(starlette_app, allowed_origins=self.allowed_origins)


# ---------------------------------------------------------------------------
# Executor implementation
# ---------------------------------------------------------------------------
if _A2A_AVAILABLE:

    class _SyncfusionDualModeExecutor(AgentExecutor):
        """Bridges A2A request/response to SyncfusionAgent, per-request.

        The A2A framework calls the SAME `execute()` for both
        `message/send` and `message/stream` — it only differs in how it
        *consumes* the queue afterward (blocking single result vs SSE).
        So we must not pick streaming-vs-buffered behaviour once at
        server startup; we check the real method on every call via
        `context.call_context.state['method']` (set by the SDK's
        jsonrpc_app._handle_requests before invoking the executor).

        - `message/send` (or `method` missing/unrecognised): buffer the
          full result and emit ONE artifact, exactly like the original
          non-streaming behaviour. This keeps `message/send` unchanged
          even when `stream_handler` is configured.
        - `message/stream`: iterate `stream_handler` (the agent's
          `handle_request_stream`) and emit one artifact per op.
        """

        def __init__(
            self,
            handler: Handler,
            stream_handler: Callable[..., Any] | None,
        ) -> None:
            self._handler = handler
            self._stream_handler = stream_handler

        @staticmethod
        def _is_streaming_request(context: RequestContext) -> bool:
            call_context = getattr(context, "call_context", None)
            method = getattr(call_context, "state", {}).get("method") if call_context else None
            return method == "message/stream"

        async def execute(
            self,
            context: RequestContext,
            event_queue: EventQueue,
        ) -> None:
            user_input = context.get_user_input() or ""
            history = _extract_history(context)
            if not context.task_id or not context.context_id:
                raise ValueError("task_id and context_id are required")
            task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
            await task_updater.submit(message=_user_message(user_input, context.context_id))
            await task_updater.start_work()

            use_stream = self._stream_handler is not None and self._is_streaming_request(context)

            if not use_stream:
                # Original buffered behaviour — unchanged for message/send.
                try:
                    result = await self._handler(user_input, history)
                except Exception as exc:
                    logger.exception("handle_request failed: %s", exc)
                    await task_updater.failed(
                        message=_text_message(f"Agent error: {exc}", context.context_id),
                    )
                    return
                await task_updater.add_artifact(
                    parts=[_data_part(result)],
                    artifact_id=uuid.uuid4().hex,
                    name="a2ui-response",
                )
                await task_updater.complete()
                return

            # Streaming path (message/stream): one artifact per op.
            surface_artifact_ids: dict[str, str] = {}
            emitted_any = False
            if not self._stream_handler:
                raise RuntimeError("stream_handler must be set for streaming requests")
            try:
                async for event in self._stream_handler(user_input, history):
                    kind = event.get("type")
                    if kind == "error":
                        await task_updater.failed(
                            message=_text_message(
                                f"Agent error: {event.get('error')}",
                                context.context_id,
                            )
                        )
                        return
                    if kind == "op":
                        op = event["op"]
                        op_kind = event.get("kind", "")
                        sid = (op.get(op_kind) or {}).get("surfaceId", "default")
                        artifact_id = surface_artifact_ids.setdefault(sid, uuid.uuid4().hex)
                        await task_updater.add_artifact(
                            parts=[_data_part({"envelope": [op], "surfaceIds": [sid]})],
                            artifact_id=artifact_id,
                            name="a2ui-response",
                        )
                        emitted_any = True
                    elif kind == "surfaceIds":
                        break
            except Exception as exc:
                logger.exception("handle_request_stream failed: %s", exc)
                await task_updater.failed(
                    message=_text_message(f"Agent error: {exc}", context.context_id)
                )
                return

            if not emitted_any:
                await task_updater.failed(
                    message=_text_message("Agent produced no envelope ops.", context.context_id)
                )
                return
            await task_updater.complete()

        async def cancel(
            self,
            context: RequestContext,
            event_queue: EventQueue,
        ) -> None:
            if not context.task_id or not context.context_id:
                raise ValueError("task_id and context_id are required")
            task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
            await task_updater.cancel()

else:  # pragma: no cover

    class _SyncfusionDualModeExecutor:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("a2a-sdk is not installed.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _extract_history(context: RequestContext) -> list[dict[str, Any]]:
    """Best-effort: project A2A prior messages into ``[{role, content}, ...]``.

    The Google A2A protocol surfaces conversation history on the
    *Task* object (``task.history``), not on the per-message request.
    For a request to have meaningful history, the caller (or the
    upstream A2A server) must have attached the prior task to the
    current request via either ``context.current_task`` or
    ``context.related_tasks``.

    We walk both, pull every :class:`a2a.types.Message` we can find,
    and project it to a plain ``{role, content}`` dict the agent
    loop can consume. ``Part`` is a discriminated union; we keep
    only :class:`a2a.types.TextPart` (text content) and skip other
    part types (files, data parts) because the agent's
    :class:`LoopEngine` expects plain string content per turn.

    When no prior history is available — the common case for the
    first request in a conversation — this returns ``[]`` and the
    agent runs with no prior turns, matching the pre-fix behaviour.
    """
    if context is None:
        return []
    # 1) The current task, if the request is mid-conversation.
    tasks: list[Any] = []
    current = getattr(context, "current_task", None)
    if current is not None:
        tasks.append(current)
    # 2) Any tasks the caller attached to this request as related
    #    (e.g. for tool-use continuation).
    related = getattr(context, "related_tasks", None) or []
    tasks.extend(related)

    if not tasks:
        return []

    history: list[dict[str, Any]] = []
    seen_message_ids: set[str] = set()
    for task in tasks:
        task_history = getattr(task, "history", None) or []
        for msg in task_history:
            msg_id = getattr(msg, "message_id", None)
            if msg_id is not None:
                if msg_id in seen_message_ids:
                    continue
                seen_message_ids.add(msg_id)
            role = getattr(msg, "role", None)
            # ``Role`` is a string enum; normalise to the lowercase
            # string the agent loop expects.
            role_value = role.value if role and hasattr(role, "value") else str(role or "")
            content = _parts_to_text(getattr(msg, "parts", None) or [])
            if not content:
                # A message with no text parts is not useful to the
                # provider; skip it rather than appending an empty turn.
                continue
            history.append({"role": role_value, "content": content})
    return history


def _parts_to_text(parts: list[Any]) -> str:
    """Flatten a list of A2A ``Part`` objects into a single text string.

    Only ``TextPart`` payloads are kept. DataPart and FilePart are
    skipped (the agent's :class:`LoopEngine` cannot consume them as
    turn content). Multiple text parts are joined with newlines.
    """
    fragments: list[str] = []
    for part in parts:
        # ``Part`` is a discriminated union whose concrete subtype
        # lives on ``.root`` (in a2a-sdk 0.3.x). Fall back to the
        # part itself for older SDK shapes.
        root = getattr(part, "root", part)
        text = getattr(root, "text", None)
        if text is None and isinstance(root, dict):
            text = root.get("text")
        if text:
            fragments.append(str(text))
    return "\n".join(fragments)


def _text_part(text: str) -> Any:
    return Part(root=TextPart(text=text))


def _data_part(payload: Any) -> Any:
    """Wrap an agent return value as an A2A DataPart.

    A2UI v0.9 responses are JSON arrays of envelope ops, not single dicts.
    The agent's `handle_request` returns:
        {"envelope": [...], "surfaceIds": [...]}
    We unwrap that and forward the envelope + surface ids to the client.
    """
    if isinstance(payload, dict) and "envelope" in payload:
        data = {
            "a2uiEnvelope": payload.get("envelope", []),
            "surfaceIds": payload.get("surfaceIds", []),
        }
    elif not isinstance(payload, dict):
        data = {"a2uiEnvelope": payload}
    else:
        data = payload
    return Part(root=DataPart(data=data))


def _text_message(text: str, context_id: str) -> Any:
    return Message(
        role=Role.agent,
        parts=[_text_part(text)],
        message_id=uuid.uuid4().hex,
        context_id=context_id,
    )


def _user_message(text: str, context_id: str) -> Any:
    return Message(
        role=Role.user,
        parts=[_text_part(text)],
        message_id=uuid.uuid4().hex,
        context_id=context_id,
    )


def _wrap_cors(app: Any, allowed_origins: list[str]) -> Any:
    """Wrap a Starlette app with permissive CORS for browser clients.

    Used in dev/test so a static test page (e.g. on :5007) can POST to the
    A2A server (e.g. on :10004). For production, customers should replace
    this with their own CORS policy or remove it entirely when the browser
    is not in the picture.
    """
    try:
        from starlette.middleware.cors import CORSMiddleware
    except Exception as exc:  # pragma: no cover
        logger.debug("starlette.middleware.cors unavailable: %s", exc)
        return app
    # ``allow_credentials=False`` because Starlette (and the browser spec)
    # refuse to combine credentials with the ``*`` wildcard origin. For
    # production, pass an explicit list of origins instead of relying on
    # the defaults.
    return CORSMiddleware(
        app=app,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# ---------------------------------------------------------------------------
# Convenience: start a uvicorn server
# ---------------------------------------------------------------------------
async def start_server(
    app: Any,
    host: str = "0.0.0.0",
    port: int = 8080,
) -> None:
    """Run the A2A Starlette app under uvicorn. Blocking.

    Keep-alive is enabled by default (30 s) so subsequent requests
    reuse the TCP connection instead of paying a fresh handshake.
    Gzip is controlled by the standard ``UVICORN_GZIP=true`` env
    var to keep the Config import surface stable. Override via
    uvicorn env vars if a different policy is needed.
    """
    try:
        import uvicorn
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "uvicorn is required to run the A2A server. `pip install uvicorn`."
        ) from exc
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
        # Hold idle TCP connections open for 30 s so subsequent
        # requests reuse them instead of paying a fresh handshake.
        timeout_keep_alive=30,
        # Gzip is controlled by the standard ``UVICORN_GZIP=true``
        # env var (set in the environment, not here, to keep the
        # ``uvicorn.Config`` import surface stable).
    )
    server = uvicorn.Server(config)
    await server.serve()
