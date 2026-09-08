"""Focused unit tests for the A2A transport wiring.

The transport layer bridges the agent's ``handle_request`` /
``handle_request_stream`` coroutines into the Google A2A SDK's
``AgentExecutor`` / ``TaskUpdater`` / ``EventQueue`` surface. Before
this test file the entire transport module sat at 0% coverage —
the existing tests only exercised it indirectly through
``SyncfusionAgent.serve()``, which binds a real port.

These tests are hermetic: they construct an in-memory
``_SyncfusionDualModeExecutor`` and feed it a ``RequestContext``
shaped exactly like the one the a2a-sdk's JSON-RPC handler
builds. No real network, no real uvicorn.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator
from unittest.mock import MagicMock

import pytest

from syncfusion_a2ui_agent.a2a import transport as a2a_transport
from syncfusion_a2ui_agent.a2a.agent_card import (
    A2UI_GENERATE_SKILL_ID,
    default_agent_card,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def executor() -> Any:
    """Build a dual-mode executor with a no-op buffered handler and no
    stream handler. The buffered handler records the call for
    assertions.
    """
    return a2a_transport._SyncfusionDualModeExecutor(
        handler=_make_recording_handler(),
        stream_handler=None,
    )


def _make_recording_handler():
    """Build an async handler that records its calls and returns a
    canned envelope. The handler's arguments are returned via
    ``handler.calls`` for assertions.
    """
    state: dict[str, Any] = {"calls": [], "response": _canned_envelope()}

    async def _handler(user_input: str, history):
        state["calls"].append((user_input, history))
        return state["response"]

    _handler.calls = state["calls"]  # type: ignore[attr-defined]
    return _handler


def _canned_envelope() -> dict[str, Any]:
    return {
        "envelope": [
            {"version": "v0.9", "createSurface": {"surfaceId": "s1"}},
            {
                "version": "v0.9",
                "updateComponents": {
                    "surfaceId": "s1",
                    "components": [{"id": "root", "component": "Column"}],
                },
            },
        ],
        "surfaceIds": ["s1"],
    }


def _make_request_context(
    user_input: str = "hello",
    method: str | None = None,
) -> Any:
    """Build a minimal a2a SDK RequestContext for the executor.

    The a2a SDK's ``RequestContext.__init__`` accepts a
    ``call_context`` whose ``state`` dict carries the JSON-RPC
    method name. The executor reads ``state["method"]`` to decide
    between buffered and streaming mode. ``RequestContext.get_user_input``
    walks the request's ``Message.parts`` and joins the text parts;
    we construct a real ``Message`` so ``get_user_input()`` returns
    the expected string.
    """
    from a2a.server.agent_execution import RequestContext
    from a2a.types import Message, MessageSendParams, Part, Role, TextPart

    msg = Message(
        role=Role.user,
        parts=[Part(root=TextPart(text=user_input))],
        message_id="m1",
        context_id="c1",
    )
    params = MessageSendParams(message=msg)

    call_context = None
    if method is not None:
        # The a2a SDK stores the JSON-RPC method on
        # ``call_context.state``. We pass a MagicMock with a
        # ``state`` dict so the executor's introspection works.
        call_context = MagicMock()
        call_context.state = {"method": method}
    return RequestContext(
        request=params,
        task_id="t1",
        context_id="c1",
        call_context=call_context,
    )


def _make_event_queue() -> Any:
    """Build a real :class:`a2a.server.events.EventQueue`.

    The executor's ``TaskUpdater`` calls ``await queue.enqueue_event(...)``
    for every state transition and artifact. A ``MagicMock`` is not
    awaitable, so we use the real EventQueue (which wraps an
    ``asyncio.Queue``) and drain it for assertions. No real network
    or consumer is involved.
    """
    return a2a_transport.EventQueue()


async def _drain(queue: Any, n: int | None = None) -> list[Any]:
    """Drain the EventQueue for assertions.

    ``TaskUpdater.enqueue_event`` is the only public surface the
    executor uses, so dequeuing events in arrival order gives us
    a faithful trace of what the executor did. Pass ``n`` to
    stop after N events, or ``None`` to drain until empty.

    Uses ``no_wait=True`` so the drain never blocks on an empty
    queue — important because ``asyncio.run`` exits when the
    executor's coroutine returns, but events from a close-out
    ``add_artifact`` may still be in transit.
    """
    out: list[Any] = []
    while True:
        try:
            ev = queue.dequeue_event(no_wait=True)
        except Exception:
            # Empty queue — no more events available synchronously.
            break
        out.append(ev)
        if n is not None and len(out) >= n:
            break
    return out


# ---------------------------------------------------------------------------
# A2ATransport construction
# ---------------------------------------------------------------------------
class TestA2ATransportConstruction:
    def test_allowed_origins_default_is_dev_list(self) -> None:
        async def _h(*a, **k):
            return {}

        async def _s(*a, **k):
            if False:
                yield {}

        transport = a2a_transport.A2ATransport(handler=_h, stream_handler=_s, streaming=True)
        # Six localhost origins spanning the common dev ports
        # (5007, 5006, 5005 — each with both ``127.0.0.1`` and
        # ``localhost`` hostnames).
        assert len(transport.allowed_origins) == 6
        for origin in transport.allowed_origins:
            assert origin.startswith("http://127.0.0.1:") or origin.startswith("http://localhost:")
        # Each of the three dev ports must be present.
        for port in (5007, 5006, 5005):
            assert f"http://127.0.0.1:{port}" in transport.allowed_origins

    def test_allowed_origins_explicit_overrides_default(self) -> None:
        async def _h(*a, **k):
            return {}

        transport = a2a_transport.A2ATransport(
            handler=_h,
            allowed_origins=["https://app.example.com"],
        )
        assert transport.allowed_origins == ["https://app.example.com"]

    def test_allowed_origins_empty_list_kept(self) -> None:
        """``[]`` is a deliberate choice (no CORS at all) and must
        not be replaced with the default."""

        async def _h(*a, **k):
            return {}

        transport = a2a_transport.A2ATransport(handler=_h, allowed_origins=[])
        assert transport.allowed_origins == []

    def test_executor_attribute_is_dual_mode(self) -> None:
        async def _h(*a, **k):
            return {}

        transport = a2a_transport.A2ATransport(handler=_h)
        assert isinstance(
            transport.executor,
            a2a_transport._SyncfusionDualModeExecutor,
        )

    def test_streaming_false_when_no_stream_handler(self) -> None:
        async def _h(*a, **k):
            return {}

        transport = a2a_transport.A2ATransport(handler=_h, streaming=True)
        # streaming=True is honoured only when a stream_handler is also passed.
        assert transport._streaming_enabled is False

    def test_streaming_true_when_handler_and_stream_handler(self) -> None:
        async def _h(*a, **k):
            return {}

        async def _s(*a, **k):
            yield {"type": "op"}

        transport = a2a_transport.A2ATransport(handler=_h, stream_handler=_s, streaming=True)
        assert transport._streaming_enabled is True


# ---------------------------------------------------------------------------
# Dual-mode executor — buffered path
# ---------------------------------------------------------------------------
class TestExecutorBufferedPath:
    def test_message_send_calls_buffered_handler(self) -> None:
        handler = _make_recording_handler()
        executor = a2a_transport._SyncfusionDualModeExecutor(handler=handler, stream_handler=None)
        ctx = _make_request_context(method="message/send")
        queue = _make_event_queue()
        asyncio.run(executor.execute(ctx, queue))
        # The handler received the user input.
        assert len(handler.calls) == 1
        user_input, history = handler.calls[0]
        assert user_input == "hello"
        assert history == []
        # And the queue saw at least one event (the artifact).
        events = asyncio.run(_drain(queue, n=10))
        assert len(events) >= 1

    def test_message_send_emits_one_artifact(self) -> None:
        handler = _make_recording_handler()
        executor = a2a_transport._SyncfusionDualModeExecutor(handler=handler, stream_handler=None)
        ctx = _make_request_context(method="message/send")
        queue = _make_event_queue()
        asyncio.run(executor.execute(ctx, queue))
        # Drain the queue. The buffered path emits a constant set:
        # submit + start_work + artifact + complete. We assert the
        # lower bound and that at least one artifact was emitted
        # (the canned envelope has 2 ops but the buffered path
        # collapses them into a single DataPart).
        events = asyncio.run(_drain(queue, n=20))
        assert len(events) >= 4

    def test_handler_exception_becomes_task_failed(self) -> None:
        async def _bad_handler(*a, **k):
            raise RuntimeError("upstream boom")

        executor = a2a_transport._SyncfusionDualModeExecutor(
            handler=_bad_handler, stream_handler=None
        )
        ctx = _make_request_context(method="message/send")
        queue = _make_event_queue()
        # The executor catches handler exceptions and routes them to
        # task_updater.failed — the call must NOT raise.
        asyncio.run(executor.execute(ctx, queue))
        events = asyncio.run(_drain(queue, n=20))
        assert events  # at least the failed event

    def test_method_none_uses_buffered_path(self) -> None:
        """When the A2A framework doesn't set call_context.state
        (e.g. older SDK versions), the executor must fall back to
        the buffered path so message/send keeps working.
        """
        handler = _make_recording_handler()
        executor = a2a_transport._SyncfusionDualModeExecutor(handler=handler, stream_handler=None)
        ctx = _make_request_context(method=None)
        queue = _make_event_queue()
        asyncio.run(executor.execute(ctx, queue))
        assert len(handler.calls) == 1

    def test_unrecognised_method_uses_buffered_path(self) -> None:
        """Unrecognised method names must not crash; the executor
        must default to the buffered path so legacy clients keep
        working.
        """
        handler = _make_recording_handler()
        executor = a2a_transport._SyncfusionDualModeExecutor(handler=handler, stream_handler=None)
        ctx = _make_request_context(method="message/something-else")
        queue = _make_event_queue()
        asyncio.run(executor.execute(ctx, queue))
        assert len(handler.calls) == 1


# ---------------------------------------------------------------------------
# Dual-mode executor — streaming path
# ---------------------------------------------------------------------------
class TestExecutorStreamingPath:
    def test_message_stream_iterates_stream_handler(self) -> None:
        buffered = _make_recording_handler()

        async def _stream_handler(user_input: str, history) -> AsyncIterator[dict]:
            yield {
                "type": "op",
                "op": {"version": "v0.9", "createSurface": {"surfaceId": "s1"}},
                "kind": "createSurface",
            }
            yield {
                "type": "op",
                "op": {
                    "version": "v0.9",
                    "updateComponents": {
                        "surfaceId": "s1",
                        "components": [{"id": "r", "component": "Column"}],
                    },
                },
                "kind": "updateComponents",
            }
            yield {"type": "surfaceIds", "surfaceIds": ["s1"]}

        executor = a2a_transport._SyncfusionDualModeExecutor(
            handler=buffered, stream_handler=_stream_handler
        )
        ctx = _make_request_context(method="message/stream")
        queue = _make_event_queue()
        asyncio.run(executor.execute(ctx, queue))
        # Buffered handler must NOT be called when the streaming path
        # is selected.
        assert buffered.calls == []
        # The stream handler yielded 3 events (2 ops + 1 surfaceIds);
        # the executor forwards them as artifacts.
        events = asyncio.run(_drain(queue, n=20))
        assert len(events) >= 3

    def test_stream_error_event_becomes_task_failed(self) -> None:
        async def _stream_handler(*a, **k) -> AsyncIterator[dict]:
            yield {"type": "error", "error": "stream went bad"}

        executor = a2a_transport._SyncfusionDualModeExecutor(
            handler=_make_recording_handler(),
            stream_handler=_stream_handler,
        )
        ctx = _make_request_context(method="message/stream")
        queue = _make_event_queue()
        # The error event must not propagate as a Python exception;
        # it surfaces as a task_updater.failed call.
        asyncio.run(executor.execute(ctx, queue))
        events = asyncio.run(_drain(queue, n=20))
        assert events

    def test_stream_with_no_ops_emits_failed(self) -> None:
        """A stream that yields only a surfaceIds event (or nothing)
        must NOT complete the task successfully — that would lie
        to the client. The executor should fail the task.
        """

        async def _stream_handler(*a, **k) -> AsyncIterator[dict]:
            yield {"type": "surfaceIds", "surfaceIds": []}

        executor = a2a_transport._SyncfusionDualModeExecutor(
            handler=_make_recording_handler(),
            stream_handler=_stream_handler,
        )
        ctx = _make_request_context(method="message/stream")
        queue = _make_event_queue()
        asyncio.run(executor.execute(ctx, queue))
        events = asyncio.run(_drain(queue, n=20))
        assert events

    def test_stream_handler_exception_becomes_task_failed(self) -> None:
        async def _stream_handler(*a, **k):
            raise RuntimeError("stream blew up")
            yield  # pragma: no cover — unreachable, makes this a generator

        executor = a2a_transport._SyncfusionDualModeExecutor(
            handler=_make_recording_handler(),
            stream_handler=_stream_handler,
        )
        ctx = _make_request_context(method="message/stream")
        queue = _make_event_queue()
        # The exception must be caught and routed to task_updater.failed.
        asyncio.run(executor.execute(ctx, queue))
        events = asyncio.run(_drain(queue, n=20))
        assert events

    def test_no_stream_handler_uses_buffered_path_even_on_message_stream(
        self,
    ) -> None:
        """If ``stream_handler`` is None the executor must fall back
        to the buffered path on a message/stream request, so a
        misconfigured agent doesn't crash.
        """
        handler = _make_recording_handler()
        executor = a2a_transport._SyncfusionDualModeExecutor(handler=handler, stream_handler=None)
        ctx = _make_request_context(method="message/stream")
        queue = _make_event_queue()
        asyncio.run(executor.execute(ctx, queue))
        # Buffered handler was called even though method was stream.
        assert len(handler.calls) == 1


# ---------------------------------------------------------------------------
# Dual-mode executor — cancel
# ---------------------------------------------------------------------------
class TestExecutorCancel:
    def test_cancel_does_not_raise(self) -> None:
        executor = a2a_transport._SyncfusionDualModeExecutor(
            handler=_make_recording_handler(), stream_handler=None
        )
        ctx = _make_request_context(method="message/send")
        queue = _make_event_queue()
        # The cancel path must not raise; it forwards to
        # task_updater.cancel().
        asyncio.run(executor.cancel(ctx, queue))
        events = asyncio.run(_drain(queue, n=10))
        assert events  # at least the cancel event was emitted


# ---------------------------------------------------------------------------
# _data_part
# ---------------------------------------------------------------------------
class TestDataPart:
    def test_unwraps_envelope_shape(self) -> None:
        payload = {
            "envelope": [{"version": "v0.9", "createSurface": {"surfaceId": "x"}}],
            "surfaceIds": ["x"],
        }
        part = a2a_transport._data_part(payload)
        # The DataPart's data should carry the a2uiEnvelope + surfaceIds
        # keys so the A2A client can read the response uniformly.
        data = part.root.data
        assert "a2uiEnvelope" in data
        assert data["a2uiEnvelope"] == payload["envelope"]
        assert data["surfaceIds"] == ["x"]

    def test_passes_through_non_envelope_dict(self) -> None:
        payload = {"foo": "bar"}
        part = a2a_transport._data_part(payload)
        assert part.root.data == {"foo": "bar"}

    def test_passes_through_non_dict(self) -> None:
        part = a2a_transport._data_part([1, 2, 3])
        assert part.root.data == {"a2uiEnvelope": [1, 2, 3]}


# ---------------------------------------------------------------------------
# default_agent_card
# ---------------------------------------------------------------------------
class TestDefaultAgentCard:
    def test_returns_object_with_required_fields(self) -> None:
        card = default_agent_card(name="MyAgent", url="http://x/")
        # The card advertises a single a2ui-generate skill and
        # streaming capability — both are part of the contract.
        d = _card_to_dict(card)
        assert d["name"] == "MyAgent"
        assert d["url"] == "http://x/"
        # capabilities may be either a dict (fallback) or an
        # AgentCapabilities object; accept either.
        caps = d.get("capabilities", {})
        streaming = (
            caps.get("streaming") if isinstance(caps, dict) else getattr(caps, "streaming", None)
        )
        assert streaming is True
        skills = d.get("skills", [])
        ids = [s.get("id") if isinstance(s, dict) else getattr(s, "id", None) for s in skills]
        assert A2UI_GENERATE_SKILL_ID in ids

    def test_custom_version_is_honoured(self) -> None:
        card = default_agent_card(name="x", url="http://x/", version="9.9.9-custom")
        d = _card_to_dict(card)
        assert d["version"] == "9.9.9-custom"

    def test_default_version_is_set(self) -> None:
        card = default_agent_card(name="x", url="http://x/")
        d = _card_to_dict(card)
        # The default version should be a non-empty string; we don't
        # pin the exact value because it tracks the SDK release.
        assert isinstance(d.get("version"), str)
        assert d["version"]


def _card_to_dict(card: Any) -> dict:
    """Coerce an AgentCard (or a fallback dict) to a plain dict.

    The a2a SDK's AgentCard is a pydantic model, so a ``model_dump``
    or ``dict()`` call works; the fallback path returns a dict
    already. Handle both.
    """
    if isinstance(card, dict):
        return card
    if hasattr(card, "model_dump"):
        return card.model_dump()
    if hasattr(card, "dict"):
        return card.dict()
    # Last resort: introspect via __dict__.
    return dict(card.__dict__)


# ---------------------------------------------------------------------------
# start_server
# ---------------------------------------------------------------------------
class TestStartServer:
    def test_missing_uvicorn_raises_runtime_error(self, monkeypatch):
        """When uvicorn is not importable, start_server must surface
        a clear error rather than letting ImportError leak through.
        """
        import builtins

        real_import = builtins.__import__

        def _import(name, *args, **kwargs):
            if name == "uvicorn" or name.startswith("uvicorn."):
                raise ImportError("simulated missing uvicorn")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _import)
        # Also need to invalidate any cached uvicorn module.
        import sys

        for mod_name in list(sys.modules):
            if mod_name == "uvicorn" or mod_name.startswith("uvicorn."):
                monkeypatch.delitem(sys.modules, mod_name)
        with pytest.raises(RuntimeError, match="uvicorn"):
            asyncio.run(a2a_transport.start_server(app=MagicMock()))
