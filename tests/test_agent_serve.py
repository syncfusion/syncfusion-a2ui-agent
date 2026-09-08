"""Tests for ``agent_serve.py`` — the extracted A2A-server boot module.

Goal: lift ``agent_serve.py`` coverage from 0% past the testing gate
without standing up a real uvicorn server. We cover:

* ``_resolve_allowed_origins`` — three-way resolution (explicit / env / default)
* ``_streaming_enabled_from_env`` — default-on + A2UI_STREAMING=0 opt-out
* ``SyncfusionAgent.serve()`` — confirms it is a thin delegator (not a copy)
* ``_serve_async()`` early wiring — verifies transport construction and
  warmup invocation without binding a port
* The teardown-suppression contract inside ``serve()`` — feeds the
  known cancel-scope RuntimeError through the BoundaryError branch and
  confirms the call returns cleanly instead of re-raising.

uvicorn is mocked out for these tests via a stubbed ``start_server``
injected through ``monkeypatch.setattr``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from factories import VALID_ENVELOPE_TEXT, FakeProvider

from syncfusion_a2ui_agent import agent as agent_module
from syncfusion_a2ui_agent import agent_serve as agent_serve_module
from syncfusion_a2ui_agent.agent_serve import (
    _resolve_allowed_origins,
    _serve_async,
    _streaming_enabled_from_env,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_agent(**kwargs: Any) -> Any:
    """Construct a real ``SyncfusionAgent`` with a fake provider.

    Agent construction under the standard env-isolation fixture means
    none of the optional MCP / orchestration layers come up, so the
    agent is safe to use in a unit test without external services.
    """
    return agent_module.SyncfusionAgent(
        model=FakeProvider(scripted_text=VALID_ENVELOPE_TEXT),
        **kwargs,
    )


def _build_subclass_agent(name: str) -> Any:
    """Build a one-off ``SyncfusionAgent`` subclass instance.

    Used by tests that need to verify the agent's class name flows
    through to the A2A agent card (the public default uses
    ``agent.__class__.__name__``).
    """

    class _Sub(agent_module.SyncfusionAgent):
        pass

    _Sub.__name__ = name
    _Sub.__qualname__ = name
    return _Sub(model=FakeProvider(scripted_text=VALID_ENVELOPE_TEXT))


@pytest.fixture
def stubbed_start_server(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace ``A2ATransport.start_server`` with an awaitable stub.

    Tests that exercise ``_serve_async`` need the uvicorn bind to be a
    no-op so the test does not actually open a TCP socket. The helper
    records the call args for assertions.
    """
    calls: MagicMock = MagicMock()

    async def _fake_start_server(app: Any, host: str, port: int) -> None:
        calls(app=app, host=host, port=port)

    monkeypatch.setattr(agent_serve_module, "start_server", _fake_start_server)
    monkeypatch.setattr("syncfusion_a2ui_agent.a2a.transport.start_server", _fake_start_server)
    return calls


# ---------------------------------------------------------------------------
# _resolve_allowed_origins
# ---------------------------------------------------------------------------
class TestResolveAllowedOrigins:
    def test_explicit_list_returned_unchanged(self) -> None:
        explicit = ["https://app.example.com", "https://admin.example.com"]
        assert _resolve_allowed_origins(explicit) == explicit

    def test_explicit_empty_list_falls_back(self) -> None:
        # ``[]`` is falsy — should fall through to env/default rather
        # than become the literal list. This matches the documented
        # "non-empty list" contract on the helper.
        assert _resolve_allowed_origins([]) == ["*"]

    def test_env_var_single_origin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(
            "SYNCFUSION_A2A_CORS_ORIGINS",
            "https://one.example.com",
        )
        assert _resolve_allowed_origins(None) == ["https://one.example.com"]

    def test_env_var_comma_separated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(
            "SYNCFUSION_A2A_CORS_ORIGINS",
            "https://a.example.com, https://b.example.com ,https://c.example.com",
        )
        assert _resolve_allowed_origins(None) == [
            "https://a.example.com",
            "https://b.example.com",
            "https://c.example.com",
        ]

    def test_env_var_empty_string_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SYNCFUSION_A2A_CORS_ORIGINS", "")
        assert _resolve_allowed_origins(None) == ["*"]

    def test_env_var_whitespace_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SYNCFUSION_A2A_CORS_ORIGINS", "   ,   ")
        # Whitespace split yields empty fragments which the helper drops.
        assert _resolve_allowed_origins(None) == ["*"]

    def test_default_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SYNCFUSION_A2A_CORS_ORIGINS", raising=False)
        assert _resolve_allowed_origins(None) == ["*"]

    def test_explicit_overrides_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(
            "SYNCFUSION_A2A_CORS_ORIGINS",
            "https://ignored.example.com",
        )
        assert _resolve_allowed_origins(["https://override.example.com"]) == [
            "https://override.example.com"
        ]


# ---------------------------------------------------------------------------
# _streaming_enabled_from_env
# ---------------------------------------------------------------------------
class TestStreamingEnabledFromEnv:
    def test_default_on_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("A2UI_STREAMING", raising=False)
        assert _streaming_enabled_from_env() is True

    def test_explicit_one_enables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("A2UI_STREAMING", "1")
        assert _streaming_enabled_from_env() is True

    def test_zero_disables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("A2UI_STREAMING", "0")
        assert _streaming_enabled_from_env() is False

    def test_only_exact_zero_disables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Any non-"0" string is treated as enabled — matches the
        # "!=" comparison in the implementation. Document this lock.
        for value in ("", "false", "False", "no", "off", "00"):
            monkeypatch.setenv("A2UI_STREAMING", value)
            assert _streaming_enabled_from_env() is True, value


# ---------------------------------------------------------------------------
# _serve_async early wiring
# ---------------------------------------------------------------------------
class TestServeAsyncEarlyWiring:
    async def test_warmup_invokes_handle_request(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stubbed_start_server: MagicMock,
    ) -> None:
        ag = _build_agent()
        calls = MagicMock(wraps=lambda: None)
        original_handle = ag.handle_request

        async def _spy_handle_request(message: str, **_: Any) -> str:
            calls(message=message)
            return await original_handle(message)

        # Replace on the instance only so we don't break the test for
        # other agents that share this provider.
        monkeypatch.setattr(ag, "handle_request", _spy_handle_request)

        await _serve_async(
            ag,
            host="127.0.0.1",
            port=0,
            warmup=True,
            allowed_origins=["https://only.example.com"],
        )

        assert calls.call_args.kwargs["message"] == "__warmup__"
        assert stubbed_start_server.call_count == 1
        assert stubbed_start_server.call_args.kwargs["host"] == "127.0.0.1"
        assert stubbed_start_server.call_args.kwargs["port"] == 0

    async def test_warmup_failure_is_swallowed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stubbed_start_server: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        ag = _build_agent()

        async def _explode(message: str) -> str:
            raise RuntimeError("simulated warmup failure")

        monkeypatch.setattr(ag, "handle_request", _explode)

        with caplog.at_level("WARNING", logger="syncfusion_a2ui_agent"):
            await _serve_async(
                ag,
                host="127.0.0.1",
                port=0,
                warmup=True,
                allowed_origins=None,
            )

        assert any("Warmup request failed" in r.message for r in caplog.records)
        # Even with a warmup failure, the server still binds —
        # matching the documented "best-effort warmup" contract.
        assert stubbed_start_server.call_count == 1

    async def test_no_warmup_skips_handle_request(
        self,
        stubbed_start_server: MagicMock,
    ) -> None:
        ag = _build_agent()
        # Replace handle_request with a sentinel that explodes if called.
        # If the implementation invokes it, the test fails loudly.
        boom = MagicMock(side_effect=AssertionError("handle_request called without warmup=True"))

        async def _fail(*_args: Any, **_kwargs: Any) -> str:
            boom()
            return ""

        ag.handle_request = _fail  # type: ignore[method-assign]

        await _serve_async(
            ag,
            host="127.0.0.1",
            port=0,
            warmup=False,
            allowed_origins=None,
        )

        assert boom.call_count == 0

    async def test_streaming_enabled_logged(
        self,
        stubbed_start_server: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        ag = _build_agent()
        with caplog.at_level("INFO", logger="syncfusion_a2ui_agent"):
            await _serve_async(
                ag,
                host="127.0.0.1",
                port=0,
                warmup=False,
                allowed_origins=None,
            )
        assert any("A2UI streaming enabled" in r.message for r in caplog.records)

    async def test_streaming_disabled_logged(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stubbed_start_server: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        ag = _build_agent()
        monkeypatch.setenv("A2UI_STREAMING", "0")
        with caplog.at_level("INFO", logger="syncfusion_a2ui_agent"):
            await _serve_async(
                ag,
                host="127.0.0.1",
                port=0,
                warmup=False,
                allowed_origins=None,
            )
        assert any("A2UI streaming disabled" in r.message for r in caplog.records)

    async def test_agent_card_uses_class_name(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stubbed_start_server: MagicMock,
    ) -> None:
        ag = _build_subclass_agent("MyDemoAgent")
        captured: dict[str, Any] = {}

        def _fake_build_app(self: Any, agent_card: Any) -> str:  # type: ignore[no-untyped-def]
            captured["agent_card"] = agent_card
            captured["name"] = getattr(agent_card, "name", None)
            return "FAKE_APP"

        monkeypatch.setattr(
            "syncfusion_a2ui_agent.a2a.transport.A2ATransport.build_app",
            _fake_build_app,
        )

        await _serve_async(
            ag,
            host="myhost",
            port=12345,
            warmup=False,
            allowed_origins=None,
        )

        assert captured["name"] == "MyDemoAgent"


# ---------------------------------------------------------------------------
# SyncfusionAgent.serve() — must be a thin delegator
# ---------------------------------------------------------------------------
class TestAgentClassServeDelegator:
    def test_serve_classmethod_delegates_to_module(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stubbed_start_server: MagicMock,
    ) -> None:
        """``agent.SyncfusionAgent.serve`` should call ``agent_serve.serve``.

        Locked by signature: the delegating wrapper uses ``from
        .agent_serve import serve as _serve`` and re-exports it on
        the class. Replacing the module-level ``serve`` and asserting
        the wrapper picks up the new target proves the wiring; if it
        ever copies the implementation, this test catches it.
        """
        ag = _build_agent()

        sentinel = MagicMock()

        def _fake_module_serve(*_args: Any, agent: Any = None, **_kwargs: Any) -> None:
            sentinel(agent)

        monkeypatch.setattr(agent_serve_module, "serve", _fake_module_serve)

        # Call through the class to prove the unbound lookup path
        # matches the public API consumers use.
        agent_module.SyncfusionAgent.serve(
            ag,
            host="127.0.0.1",
            port=9999,
            warmup=False,
        )

        sentinel.assert_called_once_with(ag)
        # And no real server was bound.
        assert stubbed_start_server.call_count == 0


# ---------------------------------------------------------------------------
# serve() — teardown-suppression contract
# ---------------------------------------------------------------------------
class TestServeTeardownSuppression:
    """The known mcp SDK bug is suppressed inside ``serve()``.

    The walker (:func:`is_mcp_teardown_exception`) inspects the error
    string for ``"cancel scope"`` to avoid swallowing unrelated
    RuntimeError s. These tests lock that contract.
    """

    def test_known_teardown_error_is_swallowed(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When ``asyncio.run`` raises the documented mcp teardown
        pattern, ``serve`` must NOT re-raise."""

        async def _exploding_serve_async(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(
                "Attempted to exit cancel scope in a different task than it was entered in",
            )

        def _fake_asyncio_run(coro: Any, *_a: Any, **_kw: Any) -> Any:
            # ``asyncio.run`` returns the result, but it *also* surfaces
            # the coroutine's exceptions synchronously. We do the same
            # by draining the coroutine ourselves via a fresh loop.
            import asyncio as _a

            loop = _a.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

        monkeypatch.setattr(agent_serve_module, "_serve_async", _exploding_serve_async)
        monkeypatch.setattr(agent_serve_module.asyncio, "run", _fake_asyncio_run)

        # Public API: must not raise.
        ag = _build_agent()
        agent_serve_module.serve(ag, host="127.0.0.1", port=9999)

    def test_unrelated_runtime_error_propagates(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A real RuntimeError with no cancel-scope text must propagate."""

        async def _exploding_serve_async(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("totally unrelated programmer error")

        def _fake_asyncio_run(coro: Any, *_a: Any, **_kw: Any) -> Any:
            import asyncio as _a

            loop = _a.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

        monkeypatch.setattr(agent_serve_module, "_serve_async", _exploding_serve_async)
        monkeypatch.setattr(agent_serve_module.asyncio, "run", _fake_asyncio_run)

        ag = _build_agent()
        with pytest.raises(RuntimeError, match="totally unrelated"):
            agent_serve_module.serve(ag, host="127.0.0.1", port=9999)

    def test_keyboard_interrupt_propagates(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Ctrl+C must cleanly stop the server.

        ``KeyboardInterrupt`` is ``BaseException``, not ``Exception``,
        so it bypasses the ``except Exception`` block. Lock this so
        nobody broadens the catch by accident.
        """

        async def _exploding_serve_async(*_args: Any, **_kwargs: Any) -> None:
            raise KeyboardInterrupt()

        def _fake_asyncio_run(coro: Any, *_a: Any, **_kw: Any) -> Any:
            import asyncio as _a

            loop = _a.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

        monkeypatch.setattr(agent_serve_module, "_serve_async", _exploding_serve_async)
        monkeypatch.setattr(agent_serve_module.asyncio, "run", _fake_asyncio_run)

        ag = _build_agent()
        with pytest.raises(KeyboardInterrupt):
            agent_serve_module.serve(ag, host="127.0.0.1", port=9999)
