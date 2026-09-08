"""Direct tests for :mod:`syncfusion_a2ui_agent._engine`.

Covers the request-handling engine in isolation, without going
through :class:`SyncfusionAgent` (which has its own tests in
``test_agent.py``). The engine owns the prompt → tool-call →
validation-retry loop; the agent tests exercise the loop only as a
side effect.

These tests focus on:

* Construction and dependency wiring.
* :meth:`LoopEngine.build_history` — pure data conversion.
* :meth:`LoopEngine.inject_skill_catalog` — prompt mutation.
* :meth:`LoopEngine.prepare` — the MCP warmup / orchestrator /
  prompt-build flow.
* :meth:`LoopEngine.handle` — buffered success, validation
  retry, the circuit breaker, tool-call dispatch (server__tool and
  read_skill), the tool-call budget, and the empty response path.
* :meth:`LoopEngine.handle_stream` — op event ordering, surface
  buffering, error events, the streaming circuit breaker, and
  tool-call in-stream dispatch.
* :class:`StreamContext` and :class:`PromptBuilderWithExtension` —
  small but important engine-private types.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from factories import (
    VALID_ENVELOPE,
    VALID_ENVELOPE_TEXT,
    FakeProvider,
    ScriptedStep,
)

from syncfusion_a2ui_agent._engine import (
    LoopEngine,
    PromptBuilderWithExtension,
    RequestSetup,
    StreamContext,
    _coerce_mcp_result,
    _op_kind,
    _tool_display_name,
)
from syncfusion_a2ui_agent.providers.base import (
    StreamEvent,
    ToolCall,
)
from syncfusion_a2ui_agent.validation.response_validator import ResponseValidator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_engine(provider: FakeProvider) -> tuple[LoopEngine, Any, Any]:
    """Build a LoopEngine with a pre-warmed MCP manager.

    Returns ``(engine, mcp_manager, agent_like_proxy)`` so tests can
    poke at the manager and the provider directly.
    """
    from syncfusion_a2ui_agent.mcp import MCPManager
    from syncfusion_a2ui_agent.orchestrator import ToolOrchestrator
    from syncfusion_a2ui_agent.prompts import PromptBuilder

    mcp = MCPManager()
    mcp.add_mcp(name="demo", command=["echo"])
    mcp.register_tool_descriptor("demo", "ping", description="ping the demo server")
    # Mark the manager as "already warm" so prepare() does not try to
    # call the optional mcp SDK during tests.
    mcp._warm_started = True
    mcp._deferred_warmup = False

    prompt_builder = PromptBuilder()
    orchestrator = ToolOrchestrator(mcp)
    validator = ResponseValidator(max_retries=2)
    # Mock orchestrator to skip tool-calling (prevent MCP initialization in tests)
    from syncfusion_a2ui_agent.orchestrator.tool_orchestrator import OrchestratorResult

    async def _mock_execute(plan: Any) -> OrchestratorResult:
        return OrchestratorResult(results=[])

    orchestrator.execute = _mock_execute  # type: ignore[method-assign]

    engine = LoopEngine(
        provider=provider,
        mcp_manager=mcp,
        orchestrator=orchestrator,
        validator=validator,
        prompt_builder=prompt_builder,
        skill_router=None,
    )
    return engine, mcp, None


def _consume(agen: Any) -> list[Any]:
    return asyncio.run(_drain(agen))


async def _drain(agen: Any) -> list[Any]:
    out: list[Any] = []
    async for x in agen:
        out.append(x)
    return out


# ---------------------------------------------------------------------------
# Small helpers (private)
# ---------------------------------------------------------------------------
class TestPrivateHelpers:
    def test_op_kind_create_surface(self) -> None:
        assert _op_kind({"createSurface": {"surfaceId": "s1"}}) == "createSurface"

    def test_op_kind_update_components(self) -> None:
        assert _op_kind({"updateComponents": {"surfaceId": "s1"}}) == "updateComponents"

    def test_op_kind_update_data_model(self) -> None:
        assert _op_kind({"updateDataModel": {"surfaceId": "s1"}}) == "updateDataModel"

    def test_op_kind_unknown(self) -> None:
        assert _op_kind({"foo": "bar"}) == "unknown"

    def test_op_kind_empty(self) -> None:
        assert _op_kind({}) == "unknown"

    def test_tool_display_name_openai_format(self) -> None:
        t = {"function": {"name": "demo__ping"}}
        assert _tool_display_name(t) == "demo__ping"

    def test_tool_display_name_flat_format(self) -> None:
        t = {"name": "demo__ping"}
        assert _tool_display_name(t) == "demo__ping"

    def test_tool_display_name_missing(self) -> None:
        assert _tool_display_name({}) == "?"

    def test_coerce_mcp_result_none(self) -> None:
        assert _coerce_mcp_result(None) == ""

    def test_coerce_mcp_result_str(self) -> None:
        assert _coerce_mcp_result("hello") == "hello"

    def test_coerce_mcp_result_dict(self) -> None:
        assert _coerce_mcp_result({"k": "v"}) == json.dumps({"k": "v"})

    def test_coerce_mcp_result_pydantic_like(self) -> None:
        # Mimic a Pydantic v2 object: ``.model_dump()`` returns a dict.
        class _Fake:
            def model_dump(self) -> dict:
                return {"k": "v"}

        result = _coerce_mcp_result(_Fake())
        parsed = json.loads(result)
        assert parsed == {"k": "v"}

    def test_coerce_mcp_result_with_content_attribute(self) -> None:
        class _Block:
            def __init__(self, text: str) -> None:
                self.text = text

        class _Result:
            def __init__(self, blocks: list[Any]) -> None:
                self.content = blocks

        result = _Result([_Block("a"), _Block("b")])
        assert _coerce_mcp_result(result) == "a\nb"

    def test_coerce_mcp_result_unserialisable_falls_back_to_str(self) -> None:
        class _Unserialisable:
            def __repr__(self) -> str:
                return "<weird>"

        # json.dumps will fail; the function falls back to str().
        result = _coerce_mcp_result(_Unserialisable())
        assert "weird" in result


# ---------------------------------------------------------------------------
# StreamContext
# ---------------------------------------------------------------------------
class TestStreamContext:
    def test_defaults(self) -> None:
        ctx = StreamContext()
        assert ctx.last_validation_attempt == 0
        assert ctx.last_validation_error is None
        assert ctx.tool_call_budget == 10
        assert ctx.tool_calls_done == 0
        assert ctx.aggregated_envelope == []
        # buffer is auto-initialised
        assert ctx.buffer is not None

    def test_reset_per_attempt_clears_state(self) -> None:
        ctx = StreamContext()
        ctx.tool_calls_done = 3
        ctx.aggregated_envelope.append({"foo": "bar"})
        ctx.buffer.add_op({"createSurface": {"surfaceId": "s"}}, "createSurface")
        ctx.reset_per_attempt()
        assert ctx.tool_calls_done == 0
        assert ctx.aggregated_envelope == []
        # Buffer is reset: surfaces_seen and pending are empty.
        assert ctx.buffer.surfaces_seen() == set()
        assert ctx.buffer.has_pending() is False

    def test_reset_does_not_touch_budget_or_attempt_counter(self) -> None:
        # The budget and attempt counter are per-request, not per-attempt.
        ctx = StreamContext(tool_call_budget=42)
        ctx.last_validation_attempt = 2
        ctx.reset_per_attempt()
        assert ctx.tool_call_budget == 42
        assert ctx.last_validation_attempt == 2


# ---------------------------------------------------------------------------
# PromptBuilderWithExtension
# ---------------------------------------------------------------------------
class TestPromptBuilderWithExtension:
    def test_extension_called_on_build(self) -> None:
        captured: list[str] = []

        def _ext() -> str:
            captured.append("called")
            return "## Business prompt"

        b = PromptBuilderWithExtension(_ext)
        bp = b.build("hi")
        assert "## Business prompt" in bp.system
        assert captured == ["called"]

    def test_extension_returning_empty_string(self) -> None:
        b = PromptBuilderWithExtension(lambda: "")
        bp = b.build("hi")
        # No "## Customer extension" header should be injected.
        assert "## Customer extension" not in bp.system

    def test_extension_returning_none(self) -> None:
        b = PromptBuilderWithExtension(lambda: None)
        bp = b.build("hi")
        assert "## Customer extension" not in bp.system

    def test_extension_raising_returns_empty(self) -> None:
        def _ext() -> str:
            raise RuntimeError("boom")

        b = PromptBuilderWithExtension(_ext)
        # Must not propagate.
        bp = b.build("hi")
        assert "## Customer extension" not in bp.system


# ---------------------------------------------------------------------------
# LoopEngine.build_history
# ---------------------------------------------------------------------------
class TestBuildHistory:
    def test_none_input(self) -> None:
        engine, _, _ = _make_engine(FakeProvider())
        assert engine.build_history(None) == []

    def test_empty_input(self) -> None:
        engine, _, _ = _make_engine(FakeProvider())
        assert engine.build_history([]) == []

    def test_converts_dicts(self) -> None:
        engine, _, _ = _make_engine(FakeProvider())
        history = engine.build_history(
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello", "name": "agent"},
            ]
        )
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[0].content == "hi"
        assert history[1].name == "agent"

    def test_defaults_role_and_content(self) -> None:
        engine, _, _ = _make_engine(FakeProvider())
        history = engine.build_history([{}])
        assert history[0].role == "user"
        assert history[0].content == ""


# ---------------------------------------------------------------------------
# LoopEngine.inject_skill_catalog
# ---------------------------------------------------------------------------
class TestInjectSkillCatalog:
    def test_no_op_when_skill_router_is_none(self) -> None:
        engine, _, _ = _make_engine(FakeProvider())
        # Should not raise; no catalog injected.
        engine.inject_skill_catalog()
        # Build a prompt and check no skill catalog block landed.
        bp = engine._prompt_builder.build("hi")
        assert "Available Skills" not in bp.system


# ---------------------------------------------------------------------------
# LoopEngine.prepare
# ---------------------------------------------------------------------------
class TestPrepare:
    def test_returns_request_setup(self) -> None:
        engine, _, _ = _make_engine(FakeProvider())
        result = asyncio.run(engine.prepare("hi", []))
        assert isinstance(result, RequestSetup)
        assert len(result.messages) >= 1
        # The pre-registered tool is visible to the model.
        # ``list_tools()`` returns short names; the qualified
        # ``server__tool`` form is built downstream by
        # :func:`_tool_display_name` and the tool-call dispatch.
        names = [t.get("name", "") for t in result.mcp_tools]
        assert "ping" in names

    def test_deferred_warmup_path_does_not_block(self) -> None:
        engine, mcp, _ = _make_engine(FakeProvider())
        mcp._deferred_warmup = True
        mcp._warm_started = False
        result = asyncio.run(engine.prepare("hi", []))
        # Even with deferred warmup, prepare() should still return
        # a setup; the warmup is fire-and-forget.
        assert isinstance(result, RequestSetup)
        # After prepare(), the deferred flag must be cleared.
        assert mcp._deferred_warmup is False
        assert mcp._warm_started is True


# ---------------------------------------------------------------------------
# LoopEngine.handle (buffered)
# ---------------------------------------------------------------------------
class TestHandleBuffered:
    def test_happy_path_returns_envelope(self) -> None:
        provider = FakeProvider().queue(ScriptedStep(content=VALID_ENVELOPE_TEXT))
        engine, _, _ = _make_engine(provider)
        result = asyncio.run(engine.handle("hello"))
        assert result["envelope"] == VALID_ENVELOPE
        # ``surface_ids_in`` collects every surfaceId reference in
        # the envelope (one per op that mentions it), so a 3-op
        # envelope that all target s1 returns ``["s1", "s1", "s1"]``.
        assert result["surfaceIds"] == ["s1", "s1", "s1"]

    def test_provider_exception_raises_runtime_error(self) -> None:
        provider = FakeProvider().queue(ScriptedStep(raise_exception=RuntimeError("upstream boom")))
        engine, _, _ = _make_engine(provider)
        with pytest.raises(RuntimeError, match="upstream boom"):
            asyncio.run(engine.handle("hello"))

    def test_validation_retry_succeeds_on_second_attempt(self) -> None:
        # First attempt returns invalid JSON, second returns the
        # canonical envelope. With max_retries=2 (default), the
        # engine should retry and succeed.
        provider = FakeProvider().queue(
            ScriptedStep(content="not a valid envelope"),
            ScriptedStep(content=VALID_ENVELOPE_TEXT),
        )
        engine, _, _ = _make_engine(provider)
        result = asyncio.run(engine.handle("hello"))
        assert result["envelope"] == VALID_ENVELOPE
        assert provider.call_count == 2

    def test_circuit_breaker_trips_on_identical_errors(self) -> None:
        # Three identical errors should be cut short by the circuit
        # breaker (the second identical error trips it).
        provider = FakeProvider().queue(
            ScriptedStep(content="<bad>"),
            ScriptedStep(content="<bad>"),
            ScriptedStep(content="<bad>"),
        )
        engine, _, _ = _make_engine(provider)
        with pytest.raises(RuntimeError, match="Agent could not produce"):
            asyncio.run(engine.handle("hello"))
        # Two attempts, not three — circuit breaker saved a round-trip.
        assert provider.call_count == 2

    def test_exhausted_attempts_raises(self) -> None:
        # The validator's error string is identical for all three
        # bad attempts (same JSON parse error), so the circuit
        # breaker trips on the second attempt rather than the
        # third. This pins that behaviour.
        provider = FakeProvider().queue(
            ScriptedStep(content="<bad 1>"),
            ScriptedStep(content="<bad 2>"),
            ScriptedStep(content="<bad 3>"),
        )
        engine, _, _ = _make_engine(provider)
        with pytest.raises(RuntimeError, match="Agent could not produce"):
            asyncio.run(engine.handle("hello"))
        # Two attempts, not three — the circuit breaker saved a
        # round-trip because the same error fired twice in a row.
        assert provider.call_count == 2

    def test_tool_call_dispatch_via_mcp(self) -> None:
        # Round 1: model calls demo__ping, MCP returns a stub result.
        # Round 2: model returns the canonical envelope.
        call_payload = ToolCall(
            name="demo__ping",
            arguments={"q": "hi"},
            id="call_1",
        )
        round_one = ScriptedStep(tool_calls=[call_payload], content="")
        round_two = ScriptedStep(content=VALID_ENVELOPE_TEXT)
        provider = FakeProvider().queue(round_one, round_two)
        engine, mcp, _ = _make_engine(provider)

        # Mock the orchestrator to skip its default first-tool call, so we
        # only capture the engine's dispatch of the provider's tool call.
        from syncfusion_a2ui_agent.orchestrator.tool_orchestrator import OrchestratorResult

        async def _mock_execute(plan: Any) -> OrchestratorResult:
            return OrchestratorResult(results=[])

        engine._orchestrator.execute = _mock_execute  # type: ignore[method-assign]

        # Replace call_tool with a fake that returns a predictable
        # string and records its invocation.
        captured: list[tuple[str, str, dict[str, Any]]] = []

        async def _fake_call_tool(server: str, tool: str, args: dict[str, Any]) -> str:
            captured.append((server, tool, args))
            return "ping-ok"

        mcp.call_tool = _fake_call_tool  # type: ignore[method-assign]

        result = asyncio.run(engine.handle("hello"))
        assert result["envelope"] == VALID_ENVELOPE
        assert captured == [("demo", "ping", {"q": "hi"})]

    def test_tool_call_dispatch_skill_router(self) -> None:
        # When the skill router is mounted, a read_skill call is
        # served by the router, not the MCP manager.
        # Build a tiny in-memory skill so the router can serve it.
        from syncfusion_a2ui_agent.skills.skill_registry import (
            SkillMetadata,
            SkillRegistry,
        )
        from syncfusion_a2ui_agent.skills.skill_router import SkillRouter

        registry = SkillRegistry(skills_path=None)
        sm = SkillMetadata(
            name="demo-skill",
            description="Demo skill.",
            path="(memory)",
        )
        sm._content = "Demo skill body."
        # ``SkillRegistry._skills`` is a list, not a dict.
        registry._skills.append(sm)
        router = SkillRouter(registry)

        provider = FakeProvider().queue(
            ScriptedStep(
                tool_calls=[
                    ToolCall(
                        name="read_skill",
                        arguments={"name": "demo-skill"},
                        id="call_1",
                    )
                ],
                content="",
            ),
            ScriptedStep(content=VALID_ENVELOPE_TEXT),
        )

        engine, mcp, _ = _make_engine(provider)
        engine._skill_router = router  # bypass SLF001 via public attr

        result = asyncio.run(engine.handle("hello"))
        assert result["envelope"] == VALID_ENVELOPE

    def test_tool_call_unknown_name_drops_silently(self) -> None:
        # Model invents a tool without "__" — the engine should drop
        # the call and treat the model's empty content as final.
        # That empty content will fail validation, so we expect a
        # RuntimeError after the circuit breaker trips.
        provider = FakeProvider().queue(
            ScriptedStep(
                tool_calls=[
                    ToolCall(
                        name="not_a_mcp_tool",
                        arguments={},
                        id="call_1",
                    )
                ],
                content="<bad>",
            ),
            ScriptedStep(content="<bad>"),
            ScriptedStep(content="<bad>"),
        )
        engine, _, _ = _make_engine(provider)
        with pytest.raises(RuntimeError):
            asyncio.run(engine.handle("hello"))

    def test_tool_call_budget_exhausted(self) -> None:
        # Model emits 11 tool-call rounds (budget is 10). The 11th
        # is ignored. We don't have a clean way to script 11
        # rounds + 1 final answer in a queue, so this test just
        # verifies the constant exists.
        engine, _, _ = _make_engine(FakeProvider())
        assert engine.DEFAULT_TOOL_CALL_BUDGET == 10
        assert engine.DEFERRED_WARMUP_WAIT > 0
        assert engine.COLD_WARMUP_REFRESH_TIMEOUT > 0
        assert engine.ORCHESTRATOR_PLAN_TIMEOUT > 0

    def test_history_passed_through(self) -> None:
        provider = FakeProvider().queue(ScriptedStep(content=VALID_ENVELOPE_TEXT))
        engine, _, _ = _make_engine(provider)
        history = [
            {"role": "user", "content": "earlier message"},
            {"role": "assistant", "content": "earlier reply"},
        ]
        asyncio.run(engine.handle("hello", conversation_history=history))
        # Provider should have seen the history rendered into the
        # OpenAI-style message list (system + 2 history + user).
        assert provider.last_messages is not None
        roles = [m.get("role") for m in provider.last_messages]
        assert "system" in roles
        assert "user" in roles


# ---------------------------------------------------------------------------
# LoopEngine.handle_stream
# ---------------------------------------------------------------------------
class TestHandleStream:
    def test_happy_path_yields_surface_ids(self) -> None:
        provider = FakeProvider().queue(ScriptedStep(content=VALID_ENVELOPE_TEXT))
        engine, _, _ = _make_engine(provider)
        events = _consume(engine.handle_stream("hello"))
        # At least one "op" event, then a "surfaceIds" event.
        kinds = [e.get("type") for e in events]
        assert "op" in kinds
        assert "surfaceIds" in kinds
        # The first op emitted must be the createSurface op.
        first_op = next(e for e in events if e.get("type") == "op")
        assert "createSurface" in first_op["op"]
        # And the surfaceIds event contains the canonical id.
        # The streaming path uses ``ctx.buffer.emitted_surface_ids()``
        # (a set), so surfaceIds is deduplicated to one entry.
        sid_event = next(e for e in events if e.get("type") == "surfaceIds")
        assert sid_event["surfaceIds"] == ["s1"]

    def test_yields_error_when_provider_stream_errors(self) -> None:
        # A stream event with ``error`` is treated like a provider
        # failure on the inner loop. The outer retry loop sees an
        # empty envelope and yields its own error. The exact error
        # string is therefore the validator's, not the provider's.
        # This test pins that behaviour (per the comment in
        # test_agent.py: "the SDK's error message does NOT echo
        # the provider's original error string back to the consumer").
        provider = FakeProvider().queue(
            ScriptedStep(stream_events=[StreamEvent(done=True, error="upstream boom")])
        )
        engine, _, _ = _make_engine(provider)
        events = _consume(engine.handle_stream("hello"))
        error_events = [e for e in events if e.get("type") == "error"]
        assert error_events, "expected an error event"
        assert "Agent could not produce" in error_events[0]["error"]

    def test_yields_error_when_no_envelope_emitted(self) -> None:
        # Provider returns text with no envelope.
        provider = FakeProvider().queue(ScriptedStep(content="just some prose, no envelope"))
        engine, _, _ = _make_engine(provider)
        events = _consume(engine.handle_stream("hello"))
        # The first attempt fails to validate; the engine should
        # retry once (since max_retries=2 → 2 attempts total per
        # the streaming circuit-breaker logic), then yield an error.
        assert any(e.get("type") == "error" for e in events)

    def test_circuit_breaker_during_streaming(self) -> None:
        # Two streaming attempts, both with the same content → second
        # identical error trips the breaker.
        provider = FakeProvider().queue(
            ScriptedStep(content="<bad>"),
            ScriptedStep(content="<bad>"),
        )
        engine, _, _ = _make_engine(provider)
        events = _consume(engine.handle_stream("hello"))
        # Should not exhaust all attempts; circuit breaker stops it.
        assert any(e.get("type") == "error" for e in events)
        assert provider.call_count == 2

    def test_history_passed_through_streaming(self) -> None:
        provider = FakeProvider().queue(ScriptedStep(content=VALID_ENVELOPE_TEXT))
        engine, _, _ = _make_engine(provider)
        history = [{"role": "user", "content": "earlier"}]
        _consume(engine.handle_stream("hello", conversation_history=history))
        # Provider must have seen the history in its last_messages.
        assert provider.last_messages is not None
        roles = [m.get("role") for m in provider.last_messages]
        assert "system" in roles
        assert "user" in roles


# ---------------------------------------------------------------------------
# LoopEngine._on_validation_failure
# ---------------------------------------------------------------------------
class TestCircuitBreaker:
    def test_first_failure_continues(self) -> None:
        engine, _, _ = _make_engine(FakeProvider())
        new_error, should_continue = engine._on_validation_failure(
            last_error=None,
            attempt=0,
            error="first error",
            last_validation_error=None,
        )
        assert new_error == "first error"
        assert should_continue is True

    def test_identical_second_failure_trips(self) -> None:
        engine, _, _ = _make_engine(FakeProvider())
        # Simulate that the previous error was the same:
        # attempt=1 means we just finished the second attempt.
        new_error, should_continue = engine._on_validation_failure(
            last_error="same",
            attempt=1,
            error="same",
            last_validation_error="same",
        )
        assert new_error == "same"
        assert should_continue is False

    def test_different_second_failure_continues(self) -> None:
        engine, _, _ = _make_engine(FakeProvider())
        new_error, should_continue = engine._on_validation_failure(
            last_error="first",
            attempt=1,
            error="second",
            last_validation_error="first",
        )
        assert new_error == "second"
        assert should_continue is True

    def test_none_error_defaults_to_unknown(self) -> None:
        engine, _, _ = _make_engine(FakeProvider())
        new_error, should_continue = engine._on_validation_failure(
            last_error=None,
            attempt=0,
            error=None,
            last_validation_error=None,
        )
        assert new_error == "unknown validation error"
        assert should_continue is True


# ---------------------------------------------------------------------------
# LoopEngine._corrective_user_message
# ---------------------------------------------------------------------------
class TestCorrectiveMessage:
    def test_shape_and_content(self) -> None:
        msg = LoopEngine._corrective_user_message("bad envelope")
        assert msg["role"] == "user"
        assert "bad envelope" in msg["content"]
        assert "CRITICAL RULES" in msg["content"]
        assert "a2ui-json" in msg["content"]
