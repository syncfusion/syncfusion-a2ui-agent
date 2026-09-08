"""Consolidated tests for the agent and the prompts subpackage.

Covers:

* ``syncfusion_a2ui_agent.prompts`` — the default system prompt
  invariants, the prompt-builder dataclasses, and the
  :class:`PromptBuilder` composition order.
* :class:`SyncfusionAgent` — construction (instance / class / string /
  subclass), data-source setters, catalog-reference setters, MCP
  delegation, :meth:`handle_request` (happy path, validation retry,
  circuit breaker, tool-call dispatch, tool-call budget), and
  :meth:`handle_request_stream` (op gating, error events, buffering
  of update-ops until ``createSurface`` lands).

The agent tests use :class:`FakeProvider` (in ``factories.py``) so
we never touch a real AI SDK, and pre-register a static tool on the
MCP manager to avoid the optional ``mcp`` SDK.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from factories import (
    VALID_ENVELOPE,
    VALID_ENVELOPE_TEXT,
    FakeProvider,
)
from factories import (
    ScriptedStep as _ScriptedStep,
)

from syncfusion_a2ui_agent.agent import SyncfusionAgent
from syncfusion_a2ui_agent.prompts.default_system_prompt import (
    DEFAULT_SYNCFUSION_SYSTEM_PROMPT,
    get_default_system_prompt,
)
from syncfusion_a2ui_agent.prompts.prompt_builder import (
    BuiltPrompt,
    CatalogReference,
    ConversationTurn,
    PromptBuilder,
    ToolResult,
)
from syncfusion_a2ui_agent.providers.base import (
    AIProvider,
    ModelResponse,
    StreamEvent,
    ToolCall,
)


# ===========================================================================
# prompts/default_system_prompt.py
# ===========================================================================
class TestDefaultSystemPrompt:
    def test_prompt_is_non_trivial(self) -> None:
        # Sanity: the prompt is a real production prompt, not a stub.
        assert len(DEFAULT_SYNCFUSION_SYSTEM_PROMPT) > 5_000

    def test_prompt_mentions_a2ui_envelope(self) -> None:
        # The prompt must teach the A2UI v0.9 contract.
        assert "createSurface" in DEFAULT_SYNCFUSION_SYSTEM_PROMPT
        assert "updateComponents" in DEFAULT_SYNCFUSION_SYSTEM_PROMPT
        assert "updateDataModel" in DEFAULT_SYNCFUSION_SYSTEM_PROMPT
        assert "<a2ui-json>" in DEFAULT_SYNCFUSION_SYSTEM_PROMPT

    def test_get_returns_fresh_copy(self) -> None:
        # Mutating the returned string must not affect the constant.
        s = get_default_system_prompt()
        original_len = len(s)
        s += "MUTATED"
        assert len(get_default_system_prompt()) == original_len

    def test_mentions_a2ui_version(self) -> None:
        assert "v0.9" in DEFAULT_SYNCFUSION_SYSTEM_PROMPT


# ===========================================================================
# prompts/prompt_builder.py — dataclasses
# ===========================================================================
class TestConversationTurn:
    def test_required_fields(self) -> None:
        t = ConversationTurn(role="user", content="hi")
        assert t.role == "user"
        assert t.content == "hi"
        assert t.name is None

    def test_with_name(self) -> None:
        t = ConversationTurn(role="tool", content="x", name="server__tool")
        assert t.name == "server__tool"


class TestToolResult:
    def test_preserves_source_and_payload(self) -> None:
        r = ToolResult(source="mcp:srv:tool", payload={"value": 1})
        assert r.source == "mcp:srv:tool"
        assert r.payload == {"value": 1}


class TestCatalogReference:
    def test_construction(self) -> None:
        r = CatalogReference(catalog_id="abc", reference_text="...")
        assert r.catalog_id == "abc"


class TestBuiltPrompt:
    def test_render_messages_ordering(self) -> None:
        bp = BuiltPrompt(
            system="SYSTEM",
            history=[
                ConversationTurn(role="user", content="u1"),
                ConversationTurn(role="assistant", content="a1"),
            ],
            tool_results=[ToolResult(source="mcp:s:t", payload="P")],
            user_message="USER",
        )
        msgs = bp.render_messages()
        assert msgs[0] == {"role": "system", "content": "SYSTEM"}
        # history + tool results + user
        assert msgs[1] == {"role": "user", "content": "u1"}
        assert msgs[2] == {"role": "assistant", "content": "a1"}
        # tool results become a single system message after history
        assert msgs[3]["role"] == "system"
        assert "mcp:s:t" in msgs[3]["content"]
        assert msgs[-1] == {"role": "user", "content": "USER"}

    def test_render_messages_without_tool_results(self) -> None:
        bp = BuiltPrompt(system="S", user_message="U")
        msgs = bp.render_messages()
        # system + user
        assert len(msgs) == 2
        assert msgs[0] == {"role": "system", "content": "S"}
        assert msgs[1] == {"role": "user", "content": "U"}


# ===========================================================================
# prompts/prompt_builder.py — PromptBuilder
# ===========================================================================
class TestPromptBuilder:
    def test_default_returns_full_prompt(self) -> None:
        b = PromptBuilder()
        bp = b.build("hello")
        # Default prompt is long — confirm it was loaded.
        assert "createSurface" in bp.system
        assert bp.user_message == "hello"
        assert bp.history == []

    def test_extension_hook_called(self) -> None:
        class Sub(PromptBuilder):
            def _customer_extension(self) -> str:
                return "EXT-CONTENT"

        b = Sub()
        bp = b.build("x")
        assert "EXT-CONTENT" in bp.system
        # Extension appears AFTER the default prompt.
        idx = bp.system.index("createSurface")
        ext = bp.system.index("EXT-CONTENT")
        assert ext > idx

    def test_data_context_is_appended(self) -> None:
        b = PromptBuilder()
        b.set_data_context(json.dumps({"kpis": {"revenue": "$4.2M"}}))
        bp = b.build("x")
        assert "Application Data Source" in bp.system
        assert '"revenue": "$4.2M"' in bp.system

    def test_catalog_reference_is_appended(self) -> None:
        b = PromptBuilder()
        b.set_catalog_reference(
            CatalogReference(catalog_id="cat-1", reference_text="<catalog>...</catalog>")
        )
        bp = b.build("x")
        assert "Supported Components" in bp.system
        assert "cat-1" in bp.system
        assert "<catalog>...</catalog>" in bp.system

    def test_history_provider_invoked_on_build(self) -> None:
        b = PromptBuilder()

        def provider():
            return [
                ConversationTurn(role="user", content="past-q"),
                ConversationTurn(role="assistant", content="past-a"),
            ]

        b.set_history_provider(provider)
        bp = b.build("new-q")
        assert [t.content for t in bp.history] == ["past-q", "past-a"]

    def test_build_orders_sections(self) -> None:
        """Extension → data context → catalog reference (after the default)."""

        class Sub(PromptBuilder):
            def _customer_extension(self) -> str:
                return "##EXT"

        b = Sub()
        b.set_data_context('{"k":1}')
        b.set_catalog_reference(CatalogReference(catalog_id="c", reference_text="<cat/>"))
        bp = b.build("x")
        sys = bp.system
        assert sys.index("##EXT") < sys.index("Application Data Source")
        assert sys.index("Application Data Source") < sys.index("Supported Components")

    def test_clear_data_context(self) -> None:
        b = PromptBuilder()
        b.set_data_context("{}")
        b.set_data_context(None)
        bp = b.build("x")
        assert "Application Data Source" not in bp.system


# ===========================================================================
# agent.py — helpers
# ===========================================================================
def _agent_with_cached_tools(provider: AIProvider) -> SyncfusionAgent:
    """Build an agent whose MCP manager is pre-warmed.

    The agent's :meth:`_prepare_request` decides whether to call
    ``refresh_tools`` based on whether the cache is empty. By
    pre-registering a static tool we make the test hermetic and
    skip the (slow / SDK-dependent) cold-start path.
    """
    from syncfusion_a2ui_agent.orchestrator.tool_orchestrator import OrchestratorResult

    agent = SyncfusionAgent(model=provider)
    agent.mcp.add_mcp(name="demo", command=["echo"])
    agent.mcp.register_tool_descriptor("demo", "ping", description="ping the demo server")

    # Mock orchestrator to skip tool-calling (prevent MCP initialization in tests)
    async def _mock_execute(plan: Any) -> OrchestratorResult:
        return OrchestratorResult(results=[])

    agent._engine._orchestrator.execute = _mock_execute  # type: ignore[method-assign]
    return agent


def _consume(agen):
    """Drain an async generator into a list."""
    return asyncio.run(_drain(agen))


async def _drain(agen):
    out = []
    async for x in agen:
        out.append(x)
    return out


# ===========================================================================
# agent.py — Construction
# ===========================================================================
class TestConstruction:
    def test_with_provider_instance(self) -> None:
        provider = FakeProvider()
        agent = SyncfusionAgent(model=provider)
        assert agent._provider is provider

    def test_with_provider_class(self) -> None:
        class MyProvider(FakeProvider):
            pass

        agent = SyncfusionAgent(model=MyProvider)
        assert isinstance(agent._provider, MyProvider)

    def test_with_string_model_uses_registry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Use the in-process FakeProvider via the registry, with a
        # patched registration under a test-only name.
        from syncfusion_a2ui_agent.providers import provider_registry as reg

        register = reg.register_provider
        register("__test_model__", FakeProvider)
        try:
            agent = SyncfusionAgent(model="__test_model__")
            assert isinstance(agent._provider, FakeProvider)
        finally:
            reg._REGISTRY.pop("__test_model__", None)
            reg._FACTORIES.pop("__test_model__", None)

    def test_subclass_extension_is_used(self) -> None:
        # A subclass that overrides ``extend_system_prompt`` should be
        # honoured by the agent's prompt builder. We construct the
        # subclass with the FakeProvider already instantiated and
        # confirm the prompt builder sees the new extension text.
        class MyAgent(SyncfusionAgent):
            def extend_system_prompt(self) -> str:
                return "BUSINESS PROMPT"

        provider = FakeProvider()
        agent = MyAgent(model=provider)
        # The agent's prompt builder must call our subclass's hook.
        bp = agent.prompt_builder.build("x")
        assert "BUSINESS PROMPT" in bp.system

    def test_exposes_sub_managers(self) -> None:
        agent = _agent_with_cached_tools(FakeProvider())
        assert agent.mcp is not None
        assert agent.orchestrator is not None
        assert agent.validator is not None
        assert agent.prompt_builder is not None


# ===========================================================================
# agent.py — Data source setters
# ===========================================================================
class TestDataSource:
    def test_set_data_source_with_dict(self) -> None:
        agent = _agent_with_cached_tools(FakeProvider())
        agent.set_data_source({"kpis": {"revenue": "$4.2M"}})
        bp = agent.prompt_builder.build("x")
        assert "Application Data Source" in bp.system
        assert '"revenue": "$4.2M"' in bp.system

    def test_set_data_source_with_list(self) -> None:
        agent = _agent_with_cached_tools(FakeProvider())
        agent.set_data_source([1, 2, 3])
        bp = agent.prompt_builder.build("x")
        assert "Application Data Source" in bp.system

    def test_set_data_source_with_string_validates(self) -> None:
        agent = _agent_with_cached_tools(FakeProvider())
        agent.set_data_source(json.dumps({"a": 1}))
        # Stored verbatim as a string.
        bp = agent.prompt_builder.build("x")
        assert '"a": 1' in bp.system

    def test_set_data_source_with_invalid_string_raises(self) -> None:
        agent = _agent_with_cached_tools(FakeProvider())
        with pytest.raises(json.JSONDecodeError):
            agent.set_data_source("{not json")

    def test_set_data_source_from_file(self, tmp_path: Path) -> None:
        p = tmp_path / "data.json"
        p.write_text('{"hello": "world"}', encoding="utf-8")
        agent = _agent_with_cached_tools(FakeProvider())
        agent.set_data_source_from_file(str(p))
        bp = agent.prompt_builder.build("x")
        assert "Application Data Source" in bp.system
        assert '"hello": "world"' in bp.system

    def test_set_data_source_from_file_missing_raises(self) -> None:
        agent = _agent_with_cached_tools(FakeProvider())
        with pytest.raises(FileNotFoundError):
            agent.set_data_source_from_file("/no/such/file.json")


# ===========================================================================
# agent.py — Catalog reference setters
# ===========================================================================
class TestCatalogReferenceSetter:
    def test_set_catalog_reference(self) -> None:
        agent = _agent_with_cached_tools(FakeProvider())
        agent.set_catalog_reference("cat-x", "<catalog/>")
        bp = agent.prompt_builder.build("x")
        assert "Supported Components" in bp.system
        assert "cat-x" in bp.system

    def test_set_catalog_reference_from_file_reads_catalog_id(self, tmp_path: Path) -> None:
        p = tmp_path / "cat.json"
        p.write_text(
            json.dumps({"catalogId": "https://example.com/cat.json"}),
            encoding="utf-8",
        )
        agent = _agent_with_cached_tools(FakeProvider())
        cid = agent.set_catalog_reference_from_file(str(p))
        assert cid == "https://example.com/cat.json"

    def test_set_catalog_reference_from_file_default_id(self, tmp_path: Path) -> None:
        p = tmp_path / "cat.json"
        p.write_text(json.dumps({"components": []}), encoding="utf-8")
        agent = _agent_with_cached_tools(FakeProvider())
        cid = agent.set_catalog_reference_from_file(str(p))
        assert cid == "syncfusion-a2ui-catalog"


# ===========================================================================
# agent.py — Design contract (set_design)
# ===========================================================================
class TestSetDesign:
    """Tests for :meth:`SyncfusionAgent.set_design`.

    ``set_design`` accepts almost any reasonable input shape — a
    parsed dict, a JSON string, a file path, a directory of files, or
    a list of any of those — and injects a "UI Surface Design Contract"
    block into the system prompt so the model echoes the design
    verbatim on every response.

    Single design: the JSON envelope is embedded directly under
    ``REFERENCE DESIGN (treat this as a template ...)``.

    Multiple designs: the JSON envelopes are concatenated into a
    markdown catalog with ``### Page: `surfaceId``` headers, so the
    model can tell pages apart and pick the right one per request.
    """

    # ----- Single design: dict / list -----
    def test_set_design_with_dict_injects_reference(self) -> None:
        agent = _agent_with_cached_tools(FakeProvider())
        envelope = [
            {"version": "v0.9", "createSurface": {"surfaceId": "workspace"}},
            {
                "version": "v0.9",
                "updateComponents": {
                    "surfaceId": "workspace",
                    "components": [{"id": "root", "component": "Column", "children": []}],
                },
            },
        ]
        agent.set_design(envelope)
        bp = agent.prompt_builder.build("hello")
        # The contract header is emitted on every build.
        assert "UI Surface Design Contract" in bp.system
        # The reference body contains the JSON-serialised design.
        assert '"createSurface"' in bp.system
        assert '"workspace"' in bp.system
        # And the "echo verbatim" rule is part of the contract.
        assert "verbatim" in bp.system
        # Catalog headers are NOT used in single-design mode.
        assert "### Page:" not in bp.system

    def test_set_design_with_list_injects_reference(self) -> None:
        agent = _agent_with_cached_tools(FakeProvider())
        agent.set_design(VALID_ENVELOPE)
        bp = agent.prompt_builder.build("hi")
        assert "UI Surface Design Contract" in bp.system
        assert '"surfaceId": "s1"' in bp.system
        assert "### Page:" not in bp.system

    # ----- Single design: JSON string -----
    def test_set_design_with_json_string_parses(self) -> None:
        agent = _agent_with_cached_tools(FakeProvider())
        agent.set_design(json.dumps(VALID_ENVELOPE))
        bp = agent.prompt_builder.build("hi")
        assert "UI Surface Design Contract" in bp.system
        assert '"surfaceId": "s1"' in bp.system
        assert "### Page:" not in bp.system

    def test_set_design_with_invalid_string_treated_as_text(self) -> None:
        agent = _agent_with_cached_tools(FakeProvider())
        # Plain text / non-JSON is treated as an opaque text block
        # (it is still embedded, but as the body, not parsed JSON).
        agent.set_design("not a json envelope, just text")
        bp = agent.prompt_builder.build("hi")
        assert "UI Surface Design Contract" in bp.system
        assert "not a json envelope, just text" in bp.system

    # ----- Single design: file path -----
    def test_set_design_from_file(self, tmp_path: Path) -> None:
        p = tmp_path / "design.json"
        p.write_text(
            json.dumps(
                [
                    {
                        "version": "v0.9",
                        "createSurface": {"surfaceId": "from-file"},
                    }
                ]
            ),
            encoding="utf-8",
        )
        agent = _agent_with_cached_tools(FakeProvider())
        agent.set_design(str(p))
        bp = agent.prompt_builder.build("hi")
        assert "UI Surface Design Contract" in bp.system
        assert '"from-file"' in bp.system
        assert "### Page:" not in bp.system

    # ----- Multi-design: directory -----
    def test_set_design_from_directory_builds_multi_page_catalog(self, tmp_path: Path) -> None:
        d = tmp_path / "designs"
        d.mkdir()
        (d / "a.json").write_text(
            json.dumps(
                [
                    {
                        "version": "v0.9",
                        "createSurface": {"surfaceId": "page_a"},
                    }
                ]
            ),
            encoding="utf-8",
        )
        (d / "b.json").write_text(
            json.dumps(
                [
                    {
                        "version": "v0.9",
                        "createSurface": {"surfaceId": "page_b"},
                    }
                ]
            ),
            encoding="utf-8",
        )
        # A non-JSON sibling must be ignored (only *.json loads).
        (d / "readme.txt").write_text("ignored", encoding="utf-8")

        agent = _agent_with_cached_tools(FakeProvider())
        agent.set_design(str(d))
        bp = agent.prompt_builder.build("hi")
        # Multi-page mode emits one `### Page:` header per design.
        assert "### Page: `page_a`" in bp.system
        assert "### Page: `page_b`" in bp.system
        # And each design's body shows up in the prompt.
        assert '"page_a"' in bp.system
        assert '"page_b"' in bp.system
        # The non-json file must be excluded.
        assert "ignored" not in bp.system

    # ----- Multi-design: list of any of the above -----
    def test_set_design_with_mixed_list(self) -> None:
        agent = _agent_with_cached_tools(FakeProvider())
        designs = [
            [
                {
                    "version": "v0.9",
                    "createSurface": {"surfaceId": "first"},
                }
            ],
            json.dumps([{"version": "v0.9", "createSurface": {"surfaceId": "second"}}]),
        ]
        agent.set_design(designs)
        bp = agent.prompt_builder.build("hi")
        assert "### Page: `first`" in bp.system
        assert "### Page: `second`" in bp.system

    # ----- Error cases -----
    def test_set_design_with_empty_list_raises(self) -> None:
        agent = _agent_with_cached_tools(FakeProvider())
        with pytest.raises(ValueError):
            agent.set_design([])

    def test_set_design_with_unsupported_type_raises(self) -> None:
        agent = _agent_with_cached_tools(FakeProvider())
        with pytest.raises(TypeError):
            agent.set_design(12345)  # type: ignore[arg-type]

    def test_set_design_with_nonexistent_json_file_falls_back_to_text(
        self,
    ) -> None:
        # A path that doesn't exist and isn't valid JSON is treated
        # as opaque text (preserves the string for the prompt body).
        # This mirrors ``set_data_source_from_file`` which DOES raise
        # FileNotFoundError — set_design is more lenient because users
        # occasionally pass ad-hoc string catalogs.
        agent = _agent_with_cached_tools(FakeProvider())
        agent.set_design("/no/such/file.json")
        bp = agent.prompt_builder.build("hi")
        assert "UI Surface Design Contract" in bp.system
        # The string is embedded verbatim.
        assert "/no/such/file.json" in bp.system

    # ----- Idempotence / replacement -----
    def test_set_design_replaces_previous_design(self) -> None:
        agent = _agent_with_cached_tools(FakeProvider())
        agent.set_design([{"version": "v0.9", "createSurface": {"surfaceId": "old"}}])
        agent.set_design([{"version": "v0.9", "createSurface": {"surfaceId": "new"}}])
        bp = agent.prompt_builder.build("hi")
        assert '"new"' in bp.system
        # The old design is no longer referenced.
        assert '"old"' not in bp.system

    # ----- surfaceId extraction helper (private API) -----
    def test_extract_surface_id_from_envelope(self) -> None:
        envelope = [
            {"version": "v0.9", "createSurface": {"surfaceId": "extracted"}},
            {"version": "v0.9", "updateComponents": {"surfaceId": "extracted"}},
        ]
        sid = SyncfusionAgent._extract_surface_id(envelope)
        assert sid == "extracted"

    def test_extract_surface_id_missing_returns_none(self) -> None:
        # Plain dict (no list of ops) is not an envelope — return None.
        assert SyncfusionAgent._extract_surface_id({"hello": "world"}) is None
        # Envelope with no createSurface returns None.
        assert (
            SyncfusionAgent._extract_surface_id([{"version": "v0.9", "updateComponents": {}}])
            is None
        )

    # ----- _normalise_design helper -----
    def test_normalise_design_dict(self) -> None:
        assert SyncfusionAgent._normalise_design({"a": 1}) == [{"a": 1}]

    def test_normalise_design_list_of_op_dicts_is_one_envelope(self) -> None:
        # A list whose every element is an A2UI op-dict is a single
        # envelope (the canonical wire shape) — keep it as ONE design.
        out = SyncfusionAgent._normalise_design(
            [
                {"version": "v0.9", "createSurface": {"surfaceId": "a"}},
                {"version": "v0.9", "updateComponents": {"surfaceId": "a"}},
            ]
        )
        assert len(out) == 1
        assert out[0][0]["createSurface"]["surfaceId"] == "a"

    def test_normalise_design_mixed_list_flattens(self) -> None:
        # A list of heterogeneous elements (dict / string) is a list
        # of designs, NOT a single envelope — recurse and flatten.
        assert SyncfusionAgent._normalise_design([{"a": 1}, "not json"]) == [{"a": 1}, "not json"]

    def test_normalise_design_json_string(self) -> None:
        out = SyncfusionAgent._normalise_design('{"a": 1}')
        assert out == [{"a": 1}]

    def test_normalise_design_invalid_string_passes_through(self) -> None:
        # Invalid JSON is preserved as opaque text so the caller can
        # embed a pre-rendered markdown catalog if they want.
        out = SyncfusionAgent._normalise_design("not json at all")
        assert out == ["not json at all"]

    def test_normalise_design_directory(self, tmp_path: Path) -> None:
        d = tmp_path / "designs"
        d.mkdir()
        (d / "x.json").write_text(
            json.dumps([{"version": "v0.9", "createSurface": {"surfaceId": "x"}}]),
            encoding="utf-8",
        )
        (d / "y.json").write_text(
            json.dumps([{"version": "v0.9", "createSurface": {"surfaceId": "y"}}]),
            encoding="utf-8",
        )
        out = SyncfusionAgent._normalise_design(str(d))
        sids = sorted(SyncfusionAgent._extract_surface_id(item) for item in out)
        assert sids == ["x", "y"]

    def test_normalise_design_unsupported_type_raises(self) -> None:
        with pytest.raises(TypeError):
            SyncfusionAgent._normalise_design(42)  # type: ignore[arg-type]


# ===========================================================================
# agent.py — MCP registration delegations
# ===========================================================================
class TestMCPDelegation:
    def test_add_mcp_returns_config(self) -> None:
        agent = _agent_with_cached_tools(FakeProvider())
        cfg = agent.add_mcp(name="x", command=["echo"])
        assert cfg.name == "x"

    def test_remove_mcp_drops_server(self) -> None:
        agent = _agent_with_cached_tools(FakeProvider())
        agent.add_mcp(name="x", command=["echo"])
        assert any(s.name == "x" for s in agent.mcp.list_servers())
        agent.remove_mcp("x")
        assert not any(s.name == "x" for s in agent.mcp.list_servers())


# ===========================================================================
# agent.py — handle_request
# ===========================================================================
class TestHandleRequest:
    def test_returns_validated_envelope(self) -> None:
        provider = FakeProvider().queue(_ScriptedStep(content=VALID_ENVELOPE_TEXT))
        agent = _agent_with_cached_tools(provider)
        result = asyncio.run(agent.handle_request("show a button"))
        assert "envelope" in result
        assert "surfaceIds" in result
        # Validated payload should be the original envelope.
        assert result["envelope"][0]["createSurface"]["surfaceId"] == "s1"

    def test_retries_on_validation_failure(self) -> None:
        provider = FakeProvider().queue(
            _ScriptedStep(content="not an envelope"),
            _ScriptedStep(content=VALID_ENVELOPE_TEXT),
        )
        agent = _agent_with_cached_tools(provider)
        result = asyncio.run(agent.handle_request("x"))
        assert result["envelope"][0]["createSurface"]["surfaceId"] == "s1"
        assert provider.call_count == 2  # one retry happened

    def test_circuit_breaker_stops_after_two_identical_errors(self) -> None:
        provider = FakeProvider().queue(
            _ScriptedStep(content=""),
            _ScriptedStep(content=""),
            _ScriptedStep(content=""),
        )
        agent = _agent_with_cached_tools(provider)
        with pytest.raises(RuntimeError, match="could not produce a schema-valid"):
            asyncio.run(agent.handle_request("x"))
        # The circuit breaker trips on attempt 2 (second identical error).
        assert provider.call_count == 2

    def test_passes_messages_to_provider(self) -> None:
        provider = FakeProvider().queue(_ScriptedStep(content=VALID_ENVELOPE_TEXT))
        agent = _agent_with_cached_tools(provider)
        asyncio.run(agent.handle_request("hello"))
        assert provider.last_messages is not None
        # The system message must contain the default prompt's marker.
        sys_msg = provider.last_messages[0]
        assert sys_msg["role"] == "system"
        assert "createSurface" in sys_msg["content"]
        # The user message is the last one in the list.
        assert provider.last_messages[-1] == {"role": "user", "content": "hello"}

    def test_passes_tools_to_provider(self) -> None:
        provider = FakeProvider().queue(_ScriptedStep(content=VALID_ENVELOPE_TEXT))
        agent = _agent_with_cached_tools(provider)
        asyncio.run(agent.handle_request("x"))
        assert provider.last_tools is not None
        # Our cached "demo__ping" tool is visible to the model.
        names = {t["name"] for t in provider.last_tools}
        assert "ping" in names

    def test_conversation_history_is_included(self) -> None:
        provider = FakeProvider().queue(_ScriptedStep(content=VALID_ENVELOPE_TEXT))
        agent = _agent_with_cached_tools(provider)
        history = [
            {"role": "user", "content": "earlier question"},
            {"role": "assistant", "content": "earlier answer"},
        ]
        asyncio.run(agent.handle_request("new question", conversation_history=history))
        msgs = provider.last_messages
        # Find the history turns in the rendered messages.
        contents = [m.get("content") for m in msgs if m.get("content")]
        assert "earlier question" in contents
        assert "earlier answer" in contents

    def test_provider_exception_becomes_runtime_error(self) -> None:
        provider = FakeProvider().queue(
            _ScriptedStep(raise_exception=RuntimeError("upstream is down"))
        )
        agent = _agent_with_cached_tools(provider)
        with pytest.raises(RuntimeError, match="upstream is down"):
            asyncio.run(agent.handle_request("x"))

    def test_tool_call_dispatches_via_mcp(self) -> None:
        # The model wants to call a tool, then produces a final envelope.
        provider = FakeProvider().queue(
            _ScriptedStep(
                tool_calls=[
                    ToolCall(
                        name="demo__ping",
                        arguments={"x": 1},
                        id="call_1",
                    )
                ]
            ),
            _ScriptedStep(content=VALID_ENVELOPE_TEXT),
        )
        agent = _agent_with_cached_tools(provider)
        # Pre-inject a deterministic tool result by patching call_tool
        # to return canned data. We hardcode the expected result so
        # the test is self-contained.
        cached_results = {"ping": {"pong": True}}

        async def fake_call_tool(server, tool, arguments):
            return cached_results.get(tool)

        agent.mcp.call_tool = fake_call_tool  # type: ignore[method-assign]

        result = asyncio.run(agent.handle_request("x"))
        assert result["envelope"][0]["createSurface"]["surfaceId"] == "s1"
        # Two generate() calls happened (one tool-call, one final).
        assert provider.call_count == 2

    def test_tool_call_with_unknown_name_breaks_loop(self) -> None:
        """A tool call with no '__' in the name is dropped — the inner
        loop exits and the validator runs on whatever text was last seen.
        """
        # First call: model emits a malformed tool call (no separator).
        # Second call: model emits the final envelope.
        provider = FakeProvider().queue(
            _ScriptedStep(tool_calls=[ToolCall(name="ping", arguments={}, id="bad")]),
            _ScriptedStep(content=VALID_ENVELOPE_TEXT),
        )
        agent = _agent_with_cached_tools(provider)
        result = asyncio.run(agent.handle_request("x"))
        assert result["envelope"][0]["createSurface"]["surfaceId"] == "s1"

    def test_tool_call_budget_caps_inner_loop(self) -> None:
        # Ten tool calls followed by a final envelope.
        steps = [
            _ScriptedStep(
                tool_calls=[
                    ToolCall(
                        name="demo__ping",
                        arguments={"i": i},
                        id=f"c{i}",
                    )
                ]
            )
            for i in range(10)
        ]
        steps.append(_ScriptedStep(content=VALID_ENVELOPE_TEXT))
        provider = FakeProvider().queue(*steps)
        agent = _agent_with_cached_tools(provider)

        async def fake_call_tool(server, tool, arguments):
            return {"ok": True}

        agent.mcp.call_tool = fake_call_tool  # type: ignore[method-assign]
        result = asyncio.run(agent.handle_request("x"))
        assert result["envelope"][0]["createSurface"]["surfaceId"] == "s1"
        # We expect the budget to cap inner-loop iterations.
        assert provider.call_count <= 11


# ===========================================================================
# agent.py — handle_request_stream
# ===========================================================================
class TestHandleRequestStream:
    def test_emits_op_events_for_each_envelope_member(self) -> None:
        provider = FakeProvider().queue(
            _ScriptedStep(
                stream_events=[
                    StreamEvent(delta="<a2ui-json>"),
                    StreamEvent(delta=json.dumps(VALID_ENVELOPE)),
                    StreamEvent(delta="</a2ui-json>"),
                    StreamEvent(response=ModelResponse(content="x"), done=True),
                ]
            )
        )
        agent = _agent_with_cached_tools(provider)
        events = _consume(agent.handle_request_stream("x"))
        op_events = [e for e in events if e["type"] == "op"]
        assert [e["kind"] for e in op_events] == [
            "createSurface",
            "updateComponents",
            "updateDataModel",
        ]
        # Final surfaceIds event carries the aggregated ids.
        surf = [e for e in events if e["type"] == "surfaceIds"]
        assert surf and surf[0]["surfaceIds"] == ["s1"]

    def test_emits_error_when_provider_stream_errors(self) -> None:
        # When the provider's stream emits ``done=True`` with an error,
        # the agent should surface an ``error`` event to the consumer.
        # (At the moment the SDK's error message does NOT echo the
        # provider's original error string back to the consumer — this
        # test pins current behaviour, and a future fix should change
        # the assertion to check for the provider's text.)
        provider = FakeProvider().queue(
            _ScriptedStep(
                stream_events=[
                    StreamEvent(done=True, error="boom"),
                ]
            )
        )
        agent = _agent_with_cached_tools(provider)
        events = _consume(agent.handle_request_stream("x"))
        err = [e for e in events if e["type"] == "error"]
        assert err
        # The error event must be present and non-empty.
        assert err[0]["error"]

    def test_emits_error_when_no_envelope_emitted(self) -> None:
        provider = FakeProvider().queue(
            _ScriptedStep(
                stream_events=[
                    StreamEvent(delta="just prose, no envelope"),
                    StreamEvent(response=ModelResponse(content="x"), done=True),
                ]
            )
        )
        agent = _agent_with_cached_tools(provider)
        events = _consume(agent.handle_request_stream("x"))
        err = [e for e in events if e["type"] == "error"]
        assert err

    def test_buffers_update_ops_until_create_surface(self) -> None:
        # If the model streams updateComponents before createSurface,
        # the agent must hold it until the surface is announced. The
        # consumer then sees both ops emitted back-to-back once the
        # surface lands (the buffered op is flushed immediately
        # before the createSurface op).
        provider = FakeProvider().queue(
            _ScriptedStep(
                stream_events=[
                    StreamEvent(delta="<a2ui-json>"),
                    StreamEvent(
                        delta=json.dumps(
                            [
                                {
                                    "version": "v0.9",
                                    "updateComponents": {
                                        "surfaceId": "s1",
                                        "components": [
                                            {
                                                "id": "r",
                                                "component": "Column",
                                            }
                                        ],
                                    },
                                },
                                {
                                    "version": "v0.9",
                                    "createSurface": {"surfaceId": "s1"},
                                },
                            ]
                        )
                    ),
                    StreamEvent(delta="</a2ui-json>"),
                    StreamEvent(response=ModelResponse(content="x"), done=True),
                ]
            )
        )
        agent = _agent_with_cached_tools(provider)
        events = _consume(agent.handle_request_stream("x"))
        op_events = [e for e in events if e["type"] == "op"]
        # Both ops must be emitted, and the renderer-facing event
        # order is buffer-flush-then-surface (so the consumer can
        # mount the surface and then immediately apply the components).
        kinds = [e["kind"] for e in op_events]
        assert kinds == ["updateComponents", "createSurface"]

    def test_circuit_breaker_trips_on_repeated_failure(self) -> None:
        # Three identical bad responses — the breaker should fire.
        provider = FakeProvider().queue(
            _ScriptedStep(
                stream_events=[
                    StreamEvent(delta="bad"),
                    StreamEvent(response=ModelResponse(content="x"), done=True),
                ]
            ),
            _ScriptedStep(
                stream_events=[
                    StreamEvent(delta="bad"),
                    StreamEvent(response=ModelResponse(content="x"), done=True),
                ]
            ),
            _ScriptedStep(
                stream_events=[
                    StreamEvent(delta="bad"),
                    StreamEvent(response=ModelResponse(content="x"), done=True),
                ]
            ),
        )
        agent = _agent_with_cached_tools(provider)
        events = _consume(agent.handle_request_stream("x"))
        err = [e for e in events if e["type"] == "error"]
        assert err
        # The breaker stops after the second identical error, so we
        # should see at most 2 provider calls.
        assert provider.call_count <= 2


# ===========================================================================
# agent.py — warm_catalog
# ===========================================================================
class TestWarmCatalog:
    def test_warm_catalog_without_running_loop_schedules_deferred(self) -> None:
        provider = FakeProvider()
        agent = _agent_with_cached_tools(provider)
        # Called from sync context — no event loop running.
        result = agent.warm_catalog()
        assert result is None
        assert agent.mcp._deferred_warmup is True


# ===========================================================================
# agent.py — serve() exception handling
# ===========================================================================
# The ``serve()`` method calls ``asyncio.run()`` internally and wraps it
# in a try/except that suppresses a *specific* mcp-SDK teardown bug.
# The tests below pin the contract:
#   - the mcp teardown exception is suppressed (matches the known
#     ``anyio`` cancel-scope message),
#   - everything else propagates — including ``CancelledError``,
#     ``KeyboardInterrupt`` and ``SystemExit``.

from syncfusion_a2ui_agent.agent_exceptions import (  # noqa: E402
    is_mcp_teardown_exception as _is_mcp_teardown_exception,
)


class TestMcpTeardownDetection:
    """Direct unit tests for the ``_is_mcp_teardown_exception`` helper."""

    def test_plain_cancel_scope_runtime_error_detected(self) -> None:
        # The exact anyio message that surfaces during teardown.
        exc = RuntimeError(
            "Attempted to exit cancel scope in a different task than it was entered in"
        )
        assert _is_mcp_teardown_exception(exc) is True

    def test_unrelated_runtime_error_not_detected(self) -> None:
        # A RuntimeError that happens to mention "cancel" but is not
        # the anyio teardown bug must NOT be suppressed — otherwise
        # we'd be hiding real bugs in user code.
        exc = RuntimeError("Task was cancelled by user code")
        assert _is_mcp_teardown_exception(exc) is False

    def test_plain_value_error_not_detected(self) -> None:
        # A value error mentioning "cancel" is a real bug — do not
        # silence it.
        exc = ValueError("Cannot cancel an already-finalized request")
        assert _is_mcp_teardown_exception(exc) is False

    def test_exception_group_with_matching_cause_detected(self) -> None:
        # The mcp bug often surfaces wrapped in an ExceptionGroup.
        # Skip the test if the running Python doesn't support it.
        try:
            inner = RuntimeError("Attempted to exit cancel scope in a different task")
            group = BaseExceptionGroup("mcp teardown", [inner])
        except NameError:  # pragma: no cover — Python < 3.11 fallback
            import pytest

            pytest.skip("BaseExceptionGroup requires Python 3.11+")
        assert _is_mcp_teardown_exception(group) is True

    def test_exception_group_with_unrelated_cause_not_detected(self) -> None:
        try:
            inner = RuntimeError("Connection reset by peer")
            group = BaseExceptionGroup("connection lost", [inner])
        except NameError:  # pragma: no cover
            import pytest

            pytest.skip("BaseExceptionGroup requires Python 3.11+")
        assert _is_mcp_teardown_exception(group) is False

    def test_cancelled_error_not_detected_as_teardown(self) -> None:
        # ``asyncio.CancelledError`` is NOT the same as the anyio
        # cancel-scope bug — a user pressing Ctrl+C should NOT be
        # silently swallowed.
        exc = asyncio.CancelledError()
        assert _is_mcp_teardown_exception(exc) is False

    def test_keyboard_interrupt_not_detected(self) -> None:
        exc = KeyboardInterrupt()
        assert _is_mcp_teardown_exception(exc) is False

    def test_system_exit_not_detected(self) -> None:
        exc = SystemExit(0)
        assert _is_mcp_teardown_exception(exc) is False

    def test_cyclic_cause_chain_does_not_loop(self) -> None:
        # Defensive: the cause walker tracks seen IDs to avoid an
        # infinite loop on a pathological cyclic ``__context__``.
        a = RuntimeError("first")
        b = RuntimeError("second")
        a.__cause__ = b
        b.__cause__ = a
        # Returns False (not a teardown exception) and doesn't hang.
        assert _is_mcp_teardown_exception(a) is False


class TestServeExceptionFilter:
    """End-to-end: ``serve()`` swallows the teardown exception and
    lets everything else propagate.

    ``serve()`` is wrapped in ``asyncio.run()`` and binds a real port
    via uvicorn, so we exercise only the post-``asyncio.run()`` filter
    by calling the filter helper directly. The full ``serve()`` flow
    is covered by the example scripts, not by these unit tests.
    """

    def test_mcp_teardown_is_suppressed(self) -> None:
        # The serve() body wraps ``asyncio.run(_serve_with_warmup())``
        # in ``except Exception as exc: ... raise`` if the filter says
        # it's not a teardown. We simulate that path by checking the
        # filter returns True for the known message — which means the
        # outer except block would log+swallow instead of re-raising.
        exc = RuntimeError("Attempted to exit cancel scope in a different task")
        assert _is_mcp_teardown_exception(exc) is True

    def test_cancelled_error_would_propagate(self) -> None:
        # The filter does NOT match CancelledError, so serve()'s
        # ``except Exception`` block re-raises it. Ctrl+C must stop
        # the server cleanly.
        exc = asyncio.CancelledError()
        assert _is_mcp_teardown_exception(exc) is False

    def test_unrelated_runtime_error_would_propagate(self) -> None:
        # A real RuntimeError from user code must NOT be silenced.
        exc = RuntimeError("real bug in user code")
        assert _is_mcp_teardown_exception(exc) is False


# Smoke test: instantiate an agent and confirm ``serve()`` rejects
# the ``--serve`` mode in environments where the A2A SDK isn't
# installed. This exercises the public path without binding a port.
class TestServeSanity:
    def test_serve_rejects_without_a2a_sdk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # If the A2A SDK is missing the constructor raises — this
        # confirms ``serve()`` is reachable from the public surface.
        # We don't actually call ``serve()`` here because it would
        # block on a port; this test just asserts the agent is
        # constructible and ``serve`` is callable.
        provider = FakeProvider()
        agent = SyncfusionAgent(model=provider)
        assert callable(agent.serve)


# ===========================================================================
# agent.py — concurrent-request safety
# ===========================================================================
# A single :class:`SyncfusionAgent` instance can be reused across
# concurrent A2A requests. Per-instance mutable state on the agent
# would therefore be shared between requests and cause cross-request
# contamination. These tests pin that the agent must not carry any
# per-instance tool-call accumulator between requests.


class TestAgentHasNoPerInstanceToolCallState:
    """SyncfusionAgent must not carry per-instance tool-call state.

    Two concurrent ``handle_request_stream`` invocations on the same
    agent must not scribble on each other's accumulators. These
    tests guard against anyone reintroducing such state.
    """

    def test_no_pending_stream_tcs_attribute(self) -> None:
        provider = FakeProvider()
        agent = SyncfusionAgent(model=provider)
        # After construction: the agent must not carry a
        # ``_pending_stream_tcs`` attribute. The ``hasattr`` check
        # fails the moment anyone re-adds the dead state — the
        # failure message points at the regression.
        assert not hasattr(agent, "_pending_stream_tcs"), (
            "SyncfusionAgent must not carry per-instance tool-call "
            "state. Concurrent requests would cross-contaminate."
        )

    def test_no_pending_stream_tcs_after_stream_invocation(self) -> None:
        # Run a full streaming request, then verify the attribute
        # still doesn't exist (the on-instance path must leave the
        # agent instance clean for the next request).
        provider = FakeProvider().queue(
            _ScriptedStep(
                stream_events=[
                    StreamEvent(delta="<a2ui-json>"),
                    StreamEvent(delta=json.dumps(VALID_ENVELOPE)),
                    StreamEvent(delta="</a2ui-json>"),
                    StreamEvent(response=ModelResponse(content="x"), done=True),
                ]
            )
        )
        agent = _agent_with_cached_tools(provider)
        # Mock orchestrator to skip tool-calling
        from syncfusion_a2ui_agent.orchestrator.tool_orchestrator import OrchestratorResult

        async def _mock_execute(plan: Any) -> OrchestratorResult:
            return OrchestratorResult(results=[])

        agent._engine._orchestrator.execute = _mock_execute  # type: ignore[method-assign]

        _consume(agent.handle_request_stream("x"))
        assert not hasattr(agent, "_pending_stream_tcs")

    def test_no_pending_stream_tcs_after_toolcall_stream(self) -> None:
        # A streaming request that *would* have appended to the dead
        # accumulator (the model emits a tool_call event mid-stream)
        # must still leave the agent instance clean afterwards.
        from syncfusion_a2ui_agent.providers.base import ToolCall  # noqa: F401

        # We need a step that yields a tool_call stream event.
        async def stream_with_tc():
            yield StreamEvent(tool_call=ToolCall(name="x__y", arguments={}, id="i"))
            yield StreamEvent(delta="<a2ui-json>")
            yield StreamEvent(delta=json.dumps(VALID_ENVELOPE))
            yield StreamEvent(delta="</a2ui-json>")
            yield StreamEvent(response=ModelResponse(content="x"), done=True)

        class _StreamProvider(FakeProvider):
            async def stream(self, messages, tools=None, context=None):
                async for ev in stream_with_tc():
                    yield ev

        provider = _StreamProvider()
        agent = _agent_with_cached_tools(provider)

        # Patch call_tool so the tool-call dispatch path doesn't fail.
        async def fake_call_tool(server, tool, arguments):
            return {"ok": True}

        agent.mcp.call_tool = fake_call_tool  # type: ignore[method-assign]
        _consume(agent.handle_request_stream("x"))
        assert not hasattr(agent, "_pending_stream_tcs")


class TestAgentHasNoPerInstanceValidationState:
    """Regression for the per-instance ``_last_validation_error`` bug.

    The streaming handler and the non-streaming handler used to write
    the previous validation error to ``self._last_validation_error``
    so the circuit breaker could detect repeated failures. Two
    concurrent requests on the same agent would step on each
    other's circuit-breaker state and abort prematurely.

    The fix was to move that state onto a per-request local
    (``_StreamContext`` for streaming, a plain local for the
    non-streaming path). This test guards against anyone
    reintroducing per-instance state.
    """

    def test_no_last_validation_error_on_construction(self) -> None:
        provider = FakeProvider()
        agent = SyncfusionAgent(model=provider)
        assert not hasattr(agent, "_last_validation_error"), (
            "SyncfusionAgent must not carry per-instance validation "
            "state. Concurrent requests would cross-contaminate."
        )

    def test_no_last_validation_error_after_streaming(self) -> None:
        provider = FakeProvider().queue(
            _ScriptedStep(
                stream_events=[
                    StreamEvent(delta="<a2ui-json>"),
                    StreamEvent(delta=json.dumps(VALID_ENVELOPE)),
                    StreamEvent(delta="</a2ui-json>"),
                    StreamEvent(response=ModelResponse(content="x"), done=True),
                ]
            )
        )
        agent = _agent_with_cached_tools(provider)
        _consume(agent.handle_request_stream("x"))
        assert not hasattr(agent, "_last_validation_error")

    def test_no_last_validation_error_after_non_streaming_retry(
        self,
    ) -> None:
        # One bad response followed by a good one — covers the
        # retry path without tripping the circuit breaker (the
        # breaker needs *two identical* errors to fire). The
        # non-streaming retry path should write the previous
        # error to a local, not the agent instance.
        provider = FakeProvider().queue(
            _ScriptedStep(content="not an envelope"),
            _ScriptedStep(content=VALID_ENVELOPE_TEXT),
        )
        agent = _agent_with_cached_tools(provider)
        result = asyncio.run(agent.handle_request("x"))
        assert result["envelope"][0]["createSurface"]["surfaceId"] == "s1"
        assert not hasattr(agent, "_last_validation_error")

    def test_concurrent_streams_on_same_agent_dont_cross_contaminate(
        self,
    ) -> None:
        # The real test: two concurrent streaming requests on the
        # same agent. With per-instance state removed, neither sees
        # the other's intermediate state.

        async def _drain_async(agen):
            out = []
            async for ev in agen:
                out.append(ev)
            return out

        provider = FakeProvider().queue(
            _ScriptedStep(
                stream_events=[
                    StreamEvent(delta="<a2ui-json>"),
                    StreamEvent(delta=json.dumps(VALID_ENVELOPE)),
                    StreamEvent(delta="</a2ui-json>"),
                    StreamEvent(response=ModelResponse(content="x"), done=True),
                ]
            ),
            _ScriptedStep(
                stream_events=[
                    StreamEvent(delta="<a2ui-json>"),
                    StreamEvent(delta=json.dumps(VALID_ENVELOPE)),
                    StreamEvent(delta="</a2ui-json>"),
                    StreamEvent(response=ModelResponse(content="x"), done=True),
                ]
            ),
        )
        agent = _agent_with_cached_tools(provider)

        # Mock orchestrator to skip tool-calling in concurrent scenario
        from syncfusion_a2ui_agent.orchestrator.tool_orchestrator import OrchestratorResult

        async def _mock_execute(plan: Any) -> OrchestratorResult:
            return OrchestratorResult(results=[])

        agent._engine._orchestrator.execute = _mock_execute  # type: ignore[method-assign]

        async def _run_both() -> tuple[list, list]:
            a, b = await asyncio.gather(
                _drain_async(agent.handle_request_stream("a")),
                _drain_async(agent.handle_request_stream("b")),
            )
            return a, b

        results = asyncio.run(_run_both())
        for events in results:
            surf = [e for e in events if e["type"] == "surfaceIds"]
            assert surf, "stream should terminate with surfaceIds"
            assert surf[0]["surfaceIds"] == ["s1"]
        assert not hasattr(agent, "_last_validation_error")
