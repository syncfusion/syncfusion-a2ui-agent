"""Consolidated unit tests for the SDK's pure-Python internals.

Covers the following source modules:

* ``syncfusion_a2ui_agent.catalog`` — envelope extraction, parsing,
  grammar validation, surface-id helpers, and the per-chunk
  streaming state machine.
* ``syncfusion_a2ui_agent.validation.response_validator`` —
  :class:`ResponseValidator` (text + per-op paths).
* ``syncfusion_a2ui_agent.mcp.mcp_manager`` — :class:`MCPManager`
  registration, tool-cache helpers, and the platform MCP catalog.
* ``syncfusion_a2ui_agent.orchestrator.tool_orchestrator`` — the
  v1 "fan out across all servers" plan/execute path.

These tests are pure-Python and never touch a real AI provider SDK
or a live MCP server. The orchestrator and MCP-manager paths run
against in-process helpers (``register_tool_descriptor``,
``call_cached``, and monkey-patched ``call_tool``).
"""

from __future__ import annotations

import json

import pytest

from syncfusion_a2ui_agent.catalog import (
    A2UI_VERSION,
    OP_KINDS,
    EnvelopeOp,
    _extract_json_array,
    _StreamingState,
    envelope_grammar_schema,
    extract_envelope,
    extract_envelope_iter,
    extract_ops_iter,
    find_create_surface,
    parse_envelope,
    surface_ids_in,
    validate_envelope_grammar,
)
from syncfusion_a2ui_agent.mcp.mcp_manager import (
    MCPManager,
    MCPServerConfig,
    MCPToolCall,
)
from syncfusion_a2ui_agent.orchestrator.tool_orchestrator import (
    OrchestratorPlan,
    OrchestratorResult,
    ToolOrchestrator,
)
from syncfusion_a2ui_agent.validation.response_validator import (
    A2UIValidationError,
    ResponseValidator,
    ValidationResult,
)


# ===========================================================================
# catalog.py
# ===========================================================================
class TestExtractJsonArray:
    def test_simple_array(self) -> None:
        assert _extract_json_array("[1, 2, 3]") == "[1, 2, 3]"

    def test_nested_arrays(self) -> None:
        text = "[1, [2, [3, 4]], 5]"
        out = _extract_json_array(text)
        assert out == text
        json.loads(out)  # must round-trip

    def test_nested_objects_with_string_brackets(self) -> None:
        # Square brackets inside a string must be ignored by the scan.
        text = '[{"key": "]"}, {"k": "["}]'
        out = _extract_json_array(text)
        json.loads(out)

    def test_escape_sequences(self) -> None:
        text = r'["\"]", "\\"]'
        out = _extract_json_array(text)
        json.loads(out)

    def test_unbalanced_returns_empty(self) -> None:
        assert _extract_json_array("[1, 2, ") == ""

    def test_no_array_returns_empty(self) -> None:
        assert _extract_json_array("hello world") == ""

    def test_leading_garbage_before_array(self) -> None:
        assert _extract_json_array("here is the array: [1, 2]") == ""


class TestExtractEnvelope:
    def test_closed_tag(self) -> None:
        text = "<a2ui-json>[1, 2, 3]</a2ui-json>"
        assert extract_envelope(text) == "[1, 2, 3]"

    def test_unclosed_tag_truncates_to_balanced_array(self) -> None:
        text = "<a2ui-json>[1, 2, 3"
        # No balanced ] — should fall through to the raw inner text so
        # the validator can report a parse error.
        out = extract_envelope(text)
        assert "[1, 2, 3" in out

    def test_fence_inside_tag(self) -> None:
        text = "<a2ui-json>```json\n[1, 2, 3]\n```</a2ui-json>"
        assert extract_envelope(text) == "[1, 2, 3]"

    def test_bare_fence(self) -> None:
        text = "```json\n[1, 2, 3]\n```"
        assert extract_envelope(text) == "[1, 2, 3]"

    def test_bare_array(self) -> None:
        # The bare-array path is only triggered when the text *starts*
        # with an array — the extractor intentionally rejects leading
        # prose to avoid false positives.
        text = "[1, 2, 3] thanks!"
        out = extract_envelope(text)
        assert json.loads(out) == [1, 2, 3]

    def test_empty_string(self) -> None:
        assert extract_envelope("") == ""

    def test_case_insensitive_tag(self) -> None:
        text = "<A2UI-JSON>[1, 2]</A2UI-JSON>"
        out = extract_envelope(text)
        assert json.loads(out) == [1, 2]

    def test_whitespace_around_inner_array(self) -> None:
        text = "<a2ui-json>\n  [1, 2]\n  </a2ui-json>"
        out = extract_envelope(text)
        assert json.loads(out) == [1, 2]


class TestParseEnvelope:
    def test_parses_three_ops(self, valid_envelope: list[dict]) -> None:
        blob = json.dumps(valid_envelope)
        ops = parse_envelope(blob)
        assert len(ops) == 3
        assert [op.kind for op in ops] == list(OP_KINDS)

    def test_single_object_normalised_to_list(self) -> None:
        blob = json.dumps({"version": "v0.9", "createSurface": {"surfaceId": "x"}})
        ops = parse_envelope(blob)
        assert len(ops) == 1
        assert ops[0].kind == "createSurface"
        assert ops[0].body == {"surfaceId": "x"}

    def test_missing_kind_raises(self) -> None:
        with pytest.raises(ValueError, match="missing one of"):
            parse_envelope(json.dumps([{"version": "v0.9"}]))

    def test_non_object_op_raises(self) -> None:
        with pytest.raises(ValueError, match="must be an object"):
            parse_envelope(json.dumps([1, 2, 3]))

    def test_non_list_non_dict_raises(self) -> None:
        with pytest.raises(ValueError, match="must be a JSON array"):
            parse_envelope(json.dumps("not a list"))

    def test_envelope_op_shape(self) -> None:
        op = EnvelopeOp(raw={"v": 1}, kind="createSurface", body={"surfaceId": "s"})
        assert op.raw == {"v": 1}
        assert op.body == {"surfaceId": "s"}


class TestValidateEnvelopeGrammar:
    def test_valid_envelope_has_no_errors(self, valid_envelope: list[dict]) -> None:
        assert validate_envelope_grammar(valid_envelope) == []

    def test_missing_version_flagged(self) -> None:
        bad = [{"createSurface": {"surfaceId": "s"}}]
        errors = validate_envelope_grammar(bad)
        assert errors

    def test_missing_surface_id_flagged(self) -> None:
        bad = [{"version": A2UI_VERSION, "createSurface": {}}]
        errors = validate_envelope_grammar(bad)
        assert errors
        assert any("surfaceId" in e for e in errors)

    def test_update_components_requires_components_array(self) -> None:
        bad = [
            {
                "version": A2UI_VERSION,
                "updateComponents": {"surfaceId": "s"},
            }
        ]
        errors = validate_envelope_grammar(bad)
        assert errors
        assert any("components" in e for e in errors)

    def test_update_data_model_requires_value(self) -> None:
        bad = [
            {
                "version": A2UI_VERSION,
                "updateDataModel": {"surfaceId": "s", "path": "/p"},
            }
        ]
        errors = validate_envelope_grammar(bad)
        assert errors

    def test_component_object_requires_id(self) -> None:
        bad = [
            {
                "version": A2UI_VERSION,
                "updateComponents": {
                    "surfaceId": "s",
                    "components": [{"component": "Row"}],
                },
            }
        ]
        errors = validate_envelope_grammar(bad)
        assert errors

    def test_component_with_id_passes_envelope_grammar(self) -> None:
        # Envelope grammar does not validate component-level props.
        ok = [
            {
                "version": A2UI_VERSION,
                "updateComponents": {
                    "surfaceId": "s",
                    "components": [{"id": "x", "component": "MadeUpComponent", "madeUpProp": 42}],
                },
            }
        ]
        assert validate_envelope_grammar(ok) == []

    def test_non_object_op_flagged(self) -> None:
        errors = validate_envelope_grammar([1, 2, 3])  # type: ignore[list-item]
        assert errors
        assert any("not an object" in e for e in errors)

    def test_schema_includes_all_three_kinds(self) -> None:
        schema = envelope_grammar_schema()
        # three oneOf branches — one per op kind
        assert len(schema["oneOf"]) == 3
        for kind in OP_KINDS:
            assert any(kind in json.dumps(branch) for branch in schema["oneOf"])


class TestSurfaceIdsIn:
    def test_collects_unique_ids(self) -> None:
        env = [
            {"version": "v0.9", "createSurface": {"surfaceId": "a"}},
            {
                "version": "v0.9",
                "updateComponents": {"surfaceId": "a", "components": []},
            },
            {
                "version": "v0.9",
                "updateDataModel": {"surfaceId": "b", "path": "/x", "value": 1},
            },
        ]
        assert surface_ids_in(env) == ["a", "a", "b"]

    def test_skips_non_dict_ops(self) -> None:
        assert surface_ids_in([1, "x", None]) == []  # type: ignore[list-item]

    def test_skips_ops_without_known_kind(self) -> None:
        assert surface_ids_in([{"version": "v0.9"}]) == []


class TestFindCreateSurface:
    def test_finds_first(self) -> None:
        env = [
            {"version": "v0.9", "createSurface": {"surfaceId": "first"}},
            {"version": "v0.9", "createSurface": {"surfaceId": "second"}},
        ]
        op = find_create_surface(env)
        assert op is not None
        assert op.body["surfaceId"] == "first"

    def test_filters_by_catalog_id(self) -> None:
        env = [
            {
                "version": "v0.9",
                "createSurface": {"surfaceId": "x", "catalogId": "c1"},
            },
            {
                "version": "v0.9",
                "createSurface": {"surfaceId": "y", "catalogId": "c2"},
            },
        ]
        op = find_create_surface(env, catalog_id="c2")
        assert op is not None
        assert op.body["surfaceId"] == "y"

    def test_returns_none_when_absent(self) -> None:
        assert (
            find_create_surface([{"version": "v0.9", "updateComponents": {"surfaceId": "x"}}])
            is None
        )


# _StreamingState / extract_envelope_iter / extract_ops_iter
def _drive(state: _StreamingState, chunks: list[str]) -> str:
    """Helper: feed chunks into the state, return the completed array."""
    for c in chunks:
        if state.phase == "pre_tag":
            rest = state.feed_pre_tag(c)
            if not rest:
                continue
            c = rest
        if state.phase == "in_tag":
            state.feed_in_tag(c)
            if state.completed_array:
                return state.completed_array
            continue
        if state.phase == "post_tag":
            state.feed_array_scan(c)
            if state.completed_array:
                return state.completed_array
    return state.completed_array


class TestStreamingState:
    def test_single_chunk_with_closed_tag(self) -> None:
        state = _StreamingState()
        out = _drive(state, ["<a2ui-json>[1, 2, 3]</a2ui-json>"])
        assert json.loads(out) == [1, 2, 3]

    def test_array_split_across_chunks(self) -> None:
        state = _StreamingState()
        out = _drive(state, ["<a2ui-json>[1,", " 2, 3]", "</a2ui-json>"])
        assert json.loads(out) == [1, 2, 3]

    def test_unclosed_tag_still_completes_via_balanced_scan(self) -> None:
        state = _StreamingState()
        out = _drive(state, ["<a2ui-json>[1, 2, 3]"])  # no close tag
        assert json.loads(out) == [1, 2, 3]

    def test_brace_counting_inside_strings(self) -> None:
        state = _StreamingState()
        # array containing a string with ']'
        out = _drive(state, ['<a2ui-json>["]", 1]</a2ui-json>'])
        assert json.loads(out) == ["]", 1]

    def test_nested_arrays_inside_main_array(self) -> None:
        state = _StreamingState()
        out = _drive(state, ["<a2ui-json>[[1, 2], [3, 4]]</a2ui-json>"])
        assert json.loads(out) == [[1, 2], [3, 4]]

    def test_empty_chunks_are_safe(self) -> None:
        state = _StreamingState()
        # Drive a no-op — must not raise.
        _drive(state, ["", "", ""])
        assert state.completed_array == ""


class TestExtractEnvelopeIter:
    def test_yields_completed_array_then_stops(self) -> None:
        chunks = ["<a2ui-json>[1, 2, 3]</a2ui-json>", " more"]
        out = list(extract_envelope_iter(chunks))
        assert len(out) == 1
        assert json.loads(out[0]) == [1, 2, 3]

    def test_stream_without_completion_yields_nothing(self) -> None:
        chunks = ["<a2ui-json>[1, 2"]
        assert list(extract_envelope_iter(chunks)) == []


class TestExtractOpsIter:
    def test_yields_one_op_per_envelope_member(self) -> None:
        env = [
            {"version": "v0.9", "createSurface": {"surfaceId": "s"}},
            {
                "version": "v0.9",
                "updateComponents": {
                    "surfaceId": "s",
                    "components": [{"id": "r", "component": "Column"}],
                },
            },
        ]
        chunks = ["<a2ui-json>", json.dumps(env), "</a2ui-json>"]
        ops = list(extract_ops_iter(chunks))
        assert [op.kind for op in ops] == ["createSurface", "updateComponents"]

    def test_malformed_inner_json_silently_yields_nothing(self) -> None:
        chunks = ["<a2ui-json>not json</a2ui-json>"]
        assert list(extract_ops_iter(chunks)) == []


# ===========================================================================
# validation/response_validator.py
# ===========================================================================
class TestResponseValidator:
    def test_valid_envelope(self, valid_envelope_text: str) -> None:
        v = ResponseValidator()
        result = v.validate_text(valid_envelope_text)
        assert result.ok is True
        assert isinstance(result.payload, list)
        assert result.error is None

    def test_missing_envelope_tag(self) -> None:
        # Plain prose with no JSON at all — the validator must report
        # failure, with a message that mentions either the missing
        # envelope tag OR a JSON parse error (the exact path depends
        # on what ``extract_envelope`` returns for prose).
        v = ResponseValidator()
        result = v.validate_text("Just some prose, no JSON.")
        assert result.ok is False
        assert result.error is not None
        assert any(marker in result.error for marker in ("no A2UI envelope", "Invalid JSON"))

    def test_truncated_envelope_returns_parse_error(self) -> None:
        v = ResponseValidator()
        result = v.validate_text("<a2ui-json>[1, 2, ")  # truncated
        assert result.ok is False
        # The extractor returns the partial content; JSON parse fails.
        assert result.error is not None
        assert "JSON" in result.error or "truncated" in result.error

    def test_non_array_envelope_rejected(self) -> None:
        v = ResponseValidator()
        # A single dict gets coerced to [dict] — so use a string
        result = v.validate_text('"a string, not a list"')
        assert result.ok is False

    def test_empty_envelope_array_rejected(self) -> None:
        v = ResponseValidator()
        result = v.validate_text("<a2ui-json>[]</a2ui-json>")
        assert result.ok is False
        assert "empty" in (result.error or "").lower()

    def test_grammar_error_reported(self) -> None:
        # Missing version field on createSurface op.
        v = ResponseValidator()
        bad = '<a2ui-json>[{"createSurface": {"surfaceId": "s"}}]</a2ui-json>'
        result = v.validate_text(bad)
        assert result.ok is False
        assert result.error is not None

    def test_max_attempts_is_retries_plus_one(self) -> None:
        assert ResponseValidator(max_retries=0).max_attempts() == 1
        assert ResponseValidator(max_retries=2).max_attempts() == 3
        assert ResponseValidator(max_retries=5).max_attempts() == 6

    def test_negative_retries_rejected(self) -> None:
        with pytest.raises(ValueError):
            ResponseValidator(max_retries=-1)

    def test_validate_op_per_op(self) -> None:
        v = ResponseValidator()
        ok, err = v.validate_op({"version": "v0.9", "createSurface": {"surfaceId": "s"}})
        assert ok is True
        assert err is None

    def test_validate_op_rejects_missing_kind(self) -> None:
        v = ResponseValidator()
        ok, err = v.validate_op({"version": "v0.9"})
        assert ok is False
        assert err is not None

    def test_validate_op_rejects_non_dict(self) -> None:
        v = ResponseValidator()
        ok, err = v.validate_op(123)  # type: ignore[arg-type]
        assert ok is False
        assert "not an object" in (err or "")

    def test_a2ui_validation_error_is_value_error(self) -> None:
        # The custom exception is a ValueError subclass so callers
        # who only catch ValueError still see it.
        assert issubclass(A2UIValidationError, ValueError)

    def test_validation_result_dataclass_fields(self) -> None:
        r = ValidationResult(ok=True, payload=[], error=None, raw_text="")
        assert r.ok is True
        assert r.payload == []
        assert r.error is None
        assert r.raw_text == ""


# ===========================================================================
# mcp/mcp_manager.py
# ===========================================================================
class TestMCPServerConfig:
    def test_defaults(self) -> None:
        cfg = MCPServerConfig(name="x", command=["npx", "-y", "@x/y"])
        assert cfg.kind == "customer"
        assert cfg.command == ["npx", "-y", "@x/y"]
        assert cfg.url is None
        assert cfg.env == {}
        assert cfg.timeout == 30.0

    def test_custom_kind(self) -> None:
        cfg = MCPServerConfig(name="x", command=["x"], kind="platform")
        assert cfg.kind == "platform"


class TestMCPToolCall:
    def test_construction(self) -> None:
        c = MCPToolCall(server="s", tool="t", arguments={"a": 1})
        assert c.server == "s"
        assert c.tool == "t"
        assert c.arguments == {"a": 1}

    def test_default_arguments(self) -> None:
        c = MCPToolCall(server="s", tool="t")
        assert c.arguments == {}


class TestAddRemove:
    def test_add_with_command(self) -> None:
        m = MCPManager()
        cfg = m.add_mcp(name="x", command=["npx", "-y", "@x/y"])
        assert cfg.name == "x"
        assert m.list_servers() == [cfg]

    def test_add_with_url(self) -> None:
        m = MCPManager()
        cfg = m.add_mcp(name="x", url="https://example.com/mcp")
        assert cfg.url == "https://example.com/mcp"

    def test_add_with_string_command_shlex_parsed(self) -> None:
        m = MCPManager()
        cfg = m.add_mcp(name="x", command="npx -y @x/y")
        assert cfg.command == ["npx", "-y", "@x/y"]

    def test_add_without_command_or_url_raises(self) -> None:
        m = MCPManager()
        with pytest.raises(ValueError, match="requires"):
            m.add_mcp(name="x")

    def test_duplicate_name_raises(self) -> None:
        m = MCPManager()
        m.add_mcp(name="x", command=["echo"])
        with pytest.raises(ValueError, match="already registered"):
            m.add_mcp(name="x", command=["echo"])

    def test_remove_drops_tools(self) -> None:
        m = MCPManager()
        m.add_mcp(name="x", command=["echo"])
        m.register_tool_descriptor("x", "do_thing", description="d")
        assert m.has_cached_tools("x")
        m.remove_mcp("x")
        assert not m.has_cached_tools("x")
        assert m.list_servers() == []

    def test_remove_unknown_is_noop(self) -> None:
        m = MCPManager()
        # Must not raise even if the server was never added.
        m.remove_mcp("never-existed")
        assert m.list_servers() == []


class TestListAndCache:
    def test_list_tools_empty_initially(self) -> None:
        m = MCPManager()
        assert m.list_tools() == []
        assert m.has_cached_tools() is False

    def test_register_tool_descriptor_visible_via_list(self) -> None:
        m = MCPManager()
        m.add_mcp(name="x", command=["echo"])
        m.register_tool_descriptor("x", "do_thing", description="a tool")
        tools = m.list_tools(server="x")
        assert len(tools) == 1
        assert tools[0]["name"] == "do_thing"
        assert tools[0]["server"] == "x"

    def test_list_tools_filters_by_server(self) -> None:
        m = MCPManager()
        m.add_mcp(name="a", command=["echo"])
        m.add_mcp(name="b", command=["echo"])
        m.register_tool_descriptor("a", "ta", description="d")
        m.register_tool_descriptor("b", "tb", description="d")
        assert {t["name"] for t in m.list_tools(server="a")} == {"ta"}
        assert {t["name"] for t in m.list_tools(server="b")} == {"tb"}
        assert {t["name"] for t in m.list_tools()} == {"ta", "tb"}

    def test_has_cached_tools_per_server(self) -> None:
        m = MCPManager()
        m.add_mcp(name="a", command=["echo"])
        m.add_mcp(name="b", command=["echo"])
        m.register_tool_descriptor("a", "ta", description="d")
        assert m.has_cached_tools("a")
        assert not m.has_cached_tools("b")

    def test_remove_evicts_tools_by_prefix(self) -> None:
        m = MCPManager()
        m.add_mcp(name="x", command=["echo"])
        m.register_tool_descriptor("x", "do_thing", description="d")
        m.add_mcp(name="xx", command=["echo"])
        m.register_tool_descriptor("xx", "do_other", description="d")
        m.remove_mcp("x")
        # x's tools are gone, xx's survive
        names = {t["name"] for t in m.list_tools()}
        assert "do_thing" not in names
        assert "do_other" in names


class TestCallCached:
    def test_returns_injected_result(self) -> None:
        import asyncio

        m = MCPManager()
        m.add_mcp(name="x", command=["echo"])
        m.register_tool_descriptor("x", "ping", description="d")
        out = asyncio.run(m.call_cached("x", "ping", {"ok": True}))
        assert out == {"ok": True}

    def test_unknown_tool_raises_keyerror(self) -> None:
        import asyncio

        m = MCPManager()
        m.add_mcp(name="x", command=["echo"])
        with pytest.raises(KeyError):
            asyncio.run(m.call_cached("x", "missing", None))


# ---------------------------------------------------------------------------
# The previous TestPlatformCatalog class (and the static
# ``platform_mcp_catalog`` module it pinned) were removed in 0.1.1.
#
# Reason: the SDK must not hardcode a list of "known" Platform MCPs.
# Customers bring their own — see the README's "Platform MCPs" section
# for the generic ``agent.add_mcp(name=..., command=..., kind="platform")``
# pattern that works for any renderer (React, Blazor, Angular, Vue, MAUI,
# WPF, WinForms, in-house, third-party, etc.).
# ---------------------------------------------------------------------------


# ===========================================================================
# orchestrator/tool_orchestrator.py
# ===========================================================================
class TestOrchestratorPlan:
    def test_default_empty(self) -> None:
        plan = OrchestratorPlan()
        assert plan.mcp_servers == []

    def test_preserves_servers(self) -> None:
        plan = OrchestratorPlan(mcp_servers=["a", "b"])
        assert plan.mcp_servers == ["a", "b"]


class TestOrchestratorResult:
    def test_default_empty(self) -> None:
        r = OrchestratorResult()
        assert r.results == []
        assert r.errors == []


class TestToolOrchestrator:
    def test_plan_returns_all_servers(self) -> None:
        mcp = MCPManager()
        mcp.add_mcp(name="a", command=["echo"])
        mcp.add_mcp(name="b", command=["echo"])
        orch = ToolOrchestrator(mcp_manager=mcp)
        plan = orch.plan("any user message", conversation_history=[])
        # ``plan.mcp_servers`` is a list of MCPServerConfig objects,
        # not strings — extract the .name attribute for the assertion.
        names = sorted(s.name for s in plan.mcp_servers)
        assert names == ["a", "b"]

    def test_execute_with_empty_plan_returns_empty_result(self) -> None:
        mcp = MCPManager()
        orch = ToolOrchestrator(mcp_manager=mcp)
        import asyncio

        plan = OrchestratorPlan(mcp_servers=[])
        out = asyncio.run(orch.execute(plan))
        assert out.results == []

    def test_execute_dispatches_hinted_tools(self) -> None:
        """When ``mcp_tool_hints`` is provided the orchestrator must call
        *exactly* those tools on the matching server, in parallel.
        """
        import asyncio

        mcp = MCPManager()
        mcp.add_mcp(name="s", command=["echo"])
        mcp.register_tool_descriptor("s", "alpha", description="a", input_schema={"type": "object"})
        mcp.register_tool_descriptor("s", "beta", description="b", input_schema={"type": "object"})
        # Inject deterministic results via a stubbed call_tool.
        cached = {"alpha": {"value": 1}, "beta": {"value": 2}}

        async def fake_call_tool(server: str, tool: str, arguments):
            return cached.get(tool)

        mcp.call_tool = fake_call_tool  # type: ignore[method-assign]

        orch = ToolOrchestrator(mcp_manager=mcp)
        plan = OrchestratorPlan(mcp_servers=[MCPServerConfig(name="s", command=["echo"])])
        out = asyncio.run(orch.execute(plan, mcp_tool_hints={"s": ["alpha", "beta"]}))
        names = sorted(r.source for r in out.results)
        assert names == ["mcp:s:alpha", "mcp:s:beta"]

    def test_execute_swallows_per_call_errors(self) -> None:
        """A tool that raises must not abort the whole batch — its
        ``ToolResult`` should carry the error string.
        """
        import asyncio

        mcp = MCPManager()
        mcp.add_mcp(name="s", command=["echo"])
        mcp.register_tool_descriptor("s", "boom", description="d")

        async def fake_call_tool(server: str, tool: str, arguments):
            if tool == "boom":
                raise RuntimeError("kaboom")
            return None

        mcp.call_tool = fake_call_tool  # type: ignore[method-assign]

        orch = ToolOrchestrator(mcp_manager=mcp)
        plan = OrchestratorPlan(mcp_servers=[MCPServerConfig(name="s", command=["echo"])])
        out = asyncio.run(orch.execute(plan, mcp_tool_hints={"s": ["boom"]}))
        assert len(out.results) == 1
        assert "kaboom" in str(out.results[0].payload)
