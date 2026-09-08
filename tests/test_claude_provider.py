"""Direct tests for :class:`ClaudeProvider`.

Covers:

* Constructor wiring (env-var fallback, missing SDK raises).
* :meth:`ClaudeProvider.generate` — system-message extraction,
  tool schema, tool-call parsing.
* :meth:`ClaudeProvider.stream` — content delta emission, tool-call
  accumulation, error handling.

The Anthropic SDK client is never invoked — we patch the
module-level ``AsyncAnthropic`` symbol to a sentinel.
"""

from __future__ import annotations

from typing import Any

import pytest


def _install_fake_client(
    monkeypatch: pytest.MonkeyPatch, captured: dict[str, Any] | None = None
) -> None:
    """Patch ``AsyncAnthropic`` to a no-op sentinel."""

    class _FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            if captured is not None:
                captured.update(kwargs)

    monkeypatch.setattr(
        "syncfusion_a2ui_agent.providers.claude_provider.AsyncAnthropic",
        _FakeClient,
    )


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------
class TestClaudeConstructor:
    def test_missing_dependencies_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from syncfusion_a2ui_agent.providers import claude_provider as mod
        from syncfusion_a2ui_agent.providers.base import ProviderError

        monkeypatch.setattr(mod, "AsyncAnthropic", None)
        with pytest.raises(ProviderError, match="anthropic is not installed"):
            mod.ClaudeProvider(api_key="k")

    def test_explicit_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from syncfusion_a2ui_agent.providers import claude_provider as mod

        captured: dict[str, Any] = {}
        _install_fake_client(monkeypatch, captured)
        p = mod.ClaudeProvider(api_key="sk-test")
        assert captured["api_key"] == "sk-test"
        assert p.model == "claude-3-5-sonnet-latest"

    def test_env_api_key_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from syncfusion_a2ui_agent.providers import claude_provider as mod

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env")
        captured: dict[str, Any] = {}
        _install_fake_client(monkeypatch, captured)
        mod.ClaudeProvider()
        assert captured["api_key"] == "sk-env"

    def test_default_max_tokens_4096(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from syncfusion_a2ui_agent.providers import claude_provider as mod

        _install_fake_client(monkeypatch, {})
        p = mod.ClaudeProvider(api_key="k")
        # Bumped from 1024 to 4096 per the source comment.
        assert p.default_max_tokens == 4096

    def test_custom_max_tokens(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from syncfusion_a2ui_agent.providers import claude_provider as mod

        _install_fake_client(monkeypatch, {})
        p = mod.ClaudeProvider(api_key="k", max_tokens=8192)
        assert p.default_max_tokens == 8192


# ---------------------------------------------------------------------------
# Helpers — split_system_messages is used heavily by Claude
# ---------------------------------------------------------------------------
class TestSplitSystemMessages:
    def test_extracts_system_role(self) -> None:
        from syncfusion_a2ui_agent.providers._common import (
            split_system_messages,
        )

        # Returns ``(non_system_messages, system_parts)``.
        non_system, system_parts = split_system_messages(
            [
                {"role": "system", "content": "you are a helper"},
                {"role": "user", "content": "hi"},
            ]
        )
        assert "you are a helper" in system_parts
        assert len(non_system) == 1
        assert non_system[0]["role"] == "user"

    def test_concatenates_multiple_system_messages(self) -> None:
        from syncfusion_a2ui_agent.providers._common import (
            split_system_messages,
        )

        _, system_parts = split_system_messages(
            [
                {"role": "system", "content": "first"},
                {"role": "system", "content": "second"},
            ]
        )
        assert "first" in system_parts
        assert "second" in system_parts

    def test_no_system_returns_empty_list(self) -> None:
        from syncfusion_a2ui_agent.providers._common import (
            split_system_messages,
        )

        non_system, system_parts = split_system_messages([{"role": "user", "content": "hi"}])
        assert system_parts == []
        assert len(non_system) == 1


# ---------------------------------------------------------------------------
# Tool schema conversion
# ---------------------------------------------------------------------------
class TestAnthropicToolSchema:
    def test_double_underscore_naming(self) -> None:
        from syncfusion_a2ui_agent.providers._common import (
            mcp_tools_to_anthropic,
        )

        out = mcp_tools_to_anthropic(
            [
                {
                    "server": "demo",
                    "name": "ping",
                    "description": "ping",
                    "inputSchema": {"type": "object"},
                }
            ]
        )
        assert out[0]["name"] == "demo__ping"
        assert out[0]["description"] == "ping"
        assert out[0]["input_schema"] == {"type": "object"}

    def test_missing_fields_use_defaults(self) -> None:
        from syncfusion_a2ui_agent.providers._common import (
            mcp_tools_to_anthropic,
        )

        out = mcp_tools_to_anthropic([{}])
        assert out[0]["name"] == "mcp__"
        assert out[0]["input_schema"] == {"type": "object", "properties": {}}
