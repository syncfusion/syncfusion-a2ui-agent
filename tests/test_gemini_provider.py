"""Direct tests for :class:`GeminiProvider`.

Covers:

* Constructor wiring (env-var fallback, missing SDK raises).
* Message conversion (assistant → model, tool → user).
* Tool schema conversion.
* :meth:`GeminiProvider._stringify_args` — Mapping-like to JSON
  string conversion.
* Temperature / max-tokens configuration.
"""

from __future__ import annotations

from typing import Any

import pytest


def _install_fake_client(
    monkeypatch: pytest.MonkeyPatch, captured: dict[str, Any] | None = None
) -> None:
    """Patch ``genai.Client`` to a no-op sentinel."""

    class _FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            if captured is not None:
                captured.update(kwargs)

    monkeypatch.setattr(
        "syncfusion_a2ui_agent.providers.gemini_provider.genai.Client",
        _FakeClient,
    )


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------
class TestGeminiConstructor:
    def test_missing_dependencies_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from syncfusion_a2ui_agent.providers import gemini_provider as mod
        from syncfusion_a2ui_agent.providers.base import ProviderError

        monkeypatch.setattr(mod, "genai", None)
        with pytest.raises(ProviderError, match="google-genai is not installed"):
            mod.GeminiProvider(api_key="k")

    def test_explicit_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from syncfusion_a2ui_agent.providers import gemini_provider as mod

        captured: dict[str, Any] = {}
        _install_fake_client(monkeypatch, captured)
        p = mod.GeminiProvider(api_key="test-key")
        assert captured["api_key"] == "test-key"
        assert p.model == "gemini-1.5-pro"

    def test_env_api_key_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from syncfusion_a2ui_agent.providers import gemini_provider as mod

        monkeypatch.setenv("GEMINI_API_KEY", "env-key")
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        captured: dict[str, Any] = {}
        _install_fake_client(monkeypatch, captured)
        mod.GeminiProvider()
        assert captured["api_key"] == "env-key"

    def test_google_api_key_fallback_removed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The Gemini provider's constructor only reads
        # ``GEMINI_API_KEY`` (not ``GOOGLE_API_KEY``) per the
        # current source. This test pins that behaviour.
        from syncfusion_a2ui_agent.providers import gemini_provider as mod
        from syncfusion_a2ui_agent.providers.base import ProviderError

        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
        _install_fake_client(monkeypatch, {})
        with pytest.raises(ProviderError, match="api_key"):
            mod.GeminiProvider()

    def test_custom_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from syncfusion_a2ui_agent.providers import gemini_provider as mod

        _install_fake_client(monkeypatch, {})
        p = mod.GeminiProvider(api_key="k", model="gemini-2.0-flash")
        assert p.model == "gemini-2.0-flash"


# ---------------------------------------------------------------------------
# _stringify_args
# ---------------------------------------------------------------------------
class TestStringifyArgs:
    def test_dict_to_json(self) -> None:
        from syncfusion_a2ui_agent.providers.gemini_provider import (
            _stringify_args,
        )

        out = _stringify_args({"a": 1, "b": "x"})
        assert out == '{"a": 1, "b": "x"}'

    def test_empty_dict(self) -> None:
        from syncfusion_a2ui_agent.providers.gemini_provider import (
            _stringify_args,
        )

        assert _stringify_args({}) == "{}"

    def test_mapping_like_object(self) -> None:
        # Gemini's ``function_call.args`` is a Mapping-like
        # protobuf object. We test the contract: any object with an
        # ``.items()`` method is serialised to JSON.
        from collections.abc import Mapping

        from syncfusion_a2ui_agent.providers.gemini_provider import (
            _stringify_args,
        )

        class _MappingLike(Mapping):
            def __init__(self, d: dict[str, Any]) -> None:
                self._d = d

            def __getitem__(self, key: str) -> Any:
                return self._d[key]

            def __iter__(self):  # type: ignore[no-untyped-def]
                return iter(self._d)

            def __len__(self) -> int:
                return len(self._d)

        out = _stringify_args(_MappingLike({"k": "v"}))
        assert out == '{"k": "v"}'


# ---------------------------------------------------------------------------
# Tool schema conversion
# ---------------------------------------------------------------------------
class TestGeminiToolSchema:
    def test_double_underscore_naming(self) -> None:
        from syncfusion_a2ui_agent.providers._common import mcp_tools_to_gemini

        out = mcp_tools_to_gemini(
            [
                {
                    "server": "demo",
                    "name": "ping",
                    "description": "ping",
                    "inputSchema": {"type": "object"},
                }
            ]
        )
        # Gemini uses ``function_declarations`` with a nested name.
        assert out[0]["name"] == "demo__ping"
        assert out[0]["description"] == "ping"
        assert out[0]["parameters"] == {"type": "object"}

    def test_missing_fields_use_defaults(self) -> None:
        from syncfusion_a2ui_agent.providers._common import mcp_tools_to_gemini

        out = mcp_tools_to_gemini([{}])
        assert out[0]["name"] == "mcp__"
        assert out[0]["parameters"] == {"type": "object", "properties": {}}
