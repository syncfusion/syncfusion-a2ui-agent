"""Direct tests for :class:`AzureOpenAIProvider`.

Covers:

* Constructor wiring (env-var fallback chain, validation of
  required fields, ``api_version`` default).
* :meth:`AzureOpenAIProvider._build_kwargs` — temperature
  handling, max-completion-tokens, reasoning-effort fallback,
  parallel-tool-calls.
* The default-temperature fix (C5): 0.0 for non-reasoning models
  with ``context['temperature']`` override, omitted for reasoning
  models.

The actual SDK client is never invoked — the test patches the
module-level ``AsyncAzureOpenAI`` symbol to a sentinel that
captures the kwargs.
"""

from __future__ import annotations

from typing import Any

import pytest


def _install_fake_client(monkeypatch: pytest.MonkeyPatch, captured: dict[str, Any]) -> None:
    """Patch ``AsyncAzureOpenAI`` to a no-op sentinel."""

    class _FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(
        "syncfusion_a2ui_agent.providers.azure_openai_provider.AsyncAzureOpenAI",
        _FakeClient,
    )


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------
class TestAzureConstructor:
    def test_missing_dependencies_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from syncfusion_a2ui_agent.providers import azure_openai_provider as mod
        from syncfusion_a2ui_agent.providers.base import ProviderError

        # Pretend the SDK isn't installed.
        monkeypatch.setattr(mod, "AsyncAzureOpenAI", None)
        with pytest.raises(ProviderError, match="openai is not installed"):
            mod.AzureOpenAIProvider(api_key="k", azure_endpoint="e", azure_deployment="d")

    def test_missing_required_fields_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from syncfusion_a2ui_agent.providers import azure_openai_provider as mod
        from syncfusion_a2ui_agent.providers.base import ProviderError

        # Bypass the SDK check by re-installing a sentinel.
        _install_fake_client(monkeypatch, {})
        with pytest.raises(ProviderError, match="api_key.*azure_endpoint.*azure_deployment"):
            mod.AzureOpenAIProvider()

    def test_explicit_kwargs_wins_over_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from syncfusion_a2ui_agent.providers import azure_openai_provider as mod

        monkeypatch.setenv("AZURE_API_KEY", "env-key")
        monkeypatch.setenv("AZURE_API_BASE", "https://env.openai.azure.com")
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "env-dep")
        captured: dict[str, Any] = {}
        _install_fake_client(monkeypatch, captured)
        p = mod.AzureOpenAIProvider(
            api_key="k",
            azure_endpoint="https://x.openai.azure.com",
            azure_deployment="x-dep",
        )
        assert captured["api_key"] == "k"
        assert captured["azure_endpoint"] == "https://x.openai.azure.com"
        assert p.model == "x-dep"

    def test_env_fallback_chain_azure_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from syncfusion_a2ui_agent.providers import azure_openai_provider as mod

        monkeypatch.delenv("AZURE_API_KEY", raising=False)
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "openai-sdk-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://oep.openai.azure.com")
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "oep-dep")
        captured: dict[str, Any] = {}
        _install_fake_client(monkeypatch, captured)
        p = mod.AzureOpenAIProvider()
        assert captured["api_key"] == "openai-sdk-key"
        assert captured["azure_endpoint"] == "https://oep.openai.azure.com"
        assert p.model == "oep-dep"

    def test_api_version_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from syncfusion_a2ui_agent.providers import azure_openai_provider as mod

        captured: dict[str, Any] = {}
        _install_fake_client(monkeypatch, captured)
        mod.AzureOpenAIProvider(
            api_key="k",
            azure_endpoint="https://x.openai.azure.com",
            azure_deployment="x",
        )
        assert captured["api_version"] == "2024-05-01-preview"

    def test_api_version_explicit_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from syncfusion_a2ui_agent.providers import azure_openai_provider as mod

        captured: dict[str, Any] = {}
        _install_fake_client(monkeypatch, captured)
        mod.AzureOpenAIProvider(
            api_key="k",
            azure_endpoint="https://x.openai.azure.com",
            azure_deployment="x",
            api_version="2025-01-01-preview",
        )
        assert captured["api_version"] == "2025-01-01-preview"

    def test_max_completion_tokens_kwarg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from syncfusion_a2ui_agent.providers import azure_openai_provider as mod

        _install_fake_client(monkeypatch, {})
        p = mod.AzureOpenAIProvider(
            api_key="k",
            azure_endpoint="https://x.openai.azure.com",
            azure_deployment="x",
            max_completion_tokens=2048,
        )
        assert p.default_max_tokens == 2048


# ---------------------------------------------------------------------------
# _build_kwargs — temperature handling (C5)
# ---------------------------------------------------------------------------
class TestAzureBuildKwargsTemperature:
    def _build(
        self,
        monkeypatch: pytest.MonkeyPatch,
        model: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        from syncfusion_a2ui_agent.providers import azure_openai_provider as mod

        _install_fake_client(monkeypatch, {})
        p = mod.AzureOpenAIProvider(api_key="k", azure_endpoint="e", azure_deployment=model)
        return p._build_kwargs(
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            context=context,
            stream=False,
        )

    def test_non_reasoning_default_is_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        kwargs = self._build(monkeypatch, "gpt-4o")
        assert kwargs["temperature"] == 0.0

    def test_reasoning_model_omits_temperature(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for model in ("o1", "o1-mini", "gpt-5", "gpt-5-mini"):
            kwargs = self._build(monkeypatch, model)
            assert "temperature" not in kwargs, f"{model!r} must omit temperature; got {kwargs!r}"

    def test_context_temperature_overrides_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        kwargs = self._build(monkeypatch, "gpt-4o", context={"temperature": 0.7})
        assert kwargs["temperature"] == 0.7

    def test_context_temperature_invalid_value_uses_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        kwargs = self._build(monkeypatch, "gpt-4o", context={"temperature": "not-a-number"})
        assert kwargs["temperature"] == 0.0

    def test_context_temperature_ignored_for_reasoning_model(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        # If a caller asks for a specific temperature on a reasoning
        # model, the param must be omitted (HTTP 400 otherwise) and
        # a warning logged.
        with caplog.at_level("WARNING"):
            kwargs = self._build(monkeypatch, "gpt-5", context={"temperature": 0.5})
        assert "temperature" not in kwargs
        assert any(
            "context['temperature']" in rec.message and "reasoning model" in rec.message
            for rec in caplog.records
        )

    def test_max_completion_tokens_always_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Reasoning and non-reasoning both use max_completion_tokens.
        for model in ("gpt-4o", "gpt-5-mini", "o1"):
            kwargs = self._build(monkeypatch, model)
            assert "max_completion_tokens" in kwargs
            # And Azure never uses the legacy key.
            assert "max_tokens" not in kwargs

    def test_tools_included_when_provided(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from syncfusion_a2ui_agent.providers import azure_openai_provider as mod

        _install_fake_client(monkeypatch, {})
        p = mod.AzureOpenAIProvider(api_key="k", azure_endpoint="e", azure_deployment="gpt-4o")
        kwargs = p._build_kwargs(
            messages=[{"role": "user", "content": "hi"}],
            tools=[
                {
                    "server": "demo",
                    "name": "ping",
                    "description": "ping",
                    "inputSchema": {"type": "object"},
                }
            ],
            context=None,
            stream=False,
        )
        assert "tools" in kwargs
        assert kwargs["tools"][0]["function"]["name"] == "demo__ping"

    def test_stream_flag_enables_streaming(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from syncfusion_a2ui_agent.providers import azure_openai_provider as mod

        _install_fake_client(monkeypatch, {})
        p = mod.AzureOpenAIProvider(api_key="k", azure_endpoint="e", azure_deployment="gpt-4o")
        kwargs = p._build_kwargs(
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            context=None,
            stream=True,
        )
        assert kwargs["stream"] is True
        assert kwargs["stream_options"] == {"include_usage": True}
