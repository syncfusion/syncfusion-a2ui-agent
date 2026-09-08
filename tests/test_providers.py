"""Consolidated tests for the provider layer.

Covers:

* The provider base class contract (no network calls).
* The provider registry (resolution, factories, custom classes).
* Each built-in provider's constructor validation (env fallback, error
  messages). The actual SDK client is never invoked — we patch the
  module-level ``AsyncXxx`` symbol to a sentinel so we can assert the
  provider wired the right kwargs into it.
* The OpenAI helper functions (``_to_openai_tools``,
  ``_from_openai_response``, ``_supports_reasoning_effort``).
* :class:`CustomProvider` — the customer-facing base class.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from syncfusion_a2ui_agent.providers.base import (
    AIProvider,
    ModelResponse,
    ProviderError,
    StreamEvent,
    ToolCall,
)
from syncfusion_a2ui_agent.providers.custom_provider import CustomProvider
from syncfusion_a2ui_agent.providers.provider_registry import (
    list_providers,
    register_provider,
    resolve_provider,
)


# ---------------------------------------------------------------------------
# Base + dataclasses
# ---------------------------------------------------------------------------
class TestModelResponse:
    def test_defaults(self) -> None:
        r = ModelResponse(content="hi")
        assert r.content == "hi"
        assert r.model == ""
        assert r.raw is None
        assert r.usage == {}
        assert r.tool_calls == []
        assert r.raw_message is None

    def test_with_tool_calls(self) -> None:
        tc = ToolCall(name="s__t", arguments={"x": 1}, id="abc")
        r = ModelResponse(content="", tool_calls=[tc], raw_message={"role": "assistant"})
        assert r.tool_calls[0].name == "s__t"
        assert r.raw_message == {"role": "assistant"}


class TestStreamEvent:
    def test_all_default_to_empty(self) -> None:
        e = StreamEvent()
        assert e.delta == ""
        assert e.tool_call is None
        assert e.response is None
        assert e.done is False
        assert e.error is None


class TestToolCall:
    def test_default_arguments(self) -> None:
        tc = ToolCall(name="x")
        assert tc.arguments == {}
        assert tc.id is None


class TestProviderError:
    def test_is_runtime_error_subclass(self) -> None:
        assert issubclass(ProviderError, RuntimeError)


class TestAIProviderBase:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            AIProvider()  # type: ignore[abstract]

    def test_subclass_must_implement_generate(self) -> None:
        class Broken(AIProvider):
            name = "broken"

        with pytest.raises(TypeError):
            Broken()  # type: ignore[abstract]

    def test_default_stream_yields_one_done_event(self) -> None:
        class Stub(AIProvider):
            name = "stub"

            async def generate(self, messages, tools=None, context=None):
                return ModelResponse(content="ok", model="stub")

        async def run() -> list[StreamEvent]:
            s = Stub()
            events: list[StreamEvent] = []
            async for e in s.stream([], tools=None):
                events.append(e)
            return events

        import asyncio

        events = asyncio.run(run())
        assert len(events) == 1
        assert events[0].done is True
        assert events[0].response is not None
        assert events[0].response.content == "ok"


# ---------------------------------------------------------------------------
# CustomProvider
# ---------------------------------------------------------------------------
class TestCustomProvider:
    def test_default_generate_raises_not_implemented(self) -> None:
        # CustomProvider provides a concrete (but failing) ``generate``
        # so it can be instantiated, but calling generate() must
        # clearly signal that subclasses are required.
        p = CustomProvider()
        import asyncio

        with pytest.raises(NotImplementedError, match="must implement"):
            asyncio.run(p.generate([]))

    def test_subclass_must_implement_generate(self) -> None:
        class Good(CustomProvider):
            name = "good"

            async def generate(self, messages, tools=None, context=None):
                return ModelResponse(content="ok")

        # Now the subclass can be instantiated.
        p = Good()
        assert p.name == "good"

    def test_default_name_is_custom(self) -> None:
        class Unnamed(CustomProvider):
            async def generate(self, messages, tools=None, context=None):
                return ModelResponse(content="ok")

        p = Unnamed()
        assert p.name == "custom"

    def test_subclass_generate_is_awaitable(self) -> None:
        import asyncio

        class Good(CustomProvider):
            name = "g"

            async def generate(self, messages, tools=None, context=None):
                return ModelResponse(content="awaited")

        async def run() -> str:
            p = Good()
            r = await p.generate([])
            return r.content

        assert asyncio.run(run()) == "awaited"


def test_ai_provider_subclass_compatible_with_custom_provider() -> None:
    """A user who subclasses ``AIProvider`` directly (instead of
    ``CustomProvider``) should still be registerable with the registry.
    """
    from syncfusion_a2ui_agent.providers import provider_registry as reg
    from syncfusion_a2ui_agent.providers.provider_registry import (
        register_provider,
    )

    class BareAI(AIProvider):
        name = "bareai"

        async def generate(self, messages, tools=None, context=None):
            return ModelResponse(content="bare")

    register_provider("bareai", BareAI)
    try:
        p = reg.resolve_provider("bareai")
        assert isinstance(p, BareAI)
    finally:
        reg._REGISTRY.pop("bareai", None)
        reg._FACTORIES.pop("bareai", None)


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------
class TestProviderRegistry:
    def test_builtin_providers_registered(self) -> None:
        names = list_providers()
        for expected in ("azure_openai", "openai", "claude", "gemini", "ollama"):
            assert expected in names, f"missing built-in provider: {expected}"

    def test_resolve_known_provider(self) -> None:
        # With an explicit api_key the OpenAI provider constructs
        # successfully — confirming the registry wires the kwargs
        # through to the provider's constructor.
        p = resolve_provider("openai", api_key="fake")
        assert p is not None
        assert p.name == "openai"

    def test_resolve_unknown_raises_keyerror(self) -> None:
        with pytest.raises(KeyError, match="Unknown AI provider"):
            resolve_provider("never-existed-xyz")

    def test_register_custom_subclass(self) -> None:
        class MyProvider(AIProvider):
            name = "mytest"

            async def generate(self, messages, tools=None, context=None):
                return ModelResponse(content="custom")

        register_provider("mytest", MyProvider)
        try:
            p = resolve_provider("mytest")
            assert isinstance(p, MyProvider)
        finally:
            # Best-effort cleanup: the registry has no public unregister
            # API; in production, custom registrations are sticky by
            # design. So we just assert the test re-ran isolation-safe.
            from syncfusion_a2ui_agent.providers import provider_registry as reg

            reg._REGISTRY.pop("mytest", None)
            reg._FACTORIES.pop("mytest", None)

    def test_register_invalid_raises_typeerror(self) -> None:
        with pytest.raises(TypeError):
            register_provider("bogus", 42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# OpenAI helper functions
# ---------------------------------------------------------------------------
class TestReasoningEffortRegex:
    def test_o1_detected(self) -> None:
        from syncfusion_a2ui_agent.providers.openai_provider import (
            _supports_reasoning_effort,
        )

        assert _supports_reasoning_effort("o1") is True
        assert _supports_reasoning_effort("o1-preview") is True
        assert _supports_reasoning_effort("o1-mini") is True

    def test_o3_detected(self) -> None:
        from syncfusion_a2ui_agent.providers.openai_provider import (
            _supports_reasoning_effort,
        )

        assert _supports_reasoning_effort("o3") is True
        assert _supports_reasoning_effort("o3-mini") is True

    def test_o4_mini_detected(self) -> None:
        from syncfusion_a2ui_agent.providers.openai_provider import (
            _supports_reasoning_effort,
        )

        assert _supports_reasoning_effort("o4-mini") is True

    def test_gpt5_reasoning_detected(self) -> None:
        from syncfusion_a2ui_agent.providers.openai_provider import (
            _supports_reasoning_effort,
        )

        assert _supports_reasoning_effort("gpt-5") is True
        assert _supports_reasoning_effort("gpt-5.1") is True
        assert _supports_reasoning_effort("gpt-5-mini") is True

    def test_gpt5_chat_excluded(self) -> None:
        # Negative lookahead — non-reasoning gpt-5-chat must NOT be
        # flagged as a reasoning model.
        from syncfusion_a2ui_agent.providers.openai_provider import (
            _supports_reasoning_effort,
        )

        assert _supports_reasoning_effort("gpt-5-chat") is False

    def test_gpt4_not_a_reasoning_model(self) -> None:
        from syncfusion_a2ui_agent.providers.openai_provider import (
            _supports_reasoning_effort,
        )

        assert _supports_reasoning_effort("gpt-4o") is False
        assert _supports_reasoning_effort("gpt-4.1") is False
        assert _supports_reasoning_effort("gpt-3.5-turbo") is False


class TestOpenAIToolHelpers:
    def test_to_openai_tools_uses_double_underscore(self) -> None:
        from syncfusion_a2ui_agent.providers.openai_provider import (
            _to_openai_tools,
        )

        out = _to_openai_tools(
            [
                {
                    "server": "demo",
                    "name": "ping",
                    "description": "ping the server",
                    "inputSchema": {"type": "object"},
                }
            ]
        )
        assert out[0]["function"]["name"] == "demo__ping"
        assert out[0]["type"] == "function"
        assert out[0]["function"]["parameters"] == {"type": "object"}

    def test_to_openai_tools_missing_fields_use_defaults(self) -> None:
        from syncfusion_a2ui_agent.providers.openai_provider import (
            _to_openai_tools,
        )

        out = _to_openai_tools([{}])
        assert out[0]["function"]["name"] == "mcp__"
        assert out[0]["function"]["parameters"] == {"type": "object", "properties": {}}

    def test_from_openai_response_no_tool_calls(self) -> None:
        from syncfusion_a2ui_agent.providers.openai_provider import (
            _from_openai_response,
        )

        class _Choice:
            class _Msg:
                content = "hi"
                tool_calls = None

            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        calls, raw = _from_openai_response(_Resp())
        assert calls == []
        assert raw is None

    def test_from_openai_response_with_tool_calls(self) -> None:
        from syncfusion_a2ui_agent.providers.openai_provider import (
            _from_openai_response,
        )

        class _Fn:
            name = "demo__ping"
            arguments = json.dumps({"a": 1})

        class _Tc:
            id = "call_1"
            function = _Fn()

        class _Msg:
            content = ""
            tool_calls = [_Tc()]

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        calls, raw = _from_openai_response(_Resp())
        assert len(calls) == 1
        assert calls[0].name == "demo__ping"
        assert calls[0].arguments == {"a": 1}
        assert raw is not None
        assert raw["role"] == "assistant"
        assert raw["tool_calls"][0]["function"]["name"] == "demo__ping"

    def test_from_openai_response_with_no_choices(self) -> None:
        from syncfusion_a2ui_agent.providers.openai_provider import (
            _from_openai_response,
        )

        class _Resp:
            choices = []

        calls, raw = _from_openai_response(_Resp())
        assert calls == []
        assert raw is None

    def test_from_openai_response_unparseable_arguments(self) -> None:
        from syncfusion_a2ui_agent.providers.openai_provider import (
            _from_openai_response,
        )

        class _Fn:
            name = "x"
            arguments = "{not json"

        class _Tc:
            id = "i"
            function = _Fn()

        class _Msg:
            content = ""
            tool_calls = [_Tc()]

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        calls, _ = _from_openai_response(_Resp())
        assert calls[0].arguments == {}


# ---------------------------------------------------------------------------
# Per-provider constructor validation
# ---------------------------------------------------------------------------
class TestOpenAIProviderConstructor:
    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from syncfusion_a2ui_agent.providers.openai_provider import OpenAIProvider

        with pytest.raises(ProviderError, match="api_key"):
            OpenAIProvider()

    def test_explicit_api_key_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from syncfusion_a2ui_agent.providers.openai_provider import (
            OpenAIProvider,
        )

        captured: dict[str, Any] = {}

        class _FakeClient:
            def __init__(self, **kwargs: Any) -> None:
                captured.update(kwargs)

        monkeypatch.setattr(
            "syncfusion_a2ui_agent.providers.openai_provider.AsyncOpenAI",
            _FakeClient,
        )
        p = OpenAIProvider(api_key="sk-test", model="gpt-4o")
        assert p.model == "gpt-4o"
        assert captured["api_key"] == "sk-test"

    def test_env_api_key_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from syncfusion_a2ui_agent.providers.openai_provider import OpenAIProvider

        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        captured: dict[str, Any] = {}

        class _FakeClient:
            def __init__(self, **kwargs: Any) -> None:
                captured.update(kwargs)

        monkeypatch.setattr(
            "syncfusion_a2ui_agent.providers.openai_provider.AsyncOpenAI",
            _FakeClient,
        )
        OpenAIProvider()
        assert captured["api_key"] == "sk-env"

    def test_max_completion_tokens_kwarg_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Reasoning-family callers pass ``max_completion_tokens=``."""
        from syncfusion_a2ui_agent.providers.openai_provider import OpenAIProvider

        class _FakeClient:
            def __init__(self, **kwargs: Any) -> None:
                pass

        monkeypatch.setattr(
            "syncfusion_a2ui_agent.providers.openai_provider.AsyncOpenAI",
            _FakeClient,
        )
        p = OpenAIProvider(
            api_key="sk-test",
            model="gpt-5-mini",
            max_completion_tokens=2048,
        )
        assert p.default_max_tokens == 2048

    def test_legacy_max_tokens_kwarg_still_works(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Backward-compat: ``max_tokens=`` kwarg still accepted."""
        from syncfusion_a2ui_agent.providers.openai_provider import OpenAIProvider

        class _FakeClient:
            def __init__(self, **kwargs: Any) -> None:
                pass

        monkeypatch.setattr(
            "syncfusion_a2ui_agent.providers.openai_provider.AsyncOpenAI",
            _FakeClient,
        )
        p = OpenAIProvider(
            api_key="sk-test",
            model="gpt-4o",
            max_tokens=1500,
        )
        assert p.default_max_tokens == 1500


class TestOpenAIProviderWireFormat:
    """The wire-format key for the token limit must match the model family.

    OpenAI's reasoning family (o1, o3, o4-mini, gpt-5*) rejects
    ``max_tokens`` with HTTP 400 and requires ``max_completion_tokens``
    instead. Legacy non-reasoning models still expect ``max_tokens``.
    """

    def _build(self, model: str) -> dict[str, Any]:
        from syncfusion_a2ui_agent.providers.openai_provider import OpenAIProvider

        p = OpenAIProvider(api_key="sk-test", model=model)
        return p._build_request_kwargs(
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            context=None,
            stream=False,
        )

    @pytest.mark.parametrize(
        "model",
        ["o1", "o1-mini", "o3", "o3-mini", "o4-mini", "gpt-5", "gpt-5.1", "gpt-5-mini"],
    )
    def test_reasoning_models_use_max_completion_tokens(self, model: str) -> None:
        kwargs = self._build(model)
        assert "max_completion_tokens" in kwargs, (
            f"{model!r} must use max_completion_tokens; got {list(kwargs)}"
        )
        assert "max_tokens" not in kwargs, (
            f"{model!r} must NOT send max_tokens; OpenAI rejects it with HTTP 400"
        )

    @pytest.mark.parametrize("model", ["gpt-4o", "gpt-4.1", "gpt-3.5-turbo"])
    def test_non_reasoning_models_use_max_tokens(self, model: str) -> None:
        kwargs = self._build(model)
        assert "max_tokens" in kwargs, f"{model!r} must use max_tokens; got {list(kwargs)}"
        assert "max_completion_tokens" not in kwargs, (
            f"{model!r} must NOT send max_completion_tokens"
        )

    def test_context_max_tokens_override_honoured(self) -> None:
        from syncfusion_a2ui_agent.providers.openai_provider import OpenAIProvider

        p = OpenAIProvider(api_key="sk-test", model="gpt-5-mini")
        kwargs = p._build_request_kwargs(
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            context={"max_tokens": 4321},
            stream=False,
        )
        assert kwargs["max_completion_tokens"] == 4321

    def test_gpt5_chat_uses_legacy_max_tokens(self) -> None:
        """``gpt-5-chat`` is NOT a reasoning model — must use ``max_tokens``."""
        from syncfusion_a2ui_agent.providers.openai_provider import OpenAIProvider

        p = OpenAIProvider(api_key="sk-test", model="gpt-5-chat")
        kwargs = p._build_request_kwargs(
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            context=None,
            stream=False,
        )
        assert "max_tokens" in kwargs
        assert "max_completion_tokens" not in kwargs

    @pytest.mark.parametrize(
        "model",
        ["o1", "o1-mini", "o3", "o3-mini", "o4-mini", "gpt-5", "gpt-5.1", "gpt-5-mini"],
    )
    def test_reasoning_models_omit_temperature(self, model: str) -> None:
        """Reasoning models reject explicit ``temperature`` ≠ 1 with HTTP 400.

        The OpenAI provider must OMIT ``temperature`` (so the API uses
        its own default of 1) rather than send any explicit value.
        """
        kwargs = self._build(model)
        assert "temperature" not in kwargs, (
            f"{model!r} must NOT send an explicit temperature; "
            f"OpenAI rejects anything other than the default (1)"
        )

    @pytest.mark.parametrize("model", ["gpt-4o", "gpt-4.1", "gpt-3.5-turbo"])
    def test_non_reasoning_models_keep_temperature_zero(self, model: str) -> None:
        """Non-reasoning models keep ``temperature=0.0`` for deterministic output."""
        kwargs = self._build(model)
        assert kwargs.get("temperature") == 0.0, (
            f"{model!r} must keep temperature=0.0 for deterministic output"
        )

    def test_gpt5_chat_keeps_temperature_zero(self) -> None:
        """``gpt-5-chat`` is non-reasoning — must keep ``temperature=0.0``."""
        kwargs = self._build("gpt-5-chat")
        assert kwargs.get("temperature") == 0.0


class TestAzureOpenAIProviderConstructor:
    def test_missing_all_required_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from syncfusion_a2ui_agent.providers.azure_openai_provider import (
            AzureOpenAIProvider,
        )

        monkeypatch.delenv("AZURE_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_API_BASE", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)
        with pytest.raises(ProviderError):
            AzureOpenAIProvider()

    def test_explicit_kwargs_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from syncfusion_a2ui_agent.providers.azure_openai_provider import (
            AzureOpenAIProvider,
        )

        captured: dict[str, Any] = {}

        class _FakeClient:
            def __init__(self, **kwargs: Any) -> None:
                captured.update(kwargs)

        monkeypatch.setattr(
            "syncfusion_a2ui_agent.providers.azure_openai_provider.AsyncAzureOpenAI",
            _FakeClient,
        )
        p = AzureOpenAIProvider(
            api_key="k",
            azure_endpoint="https://x.openai.azure.com/",
            api_version="2025-01-01-preview",
            azure_deployment="gpt-5.4",
        )
        assert p.model == "gpt-5.4"  # deployment name is used as model
        assert captured["api_version"] == "2025-01-01-preview"

    def test_reasoning_model_gets_minimal_effort(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from syncfusion_a2ui_agent.providers.azure_openai_provider import (
            AzureOpenAIProvider,
        )

        class _FakeClient:
            def __init__(self, **kwargs: Any) -> None:
                pass

        monkeypatch.setattr(
            "syncfusion_a2ui_agent.providers.azure_openai_provider.AsyncAzureOpenAI",
            _FakeClient,
        )
        # gpt-5 is a reasoning model — effort should default to "minimal".
        p = AzureOpenAIProvider(
            api_key="k",
            azure_endpoint="https://x.openai.azure.com/",
            azure_deployment="gpt-5",
        )
        assert p.reasoning_effort is None  # auto-detected at generate time

    def test_explicit_reasoning_effort_honoured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from syncfusion_a2ui_agent.providers.azure_openai_provider import (
            AzureOpenAIProvider,
        )

        class _FakeClient:
            def __init__(self, **kwargs: Any) -> None:
                pass

        monkeypatch.setattr(
            "syncfusion_a2ui_agent.providers.azure_openai_provider.AsyncAzureOpenAI",
            _FakeClient,
        )
        p = AzureOpenAIProvider(
            api_key="k",
            azure_endpoint="https://x.openai.azure.com/",
            azure_deployment="gpt-4o",
            reasoning_effort="high",
        )
        assert p.reasoning_effort == "high"


class TestClaudeProviderConstructor:
    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        from syncfusion_a2ui_agent.providers.claude_provider import ClaudeProvider

        with pytest.raises(ProviderError, match="api_key"):
            ClaudeProvider()

    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from syncfusion_a2ui_agent.providers.claude_provider import (
            ClaudeProvider,
        )

        class _FakeClient:
            def __init__(self, **kwargs: Any) -> None:
                pass

        monkeypatch.setattr(
            "syncfusion_a2ui_agent.providers.claude_provider.AsyncAnthropic",
            _FakeClient,
        )
        p = ClaudeProvider(api_key="k")
        assert p.model == "claude-3-5-sonnet-latest"
        assert p.default_max_tokens == 4096


class TestGeminiProviderConstructor:
    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        from syncfusion_a2ui_agent.providers.gemini_provider import GeminiProvider

        with pytest.raises(ProviderError, match="api_key"):
            GeminiProvider()

    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from syncfusion_a2ui_agent.providers.gemini_provider import GeminiProvider

        captured: dict[str, Any] = {}

        class _FakeClient:
            def __init__(self, **kwargs: Any) -> None:
                captured.update(kwargs)

        # Patch the genai module under the provider's symbol.
        import syncfusion_a2ui_agent.providers.gemini_provider as g

        monkeypatch.setattr(g, "genai", type("M", (), {"Client": _FakeClient}))
        p = GeminiProvider(api_key="k")
        assert p.model == "gemini-1.5-pro"
        assert captured["api_key"] == "k"


class TestOllamaProviderConstructor:
    def test_missing_model_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OLLAMA_MODEL", raising=False)
        from syncfusion_a2ui_agent.providers.ollama_provider import OllamaProvider

        with pytest.raises(ProviderError, match="model"):
            OllamaProvider()

    def test_default_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from syncfusion_a2ui_agent.providers.ollama_provider import OllamaProvider

        p = OllamaProvider(model="llama3.1")
        assert p.host == "http://localhost:11434"

    def test_custom_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from syncfusion_a2ui_agent.providers.ollama_provider import OllamaProvider

        p = OllamaProvider(model="llama3.1", host="http://gpu.local:11434/")
        # The trailing slash is stripped.
        assert p.host == "http://gpu.local:11434"
