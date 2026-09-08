"""Direct tests for :class:`OllamaProvider`.

Covers:

* Constructor wiring (env-var fallback, default model/host).
* :meth:`OllamaProvider._build_request` — payload shape,
  ``stream`` flag, tool schema.
* :meth:`OllamaProvider.generate` — happy path via a fake
  :class:`httpx.AsyncClient`.
* :meth:`OllamaProvider.stream` — NDJSON parsing, tool-call
  accumulation.

Ollama uses ``httpx`` directly (no SDK), so we patch the
``httpx.AsyncClient`` to a fake.
"""

from __future__ import annotations

import json
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.status_code = 200

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeStreamResponse:
    def __init__(self, ndjson_lines: list[str]) -> None:
        self._lines = ndjson_lines

    async def __aenter__(self) -> "_FakeStreamResponse":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self) -> Any:
        async def _gen() -> Any:
            for line in self._lines:
                yield line

        return _gen()

    def aiter_lines(self) -> Any:
        async def _gen() -> Any:
            for line in self._lines:
                yield line

        return _gen()


class _FakeAsyncClient:
    def __init__(self, response: Any = None, stream_response: Any = None) -> None:
        self._response = response
        self._stream_response = stream_response
        self.posted: list[dict[str, Any]] = []

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def post(self, url: str, **kwargs: Any) -> Any:
        self.posted.append({"url": url, **kwargs})
        if self._stream_response is not None:
            return self._stream_response
        return self._response

    def stream(self, method: str, url: str, **kwargs: Any) -> Any:
        # The Ollama provider uses ``client.stream("POST", ...)``
        # which returns an async context manager. Our fake returns
        # a stream response (which itself is an async context
        # manager).
        self.posted.append({"method": method, "url": url, **kwargs})
        return self._stream_response


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------
class TestOllamaConstructor:
    def test_default_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from syncfusion_a2ui_agent.providers.ollama_provider import OllamaProvider

        p = OllamaProvider(model="llama3")
        assert p.model == "llama3"
        assert p.host == "http://localhost:11434"
        assert p.default_max_tokens == 4096
        # The attribute is named ``default_timeout`` (not ``timeout``).
        assert p.default_timeout == 120.0

    def test_env_host_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from syncfusion_a2ui_agent.providers.ollama_provider import OllamaProvider

        monkeypatch.setenv("OLLAMA_HOST", "http://gpu-box.local:11434")
        p = OllamaProvider(model="llama3")
        assert p.host == "http://gpu-box.local:11434"

    def test_explicit_host_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from syncfusion_a2ui_agent.providers.ollama_provider import OllamaProvider

        monkeypatch.setenv("OLLAMA_HOST", "http://env-host:11434")
        p = OllamaProvider(model="llama3", host="http://explicit:9999")
        assert p.host == "http://explicit:9999"

    def test_model_required(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The constructor requires an explicit model — no default
        # model name is shipped. This pins the contract.
        from syncfusion_a2ui_agent.providers.base import ProviderError
        from syncfusion_a2ui_agent.providers.ollama_provider import OllamaProvider

        monkeypatch.delenv("OLLAMA_MODEL", raising=False)
        with pytest.raises(ProviderError, match="requires a `model`"):
            OllamaProvider()

    def test_env_model_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from syncfusion_a2ui_agent.providers.ollama_provider import OllamaProvider

        monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:7b")
        p = OllamaProvider()
        assert p.model == "qwen2.5:7b"


# ---------------------------------------------------------------------------
# _build_request
# ---------------------------------------------------------------------------
class TestOllamaBuildRequest:
    def test_shape_with_tools(self) -> None:
        from syncfusion_a2ui_agent.providers.ollama_provider import OllamaProvider

        p = OllamaProvider(model="llama3")
        payload, _ = p._build_request(
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
        assert payload["model"] == "llama3"
        assert payload["messages"] == [{"role": "user", "content": "hi"}]
        assert payload["stream"] is False
        assert "tools" in payload
        assert payload["tools"][0]["function"]["name"] == "demo__ping"

    def test_stream_flag(self) -> None:
        from syncfusion_a2ui_agent.providers.ollama_provider import OllamaProvider

        p = OllamaProvider(model="llama3")
        payload, _ = p._build_request(
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            context=None,
            stream=True,
        )
        assert payload["stream"] is True

    def test_no_tools_omits_tools_key(self) -> None:
        from syncfusion_a2ui_agent.providers.ollama_provider import OllamaProvider

        p = OllamaProvider(model="llama3")
        payload, _ = p._build_request(
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            context=None,
            stream=False,
        )
        assert "tools" not in payload

    def test_num_predict_in_options(self) -> None:
        # The token-budget is exposed via Ollama's ``options.num_predict``
        # field, not via the context.
        from syncfusion_a2ui_agent.providers.ollama_provider import OllamaProvider

        p = OllamaProvider(model="llama3")
        payload, _ = p._build_request(
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            context=None,
            stream=False,
        )
        assert payload["options"]["num_predict"] == 4096

    def test_returns_original_context(self) -> None:
        from syncfusion_a2ui_agent.providers.ollama_provider import OllamaProvider

        p = OllamaProvider(model="llama3")
        ctx = {"timeout": 60.0}
        _, returned = p._build_request(
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            context=ctx,
            stream=False,
        )
        # The build_request returns the original context unchanged.
        assert returned is ctx


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------
class TestOllamaGenerate:
    def test_happy_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from syncfusion_a2ui_agent.providers import ollama_provider as mod
        from syncfusion_a2ui_agent.providers.ollama_provider import OllamaProvider

        response_payload = {
            "message": {
                "role": "assistant",
                "content": "hello back",
            },
            "prompt_eval_count": 5,
            "eval_count": 3,
            "done": True,
        }
        fake = _FakeAsyncClient(response=_FakeResponse(response_payload))
        monkeypatch.setattr(mod.httpx, "AsyncClient", lambda **kw: fake)

        p = OllamaProvider(model="llama3")

        async def run() -> Any:
            return await p.generate(
                messages=[{"role": "user", "content": "hi"}],
                tools=None,
                context=None,
            )

        import asyncio

        result = asyncio.run(run())
        assert result.content == "hello back"
        assert result.tool_calls == []
        assert result.usage == {"prompt_tokens": 5, "completion_tokens": 3}

    def test_tool_call_parsing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from syncfusion_a2ui_agent.providers import ollama_provider as mod
        from syncfusion_a2ui_agent.providers.ollama_provider import OllamaProvider

        response_payload = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "demo__ping",
                            "arguments": {"q": "hi"},
                        }
                    }
                ],
            },
            "done": True,
        }
        fake = _FakeAsyncClient(response=_FakeResponse(response_payload))
        monkeypatch.setattr(mod.httpx, "AsyncClient", lambda **kw: fake)

        p = OllamaProvider(model="llama3")

        async def run() -> Any:
            return await p.generate(
                messages=[{"role": "user", "content": "hi"}],
                tools=None,
                context=None,
            )

        import asyncio

        result = asyncio.run(run())
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "demo__ping"
        assert result.tool_calls[0].arguments == {"q": "hi"}

    def test_http_error_raises_provider_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # HTTP failures (4xx/5xx via raise_for_status) are wrapped
        # in ProviderError. A response that contains a body-level
        # ``error`` key but is HTTP 200 is treated as a normal
        # response (the model returned an error message) — that
        # is the streaming path's responsibility to surface.
        from syncfusion_a2ui_agent.providers import ollama_provider as mod
        from syncfusion_a2ui_agent.providers.base import ProviderError
        from syncfusion_a2ui_agent.providers.ollama_provider import OllamaProvider

        class _FailingResponse:
            status_code = 404

            def raise_for_status(self) -> None:
                raise RuntimeError("HTTP 404")

            def json(self) -> dict[str, Any]:
                return {}

        fake = _FakeAsyncClient(response=_FailingResponse())
        monkeypatch.setattr(mod.httpx, "AsyncClient", lambda **kw: fake)

        p = OllamaProvider(model="missing")

        async def run() -> Any:
            return await p.generate(
                messages=[{"role": "user", "content": "hi"}],
                tools=None,
                context=None,
            )

        import asyncio

        with pytest.raises(ProviderError, match="Ollama generate failed"):
            asyncio.run(run())


# ---------------------------------------------------------------------------
# stream()
# ---------------------------------------------------------------------------
class TestOllamaStream:
    def test_ndjson_parsing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from syncfusion_a2ui_agent.providers import ollama_provider as mod
        from syncfusion_a2ui_agent.providers.ollama_provider import OllamaProvider

        # The provider yields the chunk's ``content`` field directly
        # as the ``delta`` (the consumer is expected to treat each
        # delta as a chunk of the final text, not a diff).
        lines = [
            json.dumps({"message": {"role": "assistant", "content": "hello"}}),
            json.dumps({"message": {"role": "assistant", "content": " world"}}),
            json.dumps(
                {
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                    "prompt_eval_count": 4,
                    "eval_count": 2,
                }
            ),
        ]
        fake = _FakeAsyncClient(stream_response=_FakeStreamResponse(lines))
        monkeypatch.setattr(mod.httpx, "AsyncClient", lambda **kw: fake)

        p = OllamaProvider(model="llama3")

        async def run() -> list[Any]:
            events = []
            async for ev in p.stream(
                messages=[{"role": "user", "content": "hi"}],
                tools=None,
                context=None,
            ):
                events.append(ev)
            return events

        import asyncio

        events = asyncio.run(run())
        deltas = [e.delta for e in events if e.delta]
        assert deltas == ["hello", " world"]
        # Final event is ``done=True`` with usage.
        final = next(e for e in events if e.done)
        assert final.response is not None
        assert final.response.usage == {"prompt_tokens": 4, "completion_tokens": 2}
