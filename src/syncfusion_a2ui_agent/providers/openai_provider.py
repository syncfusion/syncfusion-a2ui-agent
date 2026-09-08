"""OpenAI provider — official ``openai`` Python SDK, native tool-calling.

Tool/stream/usage plumbing is shared with all other providers via
:mod:`syncfusion_a2ui_agent.providers._common`.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import AsyncIterator, Iterable
from typing import Any

from ._common import (
    context_max_tokens,
    mcp_tools_to_openai,
    openai_response_to_tool_calls,
    usage_to_dict,
)
from .base import AIProvider, ModelResponse, ProviderError, StreamEvent, ToolCall

# Re-exported for callers (and tests) that imported the names
# directly. The implementations live in ``_common`` so all five
# providers share them.
_to_openai_tools = mcp_tools_to_openai
_from_openai_response = openai_response_to_tool_calls

try:  # pragma: no cover
    from openai import AsyncOpenAI
except Exception:  # pragma: no cover
    AsyncOpenAI = None  # type: ignore[assignment, misc]


# Matches OpenAI/Azure reasoning-family model names. The negative
# lookahead excludes non-reasoning variants (gpt-5-chat, gpt-5-turbo) that
# share the "gpt-5" prefix but do not accept the ``reasoning_effort`` /
# ``verbosity`` parameters. The o1 / o3 / o4-mini families all accept
# the parameter and have no non-reasoning variants at the time of
# writing.
_REASONING_MODEL_RE = re.compile(
    r"^(o1|o3|o4-mini|gpt-5(?!-chat|-turbo))",
    re.IGNORECASE,
)


def _supports_reasoning_effort(model: str) -> bool:
    """True for OpenAI/Azure reasoning-family models."""
    base = model.split("/", 1)[-1] if "/" in model else model
    return bool(_REASONING_MODEL_RE.match(base))


class OpenAIProvider(AIProvider):
    """OpenAI provider with native tool-calling + streaming."""

    name = "openai"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if AsyncOpenAI is None:
            raise ProviderError("openai is not installed. `pip install openai`.")
        api_key = kwargs.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ProviderError("OpenAIProvider requires `api_key` (or OPENAI_API_KEY).")
        self.model = kwargs.get("model", "gpt-4o")
        self.reasoning_effort = kwargs.get("reasoning_effort")
        # Accept both the legacy ``max_tokens=`` kwarg and OpenAI's
        # newer ``max_completion_tokens=``. Reasoning-family models
        # (o1, o3, o4-mini, gpt-5*) reject ``max_tokens`` with HTTP 400,
        # so callers targeting those models should use the latter.
        self.default_max_tokens = int(
            kwargs.get("max_completion_tokens") or kwargs.get("max_tokens") or 8000
        )
        self._client = AsyncOpenAI(
            api_key=api_key,
            organization=kwargs.get("organization"),
            # Allow pointing the OpenAI SDK at an OpenAI-compatible
            # endpoint (e.g. DeepSeek, Together, OpenRouter, a local
            # vLLM/Ollama server, etc.). The ``openai`` Python SDK
            # forwards this as the ``base_url`` of every request.
            base_url=kwargs.get("base_url"),
        )

    def _build_request_kwargs(
        self,
        messages: list[dict[str, Any]],
        tools: Iterable[dict[str, Any]] | None,
        context: dict[str, Any] | None,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        # Reasoning-family models (o1, o3, o4-mini, gpt-5*) reject any
        # explicit ``temperature`` other than the default (1) with
        # HTTP 400. The safest portable behaviour is to omit the
        # parameter entirely for reasoning models so OpenAI falls back
        # to its own default. Non-reasoning models get
        # ``temperature=0.0`` (deterministic output).
        if not _supports_reasoning_effort(self.model):
            kwargs["temperature"] = 0.0
        # Reasoning-family models (o1, o3, o4-mini, gpt-5*) reject
        # ``max_tokens`` with HTTPreject ``max_tokens`` with HTTP 400 and
        # require ``max_completion_tokens`` instead. Non-reasoning
        # models already has the right prefix + negative-lookout logic.
        token_limit_key = (
            "max_completion_tokens" if _supports_reasoning_effort(self.model) else "max_tokens"
        )
        kwargs[token_limit_key] = context_max_tokens(context, self.default_max_tokens)
        effort = self.reasoning_effort
        if effort is None:
            effort = "minimal" if _supports_reasoning_effort(self.model) else None
        if effort:
            kwargs["reasoning_effort"] = effort
            kwargs["verbosity"] = "low"
        mcp_tools = list(tools) if tools else None
        if mcp_tools:
            kwargs["tools"] = mcp_tools_to_openai(mcp_tools)
            if not effort or effort != "minimal":
                kwargs["parallel_tool_calls"] = True
        if stream:
            kwargs["stream"] = True
            kwargs["stream_options"] = {"include_usage": True}
        return kwargs

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: Iterable[dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
    ) -> ModelResponse:
        kwargs = self._build_request_kwargs(messages, tools, context, stream=False)
        mcp_tools = list(tools) if tools else None
        try:
            resp = await self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # pragma: no cover
            raise ProviderError(f"OpenAI generate failed: {exc}") from exc

        content = ""
        if resp.choices:
            content = resp.choices[0].message.content or ""
        calls, raw_message = openai_response_to_tool_calls(resp) if mcp_tools else ([], None)
        return ModelResponse(
            content=content,
            model=self.model,
            raw=resp,
            tool_calls=calls,
            raw_message=raw_message,
            usage=usage_to_dict(getattr(resp, "usage", None)),
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: Iterable[dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        kwargs = self._build_request_kwargs(messages, tools, context, stream=True)
        content_buf: list[str] = []
        pending_tcs: dict[int, dict[str, Any]] = {}
        usage_payload: dict[str, Any] = {}

        try:
            stream = await self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # pragma: no cover
            yield StreamEvent(done=True, error=f"openai stream failed: {exc}")
            return

        try:
            async for chunk in stream:
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage is not None:
                    usage_payload = usage_to_dict(chunk_usage)
                if not getattr(chunk, "choices", None):
                    continue
                choice = chunk.choices[0]
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue

                text_delta = getattr(delta, "content", None)
                if text_delta:
                    content_buf.append(text_delta)
                    yield StreamEvent(delta=text_delta)

                for tc_delta in getattr(delta, "tool_calls", None) or []:
                    idx = getattr(tc_delta, "index", 0) or 0
                    slot = pending_tcs.setdefault(
                        idx,
                        {"id": None, "name": "", "arguments_buf": "", "yields": False},
                    )
                    if getattr(tc_delta, "id", None):
                        slot["id"] = tc_delta.id
                    fn = getattr(tc_delta, "function", None)
                    if fn is not None:
                        if getattr(fn, "name", None):
                            slot["name"] = slot["name"] + fn.name
                        if getattr(fn, "arguments", None):
                            slot["arguments_buf"] += fn.arguments
                    if not slot["yields"] and slot["name"] and slot["arguments_buf"]:
                        try:
                            parsed = json.loads(slot["arguments_buf"])
                        except Exception:
                            parsed = None
                        if parsed is not None:
                            slot["yields"] = True
                            yield StreamEvent(
                                tool_call=ToolCall(
                                    name=slot["name"],
                                    arguments=parsed,
                                    id=slot["id"],
                                )
                            )
        except Exception as exc:  # pragma: no cover
            yield StreamEvent(done=True, error=f"openai stream failed: {exc}")
            return

        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(content_buf),
        }
        if pending_tcs:
            assistant_msg["tool_calls"] = [
                {
                    "id": pending_tcs[i]["id"],
                    "type": "function",
                    "function": {
                        "name": pending_tcs[i]["name"],
                        "arguments": pending_tcs[i]["arguments_buf"],
                    },
                }
                for i in sorted(pending_tcs.keys())
            ]
        yield StreamEvent(
            response=ModelResponse(
                content="".join(content_buf),
                model=self.model,
                raw=None,
                tool_calls=[
                    ToolCall(
                        name=pending_tcs[i]["name"],
                        arguments=json.loads(pending_tcs[i]["arguments_buf"] or "{}"),
                        id=pending_tcs[i]["id"],
                    )
                    for i in sorted(pending_tcs.keys())
                ],
                raw_message=assistant_msg,
                usage=usage_payload,
            ),
            done=True,
        )
