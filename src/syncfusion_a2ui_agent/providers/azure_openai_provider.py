"""Azure OpenAI provider — official ``openai`` SDK, native tool-calling.

Shares its tool/stream/usage plumbing with ``OpenAIProvider`` via
:mod:`syncfusion_a2ui_agent.providers._common`; the only delta is the
client constructor and the way Azure deployments are addressed.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import AsyncIterator, Iterable
from typing import Any

from ._common import (
    context_max_tokens,
    mcp_tools_to_openai,
    openai_response_to_tool_calls,
    usage_to_dict,
)
from .base import AIProvider, ModelResponse, ProviderError, StreamEvent, ToolCall
from .openai_provider import _supports_reasoning_effort

try:  # pragma: no cover
    from openai import AsyncAzureOpenAI
except Exception:  # pragma: no cover
    AsyncAzureOpenAI = None  # type: ignore[assignment, misc]

logger = logging.getLogger("syncfusion_a2ui_agent.providers.azure_openai")


class AzureOpenAIProvider(AIProvider):
    """Azure OpenAI provider with native tool-calling + streaming."""

    name = "azure_openai"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if AsyncAzureOpenAI is None:
            raise ProviderError("openai is not installed. `pip install openai`.")
        api_key = (
            kwargs.get("api_key")
            or os.environ.get("AZURE_API_KEY")
            or os.environ.get("AZURE_OPENAI_API_KEY")
        )
        endpoint = (
            kwargs.get("azure_endpoint")
            or os.environ.get("AZURE_API_BASE")
            or os.environ.get("AZURE_OPENAI_ENDPOINT")
        )
        deployment = kwargs.get("azure_deployment") or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        api_version = kwargs.get("api_version") or os.environ.get(
            "AZURE_OPENAI_API_VERSION", "2024-05-01-preview"
        )
        if not (api_key and endpoint and deployment):
            raise ProviderError(
                "AzureOpenAIProvider requires `api_key`, `azure_endpoint`, and `azure_deployment`."
            )
        self.model = deployment  # Azure uses deployment name as model id
        self.reasoning_effort = kwargs.get("reasoning_effort")
        self.default_max_tokens = int(kwargs.get("max_completion_tokens", 3000))
        self._client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )

    def _build_kwargs(
        self,
        messages: list[dict[str, Any]],
        tools: Iterable[dict[str, Any]] | None,
        context: dict[str, Any] | None,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        # For A2UI JSON generation, ``0.0`` is the correct deterministic
        # default — every other provider in this package either uses
        # ``0.0`` (OpenAI non-reasoning) or omits the param (Claude/
        # Gemini). The previous hardcoded ``1.0`` produced noticeably
        # less reliable structured output for all ``examples/`` agents,
        # which default to Azure. Callers can still override per-request
        # via ``context["temperature"]`` (the same escape hatch
        # :class:`GeminiProvider` exposes).
        context_temperature: float | None = None
        if isinstance(context, dict):
            raw = context.get("temperature")
            if raw is not None:
                try:
                    context_temperature = float(raw)
                except (TypeError, ValueError):
                    context_temperature = None
        # Reasoning-family models (o1, o3, o4-mini, gpt-5*) reject any
        # explicit ``temperature`` other than the default (1) with
        # HTTP 400 — both on OpenAI and on Azure OpenAI. Match
        # :class:`OpenAIProvider` and omit the param entirely for
        # reasoning models.
        is_reasoning = _supports_reasoning_effort(self.model)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": context_max_tokens(context, self.default_max_tokens),
        }
        if not is_reasoning:
            kwargs["temperature"] = context_temperature if context_temperature is not None else 0.0
        elif context_temperature is not None:
            # Caller asked for a specific temperature, but this is a
            # reasoning model that rejects the param. Surface the
            # conflict to the user rather than silently dropping it.
            logger.warning(
                "azure build_kwargs: context['temperature']=%s was ignored "
                "because deployment %r is a reasoning model that rejects "
                "the temperature parameter.",
                context_temperature,
                self.model,
            )
        effort = self.reasoning_effort
        if effort is None:
            effort = "minimal" if is_reasoning else None
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
        kwargs = self._build_kwargs(messages, tools, context, stream=False)
        mcp_tools = list(tools) if tools else None
        t0 = time.monotonic()
        fell_back = False
        try:
            resp = await self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # pragma: no cover
            # Defensive fallback: if reasoning_effort/verbosity was guessed
            # wrong for this deployment (e.g. a new model name our regex
            # doesn't recognize yet, or an API version that rejects the
            # param), strip it and retry once rather than failing outright.
            if "reasoning_effort" in kwargs and "unrecognized" in str(exc).lower():
                fell_back = True
                kwargs.pop("reasoning_effort", None)
                kwargs.pop("verbosity", None)
                kwargs["parallel_tool_calls"] = (
                    True if mcp_tools else kwargs.get("parallel_tool_calls", False)
                )
                try:
                    resp = await self._client.chat.completions.create(**kwargs)
                except Exception as exc2:  # pragma: no cover
                    raise ProviderError(f"Azure OpenAI generate failed: {exc2}") from exc2
            else:
                raise ProviderError(f"Azure OpenAI generate failed: {exc}") from exc

        elapsed = time.monotonic() - t0
        logger.info(
            "azure generate: model=%s fell_back=%s elapsed=%.2fs prompt=%s completion=%s",
            self.model,
            fell_back,
            elapsed,
            getattr(resp.usage, "prompt_tokens", "?"),
            getattr(resp.usage, "completion_tokens", "?"),
        )

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
        kwargs = self._build_kwargs(messages, tools, context, stream=True)
        mcp_tools = list(tools) if tools else None
        content_buf: list[str] = []
        pending_tcs: dict[int, dict[str, Any]] = {}
        usage_payload: dict[str, Any] = {}
        t0 = time.monotonic()
        fell_back = False

        try:
            try:
                stream = await self._client.chat.completions.create(**kwargs)
            except Exception as exc:
                if "reasoning_effort" in kwargs and "unrecognized" in str(exc).lower():
                    fell_back = True
                    kwargs.pop("reasoning_effort", None)
                    kwargs.pop("verbosity", None)
                    if mcp_tools:
                        kwargs["parallel_tool_calls"] = True
                    stream = await self._client.chat.completions.create(**kwargs)
                else:
                    raise

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
            yield StreamEvent(done=True, error=f"azure stream failed: {exc}")
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

        logger.info(
            "azure stream: model=%s fell_back=%s elapsed=%.2fs prompt=%s completion=%s",
            self.model,
            fell_back,
            time.monotonic() - t0,
            usage_payload.get("prompt_tokens", "?"),
            usage_payload.get("completion_tokens", "?"),
        )
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
