"""Ollama provider — local model server over HTTP, with streaming + tool-calling.

Uses :mod:`syncfusion_a2ui_agent.providers._common` for tool/usage helpers.
The wire shape is OpenAI-compatible when ``tools`` is supplied, so we
reuse :func:`openai_response_to_tool_calls` against Ollama's response.

Ollama's tool-calling support is model/version-dependent; if the local
model does not support tools the server returns plain text and the SDK
just sees no ``tool_calls``.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Iterable
from typing import Any

import httpx

from ._common import (
    context_max_tokens,
    mcp_tools_to_openai,
    split_system_messages,
)
from .base import AIProvider, ModelResponse, ProviderError, StreamEvent


class OllamaProvider(AIProvider):
    """Ollama provider with streaming + best-effort tool-calling."""

    name = "ollama"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model: str = kwargs.get("model") or os.environ.get("OLLAMA_MODEL") or ""
        if not self.model:
            raise ProviderError("OllamaProvider requires a `model` (or OLLAMA_MODEL).")
        self.host = (
            kwargs.get("host") or os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
        ).rstrip("/")
        self.default_timeout = float(kwargs.get("timeout", 120))
        # Total output budget; Ollama uses ``num_predict``.
        self.default_max_tokens = int(kwargs.get("max_tokens", 4096))

    def _build_request(
        self,
        messages: list[dict[str, Any]],
        tools: Iterable[dict[str, Any]] | None,
        context: dict[str, Any] | None,
        *,
        stream: bool,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        msgs, system_parts = split_system_messages(messages)
        # Ollama expects messages in OpenAI's shape with ``role`` /
        # ``content``. We can pass the SDK's messages verbatim — system
        # messages are flattened into a leading system message.
        if system_parts:
            msgs = [{"role": "system", "content": "\n\n".join(system_parts)}] + msgs
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": msgs,
            "stream": stream,
        }
        opts: dict[str, Any] = {}
        num_predict = context_max_tokens(context, self.default_max_tokens)
        if num_predict:
            opts["num_predict"] = num_predict
        if opts:
            payload["options"] = opts
        mcp_tools = list(tools) if tools else None
        if mcp_tools:
            payload["tools"] = mcp_tools_to_openai(mcp_tools)
        return payload, context

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: Iterable[dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
    ) -> ModelResponse:
        payload, _ = self._build_request(messages, tools, context, stream=False)
        timeout = (
            float(context.get("timeout", self.default_timeout)) if context else self.default_timeout
        )
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                resp = await client.post(f"{self.host}/api/chat", json=payload)
                resp.raise_for_status()
            except Exception as exc:  # pragma: no cover
                raise ProviderError(f"Ollama generate failed: {exc}") from exc
        data = resp.json()
        msg = data.get("message") or {}
        content = msg.get("content", "") or ""

        list(tools) if tools else None
        # If the model returned tool_calls we surface them; otherwise
        # they remain an empty list.
        otc = msg.get("tool_calls") or []
        from .base import ToolCall

        calls: list[Any] = []
        for c in otc:
            fn = c.get("function") or {}
            calls.append(
                ToolCall(
                    name=fn.get("name", ""),
                    arguments=fn.get("arguments") or {},
                    id=None,
                )
            )
        raw_message: dict[str, Any] | None = None
        if calls:
            raw_message = {
                "role": "assistant",
                "content": content,
                "tool_calls": otc,
            }

        return ModelResponse(
            content=content,
            model=self.model,
            raw=data,
            tool_calls=calls,
            raw_message=raw_message,
            # Ollama returns prompt_eval_count / eval_count — map to tokens.
            usage={
                "prompt_tokens": data.get("prompt_eval_count") or 0,
                "completion_tokens": data.get("eval_count") or 0,
            },
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: Iterable[dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Token-streaming via Ollama's ``/api/chat`` NDJSON stream."""

        payload, _ = self._build_request(messages, tools, context, stream=True)
        timeout = (
            float(context.get("timeout", self.default_timeout)) if context else self.default_timeout
        )

        content_buf: list[str] = []
        pending_tcs: dict[int, dict[str, Any]] = {}
        usage_payload: dict[str, Any] = {}
        # Ollama streams finish with a final chunk whose ``done`` flag is True.
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", f"{self.host}/api/chat", json=payload) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if data.get("error"):
                            yield StreamEvent(done=True, error="ollama: " + str(data["error"]))
                            return
                        msg = data.get("message") or {}
                        chunk_text = msg.get("content", "") or ""
                        if chunk_text:
                            content_buf.append(chunk_text)
                            yield StreamEvent(delta=chunk_text)
                        # Tool-call deltas (Ollama 0.5+).
                        for idx, c in enumerate(msg.get("tool_calls") or []):
                            fn = c.get("function") or {}
                            slot = pending_tcs.setdefault(
                                idx,
                                {
                                    "name": fn.get("name", ""),
                                    "arguments_buf": "",
                                    "yields": False,
                                },
                            )
                            if fn.get("name"):
                                slot["name"] = fn["name"]
                            args = fn.get("arguments")
                            if isinstance(args, dict):
                                slot["arguments_buf"] = json.dumps(args)
                                if not slot["yields"]:
                                    from .base import ToolCall

                                    slot["yields"] = True
                                    yield StreamEvent(
                                        tool_call=ToolCall(
                                            name=slot["name"],
                                            arguments=args,
                                            id=None,
                                        )
                                    )
                            elif isinstance(args, str):
                                slot["arguments_buf"] += args
                                if slot["name"] and slot["arguments_buf"] and not slot["yields"]:
                                    from .base import ToolCall

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
                                                id=None,
                                            )
                                        )
                        if data.get("done"):
                            usage_payload = {
                                "prompt_tokens": data.get("prompt_eval_count") or 0,
                                "completion_tokens": data.get("eval_count") or 0,
                            }
        except Exception as exc:  # pragma: no cover
            yield StreamEvent(done=True, error=f"ollama stream failed: {exc}")
            return

        from .base import ToolCall

        calls: list[Any] = []
        raw_message_parts: list[dict[str, Any]] = []
        if content_buf:
            raw_message_parts.append({"type": "text", "text": "".join(content_buf)})
        for slot in pending_tcs.values():
            try:
                args = json.loads(slot["arguments_buf"]) if slot["arguments_buf"] else {}
            except Exception:
                args = {}
            if not isinstance(args, dict):
                args = {}
            calls.append(ToolCall(name=slot["name"], arguments=args, id=None))
            raw_message_parts.append({"type": "tool_call", "name": slot["name"], "arguments": args})
        raw_message: dict[str, Any] | None = None
        if raw_message_parts:
            raw_message = {"role": "assistant", "content": raw_message_parts}

        yield StreamEvent(
            response=ModelResponse(
                content="".join(content_buf),
                model=self.model,
                tool_calls=calls,
                raw_message=raw_message,
                usage=usage_payload,
            ),
            done=True,
        )
