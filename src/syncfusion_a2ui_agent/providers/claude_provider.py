"""Anthropic Claude provider — official ``anthropic`` SDK, native tool-calling.

Uses :mod:`syncfusion_a2ui_agent.providers._common` for tool conversion,
usage extraction, and per-call ``max_tokens`` override. Implements both
``generate()`` and ``stream()`` so it ships full feature parity with
OpenAI/Azure.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator, Iterable
from typing import Any

from ._common import (
    anthropic_response_to_tool_calls,
    context_max_tokens,
    mcp_tools_to_anthropic,
    split_system_messages,
    usage_to_dict,
)
from .base import AIProvider, ModelResponse, ProviderError, StreamEvent

try:  # pragma: no cover
    from anthropic import AsyncAnthropic
except Exception:  # pragma: no cover
    AsyncAnthropic = None  # type: ignore[assignment, misc]

logger = logging.getLogger("syncfusion_a2ui_agent.providers.claude")


class ClaudeProvider(AIProvider):
    """Anthropic Claude provider with native tool-calling + streaming."""

    name = "claude"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if AsyncAnthropic is None:
            raise ProviderError("anthropic is not installed. `pip install anthropic`.")
        api_key = kwargs.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ProviderError("ClaudeProvider requires `api_key` (or ANTHROPIC_API_KEY).")
        self.model = kwargs.get("model", "claude-3-5-sonnet-latest")
        # Bumped default from 1024 → 4096 so first-time users get a usable
        # A2UI envelope instead of a truncated response.
        self.default_max_tokens = int(kwargs.get("max_tokens", 4096))
        self._client = AsyncAnthropic(api_key=api_key)

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: Iterable[dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
    ) -> ModelResponse:
        msgs, system_parts = split_system_messages(messages)
        system = "\n\n".join(system_parts) if system_parts else None
        max_tokens = context_max_tokens(context, self.default_max_tokens)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": msgs,
        }
        if system:
            kwargs["system"] = system
        mcp_tools = list(tools) if tools else None
        if mcp_tools:
            kwargs["tools"] = mcp_tools_to_anthropic(mcp_tools)

        try:
            resp = await self._client.messages.create(**kwargs)
        except Exception as exc:  # pragma: no cover
            raise ProviderError(f"Claude generate failed: {exc}") from exc

        calls, raw_message = anthropic_response_to_tool_calls(resp) if mcp_tools else ([], None)

        # Pull text out of the content blocks for the SDK's ``content``.
        text_chunks: list[str] = []
        for block in getattr(resp, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                text_chunks.append(text)

        return ModelResponse(
            content="".join(text_chunks),
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
        """Token-streaming via Anthropic's ``messages.stream``."""

        msgs, system_parts = split_system_messages(messages)
        system = "\n\n".join(system_parts) if system_parts else None
        max_tokens = context_max_tokens(context, self.default_max_tokens)

        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": msgs,
        }
        if system:
            request["system"] = system
        mcp_tools = list(tools) if tools else None
        if mcp_tools:
            request["tools"] = mcp_tools_to_anthropic(mcp_tools)

        content_buf: list[str] = []
        pending_tcs: dict[str, dict[str, Any]] = {}
        usage_payload: dict[str, Any] = {}

        try:
            # The Anthropic SDK exposes an async context-manager stream.
            async with self._client.messages.stream(**request) as stream:
                async for event in stream:
                    etype = getattr(event, "type", "")
                    if etype == "content_block_start":
                        block = getattr(event, "content_block", None)
                        if block is not None and getattr(block, "type", "") == "tool_use":
                            bid = getattr(event, "index", None) or getattr(block, "id", None)
                            pending_tcs[bid] = {  # type: ignore[index]
                                "name": getattr(block, "name", ""),
                                "input_buf": "",
                                "yields": False,
                            }
                    elif etype == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        if delta is None:
                            continue
                        # Text delta
                        text = getattr(delta, "text", None)
                        if text:
                            content_buf.append(text)
                            yield StreamEvent(delta=text)
                            continue
                        # Tool-use input delta — accumulate JSON fragments.
                        if getattr(delta, "type", "") == "input_json_delta":
                            bid = getattr(event, "index", None)
                            slot = pending_tcs.get(bid)  # type: ignore[arg-type]
                            if slot is not None:
                                chunk = getattr(delta, "partial_json", "") or ""
                                slot["input_buf"] += chunk
                                if slot["name"] and slot["input_buf"] and not slot["yields"]:
                                    import json as _json

                                    from .base import ToolCall

                                    try:
                                        args = _json.loads(slot["input_buf"])
                                    except Exception:
                                        args = None
                                    if args is not None:
                                        slot["yields"] = True
                                        yield StreamEvent(
                                            tool_call=ToolCall(
                                                name=slot["name"],
                                                arguments=args,
                                                id=bid,
                                            )
                                        )
                    elif etype == "message_delta":
                        # message_delta carries an updated usage with output_tokens.
                        usage_obj = getattr(event, "usage", None)
                        if usage_obj is not None:
                            usage_payload.update(usage_to_dict(usage_obj))
                    elif etype == "message_stop":
                        # Final usage arrives here on Anthropic's stream.
                        pass
                # Final message is available after the stream context exits.
                final = await stream.get_final_message()
                final_usage = getattr(final, "usage", None)
                if final_usage is not None:
                    usage_payload.update(usage_to_dict(final_usage))
        except Exception as exc:  # pragma: no cover
            yield StreamEvent(done=True, error=f"claude stream failed: {exc}")
            return

        # Rebuild the assistant raw_message in Anthropic's content-block form.
        raw_blocks: list[dict[str, Any]] = []
        if content_buf:
            raw_blocks.append({"type": "text", "text": "".join(content_buf)})
        for bid, slot in pending_tcs.items():
            raw_blocks.append(
                {
                    "type": "tool_use",
                    "id": bid,
                    "name": slot["name"],
                    "input": _safe_loads_json(slot["input_buf"]),
                }
            )
        raw_message: dict[str, Any] | None = None
        if raw_blocks:
            raw_message = {"role": "assistant", "content": raw_blocks}

        import json as _json

        from .base import ToolCall

        calls = [
            ToolCall(
                name=slot["name"],
                arguments=_safe_loads_json(slot["input_buf"]),
                id=bid,
            )
            for bid, slot in pending_tcs.items()
        ]
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


def _safe_loads_json(s: str) -> dict[str, Any]:
    """Parse *s* as JSON, returning ``{}`` on any error."""
    import json as _json

    try:
        if not s:
            return {}
        parsed = _json.loads(s)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}
