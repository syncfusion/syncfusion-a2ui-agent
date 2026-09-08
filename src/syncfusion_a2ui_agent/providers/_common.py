"""Shared provider helpers — internal to ``providers/``.

The five built-in AI providers all need the same handful of operations:

* split OpenAI-style ``{"role":"system","content":...}`` messages out of
  the prompt list,
* convert MCP tool descriptors (flat or already-OpenAI-shaped) into the
  vendor's tool schema,
* extract usage telemetry uniformly,
* convert vendor tool-call responses back into the SDK's
  :class:`providers.base.ToolCall` + ``raw_message`` shape,
* honour a per-call ``context={"max_tokens": <int>}`` override.

Centralising this in one module means each provider file stays small,
focused on its own SDK's quirks, and inherits the same feature parity.

Nothing here is part of the public API. If you need a vendor-specific
customisation, subclass the provider and override :py:meth:`generate`
or :py:meth:`stream`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

# ---------------------------------------------------------------------------
# Message normalisation
# ---------------------------------------------------------------------------


def split_system_messages(
    messages: list[dict[str, Any]],
    *,
    system_key: str = "system",
) -> tuple[list[dict[str, Any]], list[str]]:
    """Pull every ``role="system"`` message out of *messages* and join them.

    Returns ``(non_system_messages, system_texts)``. The non-system
    messages keep their original roles; the caller is responsible for
    mapping them to the vendor's expected role names.
    """
    non_system: list[dict[str, Any]] = []
    system_parts: list[str] = []
    for m in messages:
        if m.get("role") == system_key:
            text = m.get("content", "") or ""
            if text:
                system_parts.append(text)
        else:
            non_system.append(m)
    return non_system, system_parts


def context_max_tokens(
    context: dict[str, Any] | None,
    default: int,
) -> int:
    """Return ``context["max_tokens"]`` if it is an ``int``, else *default*.

    This is the single point that honours the per-call override documented
    on :py:meth:`AIProvider.generate` / :py:meth:`AIProvider.stream`.
    """
    if context and isinstance(context.get("max_tokens"), int):
        return int(context["max_tokens"])
    return default


# ---------------------------------------------------------------------------
# Usage telemetry
# ---------------------------------------------------------------------------


def usage_to_dict(usage: Any) -> dict[str, Any]:
    """Best-effort conversion of a vendor ``usage`` object to a plain dict.

    Reads ``prompt_tokens`` and ``completion_tokens`` if present —
    everything else is silently dropped (vendors vary wildly in their
    additional fields, and the SDK only consumes these two).
    """
    if usage is None:
        return {}
    out: dict[str, Any] = {}
    for key in ("prompt_tokens", "completion_tokens"):
        val = getattr(usage, key, None)
        if val is None and isinstance(usage, dict):
            val = usage.get(key)
        if isinstance(val, (int, float)):
            out[key] = int(val)
    return out


# ---------------------------------------------------------------------------
# MCP tool descriptors → OpenAI / Anthropic / Gemini schema
# ---------------------------------------------------------------------------


def mcp_tools_to_openai(mcp_tools: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate MCP tool descriptors to OpenAI's ``tools=[...]`` schema.

    Tools already in OpenAI format (``type`` + ``function`` keys) pass
    through unchanged so the synthetic ``read_skill`` tool can be
    injected without double-wrapping.
    """
    out: list[dict[str, Any]] = []
    for t in mcp_tools:
        if "type" in t and "function" in t:
            out.append(t)
            continue
        out.append(
            {
                "type": "function",
                "function": {
                    "name": f"{t.get('server', 'mcp')}__{t.get('name', '')}",
                    "description": t.get("description") or t.get("name", ""),
                    "parameters": t.get("inputSchema") or {"type": "object", "properties": {}},
                },
            }
        )
    return out


def mcp_tools_to_anthropic(mcp_tools: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate MCP tool descriptors to Anthropic's ``tools=[...]`` schema.

    Already-shaped tools (``{"name": ..., "input_schema": ...}``) are
    passed through; everything else is normalised to Anthropic's shape.
    """
    out: list[dict[str, Any]] = []
    for t in mcp_tools:
        if "input_schema" in t and "name" in t and "type" not in t:
            out.append(t)
            continue
        out.append(
            {
                "name": f"{t.get('server', 'mcp')}__{t.get('name', '')}",
                "description": t.get("description") or t.get("name", ""),
                "input_schema": t.get("inputSchema") or {"type": "object", "properties": {}},
            }
        )
    return out


def mcp_tools_to_gemini(mcp_tools: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate MCP tool descriptors to Gemini's ``function_declarations`` schema.

    Returns the list in the shape Gemini's
    ``Tool(function_declarations=[...])`` config accepts.
    """
    decls: list[dict[str, Any]] = []
    for t in mcp_tools:
        decls.append(
            {
                "name": f"{t.get('server', 'mcp')}__{t.get('name', '')}",
                "description": t.get("description") or t.get("name", ""),
                "parameters": t.get("inputSchema") or {"type": "object", "properties": {}},
            }
        )
    return decls


# ---------------------------------------------------------------------------
# Vendor tool-call responses → SDK ToolCall + raw_message
# ---------------------------------------------------------------------------


def openai_response_to_tool_calls(
    resp: Any,
) -> tuple[list[Any], dict[str, Any] | None]:
    """Pull tool calls + a re-buildable assistant message out of an
    OpenAI/Azure response. Returns ``(calls, raw_message)``.
    """
    from .base import ToolCall  # local to avoid a circular import

    calls: list[Any] = []
    choice = (resp.choices or [None])[0]
    if not choice or not getattr(choice, "message", None):
        return calls, None
    msg = choice.message
    for tc in msg.tool_calls or []:
        fn = tc.function
        name = fn.name or ""
        args: dict[str, Any] = {}
        try:
            args = json.loads(fn.arguments) if fn.arguments else {}
        except Exception:
            args = {}
        calls.append(ToolCall(name=name, arguments=args, id=tc.id))
    raw_message: dict[str, Any] | None = None
    if calls:
        raw_message = {
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name or "",
                        "arguments": tc.function.arguments or "",
                    },
                }
                for tc in (msg.tool_calls or [])
            ],
        }
    return calls, raw_message


def anthropic_response_to_tool_calls(
    resp: Any,
) -> tuple[list[Any], dict[str, Any] | None]:
    """Pull tool calls out of an Anthropic response + a re-buildable
    assistant message in *content-block* form.

    The raw_message uses Anthropic's native content-block shape
    (``[{"type":"text","text":...},{"type":"tool_use",...}]``) so the
    SDK's tool loop can echo it back before the ``role: tool`` results.
    """
    from .base import ToolCall

    calls: list[Any] = []
    blocks: list[dict[str, Any]] = []
    raw_blocks: list[dict[str, Any]] = []
    for block in getattr(resp, "content", []) or []:
        btype = getattr(block, "type", None)
        if btype == "text":
            text = getattr(block, "text", "") or ""
            if text:
                blocks.append(text)  # type: ignore[arg-type]
                raw_blocks.append({"type": "text", "text": text})
        elif btype == "tool_use":
            name = getattr(block, "name", "") or ""
            args = getattr(block, "input", {}) or {}
            bid = getattr(block, "id", None)
            calls.append(ToolCall(name=name, arguments=args, id=bid))
            raw_blocks.append(
                {
                    "type": "tool_use",
                    "id": bid,
                    "name": name,
                    "input": args,
                }
            )
    raw_message: dict[str, Any] | None = None
    if raw_blocks:
        # Only echo back what the SDK can deal with — strip the unknown
        # fields that Anthropic adds on tool_use blocks.
        raw_message = {"role": "assistant", "content": raw_blocks}
    return calls, raw_message


def gemini_response_to_tool_calls(
    resp: Any,
) -> tuple[list[Any], dict[str, Any] | None]:
    """Pull tool calls out of a Gemini response.

    Gemini uses ``function_call`` parts on the candidate message; the
    SDK converts each into a :class:`ToolCall`. We build a raw_message
    in Anthropic/OpenAI-style content-block form so the agent's tool
    loop doesn't need a Gemini-specific branch — Anthropic/OpenAI still
    won't accept it, but the call site can fall back to text-only when
    the tool dispatch finishes.
    """
    from .base import ToolCall

    calls: list[Any] = []
    raw_blocks: list[dict[str, Any]] = []
    text_buf: list[str] = []
    try:
        candidates = getattr(resp, "candidates", None) or []
        for cand in candidates:
            content = getattr(cand, "content", None)
            if content is None:
                continue
            for part in getattr(content, "parts", []) or []:
                text = getattr(part, "text", None)
                if text:
                    text_buf.append(text)
                    raw_blocks.append({"type": "text", "text": text})
                fc = getattr(part, "function_call", None)
                if fc is not None:
                    name = getattr(fc, "name", "") or ""
                    # Gemini returns args as a Mapping-like object or a
                    # dict; normalise to a plain dict.
                    raw_args = getattr(fc, "args", {}) or {}
                    args: dict[str, Any] = dict(raw_args)
                    calls.append(ToolCall(name=name, arguments=args, id=None))
                    raw_blocks.append({"type": "tool_call", "name": name, "arguments": args})
    except Exception:
        # Defensive: if Gemini's response shape changes between SDK
        # versions, return whatever we have so the agent's outer error
        # handling can decide what to do.
        pass
    raw_message: dict[str, Any] | None = None
    if raw_blocks:
        raw_message = {"role": "assistant", "content": raw_blocks}
    return calls, raw_message
