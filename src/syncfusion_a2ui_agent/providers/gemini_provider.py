"""Google Gemini provider — official ``google-genai`` SDK, native tool-calling.

Uses :mod:`syncfusion_a2ui_agent.providers._common` for tool/usage helpers
so this provider ships the same feature surface as OpenAI/Azure/Claude.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterable
from typing import Any

from ._common import (
    context_max_tokens,
    gemini_response_to_tool_calls,
    mcp_tools_to_gemini,
    split_system_messages,
    usage_to_dict,
)
from .base import AIProvider, ModelResponse, ProviderError, StreamEvent

try:  # pragma: no cover - import is environment-dependent
    from google import genai
    from google.genai import types as genai_types
except Exception:  # pragma: no cover
    genai = None  # type: ignore[assignment]
    genai_types = None  # type: ignore[assignment]


class GeminiProvider(AIProvider):
    """Google Gemini provider with native tool-calling + streaming."""

    name = "gemini"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if genai is None:
            raise ProviderError("google-genai is not installed. `pip install google-genai`.")
        api_key = kwargs.get("api_key") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ProviderError("GeminiProvider requires `api_key` (or GEMINI_API_KEY).")
        self.model = kwargs.get("model", "gemini-1.5-pro")
        self.temperature = kwargs.get("temperature")
        self.default_max_output_tokens = kwargs.get("max_output_tokens")
        self._client = genai.Client(api_key=api_key)

    def _convert_messages(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Translate OpenAI-style messages into Gemini's ``contents`` form."""
        msgs, system_parts = split_system_messages(messages)
        contents: list[dict[str, Any]] = []
        for m in msgs:
            role = m.get("role")
            content = m.get("content", "")
            # Gemini expects "user" / "model". We treat "tool" as "user"
            # so the model sees the function result text.
            if role in ("assistant", "model"):
                g_role = "model"
            elif role == "tool":
                g_role = "user"
            else:
                g_role = "user"
            contents.append({"role": g_role, "parts": [{"text": str(content)}]})
        system_text = "\n\n".join(system_parts) if system_parts else None
        return contents, system_text

    def _build_config(
        self,
        tools: Iterable[dict[str, Any]] | None,
        context: dict[str, Any] | None,
        system_text: str | None,
    ) -> Any:
        cfg_kwargs: dict[str, Any] = {}
        if system_text:
            cfg_kwargs["system_instruction"] = system_text
        if context is not None and isinstance(context.get("temperature"), (int, float)):
            cfg_kwargs["temperature"] = context["temperature"]
        elif self.temperature is not None:
            cfg_kwargs["temperature"] = self.temperature
        mr = context_max_tokens(context, self.default_max_output_tokens or 0)
        if mr > 0:
            cfg_kwargs["max_output_tokens"] = mr
        elif self.default_max_output_tokens:
            cfg_kwargs["max_output_tokens"] = self.default_max_output_tokens
        mcp_tools = list(tools) if tools else None
        if mcp_tools and genai_types is not None:
            try:
                cfg_kwargs["tools"] = [
                    genai_types.Tool(function_declarations=mcp_tools_to_gemini(mcp_tools))  # type: ignore[arg-type]
                ]
            except Exception:
                # If the SDK doesn't accept the wrapped form, fall back to
                # the raw declarations.
                cfg_kwargs["tools"] = mcp_tools_to_gemini(mcp_tools)
        if not cfg_kwargs or genai_types is None:
            return None
        return genai_types.GenerateContentConfig(**cfg_kwargs)

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: Iterable[dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
    ) -> ModelResponse:
        contents, system_text = self._convert_messages(messages)
        config = self._build_config(tools, context, system_text)
        mcp_tools = list(tools) if tools else None

        try:
            resp = await self._client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
        except Exception as exc:  # pragma: no cover
            raise ProviderError(f"Gemini generate failed: {exc}") from exc

        text = getattr(resp, "text", "") or ""
        calls, raw_message = gemini_response_to_tool_calls(resp) if mcp_tools else ([], None)
        return ModelResponse(
            content=text,
            model=self.model,
            raw=resp,
            tool_calls=calls,
            raw_message=raw_message,
            usage=usage_to_dict(getattr(resp, "usage_metadata", None)),
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: Iterable[dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Token-streaming via Gemini's ``generate_content_stream``."""

        contents, system_text = self._convert_messages(messages)
        config = self._build_config(tools, context, system_text)
        list(tools) if tools else None

        content_buf: list[str] = []
        # tool-call name -> {arguments_buf, yields}
        pending_tcs: dict[str, dict[str, Any]] = {}
        usage_payload: dict[str, Any] = {}
        final_resp: Any = None

        try:
            async for chunk in await self._client.aio.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=config,
            ):
                # Text delta.
                text = getattr(chunk, "text", "") or ""
                if text:
                    content_buf.append(text)
                    yield StreamEvent(delta=text)
                # Tool-call deltas — accumulate per name.
                try:
                    candidates = getattr(chunk, "candidates", None) or []
                    for cand in candidates:
                        for part in getattr(getattr(cand, "content", None), "parts", []) or []:
                            fc = getattr(part, "function_call", None)
                            if fc is None:
                                continue
                            raw_name = getattr(fc, "name", "") or ""
                            slot = pending_tcs.setdefault(
                                raw_name, {"arguments_buf": "", "yields": False}
                            )
                            raw_args = getattr(fc, "args", None)
                            if raw_args is not None:
                                slot["arguments_buf"] = _stringify_args(raw_args)
                                if not slot["yields"]:
                                    import json as _json

                                    from .base import ToolCall

                                    try:
                                        parsed = _json.loads(slot["arguments_buf"])
                                        if isinstance(parsed, dict):
                                            slot["yields"] = True
                                            yield StreamEvent(
                                                tool_call=ToolCall(
                                                    name=raw_name,
                                                    arguments=parsed,
                                                    id=None,
                                                )
                                            )
                                    except Exception:
                                        pass
                except Exception:
                    pass
                # Usage arrives on the final chunk in some SDK versions.
                u = getattr(chunk, "usage_metadata", None)
                if u is not None:
                    usage_payload = usage_to_dict(u)
                final_resp = chunk
        except Exception as exc:  # pragma: no cover
            yield StreamEvent(done=True, error=f"gemini stream failed: {exc}")
            return

        import json as _json

        from .base import ToolCall

        calls: list[Any] = []
        raw_blocks: list[dict[str, Any]] = []
        if content_buf:
            raw_blocks.append({"type": "text", "text": "".join(content_buf)})
        for name, slot in pending_tcs.items():
            try:
                args = _json.loads(slot["arguments_buf"]) if slot["arguments_buf"] else {}
            except Exception:
                args = {}
            if not isinstance(args, dict):
                args = {}
            calls.append(ToolCall(name=name, arguments=args, id=None))
            raw_blocks.append({"type": "tool_call", "name": name, "arguments": args})
        raw_message: dict[str, Any] | None = None
        if raw_blocks:
            raw_message = {"role": "assistant", "content": raw_blocks}

        # If the streaming path never produced final usage, fall back to the
        # last chunk's usage_metadata.
        if not usage_payload and final_resp is not None:
            usage_payload = usage_to_dict(getattr(final_resp, "usage_metadata", None))

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


def _stringify_args(args: Any) -> str:
    """Best-effort: serialise Gemini's args Mapping-like object to JSON text."""
    if isinstance(args, str):
        return args
    if hasattr(args, "items"):
        try:
            import json as _json

            return _json.dumps(dict(args), default=str)
        except Exception:
            return "{}"
    try:
        import json as _json

        return _json.dumps(args, default=str)
    except Exception:
        return "{}"
