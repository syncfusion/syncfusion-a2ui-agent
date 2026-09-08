"""Shared test helpers — the in-process ``FakeProvider`` and scripted
response steps used by the agent tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterable

from syncfusion_a2ui_agent.providers.base import (
    AIProvider,
    ModelResponse,
    StreamEvent,
    ToolCall,
)

# A canonical, valid A2UI v0.9 envelope used across the test suite.
VALID_ENVELOPE_TEXT = (
    "<a2ui-json>"
    "["
    '{ "version": "v0.9", "createSurface": { "surfaceId": "s1" } },'
    ' { "version": "v0.9", "updateComponents": { "surfaceId": "s1",'
    ' "components": [ { "id": "root", "component": "Column",'
    ' "children": [] } ] } },'
    ' { "version": "v0.9", "updateDataModel": { "surfaceId": "s1",'
    ' "path": "/_suggestions", "value": [ "Try filters" ] } }'
    "]"
    "</a2ui-json>"
)

VALID_ENVELOPE: list[dict[str, Any]] = [
    {"version": "v0.9", "createSurface": {"surfaceId": "s1"}},
    {
        "version": "v0.9",
        "updateComponents": {
            "surfaceId": "s1",
            "components": [{"id": "root", "component": "Column", "children": []}],
        },
    },
    {
        "version": "v0.9",
        "updateDataModel": {
            "surfaceId": "s1",
            "path": "/_suggestions",
            "value": ["Try filters"],
        },
    },
]


@dataclass
class ScriptedStep:
    """A single scripted behaviour for the FakeProvider.

    Each step is consumed in order. The provider advances to the next
    step on every ``generate()`` or ``stream()`` call.
    """

    tool_calls: list[ToolCall] = field(default_factory=list)
    content: str = ""
    raise_exception: Exception | None = None
    # If set, the provider yields this list of stream events instead of
    # wrapping ``content`` into a single ``done=True`` event.
    stream_events: list[StreamEvent] | None = None


class FakeProvider(AIProvider):
    """An in-process AIProvider used by the agent's test suite.

    The provider is "scripted": the test author registers a list of
    :class:`ScriptedStep` instances and the provider consumes them in
    order on every call to :meth:`generate` or :meth:`stream`. Once
    the script is exhausted, the provider returns an empty fallback.
    """

    name = "fake"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.script: list[ScriptedStep] = []
        self.call_count: int = 0
        self.last_messages: list[dict[str, Any]] | None = None
        self.last_tools: list[dict[str, Any]] | None = None
        self.model = "fake-model"

    def queue(self, *steps: ScriptedStep) -> "FakeProvider":
        """Append scripted steps and return self for fluent setup."""
        self.script.extend(steps)
        return self

    def _next(self) -> ScriptedStep:
        if self.call_count >= len(self.script):
            return ScriptedStep(content="<no script>")
        step = self.script[self.call_count]
        self.call_count += 1
        return step

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: Iterable[dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
    ) -> ModelResponse:
        self.last_messages = list(messages)
        self.last_tools = list(tools) if tools else None
        step = self._next()
        if step.raise_exception is not None:
            raise step.raise_exception
        return ModelResponse(
            content=step.content,
            model=self.model,
            tool_calls=list(step.tool_calls),
            raw_message=(
                {
                    "role": "assistant",
                    "content": step.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in step.tool_calls
                    ],
                }
                if step.tool_calls
                else None
            ),
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: Iterable[dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        self.last_messages = list(messages)
        self.last_tools = list(tools) if tools else None
        step = self._next()
        if step.raise_exception is not None:
            yield StreamEvent(done=True, error=str(step.raise_exception))
            return
        if step.stream_events is not None:
            for ev in step.stream_events:
                yield ev
            return
        # Default: emit content as a single delta + done.
        if step.content:
            yield StreamEvent(delta=step.content)
        for tc in step.tool_calls:
            yield StreamEvent(tool_call=tc)
        yield StreamEvent(
            response=ModelResponse(
                content=step.content,
                model=self.model,
                tool_calls=list(step.tool_calls),
            ),
            done=True,
        )
