"""Uniform AIProvider interface — every vendor implements this."""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from typing import Any


class ProviderError(RuntimeError):
    """Raised when an AI provider call fails in a non-recoverable way."""


@dataclass
class ToolCall:
    """A native function-call from the model that the agent should dispatch.

    `name` follows the SDK convention of `server__tool` (set by the provider
    when it translates MCP descriptors into the vendor's tool format). The
    agent's tool loop splits it back into server + tool and dispatches.
    """

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: str | None = None


@dataclass
class ModelResponse:
    """Normalised response from an AI provider.

    `content` is the raw text the model emitted. For A2UI agents the validator
    expects this to be (or contain) an A2UI JSON payload. The provider must NOT
    pre-validate against A2UI — that is the orchestrator's job.

    `tool_calls` is populated when the model wants to invoke one or more
    registered tools (e.g. MCP `tools/list`). The agent's tool loop dispatches
    them and feeds the results back to the model.

    `raw_message` is the full provider-native assistant message that contained
    the tool calls. OpenAI/Azure require this message to be appended to the
    conversation *before* the `role: tool` result messages, otherwise the
    provider returns 400 ("role 'tool' must be a response to a preceding
    message with 'tool_calls'").
    """

    content: str
    model: str = ""
    raw: Any = None
    usage: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_message: dict[str, Any] | None = None


@dataclass
class StreamEvent:
    """A single event from ``AIProvider.stream()``.

    Carries one of:
      - a ``delta`` string (text the model just emitted)
      - a ``tool_call`` (the model wants to invoke a tool mid-stream)
      - a ``response`` (the final ``ModelResponse`` once the stream completes)
      - a ``done`` flag (the stream has finished; ``response`` may be set)
      - an ``error`` string (the stream failed)

    The agent's tool-call loop only acts when ``tool_call`` is present;
    the streaming path only acts when ``delta`` is present; both paths
    finalise on the ``done=True`` event.
    """

    delta: str = ""
    tool_call: ToolCall | None = None
    response: ModelResponse | None = None
    done: bool = False
    error: str | None = None


class AIProvider(abc.ABC):
    """Abstract base for all AI providers (built-in or customer-supplied).

    Per the spec, the provider implements ONLY `generate()` — retries, timeouts,
    and error handling are owned by the SDK. `CustomProvider` subclasses this
    and adds the registration surface customers use.
    """

    name: str = "base"

    def __init__(self, **kwargs: Any) -> None:
        # Store config so subclasses can read defaults like model, temperature.
        self.config: dict[str, Any] = dict(kwargs)

    @abc.abstractmethod
    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: Iterable[dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
    ) -> ModelResponse:
        """Generate a model response from the given message list.

        `messages` is already shaped for consumption (see PromptBuilder.render_messages).
        `tools` is an optional iterable of tool schemas the provider may invoke.
        `context` is a free-form bag for orchestrator-level hints (session id, etc.).
        """

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: Iterable[dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Optional streaming variant of ``generate()``.

        Providers that support token-by-token streaming override this
        method. The default implementation falls back to ``generate()``
        and yields a single ``StreamEvent(done=True, response=...)`` so
        that every existing provider works without change.

        Agents that want to consume the stream (Level 1+ of the
        streaming plan) call this method instead of ``generate()``.
        """
        resp = await self.generate(messages, tools=tools, context=context)
        yield StreamEvent(response=resp, done=True)
