"""Request-handling engine for :class:`SyncfusionAgent`.

Owns the prompt-build, tool-call dispatch, and validation-retry loop
shared by ``handle_request`` and ``handle_request_stream``. The
agent module exposes a thin public façade; everything below is the
runtime flow that drives each request.

Public surface:

* :class:`LoopEngine` — owns the prompt → tool-call → validation-retry
  loop. Holds references to the agent's collaborators (provider,
  MCP manager, orchestrator, validator, prompt builder, optional
  skill router). One instance is created per agent and reused across
  requests; per-request state lives on the caller's frame.
* :class:`RequestSetup` / :class:`RequestSetupError` — outcome of
  :meth:`LoopEngine.prepare`.
* :class:`StreamContext` — per-stream state (counters, the per-op
  buffer). Lives on the caller's frame, not on the engine.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from ._streaming import LiveExtractor, StreamOpBuffer, emit_op_events
from .catalog import surface_ids_in
from .orchestrator.tool_orchestrator import ToolOrchestrator
from .prompts.prompt_builder import ConversationTurn, PromptBuilder
from .providers.base import AIProvider, ModelResponse
from .skills.skill_router import SkillRouter
from .validation.response_validator import ResponseValidator

logger = logging.getLogger("syncfusion_a2ui_agent.engine")


# ---------------------------------------------------------------------------
# Small helpers (private)
# ---------------------------------------------------------------------------
def _op_kind(op: dict[str, Any]) -> str:
    """Return the A2UI v0.9 op kind for a parsed op dict.

    Mirrors the discriminator in :mod:`catalog`. Used by the streaming
    path so consumers (the A2A transport, the renderer) can route
    events without re-parsing.
    """
    for kind in ("createSurface", "updateComponents", "updateDataModel"):
        if kind in op:
            return kind
    return "unknown"


def _coerce_mcp_result(result: Any) -> str:
    """Best-effort stringification of an MCP tool result.

    MCP `CallToolResult` objects typically expose ``.content[i].text``
    for text payloads; some clients return raw dicts. We try in order
    to produce something the model can read back as ``role: tool``
    content.
    """
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    # pydantic-like objects
    content = getattr(result, "content", None)
    if content is not None:
        parts: list[str] = []
        for block in content:
            text = getattr(block, "text", None)
            if text is None and isinstance(block, dict):
                text = block.get("text")
            if text is not None:
                parts.append(str(text))
        if parts:
            return "\n".join(parts)
    if hasattr(result, "model_dump"):
        try:
            return json.dumps(result.model_dump(), default=str, ensure_ascii=False)
        except Exception:
            pass
    try:
        return json.dumps(result, default=str, ensure_ascii=False)
    except Exception:
        return str(result)


def _tool_display_name(t: dict) -> str:
    """Extract a tool's display name regardless of flat MCP or OpenAI nested format."""
    if "function" in t:
        return t["function"].get("name", "?")
    return t.get("name", "?")


# ---------------------------------------------------------------------------
# Setup / state dataclasses
# ---------------------------------------------------------------------------
@dataclass
class RequestSetup:
    """Successful prep output from :meth:`LoopEngine.prepare`."""

    messages: list[dict[str, Any]]
    mcp_tools: list[dict[str, Any]]


@dataclass
class RequestSetupError:
    """Prep failure (returned instead of raising so streaming callers
    can yield a structured error event)."""

    error: str


@dataclass
class StreamContext:
    """Per-request state for the streaming path.

    Holds the validation-retry counter, the tool-call budget, the
    aggregated envelope, and the surface-buffering state. All of this
    lives on the caller's frame — *not* on the engine or the agent —
    so two concurrent A2A requests served by the same agent cannot
    see each other's intermediate state.

    Fields
    ------
    last_validation_attempt:
        How many times we've asked the model for a re-try so far.
        Compared against ``ResponseValidator.max_attempts()`` to
        decide when to stop.
    last_validation_error:
        The error string from the previous attempt. The circuit
        breaker compares the new error against this value and bails
        out when it sees the same one twice in a row.
    tool_call_budget:
        Hard cap on the number of tool calls per request. Defaults
        to 10 — well above the 1-3 the agent normally needs.
    tool_calls_done:
        Counter incremented by the tool-call dispatch loop.
    aggregated_envelope:
        The full set of ops the streaming path has emitted so far.
        Used only for logging / debugging; the consumer drives the
        render from the individual ``op`` events.
    buffer:
        The :class:`_streaming.StreamOpBuffer` that holds
        non-``createSurface`` ops until their surface is announced.
    """

    last_validation_attempt: int = 0
    last_validation_error: str | None = None
    tool_call_budget: int = 10
    tool_calls_done: int = 0
    aggregated_envelope: list[dict[str, Any]] = field(default_factory=list)
    buffer: Any = field(default=None)

    def __post_init__(self) -> None:
        if self.buffer is None:
            self.buffer = StreamOpBuffer()

    def reset_per_attempt(self) -> None:
        """Clear per-attempt state at the start of each retry."""
        self.tool_calls_done = 0
        self.aggregated_envelope = []
        self.buffer.reset()


# ---------------------------------------------------------------------------
# PromptBuilder subclass (engine-private)
# ---------------------------------------------------------------------------
class PromptBuilderWithExtension(PromptBuilder):
    """PromptBuilder that pulls the customer extension from a callable."""

    def __init__(self, extend_fn) -> None:
        super().__init__()
        self._extend_fn = extend_fn

    def _customer_extension(self) -> str:
        try:
            return self._extend_fn() or ""
        except Exception:  # pragma: no cover
            return ""


# ---------------------------------------------------------------------------
# LoopEngine — the runtime request loop
# ---------------------------------------------------------------------------
class LoopEngine:
    """Owns the prompt → tool-call → validation-retry loop.

    Construction is lightweight — the engine holds references to the
    agent's collaborators (provider, MCP manager, orchestrator,
    validator, prompt builder, optional skill router). One engine
    instance is created per agent in :class:`SyncfusionAgent.__init__`
    and reused across requests; the per-request state (counters,
    tool-call results, the streaming buffer) lives on the caller's
    frame so concurrent requests do not interfere.

    Public surface:

    * :meth:`prepare` — runs the MCP warmup / catalog-refresh dance,
      the orchestrator plan, and the prompt build. Returns a
      :class:`RequestSetup` (or :class:`RequestSetupError`).
    * :meth:`handle` — buffered ``handle_request`` body. Returns a
      dict with ``envelope`` + ``surfaceIds`` or raises on failure.
    * :meth:`handle_stream` — streaming ``handle_request_stream``
      body. Yields ``op`` / ``surfaceIds`` / ``error`` events.
    * :meth:`build_history` — utility for converting raw
      conversation-history dicts into :class:`ConversationTurn`
      records.
    * :meth:`inject_skill_catalog` — utility for setting the skill
      catalog on the prompt builder when skills are enabled.
    """

    DEFAULT_TOOL_CALL_BUDGET = 10
    DEFERRED_WARMUP_WAIT = 2.0
    COLD_WARMUP_REFRESH_TIMEOUT = 60.0
    ORCHESTRATOR_PLAN_TIMEOUT = 0.5

    def __init__(
        self,
        *,
        provider: AIProvider,
        mcp_manager: Any,
        orchestrator: ToolOrchestrator,
        validator: ResponseValidator,
        prompt_builder: PromptBuilder,
        skill_router: SkillRouter | None = None,
    ) -> None:
        self._provider = provider
        self._mcp = mcp_manager
        self._orchestrator = orchestrator
        self._validator = validator
        self._prompt_builder = prompt_builder
        self._skill_router = skill_router

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def build_history(
        conversation_history: list[dict[str, Any]] | None,
    ) -> list[ConversationTurn]:
        """Convert raw conversation-history dicts into ``ConversationTurn`` records."""
        history: list[ConversationTurn] = []
        if not conversation_history:
            return history
        for h in conversation_history:
            history.append(
                ConversationTurn(
                    role=h.get("role", "user"),
                    content=h.get("content", ""),
                    name=h.get("name"),
                )
            )
        return history

    def inject_skill_catalog(self) -> None:
        """Inject the skill catalog block into the prompt builder.

        VS Code-style: skill descriptions are added to the system
        prompt once so the model knows what skills exist. The model
        then calls ``read_skill`` (a synthetic tool) on demand to
        fetch SKILL.md content. No-op when skills are disabled.
        """
        if self._skill_router is None:
            return
        skill_catalog_block = self._skill_router.get_catalog_block()
        if skill_catalog_block:
            self._prompt_builder.set_skill_catalog(skill_catalog_block)
            logger.info("Skills: injected catalog block into system prompt.")

    # -- prepare -----------------------------------------------------------
    async def prepare(
        self,
        user_message: str,
        history: list[ConversationTurn],
    ) -> RequestSetup | RequestSetupError:
        """Shared setup for ``handle`` and ``handle_stream``.

        Runs the MCP warmup / catalog-refresh dance, the orchestrator
        plan + business tool calls, and the prompt build. Returns a
        :class:`RequestSetup` with ``messages`` and ``mcp_tools`` for
        the provider call. Returns a :class:`RequestSetupError` on
        failure (the streaming path surfaces this as an ``error``
        event instead of raising).
        """
        # Refresh MCP tool catalog so we'll have the descriptors to
        # hand to the provider as native tool definitions. The hot
        # path is a cache hit (no network round-trip); the cold path
        # fires a background warmup so the first request doesn't pay
        # a 5s spawn.
        servers = self._mcp.list_servers()
        if getattr(self._mcp, "_deferred_warmup", False):
            self._mcp._deferred_warmup = False
            self._mcp._warm_started = True
            logger.info(
                "Deferred MCP warmup — firing warm_start() on request loop for: %s",
                [s.name for s in servers],
            )
            warm_task = self._mcp.warm_start()
            try:
                await asyncio.wait_for(
                    asyncio.shield(warm_task),
                    timeout=self.DEFERRED_WARMUP_WAIT,
                )
            except (asyncio.TimeoutError, Exception):
                logger.info(
                    "Deferred MCP warmup still in flight after %s s; "
                    "continuing with this request and letting the "
                    "background task complete naturally.",
                    self.DEFERRED_WARMUP_WAIT,
                )
        elif servers and self._mcp.has_cached_tools():
            logger.debug(
                "MCP tool catalog already cached → skipping refresh for: %s",
                [s.name for s in servers],
            )
        elif not self._mcp._warm_started:
            self._mcp._warm_started = True
            logger.info(
                "Cold MCP catalog — firing warm_start() in background for: %s",
                [s.name for s in servers],
            )
            self._mcp.warm_start()
        else:
            try:
                logger.info(
                    "Refreshing MCP tool catalog for servers: %s",
                    [s.name for s in servers],
                )
                await asyncio.wait_for(
                    self._mcp.refresh_tools(),
                    timeout=self.COLD_WARMUP_REFRESH_TIMEOUT,
                )
            except (asyncio.TimeoutError, Exception) as exc:
                logger.debug("MCP tool catalog refresh failed: %s", exc)

        # Orchestrator: business-knowledge tools in parallel.
        plan = self._orchestrator.plan(user_message, conversation_history=history)
        if not plan.mcp_servers:
            tool_result = None
        else:
            tool_task = asyncio.create_task(self._orchestrator.execute(plan))
            try:
                tool_result = await asyncio.wait_for(
                    tool_task, timeout=self.ORCHESTRATOR_PLAN_TIMEOUT
                )
            except (asyncio.TimeoutError, Exception):
                tool_result = None

        prompt = self._prompt_builder.build(
            user_message=user_message,
            tool_results=tool_result.results if tool_result else None,
        )
        messages = prompt.render_messages()

        mcp_tools: list[dict[str, Any]] = self._mcp.list_tools()
        if self._skill_router is not None:
            mcp_tools = [self._skill_router.get_skill_tool_definition()] + mcp_tools
        logger.info(
            "MCP tool catalog → %d tool(s) visible to the model: %s",
            len(mcp_tools),
            [_tool_display_name(t) for t in mcp_tools],
        )
        return RequestSetup(messages=messages, mcp_tools=mcp_tools)

    # -- buffered handle ---------------------------------------------------
    async def handle(
        self,
        user_message: str,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Buffered request flow (spec §5).

        Returns a dict shaped like:
            {
                "envelope":  [...],   # the validated A2UI v0.9 envelope
                "surfaceIds":[...],
            }
        so the A2A transport can forward the response to the client.
        """
        history = self.build_history(conversation_history)
        self._prompt_builder.set_history_provider(lambda: history)
        self.inject_skill_catalog()

        setup = await self.prepare(user_message, history)
        if isinstance(setup, RequestSetupError):
            raise RuntimeError(setup.error)
        messages = setup.messages
        mcp_tools = setup.mcp_tools

        # Generate + tool-call loop with bounded validation retries on the
        # final response. Tool-calls themselves don't count against the
        # retry budget — only "the model couldn't produce valid A2UI"
        # attempts do.
        last_error: str | None = None
        last_validation_attempt = 0
        # Tracks the previous validation error so the circuit breaker
        # can detect when the model is failing the envelope
        # deterministically. Lives on the local frame, NOT on the
        # agent instance — see StreamContext for the streaming
        # equivalent. Per-instance state on the agent would be
        # shared across concurrent A2A requests.
        last_validation_error: str | None = None
        tool_call_budget = self.DEFAULT_TOOL_CALL_BUDGET
        last_resp: ModelResponse | None = None

        # Outer loop: validation retries (max N attempts).
        while last_validation_attempt < self._validator.max_attempts():
            # Inner loop: tool-call round-trip until the model returns text.
            # We always run at least one provider.generate() per outer pass —
            # even when no MCP tools are registered — because the validation
            # retry loop needs a final answer to validate.
            tool_calls_done = 0
            while True:
                try:
                    resp: ModelResponse = await self._provider.generate(messages, tools=mcp_tools)
                except Exception as exc:
                    logger.exception("Provider call failed: %s", exc)
                    last_error = f"provider error: {exc}"
                    break
                last_resp = resp

                # If the model gave us text, we're done round-tripping.
                if not resp.tool_calls:
                    logger.debug("Model returned final text (no tool calls).")
                    break
                logger.info(
                    "Model requested %d tool call(s): %s",
                    len(resp.tool_calls),
                    [tc.name for tc in resp.tool_calls],
                )

                # Dispatch each tool call via MCPManager.call_tool, feed
                # results back as OpenAI-style "tool" messages, loop.
                tool_messages, tool_calls_done = await self._dispatch_tool_calls(
                    resp, tool_calls_done
                )

                if not tool_messages:
                    # Model emitted tool calls but none had a server__ prefix
                    # (shouldn't happen with our tools=[...] schema). Treat
                    # the model's text as the final answer.
                    break

                # OpenAI / Azure require the assistant message (with the
                # original `tool_calls` field) to be present in the
                # conversation BEFORE the `role: tool` result messages.
                # The provider stored it on `last_resp.raw_message` for us.
                new_messages: list[dict[str, Any]] = []
                if last_resp and last_resp.raw_message:
                    new_messages.append(last_resp.raw_message)
                new_messages.extend(tool_messages)
                messages = list(messages) + new_messages
                if tool_calls_done >= tool_call_budget:
                    logger.warning(
                        "Tool-call budget (%d) exhausted; ignoring further model tool calls.",
                        tool_call_budget,
                    )
                    break

            # If we have a final text response, validate it.
            if last_resp is None:
                break
            vr = self._validator.validate_text(last_resp.content or "")
            if vr.ok and vr.payload is not None:
                envelope = vr.payload
                return {
                    "envelope": envelope,
                    "surfaceIds": surface_ids_in(envelope),
                }
            last_error, should_continue = self._on_validation_failure(
                last_error=last_error,
                attempt=last_validation_attempt,
                error=vr.error,
                last_validation_error=last_validation_error,
            )
            last_validation_attempt += 1
            last_validation_error = last_error
            if not should_continue:
                break
            messages = list(messages) + [self._corrective_user_message(last_error)]
            last_resp = None

        # If we get here, all attempts failed. We DO NOT return a fallback
        # surface — that would violate the A2UI envelope contract (and
        # silently mask bugs in the model or MCP plumbing). The A2A executor
        # catches this and surfaces it as a structured error to the client.
        raise RuntimeError(
            f"Agent could not produce a schema-valid A2UI v0.9 envelope after "
            f"{self._validator.max_attempts()} attempt(s). Last error: {last_error}"
        )

    # -- streaming handle --------------------------------------------------
    async def handle_stream(
        self,
        user_message: str,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Streaming variant of :meth:`handle` (Level 2 of the streaming plan).

        Yields a sequence of events shaped like::

            {"type": "op",        "op": {...}, "kind": "createSurface"|"updateComponents"|"updateDataModel"}
            {"type": "surfaceIds","surfaceIds": [...]}
            {"type": "error",     "error": "..."}

        The A2A streaming transport iterates this generator and emits
        one A2A ``Artifact`` event per ``op`` event. The first ``op``
        yielded is always the ``createSurface`` op (the renderer
        mounts the surface on that event). The final ``surfaceIds``
        event carries the aggregated ids and signals that the stream
        is complete.
        """
        history = self.build_history(conversation_history)
        self._prompt_builder.set_history_provider(lambda: history)
        self.inject_skill_catalog()

        setup = await self.prepare(user_message, history)
        if isinstance(setup, RequestSetupError):
            yield {"type": "error", "error": setup.error}
            return
        messages = setup.messages
        mcp_tools = setup.mcp_tools

        # Per-request context. Holds the circuit-breaker state and
        # per-attempt accumulators. CRITICAL: this lives on the
        # caller's frame, NOT on the engine or the agent, so two
        # concurrent A2A requests on the same agent don't see each
        # other's intermediate state.
        ctx = StreamContext(tool_call_budget=self.DEFAULT_TOOL_CALL_BUDGET)
        last_resp: ModelResponse | None = None
        last_error: str | None = None

        while ctx.last_validation_attempt < self._validator.max_attempts():
            while True:
                # Per-attempt live extractor. Fed by the provider
                # stream below, drained by the validation loop.
                extractor = LiveExtractor()

                collected_chunks: list[str] = []
                tool_call_seen = False
                try:
                    async for event in self._provider.stream(messages, tools=mcp_tools):
                        if event.error:
                            last_error = event.error
                            break
                        if event.delta:
                            collected_chunks.append(event.delta)
                            # Synchronous: feed the delta into the
                            # state machine and immediately drain
                            # any ops that have become parseable.
                            extractor.feed(event.delta)
                            for evt in emit_op_events(
                                extractor.take_ops(), ctx.buffer, self._validator
                            ):
                                ctx.aggregated_envelope.append(evt["op"])
                                yield evt
                        if event.tool_call is not None:
                            tool_call_seen = True
                            # Tool-call dispatch reads from
                            # ``last_resp.tool_calls`` on the final
                            # ``done=True`` event (see the
                            # ``if tool_call_seen and last_resp...``
                            # branch below). We intentionally do NOT
                            # accumulate tool calls on the engine
                            # instance — per-instance state would be
                            # shared across concurrent A2A requests
                            # and cause cross-request contamination.
                        if event.done:
                            last_resp = event.response
                            break
                except Exception as exc:
                    logger.exception("Provider stream failed: %s", exc)
                    last_error = f"provider error: {exc}"
                    extractor.finalize()
                    break

                # Finalize the extractor (end-of-stream flush). On
                # the done=True path the array is complete by now,
                # so this is a no-op; on the early-exit path it
                # surfaces whatever partial array was buffered.
                for evt in emit_op_events(extractor.finalize(), ctx.buffer, self._validator):
                    ctx.aggregated_envelope.append(evt["op"])
                    yield evt

                if tool_call_seen and last_resp is not None and last_resp.tool_calls:
                    tool_messages, ctx.tool_calls_done = await self._dispatch_tool_calls(
                        last_resp, ctx.tool_calls_done
                    )
                    if not tool_messages:
                        # No dispatchable tool calls — treat the
                        # accumulated text as the final answer.
                        break
                    new_messages: list[dict[str, Any]] = []
                    if last_resp.raw_message:
                        new_messages.append(last_resp.raw_message)
                    new_messages.extend(tool_messages)
                    messages = list(messages) + new_messages
                    if ctx.tool_calls_done >= ctx.tool_call_budget:
                        logger.warning(
                            "Stream tool-call budget (%d) exhausted; treating as final answer.",
                            ctx.tool_call_budget,
                        )
                        break
                    last_resp = None
                    collected_chunks = []
                    continue

                # No tool call — this is the final text stream for
                # this validation attempt.
                text_so_far = "".join(collected_chunks)
                if not ctx.aggregated_envelope:
                    last_error = (
                        "Model streamed a response that contained no "
                        "A2UI envelope. Expected a JSON array wrapped "
                        "in <a2ui-json>...</a2ui-json>."
                    )
                    break
                vr = self._validator.validate_text(text_so_far)
                if vr.ok and vr.payload is not None:
                    yield {
                        "type": "surfaceIds",
                        "surfaceIds": list(ctx.buffer.emitted_surface_ids()),
                    }
                    return
                last_error = vr.error or "unknown validation error"
                break

            # If we exited the inner loop with a final text answer
            # that validated, we're done.
            if last_resp is not None and not last_resp.tool_calls:
                vr = self._validator.validate_text(last_resp.content or "")
                if vr.ok and vr.payload is not None:
                    yield {
                        "type": "surfaceIds",
                        "surfaceIds": list(ctx.buffer.emitted_surface_ids()),
                    }
                    return
                last_error = vr.error or "unknown validation error"
            ctx.last_validation_attempt += 1
            logger.debug(
                "Streaming validation failed (attempt %d/%d): %s",
                ctx.last_validation_attempt,
                self._validator.max_attempts(),
                last_error,
            )
            if last_error == ctx.last_validation_error and ctx.last_validation_attempt >= 2:
                logger.warning(
                    "Streaming validator circuit breaker tripped after %d "
                    "identical errors; aborting. Last error: %s",
                    ctx.last_validation_attempt,
                    last_error,
                )
                break
            ctx.last_validation_error = last_error
            # Same corrective user message as the non-streaming path.
            messages = list(messages) + [self._corrective_user_message(last_error or "")]
            # Reset per-attempt state.
            last_resp = None
            ctx.reset_per_attempt()

        # All attempts failed.
        yield {
            "type": "error",
            "error": (
                f"Agent could not produce a schema-valid A2UI v0.9 "
                f"envelope after {self._validator.max_attempts()} "
                f"attempt(s). Last error: {last_error}"
            ),
        }

    # -- internal helpers --------------------------------------------------
    async def _dispatch_tool_calls(
        self,
        resp: ModelResponse,
        tool_calls_done: int,
    ) -> tuple[list[dict[str, Any]], int]:
        """Dispatch each tool call in *resp* and return OpenAI-style
        ``role: tool`` messages plus the updated counter.

        Handles two tool flavours:

        * The synthetic ``read_skill`` tool — served by the
          :class:`SkillRouter` when skills are enabled.
        * MCP tools named ``server__tool`` — split on ``__`` and
          routed through :class:`MCPManager.call_tool`.

        Tool calls with no ``__`` in the name (i.e. the model
        invented a non-MCP name) are silently dropped — they don't
        match the schema the model was given, and surfacing them as
        errors just wastes a round-trip.
        """
        tool_messages: list[dict[str, Any]] = []
        for tc in resp.tool_calls:
            # Dispatch read_skill synthetic tool (VS Code-style skill fetch).
            if self._skill_router is not None and SkillRouter.is_read_skill_call(tc.name):
                skill_name = tc.arguments.get("name", "")
                content_str = self._skill_router.handle_read_skill_call(skill_name)
                logger.info(
                    "read_skill tool call → skill=%r, content_length=%d",
                    skill_name,
                    len(content_str),
                )
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id or "",
                        "content": content_str,
                    }
                )
                tool_calls_done += 1
                continue
            if "__" not in tc.name:
                continue
            server, tool = tc.name.split("__", 1)
            try:
                result = await self._mcp.call_tool(server, tool, tc.arguments)
                content_str = _coerce_mcp_result(result)
            except Exception as exc:
                content_str = f"<error: {exc}>"
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id or "",
                    "content": content_str,
                }
            )
            tool_calls_done += 1
        return tool_messages, tool_calls_done

    def _on_validation_failure(
        self,
        *,
        last_error: str | None,
        attempt: int,
        error: str | None,
        last_validation_error: str | None,
    ) -> tuple[str, bool]:
        """Apply the circuit-breaker policy on a validation failure.

        Returns ``(new_last_error, should_continue)``. When the same
        envelope error fires twice, the circuit breaker trips and
        ``should_continue`` is ``False`` — burning a 3rd attempt on a
        deterministic failure almost never recovers.
        """
        err = error or "unknown validation error"
        new_attempt = attempt + 1
        logger.debug(
            "Validation failed (attempt %d/%d): %s",
            new_attempt,
            self._validator.max_attempts(),
            err,
        )
        if err == last_validation_error and new_attempt >= 2:
            logger.warning(
                "Validator circuit breaker tripped after %d identical "
                "errors; aborting retries. Last error: %s",
                new_attempt,
                err,
            )
            return err, False
        return err, True

    @staticmethod
    def _corrective_user_message(last_error: str) -> dict[str, str]:
        """Build the corrective user message appended after a validation failure.

        Same corrective prompt for buffered and streaming paths.
        """
        return {
            "role": "user",
            "content": (
                "Your previous response failed A2UI envelope validation.\n"
                f"Error: {last_error}\n\n"
                "CRITICAL RULES for your next response:\n"
                "1. Output ONLY the JSON array — no explanation, no markdown prose, no code comments.\n"
                "2. Wrap the entire array in <a2ui-json>[ ... ]</a2ui-json> with no code fence inside.\n"
                "3. Do NOT use ```json or ``` inside the <a2ui-json> tags.\n"
                "4. If the component tree is large, reduce the number of components — keep it under 25.\n"
                "5. The array must contain: one createSurface, one updateComponents, "
                "one or more updateDataModel ops.\n"
                "Respond with ONLY the <a2ui-json> block, nothing else."
            ),
        }


__all__ = [
    "LoopEngine",
    "PromptBuilderWithExtension",
    "RequestSetup",
    "RequestSetupError",
    "StreamContext",
    "_coerce_mcp_result",
    "_op_kind",
    "_tool_display_name",
]
