"""Direct tests for the A2A transport's history extraction.

Covers the C4 fix to ``_extract_history`` in
:mod:`syncfusion_a2ui_agent.a2a.transport` — the prior version
returned ``[]`` for every request, silently dropping
multi-turn conversation history.

These tests exercise the new implementation directly with a
real :class:`a2a.types.RequestContext` and real
:class:`a2a.types.Task` objects (the function is private but
the behaviour is the public contract for the A2A executor).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock


def _msg(role: str, text: str, msg_id: str) -> Any:
    from a2a.types import Message, Part, Role, TextPart

    return Message(
        role=Role(role) if role != "agent" else Role.agent,
        parts=[Part(root=TextPart(text=text))],
        message_id=msg_id,
        context_id="c1",
    )


def _task(messages: list[Any], task_id: str = "t1") -> Any:
    from a2a.types import Task, TaskStatus

    return Task(
        id=task_id,
        context_id="c1",
        history=messages,
        status=TaskStatus(state="completed"),
    )


# ---------------------------------------------------------------------------
# _extract_history
# ---------------------------------------------------------------------------
class TestExtractHistory:
    def _import(self) -> Any:
        from syncfusion_a2ui_agent.a2a.transport import _extract_history

        return _extract_history

    def test_none_context_returns_empty(self) -> None:
        extract = self._import()
        assert extract(None) == []

    def test_context_with_no_tasks_returns_empty(self) -> None:
        extract = self._import()
        ctx = MagicMock()
        ctx.current_task = None
        ctx.related_tasks = None
        assert extract(ctx) == []

    def test_extracts_from_current_task(self) -> None:
        extract = self._import()
        ctx = MagicMock()
        ctx.current_task = _task(
            [
                _msg("user", "hi", "m1"),
                _msg("agent", "hello back", "m2"),
            ]
        )
        ctx.related_tasks = None
        result = extract(ctx)
        assert result == [
            {"role": "user", "content": "hi"},
            {"role": "agent", "content": "hello back"},
        ]

    def test_extracts_from_related_tasks(self) -> None:
        extract = self._import()
        ctx = MagicMock()
        ctx.current_task = _task([_msg("agent", "current", "m2")])
        ctx.related_tasks = [_task([_msg("user", "earlier", "m1")])]
        result = extract(ctx)
        # ``current_task`` is processed first, then related tasks.
        assert result[0] == {"role": "agent", "content": "current"}
        assert result[1] == {"role": "user", "content": "earlier"}

    def test_dedupes_by_message_id(self) -> None:
        extract = self._import()
        ctx = MagicMock()
        m1 = _msg("user", "hi", "m1")
        # Same message_id appears in both current and related task.
        ctx.current_task = _task([m1, _msg("agent", "hi back", "m2")])
        ctx.related_tasks = [_task([m1])]
        result = extract(ctx)
        # m1 should appear exactly once.
        assert len([r for r in result if r["content"] == "hi"]) == 1

    def test_skips_messages_with_no_text_content(self) -> None:
        extract = self._import()
        from a2a.types import DataPart, Message, Part, Role

        text_msg = _msg("user", "real text", "m1")
        data_msg = Message(
            role=Role.user,
            parts=[Part(root=DataPart(data={"foo": "bar"}))],
            message_id="m2",
            context_id="c1",
        )
        ctx = MagicMock()
        ctx.current_task = _task([text_msg, data_msg])
        ctx.related_tasks = None
        result = extract(ctx)
        # The data-part message has no text, so it's skipped.
        assert result == [{"role": "user", "content": "real text"}]

    def test_real_a2a_request_context(self) -> None:
        # End-to-end test using a real RequestContext from the
        # a2a-sdk, not a mock — exercises the public path.
        from a2a.server.agent_execution import RequestContext
        from a2a.types import MessageSendParams

        from syncfusion_a2ui_agent.a2a.transport import _extract_history

        task = _task([_msg("user", "earlier", "m1")])
        ctx = RequestContext(
            request=MessageSendParams(message=_msg("user", "new", "m_new")),
            task_id="t1",
            context_id="c1",
            task=task,
            related_tasks=[],
        )
        result = _extract_history(ctx)
        # The function should pick up ``ctx.task`` (the current
        # task) and project its history.
        assert result == [{"role": "user", "content": "earlier"}]
