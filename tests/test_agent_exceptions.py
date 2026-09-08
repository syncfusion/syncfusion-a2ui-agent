"""Direct tests for :mod:`syncfusion_a2ui_agent.agent_exceptions`.

The module exposes a single helper:
:func:`is_mcp_teardown_exception` — a walker that distinguishes
the ``mcp`` SDK's cosmetic teardown ``RuntimeError`` (anyio
cancel-scope bug) from real user-code errors.
"""

from __future__ import annotations

from typing import Any

from syncfusion_a2ui_agent.agent_exceptions import is_mcp_teardown_exception


class TestIsMcpTeardownException:
    def test_returns_true_for_anyio_cancel_scope_runtime_error(self) -> None:
        # The exact anyio message the SDK emits at teardown.
        exc = RuntimeError("Attempted to exit cancel scope in a different task")
        assert is_mcp_teardown_exception(exc) is True

    def test_returns_false_for_unrelated_runtime_error(self) -> None:
        exc = RuntimeError("Connection refused")
        assert is_mcp_teardown_exception(exc) is False

    def test_returns_false_for_non_runtime_error(self) -> None:
        assert is_mcp_teardown_exception(ValueError("bad input")) is False
        assert is_mcp_teardown_exception(KeyError("k")) is False
        assert is_mcp_teardown_exception(TypeError("bad type")) is False

    def test_walks_cause_chain(self) -> None:
        # The cosmetic error may be wrapped in another exception.
        outer = RuntimeError("wrapper")
        outer.__cause__ = RuntimeError("Attempted to exit cancel scope in a different task")
        assert is_mcp_teardown_exception(outer) is True

    def test_walks_context_chain(self) -> None:
        outer = RuntimeError("wrapper")
        outer.__context__ = RuntimeError("Attempted to exit cancel scope in a different task")
        assert is_mcp_teardown_exception(outer) is True

    def test_handles_cyclic_exception_chain(self) -> None:
        # Two exceptions that reference each other — the walker
        # must not infinite-loop. There's a ``seen: set[int]``
        # guard in the source.
        a: Any = RuntimeError("a")
        b: Any = RuntimeError("b")
        a.__cause__ = b
        b.__cause__ = a
        # Neither matches the cancel-scope message, so the
        # result is False; the test is that it RETURNS.
        assert is_mcp_teardown_exception(a) is False

    def test_returns_false_for_keyboard_interrupt(self) -> None:
        # BaseException subclasses bypass the filter naturally.
        assert is_mcp_teardown_exception(KeyboardInterrupt()) is False
        assert is_mcp_teardown_exception(SystemExit()) is False

    def test_handles_empty_message_runtime_error(self) -> None:
        # A RuntimeError with an empty message — the walker falls
        # through to a False result.
        assert is_mcp_teardown_exception(RuntimeError("")) is False
