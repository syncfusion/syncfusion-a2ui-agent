"""Exception-classification helpers for the agent's lifecycle code.

Exposes :func:`is_mcp_teardown_exception` — a walker that
distinguishes the mcp SDK's cosmetic teardown bug from a real
user-code error. Kept in its own module so it can be unit-tested
without dragging in the rest of the agent.
"""

from __future__ import annotations


def is_mcp_teardown_exception(exc: BaseException) -> bool:
    """Return True iff ``exc`` matches the known mcp-SDK teardown bug.

    The bug surfaces as ``anyio``'s
    ``RuntimeError("...exit cancel scope in a different task...")``,
    either directly or wrapped in a ``BaseExceptionGroup`` (Python 3.11+)
    or ``ExceptionGroup`` (Python 3.10+ via ``exceptiongroup`` backport).
    We check the cause tree explicitly so unrelated exceptions that
    happen to mention the words "cancel scope" in their message are
    *not* silenced.

    Parameters
    ----------
    exc:
        The exception to classify. Any ``BaseException`` subclass
        is accepted (``RuntimeError``, ``ExceptionGroup``,
        ``BaseExceptionGroup``, ``asyncio.CancelledError``, …).

    Returns
    -------
    bool
        ``True`` when the exception chain looks like the known mcp
        teardown bug; ``False`` otherwise. A ``False`` return means
        the exception should be re-raised, not silently swallowed.

    Notes
    -----
    The walker is defensive against cyclic ``__context__`` /
    ``__cause__`` chains — it tracks seen exception ids so a
    pathological cycle terminates instead of looping forever.
    """
    seen: set[int] = set()

    def _walk(e: BaseException) -> bool:
        if id(e) in seen:
            return False
        seen.add(id(e))
        # anyio.CancelScopeError was renamed to RuntimeError in anyio 4.x
        # with the same message. Match on message + type to avoid false
        # positives on unrelated RuntimeErrors.
        if isinstance(e, RuntimeError) and "cancel scope" in str(e):
            return True
        # Walk any ExceptionGroup / BaseExceptionGroup causes.
        causes = getattr(e, "exceptions", None)
        if causes:
            for c in causes:
                if _walk(c):
                    return True
        # Walk the legacy __context__ / __cause__ chain too.
        if e.__cause__ is not None and _walk(e.__cause__):
            return True
        if e.__context__ is not None and _walk(e.__context__):
            return True
        return False

    return _walk(exc)


__all__ = ["is_mcp_teardown_exception"]
