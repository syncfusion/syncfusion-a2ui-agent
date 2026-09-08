"""Streaming helpers for the per-op envelope extractor.

Two pieces live here:

* :class:`LiveExtractor` — a synchronous wrapper around
  :class:`catalog._StreamingState` that drains the provider's delta
  stream and surfaces ops as they become parseable.
* :func:`emit_op_events` — the per-op flush logic shared by both the
  in-band path (during a stream chunk) and the end-of-stream
  ``finalize()`` path.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from .catalog import _StreamingState, parse_envelope
from .providers.base import StreamEvent

logger = logging.getLogger("syncfusion_a2ui_agent.streaming")


def _op_kind(op: dict[str, Any]) -> str:
    """Return the A2UI v0.9 op kind for a parsed op dict."""
    for kind in ("createSurface", "updateComponents", "updateDataModel"):
        if kind in op:
            return kind
    return "unknown"


class LiveExtractor:
    """Push deltas in synchronously, pull ops out synchronously.

    The whole extractor is CPU-bound. ``feed(delta)`` runs the
    streaming state machine on the delta and stashes any
    newly-parseable ops in an internal list. ``take_ops()`` returns
    and clears that list. ``finalize()`` flushes any partial inner
    array as a best-effort parse on end-of-stream.

    Lifetime: one extractor per provider round-trip. Construct a
    fresh one at the top of every retry attempt so a partial
    envelope from a prior failed attempt doesn't leak into the
    next one.
    """

    def __init__(self) -> None:
        self._state = _StreamingState()
        self._pending: list[Any] = []
        self._closed = False

    def feed(self, delta: str) -> None:
        """Feed a delta. Newly-parseable ops land in ``take_ops()``."""
        if self._closed:
            return
        state = self._state
        item = delta
        if state.phase == "pre_tag":
            rest = state.feed_pre_tag(item)
            if not rest:
                return
            item = rest
        if state.phase == "in_tag":
            state.feed_in_tag(item)
            if state.completed_array:
                self._emit_array(state.completed_array)
                # Mark closed so subsequent deltas don't re-emit the
                # same array (the state machine has nothing left to
                # do and would re-parse ``completed_array`` every
                # call otherwise).
                self._closed = True
            return
        if state.phase == "post_tag":
            state.feed_array_scan(item)
            if state.completed_array:
                self._emit_array(state.completed_array)
                self._closed = True
            return

    def _emit_array(self, blob: str) -> None:
        try:
            for op in parse_envelope(blob):
                self._pending.append(op)
        except Exception:
            # Malformed JSON — let the non-streaming validator
            # surface the real error on the aggregated text.
            pass

    def take_ops(self) -> list[Any]:
        """Return and clear the list of newly-parseable ops."""
        out = self._pending
        self._pending = []
        return out

    def finalize(self) -> list[Any]:
        """End-of-stream: best-effort flush of any partial array.

        Returns any newly-parseable ops. Safe to call multiple times.
        """
        self._closed = True
        state = self._state
        if state.array_chars and not state.completed_array:
            self._emit_array("".join(state.array_chars))
        return self.take_ops()


class StreamOpBuffer:
    """Buffers non-createSurface ops until their surface is announced.

    The renderer cannot mount ``updateComponents`` / ``updateDataModel``
    ops before the matching ``createSurface`` has been emitted — that
    would leave the consumer referencing a surface it hasn't seen yet.
    This class holds the buffered ops in surface-id-keyed buckets and
    flushes them eagerly the moment a matching surface is announced.

    Lifetime: one buffer per request. Reusing the same buffer across
    requests would leak ops from one request into the next.
    """

    def __init__(self) -> None:
        self._pending: list[dict[str, Any]] = []
        self._surfaces_seen: set[str] = set()
        self._emitted: set[str] = set()

    def surfaces_seen(self) -> set[str]:
        return set(self._surfaces_seen)

    def emitted_surface_ids(self) -> set[str]:
        return set(self._emitted)

    def has_pending(self) -> bool:
        return bool(self._pending)

    def add_op(self, op: dict[str, Any], kind: str) -> None:
        """Record an op; update surface bookkeeping.

        ``createSurface`` ops update the seen set immediately. Other
        ops are appended to the pending list and may be flushed
        synchronously if their surface has already been announced.
        """
        if kind == "createSurface":
            cs = op.get("createSurface") or {}
            sid = cs.get("surfaceId")
            if isinstance(sid, str):
                self._surfaces_seen.add(sid)
        else:
            self._pending.append(op)

    def drain_flushable(self) -> list[dict[str, Any]]:
        """Return and remove pending ops whose surface has been seen.

        Called every time a ``createSurface`` is emitted so the
        renderer can immediately apply the components that were
        waiting for the surface.
        """
        out: list[dict[str, Any]] = []
        still_pending: list[dict[str, Any]] = []
        for o in self._pending:
            sid = o.get("updateComponents", {}).get("surfaceId") or o.get(
                "updateDataModel", {}
            ).get("surfaceId")
            if isinstance(sid, str) and sid in self._surfaces_seen:
                out.append(o)
            else:
                still_pending.append(o)
        self._pending = still_pending
        return out

    def record_emitted(self, surface_id: str | None) -> None:
        if isinstance(surface_id, str):
            self._emitted.add(surface_id)

    def reset(self) -> None:
        """Clear per-attempt state at the start of each retry."""
        self._pending = []
        self._surfaces_seen = set()
        self._emitted = set()


def emit_op_events(
    ops: Iterable[Any],
    buffer: StreamOpBuffer,
    validator: Any,
) -> list[dict[str, Any]]:
    """Validate ops and yield them in renderer-safe order.

    For each op in *ops*:

    1. Validate it against the A2UI v0.9 envelope grammar. Invalid
       ops are dropped with a warning — the renderer cannot mount
       them, and dropping them is preferable to failing the whole
       envelope.
    2. If it's a ``createSurface``, flush any pending ops that were
       waiting for the same surface, then yield the ``createSurface``
       itself.
    3. If it's a non-``createSurface`` and its surface is already
       seen, yield it directly. Otherwise, buffer it for later
       flushing.
    """
    events: list[dict[str, Any]] = []
    for op in ops:
        ok, err = validator.validate_op(op.raw)
        if not ok:
            logger.warning("Skipping invalid op in stream: %s", err)
            continue
        kind = op.kind
        if kind == "createSurface":
            sid = op.body.get("surfaceId") if isinstance(op.body, dict) else None
            buffer.add_op(op.raw, kind)
            for flushed in buffer.drain_flushable():
                events.append({"type": "op", "op": flushed, "kind": _op_kind(flushed)})
            events.append({"type": "op", "op": op.raw, "kind": kind})
            buffer.record_emitted(sid)
        else:
            buffer.add_op(op.raw, kind)
            sid = op.body.get("surfaceId") if isinstance(op.body, dict) else None
            if isinstance(sid, str) and sid in buffer.surfaces_seen():
                events.append({"type": "op", "op": op.raw, "kind": kind})
                # Remove the just-emitted op from the pending list.
                buffer.drain_flushable()  # no-op, surfaces unchanged
                # We appended it above; pop it back out.
                # The drain above only removed already-flushed ops,
                # not this new one — pull it explicitly.
                try:
                    buffer._pending.remove(op.raw)  # noqa: SLF001 — internal but stable
                except ValueError:
                    pass
    return events


__all__ = [
    "LiveExtractor",
    "StreamOpBuffer",
    "emit_op_events",
    "StreamEvent",
]
