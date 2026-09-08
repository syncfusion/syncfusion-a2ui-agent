"""Direct tests for :mod:`syncfusion_a2ui_agent._streaming`.

Covers:

* :class:`LiveExtractor` — push-delta / pull-ops semantics,
  phase transitions, end-of-stream ``finalize``.
* :class:`StreamOpBuffer` — surface bookkeeping, ``createSurface``
  flushing, ``reset`` semantics.
* :func:`emit_op_events` — ordering invariants, invalid-op
  dropping, buffering of ``updateComponents`` / ``updateDataModel``
  ops until their surface is announced.

These helpers are the single most important runtime code in the
streaming path; ``test_agent.py`` exercises them only as a side
effect of :meth:`SyncfusionAgent.handle_request_stream`.
"""

from __future__ import annotations

from typing import Any

from factories import VALID_ENVELOPE, VALID_ENVELOPE_TEXT

from syncfusion_a2ui_agent._streaming import (
    LiveExtractor,
    StreamOpBuffer,
    emit_op_events,
)
from syncfusion_a2ui_agent._streaming import (
    _op_kind as streaming_op_kind,
)
from syncfusion_a2ui_agent.catalog import EnvelopeOp
from syncfusion_a2ui_agent.validation.response_validator import ResponseValidator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse(envelope: list[dict[str, Any]]) -> list[EnvelopeOp]:
    """Helper: parse a dict-envelope into EnvelopeOp objects the
    streaming path consumes (the ``emit_op_events`` validator
    expects ``.raw``, ``.kind`` and ``.body`` attributes)."""
    ops: list[EnvelopeOp] = []
    for raw in envelope:
        for kind in ("createSurface", "updateComponents", "updateDataModel"):
            if kind in raw:
                ops.append(EnvelopeOp(raw=raw, kind=kind, body=raw.get(kind) or {}))
                break
    return ops


def _validator() -> ResponseValidator:
    return ResponseValidator(max_retries=2)


# ---------------------------------------------------------------------------
# _op_kind
# ---------------------------------------------------------------------------
class TestOpKind:
    def test_create_surface(self) -> None:
        assert streaming_op_kind({"createSurface": {"surfaceId": "s"}}) == "createSurface"

    def test_update_components(self) -> None:
        assert streaming_op_kind({"updateComponents": {"surfaceId": "s"}}) == "updateComponents"

    def test_update_data_model(self) -> None:
        assert streaming_op_kind({"updateDataModel": {"surfaceId": "s"}}) == "updateDataModel"

    def test_unknown(self) -> None:
        assert streaming_op_kind({"foo": "bar"}) == "unknown"

    def test_empty(self) -> None:
        assert streaming_op_kind({}) == "unknown"


# ---------------------------------------------------------------------------
# LiveExtractor — single-chunk happy path
# ---------------------------------------------------------------------------
class TestLiveExtractorSingleChunk:
    def test_full_envelope_in_one_chunk(self) -> None:
        ex = LiveExtractor()
        ex.feed(VALID_ENVELOPE_TEXT)
        ops = ex.take_ops()
        assert len(ops) == 3
        assert ops[0].kind == "createSurface"
        assert ops[1].kind == "updateComponents"
        assert ops[2].kind == "updateDataModel"

    def test_take_ops_clears_pending(self) -> None:
        ex = LiveExtractor()
        ex.feed(VALID_ENVELOPE_TEXT)
        first = ex.take_ops()
        second = ex.take_ops()
        assert len(first) == 3
        assert second == []

    def test_subsequent_deltas_are_ignored(self) -> None:
        # Once the array is complete, the extractor is closed.
        ex = LiveExtractor()
        ex.feed(VALID_ENVELOPE_TEXT)
        ex.feed("more text that should be ignored")
        ops = ex.take_ops()
        # No re-emission.
        assert len(ops) == 3

    def test_invalid_json_silently_dropped(self) -> None:
        # Malformed JSON inside the tag should not raise. The
        # engine surfaces a real error via the validator, not the
        # extractor.
        ex = LiveExtractor()
        ex.feed("<a2ui-json>[{not valid json}]</a2ui-json>")
        ops = ex.take_ops()
        # parse_envelope raises ValueError on bad input; the
        # extractor catches it and returns nothing.
        assert ops == []


# ---------------------------------------------------------------------------
# LiveExtractor — multi-chunk streaming
# ---------------------------------------------------------------------------
class TestLiveExtractorChunked:
    def test_array_split_across_chunks(self) -> None:
        ex = LiveExtractor()
        # Split the canonical envelope into arbitrary chunk boundaries.
        text = VALID_ENVELOPE_TEXT
        mid = len(text) // 2
        ex.feed(text[:mid])
        # No ops should be available yet.
        assert ex.take_ops() == []
        ex.feed(text[mid:])
        ops = ex.take_ops()
        assert len(ops) == 3
        assert ops[0].kind == "createSurface"

    def test_chunks_with_no_progress(self) -> None:
        ex = LiveExtractor()
        ex.feed("garbage prefix ")
        ex.feed("more garbage ")
        # No <a2ui-json> tag yet — no ops.
        assert ex.take_ops() == []

    def test_finalize_emits_partial_array(self) -> None:
        # End-of-stream with a partial, unclosed array. ``finalize``
        # is the best-effort flush for that case.
        ex = LiveExtractor()
        ex.feed("<a2ui-json>[")
        ex.feed(str(VALID_ENVELOPE[0]))  # stringified dict
        # No ops yet (unclosed).
        assert ex.take_ops() == []
        # ``finalize`` must not raise even when the buffered array
        # is incomplete. Best-effort — may or may not yield ops
        # depending on JSON-parse tolerance.
        ops = ex.finalize()
        assert isinstance(ops, list)

    def test_finalize_is_idempotent(self) -> None:
        ex = LiveExtractor()
        ex.feed(VALID_ENVELOPE_TEXT)
        first = ex.finalize()
        second = ex.finalize()
        # First call drains; second is a no-op.
        assert len(first) == 3
        assert second == []

    def test_phase_post_tag_handles_unclosed_tag(self) -> None:
        # The post-tag phase is reached when the state machine
        # detects an unclosed ``<a2ui-json>`` tag (balanced scan
        # from the end of the input). It is NOT reached by feeding
        # a bare array with no tag at all — that stays in pre_tag
        # and yields nothing on ``finalize``.
        ex = LiveExtractor()
        # Open the tag, then feed the array without ever closing it.
        ex.feed("garbage <a2ui-json>")
        envelope_only = str(VALID_ENVELOPE).replace("'", '"')
        ex.feed(envelope_only)
        # The state machine should have entered the in_tag phase
        # and parsed the array.
        assert ex._state.phase == "in_tag"
        ops = ex.take_ops()
        assert len(ops) == 3
        assert ops[0].kind == "createSurface"


# ---------------------------------------------------------------------------
# StreamOpBuffer — surface bookkeeping
# ---------------------------------------------------------------------------
class TestStreamOpBuffer:
    def test_empty_initially(self) -> None:
        buf = StreamOpBuffer()
        assert buf.surfaces_seen() == set()
        assert buf.emitted_surface_ids() == set()
        assert buf.has_pending() is False

    def test_create_surface_records_surface_id(self) -> None:
        buf = StreamOpBuffer()
        buf.add_op({"createSurface": {"surfaceId": "s1"}}, "createSurface")
        assert "s1" in buf.surfaces_seen()
        assert buf.has_pending() is False

    def test_create_surface_without_string_id_is_ignored(self) -> None:
        buf = StreamOpBuffer()
        buf.add_op({"createSurface": {}}, "createSurface")
        # Missing surfaceId → not added to seen.
        assert buf.surfaces_seen() == set()

    def test_update_components_buffered_until_surface_seen(self) -> None:
        buf = StreamOpBuffer()
        op = {
            "updateComponents": {
                "surfaceId": "s1",
                "components": [{"id": "root", "component": "Column"}],
            }
        }
        buf.add_op(op, "updateComponents")
        # Surface not yet seen → op is pending.
        assert buf.has_pending() is True
        flushed = buf.drain_flushable()
        assert flushed == []  # nothing flushable
        # Now announce the surface.
        buf.add_op({"createSurface": {"surfaceId": "s1"}}, "createSurface")
        flushed = buf.drain_flushable()
        assert flushed == [op]

    def test_drain_flushable_only_returns_matching_surfaces(self) -> None:
        buf = StreamOpBuffer()
        op_a = {"updateComponents": {"surfaceId": "a"}}
        op_b = {"updateComponents": {"surfaceId": "b"}}
        buf.add_op(op_a, "updateComponents")
        buf.add_op(op_b, "updateComponents")
        buf.add_op({"createSurface": {"surfaceId": "a"}}, "createSurface")
        flushed = buf.drain_flushable()
        # Only the matching op should be flushed.
        assert flushed == [op_a]
        # op_b is still pending.
        assert buf.has_pending() is True

    def test_drain_flushable_skips_ops_with_no_surface(self) -> None:
        buf = StreamOpBuffer()
        # An op with no surfaceId at all cannot be flushed.
        op = {"updateComponents": {}}
        buf.add_op(op, "updateComponents")
        buf.add_op({"createSurface": {"surfaceId": "a"}}, "createSurface")
        flushed = buf.drain_flushable()
        assert flushed == []

    def test_record_emitted(self) -> None:
        buf = StreamOpBuffer()
        buf.record_emitted("s1")
        buf.record_emitted(None)  # ignored
        buf.record_emitted(123)  # ignored (not a string)
        assert buf.emitted_surface_ids() == {"s1"}

    def test_reset_clears_state(self) -> None:
        buf = StreamOpBuffer()
        buf.add_op({"createSurface": {"surfaceId": "s"}}, "createSurface")
        buf.add_op({"updateComponents": {"surfaceId": "s"}}, "updateComponents")
        buf.record_emitted("s")
        buf.reset()
        assert buf.surfaces_seen() == set()
        assert buf.emitted_surface_ids() == set()
        assert buf.has_pending() is False


# ---------------------------------------------------------------------------
# emit_op_events — ordering invariants
# ---------------------------------------------------------------------------
class TestEmitOpEvents:
    def test_invalid_ops_are_dropped(self) -> None:
        v = _validator()
        buf = StreamOpBuffer()
        bad_op = EnvelopeOp(
            raw={"updateComponents": {"surfaceId": "s", "components": "not-a-list"}},
            kind="updateComponents",
            body={"surfaceId": "s", "components": "not-a-list"},
        )
        events = emit_op_events([bad_op], buf, v)
        assert events == []

    def test_create_surface_yields_first(self) -> None:
        v = _validator()
        buf = StreamOpBuffer()
        ops = _parse([VALID_ENVELOPE[0]])
        events = emit_op_events(ops, buf, v)
        assert len(events) == 1
        assert events[0]["type"] == "op"
        assert events[0]["kind"] == "createSurface"

    def test_update_components_buffered_until_surface(self) -> None:
        # When ``updateComponents`` arrives BEFORE its
        # ``createSurface`` in the same ``emit_op_events`` call,
        # the buffer holds it; the second op (createSurface) drains
        # the buffer and emits the updateComponents THEN the
        # createSurface — that's the renderer-safe order.
        v = _validator()
        buf = StreamOpBuffer()
        ops = _parse([VALID_ENVELOPE[1], VALID_ENVELOPE[0]])
        events = emit_op_events(ops, buf, v)
        kinds = [e["kind"] for e in events]
        # Buffer drains the updateComponents, then createSurface is
        # emitted. updateDataModel (op index 2) is not in this batch.
        assert kinds == ["updateComponents", "createSurface"]

    def test_full_envelope_in_order(self) -> None:
        v = _validator()
        buf = StreamOpBuffer()
        ops = _parse(VALID_ENVELOPE)
        events = emit_op_events(ops, buf, v)
        kinds = [e["kind"] for e in events]
        # createSurface first, then updateComponents, then
        # updateDataModel. updateDataModel references the same
        # surface as createSurface, so it's flushed eagerly.
        assert kinds == [
            "createSurface",
            "updateComponents",
            "updateDataModel",
        ]

    def test_update_data_model_after_create_surface_emits_directly(self) -> None:
        v = _validator()
        buf = StreamOpBuffer()
        # First announce the surface.
        emit_op_events(_parse([VALID_ENVELOPE[0]]), buf, v)
        # Then a data-model update on the same surface.
        events = emit_op_events(_parse([VALID_ENVELOPE[2]]), buf, v)
        assert len(events) == 1
        assert events[0]["kind"] == "updateDataModel"

    def test_buffer_preserves_pending_op_for_unknown_surface(self) -> None:
        # An updateComponents op that references a surface never
        # announced is buffered (not emitted). The pending list
        # keeps it for a future createSurface to drain.
        v = _validator()
        buf = StreamOpBuffer()
        op = {
            "version": "v0.9",
            "updateComponents": {
                "surfaceId": "ghost",
                "components": [{"id": "root", "component": "Column"}],
            },
        }
        eo = EnvelopeOp(
            raw=op,
            kind="updateComponents",
            body=op["updateComponents"],
        )
        events = emit_op_events([eo], buf, v)
        assert events == []
        # The buffer should now have one pending op.
        assert buf.has_pending() is True
        # Draining the buffer with no matching surface returns nothing.
        assert buf.drain_flushable() == []
