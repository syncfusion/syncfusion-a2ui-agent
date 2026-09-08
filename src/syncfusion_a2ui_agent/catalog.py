"""A2UI v0.9 envelope schema grammar — generic to all platforms.

A2UI v0.9 responses are a JSON array of envelope operations. There are
three operation kinds:

  - `createSurface`    — open a new renderer surface
  - `updateComponents` — replace / add / remove components on that surface
  - `updateDataModel`  — write data into the surface's data model
                         (typically used for suggestions, lookups, etc.)

This module does NOT validate component-level properties (those depend on
the renderer / catalog being mounted on the client). The platform MCP is
treated as Knowledge Transfer (KT) — it teaches the AI the component
shapes via the prompt builder, NOT as a hard runtime validator.

What IS validated here is the **envelope grammar** itself:
  - Every op is an object with `version: "v0.9"`.
  - `createSurface` must declare `surfaceId`; `catalogId` is optional.
  - `updateComponents` must declare `surfaceId` + `components` (array).
  - `updateDataModel` must declare `surfaceId`, `path`, and `value`.

Per-component property validation is the renderer adapter's responsibility.
That keeps one grammar shared across React / Blazor / Angular / WinForms / etc.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

A2UI_VERSION: str = "v0.9"

_A2UI_TAG_RE = re.compile(r"<a2ui-json>\s*([\s\S]*?)\s*</a2ui-json>", re.IGNORECASE)
# Matches an UNCLOSED opening tag — handles truncated LLM responses.
_A2UI_TAG_OPEN_RE = re.compile(r"<a2ui-json>\s*([\s\S]*)", re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)

OP_KINDS = ("createSurface", "updateComponents", "updateDataModel")


@dataclass
class EnvelopeOp:
    """A single A2UI v0.9 envelope operation."""

    raw: dict[str, Any]
    kind: str
    body: dict[str, Any]


# ---------------------------------------------------------------- schema
def envelope_grammar_schema() -> dict[str, Any]:
    """Return the generic A2UI v0.9 envelope grammar schema."""
    return {
        "$schema": "https://json-schema.org/draft-07/schema#",
        "$id": "https://a2ui.org/specification/v0_9/envelope.json",
        "title": "A2UI v0.9 Envelope Grammar",
        "description": (
            "Generic A2UI v0.9 envelope grammar — shared by every platform "
            "(React, Blazor, Angular, WinForms, etc.). Per-component property "
            "validation is NOT enforced here; the renderer adapter validates "
            "against its own catalog."
        ),
        "oneOf": [
            _create_surface_op(),
            _update_components_op(),
            _update_data_model_op(),
        ],
    }


def _create_surface_op() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["version", "createSurface"],
        "properties": {
            "version": {"const": A2UI_VERSION},
            "createSurface": {
                "type": "object",
                "additionalProperties": False,
                "required": ["surfaceId"],
                "properties": {
                    "surfaceId": {"type": "string", "minLength": 1},
                    "catalogId": {"type": "string"},
                },
            },
        },
    }


def _update_components_op() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["version", "updateComponents"],
        "properties": {
            "version": {"const": A2UI_VERSION},
            "updateComponents": {
                "type": "object",
                "additionalProperties": False,
                "required": ["surfaceId", "components"],
                "properties": {
                    "surfaceId": {"type": "string", "minLength": 1},
                    "components": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            # Component-level property validation is the
                            # renderer's job (per-platform catalog). At the
                            # envelope grammar level we only require `id` and
                            # allow everything else through.
                            "required": ["id"],
                            "properties": {
                                "id": {"type": "string", "minLength": 1},
                            },
                        },
                    },
                },
            },
        },
    }


def _update_data_model_op() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["version", "updateDataModel"],
        "properties": {
            "version": {"const": A2UI_VERSION},
            "updateDataModel": {
                "type": "object",
                "additionalProperties": False,
                "required": ["surfaceId", "path", "value"],
                "properties": {
                    "surfaceId": {"type": "string", "minLength": 1},
                    "path": {"type": "string", "minLength": 1},
                    "value": {},
                },
            },
        },
    }


# ---------------------------------------------------------------- utils
def _extract_json_array(text: str) -> str:
    """Extract a balanced JSON array from the start of *text*. Returns '' if not found."""
    stripped = text.strip()
    if not stripped.startswith("["):
        return ""
    depth = 0
    in_str = False
    escape = False
    for i, ch in enumerate(stripped):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return stripped[: i + 1]
    return ""


def _strip_fences(text: str) -> str:
    """If *text* is wrapped in a markdown code fence, unwrap it."""
    m = _FENCE_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def extract_envelope(text: str) -> str:
    """Best-effort: pull the JSON array out of fences / ``<a2ui-json>`` tags.

    Handles all common LLM output patterns:
    - ``<a2ui-json>[...]</a2ui-json>``  (normal)
    - ``<a2ui-json>[...`` (truncated — no closing tag)
    - ``<a2ui-json>```json\\n[...]\\n```</a2ui-json>``  (fence inside tag)
    - ` ``` `json\\n[...]\\n` ``` ` (bare fence, no tags)
    - bare JSON array
    """
    if not text:
        return ""

    # 1. Closed <a2ui-json>...</a2ui-json> tag — best case.
    tag = _A2UI_TAG_RE.search(text)
    if tag:
        inner = tag.group(1).strip()
        # The model sometimes wraps the array in a code fence inside the tag.
        if not inner.startswith("["):
            inner = _strip_fences(inner)
        candidate = _extract_json_array(inner) or inner
        return candidate

    # 2. Unclosed <a2ui-json> — model was truncated before writing </a2ui-json>.
    #    Extract whatever follows the opening tag and try to parse the array.
    open_tag = _A2UI_TAG_OPEN_RE.search(text)
    if open_tag:
        inner = open_tag.group(1).strip()
        if not inner.startswith("["):
            inner = _strip_fences(inner)
        # Use the balanced extractor — it returns a complete array if found,
        # or empty string if the JSON itself is also truncated.
        candidate = _extract_json_array(inner)
        if candidate:
            return candidate
        # JSON is truncated too — return the partial content so the validator
        # can report a meaningful parse error (not a silent empty-string error).
        return inner if inner else ""

    # 3. Bare code fence.
    fence = _FENCE_RE.search(text)
    if fence:
        inner = fence.group(1).strip()
        candidate = _extract_json_array(inner) or inner
        return candidate

    # 4. Raw JSON array anywhere in the text.
    candidate = _extract_json_array(text)
    if candidate:
        return candidate

    # 5. Return stripped text and let the caller report the parse error.
    return text.strip()


def parse_envelope(blob: str) -> list[EnvelopeOp]:
    """Parse a JSON array (or single-object) envelope into typed ops."""
    obj = json.loads(blob) if blob else []
    if isinstance(obj, dict):
        obj = [obj]
    if not isinstance(obj, list):
        raise ValueError("A2UI v0.9 response must be a JSON array of envelope ops.")
    ops: list[EnvelopeOp] = []
    for item in obj:
        if not isinstance(item, dict):
            raise ValueError("Each envelope op must be an object.")
        kind = next((k for k in OP_KINDS if k in item), None)
        if kind is None:
            raise ValueError(f"Envelope op missing one of {OP_KINDS}: keys={list(item.keys())}")
        ops.append(EnvelopeOp(raw=item, kind=kind, body=item[kind]))
    return ops


def validate_envelope_grammar(envelope: list[dict[str, Any]]) -> list[str]:
    """Validate a parsed envelope against the generic grammar schema."""
    try:
        import jsonschema
    except Exception:  # pragma: no cover
        return ["jsonschema is not installed; cannot validate envelope."]
    schema = envelope_grammar_schema()
    subschemas: dict[str, dict[str, Any]] = {
        "createSurface": schema["oneOf"][0],
        "updateComponents": schema["oneOf"][1],
        "updateDataModel": schema["oneOf"][2],
    }
    errors: list[str] = []
    for i, op in enumerate(envelope):
        if not isinstance(op, dict):
            errors.append(f"op[{i}] is not an object")
            continue
        # Match the op against the right sub-schema (each is a discriminator
        # by required key) so error messages point at the offending field
        # rather than the unhelpful "is not valid under any of the given
        # schemas" oneOf fallback.
        matched_kind: str | None = None
        for kind in OP_KINDS:
            if kind in op:
                matched_kind = kind
                break
        if matched_kind is None:
            errors.append(f"op[{i}] missing one of {OP_KINDS}: keys={list(op.keys())}")
            continue
        sub = subschemas[matched_kind]
        for e in jsonschema.Draft7Validator(sub).iter_errors(op):
            path = "/".join(str(p) for p in e.absolute_path) or "<root>"
            errors.append(f"op[{i}] {e.message} (at {path})")
    return errors


def surface_ids_in(envelope: list[dict[str, Any]]) -> list[str]:
    """Collect every surfaceId referenced in the envelope."""
    out: list[str] = []
    for op in envelope:
        if not isinstance(op, dict):
            continue
        for kind in OP_KINDS:
            body = op.get(kind) or {}
            sid = body.get("surfaceId")
            if isinstance(sid, str):
                out.append(sid)
    return out


def find_create_surface(
    envelope: list[dict[str, Any]],
    catalog_id: str | None = None,
) -> EnvelopeOp | None:
    """Return the first `createSurface` op (optionally filtering by catalogId)."""
    for op in envelope:
        if not isinstance(op, dict):
            continue
        cs = op.get("createSurface")
        if not cs:
            continue
        if catalog_id is None or cs.get("catalogId") == catalog_id:
            return EnvelopeOp(raw=op, kind="createSurface", body=cs)
    return None


# ---------------------------------------------------------------- streaming extractors
# Level 2 of the live-streaming plan needs to parse the token stream
# incrementally and yield each `EnvelopeOp` as soon as it's complete.
# `extract_envelope_iter(chunks)` and `extract_ops_iter(chunks)` are the
# per-chunk siblings of the regex-based `extract_envelope(text)` /
# `parse_envelope(blob)`. They share the same three phases:
#
#   1. Tag-closing detection — buffer chunks, watch for `</a2ui-json>`.
#   2. Balanced-bracket scan — reuse the same logic as
#      `_extract_json_array` to find the matching `]` for the inner
#      array. The scan is stateful so it survives chunk boundaries.
#   3. Per-op split — once the array is complete, walk it and yield
#      each op as a separate `EnvelopeOp`.
#
# A "completed" inner array is yielded the moment its matching `]`
# arrives; ops inside the array are then yielded in order. The state
# machine is intentionally minimal: a per-call ``_StreamingState``
# object holds the buffers and the parser phase.

_A2UI_CLOSE_TAG = "</a2ui-json>"


def _strip_inner_fence(text: str) -> str:
    """Mirror of ``_strip_fences`` for streaming — drop a ``` fence if present."""
    fence = _FENCE_RE.search(text)
    return fence.group(1).strip() if fence else text


class _StreamingState:
    """Mutable per-call state for the per-chunk extractors."""

    def __init__(self) -> None:
        self.phase: str = "pre_tag"  # pre_tag | in_tag | post_tag | done
        self.tag_buffer: str = ""  # content carried across chunks
        # For the balanced-bracket scan (phase == 'in_tag', once we've
        # found the opening `[`):
        self.array_start: int = -1
        self.scan_index: int = 0
        self.in_str: bool = False
        self.escape: bool = False
        self.depth: int = 0
        self.stripped_lead: bool = False
        # Buffer the inner array (between `[` and `]`) so the
        # post-tag phase can parse it.
        self.array_chars: list[str] = []
        # The full completed inner array string once we have it.
        self.completed_array: str = ""

    def feed_pre_tag(self, chunk: str) -> str:
        """Phase 1: return whatever part of *chunk* (combined with the
        previous tail) comes after the opening ``<a2ui-json>`` tag.
        Returns ``""`` if the tag is not yet visible.
        """
        if self.phase != "pre_tag":
            return chunk
        combined = self.tag_buffer + chunk
        idx = combined.lower().find("<a2ui-json>")
        if idx < 0:
            # Hold a tail in case the opening tag is split across
            # chunks. 32 chars is more than enough.
            self.tag_buffer = combined[-32:]
            return ""
        tag_end = idx + len("<a2ui-json>")
        rest = combined[tag_end:]
        self.tag_buffer = ""
        self.phase = "in_tag"
        return rest

    def feed_in_tag(self, rest: str) -> bool:
        """Phase 2: append *rest* to the inner buffer. If the closing
        ``</a2ui-json>`` tag is now present, transition to
        ``post_tag`` and start feeding the inner content to the
        array scan. Returns ``True`` if the tag was found in this call.

        The array scan is fed incrementally — the moment the closing
        tag arrives, most of the array has already been processed.
        """
        if self.phase == "post_tag":
            return True
        if self.phase != "in_tag":
            return False
        # First, feed whatever's already in the buffer (from previous
        # chunks) into the array scan.
        if self.tag_buffer:
            self.feed_array_scan(self.tag_buffer)
            self.tag_buffer = ""
        close_idx = rest.lower().find(_A2UI_CLOSE_TAG)
        if close_idx < 0:
            # Closing tag not yet visible. Feed the array scan with
            # this chunk.
            self.feed_array_scan(rest)
            if self.completed_array:
                return True
            # Keep a 14-char tail in case the closing tag starts
            # here but doesn't finish (length of `</a2ui-json>` is
            # 13, +1 to be safe). Only do so if the chunk ends with
            # the start of the closing tag.
            tail_start = rest.rfind("<")
            if tail_start >= 0 and tail_start >= len(rest) - 13:
                self.tag_buffer = rest[tail_start:]
            return False
        inner = rest[:close_idx]
        self.phase = "post_tag"
        self.feed_array_scan(inner)
        return True

    def feed_array_scan(self, inner: str) -> None:
        """Run the balanced-bracket scan over the inner content,
        starting from the first ``[``. Sets ``self.completed_array`` if
        the matching ``]`` is found, otherwise leaves
        ``self.array_chars`` partially filled and ``self.scan_index``
        pointing at the first unconsumed position so the next chunk
        can resume.

        This is a stateful variant of :func:`_extract_json_array` that
        handles chunk boundaries, string escaping, and nested
        arrays/objects correctly.
        """
        if not inner:
            return
        # Drop a leading code fence if the model wrapped the array in one.
        if not self.stripped_lead:
            inner = _strip_inner_fence(inner)
            self.stripped_lead = True
        if not inner:
            return
        # If we haven't found the opening `[` yet, find it.
        if self.array_start < 0:
            idx = inner.find("[")
            if idx < 0:
                # No array start in this chunk — discard the prefix.
                self.scan_index += len(inner)
                return
            self.array_start = self.scan_index + idx
            # Reset scan state to begin parsing at the `[`.
            self.scan_index = self.array_start
            self.depth = 0
            self.in_str = False
            self.escape = False
            inner = inner[idx:]
        # Continue the balanced-bracket scan from `self.scan_index`
        # forward. The state machine mirrors `_extract_json_array`
        # but ALSO appends characters to `array_chars` so the
        # completed string is a valid JSON array.
        for ch in inner:
            if self.escape:
                self.escape = False
                if self.depth > 0:
                    self.array_chars.append(ch)
                self.scan_index += 1
                continue
            if ch == "\\":
                self.escape = True
                if self.depth > 0:
                    self.array_chars.append(ch)
                self.scan_index += 1
                continue
            if ch == '"':
                self.in_str = not self.in_str
                if self.depth > 0:
                    self.array_chars.append(ch)
                self.scan_index += 1
                continue
            if self.in_str:
                if self.depth > 0:
                    self.array_chars.append(ch)
                self.scan_index += 1
                continue
            if ch == "[":
                # Append the opening `[` (depth goes 0 → 1, the
                # post-condition `self.depth > 0` will include it).
                self.depth += 1
                if self.depth > 0:
                    self.array_chars.append(ch)
                self.scan_index += 1
                continue
            if ch == "]":
                # Append the closing `]` BEFORE decrementing so the
                # completed array string is a valid JSON array.
                self.array_chars.append(ch)
                self.depth -= 1
                if self.depth == 0:
                    # Done — the array is complete.
                    self.completed_array = "".join(self.array_chars)
                    return
                self.scan_index += 1
                continue
            if self.depth > 0:
                self.array_chars.append(ch)
            self.scan_index += 1


def extract_envelope_iter(chunks: Iterable[str]) -> Iterator[str]:
    """Yield the **inner JSON array** of ``<a2ui-json>...</a2ui-json>`` once.

    Drives a per-chunk state machine that:
      - buffers until ``<a2ui-json>`` is seen,
      - buffers until the matching ``</a2ui-json>`` is seen,
      - runs a balanced-bracket scan over the inner content (handling
        string-escapes, nested arrays/objects, chunk boundaries),
      - yields the complete inner array as a single string the
        moment the matching ``]`` is consumed.

    The caller is expected to iterate the chunks in order (e.g. from
    a provider's streaming response). Any error condition yields
    nothing — the caller decides how to surface a partial / truncated
    envelope.
    """
    state = _StreamingState()
    for chunk in chunks:
        if state.phase == "pre_tag":
            rest = state.feed_pre_tag(chunk)
            if not rest:
                # Still waiting for the opening tag.
                continue
            chunk = rest
            # fall through to in_tag processing
        if state.phase == "in_tag":
            state.feed_in_tag(chunk)
            if state.completed_array:
                yield state.completed_array
                return
            continue
        if state.phase == "post_tag":
            # Closing tag is past; the array scan is in progress
            # (or done). Feed the new chunk directly to the scan.
            state.feed_array_scan(chunk)
            if state.completed_array:
                yield state.completed_array
                return
            continue
        return
    # Stream ended without yielding — nothing more to do.


def extract_ops_iter(chunks: Iterable[str]) -> Iterator[EnvelopeOp]:
    """Yield each ``EnvelopeOp`` as it becomes parseable in the stream.

    Drives :func:`extract_envelope_iter` to find the complete inner
    array, then walks it with :func:`parse_envelope` and yields each
    op in order. The inner array is yielded exactly once, so the
    per-op fan-out happens only after the entire JSON array is
    available — this is the minimum fidelity Level 2 needs (per-op,
    not per-token).

    For a Tier 3 dashboard, the array is parseable in one piece
    within ~100 ms of the closing `</a2ui-json>`. The renderer
    mounts the surface (after `createSurface`) ~100 ms after the
    model finishes writing the array, and components stream in over
    the next ~50–200 ms while the ADK fans them out.
    """
    for blob in extract_envelope_iter(chunks):
        try:
            yield from parse_envelope(blob)
        except Exception:
            # Malformed JSON in the inner array — surface nothing.
            # The non-streaming validator is the safety net on the
            # full text path.
            return
        return
