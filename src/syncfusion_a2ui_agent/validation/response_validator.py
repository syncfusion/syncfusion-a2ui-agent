"""A2UI v0.9 response validator — generic envelope grammar only.

Validates that the model produced a recognised A2UI v0.9 envelope:
a JSON array of `createSurface` / `updateComponents` / `updateDataModel`
ops. Component-level properties are NOT validated here — that's the
renderer adapter's responsibility against its own per-platform catalog.

On failure, returns the error so the agent can re-call the provider
with the error appended to the prompt (max 2 retries = 3 attempts total,
per spec §12.2).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft7Validator

from ..catalog import (
    envelope_grammar_schema,
    extract_envelope,
    validate_envelope_grammar,
)


class A2UIValidationError(ValueError):
    """Raised when model output fails A2UI envelope validation."""


@dataclass
class ValidationResult:
    """Outcome of a validation pass."""

    ok: bool
    payload: Any | None = None
    error: str | None = None
    raw_text: str = ""


class ResponseValidator:
    """Validate a model response against the generic A2UI v0.9 envelope grammar."""

    def __init__(
        self,
        max_retries: int = 2,
        validator: Draft7Validator | None = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.max_retries = max_retries
        self._grammar_validator = validator or Draft7Validator(envelope_grammar_schema())

    def validate_text(self, text: str) -> ValidationResult:
        """Validate raw model text. Returns a result; never raises."""
        import logging

        _log = logging.getLogger(__name__)
        raw = text or ""
        _log.debug("[ResponseValidator] raw model output (first 2000 chars): %s", raw[:2000])
        candidate = extract_envelope(raw)
        if not candidate:
            _log.warning(
                "[ResponseValidator] No envelope found. Raw output starts with: %r",
                raw[:200],
            )
            return ValidationResult(
                ok=False,
                error="Model response contained no A2UI envelope. Expected a JSON array wrapped in <a2ui-json>...</a2ui-json>.",
                raw_text=raw,
            )
        if not candidate.strip():
            return ValidationResult(
                ok=False,
                error="Extracted envelope is empty. The model may have truncated its response.",
                raw_text=raw,
            )
        try:
            envelope = json.loads(candidate)
        except json.JSONDecodeError as exc:
            _log.warning(
                "[ResponseValidator] JSON parse failed on candidate (first 500 chars): %r. Error: %s",
                candidate[:500],
                exc,
            )
            return ValidationResult(
                ok=False,
                error=f"Invalid JSON in extracted envelope: {exc.msg} at line {exc.lineno} col {exc.colno}. The response may have been truncated.",
                raw_text=raw,
            )
        if isinstance(envelope, dict):
            envelope = [envelope]
        if not isinstance(envelope, list):
            return ValidationResult(
                ok=False,
                error="A2UI v0.9 response must be a JSON array of envelope ops.",
                raw_text=raw,
            )
        if not envelope:
            return ValidationResult(
                ok=False,
                error="Envelope array is empty.",
                raw_text=raw,
            )
        errors = validate_envelope_grammar(envelope)
        if errors:
            return ValidationResult(ok=False, error=errors[0], raw_text=raw)
        return ValidationResult(ok=True, payload=envelope, raw_text=raw)

    def max_attempts(self) -> int:
        return 1 + self.max_retries

    def validate_op(self, op: dict[str, Any]) -> tuple[bool, str | None]:
        """Validate a single envelope op against the generic A2UI v0.9 grammar.

        Returns ``(True, None)`` if the op is valid, otherwise
        ``(False, <error message>)``. This is the per-op primitive
        Level 2 (per-op streaming) uses to validate each op as it
        becomes parseable out of the token stream.

        The full-envelope :meth:`validate_text` path still exists; this
        method just exposes the per-op slice so the streaming
        transport can drop invalid ops with a structured warning
        instead of failing the whole envelope.
        """
        if not isinstance(op, dict):
            return False, f"op is not an object (got {type(op).__name__})"
        errors = validate_envelope_grammar([op])
        if errors:
            return False, errors[0]
        return True, None
