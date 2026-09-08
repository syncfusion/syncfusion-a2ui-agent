"""Shared test fixtures for the syncfusion-a2ui-agent test suite.

Pytest is configured (in ``pyproject.toml``) with ``asyncio_mode = "auto"``
so ``async def test_*`` functions run on the default event loop without
an explicit ``@pytest.mark.asyncio`` decorator.
"""

from __future__ import annotations

from typing import Any

import pytest
from factories import (
    VALID_ENVELOPE,
    VALID_ENVELOPE_TEXT,
    FakeProvider,
    ScriptedStep,
)

# Re-export the helpers so test modules that already ``from conftest
# import ...`` keep working.
__all__ = [
    "VALID_ENVELOPE",
    "VALID_ENVELOPE_TEXT",
    "FakeProvider",
    "ScriptedStep",
]


# ---------------------------------------------------------------------------
# Environment isolation
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip provider credentials and feature flags so tests never touch
    a real API or pick up stale environment state from the host."""
    for var in (
        "AZURE_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "AZURE_API_BASE",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT",
        "AZURE_OPENAI_API_VERSION",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "OLLAMA_MODEL",
        "OLLAMA_HOST",
        "SYNCFUSION_API_KEY",
        "Syncfusion_API_Key",
        "GOOGLE_API_KEY",
        # Feature flags that influence agent behaviour at construction
        # time must also be cleared so the tests don't depend on what
        # the developer happens to have exported in their shell.
        "SKILLS_PATH",
        "A2UI_STREAMING",
        "A2UI_DATA_SOURCE_FILTER",
    ):
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# A2UI envelope fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def valid_envelope_text() -> str:
    return VALID_ENVELOPE_TEXT


@pytest.fixture
def valid_envelope() -> list[dict[str, Any]]:
    return [dict(op) for op in VALID_ENVELOPE]


@pytest.fixture
def fake_provider() -> FakeProvider:
    return FakeProvider()


@pytest.fixture
def valid_envelope_step() -> ScriptedStep:
    return ScriptedStep(content=VALID_ENVELOPE_TEXT)
