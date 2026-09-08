"""CustomProvider — base class for customer-supplied providers.

Customers subclass this and pass the class (not an instance) to
`register_provider()`, or they pass an instance to
`SyncfusionAgent(model=MyProvider(api_key=...))`.
"""

from __future__ import annotations

from .base import AIProvider, ModelResponse


class CustomProvider(AIProvider):
    """Base class for customer-defined AI providers.

    Subclasses only need to implement `generate()`. The SDK owns retry,
    timeout, and error handling centrally (see ResponseValidator).
    """

    name: str = "custom"

    async def generate(  # pragma: no cover - abstract re-export
        self,
        messages: list[dict],
        tools=None,
        context: dict | None = None,
    ) -> ModelResponse:
        raise NotImplementedError("Subclasses of CustomProvider must implement `generate()`.")
