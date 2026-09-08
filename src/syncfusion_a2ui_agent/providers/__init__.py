"""AI provider manager — uniform interface across vendors, registry-driven."""

from .base import AIProvider, ModelResponse, ProviderError
from .custom_provider import CustomProvider
from .provider_registry import (
    _ensure_builtins,
    list_providers,
    register_provider,
    resolve_provider,
)

# Eagerly register built-in vendors so callers can resolve by name.
_ensure_builtins()

__all__ = [
    "AIProvider",
    "ModelResponse",
    "ProviderError",
    "CustomProvider",
    "register_provider",
    "resolve_provider",
    "list_providers",
]
