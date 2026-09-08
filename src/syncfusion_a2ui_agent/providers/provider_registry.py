"""Provider registry — maps names to AIProvider classes.

Built-ins are auto-registered on import. Customers can register additional
providers (subclasses of `CustomProvider`) at any time.
"""

from __future__ import annotations

from collections.abc import Callable
from threading import RLock

from .base import AIProvider

# Registry of name -> provider class
_REGISTRY: dict[str, type[AIProvider]] = {}
# A factory is anything callable that returns an AIProvider instance given kwargs.
_FACTORIES: dict[str, Callable[..., AIProvider]] = {}
_LOCK = RLock()
_BUILTINS_REGISTERED = False


def register_provider(
    name: str,
    provider: type[AIProvider] | Callable[..., AIProvider] | AIProvider,
) -> None:
    """Register a provider under `name`.

    Accepts:
      - a subclass of AIProvider (or CustomProvider) — instantiated with the
        kwargs passed to `SyncfusionAgent(...)`.
      - a callable factory (kwargs) -> AIProvider instance.
      - an already-instantiated AIProvider (used as-is, kwargs ignored).
    """
    with _LOCK:
        if isinstance(provider, type) and issubclass(provider, AIProvider):
            _REGISTRY[name] = provider
        elif isinstance(provider, AIProvider):
            _FACTORIES[name] = lambda _provider=provider, **_kw: _provider
        elif callable(provider):
            _FACTORIES[name] = provider  # type: ignore[assignment]
        else:
            raise TypeError(
                "register_provider expects a class, a callable, or an AIProvider instance."
            )


def resolve_provider(name: str, **kwargs) -> AIProvider:
    """Resolve a provider by name and instantiate it with the given kwargs.

    Factories take precedence over class registrations because they are
    strictly more specific (e.g. already-bound instance).
    """
    _ensure_builtins()
    with _LOCK:
        if name in _FACTORIES:
            return _FACTORIES[name](**kwargs)
        if name in _REGISTRY:
            return _REGISTRY[name](**kwargs)
    registered = sorted(set(_REGISTRY) | set(_FACTORIES))
    raise KeyError(
        f"Unknown AI provider {name!r}. Did you mean one of: {registered}? "
        f"Register custom providers via `register_provider(name, cls)`."
    )


def list_providers() -> list[str]:
    """Return all registered provider names (factories + classes)."""
    _ensure_builtins()
    with _LOCK:
        names = set(_REGISTRY) | set(_FACTORIES)
    return sorted(names)


def _ensure_builtins() -> None:
    global _BUILTINS_REGISTERED
    with _LOCK:
        if _BUILTINS_REGISTERED:
            return
        # Import inside the function to avoid circular imports at package import.
        from .azure_openai_provider import AzureOpenAIProvider
        from .claude_provider import ClaudeProvider
        from .gemini_provider import GeminiProvider
        from .ollama_provider import OllamaProvider
        from .openai_provider import OpenAIProvider

        for cls in (
            GeminiProvider,
            OpenAIProvider,
            AzureOpenAIProvider,
            ClaudeProvider,
            OllamaProvider,
        ):
            _REGISTRY.setdefault(cls.name, cls)
        _BUILTINS_REGISTERED = True
