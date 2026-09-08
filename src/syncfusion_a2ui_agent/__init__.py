"""syncfusion-a2ui-agent — thin orchestration ADK for Syncfusion A2UI agents.

The ADK ships zero Syncfusion component knowledge. UI knowledge lives
exclusively behind Syncfusion Platform MCPs registered at runtime.
"""

from .agent import SyncfusionAgent
from .mcp.mcp_manager import MCPServerConfig
from .providers.base import AIProvider, ModelResponse
from .providers.custom_provider import CustomProvider
from .providers.provider_registry import register_provider, resolve_provider
from .skills import SkillMetadata, SkillRegistry, SkillRouter
from .validation.response_validator import ResponseValidator

__all__ = [
    "SyncfusionAgent",
    "AIProvider",
    "ModelResponse",
    "CustomProvider",
    "register_provider",
    "resolve_provider",
    "MCPServerConfig",
    "SkillMetadata",
    "SkillRegistry",
    "SkillRouter",
    "ResponseValidator",
]

__version__ = "0.1.0"
