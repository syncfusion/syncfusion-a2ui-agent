"""MCP manager — fully pluggable, no platform gating."""

from .mcp_manager import MCPManager, MCPServerConfig, MCPToolCall

__all__ = [
    "MCPServerConfig",
    "MCPManager",
    "MCPToolCall",
]
