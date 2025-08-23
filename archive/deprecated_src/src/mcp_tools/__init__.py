"""
MCP tools for the iMessage Advanced Insights server.

This package contains all the MCP tools that are exposed to the client.
Each module in this package should contain one or more tools registered
with the tool registry.
"""

# Import the registry for other modules to use
from .registry import register_tool, set_mcp_instance, list_tools, get_tool

# Import consent tools - this is done explicitly to ensure they're registered
from . import consent

# Make sure the core modules are imported so their tools are registered
__all__ = [
    "register_tool",
    "set_mcp_instance",
    "list_tools",
    "get_tool",
    "consent",
]
