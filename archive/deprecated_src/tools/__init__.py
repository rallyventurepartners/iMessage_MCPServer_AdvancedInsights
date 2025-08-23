"""
MCP tools for iMessage Advanced Insights.

This module provides all tool implementations following the MCP protocol.
Each tool is registered with proper JSON schema validation and error handling.
"""

import logging
from typing import Any, Dict

from mcp import Server

from ..config import Config
from .health import register_health_tools
from .overview import register_overview_tools
from .analytics import register_analytics_tools
from .messages import register_message_tools
from .network import register_network_tools
from .predictions import register_prediction_tools

logger = logging.getLogger(__name__)


def register_all_tools(server: Server, config: Config) -> None:
    """
    Register all available tools with the MCP server.
    
    Args:
        server: MCP server instance
        config: Server configuration
    """
    logger.info("Registering MCP tools...")
    
    # Register tool groups
    register_health_tools(server, config)
    register_overview_tools(server, config)
    register_analytics_tools(server, config)
    register_message_tools(server, config)
    register_network_tools(server, config)
    register_prediction_tools(server, config)
    
    logger.info("All tools registered successfully")