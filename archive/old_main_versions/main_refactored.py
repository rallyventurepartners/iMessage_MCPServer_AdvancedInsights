#!/usr/bin/env python3
"""
iMessage Advanced Insights MCP Server - Refactored Entry Point.

This is the production-ready MCP server for Claude Desktop integration
with properly modularized tool implementations.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# MCP SDK imports
from mcp.server.fastmcp import FastMCP

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Local imports
from mcp_server.config import Config, load_config
from mcp_server.consent import ConsentManager
from mcp_server.db import close_database, get_database

# Tool imports
from mcp_server.tools.consent import (
    check_consent_tool,
    request_consent_tool,
    revoke_consent_tool,
)
from mcp_server.tools.health import health_check_tool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("iMessage Advanced Insights")

# Global instances
config: Optional[Config] = None
consent_manager: Optional[ConsentManager] = None


# Register consent tools
@mcp.tool()
async def request_consent(expiry_hours: int = 24):
    """Request user consent to access iMessage data."""
    return await request_consent_tool(consent_manager, expiry_hours)


@mcp.tool()
async def check_consent():
    """Check current consent status."""
    return await check_consent_tool(consent_manager)


@mcp.tool()
async def revoke_consent():
    """Revoke user consent to access iMessage data."""
    return await revoke_consent_tool(consent_manager)


# Register health check tool
@mcp.tool()
async def imsg_health_check(db_path: str = "~/Library/Messages/chat.db"):
    """Validate DB access, schema presence, index hints, and read-only mode."""
    return await health_check_tool(db_path)


# Server lifecycle handlers
@mcp.on_startup()
async def startup():
    """Initialize server resources on startup."""
    global config, consent_manager

    logger.info("Starting iMessage Advanced Insights MCP Server...")

    # Load configuration
    config = load_config()
    logger.info(f"Loaded configuration: {config}")

    # Initialize consent manager
    consent_manager = ConsentManager(config.consent_db_path)
    await consent_manager.initialize()
    logger.info("Consent manager initialized")

    # Initialize database connection
    try:
        db = await get_database(config.db_path)
        logger.info("Database connection established")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        # Continue anyway - tools will handle the error

    logger.info("Server startup complete")


@mcp.on_shutdown()
async def shutdown():
    """Clean up resources on shutdown."""
    logger.info("Shutting down iMessage Advanced Insights MCP Server...")

    # Close database connection
    await close_database()

    logger.info("Server shutdown complete")


def main():
    """Main entry point for the server."""
    try:
        # Run the MCP server
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
