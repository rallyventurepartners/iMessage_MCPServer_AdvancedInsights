#!/usr/bin/env python3
"""Test minimal MCP server to debug startup issue."""

import logging
from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create minimal server
mcp = FastMCP("Test Server")

@mcp.tool()
async def test_tool():
    """Test tool."""
    return {"message": "Hello from test tool"}

if __name__ == "__main__":
    logger.info("Starting test server...")
    try:
        mcp.run(transport='stdio')
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)