#!/bin/bash
# iMessage MCP Server Startup Script (Full Version)
# Created: 2025-07-31

# Navigate to project directory
cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Export environment variables
export PYTHONPATH="$(pwd)"
export MCP_DB_PATH="$HOME/Library/Messages/chat.db"
export MCP_USE_SHARDS="false"
export MCP_MEMORY_LIMIT_MB="2048"
export MCP_DISABLE_MEMORY_MONITOR="false"

echo "Starting iMessage Advanced Insights MCP Server (Full Version)..."
echo "Database: $MCP_DB_PATH"
echo "Memory Limit: ${MCP_MEMORY_LIMIT_MB}MB"
echo ""

# Run the full MCP server
python mcp_server_full.py

# Deactivate virtual environment on exit
deactivate