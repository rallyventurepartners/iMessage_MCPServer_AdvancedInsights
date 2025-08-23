#!/bin/bash
# iMessage MCP Server Startup Script
# Created: 2025-07-31

# Navigate to project directory
cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Start the server with appropriate settings
echo "Starting iMessage MCP Server..."
echo "Database: ~/Library/Messages/chat.db"
echo "Port: 5000"
echo ""

# Run the server
python server.py \
    --db-path ~/Library/Messages/chat.db \
    --memory-limit 2048 \
    --log-level INFO

# Deactivate virtual environment on exit
deactivate