#!/bin/bash
# Run tests for iMessage MCP Server

echo "Running iMessage MCP Server Tests..."
echo "===================================="

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "pytest is not installed. Install with: pip install pytest pytest-asyncio"
    exit 1
fi

# Run tests with coverage if available
if command -v coverage &> /dev/null; then
    echo "Running tests with coverage..."
    coverage run -m pytest tests/ -v
    coverage report
    coverage html
    echo "Coverage report generated in htmlcov/"
else
    echo "Running tests without coverage..."
    pytest tests/ -v
fi

echo ""
echo "Test run complete!"