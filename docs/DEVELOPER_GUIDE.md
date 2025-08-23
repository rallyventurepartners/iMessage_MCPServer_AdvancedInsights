# iMessage Advanced Insights Developer Guide

This guide provides technical information for developers who want to understand, modify, or extend the iMessage MCP Server. It covers the architecture, APIs, and development practices.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Project Structure](#project-structure)
3. [Core Components](#core-components)
4. [MCP Tool Implementation](#mcp-tool-implementation)
5. [Database Interface](#database-interface)
6. [Performance Utilities](#performance-utilities)
7. [Testing](#testing)
8. [Contribution Guidelines](#contribution-guidelines)

## Architecture Overview

The iMessage MCP Server is built around the Model Context Protocol (MCP), which enables Claude Desktop to communicate with your local iMessage database. The server provides tools that Claude can use to query and analyze message data.

### High-Level Architecture

```
┌─────────────────┐      ┌──────────────────┐      ┌───────────────┐
│                 │      │                  │      │               │
│ Claude Desktop ├─────►│ iMessage MCP     ├─────►│ macOS         │
│                 │      │ Server           │      │ iMessage DB   │
│                 │◄─────┤                  │◄─────┤               │
└─────────────────┘      └──────────────────┘      └───────────────┘
       MCP                  Tools & Analysis          SQLite Access
    Communication
```

### Communication Flow

1. Claude Desktop sends a request to the MCP server using the MCP protocol
2. The MCP server processes the request via registered tools
3. Tools access the iMessage database via the database interface
4. Results are processed, sanitized, and returned to Claude
5. Claude presents the results to the user

### Key Design Principles

- **Privacy First**: All processing happens locally, with data sanitization
- **Performance**: Optimizations for handling large iMessage databases
- **Modularity**: Separation of concerns allowing easy extension
- **Error Handling**: Comprehensive error handling and recovery

## Project Structure

```
imessage-mcp-server/
├── docs/                    # Documentation
├── scripts/                 # Utility scripts
├── src/                     # Source code
│   ├── database/            # Database access and management
│   ├── exceptions/          # Custom exceptions
│   ├── mcp_tools/           # MCP tool implementations
│   ├── server/              # MCP server implementation
│   ├── utils/               # Utility functions and helpers
│   └── app_async.py         # Main application entry point
├── tests/                   # Test suite
├── config.sample.json       # Sample configuration
└── setup.py                 # Package setup
```

## Core Components

### Server Module

The server module handles MCP protocol communication with Claude Desktop.

```python
# src/server/server.py
class MCPServer:
    def __init__(self, config):
        self.config = config
        self.tool_registry = ToolRegistry()
        
    async def start(self):
        # Initialize and start the server
        
    async def handle_request(self, request):
        # Process MCP requests
```

### Tool Registry

The tool registry manages all available MCP tools.

```python
# src/mcp_tools/registry.py
class ToolRegistry:
    def __init__(self):
        self.tools = {}
        
    def register_tool(self, name, description, handler):
        # Register a new tool
        
    def get_tool(self, name):
        # Retrieve a registered tool
        
    def list_tools(self):
        # List all available tools
```

### Database Interface

The database module provides a unified interface to the iMessage database.

```python
# src/database/db_base.py
class DatabaseInterface:
    async def connect(self):
        # Connect to the database
        
    async def get_messages(self, query_params):
        # Retrieve messages matching parameters
        
    async def get_contacts(self, query_params):
        # Retrieve contacts matching parameters
```

## MCP Tool Implementation

### Creating a New Tool

Tools are implemented as async functions that process requests and return responses.

```python
# src/mcp_tools/example.py
from .registry import register_tool

@register_tool(
    name="example_tool",
    description="Example tool that demonstrates the tool structure",
)
async def example_tool(param1, param2=None):
    """
    Example tool documentation.
    
    Args:
        param1: First parameter description
        param2: Optional second parameter description
        
    Returns:
        Dictionary with results
    """
    # Tool implementation
    result = await process_data(param1, param2)
    return success_response(result)
```

### Tool Response Structure

Tools should return responses in a consistent format:

```python
# src/utils/responses.py
def success_response(data):
    return {
        "status": "success",
        "data": data
    }

def error_response(error):
    return {
        "status": "error",
        "error": str(error),
        "error_type": type(error).__name__
    }

def paginated_response(items, page, page_size, total_items, additional_data=None):
    response = {
        "status": "success",
        "data": items,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_items": total_items,
            "total_pages": (total_items + page_size - 1) // page_size
        }
    }
    
    if additional_data:
        response.update(additional_data)
        
    return response
```

### Error Handling

Tools should use structured exception handling:

```python
# Example of error handling in a tool
try:
    # Attempt operation
    result = await db.get_data()
    return success_response(result)
except DatabaseError as e:
    # Handle database-specific errors
    logger.error(f"Database error: {e}")
    return error_response(e)
except Exception as e:
    # Handle unexpected errors
    logger.error(f"Unexpected error: {e}")
    logger.error(traceback.format_exc())
    return error_response(ToolExecutionError(f"Failed to execute: {str(e)}"))
```

## Database Interface

### Database Factory

The database factory creates the appropriate database interface based on configuration.

```python
# src/database/db_factory.py
async def get_database():
    """
    Get the appropriate database interface based on configuration.
    """
    config = get_config()
    db_type = config.get("database", {}).get("type", "default")
    
    if db_type == "sharded":
        return ShardedAsyncMessagesDb()
    else:
        return AsyncMessagesDb()
```

### Database Implementations

#### Standard Database

```python
# src/database/async_messages_db.py
class AsyncMessagesDb(DatabaseInterface):
    async def connect(self):
        # Connect to the standard iMessage database
        
    async def get_messages(self, **kwargs):
        # Implement message retrieval
```

#### Sharded Database

```python
# src/database/sharded_async_messages_db.py
class ShardedAsyncMessagesDb(DatabaseInterface):
    async def connect(self):
        # Connect to sharded database configuration
        
    async def get_messages(self, **kwargs):
        # Implement message retrieval across shards
```

### Query Optimization

Example of using the query cache with the database:

```python
# Using query cache with database methods
from ..utils.query_cache import cached_query

class OptimizedDatabase(DatabaseInterface):
    @cached_query(ttl=300)  # Cache for 5 minutes
    async def get_frequently_accessed_data(self, **kwargs):
        # This result will be cached
        return await self._expensive_query(**kwargs)
```

## Performance Utilities

### Query Caching

The query cache system improves performance by storing results of expensive database operations.

```python
# src/utils/query_cache.py
@cached_query(ttl=60)  # Cache for 60 seconds
async def example_cached_function(param1, param2):
    # Expensive operation
    result = await db.complex_query(param1, param2)
    return result
```

### Connection Pooling

The connection pooling system manages database connections efficiently.

```python
# src/database/connection_manager.py
async with db_pool.connection() as conn:
    # Use the connection
    result = await conn.execute_query("SELECT * FROM messages")
```

### Batch Processing

For operations that benefit from batching:

```python
# Creating a batch processor
processor = BatchProcessor(
    processor=db.bulk_insert,
    max_batch_size=100,
    max_wait_time=0.05
)

# Using the batch processor
await processor.submit(item)
```

## Testing

### Unit Tests

Unit tests are in the `tests/` directory and use pytest:

```python
# tests/test_tool.py
import pytest
from src.mcp_tools.example import example_tool

@pytest.mark.asyncio
async def test_example_tool():
    # Test the tool with various inputs
    result = await example_tool("test_param")
    assert result["status"] == "success"
    assert "data" in result
```

### Integration Tests

Integration tests verify that components work together:

```python
# tests/integration/test_database_tools.py
@pytest.mark.asyncio
async def test_contact_analytics_with_database():
    # Set up test database
    test_db = TestDatabase()
    await test_db.populate_test_data()
    
    # Test tool with the database
    result = await get_contact_analytics_tool("test_contact")
    
    # Verify results
    assert result["status"] == "success"
    assert result["data"]["contact"]["name"] == "Test Contact"
```

### Running Tests

To run the tests:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_tool.py

# Run with coverage
pytest --cov=src
```

## Contribution Guidelines

### Code Style

This project follows the PEP 8 style guide with a few exceptions:

- Line length limit: 100 characters
- Use double quotes for strings
- Use trailing commas in multi-line collections

We use black and isort for code formatting. Run before committing:

```bash
black src tests
isort src tests
```

### Commit Messages

Follow the conventional commits specification:

- feat: A new feature
- fix: A bug fix
- docs: Documentation changes
- style: Code style changes (formatting, etc.)
- refactor: Code refactoring without changing functionality
- perf: Performance improvements
- test: Adding or updating tests
- chore: Maintenance tasks

### Pull Request Process

1. Fork the repository and create a feature branch
2. Implement your changes with tests
3. Ensure all tests pass and code is formatted
4. Submit a pull request with a clear description
5. Address any review comments

### Development Workflow

1. Create an issue for the feature or bug
2. Discuss the approach in the issue
3. Implement the change
4. Add tests
5. Update documentation
6. Submit a pull request

## API Reference

### Contact Analytics API

#### get_contacts

Returns a paginated list of contacts with basic statistics.

Parameters:
- `search_term` (optional): Filter contacts by name or identifier
- `active_since` (optional): Only include contacts active after this date
- `page` (optional): Page number (default: 1)
- `page_size` (optional): Results per page (default: 30)

#### get_contact_analytics

Returns detailed analytics for a specific contact.

Parameters:
- `contact_id`: Contact identifier
- `start_date` (optional): Start of analysis period
- `end_date` (optional): End of analysis period

#### get_communication_timeline

Returns a timeline of communication frequency with a contact.

Parameters:
- `contact_id`: Contact identifier
- `interval` (optional): Aggregation interval (day, week, month)
- `start_date` (optional): Start of timeline
- `end_date` (optional): End of timeline

#### get_relationship_strength

Calculates relationship strength indicators for a contact.

Parameters:
- `contact_id` (optional): Contact identifier (if none, returns top contacts)
- `limit` (optional): Number of top contacts to return (default: 10)

### Group Chat API

#### get_group_chats

Returns a paginated list of group chats.

Parameters:
- `search_term` (optional): Filter group chats by name
- `active_since` (optional): Only include group chats active after this date
- `page` (optional): Page number (default: 1)
- `page_size` (optional): Results per page (default: 20)

#### get_group_chat_analytics

Returns detailed analytics for a specific group chat.

Parameters:
- `chat_id`: Group chat identifier
- `start_date` (optional): Start of analysis period
- `end_date` (optional): End of analysis period

### Topic Analysis API

#### analyze_conversation_topics

Extracts and analyzes topics from conversations.

Parameters:
- `contact_id`: Contact identifier
- `start_date` (optional): Start of analysis period
- `end_date` (optional): End of analysis period
- `topic_count` (optional): Number of topics to return (default: 5)
- `include_sentiment` (optional): Whether to include sentiment analysis (default: true)

#### get_topic_trends

Analyzes how topics change over time.

Parameters:
- `contact_id` (optional): Contact identifier
- `topic` (optional): Specific topic to analyze
- `interval` (optional): Time interval for trend analysis (day, week, month)
- `start_date` (optional): Start of analysis period
- `end_date` (optional): End of analysis period
- `topic_count` (optional): Number of topics to return (default: 5)

### Visualization API

#### visualize_message_network

Generates a visualization of the messaging network.

Parameters:
- `time_period` (optional): Time period to analyze (default: "1 year")
- `min_message_count` (optional): Minimum message count to include a contact (default: 10)
- `format` (optional): Output format (svg, png, json) (default: "svg")
- `include_group_chats` (optional): Whether to include group chats (default: true)
- `layout` (optional): Network layout algorithm (default: "force")

#### visualize_contact_timeline

Generates a timeline visualization for a contact.

Parameters:
- `contact_id`: Contact identifier
- `interval` (optional): Time interval (hour, day, week, month) (default: "day")
- `start_date` (optional): Start date
- `end_date` (optional): End date
- `include_sentiment` (optional): Whether to include sentiment (default: true)
- `format` (optional): Output format (svg, png, json) (default: "svg")

## Internal Module Reference

For additional details on internal modules and classes, see:

- [Database Class Reference](./DATABASE_REFERENCE.md)
- [MCP Server Implementation](./MCP_SERVER_REFERENCE.md)
- [Tool Registry API](./TOOL_REGISTRY_REFERENCE.md)

---

This Developer Guide provides an overview of the iMessage MCP Server architecture and implementation. For more specific questions or issues, please refer to the GitHub repository or contact the maintainers.
