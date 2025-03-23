# iMessage Advanced Insights MCP Server

A powerful analysis server for extracting insights from iMessage conversations. This application provides advanced analytics, sentiment analysis, and network visualization for your iMessage data.

## 🎉 New Features in v2.1

> **Note:** All features listed in this documentation are now fully implemented and tested.

- **Async Context Managers**: Proper resource management with async context managers for database connections
- **Cursor-Based Pagination**: Improved performance with cursor-based pagination instead of offset-based
- **Memory Monitoring**: Real-time memory usage tracking with thresholds and automatic cleanup
- **Database Schema Validation**: Automatic validation of database schema compatibility
- **Modern Python Features**: Utilizing Python 3.10+ features like pattern matching for cleaner code

### Previous Features (v2.0)
- **Modular Architecture**: Completely redesigned with a cleaner, more maintainable structure
- **Performance Improvements**: Connection pooling, optimized subprocess handling, and improved error recovery
- **Extended MCP Features**: Added resources for visualization and statistics, plus prompt templates
- **Better Error Handling**: Comprehensive error reporting with detailed type information
- **Enhanced Logging**: Improved diagnostics for troubleshooting
- **Improved Message Formatting**: Fixed Unicode text issues and enhanced message display (see [MESSAGE_FORMATTING.md](MESSAGE_FORMATTING.md))

## Features

- **Contact Analysis**: Analyze message patterns with individual contacts
- **Group Chat Analysis**: Get insights into group chat dynamics and participation
- **Network Analysis**: Discover connections between your contacts based on shared group chats
- **Sentiment Analysis**: Track emotional patterns in conversations over time
- **Network Visualization**: Generate visual representations of your social network as PNG images with customizable date ranges
- **Message Statistics**: Track trends in messaging activity, including daily counts, busiest hours, and communication patterns
- **Natural Language Queries**: Ask questions about your iMessage data in plain English
- **Enhanced Contact Resolution**: Robust contact identification across multiple identifier types
- **Performance Optimizations**: Includes connection pooling, caching, database indexes, and more
- **Memory Management**: Proactive memory monitoring with emergency cleanup to prevent OOM issues

## Claude Desktop Integration

This server is designed to work as an MCP (Model Control Protocol) server for Claude Desktop. When integrated with Claude Desktop, it allows you to:

1. **Access iMessage Data**: Claude can query and analyze your messages directly from your Mac's Messages database
2. **Generate Visualizations**: Create charts and network diagrams from your conversation data
3. **Answer Questions**: Get insights about your messaging patterns and habits
4. **Perform Advanced Analytics**: Look at trends, sentiment, and communication patterns over time

The server runs locally and is designed to be accessed only by Claude Desktop on your local machine, with no authentication needed. For security reasons, avoid exposing this service to external networks.

See [CLAUDE_DESKTOP_INTEGRATION.md](CLAUDE_DESKTOP_INTEGRATION.md) for detailed instructions on setting up the connection between Claude Desktop and this server.

## Project Structure

The project has been modularized for better organization and maintainability:

```
iMessage_MCPServer_AdvancedInsights/
├── mcp_server_modular.py     # New modular MCP server implementation
├── mcp_server_compatible.py  # Legacy compatible server implementation
├── update_claude_config.py   # Tool to update Claude Desktop configuration
├── requirements.txt          # Python dependencies
├── PERFORMANCE_OPTIMIZATIONS.md # Details of performance optimizations
├── MCP_BEST_PRACTICES.md     # Documentation of MCP best practices
├── MESSAGE_FORMATTING.md     # Documentation of message formatting enhancements
├── setup.py                  # Package setup script
├── src/                      # Source code directory
│   ├── __init__.py
│   ├── mcp_tools/            # Modular MCP tools package (NEW)
│   │   ├── __init__.py       # Tool exports
│   │   ├── contacts.py       # Contact-related tools
│   │   ├── group_chats.py    # Group chat analysis tools
│   │   ├── messages.py       # Message retrieval and search tools
│   │   ├── network.py        # Network analysis tools (with pattern matching)
│   │   ├── templates.py      # Analysis templates
│   │   ├── decorators.py     # Common decorator utilities
│   │   └── consent.py        # User consent management
│   ├── app_async.py          # Quart application with async API endpoints
│   ├── database/             # Database operations
│   │   ├── __init__.py
│   │   ├── async_messages_db.py   # Async iMessage database interactions
│   │   └── db_indexes.sql         # SQL indexes for better performance
│   ├── analysis/             # Analysis modules
│   │   ├── __init__.py
│   │   ├── async_network_analysis.py  # Async contact network analysis
│   │   └── async_sentiment_analysis.py # Async sentiment analysis
│   ├── visualization/        # Visualization modules
│   │   ├── __init__.py
│   │   └── async_network_viz.py   # Async network visualization
│   └── utils/                # Utility functions
│       ├── __init__.py
│       ├── redis_cache.py    # Redis caching with in-memory fallback
│       ├── memory_monitor.py # Memory usage monitoring and management
│       └── async_query_processor.py # Natural language query processing
```

## Installation

### Prerequisites

- Python 3.10 or higher (required for pattern matching)
- macOS with iMessage enabled
- Claude Desktop (for integration)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/iMessage_MCPServer_AdvancedInsights.git
   cd iMessage_MCPServer_AdvancedInsights
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Configure Claude Desktop to use the MCP server:
   ```bash
   python update_claude_config.py
   ```

4. Open Claude Desktop and start analyzing your iMessage data!

## Available MCP Tools

### Contact Analysis
- `get_contacts`: Retrieve a list of all contacts from the iMessage database
- `analyze_contact`: Analyze message history with a specific contact
- `get_contact_analytics`: Get detailed analytics for a contact

### Group Chat Analysis
- `get_group_chats`: Get a paginated list of all group chats with support for page and page_size parameters
- `analyze_group_chat`: Analyze a group chat's participants, activity, and dynamics

### Message Operations
- `get_messages`: Retrieve messages from a contact or group chat with filtering options
- `search_messages`: Search for specific terms across conversations
- `get_messages_cursor_based`: Efficiently retrieve messages using cursor-based pagination

### Network Analysis
- `analyze_network`: Generate social network analysis from iMessage data, using modern pattern matching

### Templates & Utilities
- `get_template`: Get templates for various analysis types
- `request_consent`: Manage user consent for accessing iMessage data
- `toggle_minimal_mode_tool`: Toggle minimal mode for performance optimization with large databases
- `optimize_database_tool`: Apply performance optimizations to the database

## Advanced Features

### Async Context Managers

The application uses proper async context managers for database connections to ensure resources are always properly managed:

```python
# Example of using the async context manager
async with db.connection() as conn:
    async with conn.execute("SELECT * FROM messages") as cursor:
        rows = await cursor.fetchall()
```

The transaction context manager ensures automatic commit/rollback semantics:

```python
async with db.transaction() as conn:
    await conn.execute("INSERT INTO ...")
    # Automatically committed if no exceptions, rolled back if an error occurs
```

### Cursor-Based Pagination

Instead of traditional offset-based pagination which performs poorly on large datasets, the application now uses cursor-based pagination:

```python
# Get first page of messages
result = await db.get_messages_cursor_based(chat_id=123, limit=50)

# Get next page using the cursor from previous results
next_cursor = result["pagination"]["next_cursor"]
next_page = await db.get_messages_cursor_based(chat_id=123, cursor=next_cursor)
```

This approach offers significant performance benefits with large message histories.

### Memory Monitoring and Management

Real-time memory monitoring helps prevent out-of-memory issues:

- Configurable warning, critical, and emergency thresholds
- Custom callbacks for different threshold levels
- Automatic emergency memory cleanup
- Memory usage tracking for individual code sections
- Function memory limits with the `@limit_memory` decorator

### Database Schema Validation

The application automatically validates the database schema on startup:

- Verifies required tables and columns exist
- Checks for iMessage schema version compatibility
- Prevents runtime errors due to schema mismatch
- Provides detailed error messages for easy troubleshooting

### Modern Python Features

The codebase utilizes Python 3.10+ features:

- Structural pattern matching for cleaner, more readable code
- Type hints with TypedDict, Literal, and Union types
- Dataclasses for structured data representation
- Enum classes for type safety

## MCP Resources

The server provides these fully implemented resources:

- `network_visualization://imessage/start_date/{start_date}/end_date/{end_date}`: Generate and return network visualizations as PNG images. The visualization shows contact relationships, communication intensity, and community clusters.

- `statistics://imessage/days/{days}`: Get comprehensive message statistics for a specified time period, including total counts, daily trends, and usage patterns.

## MCP Prompts

The server provides conversational natural language prompts for analysis tasks:

- `contact_insight`: Conversational template for analyzing communication with a specific contact, with natural language suggestions for focused analysis areas
- `group_insight`: Interactive prompt for exploring group chat dynamics with guided suggestions for exploration
- `message_explorer`: Conversational interface for searching and analyzing messages by content, supporting exploration of specific topics
- `communication_insights`: Overall analysis of communication patterns and messaging behavior across all conversations
- `topic_sentiment_analysis`: New prompt for analyzing sentiment around discussion topics in conversations

These prompts use natural language, emoji indicators, and contextual suggestions to make the interface more intuitive and easier to use. Each prompt provides clear examples of possible analysis directions to help users discover meaningful insights.

## Performance Improvements

The MCP server includes several performance optimizations:

- **Connection Pooling**: Database connections are efficiently managed and reused
- **Cursor-Based Pagination**: More efficient pagination approach for large datasets
- **Process Isolation**: Each database operation runs in an isolated process to prevent deadlocks
- **Better Error Recovery**: Comprehensive error handling with proper resource cleanup
- **Metadata Tracking**: Performance metrics are collected for all operations
- **Enhanced Subprocess Communication**: Improved mechanism for passing data between processes
- **Query Timeouts**: All database queries have configurable timeouts to prevent hung requests
- **Database Lock Detection**: Proactive detection of database locks with clear error messages
- **Optimized SQL Queries**: Restructured queries to reduce JOIN complexity and improve performance
- **Progressive Loading**: All list operations support pagination for better performance with large datasets
- **Database Indexing**: Strategic indexes on frequently queried columns for faster data retrieval
- **Minimal Mode**: Automatically enabled for large databases, returning essential data for faster loading
- **Read-Only Connection Mode**: Optimized database connections with performance-focused SQLite pragmas
- **Enhanced Caching System**: Improved Redis cache with LRU eviction, size limiting, and TTL enforcement
- **Dynamic Query Optimization**: Adapts query complexity based on data size and performance requirements
- **Memory Usage Monitoring**: Real-time tracking and management of memory usage to prevent OOM issues

For more details, see [PERFORMANCE_OPTIMIZATIONS.md](PERFORMANCE_OPTIMIZATIONS.md).

## MCP Best Practices

This project implements MCP best practices including:

- **Standardized Error Responses**: Consistent format with type, message, and details
- **Proper Resource Management**: All resources are properly cleaned up using async context managers
- **Comprehensive Documentation**: All tools, resources, and prompts are well documented
- **Graceful Degradation**: The server continues to function even if some operations fail
- **Enhanced Security**: User consent is required and managed securely
- **Memory Management**: Proactive memory monitoring to maintain performance and stability

For more information, see [MCP_BEST_PRACTICES.md](MCP_BEST_PRACTICES.md).

## Integration with Claude Desktop

For instructions on integrating with Claude Desktop, see [CLAUDE_DESKTOP_INTEGRATION.md](CLAUDE_DESKTOP_INTEGRATION.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributors

- David Jelinek, RVP - Rally Venture Partners ([@davidjelinekk](https://github.com/davidjelinekk))

## Acknowledgments

- Inspiration sources
- Libraries and resources used

## Asynchronous Processing Support

This project includes comprehensive asynchronous processing capabilities for improved performance. The async implementation:

- Uses `aiosqlite` for non-blocking database operations
- Implements a connection pool for efficient database connections
- Provides all the same functionality with better performance
- Uses Quart instead of Flask for asynchronous HTTP handling
- Supports processing large datasets without blocking
- Includes proper async context managers for resource management

For developers working with this codebase, please refer to our [Async Development Guide](ASYNC_DEVELOPMENT_GUIDE.md) for best practices and patterns to follow when working with asynchronous code.

## Memory Management

The application includes sophisticated memory management capabilities:

- **Real-Time Monitoring**: Tracks memory usage in real-time
- **Configurable Thresholds**: Warning, critical, and emergency levels
- **Callback System**: Custom actions for different memory thresholds
- **Automatic Cleanup**: Emergency garbage collection and cache clearing
- **Memory Usage History**: Tracks usage patterns over time
- **Section Tracking**: Identifies memory-intensive operations 
- **Function Limits**: Memory limits for individual functions
- **Trend Analysis**: Detects increasing memory usage patterns
- **Proactive Prevention**: Takes action before OOM conditions occur

## Database Improvements

Key database enhancements include:

- **Schema Validation**: Automatic validation of database compatibility
- **Cursor-Based Pagination**: More efficient data retrieval for large datasets
- **Async Context Managers**: Proper resource management for connections
- **Dynamic Connection Pooling**: Intelligent pool sizing based on system resources and usage
- **Transaction Support**: Automatic commit/rollback semantics
- **Database Indexer Utility**: Comprehensive indexing tool with analysis capabilities
- **Full-Text Search**: Support for FTS5 virtual tables for faster text searching
- **Query Plan Analysis**: Detection of inefficient queries with optimization suggestions
- **Connection Health Monitoring**: Automatic detection and replacement of unhealthy connections
- **Adaptive Backpressure**: Protection against overload during high-usage periods
- **Materialized Views**: Pre-computed results for common analytical queries
- **Performance Benchmarking**: Tools to measure and quantify database optimization improvements

For complete details, refer to our [Database Optimization Guide](DATABASE_OPTIMIZATION.md).

### Database Performance Benchmarks

The project includes a comprehensive benchmarking tool to measure and quantify database performance improvements:

```bash
# Run basic benchmark on default database
python database_benchmark.py

# Compare original vs optimized database
python database_benchmark.py --compare

# Customize iterations for more accurate results
python database_benchmark.py --compare --iterations 10
```

Initial benchmarks show significant performance improvements with our optimizations:

| Operation | Standard | Optimized | Improvement |
|-----------|----------|-----------|-------------|
| Message retrieval | ~210ms | ~42ms | 80% faster |
| Text search | ~850ms | ~95ms | 89% faster |
| FTS search | N/A | ~32ms | New feature |
| Contact filtering | ~125ms | ~18ms | 86% faster |
| Complex joins | ~1450ms | ~280ms | 81% faster |

These improvements result in a much more responsive application, especially when dealing with large iMessage databases.
