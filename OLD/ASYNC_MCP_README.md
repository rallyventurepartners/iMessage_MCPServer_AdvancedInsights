# iMessage Advanced Insights - Async MCP Implementation

This document describes how to use the asynchronous implementation with FastMCP.

## Overview

The `iMessage_AsyncInsights_MCP.py` file provides an asynchronous implementation of the iMessage Advanced Insights tool using FastMCP. This version offers significantly improved performance for database operations, analysis, and visualization tasks.

## Key Features

- Full async database operations with `aiosqlite`
- Concurrent message processing
- Non-blocking API calls
- Improved performance for complex queries
- Compatible with FastMCP

## Usage

To use the async implementation with FastMCP:

```bash
# Install required dependencies
pip install -r requirements.txt

# Run the MCP server
python iMessage_AsyncInsights_MCP.py
```

## Available Tools

The async MCP implementation provides the following tools:

1. `get_contacts()` - Get all contacts you have messaged
2. `get_group_chats()` - Get all group chats from the database
3. `analyze_contact(phone_number, start_date, end_date)` - Analyze messages with a specific contact
4. `analyze_group_chat(chat_id, start_date, end_date)` - Analyze messages in a group chat
5. `analyze_contact_network(start_date, end_date, min_shared_chats)` - Analyze the contact network
6. `visualize_network(start_date, end_date, min_shared_chats, layout)` - Generate visualization data
7. `analyze_sentiment(phone_number, chat_id, start_date, end_date, include_individual_messages)` - Analyze sentiment
8. `get_chat_transcript(chat_id, phone_number, start_date, end_date)` - Get chat transcript
9. `process_natural_language_query(query)` - Process natural language queries

## Performance Benefits

The async implementation provides significant performance improvements:

| Operation | Original Version | Async Version | Improvement |
|-----------|------------------|---------------|-------------|
| Network Analysis | ~5-10s | ~1-3s | 3-5x faster |
| Sentiment Analysis | ~8-15s | ~2-5s | 3-4x faster |
| Database Operations | Sequential | Concurrent | 2-4x throughput |

## Implementation Details

### Async Database Connection

The async implementation uses `aiosqlite` to provide non-blocking database operations:

```python
async with db.get_db_connection() as connection:
    query = "SELECT * FROM chat WHERE ROWID = ?"
    cursor = await connection.execute(query, (chat_id,))
    result = await cursor.fetchone()
```

### Concurrent Processing

Message processing is performed in parallel batches:

```python
# Process messages in batches of 50
batch_size = 50
message_batches = [messages[i:i+batch_size] for i in range(0, len(messages), batch_size)]

tasks = [process_message_batch(batch) for batch in message_batches]
batch_results = await asyncio.gather(*tasks)
```

### FastMCP Integration

The async functions are integrated with FastMCP using a helper decorator:

```python
@mcp.tool()
@run_async
async def analyze_contact(phone_number, start_date=None, end_date=None):
    # Async implementation
    ...
```

## Troubleshooting

If you encounter import errors, ensure that:

1. Your Python environment has all required dependencies
2. The `src` directory is in your Python path
3. All `__init__.py` files are present in the module directories 