# iMessage Advanced Insights User Guide

This guide provides detailed information on using the iMessage MCP Server tools with Claude Desktop. These tools allow you to analyze and visualize your iMessage conversations in various ways.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Contact Analytics](#contact-analytics)
3. [Group Chat Analysis](#group-chat-analysis)
4. [Topic Analysis](#topic-analysis)
5. [Visualizations](#visualizations)
6. [Performance Considerations](#performance-considerations)
7. [Privacy and Security](#privacy-and-security)
8. [Troubleshooting](#troubleshooting)

## Getting Started

### Prerequisites

- macOS 11.0 or higher
- Claude Desktop application
- Python 3.8 or higher
- Access to your iMessage database

### Installation

1. Install the iMessage MCP Server:
   ```bash
   pip install imessage-mcp-server
   ```

2. Configure the server by editing the config file:
   ```bash
   imessage-mcp-server --generate-config
   ```

3. Start the server:
   ```bash
   imessage-mcp-server start
   ```

4. Connect Claude Desktop to the MCP server by adding the following to your Claude Desktop configuration:
   ```json
   "mcpServers": {
     "imessage": {
       "command": "imessage-mcp-server",
       "args": ["serve"],
       "env": {}
     }
   }
   ```

### First-Time Setup

When you first use the MCP server, you'll be asked to provide consent for Claude to access your iMessage data. You can choose from several consent levels:

- **Basic**: Only allows access to metadata (contact names, message counts)
- **Standard**: Allows access to message content but with sensitive information redacted
- **Full**: Allows access to all message content

You can modify your consent settings at any time using the `consent` tool.

## Contact Analytics

Contact analytics tools provide insights into your messaging patterns with individual contacts.

### Listing Contacts

To get a list of your contacts with basic activity statistics:

```
Use the get_contacts tool to list my active contacts.
```

**Example Response:**
```json
{
  "contacts": [
    {
      "name": "John Smith",
      "message_count": 1234,
      "last_active": "2025-03-15T10:20:30Z",
      "contact_info": "***-***-1234"
    },
    ...
  ],
  "total": 15,
  "page": 1,
  "page_size": 30
}
```

You can filter contacts by activity:

```
Show me contacts I've messaged in the last week.
```

### Detailed Contact Analytics

To get comprehensive analytics for a specific contact:

```
Show analytics for my conversations with John Smith.
```

This provides various metrics including:
- Message counts (sent, received, attachments)
- Activity patterns (most active days/hours)
- Conversation metrics (typical conversation length, response times)
- Content metrics (average message length, emoji usage)

### Relationship Strength

To analyze relationship strength based on communication patterns:

```
What's my messaging relationship like with Sarah?
```

This provides:
- Overall relationship strength score
- Specific indicators (response rate, conversation depth, etc.)
- Comparison to other contacts
- Changes over time

## Group Chat Analysis

Group chat analysis tools help you understand dynamics in your group conversations.

### Listing Group Chats

To get a list of your group chats:

```
Show me all my active group chats.
```

**Example Response:**
```json
{
  "group_chats": [
    {
      "name": "Family Chat",
      "participants": ["Mom", "Dad", "Sister"],
      "message_count": 5678,
      "last_active": "2025-04-01T15:30:45Z"
    },
    ...
  ],
  "total": 8,
  "page": 1,
  "page_size": 20
}
```

### Group Chat Analytics

For detailed analysis of a specific group chat:

```
Analyze my Family Chat group.
```

This provides:
- Message distribution among participants
- Activity patterns
- Conversation flow metrics
- Topic distribution

### Member Dynamics

To analyze interaction patterns between members:

```
Show the interaction dynamics in my Work Team chat.
```

This reveals:
- Who interacts most with whom
- Conversation starters and active participants
- Subgroup detection
- Communication role analysis (mediator, information provider, etc.)

## Topic Analysis

Topic analysis tools extract and analyze conversation topics.

### Conversation Topics

To identify main topics in your conversations:

```
What topics do I discuss most with Alex?
```

**Example Response:**
```json
{
  "contact": "Alex Chen",
  "topics": [
    {
      "topic": "Work Projects",
      "percentage": 35.2,
      "keywords": ["deadline", "meeting", "client", "project"]
    },
    ...
  ],
  "date_range": {
    "start": "2025-01-01T00:00:00Z",
    "end": "2025-04-01T00:00:00Z"
  }
}
```

### Topic Trends

To analyze how topics change over time:

```
How have my conversation topics with Chris changed over the past 3 months?
```

This shows:
- Topic distribution changes
- New topics emerging
- Topics fading out
- Sentiment shifts within topics

### Topic Sentiment

To analyze sentiment for specific topics:

```
What's the sentiment when I discuss work with Jordan?
```

This provides:
- Overall sentiment for the topic
- Sentiment trends over time
- Comparison to other topics
- Sample positive/negative messages

## Visualizations

Visualization tools help you see patterns in your messaging data.

### Network Visualization

To visualize your messaging network:

```
Show me a visualization of my messaging network over the past year.
```

This creates a graph showing:
- Contacts as nodes
- Connection strength as edge weight
- Group chat connections
- Communication clusters

### Timeline Visualizations

To visualize messaging activity over time:

```
Generate a timeline visualization of my messages with Taylor over the past month.
```

This creates a chart showing:
- Message frequency
- Sent vs. received distribution
- Time-of-day patterns
- Optional sentiment overlay

### Topic Distribution Visualizations

To visualize topic distribution:

```
Create a visualization of the topics in my Book Club chat.
```

This generates a chart (pie, bar, or treemap) showing the relative frequency of different topics.

## Performance Considerations

### Large Databases

If you have a large iMessage database (>100k messages), some operations may take longer to complete. The server implements various optimizations:

- Query caching
- Connection pooling
- Asynchronous processing

For best performance:
- Use date ranges to limit the scope of analysis
- Start with summary views before drilling down
- Allow long-running operations to complete without interruption

### Database Sharding

For very large databases, the server supports database sharding:

```
Enable database sharding for improved performance.
```

### Memory Usage

The server monitors memory usage and will automatically adjust caching behavior to prevent excessive memory consumption. You can manually control memory usage:

```
Set maximum memory usage to 1GB.
```

## Privacy and Security

### Data Protection

Your message data remains on your device and is processed locally. The MCP server:
- Never sends your messages to external servers
- Sanitizes personal information when returning results
- Respects your consent settings
- Provides detailed access logs

### Managing Consent

You can update your consent settings at any time:

```
Update my consent settings to standard level.
```

Or revoke consent completely:

```
Revoke access to my iMessage data.
```

### Access Logs

To review when and how your data has been accessed:

```
Show me the access logs for my iMessage data.
```

## Troubleshooting

### Common Issues

#### Connection Problems

If Claude cannot connect to the MCP server:

1. Verify the server is running:
   ```bash
   imessage-mcp-server status
   ```

2. Check the server logs:
   ```bash
   cat ~/.imessage-mcp-server/logs/server.log
   ```

3. Restart the server:
   ```bash
   imessage-mcp-server restart
   ```

#### Permission Issues

If the server cannot access your iMessage database:

1. Verify Terminal has Full Disk Access (System Preferences > Privacy)
2. Check the permission logs:
   ```bash
   cat ~/.imessage-mcp-server/logs/permissions.log
   ```

#### Slow Performance

If analyses are taking too long:

1. Check server load:
   ```bash
   imessage-mcp-server stats
   ```

2. Consider enabling database sharding
3. Use more specific queries with narrower date ranges

### Getting Help

If you encounter issues not covered in this guide:

1. Check the FAQ documentation
2. Look at the GitHub issues
3. Generate a diagnostic report:
   ```bash
   imessage-mcp-server diagnostics
   ```

## Advanced Usage

### Custom Analyses

Claude can combine multiple tools to perform custom analyses:

```
Compare my messaging patterns with Alex and Jordan over the past 6 months, focusing on response times and topic differences.
```

### Data Export

You can export analysis results:

```
Export my conversation analytics with Sam to CSV.
```

### Batch Processing

For analyzing multiple contacts or chats:

```
Analyze topic trends for all my family group chats.
```

---

This guide covers the basic usage of the iMessage MCP Server tools. For more advanced features and detailed API documentation, please refer to the Developer Documentation.
