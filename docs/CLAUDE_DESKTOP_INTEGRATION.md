# Claude Desktop Integration Guide

This document explains how to set up and use the iMessage Advanced Insights MCP server with Claude Desktop.

## Overview

The iMessage Advanced Insights server implements the Model Context Protocol (MCP), allowing Claude Desktop to access and analyze your iMessage data. This integration enables Claude to:

- Search and retrieve iMessage conversations
- Analyze communication patterns
- Visualize messaging networks
- Provide insights based on message content and metadata

## Setup Instructions

### 1. Configure Claude Desktop

Add the MCP server to your Claude Desktop configuration:

1. Open the Claude Desktop app
2. Go to Settings > MCP Servers
3. Click "Add Server"
4. Add the following configuration:

```json
{
  "mcpServers": {
    "imessage-insights": {
      "command": "python",
      "args": ["/path/to/your/installation/server.py"],
      "env": {
        "PYTHONPATH": "/path/to/your/installation"
      },
      "disabled": false,
      "autoApprove": []
    }
  }
}
```

### 2. Grant Required Permissions

For Claude Desktop to access your iMessage database, you need to:

1. Grant Full Disk Access to Terminal.app (or your terminal of choice)
   - Open System Preferences > Security & Privacy > Privacy
   - Select "Full Disk Access"
   - Add Terminal.app to the list

2. The first time Claude attempts to access your messages, the server will request consent. You can manage this consent using the consent management tools.

## Usage with Claude

Once set up, you can ask Claude questions about your iMessage communications:

- "Show me recent messages from [contact name]"
- "Summarize my conversations with [contact name] over the past week"
- "What topics do I discuss most frequently with [contact name]?"
- "Show me all messages containing [keyword]"
- "Analyze the sentiment of my conversations with [contact name]"
- "Visualize my messaging network"

## Available MCP Tools

When using Claude Desktop with this MCP server, the following tools are available:

### Consent Management
- `request_consent`: Request user consent to access iMessage data
- `revoke_consent`: Revoke previously granted consent
- `check_consent`: Check current consent status

### Message Access
- `get_messages`: Retrieve messages from a specific contact or group chat
- `search_messages`: Search across your messages for specific text

### Analysis Tools
- `analyze_sentiment`: Analyze sentiment in conversations
- `analyze_topics`: Extract common topics from conversations
- `get_contact_stats`: Get statistics about communications with a contact

### Visualization
- `visualize_network`: Generate a visualization of your messaging network
- `visualize_timeline`: Create a timeline visualization of message frequency

## Security and Privacy

Your privacy is a priority with this integration:

1. **Local Processing**: All data processing happens locally on your Mac
2. **Consent Management**: Explicit consent is required and can be revoked at any time
3. **Data Sanitization**: Personal information is sanitized before being presented
4. **Audit Trail**: Access to your message data is logged

## Troubleshooting

If you encounter issues with the Claude Desktop integration:

1. **Consent Issues**: Use the `check_consent` tool to verify consent status
2. **Access Issues**: Ensure Terminal.app has Full Disk Access permissions
3. **Server Not Found**: Verify the server path in your Claude Desktop configuration
4. **Memory Issues**: Try running with lower memory limits using `--memory-limit 1024`

For more detailed logging, start the server with: `python server.py --log-level DEBUG`

## Advanced Configuration

For advanced users, you can customize the integration by modifying:

1. `config.json`: Create this file based on the `config.sample.json` template
2. Environment variables: Override configuration using environment variables like `MCP_SERVER_PORT=5001`
3. Command-line options: Add options to the `args` array in your Claude Desktop configuration

## Upgrading

When upgrading to a new version:

1. Stop Claude Desktop
2. Replace the server files
3. Restart Claude Desktop

Your configuration and consent settings will be preserved.
