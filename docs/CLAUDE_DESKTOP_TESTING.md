# Claude Desktop Testing Guide

This guide will walk you through testing the iMessage MCP Server with Claude Desktop.

## Prerequisites

- macOS 10.15 or later
- Claude Desktop app installed
- Python 3.11 or later
- Terminal with Full Disk Access permission

## Step 1: Grant Full Disk Access

The server needs permission to read your iMessage database.

1. Open **System Settings** → **Privacy & Security** → **Full Disk Access**
2. Click the lock to make changes
3. Add Terminal.app (or your terminal of choice)
4. Restart Terminal after granting access

## Step 2: Install Dependencies

```bash
# Clone this repository
git clone <your-repo-url>
cd iMessage_MCPServer_AdvancedInsights

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements_mcp.txt
```

## Step 3: Configure Claude Desktop

1. Find your Claude Desktop configuration file:
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Linux: `~/.config/Claude/claude_desktop_config.json`
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`

2. Edit the configuration file:

```json
{
  "mcpServers": {
    "imessage": {
      "command": "/path/to/iMessage_MCPServer_AdvancedInsights/venv/bin/python",
      "args": ["-m", "mcp_server.main"],
      "cwd": "/path/to/iMessage_MCPServer_AdvancedInsights",
      "env": {
        "PYTHONPATH": "/path/to/iMessage_MCPServer_AdvancedInsights"
      }
    }
  }
}
```

Replace `/path/to/iMessage_MCPServer_AdvancedInsights` with the actual path to your cloned repository.

## Step 4: Restart Claude Desktop

1. Quit Claude Desktop completely (Cmd+Q on macOS)
2. Reopen Claude Desktop
3. The server should start automatically

## Step 5: Test Basic Functionality

### 5.1 Health Check

Ask Claude:
```
Can you check my iMessage database health?
```

Expected: Claude should use the `imsg_health_check` tool and report:
- Database version
- Available tables
- Index recommendations
- Read-only status

### 5.2 Request Consent

Ask Claude:
```
I'd like to analyze my iMessage data. Can you request consent for 24 hours?
```

Expected: Claude should use the `request_consent` tool and confirm consent is granted.

### 5.3 Summary Overview

Ask Claude:
```
Can you give me a summary of my iMessage activity?
```

Expected: Claude should use the `imsg_summary_overview` tool and report:
- Total messages
- Number of unique contacts
- Date range
- Message distribution (sent/received)
- Platform breakdown (iMessage/SMS)

## Step 6: Test Analytics Tools

### 6.1 Relationship Intelligence

Ask Claude:
```
Show me my communication patterns from the last 30 days
```

Expected: Claude should use the `imsg_relationship_intelligence` tool to show:
- Top contacts by message volume
- Response time patterns
- Communication balance

### 6.2 Conversation Topics

Ask Claude:
```
What topics have I been discussing recently?
```

Expected: Claude should use the `imsg_conversation_topics` tool to show:
- Frequent keywords/topics
- Trending terms

### 6.3 Best Contact Times

Ask Claude:
```
When's the best time to contact people based on my messaging history?
```

Expected: Claude should use the `imsg_best_contact_time` tool to show optimal contact windows.

## Step 7: Test Privacy Features

### 7.1 Contact Resolution

Ask Claude:
```
Can you resolve the contact for phone number +1-555-123-4567?
```

Expected: Claude should use the `imsg_contact_resolve` tool and return a hashed contact ID.

### 7.2 Sample Messages

Ask Claude:
```
Show me a few recent messages
```

Expected: Claude should use the `imsg_sample_messages` tool and show:
- Redacted/truncated message previews
- Hashed contact IDs
- Limited to 20 messages max

### 7.3 Revoke Consent

Ask Claude:
```
Please revoke my consent to access iMessage data
```

Expected: Claude should use the `revoke_consent` tool and confirm consent is revoked.

## Step 8: Performance Testing

Ask Claude to run multiple analyses:
```
Can you:
1. Check database health
2. Show me a summary overview
3. Analyze conversation topics from the last week
4. Show response time patterns
5. Display communication cadence
```

Expected: All operations should complete quickly (each under 1.5 seconds).

## Troubleshooting

### Server Not Appearing in Claude

1. Check Claude logs:
   ```bash
   # macOS
   tail -f ~/Library/Logs/Claude/mcp-server-imessage.log
   ```

2. Verify configuration syntax:
   ```bash
   python3 -m json.tool ~/Library/Application\ Support/Claude/claude_desktop_config.json
   ```

3. Test server manually:
   ```bash
   cd /path/to/iMessage_MCPServer_AdvancedInsights
   source venv/bin/activate
   python -m mcp_server.main
   ```
   
   You should see: `Server running on stdio`

### Permission Errors

1. Ensure Terminal has Full Disk Access
2. Check database exists:
   ```bash
   ls -la ~/Library/Messages/chat.db
   ```

### Import Errors

1. Ensure you're using the virtual environment:
   ```bash
   which python
   # Should show: /path/to/iMessage_MCPServer_AdvancedInsights/venv/bin/python
   ```

2. Verify all dependencies installed:
   ```bash
   pip list | grep mcp
   # Should show: mcp 1.x.x
   ```

## Expected Behavior

When everything is working correctly:

1. **Consent Required**: All data access tools require active consent
2. **Privacy by Default**: Contact IDs are hashed, PII is redacted
3. **Preview Caps**: Message samples limited to 20 messages, 160 chars each
4. **Performance**: All queries complete within 1.5 seconds
5. **Read-Only**: Database cannot be modified

## Security Notes

- The server operates in read-only mode
- All processing happens locally
- Contact identifiers are hashed with per-session salts
- Sensitive information is automatically redacted
- Consent expires after the specified duration

## Example Conversation Flow

1. "Check my iMessage database health" → Health check results
2. "Grant consent for 24 hours" → Consent granted
3. "Show me a summary of my messages" → Overview statistics
4. "Who do I message most?" → Top contacts (hashed IDs)
5. "What are we talking about?" → Topic analysis
6. "When should I contact them?" → Best contact times
7. "Revoke consent" → Access revoked

## Additional Tools Available

- `imsg_sentiment_evolution`: Analyze sentiment trends over time
- `imsg_response_time_distribution`: See response time patterns
- `imsg_cadence_calendar`: View communication patterns by hour/day
- `imsg_anomaly_scan`: Detect unusual communication patterns
- `imsg_network_intelligence`: Analyze group chat networks

Feel free to explore these tools by asking Claude relevant questions!