# iMessage MCP Server - Setup Guide

## Prerequisites

- macOS 10.15 or later (required for iMessage database access)
- Python 3.11 or later
- Claude Desktop app
- Full Disk Access permission for Terminal/Python

## Step 1: Install Dependencies

### Option A: Using pip (recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/imessage-advanced-insights.git
cd imessage-advanced-insights

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements_mcp.txt
```

### Option B: Using uv (alternative)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create project
uv init imessage-mcp-server
cd imessage-mcp-server

# Add dependencies
uv add mcp aiosqlite pydantic numpy scipy psutil
```

## Step 2: Grant Full Disk Access

The server needs permission to read the iMessage database.

1. Open **System Settings** → **Privacy & Security** → **Full Disk Access**
2. Click the lock to make changes
3. Add Terminal.app (or your terminal of choice)
4. If using an IDE, add the IDE as well
5. Restart Terminal after granting access

## Step 3: Test Database Access

Run the test script to verify everything is working:

```bash
python test_mcp_server.py
```

Expected output:
```
✅ FastMCP imported successfully
✅ All local modules imported successfully
✅ Configuration defaults are correct
✅ Contact hashing works
✅ PII redaction works
✅ Database connected, found 4 tables
✅ All tests passed!
```

## Step 4: Configure Claude Desktop

### Find Configuration File

```bash
# macOS
~/Library/Application Support/Claude/claude_desktop_config.json

# Linux
~/.config/Claude/claude_desktop_config.json

# Windows
%APPDATA%\Claude\claude_desktop_config.json
```

### Edit Configuration

Add the iMessage MCP server to your configuration:

```json
{
  "mcpServers": {
    "imessage": {
      "command": "python",
      "args": ["-m", "imessage_mcp_server.main"],
      "cwd": "/path/to/imessage-advanced-insights",
      "env": {
        "PYTHONPATH": "/path/to/imessage-advanced-insights/src"
      }
    }
  }
}
```

Replace `/path/to/imessage-mcp-server` with the actual path to your installation.

### Alternative: Using Absolute Python Path

If you're using a virtual environment:

```json
{
  "mcpServers": {
    "imessage": {
      "command": "/path/to/imessage-advanced-insights/venv/bin/python",
      "args": ["-m", "imessage_mcp_server.main"],
      "cwd": "/path/to/imessage-advanced-insights"
    }
  }
}
```

## Step 5: Restart Claude Desktop

1. Quit Claude Desktop completely (Cmd+Q on macOS)
2. Reopen Claude Desktop
3. The server should start automatically

## Step 6: Verify Installation

In Claude, try:

```
Can you check my iMessage database health?
```

Claude should use the `imsg_health_check` tool and report:
- Database version
- Available tables
- Index status
- Any warnings

## Troubleshooting

### "Database not found" Error

1. Verify the database exists:
   ```bash
   ls -la ~/Library/Messages/chat.db
   ```

2. Check permissions:
   ```bash
   # Should show your username as owner
   ls -la ~/Library/Messages/
   ```

3. Ensure Full Disk Access is granted (see Step 2)

### "No module named 'mcp'" Error

Install the MCP SDK:
```bash
pip install mcp
```

### "Permission denied" Error

1. Check that Terminal has Full Disk Access
2. Try running with sudo (not recommended for production):
   ```bash
   sudo python test_mcp_server.py
   ```

### Server Not Appearing in Claude

1. Check Claude logs:
   ```bash
   # macOS
   tail -f ~/Library/Logs/Claude/mcp-server-imessage.log
   ```

2. Verify JSON syntax in config file:
   ```bash
   python -m json.tool ~/Library/Application\ Support/Claude/claude_desktop_config.json
   ```

3. Try running the server manually:
   ```bash
   cd /path/to/imessage-advanced-insights
   python -m imessage_mcp_server.main
   ```
   
   The server will wait for stdio input (this is normal)

### Messages in iCloud

If most of your messages are stored in iCloud:

1. Check availability:
   ```bash
   python test_cloud_aware_tools.py
   ```

2. Download messages from iCloud:
   ```bash
   # Force download all messages
   brctl download ~/Library/Messages/
   
   # Or open Messages app and scroll through conversations
   ```

3. Use cloud-aware tools that adapt to available data

### Performance Issues

Run the benchmark:
```bash
python scripts/benchmark_mcp.py
```

This will show:
- Query performance metrics
- Memory usage
- p95 latency measurements

## Configuration Options

### Environment Variables

```bash
# Privacy settings
export IMSG_REDACT_DEFAULT=true
export IMSG_HASH_IDENTIFIERS=true

# Performance settings
export IMSG_MEMORY_LIMIT_MB=250

# Consent window (hours)
export IMSG_CONSENT_WINDOW_HOURS=24
```

### Configuration File

Create `config.json` in the project root:

```json
{
  "privacy": {
    "redact_by_default": true,
    "hash_identifiers": true,
    "preview_caps": {
      "enabled": true,
      "max_messages": 20,
      "max_chars": 160
    }
  },
  "consent": {
    "default_duration_hours": 24,
    "max_duration_hours": 720
  },
  "database": {
    "path": "~/Library/Messages/chat.db",
    "timeout_seconds": 30
  },
  "performance": {
    "memory_limit_mb": 250,
    "query_timeout_s": 30
  }
}
```

## Security Notes

1. **Consent Required**: The server requires explicit consent before accessing messages
2. **Read-Only**: The server cannot modify your iMessage database
3. **Local Only**: All processing happens on your device
4. **Hashed IDs**: Contact identifiers are hashed by default
5. **PII Redaction**: Sensitive information is automatically redacted

## Usage Examples

Once configured, you can ask Claude:

- "Show me my communication patterns from the last month"
- "When's the best time to contact John?"
- "What topics have I been discussing recently?"
- "Are there any anomalies in my messaging patterns?"
- "Show me my relationship dynamics with my top contacts"

## Updating

To update the server:

```bash
cd /path/to/imessage-mcp-server
git pull
pip install -r requirements_mcp.txt --upgrade
```

Then restart Claude Desktop.

## Support

- GitHub Issues: [Report bugs or request features]
- Documentation: See `MCP_TOOLS_REFERENCE.md` for tool details
- Privacy: See `PRIVACY_SECURITY.md` for security information