# iMessage Advanced Insights - MCP Server

A privacy-first MCP server that provides advanced analytics for iMessage conversations through Claude Desktop.

## Overview

This server transforms your iMessage data into profound insights about relationships and communication patterns. All processing is 100% local with strong privacy guarantees.

### Key Features

- **Relationship Intelligence**: Track communication patterns, response times, and conversation balance
- **Topic Analysis**: Extract conversation themes with trend visualization
- **Sentiment Evolution**: Monitor emotional dynamics over time
- **Network Intelligence**: Discover social connections through group chats
- **Predictive Analytics**: Best contact times and anomaly detection
- **Privacy-First**: Contact ID hashing, PII redaction, consent management

## Installation

### Requirements

- macOS 10.15+ (for iMessage database access)
- Python 3.11+
- Claude Desktop app

### Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/imessage-mcp-server.git
cd imessage-mcp-server
```

2. Install dependencies:
```bash
pip install -r requirements_mcp.txt
```

3. Configure Claude Desktop:

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "imessage": {
      "command": "python",
      "args": ["-m", "mcp_server.main"],
      "cwd": "/path/to/imessage-mcp-server"
    }
  }
}
```

4. Restart Claude Desktop

## Usage

### First Time Setup

In Claude, request consent to access your messages:

```
Can you analyze my iMessage conversations?
```

Claude will use the `request_consent` tool to get your permission.

### Example Queries

**Relationship Analysis:**
```
Show me my communication patterns with my top contacts
```

**Best Contact Times:**
```
When's the best time to reach John?
```

**Topic Trends:**
```
What have I been talking about most in the last month?
```

**Anomaly Detection:**
```
Have there been any unusual changes in my communication patterns?
```

## Privacy & Security

### Data Protection

- **100% Local**: No data leaves your device
- **Read-Only**: Cannot modify your iMessage database
- **Contact Hashing**: IDs are hashed with per-session salts
- **PII Redaction**: Automatic removal of sensitive information
- **Preview Caps**: Limited to 20 messages, 160 chars each

### Consent Model

- Explicit consent required for all operations
- Time-limited access (default 24 hours)
- Revocable at any time
- Audit trail of all tool usage

See [PRIVACY_SECURITY.md](PRIVACY_SECURITY.md) for detailed information.

## Available Tools

### System Tools
- `imsg.health_check` - Validate database access and optimization
- `imsg.summary_overview` - Global statistics and overview

### Analytics Tools
- `imsg.relationship_intelligence` - Per-contact communication profiles
- `imsg.conversation_topics` - Topic extraction and trends
- `imsg.sentiment_evolution` - Sentiment tracking over time
- `imsg.response_time_distribution` - Response latency analysis
- `imsg.cadence_calendar` - Message timing heatmaps

### Prediction Tools
- `imsg.best_contact_time` - Optimal contact time recommendations
- `imsg.anomaly_scan` - Detect unusual patterns
- `imsg.network_intelligence` - Social graph analysis

### Message Tools
- `imsg.sample_messages` - Redacted message previews

See [MCP_TOOLS_REFERENCE.md](MCP_TOOLS_REFERENCE.md) for complete documentation.

## Configuration

### Environment Variables

```bash
# Privacy settings
IMSG_REDACT_DEFAULT=true
IMSG_HASH_IDENTIFIERS=true

# Consent settings
IMSG_CONSENT_WINDOW_HOURS=24

# Database settings
IMSG_DATABASE_PATH=~/Library/Messages/chat.db

# Performance settings
IMSG_MEMORY_LIMIT_MB=250
```

### Configuration File

Create `config.json`:

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
  "performance": {
    "memory_limit_mb": 250,
    "query_timeout_s": 30
  }
}
```

## Troubleshooting

### Common Issues

**"Database not found" error:**
- Ensure Full Disk Access is granted to Terminal/Python
- Check database path: `~/Library/Messages/chat.db`

**"Read-only mode failed" error:**
- Database might be locked by Messages app
- Try closing Messages app

**Performance issues:**
- Large databases (>10GB) may be slow
- Consider using the sharding feature
- Run `imsg.health_check` for optimization tips

### Debug Mode

Run with debug logging:
```bash
IMSG_LOG_LEVEL=DEBUG python -m mcp_server.main
```

## Development

### Running Tests

```bash
# Unit tests
pytest tests/

# Integration tests
pytest tests/integration/

# Performance benchmarks
python scripts/benchmark_mcp.py
```

### Code Style

```bash
# Format code
black mcp_server/

# Lint
ruff check mcp_server/

# Type checking
mypy mcp_server/
```

## Migration from FastMCP

If you're migrating from the FastMCP version, see [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md).

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - See LICENSE file

## Security

To report security issues, please email security@[yourdomain].com

Do NOT create public GitHub issues for security vulnerabilities.

## Acknowledgments

- Built for Claude Desktop using Anthropic's MCP protocol
- Privacy-first design inspired by Apple's differential privacy
- Community feedback and contributions