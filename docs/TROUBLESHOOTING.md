# Troubleshooting Guide

This guide helps resolve common issues with the iMessage Advanced Insights MCP Server.

## Common Issues

### 1. Claude Desktop Can't Find the Server

**Symptoms**: 
- "Failed to connect to MCP server" error
- Server doesn't appear in Claude's tool list

**Solutions**:
1. Verify server is installed:
   ```bash
   pip show imessage-advanced-insights
   ```

2. Check Claude Desktop configuration:
   ```bash
   cat ~/Library/Application\ Support/Claude/claude_desktop_config.json
   ```

3. Ensure the command path is correct:
   ```json
   {
     "mcpServers": {
       "imessage-insights": {
         "command": "python",
         "args": ["-m", "mcp_server.main"]
       }
     }
   }
   ```

4. Restart Claude Desktop after configuration changes

### 2. Database Access Errors

**Symptoms**:
- "Database locked" errors
- "Permission denied" accessing chat.db
- "Database not found" messages

**Solutions**:
1. Check database location:
   ```bash
   ls -la ~/Library/Messages/chat.db
   ```

2. Verify permissions:
   ```bash
   # Should show your username as owner
   stat ~/Library/Messages/chat.db
   ```

3. Ensure Messages app is closed during initial setup

4. Test database access:
   ```bash
   sqlite3 ~/Library/Messages/chat.db "SELECT COUNT(*) FROM message;"
   ```

### 3. Performance Issues

**Symptoms**:
- Slow tool responses
- Timeouts on large contact lists
- High memory usage

**Solutions**:
1. Check database size:
   ```bash
   du -h ~/Library/Messages/chat.db
   ```

2. For databases >5GB, create indexes:
   ```bash
   python scripts/add_performance_indexes.py
   ```

3. Use database sharding for very large databases (>20GB):
   ```bash
   python scripts/shard_large_database.py --input-db ~/Library/Messages/chat.db
   ```

4. Monitor memory usage:
   ```bash
   # In server config, set memory limits
   export MCP_MEMORY_LIMIT_GB=4
   ```

### 4. Tool Consent Issues

**Symptoms**:
- "Consent required" errors
- Tools not working despite granting consent
- Consent expiring unexpectedly

**Solutions**:
1. Check current consent status:
   ```
   Ask Claude: "Check my consent status for iMessage tools"
   ```

2. Request fresh consent:
   ```
   Ask Claude: "Request consent for all iMessage tools for 24 hours"
   ```

3. Clear consent cache if stuck:
   ```bash
   rm ~/.imessage_insights/consent.json
   ```

### 5. Privacy/Redaction Issues

**Symptoms**:
- Seeing phone numbers or emails in output
- PII not being redacted properly
- Contact IDs not hashed

**Solutions**:
1. Verify privacy mode is enabled (default):
   ```python
   # In config.json
   {
     "privacy": {
       "hash_contacts": true,
       "redact_pii": true,
       "preview_length": 50
     }
   }
   ```

2. Report any PII leaks as security issues immediately

### 6. Installation Problems

**Symptoms**:
- Import errors
- Missing dependencies
- Python version conflicts

**Solutions**:
1. Ensure Python 3.9+:
   ```bash
   python --version
   ```

2. Use virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   pip install -r requirements.txt
   ```

3. Clear pip cache if corrupted:
   ```bash
   pip cache purge
   pip install --no-cache-dir -r requirements.txt
   ```

## Debugging Steps

### Enable Debug Logging

1. Set environment variable:
   ```bash
   export MCP_DEBUG=true
   ```

2. Check logs:
   ```bash
   tail -f ~/.imessage_insights/debug.log
   ```

### Test Individual Components

1. Test database connection:
   ```python
   from mcp_server.db import get_database
   db = get_database()
   print(db.get_stats())
   ```

2. Test privacy functions:
   ```python
   from mcp_server.privacy import hash_contact_id, redact_pii
   print(hash_contact_id("+1234567890"))
   print(redact_pii("Call me at 123-456-7890"))
   ```

### Common Error Messages

| Error | Meaning | Solution |
|-------|---------|----------|
| `SQLITE_BUSY` | Database locked by another process | Close Messages app |
| `SQLITE_PERM` | Permission denied | Check file permissions |
| `ConnectionRefused` | Server not running | Restart Claude Desktop |
| `ImportError: mcp` | MCP SDK not installed | `pip install mcp` |
| `ConsentRequired` | User hasn't granted consent | Request consent through Claude |

## Getting Help

1. **Check logs**: Most issues are explained in the debug logs
2. **Search issues**: Check GitHub issues for similar problems
3. **Ask Claude**: Claude can help diagnose many issues
4. **File an issue**: Include logs, system info, and steps to reproduce

## System Requirements

- macOS 10.15+ (for iMessage database access)
- Python 3.9+
- 4GB RAM minimum (8GB recommended)
- 100MB free disk space (more for large databases)
- Claude Desktop installed and configured