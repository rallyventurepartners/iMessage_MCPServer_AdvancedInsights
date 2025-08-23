# Migration Guide: FastMCP to Anthropic MCP

## Overview

This guide documents the migration from FastMCP to Anthropic's official MCP SDK for Claude Desktop compatibility.

## Key Changes

### 1. Protocol Implementation

**Before (FastMCP):**
- HTTP-based server
- Custom tool registration
- Port-based communication

**After (Anthropic MCP):**
- stdio-based JSON-RPC
- Standard MCP tool registration
- Direct Claude Desktop integration

### 2. Directory Structure

```
OLD:
src/
├── mcp_tools/      # Mixed tools and utilities
├── server/         # FastMCP server
└── utils/          # Various helpers

NEW:
mcp_server/
├── main.py         # stdio entry point
├── config.py       # Simplified config
├── db.py          # Read-only database
├── models.py      # Pydantic schemas
├── privacy.py     # Hashing & redaction
├── tools/         # One file per tool group
│   ├── health.py
│   ├── overview.py
│   ├── analytics.py
│   ├── messages.py
│   ├── network.py
│   └── predictions.py
└── analytics/     # Deterministic algorithms
```

### 3. Tool Registration

**Before:**
```python
@register_tool(name="get_messages", description="...")
async def get_messages_tool(...):
    ...
```

**After:**
```python
@server.tool()
async def imsg_get_messages(arguments: Dict[str, Any]) -> Dict[str, Any]:
    # Validate with Pydantic
    params = GetMessagesInput(**arguments)
    # Process...
    output = GetMessagesOutput(...)
    return output.model_dump()
```

### 4. Privacy Enhancements

**New Features:**
- BLAKE2b hashing with per-session salts
- Default redaction ON
- Preview caps (20 messages, 160 chars)
- Structured audit logging

### 5. Configuration

**Before:** Multiple config files, environment variables
**After:** Single config with clear defaults

```json
{
  "privacy": {
    "redact_by_default": true,
    "hash_identifiers": true
  },
  "consent": {
    "default_duration_hours": 24
  }
}
```

## Migration Steps

### Phase 1: Parallel Installation

1. Install new dependencies:
   ```bash
   pip install -r requirements_mcp.txt
   ```

2. Keep existing code intact
3. Build new implementation in `mcp_server/` directory

### Phase 2: Tool Migration

Migrate tools in order of priority:

1. **System Tools** (Week 1)
   - [x] health_check
   - [x] summary_overview
   - [x] contact_resolve

2. **Analytics Tools** (Week 2)
   - [ ] relationship_intelligence
   - [ ] conversation_topics
   - [ ] sentiment_evolution
   - [ ] response_time_distribution
   - [ ] cadence_calendar

3. **Other Tools** (Week 3)
   - [ ] anomaly_scan
   - [ ] best_contact_time
   - [ ] network_intelligence
   - [ ] sample_messages

### Phase 3: Testing & Validation

1. **Unit Tests**
   ```bash
   pytest tests/test_mcp_tools.py
   ```

2. **Integration Tests**
   ```bash
   python -m mcp_server.main --test-mode
   ```

3. **Performance Validation**
   ```bash
   python scripts/benchmark_mcp.py
   ```

### Phase 4: Claude Desktop Integration

1. Update Claude Desktop config:
   ```json
   {
     "mcpServers": {
       "imessage": {
         "command": "python",
         "args": ["-m", "mcp_server.main"],
         "cwd": "/path/to/imessage-mcp"
       }
     }
   }
   ```

2. Test with Claude Desktop
3. Monitor logs for errors

## Tool Mapping

| Old Tool Name | New Tool Name | Status |
|--------------|---------------|---------|
| get_messages | imsg.sample_messages | Needs refactor |
| get_contacts | imsg.relationship_intelligence | Needs simplification |
| analyze_conversation | imsg.conversation_topics | Partial |
| get_sentiment | imsg.sentiment_evolution | Needs deterministic mode |
| - | imsg.health_check | ✅ New |
| - | imsg.summary_overview | ✅ New |
| - | imsg.response_time_distribution | ✅ Implemented |
| - | imsg.cadence_calendar | ✅ Implemented |

## Breaking Changes

### 1. Tool Names
All tools now prefixed with `imsg.` for namespace clarity

### 2. Contact IDs
Now hashed by default: `+1-555-1234` → `hash:a3f2b8c9`

### 3. Response Format
Strict Pydantic validation on all inputs/outputs

### 4. Consent Required
All tools (except consent management) require active consent

## Rollback Plan

If issues arise:

1. Keep both implementations side-by-side
2. Use environment variable to switch:
   ```bash
   USE_LEGACY_MCP=true python server.py
   ```
3. Gradual tool migration allows partial rollback

## Performance Considerations

- New implementation targets p95 < 1.5s
- Memory limit enforced at 250MB
- Streaming disabled for security
- Connection pooling for large databases

## Security Improvements

1. **Read-Only Enforcement**
   ```python
   connection = sqlite3.connect(
       f"file:{db_path}?mode=ro",
       uri=True
   )
   ```

2. **Query Timeouts**
   - 30 second maximum per query
   - Automatic cancellation

3. **Row Limits**
   - Automatic LIMIT clauses
   - Preview caps enforced

## Support

- GitHub Issues: For bugs and features
- Documentation: See MCP_TOOLS_REFERENCE.md
- Testing: Run test suite before deployment