# Privacy & Security Documentation

## Overview

The iMessage Advanced Insights MCP Server is designed with privacy-by-default principles. All analysis is performed locally on your device, with no data transmitted to external servers. This document outlines our security measures, privacy guarantees, and data handling practices.

## Core Privacy Principles

1. **100% Local Processing**: All data analysis occurs on your device
2. **Read-Only Access**: The server never modifies your iMessage database
3. **Explicit Consent**: Access requires user consent with configurable expiration
4. **Data Minimization**: Only necessary data is accessed and processed
5. **No Persistence**: No analysis results are stored permanently

## Consent Model

### Consent Flow

```
User Request → Consent Check → Prompt for Consent → Time-Limited Access → Auto-Expiration
```

### Consent Configuration

Default consent window: **24 hours** (configurable 1-720 hours)

```json
{
  "consent": {
    "default_duration_hours": 24,
    "max_duration_hours": 720,
    "require_explicit": true
  }
}
```

### Consent Tools

- `request_consent`: Explicitly request access with duration
- `check_consent`: Verify current consent status
- `revoke_consent`: Immediately revoke all access
- `get_access_log`: Review tool usage history

## Data Protection Measures

### 1. Contact ID Hashing

All contact identifiers are hashed using BLAKE2b with per-session salts:

```python
# Per-session salt (never persisted)
session_salt = os.urandom(32)

# Contact hashing
hashed_id = blake2b(
    contact_id.encode('utf-8'),
    salt=session_salt,
    digest_size=16
).hexdigest()
```

### 2. Automatic PII Redaction

The following data types are automatically redacted:

| Data Type | Redaction Pattern | Example |
|-----------|------------------|---------|
| Credit Cards | `[CREDIT CARD REDACTED]` | 4111 1111 1111 1111 → [CREDIT CARD REDACTED] |
| SSN | `[SSN REDACTED]` | 123-45-6789 → [SSN REDACTED] |
| Phone Numbers | Partial masking | +1-555-123-4567 → +1XXXXXX67 |
| Email Addresses | Username masking | john.doe@email.com → jXe@email.com |
| Street Addresses | `[ADDRESS REDACTED]` | 123 Main St → [ADDRESS REDACTED] |
| Financial Amounts | `[AMOUNT REDACTED]` | $1,234.56 → [AMOUNT REDACTED] |

### 3. Preview Limitations

Message previews are strictly limited to prevent bulk data extraction:

```json
{
  "preview_limits": {
    "max_messages": 20,
    "max_chars_per_message": 160,
    "max_total_chars": 3200
  }
}
```

### 4. Database Access Controls

- **Read-Only Connection**: Enforced at connection level
- **Query Timeouts**: 30-second maximum query time
- **Row Limits**: Automatic LIMIT clauses on all queries
- **No Raw Dumps**: Bulk exports disabled by default

## Security Architecture

### Database Connection

```python
# Read-only connection enforcement
connection = sqlite3.connect(
    f"file:{db_path}?mode=ro",
    uri=True,
    check_same_thread=False
)
connection.execute("PRAGMA query_only = ON")
```

### Tool Access Control

Each tool access is:
1. Logged with timestamp and parameters
2. Validated against consent status
3. Rate-limited to prevent abuse
4. Monitored for anomalous patterns

### Memory Security

- Sensitive data cleared after use
- No caching of message content
- Garbage collection forced after large operations
- Memory limits enforced (default 250MB)

## Audit Trail

All tool usage is logged with:

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "tool": "imsg.relationship_intelligence",
  "args_schema_valid": true,
  "rows_accessed": 150,
  "duration_ms": 245,
  "consent_valid": true
}
```

Note: Actual message content is NEVER logged.

## Data Handling by Tool Type

### Analytics Tools
- Process aggregated data only
- Return statistical summaries
- No individual message content

### Message Tools  
- Apply preview caps
- Redact PII automatically
- Return structured insights

### Network Tools
- Analyze connection patterns
- Hash all identifiers
- Focus on graph structure

## Configuration Options

### Privacy Settings

```json
{
  "privacy": {
    "redact_by_default": true,
    "hash_identifiers": true,
    "preview_caps": {
      "enabled": true,
      "max_messages": 20,
      "max_chars": 160
    },
    "audit_logging": true
  }
}
```

### Feature Flags

```json
{
  "features": {
    "allow_exports": false,
    "enable_network_egress": false,
    "use_transformer_nlp": false
  }
}
```

## Threat Model

### Protected Against

1. **Unauthorized Access**: Consent required for all operations
2. **Data Exfiltration**: Preview caps and no bulk exports
3. **Database Modification**: Read-only connections
4. **PII Exposure**: Automatic redaction and hashing
5. **Memory Attacks**: Limits and cleanup procedures

### Out of Scope

1. **Physical Device Access**: Assumes device security
2. **macOS Permissions**: Requires user to grant access
3. **Network Monitoring**: All processing is local
4. **Malicious Extensions**: Trust boundary at MCP level

## Best Practices for Users

1. **Review Consent Requests**: Understand what access you're granting
2. **Use Short Durations**: Limit consent to needed timeframe
3. **Check Access Logs**: Review what tools have been used
4. **Revoke When Done**: Explicitly revoke access after use
5. **Update Regularly**: Keep the server updated for security fixes

## Compliance Notes

This implementation is designed to support compliance with:

- **GDPR**: Local processing, explicit consent, data minimization
- **CCPA**: No data sale, user control, transparency
- **HIPAA**: While not certified, follows security best practices

Note: This software is provided as-is. Users are responsible for compliance with applicable laws in their jurisdiction.

## Security Disclosures

To report security issues:
1. Do NOT create public GitHub issues
2. Email: security@[your-domain].com
3. Include: Description, reproduction steps, impact assessment

We aim to respond within 48 hours and provide fixes within 30 days for confirmed vulnerabilities.

## Version History

| Version | Date | Security Changes |
|---------|------|-----------------|
| 2.4.0 | 2024-01 | Added BLAKE2b hashing, preview caps |
| 2.3.0 | 2023-12 | Implemented consent system |
| 2.2.0 | 2023-11 | Added PII redaction |
| 2.1.0 | 2023-10 | Read-only enforcement |
| 2.0.0 | 2023-09 | Initial privacy-focused release |