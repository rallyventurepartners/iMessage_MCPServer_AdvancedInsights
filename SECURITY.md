# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.x.x   | :white_check_mark: |

## Reporting a Vulnerability

The iMessage Advanced Insights team takes security seriously. We appreciate your efforts to responsibly disclose your findings.

### Where to Report

Please report security vulnerabilities by emailing: [INSERT SECURITY EMAIL]

### What to Include

* Description of the vulnerability
* Steps to reproduce
* Potential impact
* Suggested fix (if any)

### What to Expect

* **Acknowledgment**: Within 48 hours
* **Initial Assessment**: Within 1 week
* **Fix Timeline**: Depending on severity
* **Credit**: Security researchers will be credited (unless anonymity requested)

## Security Considerations

### Data Privacy

This project processes sensitive personal communication data. Key protections:

1. **Local Processing Only**: All data analysis happens on the user's machine
2. **No Network Calls**: No telemetry, analytics, or external API calls
3. **Hashed Identifiers**: Contact information is SHA-256 hashed
4. **Consent Required**: Explicit user consent before any analysis
5. **Memory Safety**: Automatic cleanup of sensitive data

### Known Security Measures

* **Input Validation**: All user inputs are validated and sanitized
* **SQL Injection Prevention**: Parameterized queries only
* **Path Traversal Protection**: Strict path validation
* **PII Redaction**: Automatic removal of detected personal information
* **Secure Defaults**: Privacy-preserving settings by default

### Out of Scope

The following are explicitly out of scope:

* Physical access to user's machine
* Social engineering of users
* Attacks on Claude Desktop or the MCP protocol itself
* Issues in upstream dependencies (report to respective projects)

## Best Practices for Users

1. **Keep Software Updated**: Always use the latest version
2. **Secure Your Database**: Ensure proper file permissions on chat.db
3. **Review Consent**: Regularly review what tools have consent
4. **Audit Access**: Check Claude Desktop logs for usage
5. **Backup Safely**: If backing up chat.db, encrypt the backup

## Development Security

Contributors should:

* Never commit real message data
* Use synthetic data for all tests
* Run security linters before submitting PRs
* Follow secure coding practices
* Report any concerns immediately

## Incident Response

In case of a security incident:

1. **Isolate**: Stop the MCP server immediately
2. **Assess**: Determine what data may have been affected
3. **Report**: Contact the security team
4. **Patch**: Apply fixes when available
5. **Review**: Audit logs for any suspicious activity