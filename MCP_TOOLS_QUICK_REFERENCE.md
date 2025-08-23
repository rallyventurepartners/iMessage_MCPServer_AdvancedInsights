# iMessage MCP Server - Quick Reference Guide

## 🚀 Server Status: READY

The iMessage Advanced Insights MCP server provides 25 powerful tools for privacy-first message analysis, including cloud-aware capabilities for handling iCloud-stored messages!

## Available Tools (25 Total)

### 🔐 Consent Management (Required First!)
- `request_consent` - Request permission to access iMessage data
- `check_consent` - Check if consent is active
- `revoke_consent` - Revoke access permission

### 🏥 System Tools
- `imsg_health_check` - Verify database access and system status
- `imsg_summary_overview` - Global messaging statistics
- `imsg_contact_resolve` - Resolve contact identifiers

### 📊 Core Analysis
- `imsg_relationship_intelligence` - Deep relationship analysis (now with visualizations!)
- `imsg_conversation_quality` - Multi-dimensional conversation quality scoring
- `imsg_relationship_comparison` - Compare multiple relationships
- `imsg_conversation_topics` - Extract conversation themes
- `imsg_sentiment_evolution` - Track emotional patterns
- `imsg_response_time_distribution` - Response time analysis
- `imsg_cadence_calendar` - Communication heatmaps (now supports 36 months + charts!)

### 🎯 Advanced Analytics
- `imsg_best_contact_time` - Predict optimal contact times
- `imsg_anomaly_scan` - Detect unusual patterns
- `imsg_network_intelligence` - Social network analysis
- `imsg_sample_messages` - Get redacted message samples
- `imsg_group_dynamics` - Analyze group chat dynamics and health
- `imsg_predictive_engagement` - ML-powered engagement predictions

### ☁️ Cloud-Aware Tools (NEW!)
- `imsg_cloud_status` - Check cloud vs local message availability
- `imsg_smart_query` - Query with automatic cloud awareness
- `imsg_progressive_analysis` - Analysis with confidence scores

### 🤖 ML-Powered Tools (Optional)
- `imsg_semantic_search` - Natural language message search
- `imsg_emotion_timeline` - Track emotional dimensions
- `imsg_topic_clusters` - Discover topic clusters

## Quick Start Examples

### 1. Check System and Request Consent
```
Claude: "Check if iMessage tools are ready"
> Uses: imsg_health_check

Claude: "I need your consent to analyze messages"
> Uses: request_consent
```

### 2. Check Data Availability (Important!)
```
Claude: "How much of my message data is available locally?"
> Uses: imsg_cloud_status
```

### 3. Analyze a Relationship
```
Claude: "Analyze my communication with John"
> Uses: imsg_contact_resolve → imsg_relationship_intelligence
```

### 4. Find Communication Patterns
```
Claude: "When is the best time to contact Sarah?"
> Uses: imsg_best_contact_time
```

### 5. Long-Term Analysis with Charts
```
Claude: "Show me 2 years of messaging patterns with John"
> Uses: imsg_relationship_intelligence with include_visualizations=True
```

### 6. Compare Multiple Contacts
```
Claude: "Compare my top 5 contacts over the past year"
> Uses: imsg_cadence_calendar with comparison_contacts parameter
```

## ⚡ Performance Tips

1. **Always check cloud status first** - Most messages may be in iCloud
2. **Use progressive analysis** for partial data with confidence scores
3. **Cache contact IDs** to avoid repeated lookups
4. **Use redaction by default** for privacy

## 🔒 Privacy Features

- All contacts are SHA-256 hashed
- Messages are heavily redacted
- No network access
- Consent expires after 24 hours
- All processing is local

## 🚨 Common Issues

**"99% of messages in iCloud"**
- Use cloud-aware tools for adaptive analysis
- Consider downloading data with `brctl download ~/Library/Messages/`

**"No consent active"**
- Run `request_consent` first

**"Contact not found"**
- Try different formats: name, email, phone number

## 📚 Full Documentation

- [Complete Tool Reference](docs/MCP_TOOLS_REFERENCE.md)
- [Cloud-Aware Tools Guide](docs/CLOUD_AWARE_TOOLS.md)
- [Large Database Guide](docs/LARGE_DATABASE_GUIDE.md)
- [Privacy & Security](PRIVACY_SECURITY.md)