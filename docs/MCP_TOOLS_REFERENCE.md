# MCP Tools Reference

## Overview

This document provides a comprehensive reference for all MCP tools available in the iMessage Advanced Insights server. Each tool includes its purpose, parameters, return schema, and example usage.

## Tool Categories

1. [Consent Management](#consent-management) - Privacy consent controls
2. [System Tools](#system-tools) - Health checks and configuration
3. [Analytics Tools](#analytics-tools) - Relationship and conversation intelligence  
4. [Message Tools](#message-tools) - Message retrieval and search
5. [Network Tools](#network-tools) - Social graph analysis
6. [Prediction Tools](#prediction-tools) - Forecasting and recommendations
7. [ML-Powered Tools](#ml-powered-tools) - Optional machine learning features

## Consent Management

### request_consent

Request user consent to access iMessage data.

**Parameters:**
```typescript
{
  expiry_hours: number  // Hours until consent expires (default: 24)
}
```

**Returns:**
```typescript
{
  consent_granted: boolean
  expires_at: string    // ISO timestamp
}
```

**Example:**
```json
// Request
{
  "tool": "request_consent",
  "args": {
    "expiry_hours": 48
  }
}

// Response
{
  "consent_granted": true,
  "expires_at": "2024-12-22T10:00:00Z"
}
```

### check_consent

Check current consent status.

**Parameters:**
```typescript
{}  // No parameters required
```

**Returns:**
```typescript
{
  has_consent: boolean
  expires_at?: string      // ISO timestamp if consent active
  remaining_hours?: number // Hours remaining
}
```

### revoke_consent

Revoke user consent to access iMessage data.

**Parameters:**
```typescript
{}  // No parameters required
```

**Returns:**
```typescript
{
  consent_revoked: boolean
}
```

## System Tools

### imsg.health_check

Validates database access, schema compatibility, and system readiness.

**Parameters:**
```typescript
{
  db_path: string  // Path to iMessage database
}
```

**Returns:**
```typescript
{
  db_version: string         // SQLite version
  tables: string[]           // Available tables
  indices_ok: boolean        // Index optimization status
  read_only_ok: boolean      // Read-only enforcement
  warnings: string[]         // Actionable warnings
}
```

**Example:**
```json
// Request
{
  "tool": "imsg.health_check",
  "args": {
    "db_path": "~/Library/Messages/chat.db"
  }
}

// Response
{
  "db_version": "3.39.5",
  "tables": ["message", "handle", "chat", "attachment"],
  "indices_ok": true,
  "read_only_ok": true,
  "warnings": ["Consider running VACUUM for 15% space reduction"]
}
```

### imsg.summary_overview

Provides global statistics about the message database.

**Parameters:**
```typescript
{
  db_path: string      // Path to iMessage database
  redact: boolean      // Apply PII redaction (default: true)
}
```

**Returns:**
```typescript
{
  total_messages: number
  unique_contacts: number
  date_range: {
    start: string    // ISO date
    end: string      // ISO date
  }
  by_direction: {
    sent: number
    received: number
  }
  by_platform: {
    iMessage: number
    SMS: number
  }
  attachments: {
    images: number
    videos: number
    other: number
  }
  notes: string[]
}
```

**Example:**
```json
// Request
{
  "tool": "imsg.summary_overview",
  "args": {
    "db_path": "~/Library/Messages/chat.db",
    "redact": true
  }
}

// Response
{
  "total_messages": 125847,
  "unique_contacts": 342,
  "date_range": {
    "start": "2019-03-15",
    "end": "2024-01-15"
  },
  "by_direction": {
    "sent": 58234,
    "received": 67613
  },
  "by_platform": {
    "iMessage": 98234,
    "SMS": 27613
  },
  "attachments": {
    "images": 8234,
    "videos": 1823,
    "other": 923
  },
  "notes": ["5 group chats detected", "12% messages have attachments"]
}
```

### imsg.contact_resolve

Resolve phone/email/handle to hashed contact ID.

**Parameters:**
```typescript
{
  query: string  // Phone number, email, or name to search
}
```

**Returns:**
```typescript
{
  matches: Array<{
    contact_id: string     // Hashed identifier
    match_type: string     // "exact" | "partial"
    confidence: number     // 0.0-1.0
  }>
}
```

**Example:**
```json
// Request
{
  "tool": "imsg_contact_resolve",
  "args": {
    "query": "+1-555-123-4567"
  }
}

// Response
{
  "matches": [
    {
      "contact_id": "hash:a3f2b8c9...",
      "match_type": "exact",
      "confidence": 1.0
    }
  ]
}
```

## Analytics Tools

### imsg.relationship_intelligence

Analyzes communication patterns and relationship dynamics per contact.

**Parameters:**
```typescript
{
  db_path: string               // Path to iMessage database
  contact_filters?: string[]    // Filter to specific contacts
  window_days: number          // Analysis window (default: 365)
  redact: boolean              // Apply PII redaction (default: true)
}
```

**Returns:**
```typescript
{
  contacts: Array<{
    contact_id: string         // Hashed identifier
    display_name?: string      // Resolved name (if not redacted)
    messages_total: number
    sent_pct: number          // Percentage sent by user
    median_response_time_s: number
    avg_daily_msgs: number
    streak_days_max: number   // Longest conversation streak
    last_contacted: string    // ISO date
    engagement_score: number  // 0-1 composite engagement metric
    engagement_trend: string  // "increasing", "stable", "decreasing"
    flags: string[]           // Enhanced behavioral flags
  }>,
  insights: string[]          // Natural language insights
  recommendations: string[]   // Actionable recommendations
}
```

**Enhanced Features:**
- **Engagement Score**: Composite metric (0-1) combining balance, frequency, responsiveness, and recency
- **Behavioral Flags**: 
  - Volume: "high-volume", "active-communicator", "low-volume"
  - Balance: "conversation-initiator", "responsive-communicator", "balanced-communicator"
  - Response: "quick-responder", "thoughtful-responder"
  - Engagement: "highly-engaged", "low-engagement"
  - Recency: "recently-active", "reconnect-suggested"
- **Insights**: Auto-generated observations about relationship patterns
- **Recommendations**: Actionable suggestions for improving relationships

**Example:**
```json
// Request
{
  "tool": "imsg.relationship_intelligence",
  "args": {
    "db_path": "~/Library/Messages/chat.db",
    "contact_filters": null,
    "window_days": 180,
    "redact": true
  }
}

// Response
{
  "contacts": [
    {
      "contact_id": "hash:a3f2b8c9",
      "display_name": null,
      "messages_total": 3421,
      "sent_pct": 48.5,
      "median_response_time_s": 245,
      "avg_daily_msgs": 18.9,
      "streak_days_max": 47,
      "last_contacted": "2024-01-15",
      "engagement_score": 0.87,
      "engagement_trend": "stable",
      "flags": ["balanced-communicator", "high-volume", "quick-responder", "highly-engaged"]
    }
  ],
  "insights": [
    "Contact a3f2b8c9 is one of your most engaged relationships",
    "You maintain balanced conversations with a3f2b8c9"
  ],
  "recommendations": [
    "Continue your current communication pattern - it's working well"
  ]
}
```

### imsg.conversation_topics

Extracts topics and keywords from conversations.

**Parameters:**
```typescript
{
  db_path: string            // Path to iMessage database
  contact_id?: string        // Specific contact (null for all)
  since_days: number         // Lookback period
  top_k: number             // Number of topics (default: 25)
  use_transformer: boolean   // Use ML models (default: false)
}
```

**Returns:**
```typescript
{
  terms: Array<{
    term: string
    count: number
  }>
  trend: Array<{
    term: string
    spark: string    // Unicode sparkline: "▁▃▅▇▅▃▁"
  }>
  notes: string[]
}
```

### imsg.sentiment_evolution

Tracks sentiment changes over time.

**Parameters:**
```typescript
{
  db_path: string         // Path to iMessage database
  contact_id?: string     // Specific contact (null for all)
  window_days: number     // Rolling window size
}
```

**Returns:**
```typescript
{
  series: Array<{
    ts: string          // ISO timestamp
    score: number       // -1.0 to 1.0
  }>
  summary: {
    mean: number
    delta_30d: number   // Change over 30 days
    volatility_index: number  // 0-1 emotional stability measure
    emotional_stability: string  // "stable", "variable", "volatile"
  }
  peak_sentiment_times: {
    most_positive_hour: number   // 0-23
    most_negative_hour: number   // 0-23
    pattern: string             // "morning_person", "evening_person", "neutral"
  }
  insights: string[]            // Natural language insights
  recommendations: string[]     // Actionable recommendations
}
```

**Enhanced Features:**
- **Volatility Index**: Measures emotional consistency (0=stable, 1=volatile)
- **Peak Sentiment Times**: Identifies when conversations are most positive/negative
- **Emotional Stability**: Categorizes overall emotional patterns
- **Insights**: Auto-generated observations about sentiment trends
- **Recommendations**: Suggestions for optimal communication timing

### imsg.response_time_distribution

Analyze response time patterns.

**Parameters:**
```typescript
{
  contact_id?: string    // Optional: filter by contact
  days: number          // Lookback period (default: 90)
  db_path: string       // Path to database
}
```

**Returns:**
```typescript
{
  contact_id?: string
  your_response_times: {
    median_minutes: number
    p25_minutes: number     // 25th percentile
    p75_minutes: number     // 75th percentile
    p95_minutes: number     // 95th percentile
  }
  their_response_times: {
    median_minutes: number
    p25_minutes: number
    p75_minutes: number
    p95_minutes: number
  }
  analysis_period_days: number
  total_conversations_analyzed: number
}
```

**Example:**
```json
// Response
{
  "contact_id": "hash:b4f9a2c1...",
  "your_response_times": {
    "median_minutes": 4.5,
    "p25_minutes": 2.1,
    "p75_minutes": 12.3,
    "p95_minutes": 45.2
  },
  "their_response_times": {
    "median_minutes": 8.2,
    "p25_minutes": 3.5,
    "p75_minutes": 22.1,
    "p95_minutes": 120.5
  },
  "analysis_period_days": 90,
  "total_conversations_analyzed": 342
}
```

### imsg.cadence_calendar

Generate communication frequency heatmap data.

**Parameters:**
```typescript
{
  contact_id?: string    // Optional: filter by contact
  days: number          // Lookback period (default: 90)
  db_path: string       // Path to database
}
```

**Returns:**
```typescript
{
  contact_id?: string
  heatmap: {
    [day: string]: {      // "Monday", "Tuesday", etc.
      [hour: string]: number  // Message count for that hour
    }
  }
  peak_hours: string[]    // Top 3 active hours
  peak_days: string[]     // Top 3 active days
  total_messages: number
}
```

**Example:**
```json
// Response
{
  "contact_id": "hash:c5d8e9f2...",
  "heatmap": {
    "Monday": {
      "9": 5,
      "10": 8,
      "14": 12,
      "19": 15
    },
    "Tuesday": {
      "9": 3,
      "11": 6,
      "15": 10,
      "20": 18
    }
  },
  "peak_hours": ["19:00", "20:00", "15:00"],
  "peak_days": ["Wednesday", "Thursday"],
  "total_messages": 523
}
```

## Message Tools

### imsg.sample_messages

Returns redacted message previews for validation.

**Parameters:**
```typescript
{
  db_path: string         // Path to iMessage database
  contact_id?: string     // Filter by contact
  limit: number          // Max messages (capped at 20)
}
```

**Returns:**
```typescript
Array<{
  ts: string             // ISO timestamp
  direction: string      // "sent" | "received"
  contact_id: string     // Hashed identifier
  preview: string        // Truncated & redacted text
}>
```

## Network Tools

### imsg.network_intelligence

Analyzes social connections from group chats.

**Parameters:**
```typescript
{
  db_path: string       // Path to iMessage database
  since_days: number    // Lookback period
}
```

**Returns:**
```typescript
{
  nodes: Array<{
    id: string          // Hashed identifier
    label?: string      // Display name if available
    degree: number      // Connection count
  }>
  edges: Array<{
    source: string
    target: string
    weight: number      // Interaction count
  }>
  communities: Array<{
    community_id: number
    members: string[]
  }>
  key_connectors: Array<{
    contact_id: string
    score: number       // Betweenness centrality
  }>
  network_health: {
    overall_score: number        // 0-1 health metric
    diversity_score: number      // Community diversity
    connectivity_score: number   // Network connectivity
    redundancy_score: number     // Path redundancy
    risk_level: string          // "low", "medium", "high"
    metrics: {
      total_nodes: number
      total_edges: number
      avg_connections: number
    }
  }
  insights: string[]            // Natural language insights
  recommendations: string[]     // Actionable recommendations
}
```

**Enhanced Features:**
- **Network Health Score**: Composite metric assessing network resilience
- **Diversity Score**: Measures variety of social connections
- **Connectivity Score**: Average connections per person
- **Redundancy Score**: Multiple paths between contacts
- **Risk Assessment**: Identifies isolation risks
- **Insights**: Observations about social network structure
- **Recommendations**: Suggestions for strengthening connections

## Prediction Tools

### imsg.best_contact_time

Recommends optimal times to contact someone.

**Parameters:**
```typescript
{
  db_path: string         // Path to iMessage database
  contact_id?: string     // Specific contact
}
```

**Returns:**
```typescript
{
  windows: Array<{
    weekday: string      // "Monday", "Tuesday", etc.
    hour: number         // 0-23
    score: number        // 0.0-1.0 confidence
  }>
}
```

**Example:**
```json
// Request
{
  "tool": "imsg.best_contact_time",
  "args": {
    "db_path": "~/Library/Messages/chat.db",
    "contact_id": "hash:a3f2b8c9"
  }
}

// Response
{
  "windows": [
    {
      "weekday": "Tuesday",
      "hour": 19,
      "score": 0.89
    },
    {
      "weekday": "Thursday",
      "hour": 20,
      "score": 0.85
    },
    {
      "weekday": "Saturday",
      "hour": 11,
      "score": 0.78
    }
  ]
}
```

### imsg.anomaly_scan

Detects unusual communication patterns.

**Parameters:**
```typescript
{
  db_path: string         // Path to iMessage database
  contact_id?: string     // Specific contact
  lookback_days: number   // Analysis period
}
```

**Returns:**
```typescript
{
  anomalies: Array<{
    ts: string           // ISO timestamp
    type: string         // "silence", "burst", "pattern_change"
    severity: number     // 0.0-1.0
    note: string        // Human-readable description
  }>
}
```

## Common Patterns

### Consent Required

All tools except consent management require active consent:

```python
# First request consent
claude: Let me request access to analyze your messages
> Tool: request_consent(expiry_hours=24)

# Then use analytics tools
> Tool: relationship_intelligence(...)
```

### Pagination

Tools returning large datasets support pagination:

```typescript
{
  page: number         // Starting from 1
  page_size: number    // Items per page (max varies by tool)
}
```

### Date Filtering

Most tools accept flexible date formats:
- ISO dates: `"2024-01-15"`
- Relative: `"7 days ago"`, `"3 months ago"`
- Ranges: `start_date` and `end_date` parameters

### Contact Resolution

Tools accept multiple contact formats:
- Phone: `"+1-555-123-4567"`
- Email: `"user@example.com"`
- Hashed ID: `"hash:a3f2b8c9"`

## Error Handling

All tools return consistent error responses:

```typescript
{
  error: string          // Error message
  error_type: string     // "consent", "database", "validation", etc.
  details?: any          // Additional context
}
```

Common errors:
- `ConsentRequired`: No active consent
- `DatabaseNotFound`: Can't access iMessage database  
- `InvalidParameters`: Validation failed
- `RateLimited`: Too many requests

## Performance Considerations

- Most tools complete in < 500ms for typical databases
- Large databases (>100k messages) may take 1-2 seconds
- Network analysis is most computationally intensive
- All tools respect memory limits (default 250MB)

## Example Claude Interactions

### Relationship Analysis
```
Human: Can you analyze my communication patterns with my close contacts?

Claude: I'll analyze your communication patterns with your close contacts. First, I need your consent to access the iMessage database.

> Tool: request_consent(expiry_hours=24)
< Consent granted for 24 hours

Now analyzing your relationships:

> Tool: relationship_intelligence(window_days=90, redact=false)
< Returns detailed contact patterns

Based on the analysis, here are your communication patterns...
```

### Best Time to Reach Someone
```
Human: When's the best time to text John?

Claude: I'll analyze your conversation history with John to find the best times they typically respond.

> Tool: contact_resolve(query="John")
< Found contact: hash:b4f9a2c1

> Tool: best_contact_time(contact_id="hash:b4f9a2c1")
< Returns optimal contact windows

Based on your conversation history, John responds most reliably:
- Weekday evenings around 7-8 PM (89% response rate)
- Saturday mornings around 10-11 AM (78% response rate)
```

## Testing Tools

For development and testing:

```bash
# Test with synthetic data
python -m pytest tests/test_mcp_tools.py

# Benchmark performance
python scripts/benchmark_tools.py --tool relationship_intelligence

# Validate schemas
python scripts/validate_tool_schemas.py
```

## Cloud-Aware Tools

### imsg.cloud_status

**Purpose:** Check cloud vs local message availability to understand data completeness.

**Parameters:**
- `db_path` (string, optional): Path to iMessage database
- `check_specific_dates` (array, optional): List of specific dates to check availability

**Returns:**
- `summary`: Overall cloud vs local statistics
- `availability_gaps`: Time periods with limited local data
- `date_availability`: Availability for specific requested dates
- `recommendations`: Actions to improve data access
- `download_status`: Current iCloud download status

**Example:**
```json
// Request
{
  "tool": "imsg.cloud_status",
  "args": {
    "check_specific_dates": ["2024-01-15", "2024-06-20"]
  }
}

// Response
{
  "summary": {
    "total_messages": 323157,
    "local_messages": 212,
    "cloud_messages": 322945,
    "local_percentage": 0.1,
    "cloud_percentage": 99.9
  },
  "availability_gaps": [
    {
      "period": "2023-12",
      "total_messages": 2834,
      "cloud_percentage": 98.5
    }
  ],
  "recommendations": [
    {
      "priority": "high",
      "action": "download_from_cloud",
      "command": "brctl download ~/Library/Messages/"
    }
  ]
}
```

### imsg.smart_query

**Purpose:** Query messages with automatic cloud awareness and optional download triggering.

**Parameters:**
- `query_type` (string, required): Type of query - 'messages', 'stats', or 'patterns'
- `contact_id` (string, optional): Hashed contact identifier
- `date_range` (object, optional): Date range with 'start' and 'end' ISO dates
- `auto_download` (boolean, optional): Attempt to download missing data
- `db_path` (string, optional): Path to iMessage database

**Returns:**
- Query results based on type
- `_metadata`: Information about data availability and completeness

**Example:**
```json
// Request
{
  "tool": "imsg.smart_query",
  "args": {
    "query_type": "stats",
    "contact_id": "hash:a3f2b8c9",
    "date_range": {
      "start": "2024-01-01",
      "end": "2024-01-31"
    },
    "auto_download": false
  }
}

// Response
{
  "total_messages": 1523,
  "sent": 745,
  "received": 778,
  "_metadata": {
    "total_matching_messages": 1523,
    "locally_available": 45,
    "availability_percentage": 3.0,
    "data_completeness": "limited"
  }
}
```

### imsg.progressive_analysis

**Purpose:** Perform analysis that adapts to available data with confidence scoring.

**Parameters:**
- `analysis_type` (string, required): Type - 'sentiment', 'topics', or 'patterns'
- `contact_id` (string, optional): Hashed contact identifier
- `options` (object, optional): Analysis options including window ('recent', 'quarterly', 'annual', 'historical')
- `db_path` (string, optional): Path to iMessage database

**Returns:**
- `analysis_type`: Type of analysis performed
- `time_window`: Period analyzed
- `overall_confidence`: Confidence score based on data availability
- `results`: Analysis results by time chunk with individual confidence scores
- `data_quality`: Overall assessment of data quality
- `recommendations`: Suggestions for improving analysis accuracy

**Example:**
```json
// Request
{
  "tool": "imsg.progressive_analysis",
  "args": {
    "analysis_type": "sentiment",
    "contact_id": "hash:a3f2b8c9",
    "options": {
      "window": "quarterly"
    }
  }
}

// Response
{
  "analysis_type": "sentiment",
  "time_window": "90 days",
  "overall_confidence": 0.45,
  "results": [
    {
      "period": "0-30 days ago",
      "data": {
        "average_sentiment": 0.72,
        "trend": "stable"
      },
      "confidence": 0.85,
      "messages_analyzed": 145,
      "messages_total": 170
    }
  ],
  "data_quality": "limited",
  "recommendations": [
    {
      "priority": "high",
      "action": "Download more data from iCloud before analysis"
    }
  ]
}
```

## ML-Powered Tools

Optional tools that require machine learning dependencies (`pip install -e ".[ml]"`).

### imsg.semantic_search

Search messages using semantic similarity (requires ML dependencies).

**Parameters:**
```typescript
{
  query: string          // Natural language search query
  contact_id?: string    // Optional: filter by contact
  k: number             // Number of results (default: 10)
  db_path: string       // Path to database
}
```

**Returns:**
```typescript
{
  results: Array<{
    message_id: string
    contact_id: string
    timestamp: string
    preview: string      // Redacted message preview
    similarity_score: number  // 0.0-1.0
    is_from_me: boolean
  }>
  query_embedding_cached: boolean
}
```

**Example:**
```json
// Request
{
  "tool": "imsg_semantic_search",
  "args": {
    "query": "discussions about vacation planning",
    "k": 5
  }
}

// Response
{
  "results": [
    {
      "message_id": "msg_123",
      "contact_id": "hash:d7f8a9b2...",
      "timestamp": "2024-12-10T14:30:00Z",
      "preview": "Let's plan our [LOCATION] trip for [DATE]",
      "similarity_score": 0.92,
      "is_from_me": true
    }
  ],
  "query_embedding_cached": true
}
```

### imsg.emotion_timeline

Track emotional dimensions over time (requires ML dependencies).

**Parameters:**
```typescript
{
  contact_id?: string    // Optional: filter by contact
  days: number          // Lookback period (default: 90)
  db_path: string       // Path to database
}
```

**Returns:**
```typescript
{
  timeline: Array<{
    date: string         // ISO date
    emotions: {
      joy: number        // 0.0-1.0
      trust: number
      fear: number
      surprise: number
      sadness: number
      disgust: number
      anger: number
      anticipation: number
    }
    dominant_emotion: string
    emotional_valence: number  // -1.0 to 1.0
    message_count: number
  }>
  summary: {
    overall_valence: number
    emotional_stability: number  // 0.0-1.0
    dominant_emotions: string[]
  }
}
```

### imsg.topic_clusters

Discover topic clusters using ML (requires ML dependencies).

**Parameters:**
```typescript
{
  contact_id?: string    // Optional: filter by contact
  k: number             // Number of clusters (default: 10)
  db_path: string       // Path to database
}
```

**Returns:**
```typescript
{
  clusters: Array<{
    cluster_id: number
    size: number         // Number of messages
    keywords: string[]   // Top keywords (redacted if needed)
    summary: string      // AI-generated summary
    sample_messages: Array<{
      preview: string    // Redacted preview
      timestamp: string
    }>
  }>
  unclustered_ratio: number  // Percentage not fitting clusters
}
```