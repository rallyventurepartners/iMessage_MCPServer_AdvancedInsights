# Cloud-Aware Tools for iMessage MCP Server

## Overview

When your iMessage database has messages stored in iCloud, standard queries return limited results. These cloud-aware tools help you work with partial data intelligently.

## New Tools

### 1. `imsg_cloud_status`
Check what data is available locally vs in iCloud.

```python
# Check overall cloud storage status
result = await imsg_cloud_status()

# Check specific dates
result = await imsg_cloud_status(
    check_specific_dates=["2024-01-15", "2024-06-20"]
)
```

**Returns:**
```json
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

### 2. `imsg_smart_query`
Query with automatic cloud awareness and optional download triggering.

```python
# Query with auto-download if needed
result = await imsg_smart_query(
    query_type="messages",
    contact_id="+1234567890",
    date_range={"start": "2024-01-01", "end": "2024-01-31"},
    auto_download=True  # Attempts to download missing data
)
```

**Query Types:**
- `messages` - Retrieve messages with metadata
- `stats` - Get statistics about conversations
- `patterns` - Analyze communication patterns

**Returns:**
```json
{
  "results": [...],
  "_metadata": {
    "total_matching_messages": 1523,
    "locally_available": 45,
    "availability_percentage": 3.0,
    "data_completeness": "limited"
  }
}
```

### 3. `imsg_progressive_analysis`
Perform analysis that adapts to available data with confidence scoring.

```python
# Analyze what's available with confidence scores
result = await imsg_progressive_analysis(
    analysis_type="sentiment",
    contact_id="+1234567890",
    options={"window": "quarterly"}  # recent, quarterly, annual, historical
)
```

**Analysis Types:**
- `sentiment` - Emotional tone analysis
- `topics` - Conversation topics
- `patterns` - Communication patterns

**Returns:**
```json
{
  "analysis_type": "sentiment",
  "time_window": "90 days",
  "overall_confidence": 0.45,
  "results": [
    {
      "period": "0-30 days ago",
      "data": {"average_sentiment": 0.72},
      "confidence": 0.85,
      "messages_analyzed": 145,
      "messages_total": 170
    },
    {
      "period": "30-60 days ago",
      "data": {"average_sentiment": 0.65},
      "confidence": 0.15,
      "messages_analyzed": 23,
      "messages_total": 156
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

## Usage Patterns

### Pattern 1: Check Before Query
```python
# Always check availability first
status = await imsg_cloud_status()
if status["summary"]["local_percentage"] < 50:
    print("Warning: Most data is in iCloud")
    # Proceed with limited data or trigger download
```

### Pattern 2: Progressive Analysis
```python
# Start with recent data (usually local)
recent = await imsg_progressive_analysis(
    analysis_type="patterns",
    options={"window": "recent"}
)

# If confidence is good, expand the window
if recent["overall_confidence"] > 0.8:
    quarterly = await imsg_progressive_analysis(
        analysis_type="patterns",
        options={"window": "quarterly"}
    )
```

### Pattern 3: Smart Querying
```python
# Let the tool handle cloud awareness
result = await imsg_smart_query(
    query_type="stats",
    date_range={"start": "2023-01-01", "end": "2023-12-31"},
    auto_download=False  # Just work with what's available
)

print(f"Analysis based on {result['_metadata']['availability_percentage']}% of data")
```

## Implementation in Claude

When using these tools through Claude, you can:

1. **Check Status First**
   ```
   "What percentage of my messages are stored locally vs in iCloud?"
   ```

2. **Request Progressive Analysis**
   ```
   "Analyze my conversations with John, working with whatever data is available locally"
   ```

3. **Get Confidence Scores**
   ```
   "Show me sentiment analysis for 2024 with confidence scores for each month"
   ```

## Technical Details

### How It Works

1. **Availability Checking**: Each query first checks if message `text` is NULL (indicating cloud storage)
2. **Confidence Scoring**: Results include confidence based on data completeness
3. **Progressive Loading**: Analyzes in chunks, starting with most likely available data
4. **Smart Recommendations**: Suggests when to download more data

### Triggering Downloads

The tools can attempt to trigger downloads using:
- `brctl download` command (requires local permissions)
- Opening Messages app to specific conversations
- Disabling/re-enabling Messages in iCloud

### Limitations

- Cannot directly access iCloud servers
- Download triggering is best-effort only
- Full downloads can take hours for large databases
- Some older messages may never sync back

## Best Practices

1. **Always Check First**: Use `imsg_cloud_status` before deep analysis
2. **Start Recent**: Recent data is more likely to be local
3. **Accept Partial Results**: Work with confidence scores
4. **Plan Downloads**: Schedule full downloads during off-hours
5. **Cache Results**: Save analysis results to avoid re-querying

## Error Handling

All cloud-aware tools handle common scenarios:
- Database locked by Messages app
- Download in progress
- Insufficient local data
- Network connectivity issues

Each tool returns appropriate error messages and recommendations for resolution.