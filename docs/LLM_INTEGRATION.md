# LLM Integration Guide

This document describes how the iMessage MCP Server integrates with Large Language Models (LLMs) to provide advanced conversational insights.

## Overview

The LLM integration enhances the basic NLP analysis with sophisticated insights about relationships, communication patterns, and actionable recommendations. The system is designed to work with Claude or other LLMs while maintaining privacy and efficiency through intelligent caching.

## Architecture

### Core Components

1. **LLMInsightGenerator** (`src/utils/llm_integration.py`)
   - Main class handling LLM interactions
   - Manages context preparation and prompt generation
   - Provides structured fallback insights if LLM is unavailable

2. **Analysis Cache** (`src/utils/analysis_cache.py`)
   - Caches expensive NLP operations (sentiment, topics, LLM calls)
   - Reduces redundant API calls and improves performance
   - Configurable TTLs for different analysis types

3. **Integration Points**
   - `conversation_intelligence.py`: Primary consumer of LLM insights
   - Activated when `analysis_depth="comprehensive"`
   - Supports custom focus areas for targeted analysis

## Usage

### Basic Usage

```python
from src.utils.llm_integration import generate_llm_insights

# Generate insights for a conversation
insights = await generate_llm_insights(
    messages=conversation_messages,
    analysis=traditional_analysis_results,
    focus_areas=["emotional_support", "growth"]
)
```

### With Caching

The LLM integration automatically uses caching to avoid redundant API calls:

```python
# First call - generates fresh insights
insights1 = await generate_llm_insights(messages, analysis, ["emotional_support"])

# Second call with same data - returns cached result
insights2 = await generate_llm_insights(messages, analysis, ["emotional_support"])
```

### Focus Areas

The system supports several focus areas for targeted analysis:

- **"emotional_support"**: Analyzes emotional support patterns and reciprocity
- **"conflict"**: Identifies conflict patterns and resolution strategies
- **"growth"**: Tracks relationship development and milestones

## Insight Categories

### 1. Relationship Summary
Provides a high-level overview of the relationship quality and characteristics based on:
- Conversation depth scores
- Topic diversity
- Emotional tone
- Health indicators

### 2. Communication Style
Analyzes how both parties communicate:
- Response patterns and timing
- Message balance and reciprocity
- Question-asking behavior
- Engagement levels

### 3. Areas of Strength
Identifies positive aspects of the relationship:
- Meaningful conversation ability
- Emotional support patterns
- Shared interests
- Communication consistency

### 4. Growth Opportunities
Suggests areas for improvement:
- Conversation depth enhancement
- Topic exploration
- Emotional expression
- Balance improvement

### 5. Actionable Recommendations
Provides specific, time-bound suggestions:
- Priority level (high/medium/low)
- Specific actions to take
- Rationale for each recommendation
- Suggested timeframe

## Implementation Details

### Context Preparation

The system prepares context for the LLM by:

1. **Message Sampling**: Selects representative messages (max 50) evenly distributed across time
2. **Statistical Summary**: Includes depth scores, topic diversity, emotional metrics
3. **Pattern Extraction**: Identifies key communication patterns
4. **Exchange Examples**: Provides sample conversation exchanges

### Prompt Engineering

The prompt is structured to guide the LLM towards actionable insights:

```
Analyze this conversation data and provide deep insights:

Conversation Overview:
- Total messages: [count]
- Date range: [range]
- Depth score: [score]/100
...

Key Patterns Detected:
[patterns]

Sample Exchanges:
[examples]

Please provide insights on:
1. Relationship dynamics and communication patterns
2. Emotional support and reciprocity
3. Areas of strength in the relationship
4. Opportunities for deeper connection

Specific Focus Areas: [if provided]

Provide actionable insights and specific recommendations.
```

### Fallback Mechanism

If the LLM is unavailable or fails, the system provides structured fallback insights based on the traditional analysis data. This ensures the user always receives some level of insight.

## Caching Strategy

### Cache TTLs

- **Sentiment Analysis**: 1 hour (historical messages don't change)
- **Topic Analysis**: 2 hours (topics are relatively stable)
- **LLM Insights**: 30 minutes (balance freshness with efficiency)
- **Conversation Depth**: 1 hour
- **Network Analysis**: 2 hours

### Cache Keys

Cache keys are generated using:
- Message content hashes (for consistency)
- Analysis parameters
- Focus areas
- Time windows

### Cache Invalidation

Caches can be invalidated:
- Manually via `invalidate_analysis_cache()`
- Automatically when TTL expires
- Selectively by contact or analysis type

## Performance Considerations

1. **Message Limits**: Analysis limited to 1000 messages to prevent memory issues
2. **Context Limits**: LLM context limited to 8000 tokens
3. **Parallel Processing**: Multiple analyses can run concurrently
4. **Streaming**: Large conversations processed in batches

## Privacy and Security

1. **Local Processing**: All analysis happens locally; no data leaves the system
2. **PII Sanitization**: Messages are sanitized before any external API calls
3. **Consent Required**: LLM insights only generated with user consent
4. **Cache Security**: Cached data respects the same privacy controls

## Future Enhancements

1. **Custom Models**: Support for self-hosted LLMs
2. **Streaming Insights**: Real-time insight generation for active conversations
3. **Multi-Language**: Support for non-English conversations
4. **Voice Analysis**: Integration with voice message transcription
5. **Predictive Insights**: Forecasting relationship trajectories

## Configuration

Add to your config file:

```json
{
  "llm": {
    "provider": "claude",
    "model": "claude-3",
    "max_context": 8000,
    "cache_ttl": 1800
  },
  "analysis_cache": {
    "enabled": true,
    "sentiment_ttl": 3600,
    "topic_ttl": 7200,
    "llm_ttl": 1800
  }
}
```

## Troubleshooting

### Common Issues

1. **"Advanced insights temporarily unavailable"**
   - Check LLM API configuration
   - Verify network connectivity
   - Check error logs for specific issues

2. **Slow insight generation**
   - Enable caching in configuration
   - Reduce message count for analysis
   - Check system resources

3. **Generic insights**
   - Ensure sufficient message history
   - Provide specific focus areas
   - Use "comprehensive" analysis depth

### Debug Mode

Enable debug logging for detailed troubleshooting:

```python
import logging
logging.getLogger("src.utils.llm_integration").setLevel(logging.DEBUG)
```