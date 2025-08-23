# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

This is a sophisticated data science platform for analyzing human-to-human communication through iMessage data. The primary goal is to provide deep, actionable insights about relationships, communication patterns, and social dynamics by leveraging advanced NLP, machine learning, and Claude's language understanding capabilities.

## Key Principles

1. **Data Science First**: Every feature should provide meaningful insights, not just raw data
2. **LLM Integration**: Maximize use of Claude's capabilities for contextual understanding
3. **Privacy Paramount**: All processing is local-only with strong consent controls
4. **Actionable Insights**: Focus on recommendations and predictions, not just analysis
5. **Human-Centric**: Design for understanding human relationships and improving communication

## Commands

### Running the Server
```bash
# Start the MCP server with default settings
python server.py

# With custom configuration
python server.py --config config.json

# With database sharding enabled (for large databases 20GB+)
python server.py --use-shards --shards-dir /path/to/shards

# With memory monitoring disabled
python server.py --disable-memory-monitor
```

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_improvements.py

# With coverage
pytest --cov=src --cov-report=term-missing

# Run data science tests
pytest tests/test_conversation_intelligence.py -v
```

### Code Quality
```bash
# Format code with black
black src/ tests/ server.py

# Sort imports
isort src/ tests/ server.py

# Lint with flake8
flake8 src/ tests/ server.py

# Type checking
mypy src/ server.py

# Run pylint
pylint src/
```

### Data Science Scripts
```bash
# Index database for better performance
python scripts/index_imessage_db.py

# Shard large databases for analysis
python scripts/shard_large_database.py --input-db chat.db --output-dir shards/

# Run performance benchmarks
python scripts/performance_testing.py

# Generate insight reports
python scripts/generate_insights_report.py --format pdf --period monthly

# Train ML models for relationship classification
python scripts/train_relationship_models.py
```

## Architecture for Data Science

### Core Analytics Components

1. **Conversation Intelligence** (`src/mcp_tools/conversation_intelligence.py`):
   - Deep conversation analysis with quality metrics
   - Relationship trajectory tracking
   - Emotional dynamics analysis
   - LLM-powered insight generation

2. **Advanced NLP Pipeline** (`src/utils/advanced_nlp.py`):
   - Semantic analysis beyond keyword matching
   - Context-aware topic modeling
   - Sentiment trajectory analysis
   - Named entity recognition for life events

3. **Machine Learning Models** (`src/ml_models/`):
   - Relationship classification (friend, family, colleague, romantic)
   - Conversation quality scoring
   - Anomaly detection for behavior changes
   - Response time prediction

4. **Statistical Analysis** (`src/analytics/`):
   - Time series analysis for communication patterns
   - Correlation analysis between features
   - Hypothesis testing for significant changes
   - Clustering for contact grouping

### Data Science Tools

**Current Tools** (Basic Analytics):
- `get_messages`: Retrieve messages with basic filtering
- `get_contacts`: List contacts with activity stats
- `analyze_conversation_topics`: Basic topic extraction

**Enhanced Tools** (Deep Insights):
- `analyze_conversation_intelligence`: Comprehensive conversation analysis
- `analyze_relationship_trajectory`: Relationship evolution over time
- `predict_communication_patterns`: Forecast future interactions
- `detect_life_events`: Identify major life changes
- `profile_communication_style`: Detailed style analysis
- `generate_insights_report`: Automated insight reports

### Key Algorithms & Techniques

1. **Feature Engineering**:
   - Temporal features (hour, day, seasonality)
   - Linguistic features (vocabulary richness, complexity)
   - Behavioral features (initiation patterns, response times)
   - Network features (centrality, clustering coefficient)

2. **Analysis Techniques**:
   - Sliding window analysis for trend detection
   - Change point detection for relationship phases
   - Markov chains for conversation flow
   - Graph analysis for social network insights

3. **LLM Integration**:
   - Use Claude for semantic understanding
   - Generate natural language insights
   - Provide personalized recommendations
   - Contextual interpretation of patterns

### Performance Optimizations for Analytics

- **Incremental Processing**: Update insights without full recomputation
- **Feature Caching**: Store computed features for reuse
- **Parallel Analysis**: Concurrent processing across contacts
- **Streaming Algorithms**: Handle large datasets without loading all data
- **Approximate Algorithms**: Trade precision for speed when appropriate

### Privacy-Preserving Analytics

- **Differential Privacy**: Add noise to aggregate statistics
- **K-Anonymity**: Ensure patterns can't identify individuals
- **Secure Aggregation**: Compute insights without exposing raw data
- **Consent-Based Features**: Only compute what user approves

## Best Practices for Data Science Features

1. **Always Provide Context**: Don't just show numbers, explain what they mean
2. **Use Visualizations**: Complex patterns are easier to understand visually
3. **Progressive Disclosure**: Start with high-level insights, allow drilling down
4. **Comparative Analysis**: Show how metrics compare to baselines
5. **Actionable Recommendations**: Every insight should suggest actions

## Common Patterns

### Adding a New Insight Tool
```python
@register_tool(
    name="analyze_[specific_aspect]",
    description="Deep analysis of [aspect] using advanced data science"
)
async def analyze_[aspect]_tool(
    contact_id: str,
    time_period: str = "6 months",
    analysis_depth: str = "comprehensive"
) -> Dict[str, Any]:
    # 1. Get relevant data
    # 2. Extract features
    # 3. Apply ML/statistical models
    # 4. Generate LLM insights
    # 5. Return actionable recommendations
```

### Integrating LLM Insights
```python
def _generate_llm_insights(data: Dict, context: Dict) -> Dict[str, Any]:
    # 1. Prepare context for Claude
    # 2. Ask for specific insights
    # 3. Parse and structure response
    # 4. Validate insights
    # 5. Return formatted insights
```

## Future Enhancements Priority

1. **Real-time Dashboard**: Live updating insights
2. **Predictive Alerts**: Proactive notifications
3. **Cross-Platform Analysis**: WhatsApp, Telegram integration
4. **Wellness Tracking**: Mental health indicators
5. **Conversation Coaching**: Real-time suggestions