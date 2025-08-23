# New Tools Implementation Specification
## iMessage Advanced Insights MCP Server

This document provides detailed implementation specifications for the proposed new tools, including schemas, algorithms, and privacy considerations.

---

## 1. Conversation Dynamics Analyzer

### Tool Definition
```python
@mcp.tool()
async def imsg_conversation_dynamics(
    db_path: str = "~/Library/Messages/chat.db",
    contact_id: Optional[str] = None,
    window_days: int = 90,
    include_samples: bool = False,
    redact: bool = True
) -> Dict[str, Any]:
    """
    Analyze conversational flow patterns, turn-taking dynamics, and engagement quality.
    
    Provides insights into who initiates conversations, how balanced the exchange is,
    and the overall health of communication patterns.
    """
```

### Implementation Algorithm

```python
# Core metrics to calculate:

1. Initiation Patterns:
   - Group messages by conversation (gap > 4 hours = new conversation)
   - Track first message sender for each conversation
   - Calculate initiation ratio over time window

2. Turn-Taking Analysis:
   - Measure response intervals
   - Count alternating message sequences
   - Identify monologue vs dialogue patterns

3. Engagement Quality:
   - Message length distribution
   - Media attachment frequency
   - Emoji usage density
   - Topic diversity (via TF-IDF on message content)

4. Conversation Depth:
   - Average messages per conversation
   - Conversation duration (first to last message)
   - Peak engagement hours
```

### SQL Query Pattern
```sql
-- Conversation detection with 4-hour gap
WITH conversations AS (
  SELECT 
    m.*,
    CASE 
      WHEN LAG(date, 1) OVER (PARTITION BY handle_id ORDER BY date) IS NULL 
        OR date - LAG(date, 1) OVER (PARTITION BY handle_id ORDER BY date) > 14400000000000
      THEN 1 
      ELSE 0 
    END as new_conversation
  FROM message m
  WHERE date >= ? AND date <= ?
),
conversation_ids AS (
  SELECT 
    *,
    SUM(new_conversation) OVER (PARTITION BY handle_id ORDER BY date) as conversation_id
  FROM conversations
),
conversation_stats AS (
  SELECT 
    handle_id,
    conversation_id,
    MIN(date) as conv_start,
    MAX(date) as conv_end,
    COUNT(*) as message_count,
    SUM(CASE WHEN is_from_me = 1 THEN 1 ELSE 0 END) as sent_count,
    MIN(CASE WHEN is_from_me = 1 THEN date ELSE NULL END) as first_sent,
    MIN(CASE WHEN is_from_me = 0 THEN date ELSE NULL END) as first_received
  FROM conversation_ids
  GROUP BY handle_id, conversation_id
)
SELECT * FROM conversation_stats;
```

### Return Schema
```json
{
  "initiation_patterns": {
    "user_initiated_pct": 0.62,
    "contact_initiated_pct": 0.38,
    "total_conversations": 47,
    "trend": "increasing_user_initiation",
    "recent_shift": "+15% user initiation last 30 days"
  },
  "turn_taking": {
    "avg_turns_per_conversation": 8.4,
    "response_consistency": 0.85,
    "balanced_exchange_score": 0.73,
    "monologue_frequency": 0.12,
    "conversation_depth": "meaningful"
  },
  "engagement_quality": {
    "message_richness_score": 0.72,
    "avg_words_per_message": 42,
    "media_share_rate": 0.15,
    "emoji_density": 0.08,
    "topic_diversity_score": 0.68,
    "emotional_range": 0.54
  },
  "temporal_patterns": {
    "avg_conversation_duration_minutes": 28,
    "peak_engagement_hours": [19, 20, 21],
    "weekend_vs_weekday_ratio": 1.34
  },
  "conversation_health": {
    "overall_score": 0.81,
    "strengths": ["balanced_exchange", "diverse_topics", "consistent_engagement"],
    "improvement_areas": ["increase_media_sharing", "deeper_weekend_conversations"],
    "flags": ["healthy_dynamic"],
    "recommendations": [
      "Your conversations show good balance and engagement",
      "Try sharing more photos to enrich conversations",
      "Weekend chats could be longer and more meaningful"
    ]
  }
}
```

---

## 2. Emotional Intelligence Monitor

### Tool Definition
```python
@mcp.tool()
async def imsg_emotional_intelligence(
    db_path: str = "~/Library/Messages/chat.db",
    contact_id: Optional[str] = None,
    window_days: int = 180,
    emotion_categories: List[str] = None,
    include_conflict_analysis: bool = True,
    redact: bool = True
) -> Dict[str, Any]:
    """
    Deep emotional pattern analysis including volatility, resilience, support patterns,
    and conflict resolution dynamics.
    """
```

### Emotion Detection Algorithm
```python
# Enhanced emotion categories with weighted keywords
EMOTION_PATTERNS = {
    "joy": {
        "keywords": ["happy", "excited", "love", "amazing", "wonderful", "great", "😊", "😄", "❤️"],
        "weight": 1.0,
        "context_boost": ["birthday", "congratulations", "celebration"]
    },
    "stress": {
        "keywords": ["stressed", "worried", "anxious", "overwhelmed", "busy", "deadline", "😰", "😟"],
        "weight": 1.2,
        "context_boost": ["work", "exam", "meeting"]
    },
    "affection": {
        "keywords": ["love", "miss", "care", "thinking of", "heart", "💕", "😘", "🤗"],
        "weight": 1.1,
        "context_boost": ["you", "us", "together"]
    },
    "frustration": {
        "keywords": ["annoyed", "frustrated", "angry", "upset", "hate", "😤", "😠", "🙄"],
        "weight": 1.3,
        "context_boost": ["always", "never", "again"]
    },
    "support": {
        "keywords": ["here for you", "help", "support", "understand", "listen", "🤝", "💪"],
        "weight": 1.0,
        "context_boost": ["anything", "need", "available"]
    }
}

# Volatility calculation
def calculate_emotional_volatility(emotion_series):
    """
    Measure emotional swings using standard deviation of emotion scores
    normalized by mean absolute emotion level.
    """
    if len(emotion_series) < 2:
        return 0.0
    
    std_dev = np.std(emotion_series)
    mean_abs = np.mean(np.abs(emotion_series))
    
    return min(std_dev / (mean_abs + 0.1), 1.0)  # Normalize to 0-1

# Conflict detection
def detect_conflict_cycles(messages):
    """
    Identify negative sentiment spikes followed by resolution patterns.
    """
    conflicts = []
    in_conflict = False
    conflict_start = None
    
    for i, msg in enumerate(messages):
        sentiment = msg['sentiment_score']
        
        if sentiment < -0.3 and not in_conflict:
            in_conflict = True
            conflict_start = i
        elif in_conflict and sentiment > 0.2:
            # Resolution detected
            conflicts.append({
                'start_idx': conflict_start,
                'end_idx': i,
                'duration_hours': (messages[i]['date'] - messages[conflict_start]['date']) / 3600,
                'severity': abs(messages[conflict_start]['sentiment_score'])
            })
            in_conflict = False
    
    return conflicts
```

### Return Schema
```json
{
  "emotional_profile": {
    "dominant_emotions": [
      {"emotion": "joy", "frequency": 0.35},
      {"emotion": "support", "frequency": 0.28},
      {"emotion": "affection", "frequency": 0.22}
    ],
    "emotional_range": 0.76,
    "expression_complexity": "high",
    "volatility_score": 0.28,
    "stability_trend": "improving",
    "baseline_sentiment": 0.42
  },
  "temporal_emotions": {
    "by_hour": {
      "morning": {"dominant": "neutral", "avg_sentiment": 0.15},
      "afternoon": {"dominant": "focused", "avg_sentiment": 0.25},
      "evening": {"dominant": "joy", "avg_sentiment": 0.48},
      "night": {"dominant": "affection", "avg_sentiment": 0.38}
    },
    "by_day": {
      "weekday": {"stress": 0.22, "joy": 0.31},
      "weekend": {"stress": 0.08, "joy": 0.52}
    }
  },
  "stress_indicators": {
    "overall_stress_level": 0.18,
    "late_night_negativity": 0.15,
    "weekend_stress": 0.08,
    "workday_pressure": 0.34,
    "stress_topics": ["deadlines", "meetings", "traffic"],
    "stress_recovery_time_hours": 4.2
  },
  "support_patterns": {
    "gives_emotional_support": 0.76,
    "seeks_emotional_support": 0.45,
    "mutual_support_balance": 0.61,
    "support_response_time_minutes": 12,
    "support_effectiveness": 0.84
  },
  "conflict_analysis": {
    "conflict_frequency_per_month": 1.3,
    "avg_resolution_hours": 16,
    "repair_success_rate": 0.89,
    "unresolved_tensions": 1,
    "recent_conflicts": [
      {
        "date": "2024-12-15",
        "duration_hours": 18,
        "severity": "minor",
        "resolved": true,
        "trigger_topic": "scheduling"
      }
    ],
    "conflict_triggers": ["miscommunication", "scheduling", "expectations"]
  },
  "emotional_intelligence_score": 0.78,
  "well_being_flags": [
    "emotionally_stable",
    "quick_conflict_recovery",
    "strong_support_giver",
    "healthy_stress_levels"
  ],
  "insights": [
    "You maintain emotional stability even during stressful periods",
    "Your support-giving is stronger than support-seeking - consider balance",
    "Conflicts resolve quickly, showing good relationship resilience",
    "Evening conversations tend to be most positive"
  ],
  "recommendations": [
    "Share your own challenges more to balance support dynamics",
    "Address scheduling conflicts proactively to prevent tension",
    "Maintain your excellent conflict resolution practices"
  ]
}
```

---

## 3. Implementation Guidelines

### Privacy Controls
```python
class PrivacyEnforcer:
    """Ensure all new tools respect privacy settings."""
    
    @staticmethod
    async def apply_privacy_filters(data: Dict, config: Config) -> Dict:
        """Apply redaction and hashing based on config."""
        if config.privacy.hash_identifiers:
            data = hash_all_identifiers(data)
        
        if config.privacy.redact_by_default:
            data = redact_sensitive_content(data)
        
        if config.privacy.preview_caps.get('enabled'):
            data = apply_message_caps(data)
        
        return data
    
    @staticmethod
    def validate_consent_scope(tool_name: str, requested_data: List[str]) -> bool:
        """Ensure requested data is within consent scope."""
        allowed_scopes = {
            'basic': ['counts', 'timestamps', 'metadata'],
            'analytics': ['sentiment', 'topics', 'patterns'],
            'deep': ['emotional', 'behavioral', 'predictive']
        }
        # Check if current consent covers requested scope
        return True  # Simplified
```

### Performance Optimization
```python
class QueryOptimizer:
    """Optimize database queries for new tools."""
    
    @staticmethod
    def batch_contact_queries(contact_ids: List[str], queries: List[str]) -> str:
        """Combine multiple contact queries into single transaction."""
        return f"""
        WITH contact_batch AS (
            SELECT * FROM handle WHERE id IN ({','.join(['?'] * len(contact_ids))})
        )
        {' UNION ALL '.join(queries)}
        """
    
    @staticmethod
    async def cache_expensive_calculations(key: str, compute_func, ttl: int = 3600):
        """Cache expensive computations with TTL."""
        cached = await cache.get(key)
        if cached and not cached.expired:
            return cached.value
        
        result = await compute_func()
        await cache.set(key, result, ttl=ttl)
        return result
```

### Error Handling
```python
class ToolErrorHandler:
    """Consistent error handling for new tools."""
    
    @staticmethod
    def wrap_tool_execution(func):
        async def wrapper(*args, **kwargs):
            try:
                # Check consent
                if not await check_consent():
                    return {"error": "No active consent", "error_type": "consent_required"}
                
                # Execute tool
                result = await func(*args, **kwargs)
                
                # Apply privacy filters
                result = await PrivacyEnforcer.apply_privacy_filters(result, get_config())
                
                return result
                
            except DatabaseError as e:
                logger.error(f"Database error in {func.__name__}: {e}")
                return {"error": "Database access failed", "error_type": "database_error"}
                
            except Exception as e:
                logger.error(f"Unexpected error in {func.__name__}: {e}")
                return {"error": "Internal error", "error_type": "internal_error"}
        
        return wrapper
```

---

## 4. Testing Strategy

### Unit Tests
```python
# test_conversation_dynamics.py
async def test_initiation_detection():
    """Test conversation initiation pattern detection."""
    messages = [
        {"date": 1000, "is_from_me": 1, "handle_id": "test"},  # User initiates
        {"date": 2000, "is_from_me": 0, "handle_id": "test"},
        {"date": 20000000, "is_from_me": 0, "handle_id": "test"},  # Contact initiates (new conv)
        {"date": 20001000, "is_from_me": 1, "handle_id": "test"},
    ]
    
    result = await detect_initiation_patterns(messages)
    assert result['user_initiated'] == 1
    assert result['contact_initiated'] == 1
    assert result['user_initiated_pct'] == 0.5

# test_emotional_intelligence.py  
async def test_volatility_calculation():
    """Test emotional volatility scoring."""
    emotion_series = [0.5, 0.4, -0.8, 0.9, -0.6, 0.7]  # High swings
    volatility = calculate_emotional_volatility(emotion_series)
    assert 0.6 < volatility < 0.8  # Should indicate high volatility
    
    stable_series = [0.4, 0.5, 0.4, 0.6, 0.5, 0.4]  # Stable
    stability = calculate_emotional_volatility(stable_series)
    assert stability < 0.3  # Should indicate low volatility
```

### Integration Tests
```python
async def test_full_conversation_dynamics_flow():
    """Test complete conversation dynamics analysis."""
    # Grant consent
    await consent_manager.grant_consent(expires_hours=1)
    
    # Run analysis
    result = await imsg_conversation_dynamics(
        window_days=30,
        include_samples=False
    )
    
    # Verify structure
    assert 'initiation_patterns' in result
    assert 'turn_taking' in result
    assert 'engagement_quality' in result
    assert 'conversation_health' in result
    
    # Verify privacy
    assert 'hash:' in str(result)  # Contact IDs should be hashed
    assert '@' not in str(result)   # No email addresses
    assert not any(char.isdigit() for char in str(result).replace('hash:', ''))  # No phone numbers
```

---

## 5. Deployment Checklist

- [ ] Implement core algorithms with error handling
- [ ] Add comprehensive logging for debugging
- [ ] Create database indexes for new query patterns
- [ ] Implement caching layer for expensive calculations
- [ ] Add privacy filters to all outputs
- [ ] Write unit tests (>90% coverage)
- [ ] Write integration tests with real data
- [ ] Performance test with large databases (10GB+)
- [ ] Update MCP_TOOLS_REFERENCE.md
- [ ] Create user documentation with examples
- [ ] Add to Claude Desktop config validation
- [ ] Test end-to-end with Claude Desktop