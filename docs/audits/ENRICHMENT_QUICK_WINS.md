# Quick Win Enrichments for Existing Tools
## Implementation-Ready Enhancements

These enrichments can be added to existing tools with minimal effort while providing significant value.

---

## 1. Relationship Intelligence Enrichments

### Add Engagement Score (30 min implementation)
```python
# Add to imsg_relationship_intelligence

def calculate_engagement_score(sent, received, response_time, last_contact_days):
    """Composite engagement metric (0-1 scale)."""
    # Balance factor (0-1, where 0.5 is perfect balance)
    balance = 1 - abs(0.5 - (sent / (sent + received)))
    
    # Frequency factor (messages per day, capped at 1.0)
    frequency = min((sent + received) / 365, 1.0)
    
    # Responsiveness factor (faster = better, capped at 1.0)
    responsiveness = max(0, 1 - (response_time / 3600))  # 1 hour baseline
    
    # Recency factor (exponential decay)
    recency = math.exp(-last_contact_days / 30)  # 30-day half-life
    
    # Weighted combination
    score = (
        balance * 0.25 +
        frequency * 0.25 +
        responsiveness * 0.25 +
        recency * 0.25
    )
    
    return round(score, 2)

# Add to output:
"engagement_score": calculate_engagement_score(sent, received, avg_response_time, last_contact_days),
"engagement_trend": "increasing" if current_score > previous_score else "decreasing"
```

### Add Behavioral Flags (20 min implementation)
```python
# Add pattern detection flags

behavioral_flags = []

if sent_pct > 70:
    behavioral_flags.append("conversation_initiator")
elif sent_pct < 30:
    behavioral_flags.append("responsive_communicator")
else:
    behavioral_flags.append("balanced_communicator")

if avg_response_time < 300:  # 5 minutes
    behavioral_flags.append("quick_responder")
elif avg_response_time > 3600:  # 1 hour
    behavioral_flags.append("thoughtful_responder")

if messages_per_day > 10:
    behavioral_flags.append("high_frequency")
elif messages_per_day < 0.5:
    behavioral_flags.append("low_frequency")

# Add to output:
"behavioral_flags": behavioral_flags
```

---

## 2. Sentiment Evolution Enrichments

### Add Volatility Index (15 min implementation)
```python
# Add to imsg_sentiment_evolution

def calculate_sentiment_volatility(sentiment_values):
    """Measure emotional stability (0=stable, 1=volatile)."""
    if len(sentiment_values) < 2:
        return 0.0
    
    # Calculate rolling standard deviation
    volatilities = []
    window_size = min(7, len(sentiment_values))
    
    for i in range(window_size, len(sentiment_values)):
        window = sentiment_values[i-window_size:i]
        volatilities.append(np.std(window))
    
    return round(np.mean(volatilities), 2) if volatilities else 0.0

# Add to output:
"volatility_index": calculate_sentiment_volatility(daily_sentiments),
"emotional_stability": "stable" if volatility < 0.3 else "variable" if volatility < 0.6 else "volatile"
```

### Add Peak Sentiment Times (10 min implementation)
```python
# Analyze sentiment by time of day

sentiment_by_hour = defaultdict(list)

for msg in messages:
    hour = datetime.fromtimestamp(msg['date'] / 1e9 + 978307200).hour
    sentiment_by_hour[hour].append(msg['sentiment'])

hourly_averages = {hour: np.mean(sentiments) for hour, sentiments in sentiment_by_hour.items()}

# Add to output:
"peak_sentiment_times": {
    "most_positive_hour": max(hourly_averages, key=hourly_averages.get),
    "most_negative_hour": min(hourly_averages, key=hourly_averages.get),
    "pattern": "morning_person" if max(hourly_averages, key=hourly_averages.get) < 12 else "evening_person"
}
```

---

## 3. Network Intelligence Enrichments

### Add Network Health Score (25 min implementation)
```python
# Add to imsg_network_intelligence

def calculate_network_health(nodes, edges, communities):
    """Assess overall network health and resilience."""
    
    # Diversity: number of distinct communities
    diversity = len(set(c['id'] for c in communities))
    diversity_score = min(diversity / 5, 1.0)  # 5+ communities = max score
    
    # Connectivity: average connections per node
    avg_connections = len(edges) / len(nodes) if nodes else 0
    connectivity_score = min(avg_connections / 3, 1.0)  # 3+ avg = max score
    
    # Redundancy: multiple paths between nodes
    redundancy_score = 0.0
    if len(edges) > len(nodes):
        redundancy_score = min((len(edges) - len(nodes)) / len(nodes), 1.0)
    
    # Overall health score
    health_score = (diversity_score + connectivity_score + redundancy_score) / 3
    
    return {
        "overall_score": round(health_score, 2),
        "diversity_score": round(diversity_score, 2),
        "connectivity_score": round(connectivity_score, 2),
        "redundancy_score": round(redundancy_score, 2),
        "risk_level": "low" if health_score > 0.7 else "medium" if health_score > 0.4 else "high"
    }

# Add to output:
"network_health": calculate_network_health(nodes, edges, communities)
```

### Add Communication Clusters (20 min implementation)
```python
# Identify natural groupings

def identify_clusters(edges, nodes):
    """Group contacts into communication clusters."""
    
    # Build adjacency matrix
    node_map = {n['id']: i for i, n in enumerate(nodes)}
    adj_matrix = np.zeros((len(nodes), len(nodes)))
    
    for edge in edges:
        if edge['source'] in node_map and edge['target'] in node_map:
            i, j = node_map[edge['source']], node_map[edge['target']]
            adj_matrix[i][j] = edge['weight']
            adj_matrix[j][i] = edge['weight']
    
    # Simple clustering based on connection strength
    clusters = []
    visited = set()
    
    for i, node in enumerate(nodes):
        if i not in visited:
            cluster = {'members': [], 'total_messages': 0}
            queue = [i]
            
            while queue:
                current = queue.pop(0)
                if current not in visited:
                    visited.add(current)
                    cluster['members'].append(nodes[current]['id'])
                    
                    # Add strongly connected nodes
                    for j in range(len(nodes)):
                        if adj_matrix[current][j] > 10 and j not in visited:
                            queue.append(j)
            
            if len(cluster['members']) > 1:
                clusters.append(cluster)
    
    return clusters

# Add to output:
"communication_clusters": identify_clusters(edges, nodes)
```

---

## 4. Universal Enhancements

### Add Natural Language Insights (All Tools)
```python
def generate_insights(tool_name: str, metrics: Dict) -> List[str]:
    """Generate human-readable insights from metrics."""
    
    insights = []
    
    if tool_name == "relationship_intelligence":
        if metrics.get('engagement_score', 0) > 0.8:
            insights.append("This is one of your most engaged relationships")
        if metrics.get('sent_pct', 0) > 70:
            insights.append("You typically initiate conversations with this person")
        if metrics.get('response_time', float('inf')) < 300:
            insights.append("Both of you respond very quickly to each other")
    
    elif tool_name == "sentiment_evolution":
        if metrics.get('volatility_index', 0) < 0.3:
            insights.append("Your emotional tone is very consistent")
        if metrics.get('trend', 0) > 0:
            insights.append("Conversations have become more positive recently")
    
    elif tool_name == "network_intelligence":
        if metrics.get('network_health', {}).get('risk_level') == 'high':
            insights.append("Your communication network could benefit from more connections")
        if len(metrics.get('key_connectors', [])) > 0:
            insights.append(f"You have {len(metrics['key_connectors'])} people who connect different groups")
    
    return insights

# Add to all tool outputs:
"insights": generate_insights(tool_name, result_metrics)
```

### Add Actionable Recommendations (All Tools)
```python
def generate_recommendations(tool_name: str, metrics: Dict) -> List[str]:
    """Generate actionable recommendations based on metrics."""
    
    recommendations = []
    
    if tool_name == "relationship_intelligence":
        if metrics.get('last_contact_days', 0) > 30:
            recommendations.append("Reach out to reconnect - it's been over a month")
        if metrics.get('sent_pct', 0) > 80:
            recommendations.append("Try asking more questions to balance the conversation")
    
    elif tool_name == "sentiment_evolution":
        if metrics.get('volatility_index', 0) > 0.6:
            recommendations.append("Consider more consistent communication to stabilize emotions")
        if metrics.get('negative_streak_days', 0) > 3:
            recommendations.append("Address any ongoing issues - negativity persisting")
    
    elif tool_name == "best_contact_time":
        best_hour = metrics.get('best_hour', 0)
        recommendations.append(f"Schedule important conversations around {best_hour}:00")
    
    return recommendations

# Add to all tool outputs:
"recommendations": generate_recommendations(tool_name, result_metrics)
```

---

## 5. Performance Quick Wins

### Add Result Caching (All Tools)
```python
# Simple TTL cache decorator
from functools import lru_cache
import hashlib

def cache_tool_result(ttl_seconds=3600):
    """Cache tool results with TTL."""
    cache = {}
    
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Create cache key
            key_data = f"{func.__name__}:{args}:{sorted(kwargs.items())}"
            cache_key = hashlib.md5(key_data.encode()).hexdigest()
            
            # Check cache
            if cache_key in cache:
                cached_result, timestamp = cache[cache_key]
                if time.time() - timestamp < ttl_seconds:
                    return {**cached_result, "_cached": True}
            
            # Compute result
            result = await func(*args, **kwargs)
            
            # Store in cache
            cache[cache_key] = (result, time.time())
            
            return result
        
        return wrapper
    
    return decorator

# Apply to expensive tools:
@cache_tool_result(ttl_seconds=1800)  # 30 min cache
async def imsg_network_intelligence(...):
    # ... existing implementation
```

### Add Query Batching
```python
# Batch multiple contact queries
async def batch_contact_analysis(contact_ids: List[str]) -> Dict[str, Any]:
    """Analyze multiple contacts in single DB transaction."""
    
    placeholders = ','.join(['?' for _ in contact_ids])
    query = f"""
    SELECT 
        h.id as handle_id,
        COUNT(*) as message_count,
        AVG(LENGTH(m.text)) as avg_message_length,
        SUM(CASE WHEN m.is_from_me = 1 THEN 1 ELSE 0 END) as sent_count
    FROM message m
    JOIN handle h ON m.handle_id = h.ROWID
    WHERE h.id IN ({placeholders})
    GROUP BY h.id
    """
    
    results = await db.execute_query(query, contact_ids)
    return {r['handle_id']: r for r in results}
```

---

## Implementation Priority

1. **High Value, Low Effort** (Week 1):
   - Engagement scores
   - Behavioral flags  
   - Volatility index
   - Natural language insights

2. **Medium Value, Medium Effort** (Week 2):
   - Network health scoring
   - Peak sentiment times
   - Communication clusters
   - Recommendations engine

3. **Performance Optimizations** (Week 3):
   - Result caching
   - Query batching
   - Incremental updates

Each enhancement maintains privacy (hashing, redaction) and can be deployed independently.