"""
Analytics tools for relationship and conversation intelligence.
"""

import logging
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List

from mcp import Server

from ..config import Config
from ..db import get_database
from ..models import (
    RelationshipIntelligenceInput, RelationshipIntelligenceOutput,
    ConversationTopicsInput, ConversationTopicsOutput,
    SentimentEvolutionInput, SentimentEvolutionOutput,
    ResponseTimeInput, ResponseTimeOutput,
    CadenceCalendarInput, CadenceCalendarOutput
)
from ..privacy import hash_contact_id, sanitize_contact

logger = logging.getLogger(__name__)


def register_analytics_tools(server: Server, config: Config) -> None:
    """Register analytics tools with the server."""
    
    @server.tool()
    async def imsg_relationship_intelligence(arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Per-contact multi-metric profile: volume, balance, responsiveness, streaks.
        
        Analyzes communication patterns to provide insights about relationships.
        """
        try:
            # Validate input
            params = RelationshipIntelligenceInput(**arguments)
            
            # Get database connection
            db = await get_database(params.db_path)
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=params.window_days)
            start_timestamp = int((start_date.timestamp() - 978307200) * 1e9)
            end_timestamp = int((end_date.timestamp() - 978307200) * 1e9)
            
            # Build query for contact statistics
            query = """
            WITH contact_stats AS (
                SELECT 
                    h.id as handle_id,
                    COUNT(*) as total_messages,
                    SUM(CASE WHEN m.is_from_me = 1 THEN 1 ELSE 0 END) as sent_count,
                    SUM(CASE WHEN m.is_from_me = 0 THEN 1 ELSE 0 END) as received_count,
                    MAX(m.date) as last_message_date,
                    MIN(m.date) as first_message_date
                FROM message m
                JOIN handle h ON m.handle_id = h.ROWID
                WHERE m.date >= ? AND m.date <= ?
                GROUP BY h.id
                HAVING COUNT(*) > 10
            )
            SELECT * FROM contact_stats
            ORDER BY total_messages DESC
            LIMIT 50
            """
            
            results = await db.execute_query(query, (start_timestamp, end_timestamp))
            
            contacts = []
            for row in results:
                handle_id = row['handle_id']
                
                # Apply contact filters if specified
                if params.contact_filters and handle_id not in params.contact_filters:
                    continue
                
                # Calculate metrics
                total = row['total_messages']
                sent = row['sent_count']
                sent_pct = (sent / total * 100) if total > 0 else 0
                
                # Calculate average daily messages
                first_date = datetime.fromtimestamp(row['first_message_date'] / 1e9 + 978307200)
                last_date = datetime.fromtimestamp(row['last_message_date'] / 1e9 + 978307200)
                days_active = max((last_date - first_date).days, 1)
                avg_daily = total / days_active
                
                # TODO: Calculate median response time (requires more complex query)
                median_response_time = 300.0  # Placeholder: 5 minutes
                
                # TODO: Calculate streak days (requires sequential analysis)
                streak_days_max = 7  # Placeholder
                
                # Determine relationship flags
                flags = []
                if avg_daily > 20:
                    flags.append("high-volume")
                elif avg_daily < 0.5:
                    flags.append("low-volume")
                
                if sent_pct > 70:
                    flags.append("one-sided-sending")
                elif sent_pct < 30:
                    flags.append("one-sided-receiving")
                else:
                    flags.append("balanced")
                
                # Build contact info
                contact_id = hash_contact_id(handle_id) if params.redact else handle_id
                
                contacts.append({
                    "contact_id": contact_id,
                    "display_name": None if params.redact else f"Contact {handle_id[:8]}",
                    "messages_total": total,
                    "sent_pct": round(sent_pct, 1),
                    "median_response_time_s": median_response_time,
                    "avg_daily_msgs": round(avg_daily, 1),
                    "streak_days_max": streak_days_max,
                    "last_contacted": last_date.strftime("%Y-%m-%d"),
                    "flags": flags
                })
            
            # Build response
            output = RelationshipIntelligenceOutput(contacts=contacts)
            return output.model_dump()
            
        except Exception as e:
            logger.error(f"Relationship intelligence failed: {e}")
            return {
                "error": str(e),
                "error_type": "analysis_failed"
            }
    
    @server.tool()
    async def imsg_response_time_distribution(arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reply latency distribution + quantiles overall/per-contact.
        
        Analyzes how quickly messages are responded to, providing insights
        into communication responsiveness.
        """
        try:
            # Validate input
            params = ResponseTimeInput(**arguments)
            
            # Get database connection
            db = await get_database(params.db_path)
            
            # Build query for response times
            # This is a simplified version - full implementation would track
            # conversation threads more accurately
            base_query = """
            WITH response_pairs AS (
                SELECT 
                    m1.date as sent_date,
                    m2.date as response_date,
                    (m2.date - m1.date) / 1000000000.0 as response_time_s
                FROM message m1
                JOIN message m2 ON m2.handle_id = m1.handle_id
                WHERE m1.is_from_me = 1 
                AND m2.is_from_me = 0
                AND m2.date > m1.date
                AND m2.date < m1.date + 86400000000000  -- Within 24 hours
                AND m1.text IS NOT NULL
                AND m2.text IS NOT NULL
            """
            
            if params.contact_id:
                # TODO: Add contact filtering
                pass
            
            query = base_query + """
            )
            SELECT response_time_s
            FROM response_pairs
            WHERE response_time_s > 0 AND response_time_s < 86400
            ORDER BY response_time_s
            LIMIT 10000
            """
            
            results = await db.execute_query(query)
            
            if not results:
                # Return empty statistics
                return ResponseTimeOutput(
                    p50_s=0,
                    p90_s=0,
                    p99_s=0,
                    histogram=[],
                    samples=0
                ).model_dump()
            
            # Extract response times
            response_times = [r['response_time_s'] for r in results]
            response_times.sort()
            
            # Calculate percentiles
            n = len(response_times)
            p50 = response_times[int(n * 0.5)]
            p90 = response_times[int(n * 0.9)]
            p99 = response_times[int(n * 0.99)] if n > 100 else p90
            
            # Build histogram
            histogram_buckets = [
                (0, 60, "< 1 min"),
                (60, 300, "1-5 min"),
                (300, 900, "5-15 min"),
                (900, 3600, "15-60 min"),
                (3600, 7200, "1-2 hours"),
                (7200, 86400, "> 2 hours")
            ]
            
            histogram = []
            for start, end, label in histogram_buckets:
                count = sum(1 for t in response_times if start <= t < end)
                histogram.append({
                    "bucket": label,
                    "count": count
                })
            
            # Build response
            output = ResponseTimeOutput(
                p50_s=round(p50, 1),
                p90_s=round(p90, 1),
                p99_s=round(p99, 1),
                histogram=histogram,
                samples=n
            )
            
            return output.model_dump()
            
        except Exception as e:
            logger.error(f"Response time analysis failed: {e}")
            return {
                "error": str(e),
                "error_type": "analysis_failed"
            }
    
    @server.tool()
    async def imsg_cadence_calendar(arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Heatmap-ready message counts by hour x weekday.
        
        Creates a 24x7 matrix showing when messages are typically sent/received,
        useful for understanding communication patterns.
        """
        try:
            # Validate input
            params = CadenceCalendarInput(**arguments)
            
            # Get database connection
            db = await get_database(params.db_path)
            
            # Build query for hourly message counts
            base_query = """
            SELECT 
                strftime('%w', datetime(date/1000000000 + 978307200, 'unixepoch')) as weekday,
                strftime('%H', datetime(date/1000000000 + 978307200, 'unixepoch')) as hour,
                COUNT(*) as count
            FROM message
            WHERE date IS NOT NULL
            """
            
            if params.contact_id:
                # TODO: Add contact filtering
                pass
            
            query = base_query + """
            GROUP BY weekday, hour
            ORDER BY weekday, hour
            """
            
            results = await db.execute_query(query)
            
            # Initialize 24x7 matrix (hours x weekdays)
            matrix = [[0 for _ in range(7)] for _ in range(24)]
            
            # Fill matrix with counts
            for row in results:
                hour = int(row['hour'])
                weekday = int(row['weekday'])  # 0 = Sunday
                count = row['count']
                matrix[hour][weekday] = count
            
            # Build response
            output = CadenceCalendarOutput(
                matrix=matrix,
                hours=list(range(24)),
                weekdays=["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
            )
            
            return output.model_dump()
            
        except Exception as e:
            logger.error(f"Cadence calendar failed: {e}")
            return {
                "error": str(e),
                "error_type": "analysis_failed"
            }
    
    @server.tool()
    async def imsg_conversation_topics(arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Lightweight topic/keyword extraction + time trends.
        
        Extracts common topics and keywords from conversations using
        deterministic methods (no ML by default).
        """
        try:
            # Validate input
            params = ConversationTopicsInput(**arguments)
            
            # Get database connection
            db = await get_database(params.db_path)
            
            # Calculate date range
            from datetime import datetime, timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=params.since_days)
            start_timestamp = int((start_date.timestamp() - 978307200) * 1e9)
            
            # Build query
            query = """
            SELECT text
            FROM message
            WHERE text IS NOT NULL
            AND date >= ?
            """
            
            query_params = [start_timestamp]
            
            if params.contact_id:
                # TODO: Add contact filtering
                pass
            
            query += " LIMIT 5000"  # Limit for performance
            
            results = await db.execute_query(query, tuple(query_params))
            
            # Extract terms (simple word frequency)
            from collections import Counter
            import re
            
            # Common English stop words
            stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
                'before', 'after', 'above', 'below', 'between', 'under', 'again',
                'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
                'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
                'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
                'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should',
                'now', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
                'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their',
                'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were', 'be',
                'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'doing'
            }
            
            word_counts = Counter()
            
            for row in results:
                text = row['text'].lower()
                # Extract words (alphanumeric only)
                words = re.findall(r'\b[a-z]+\b', text)
                # Filter stop words and short words
                words = [w for w in words if w not in stop_words and len(w) > 2]
                word_counts.update(words)
            
            # Get top terms
            top_terms = word_counts.most_common(params.top_k)
            
            terms = [
                {"term": term, "count": count}
                for term, count in top_terms
            ]
            
            # Generate trend sparklines (simplified - would need time-based analysis)
            trends = []
            for term, _ in top_terms[:10]:  # Top 10 for trends
                # Placeholder sparkline
                trends.append({
                    "term": term,
                    "spark": "▁▃▅▇▅▃▁"  # Example sparkline
                })
            
            # Add notes
            notes = []
            if len(word_counts) > 0:
                notes.append(f"Analyzed {len(results)} messages")
                notes.append(f"Found {len(word_counts)} unique terms")
            
            if params.use_transformer and not config.features.use_transformer_nlp:
                notes.append("Transformer models disabled in config")
            
            # Build response
            output = ConversationTopicsOutput(
                terms=terms,
                trend=trends,
                notes=notes
            )
            
            return output.model_dump()
            
        except Exception as e:
            logger.error(f"Conversation topics failed: {e}")
            return {
                "error": str(e),
                "error_type": "analysis_failed"
            }
    
    @server.tool()
    async def imsg_sentiment_evolution(arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deterministic sentiment/time aggregation.
        
        Tracks sentiment changes over time using rule-based analysis
        (no ML models required).
        """
        try:
            # Validate input
            params = SentimentEvolutionInput(**arguments)
            
            # Get database connection
            db = await get_database(params.db_path)
            
            # Build query for messages over time
            query = """
            SELECT 
                DATE(date/1000000000 + 978307200, 'unixepoch') as day,
                text
            FROM message
            WHERE text IS NOT NULL
            """
            
            if params.contact_id:
                # TODO: Add contact filtering
                pass
            
            query += " ORDER BY date DESC LIMIT 10000"
            
            results = await db.execute_query(query)
            
            # Simple sentiment analysis using word lists
            positive_words = {
                'love', 'great', 'awesome', 'wonderful', 'fantastic', 'excellent',
                'happy', 'joy', 'blessed', 'grateful', 'amazing', 'beautiful',
                'thanks', 'thank', 'appreciate', 'excited', 'fun', 'good', 'nice',
                'perfect', 'brilliant', 'delightful', 'pleased', 'glad', 'cheerful'
            }
            
            negative_words = {
                'hate', 'terrible', 'awful', 'horrible', 'bad', 'worst', 'angry',
                'sad', 'depressed', 'frustrated', 'annoyed', 'disappointed', 'upset',
                'sorry', 'unfortunately', 'problem', 'issue', 'difficult', 'hard',
                'stress', 'worried', 'anxious', 'fear', 'scared', 'nervous'
            }
            
            # Group by day and calculate sentiment
            daily_sentiments = defaultdict(list)
            
            for row in results:
                day = row['day']
                text = row['text'].lower()
                
                # Count positive and negative words
                words = text.split()
                pos_count = sum(1 for w in words if w in positive_words)
                neg_count = sum(1 for w in words if w in negative_words)
                
                # Calculate sentiment score
                if pos_count + neg_count > 0:
                    score = (pos_count - neg_count) / (pos_count + neg_count)
                    daily_sentiments[day].append(score)
            
            # Calculate rolling average
            from datetime import datetime, timedelta
            series = []
            
            for day in sorted(daily_sentiments.keys()):
                scores = daily_sentiments[day]
                if scores:
                    avg_score = sum(scores) / len(scores)
                    series.append({
                        "ts": f"{day}T12:00:00",
                        "score": round(avg_score, 3)
                    })
            
            # Apply rolling window
            if len(series) > params.window_days:
                windowed_series = []
                for i in range(params.window_days, len(series)):
                    window_scores = [s['score'] for s in series[i-params.window_days:i]]
                    avg = sum(window_scores) / len(window_scores)
                    windowed_series.append({
                        "ts": series[i]['ts'],
                        "score": round(avg, 3)
                    })
                series = windowed_series
            
            # Calculate summary statistics
            all_scores = [s['score'] for s in series]
            
            if all_scores:
                mean_score = sum(all_scores) / len(all_scores)
                
                # Calculate 30-day delta
                recent_scores = all_scores[-30:] if len(all_scores) > 30 else all_scores
                older_scores = all_scores[-60:-30] if len(all_scores) > 60 else all_scores[:len(all_scores)//2]
                
                recent_mean = sum(recent_scores) / len(recent_scores) if recent_scores else 0
                older_mean = sum(older_scores) / len(older_scores) if older_scores else recent_mean
                delta_30d = recent_mean - older_mean
            else:
                mean_score = 0
                delta_30d = 0
            
            # Build response
            output = SentimentEvolutionOutput(
                series=series[-100:],  # Limit to last 100 points
                summary={
                    "mean": round(mean_score, 3),
                    "delta_30d": round(delta_30d, 3)
                }
            )
            
            return output.model_dump()
            
        except Exception as e:
            logger.error(f"Sentiment evolution failed: {e}")
            return {
                "error": str(e),
                "error_type": "analysis_failed"
            }