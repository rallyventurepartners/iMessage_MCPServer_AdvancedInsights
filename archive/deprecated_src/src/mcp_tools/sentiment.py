"""
Sentiment analysis tools for the iMessage Advanced Insights server.

This module provides tools for analyzing sentiment patterns and emotional tone
in messages over time.
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, Counter
import statistics

from ..database import get_database
from ..exceptions import DatabaseError, ToolExecutionError
from ..utils.responses import error_response, success_response
from ..utils.decorators import requires_consent, parse_date
from ..utils.analysis_cache import cached_analysis, cache_sentiment_analysis
from .registry import register_tool

logger = logging.getLogger(__name__)


# Sentiment lexicons
POSITIVE_WORDS = {
    'love', 'happy', 'great', 'awesome', 'excellent', 'good', 'wonderful',
    'fantastic', 'amazing', 'perfect', 'beautiful', 'joy', 'excited',
    'thankful', 'grateful', 'blessed', 'smile', 'laugh', 'fun', 'enjoy',
    'appreciate', 'best', 'brilliant', 'super', 'nice', 'sweet', 'cool',
    'glad', 'pleased', 'delighted', 'cheerful', 'optimistic', 'yes', 'yeah',
    'haha', 'lol', 'yay', 'congrats', 'congratulations', 'thanks', 'thank you'
}

NEGATIVE_WORDS = {
    'hate', 'sad', 'angry', 'bad', 'terrible', 'awful', 'horrible', 'upset',
    'disappointed', 'frustrate', 'annoying', 'worry', 'stress', 'anxiety',
    'fear', 'scared', 'nervous', 'sorry', 'apologize', 'mistake', 'wrong',
    'problem', 'issue', 'difficult', 'hard', 'fail', 'lost', 'miss', 'hurt',
    'pain', 'sick', 'tired', 'exhausted', 'boring', 'sucks', 'damn', 'hell',
    'crap', 'shit', 'fuck', 'no', 'nope', 'never', 'cannot', 'cant', 'won\'t'
}

# Emoji sentiment mapping
POSITIVE_EMOJIS = {'😊', '😃', '😄', '😁', '😆', '😍', '🥰', '😘', '❤️', '💕', 
                   '💖', '💗', '💓', '💞', '💝', '🎉', '🎊', '✨', '🌟', '⭐',
                   '👍', '👏', '🙌', '💪', '🔥', '🎯', '💯', '🥳', '🤗', '🙏'}

NEGATIVE_EMOJIS = {'😢', '😭', '😔', '😞', '😟', '😕', '🙁', '☹️', '😣', '😖',
                   '😫', '😩', '😤', '😠', '😡', '🤬', '😨', '😰', '😥', '😓',
                   '🤒', '🤕', '🤢', '🤮', '💔', '😴', '😪', '👎', '😒', '🙄'}


@cached_analysis("sentiment", ["text"])
async def calculate_message_sentiment(text: str) -> Dict[str, Any]:
    """
    Calculate sentiment score for a single message.
    
    Returns dict with:
    - score: -1 to 1 (negative to positive)
    - magnitude: 0 to 1 (strength of sentiment)
    - dominant: 'positive', 'negative', or 'neutral'
    """
    if not text:
        return {'score': 0, 'magnitude': 0, 'dominant': 'neutral'}
    
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    
    # Count sentiment words
    positive_count = sum(1 for word in words if word in POSITIVE_WORDS)
    negative_count = sum(1 for word in words if word in NEGATIVE_WORDS)
    
    # Count sentiment emojis
    positive_emoji_count = sum(1 for char in text if char in POSITIVE_EMOJIS)
    negative_emoji_count = sum(1 for char in text if char in NEGATIVE_EMOJIS)
    
    # Weight emojis slightly higher
    total_positive = positive_count + (positive_emoji_count * 1.5)
    total_negative = negative_count + (negative_emoji_count * 1.5)
    total_words = len(words) + positive_emoji_count + negative_emoji_count
    
    if total_words == 0:
        return {'score': 0, 'magnitude': 0, 'dominant': 'neutral'}
    
    # Calculate score
    if total_positive + total_negative == 0:
        score = 0
    else:
        score = (total_positive - total_negative) / (total_positive + total_negative)
    
    # Calculate magnitude (strength)
    magnitude = (total_positive + total_negative) / total_words
    magnitude = min(1.0, magnitude)  # Cap at 1.0
    
    # Determine dominant sentiment
    if score > 0.1:
        dominant = 'positive'
    elif score < -0.1:
        dominant = 'negative'
    else:
        dominant = 'neutral'
    
    return {
        'score': round(score, 3),
        'magnitude': round(magnitude, 3),
        'dominant': dominant
    }


@register_tool(
    name="analyze_sentiment_trends",
    description="Analyze sentiment trends over time"
)
@requires_consent
async def analyze_sentiment_trends_tool(
    contact_id: Optional[str] = None,
    time_period: str = "30 days",
    granularity: str = "daily"
) -> Dict[str, Any]:
    """
    Analyze sentiment trends in conversations over time.
    
    Args:
        contact_id: Optional contact to analyze (None = all conversations)
        time_period: Time period to analyze (e.g., "30 days", "3 months")
        granularity: Analysis granularity ("hourly", "daily", "weekly", "monthly")
        
    Returns:
        Sentiment analysis with trends, patterns, and insights
    """
    try:
        # Get database connection
        db = await get_database()
        
        # Parse time period
        end_date = datetime.now()
        start_date = parse_date(time_period)
        if not start_date:
            # Default to 30 days if parsing fails
            start_date = end_date - timedelta(days=30)
        
        # Determine time buckets based on granularity
        if granularity == "hourly":
            bucket_seconds = 3600
            date_format = "%Y-%m-%d %H:00"
        elif granularity == "weekly":
            bucket_seconds = 604800  # 7 days
            date_format = "%Y-W%W"
        elif granularity == "monthly":
            bucket_seconds = 2592000  # 30 days
            date_format = "%Y-%m"
        else:  # daily
            bucket_seconds = 86400
            date_format = "%Y-%m-%d"
        
        # Get messages with memory-safe pagination
        if contact_id:
            # Check if memory monitor is available
            from ..utils.memory_monitor import MemoryMonitor
            memory_monitor = None
            try:
                memory_monitor = await MemoryMonitor.get_instance()
                memory_stats = memory_monitor.get_memory_stats()
                if memory_stats["process"]["percent"] > 60:
                    # High memory usage, use smaller page size
                    page_size = 1000
                    logger.warning("High memory usage detected, limiting to 1000 messages per page")
                else:
                    page_size = 5000  # Reduced from 10000 for safety
            except:
                page_size = 5000  # Default safe size
            
            result = await db.get_messages_from_contact(
                phone_number=contact_id,
                start_date=start_date,
                end_date=end_date,
                page=1, page_size=page_size
            )
        else:
            # Analyze all messages - get recent messages without contact filter
            # Note: There's no direct method to get all messages with date filters
            # We'll use get_messages_from_contact with empty contact to get recent messages
            
            # Use same memory-safe page size
            result = await db.get_messages_from_contact(
                phone_number="",  # Empty to get all messages
                start_date=start_date,
                end_date=end_date,
                page=1, page_size=page_size
            )
        
        messages = result.get("messages", [])
        
        if not messages:
            return success_response({
                "time_period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                    "days": (end_date - start_date).days
                },
                "summary": "No messages found in the specified time period",
                "trends": [],
                "insights": {}
            })
        
        # Analyze sentiment by time bucket
        sentiment_buckets = defaultdict(lambda: {
            'messages': 0,
            'total_score': 0,
            'positive': 0,
            'negative': 0,
            'neutral': 0,
            'scores': []
        })
        
        # Overall statistics
        all_scores = []
        sentiment_by_contact = defaultdict(lambda: {'scores': [], 'count': 0})
        word_frequency = Counter()
        emoji_frequency = Counter()
        
        for msg in messages:
            # Skip empty messages
            if not msg.get('text'):
                continue
            
            # Calculate sentiment with caching
            sentiment = await calculate_message_sentiment(msg['text'])
            all_scores.append(sentiment['score'])
            
            # Determine time bucket
            msg_date = datetime.fromisoformat(msg['date'])
            bucket_key = msg_date.strftime(date_format)
            
            # Update bucket statistics
            bucket = sentiment_buckets[bucket_key]
            bucket['messages'] += 1
            bucket['total_score'] += sentiment['score']
            bucket['scores'].append(sentiment['score'])
            bucket[sentiment['dominant']] += 1
            
            # Track by contact
            contact = msg.get('contact_name', msg.get('contact_id', 'Unknown'))
            sentiment_by_contact[contact]['scores'].append(sentiment['score'])
            sentiment_by_contact[contact]['count'] += 1
            
            # Extract words and emojis for frequency analysis
            words = re.findall(r'\b\w+\b', msg['text'].lower())
            word_frequency.update(word for word in words 
                                if word in POSITIVE_WORDS or word in NEGATIVE_WORDS)
            
            for char in msg['text']:
                if char in POSITIVE_EMOJIS or char in NEGATIVE_EMOJIS:
                    emoji_frequency[char] += 1
        
        # Calculate trends
        trends = []
        for bucket_key in sorted(sentiment_buckets.keys()):
            bucket = sentiment_buckets[bucket_key]
            if bucket['messages'] > 0:
                avg_score = bucket['total_score'] / bucket['messages']
                trends.append({
                    'period': bucket_key,
                    'message_count': bucket['messages'],
                    'average_sentiment': round(avg_score, 3),
                    'sentiment_distribution': {
                        'positive': bucket['positive'],
                        'negative': bucket['negative'],
                        'neutral': bucket['neutral']
                    },
                    'volatility': round(statistics.stdev(bucket['scores']), 3) if len(bucket['scores']) > 1 else 0
                })
        
        # Calculate overall statistics
        overall_sentiment = sum(all_scores) / len(all_scores) if all_scores else 0
        sentiment_volatility = statistics.stdev(all_scores) if len(all_scores) > 1 else 0
        
        # Find peaks and valleys
        if trends:
            sentiment_values = [t['average_sentiment'] for t in trends]
            max_idx = sentiment_values.index(max(sentiment_values))
            min_idx = sentiment_values.index(min(sentiment_values))
            
            peaks_valleys = {
                'most_positive_period': trends[max_idx]['period'],
                'most_positive_score': trends[max_idx]['average_sentiment'],
                'most_negative_period': trends[min_idx]['period'],
                'most_negative_score': trends[min_idx]['average_sentiment']
            }
        else:
            peaks_valleys = {}
        
        # Top contributors to sentiment
        top_positive_contacts = sorted(
            [(contact, sum(scores)/len(scores)) 
             for contact, data in sentiment_by_contact.items() 
             for scores in [data['scores']] if scores],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        top_negative_contacts = sorted(
            [(contact, sum(scores)/len(scores)) 
             for contact, data in sentiment_by_contact.items() 
             for scores in [data['scores']] if scores],
            key=lambda x: x[1]
        )[:5]
        
        # Generate insights
        insights = {
            'overall_sentiment': 'positive' if overall_sentiment > 0.1 else 'negative' if overall_sentiment < -0.1 else 'neutral',
            'sentiment_score': round(overall_sentiment, 3),
            'sentiment_volatility': round(sentiment_volatility, 3),
            'trend_direction': _calculate_trend_direction(trends),
            'peaks_valleys': peaks_valleys,
            'top_positive_contacts': [
                {'contact': c[0], 'score': round(c[1], 3)} 
                for c in top_positive_contacts
            ],
            'top_negative_contacts': [
                {'contact': c[0], 'score': round(c[1], 3)} 
                for c in top_negative_contacts
            ],
            'most_used_positive_words': word_frequency.most_common(10),
            'most_used_emojis': emoji_frequency.most_common(10),
            'recommendations': _generate_sentiment_recommendations(
                overall_sentiment, sentiment_volatility, trends
            )
        }
        
        return success_response({
            'time_period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat(),
                'days': (end_date - start_date).days,
                'granularity': granularity
            },
            'summary': {
                'total_messages_analyzed': len(messages),
                'overall_sentiment': insights['overall_sentiment'],
                'sentiment_score': insights['sentiment_score'],
                'trend': insights['trend_direction']
            },
            'trends': trends,
            'insights': insights
        })
        
    except Exception as e:
        logger.error(f"Error analyzing sentiment trends: {e}", exc_info=True)
        return error_response(f"Failed to analyze sentiment trends: {str(e)}")


def _calculate_trend_direction(trends: List[Dict]) -> str:
    """Calculate overall trend direction from sentiment data."""
    if len(trends) < 2:
        return "stable"
    
    # Simple linear regression on sentiment scores
    x_values = list(range(len(trends)))
    y_values = [t['average_sentiment'] for t in trends]
    
    n = len(x_values)
    if n == 0:
        return "stable"
    
    x_mean = sum(x_values) / n
    y_mean = sum(y_values) / n
    
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    denominator = sum((x - x_mean) ** 2 for x in x_values)
    
    if denominator == 0:
        return "stable"
    
    slope = numerator / denominator
    
    if slope > 0.01:
        return "improving"
    elif slope < -0.01:
        return "declining"
    else:
        return "stable"


def _generate_sentiment_recommendations(
    overall_sentiment: float,
    volatility: float,
    trends: List[Dict]
) -> List[str]:
    """Generate actionable recommendations based on sentiment analysis."""
    recommendations = []
    
    if overall_sentiment < -0.2:
        recommendations.append(
            "Consider reaching out with positive messages to improve relationship sentiment"
        )
    
    if volatility > 0.5:
        recommendations.append(
            "High emotional volatility detected - consider more consistent communication tone"
        )
    
    if trends and len(trends) > 3:
        recent_trend = sum(t['average_sentiment'] for t in trends[-3:]) / 3
        older_trend = sum(t['average_sentiment'] for t in trends[:-3]) / (len(trends) - 3)
        
        if recent_trend < older_trend - 0.2:
            recommendations.append(
                "Recent sentiment decline detected - check in on recent conversations"
            )
        elif recent_trend > older_trend + 0.2:
            recommendations.append(
                "Great improvement in recent sentiment - keep up the positive communication!"
            )
    
    if not recommendations:
        recommendations.append("Sentiment patterns appear healthy and stable")
    
    return recommendations