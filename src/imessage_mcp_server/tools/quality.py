"""
Conversation Quality Score tool for comprehensive relationship analysis.

Provides multi-dimensional scoring of conversation quality including depth,
balance, emotional health, and consistency metrics.
"""

import asyncio
import logging
import statistics
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from imessage_mcp_server.db import get_database
from imessage_mcp_server.privacy import apply_privacy_filters, hash_contact_id

logger = logging.getLogger(__name__)


async def conversation_quality_tool(
    contact_id: str,
    time_period: str = "30d",
    include_recommendations: bool = True,
    db_path: str = "~/Library/Messages/chat.db",
    redact: bool = True,
) -> Dict[str, Any]:
    """
    Calculate comprehensive conversation quality score.
    
    Analyzes multiple dimensions of conversation health:
    - Depth: Message length, vocabulary richness, question frequency
    - Balance: Initiation ratio, word count balance, response patterns
    - Emotion: Sentiment positivity, emotional range, support language
    - Consistency: Communication regularity, response time stability
    
    Args:
        contact_id: Contact identifier (phone/email)
        time_period: Analysis period (e.g., "30d", "90d", "6m")
        include_recommendations: Whether to include improvement suggestions
        db_path: Path to iMessage database
        redact: Whether to apply privacy filters
        
    Returns:
        Dict containing overall score, dimension breakdown, and recommendations
    """
    try:
        # Parse time period
        days = _parse_time_period(time_period)
        
        # Expand path
        db_path = Path(db_path).expanduser()
        db = await get_database(str(db_path))
        
        # Get messages for analysis
        messages = await _fetch_messages_for_quality(db, contact_id, days)
        
        if not messages:
            return {
                "error": "No messages found for quality analysis",
                "contact_id": hash_contact_id(contact_id) if redact else contact_id,
                "time_period": time_period,
            }
        
        # Calculate dimension scores
        depth_analysis = await _calculate_depth_score(messages)
        balance_analysis = await _calculate_balance_score(messages)
        emotion_analysis = await _calculate_emotion_score(messages)
        consistency_analysis = await _calculate_consistency_score(messages)
        
        # Calculate overall score
        overall_score = (
            depth_analysis["score"] * 0.25 +
            balance_analysis["score"] * 0.25 +
            emotion_analysis["score"] * 0.25 +
            consistency_analysis["score"] * 0.25
        )
        
        # Determine grade and trajectory
        grade = _score_to_grade(overall_score)
        trajectory = await _calculate_trajectory(db, contact_id, overall_score, days)
        
        # Generate recommendations if requested
        recommendations = []
        if include_recommendations:
            recommendations = _generate_recommendations(
                depth_analysis,
                balance_analysis,
                emotion_analysis,
                consistency_analysis,
            )
        
        result = {
            "contact_id": hash_contact_id(contact_id) if redact else contact_id,
            "time_period": time_period,
            "overall_score": round(overall_score, 1),
            "grade": grade,
            "trajectory": trajectory,
            "dimensions": {
                "depth": {
                    "score": round(depth_analysis["score"], 1),
                    "insights": depth_analysis["insights"],
                },
                "balance": {
                    "score": round(balance_analysis["score"], 1),
                    "insights": balance_analysis["insights"],
                },
                "emotion": {
                    "score": round(emotion_analysis["score"], 1),
                    "insights": emotion_analysis["insights"],
                },
                "consistency": {
                    "score": round(consistency_analysis["score"], 1),
                    "insights": consistency_analysis["insights"],
                },
            },
            "action_items": recommendations[:3],  # Top 3 recommendations
            "analysis_details": {
                "messages_analyzed": len(messages),
                "date_range": f"{messages[0]['date']} to {messages[-1]['date']}",
            },
        }
        
        if redact:
            result = apply_privacy_filters(result)
            
        return result
        
    except Exception as e:
        logger.error(f"Error calculating conversation quality: {e}")
        return {
            "error": str(e),
            "error_type": "calculation_error",
        }


async def _fetch_messages_for_quality(
    db: Any, contact_id: str, days: int
) -> List[Dict[str, Any]]:
    """Fetch messages with metadata for quality analysis."""
    query = """
    SELECT 
        m.text,
        m.is_from_me,
        m.date,
        LENGTH(m.text) as length,
        CASE WHEN m.text LIKE '%?%' THEN 1 ELSE 0 END as has_question,
        CASE WHEN LENGTH(m.text) > 200 THEN 1 ELSE 0 END as is_long,
        h.id as handle_id
    FROM message m
    JOIN handle h ON m.handle_id = h.ROWID
    WHERE h.id = ?
    AND m.text IS NOT NULL
    AND m.date > (strftime('%s', 'now') - ?) * 1000000000
    ORDER BY m.date
    """
    
    seconds = days * 86400
    cursor = await db.execute(query, [contact_id, seconds])
    rows = await cursor.fetchall()
    
    messages = []
    for row in rows:
        messages.append({
            "text": row[0],
            "is_from_me": row[1],
            "date": datetime.fromtimestamp(row[2] / 1000000000 + 978307200),
            "length": row[3],
            "has_question": row[4],
            "is_long": row[5],
        })
    
    return messages


async def _calculate_depth_score(messages: List[Dict]) -> Dict[str, Any]:
    """Calculate conversation depth metrics."""
    # Message length analysis
    avg_length = statistics.mean([m["length"] for m in messages])
    long_messages = sum(1 for m in messages if m["is_long"])
    
    # Question frequency
    questions = sum(1 for m in messages if m["has_question"])
    question_ratio = questions / len(messages) if messages else 0
    
    # Vocabulary richness (unique words / total words)
    all_words = []
    for msg in messages:
        words = msg["text"].lower().split()
        all_words.extend(words)
    
    vocab_richness = len(set(all_words)) / len(all_words) if all_words else 0
    
    # Calculate depth score (0-100)
    depth_score = (
        min(100, avg_length / 2) * 0.3 +  # Length component
        min(100, (long_messages / len(messages)) * 200) * 0.2 +  # Long messages
        (question_ratio * 100) * 0.2 +  # Questions
        (vocab_richness * 100) * 0.3  # Vocabulary
    )
    
    insights = []
    if avg_length < 50:
        insights.append("Messages tend to be brief - try sharing more details")
    if question_ratio < 0.1:
        insights.append("Ask more questions to deepen engagement")
    if vocab_richness < 0.3:
        insights.append("Conversations could benefit from more varied topics")
    
    return {
        "score": depth_score,
        "metrics": {
            "avg_message_length": round(avg_length, 1),
            "long_messages_pct": round((long_messages / len(messages)) * 100, 1),
            "question_ratio": round(question_ratio, 2),
            "vocabulary_richness": round(vocab_richness, 2),
        },
        "insights": insights,
    }


async def _calculate_balance_score(messages: List[Dict]) -> Dict[str, Any]:
    """Calculate conversation balance metrics."""
    my_messages = [m for m in messages if m["is_from_me"]]
    their_messages = [m for m in messages if not m["is_from_me"]]
    
    # Initiation patterns (simplified - who sends first after gaps)
    initiation_ratio = len(my_messages) / len(messages) if messages else 0.5
    
    # Word count balance
    my_words = sum(len(m["text"].split()) for m in my_messages)
    their_words = sum(len(m["text"].split()) for m in their_messages)
    total_words = my_words + their_words
    word_balance = min(my_words, their_words) / max(my_words, their_words) if total_words > 0 else 0
    
    # Response rate (simplified)
    response_rate = min(len(my_messages), len(their_messages)) / max(len(my_messages), len(their_messages))
    
    # Calculate balance score
    balance_score = (
        (1 - abs(0.5 - initiation_ratio)) * 100 * 0.3 +  # Initiation balance
        word_balance * 100 * 0.35 +  # Word count balance
        response_rate * 100 * 0.35  # Response balance
    )
    
    insights = []
    if initiation_ratio > 0.7:
        insights.append("You initiate most conversations - good leadership")
    elif initiation_ratio < 0.3:
        insights.append("They usually start conversations - try initiating more")
    
    if word_balance < 0.5:
        insights.append("Message lengths are imbalanced - aim for equality")
    
    return {
        "score": balance_score,
        "metrics": {
            "initiation_ratio": round(initiation_ratio, 2),
            "word_balance": round(word_balance, 2),
            "response_rate": round(response_rate, 2),
            "my_messages": len(my_messages),
            "their_messages": len(their_messages),
        },
        "insights": insights,
    }


async def _calculate_emotion_score(messages: List[Dict]) -> Dict[str, Any]:
    """Calculate emotional health metrics."""
    # Simple sentiment analysis based on keywords
    positive_words = {"love", "great", "awesome", "happy", "excited", "wonderful", "lol", "haha", "😊", "❤️", "🎉"}
    negative_words = {"sad", "sorry", "angry", "frustrated", "disappointed", "hate", "😢", "😡", "👎"}
    support_words = {"help", "support", "there for you", "understand", "feel", "care"}
    
    positive_count = 0
    negative_count = 0
    support_count = 0
    
    for msg in messages:
        text_lower = msg["text"].lower()
        positive_count += sum(1 for word in positive_words if word in text_lower)
        negative_count += sum(1 for word in negative_words if word in text_lower)
        support_count += sum(1 for word in support_words if word in text_lower)
    
    # Calculate metrics
    total_emotional_words = positive_count + negative_count
    positivity_ratio = positive_count / total_emotional_words if total_emotional_words > 0 else 0.5
    emotional_expression = total_emotional_words / len(messages) if messages else 0
    support_frequency = support_count / len(messages) if messages else 0
    
    # Calculate emotion score
    emotion_score = (
        positivity_ratio * 100 * 0.4 +  # Positive sentiment
        min(100, emotional_expression * 50) * 0.3 +  # Emotional expression
        min(100, support_frequency * 200) * 0.3  # Support language
    )
    
    insights = []
    if positivity_ratio < 0.5:
        insights.append("Conversations could be more positive")
    if emotional_expression < 0.2:
        insights.append("Express emotions more openly")
    if support_frequency < 0.05:
        insights.append("Show more emotional support")
    
    return {
        "score": emotion_score,
        "metrics": {
            "positivity_ratio": round(positivity_ratio, 2),
            "emotional_expression": round(emotional_expression, 2),
            "support_frequency": round(support_frequency, 2),
        },
        "insights": insights,
    }


async def _calculate_consistency_score(messages: List[Dict]) -> Dict[str, Any]:
    """Calculate communication consistency metrics."""
    if len(messages) < 2:
        return {
            "score": 50,
            "metrics": {},
            "insights": ["Not enough data for consistency analysis"],
        }
    
    # Calculate message gaps
    gaps = []
    for i in range(1, len(messages)):
        gap = (messages[i]["date"] - messages[i-1]["date"]).total_seconds() / 3600  # Hours
        if gap < 24 * 7:  # Only consider gaps less than a week
            gaps.append(gap)
    
    if not gaps:
        return {
            "score": 50,
            "metrics": {},
            "insights": ["Communication pattern unclear"],
        }
    
    # Calculate consistency metrics
    avg_gap = statistics.mean(gaps)
    gap_std = statistics.stdev(gaps) if len(gaps) > 1 else 0
    regularity = 1 - (gap_std / avg_gap) if avg_gap > 0 else 0
    
    # Daily message count
    date_range = (messages[-1]["date"] - messages[0]["date"]).days or 1
    daily_avg = len(messages) / date_range
    
    # Calculate consistency score
    consistency_score = (
        max(0, regularity) * 100 * 0.5 +  # Gap regularity
        min(100, daily_avg * 10) * 0.5  # Message frequency
    )
    
    insights = []
    if regularity < 0.3:
        insights.append("Communication timing is irregular")
    if daily_avg < 0.5:
        insights.append("Try to maintain more frequent contact")
    elif daily_avg > 10:
        insights.append("Very active communication - well done!")
    
    return {
        "score": consistency_score,
        "metrics": {
            "avg_gap_hours": round(avg_gap, 1),
            "regularity": round(regularity, 2),
            "daily_avg_messages": round(daily_avg, 1),
        },
        "insights": insights,
    }


def _parse_time_period(period: str) -> int:
    """Parse time period string to days."""
    if period.endswith("d"):
        return int(period[:-1])
    elif period.endswith("w"):
        return int(period[:-1]) * 7
    elif period.endswith("m"):
        return int(period[:-1]) * 30
    elif period.endswith("y"):
        return int(period[:-1]) * 365
    else:
        return 30  # Default to 30 days


def _score_to_grade(score: float) -> str:
    """Convert numeric score to letter grade."""
    if score >= 93:
        return "A"
    elif score >= 90:
        return "A-"
    elif score >= 87:
        return "B+"
    elif score >= 83:
        return "B"
    elif score >= 80:
        return "B-"
    elif score >= 77:
        return "C+"
    elif score >= 73:
        return "C"
    elif score >= 70:
        return "C-"
    elif score >= 67:
        return "D+"
    elif score >= 63:
        return "D"
    else:
        return "F"


async def _calculate_trajectory(
    db: Any, contact_id: str, current_score: float, days: int
) -> str:
    """Calculate score trajectory by comparing to previous period."""
    # Get previous period score (simplified - would need to calculate)
    # For now, return static trajectory
    return "stable"  # Options: improving, declining, stable


def _generate_recommendations(
    depth: Dict, balance: Dict, emotion: Dict, consistency: Dict
) -> List[str]:
    """Generate personalized recommendations based on scores."""
    recommendations = []
    
    # Collect all insights
    all_insights = []
    all_insights.extend([(depth["score"], insight) for insight in depth["insights"]])
    all_insights.extend([(balance["score"], insight) for insight in balance["insights"]])
    all_insights.extend([(emotion["score"], insight) for insight in emotion["insights"]])
    all_insights.extend([(consistency["score"], insight) for insight in consistency["insights"]])
    
    # Sort by score (lowest first) to prioritize areas needing improvement
    all_insights.sort(key=lambda x: x[0])
    
    # Take top recommendations
    for _, insight in all_insights[:3]:
        recommendations.append(insight)
    
    # Add positive reinforcement if doing well
    if not recommendations:
        recommendations.append("Your conversation quality is excellent - keep it up!")
    
    return recommendations