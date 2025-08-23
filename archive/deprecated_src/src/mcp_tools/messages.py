"""
Message-related MCP tools for the iMessage Advanced Insights server.

This module provides tools for retrieving and analyzing messages
from the iMessage database.
"""

import logging
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from collections import defaultdict, Counter
import statistics

from ..database import get_database
from ..exceptions import DatabaseError, ToolExecutionError
from ..utils.responses import error_response, paginated_response, success_response
from ..utils.sanitization import sanitize_message, sanitize_messages
from .registry import register_tool

logger = logging.getLogger(__name__)


@register_tool(
    name="get_messages",
    description="Get messages from a contact or group chat with improved formatting",
)
async def get_messages_tool(
    contact_or_chat: str,
    is_group: bool = False,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    search_term: Optional[str] = None,
    page: int = 1,
    page_size: int = 30,
) -> Dict[str, Any]:
    """
    Get messages from a contact or group chat with proper formatting.
    
    Args:
        contact_or_chat: Phone number (for contact) or chat ID/name (for group chat)
        is_group: Whether this is a group chat or individual contact
        start_date: Start date for filtering (format: YYYY-MM-DD or "X days/weeks/months ago")
        end_date: End date for filtering (format: YYYY-MM-DD or "X days/weeks/months ago")
        search_term: Optional search term to filter messages
        page: Page number for paginated results
        page_size: Number of messages per page
        
    Returns:
        Paginated messages with proper text formatting and dates
    """
    try:
        # Validate parameters
        if not contact_or_chat:
            return error_response("Contact or chat identifier is required")
        
        if page < 1:
            return error_response("Page number must be at least 1")
        
        if page_size < 1 or page_size > 100:
            return error_response("Page size must be between 1 and 100")
        
        # Get database connection
        db = await get_database()
        
        # Parse dates using our utility function
        from ..utils.decorators import parse_date
        
        start_date_obj = parse_date(start_date)
        end_date_obj = parse_date(end_date)
        
        # Get messages based on whether it's a group chat or individual contact
        if is_group:
            # Handle group chat
            result = await db.get_messages_from_chat(
                chat_id=contact_or_chat,
                start_date=start_date_obj,
                end_date=end_date_obj,
                page=page,
                page_size=page_size,
            )
        else:
            # Handle individual contact
            result = await db.get_messages_from_contact(
                phone_number=contact_or_chat,
                start_date=start_date_obj,
                end_date=end_date_obj,
                page=page,
                page_size=page_size,
            )
        
        # Get messages from the result
        messages = result.get("messages", [])
        total_count = result.get("total", 0)
        
        # Filter by search term if provided
        if search_term:
            search_lower = search_term.lower()
            messages = [msg for msg in messages if msg.get("text") and search_lower in msg["text"].lower()]
            total_count = len(messages)
        has_more = result.get("has_more", False)
        
        # Sanitize messages to protect sensitive information
        sanitized_messages = sanitize_messages(messages)
        
        # Format dates consistently
        for message in sanitized_messages:
            if "date" in message and isinstance(message["date"], (str, datetime)):
                # Convert to ISO format if it's a datetime object
                if isinstance(message["date"], datetime):
                    message["date"] = message["date"].isoformat()
        
        # Create pagination info
        return paginated_response(
            items=sanitized_messages,
            page=page,
            page_size=page_size,
            total_items=total_count,
            additional_data={
                "is_group": is_group,
                "contact_or_chat": contact_or_chat,
                "date_range": {
                    "start": start_date_obj.isoformat() if start_date_obj else None,
                    "end": end_date_obj.isoformat() if end_date_obj else None,
                },
                "has_search_filter": bool(search_term),
                "search_term": search_term,
            },
        )
    except DatabaseError as e:
        logger.error(f"Database error in get_messages_tool: {e}")
        return error_response(e)
    except Exception as e:
        logger.error(f"Unexpected error in get_messages_tool: {e}")
        logger.error(traceback.format_exc())
        return error_response(ToolExecutionError(f"Failed to get messages: {str(e)}"))


@register_tool(
    name="search_messages",
    description="Search messages across all conversations or a specific contact",
)
async def search_messages_tool(
    query: str,
    contact: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
) -> Dict[str, Any]:
    """
    Search messages across all conversations or a specific contact.
    
    Args:
        query: Search query
        contact: Optional contact to restrict search to
        start_date: Start date for search (format: YYYY-MM-DD or "X days/weeks/months ago")
        end_date: End date for search (format: YYYY-MM-DD or "X days/weeks/months ago")
        page: Page number for paginated results
        page_size: Number of messages per page
        
    Returns:
        Search results with messages matching the query
    """
    try:
        # Validate parameters
        if not query:
            return error_response("Search query is required")
        
        if page < 1:
            return error_response("Page number must be at least 1")
        
        if page_size < 1 or page_size > 100:
            return error_response("Page size must be between 1 and 100")
        
        # Get database connection
        db = await get_database()
        
        # Parse dates using our utility function
        from ..utils.decorators import parse_date
        
        start_date_obj = parse_date(start_date)
        end_date_obj = parse_date(end_date)
        
        # Search messages
        # Convert page to offset for the database method
        offset = (page - 1) * page_size
        result = await db.search_messages(
            query=query,
            contact_id=contact,
            start_date=start_date_obj,
            end_date=end_date_obj,
            limit=page_size,
            offset=offset,
        )
        
        # Get messages from the result
        messages = result.get("messages", [])
        total_count = result.get("total", 0)
        has_more = result.get("has_more", False)
        
        # Sanitize messages to protect sensitive information
        sanitized_messages = sanitize_messages(messages)
        
        # Format dates consistently
        for message in sanitized_messages:
            if "date" in message and isinstance(message["date"], (str, datetime)):
                # Convert to ISO format if it's a datetime object
                if isinstance(message["date"], datetime):
                    message["date"] = message["date"].isoformat()
        
        # Create pagination info
        return paginated_response(
            items=sanitized_messages,
            page=page,
            page_size=page_size,
            total_items=total_count,
            additional_data={
                "query": query,
                "contact": contact,
                "date_range": {
                    "start": start_date_obj.isoformat() if start_date_obj else None,
                    "end": end_date_obj.isoformat() if end_date_obj else None,
                },
                "message_match_count": len(sanitized_messages),
            },
        )
    except DatabaseError as e:
        logger.error(f"Database error in search_messages_tool: {e}")
        return error_response(e)
    except Exception as e:
        logger.error(f"Unexpected error in search_messages_tool: {e}")
        logger.error(traceback.format_exc())
        return error_response(ToolExecutionError(f"Failed to search messages: {str(e)}"))


@register_tool(
    name="analyze_messages_with_insights",
    description="Get messages with deep analytical insights"
)
async def analyze_messages_with_insights_tool(
    contact_or_chat: str,
    time_period: Optional[str] = None,
    analysis_focus: Optional[List[str]] = None,
    include_predictions: bool = True,
    page: int = 1,
    page_size: int = 30
) -> Dict[str, Any]:
    """
    Enhanced message retrieval with deep insights and analysis.
    
    This tool goes beyond simple message retrieval to provide:
    - Messages with contextual analysis
    - Conversation quality metrics
    - Emotional dynamics analysis
    - Topic evolution tracking
    - Key moment identification
    - Future conversation predictions
    - Personalized recommendations
    
    Args:
        contact_or_chat: Contact identifier (phone number, email, or handle)
        time_period: Time period to analyze (e.g., "30 days", "3 months")
        analysis_focus: Specific aspects to analyze (default: ["quality", "sentiment", "topics"])
        include_predictions: Whether to include predictive insights
        page: Page number for paginated results
        page_size: Number of messages per page
        
    Returns:
        Messages with comprehensive insights and recommendations
    """
    try:
        # Validate parameters
        if not contact_or_chat:
            return error_response("Contact or chat identifier is required")
        
        if analysis_focus is None:
            analysis_focus = ["quality", "sentiment", "topics", "patterns"]
        
        # Get database connection
        db = await get_database()
        
        # Parse time period
        from ..utils.decorators import parse_date
        start_date = parse_date(time_period) if time_period else None
        end_date = datetime.now()
        
        # Get messages
        result = await db.get_messages_from_contact(
            phone_number=contact_or_chat,
            start_date=start_date,
            end_date=end_date,
            page=page,
            page_size=page_size
        )
        
        messages = result.get("messages", [])
        total_count = result.get("total", 0)
        
        if not messages:
            return success_response({
                "messages": [],
                "insights": {
                    "status": "no_data",
                    "message": "No messages found for analysis"
                }
            })
        
        # Sanitize messages
        sanitized_messages = sanitize_messages(messages)
        
        # Perform multi-dimensional analysis
        insights = {}
        
        # 1. Conversation Quality Analysis
        if "quality" in analysis_focus:
            insights["conversation_quality"] = _analyze_conversation_quality(messages)
        
        # 2. Sentiment Analysis
        if "sentiment" in analysis_focus:
            insights["emotional_dynamics"] = _analyze_message_sentiment(messages)
        
        # 3. Topic Analysis
        if "topics" in analysis_focus:
            insights["topic_analysis"] = _analyze_message_topics(messages)
        
        # 4. Pattern Detection
        if "patterns" in analysis_focus:
            insights["communication_patterns"] = _analyze_communication_patterns(messages)
        
        # 5. Key Moments Identification
        insights["key_moments"] = _identify_key_moments(messages)
        
        # 6. Predictions (if requested)
        predictions = None
        if include_predictions:
            predictions = _generate_message_predictions(messages, insights)
        
        # 7. Generate Recommendations
        recommendations = _generate_message_recommendations(insights, predictions)
        
        # Format response
        return success_response({
            "messages": sanitized_messages,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_messages": total_count,
                "has_more": total_count > page * page_size
            },
            "insights": insights,
            "predictions": predictions,
            "recommendations": recommendations,
            "metadata": {
                "time_period": time_period or "all_time",
                "analysis_focus": analysis_focus,
                "contact": contact_or_chat
            }
        })
        
    except DatabaseError as e:
        logger.error(f"Database error in analyze_messages_with_insights_tool: {e}")
        return error_response(e)
    except Exception as e:
        logger.error(f"Unexpected error in analyze_messages_with_insights_tool: {e}")
        logger.error(traceback.format_exc())
        return error_response(ToolExecutionError(f"Failed to analyze messages: {str(e)}"))


def _analyze_conversation_quality(messages: List[Dict]) -> Dict[str, Any]:
    """Analyze the quality and depth of conversations."""
    
    # Calculate message length statistics
    message_lengths = [len(msg.get("text", "")) for msg in messages if msg.get("text")]
    
    if not message_lengths:
        return {"status": "no_text_messages"}
    
    avg_length = statistics.mean(message_lengths)
    
    # Count questions and substantive messages
    questions = sum(1 for msg in messages if msg.get("text", "").strip().endswith("?"))
    long_messages = sum(1 for length in message_lengths if length > 100)
    
    # Calculate quality score
    quality_score = min(100, int(
        (avg_length / 50) * 30 +  # Length component
        (questions / len(messages)) * 200 +  # Question component
        (long_messages / len(messages)) * 170  # Depth component
    ))
    
    # Determine quality category
    if quality_score >= 70:
        quality_category = "deep_meaningful"
        interpretation = "Rich, engaging conversations with depth"
    elif quality_score >= 40:
        quality_category = "moderate"
        interpretation = "Balanced mix of casual and meaningful exchanges"
    else:
        quality_category = "light"
        interpretation = "Primarily brief, casual interactions"
    
    return {
        "quality_score": quality_score,
        "quality_category": quality_category,
        "metrics": {
            "average_message_length": round(avg_length, 1),
            "question_ratio": round(questions / len(messages), 3),
            "substantive_messages": long_messages,
            "total_analyzed": len(messages)
        },
        "interpretation": interpretation
    }


def _analyze_message_sentiment(messages: List[Dict]) -> Dict[str, Any]:
    """Analyze emotional dynamics in messages."""
    
    positive_words = {"love", "great", "awesome", "happy", "excited", "wonderful", "amazing", 
                     "good", "fantastic", "excellent", "perfect", "beautiful", "lol", "haha"}
    negative_words = {"sad", "angry", "frustrated", "upset", "worried", "stressed", "bad", 
                     "terrible", "awful", "horrible", "hate", "annoyed", "disappointed"}
    support_words = {"help", "support", "there", "understand", "sorry", "care", "hope", "hugs"}
    
    sentiment_timeline = []
    overall_positive = 0
    overall_negative = 0
    support_count = 0
    
    for msg in messages:
        if not msg.get("text"):
            continue
        
        text_lower = msg["text"].lower()
        words = set(text_lower.split())
        
        # Count sentiment words
        pos_count = len(words & positive_words)
        neg_count = len(words & negative_words)
        sup_count = len(words & support_words)
        
        overall_positive += pos_count
        overall_negative += neg_count
        support_count += sup_count
        
        # Track sentiment over time
        if pos_count > 0 or neg_count > 0:
            sentiment_timeline.append({
                "date": msg["date"],
                "sentiment": "positive" if pos_count > neg_count else "negative",
                "score": pos_count - neg_count
            })
    
    # Calculate overall sentiment
    total_emotional_words = overall_positive + overall_negative
    sentiment_balance = (overall_positive - overall_negative) / max(total_emotional_words, 1)
    
    # Determine emotional tone
    if sentiment_balance > 0.3:
        tone = "very_positive"
        interpretation = "Conversations are filled with joy and positivity"
    elif sentiment_balance > 0:
        tone = "positive"
        interpretation = "Generally upbeat and optimistic exchanges"
    elif sentiment_balance > -0.3:
        tone = "neutral"
        interpretation = "Balanced emotional expression"
    else:
        tone = "concerning"
        interpretation = "Notable presence of negative emotions - may need support"
    
    return {
        "emotional_tone": tone,
        "sentiment_balance": round(sentiment_balance, 3),
        "metrics": {
            "positive_expressions": overall_positive,
            "negative_expressions": overall_negative,
            "support_expressions": support_count,
            "emotional_message_ratio": round(len(sentiment_timeline) / len(messages), 3)
        },
        "sentiment_trend": _calculate_sentiment_trend(sentiment_timeline),
        "interpretation": interpretation
    }


def _calculate_sentiment_trend(timeline: List[Dict]) -> str:
    """Calculate trend in sentiment over time."""
    if len(timeline) < 5:
        return "insufficient_data"
    
    # Compare recent to historical sentiment
    midpoint = len(timeline) // 2
    recent_scores = [item["score"] for item in timeline[midpoint:]]
    historical_scores = [item["score"] for item in timeline[:midpoint]]
    
    recent_avg = statistics.mean(recent_scores) if recent_scores else 0
    historical_avg = statistics.mean(historical_scores) if historical_scores else 0
    
    if recent_avg > historical_avg + 0.5:
        return "improving"
    elif recent_avg < historical_avg - 0.5:
        return "declining"
    else:
        return "stable"


def _analyze_message_topics(messages: List[Dict]) -> Dict[str, Any]:
    """Analyze topics discussed in messages."""
    
    # Extract words as simplified topics
    all_words = []
    topic_evolution = defaultdict(list)
    
    for msg in messages:
        if msg.get("text"):
            # Extract meaningful words (longer than 4 chars)
            words = [w.lower() for w in msg["text"].split() 
                    if len(w) > 4 and w.isalpha()]
            all_words.extend(words)
            
            # Track topic evolution over time
            for word in set(words):
                topic_evolution[word].append(msg["date"])
    
    # Count topic frequency
    topic_counts = Counter(all_words)
    top_topics = topic_counts.most_common(10)
    
    # Calculate topic diversity
    unique_topics = len(set(all_words))
    diversity_score = min(100, (unique_topics / len(all_words) * 100)) if all_words else 0
    
    # Identify emerging topics (recent but not historical)
    if len(messages) > 20:
        recent_messages = messages[-10:]
        historical_messages = messages[:-10]
        
        recent_words = set()
        historical_words = set()
        
        for msg in recent_messages:
            if msg.get("text"):
                recent_words.update(w.lower() for w in msg["text"].split() 
                                  if len(w) > 4 and w.isalpha())
        
        for msg in historical_messages:
            if msg.get("text"):
                historical_words.update(w.lower() for w in msg["text"].split() 
                                      if len(w) > 4 and w.isalpha())
        
        emerging_topics = list(recent_words - historical_words)[:5]
    else:
        emerging_topics = []
    
    return {
        "top_topics": [{"topic": topic, "frequency": count} for topic, count in top_topics],
        "topic_diversity": {
            "unique_topics": unique_topics,
            "diversity_score": round(diversity_score, 1),
            "interpretation": "High variety" if diversity_score > 30 else "Focused discussions"
        },
        "emerging_topics": emerging_topics,
        "topic_consistency": _calculate_topic_consistency(topic_evolution)
    }


def _calculate_topic_consistency(topic_evolution: Dict[str, List]) -> str:
    """Determine how consistent topics are over time."""
    if not topic_evolution:
        return "no_topics"
    
    # Find topics that appear throughout the conversation
    consistent_topics = []
    for topic, dates in topic_evolution.items():
        if len(dates) > 3:  # Appears multiple times
            consistent_topics.append(topic)
    
    if len(consistent_topics) > 5:
        return "very_consistent"
    elif len(consistent_topics) > 2:
        return "moderately_consistent"
    else:
        return "varied"


def _analyze_communication_patterns(messages: List[Dict]) -> Dict[str, Any]:
    """Analyze patterns in communication timing and frequency."""
    
    # Group messages by hour and day
    hourly_distribution = defaultdict(int)
    daily_distribution = defaultdict(int)
    response_times = []
    
    for i, msg in enumerate(messages):
        if msg.get("date"):
            msg_time = datetime.fromisoformat(msg["date"])
            hourly_distribution[msg_time.hour] += 1
            daily_distribution[msg_time.weekday()] += 1
            
            # Calculate response times
            if i > 0:
                prev_msg = messages[i-1]
                if prev_msg.get("is_from_me") != msg.get("is_from_me"):
                    prev_time = datetime.fromisoformat(prev_msg["date"])
                    response_minutes = (msg_time - prev_time).total_seconds() / 60
                    if 0 < response_minutes < 1440:  # Within 24 hours
                        response_times.append(response_minutes)
    
    # Find peak communication times
    peak_hours = sorted(hourly_distribution.items(), key=lambda x: x[1], reverse=True)[:3]
    peak_days = sorted(daily_distribution.items(), key=lambda x: x[1], reverse=True)[:3]
    
    # Calculate average response time
    avg_response_time = statistics.mean(response_times) if response_times else 0
    
    # Determine communication style
    if avg_response_time < 5:
        response_style = "immediate"
        style_interpretation = "Very quick, real-time conversations"
    elif avg_response_time < 30:
        response_style = "prompt"
        style_interpretation = "Responsive and engaged communication"
    elif avg_response_time < 120:
        response_style = "relaxed"
        style_interpretation = "Thoughtful, asynchronous exchanges"
    else:
        response_style = "delayed"
        style_interpretation = "Sporadic check-ins and updates"
    
    return {
        "peak_times": [
            {"hour": h, "count": c, "period": _hour_to_period_name(h)} 
            for h, c in peak_hours
        ],
        "peak_days": [
            {"day": _day_name(d), "count": c} 
            for d, c in peak_days
        ],
        "response_behavior": {
            "average_response_minutes": round(avg_response_time, 1),
            "response_style": response_style,
            "interpretation": style_interpretation
        },
        "message_frequency": {
            "total_messages": len(messages),
            "daily_average": round(len(messages) / max((messages[-1]["date"][:10] != messages[0]["date"][:10]), 1), 1)
        }
    }


def _hour_to_period_name(hour: int) -> str:
    """Convert hour to period name."""
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 22:
        return "evening"
    else:
        return "night"


def _day_name(day: int) -> str:
    """Convert day number to name."""
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    return days[day]


def _identify_key_moments(messages: List[Dict]) -> List[Dict[str, Any]]:
    """Identify key moments in the conversation."""
    key_moments = []
    
    # Look for long messages (potentially important)
    for i, msg in enumerate(messages):
        if msg.get("text") and len(msg["text"]) > 200:
            key_moments.append({
                "type": "deep_share",
                "date": msg["date"],
                "preview": msg["text"][:100] + "...",
                "significance": "Extended personal sharing"
            })
    
    # Look for emotional moments
    emotional_words = {"love", "sorry", "miss", "proud", "worried", "excited", "grateful"}
    for msg in messages:
        if msg.get("text"):
            text_lower = msg["text"].lower()
            if any(word in text_lower for word in emotional_words):
                key_moments.append({
                    "type": "emotional_expression",
                    "date": msg["date"],
                    "preview": msg["text"][:100],
                    "significance": "Important emotional moment"
                })
                break  # Limit to avoid too many
    
    # Sort by date and return top moments
    key_moments.sort(key=lambda x: x["date"], reverse=True)
    return key_moments[:5]


def _generate_message_predictions(messages: List[Dict], insights: Dict) -> Dict[str, Any]:
    """Generate predictions based on message analysis."""
    
    predictions = {
        "next_interaction": {},
        "conversation_trajectory": {},
        "topic_predictions": []
    }
    
    # Predict next interaction based on patterns
    if insights.get("communication_patterns"):
        patterns = insights["communication_patterns"]
        avg_response = patterns["response_behavior"]["average_response_minutes"]
        
        # Estimate next message time
        last_message_date = datetime.fromisoformat(messages[-1]["date"])
        hours_since = (datetime.now() - last_message_date).total_seconds() / 3600
        
        if hours_since > avg_response / 60 * 2:
            predictions["next_interaction"] = {
                "likelihood": "overdue",
                "explanation": f"Usually respond within {avg_response:.0f} minutes",
                "recommendation": "Consider reaching out"
            }
        else:
            predictions["next_interaction"] = {
                "likelihood": "soon",
                "explanation": f"Based on {avg_response:.0f} minute average response time"
            }
    
    # Predict conversation trajectory
    quality_score = insights.get("conversation_quality", {}).get("quality_score", 50)
    sentiment_trend = insights.get("emotional_dynamics", {}).get("sentiment_trend", "stable")
    
    if quality_score > 70 and sentiment_trend in ["stable", "improving"]:
        predictions["conversation_trajectory"] = {
            "direction": "deepening",
            "confidence": 0.8,
            "explanation": "High quality exchanges with positive emotional dynamics"
        }
    elif quality_score < 40 or sentiment_trend == "declining":
        predictions["conversation_trajectory"] = {
            "direction": "needs_attention",
            "confidence": 0.7,
            "explanation": "Conversations becoming less engaged or more negative"
        }
    else:
        predictions["conversation_trajectory"] = {
            "direction": "maintaining",
            "confidence": 0.6,
            "explanation": "Steady, consistent communication patterns"
        }
    
    # Predict emerging topics
    if insights.get("topic_analysis", {}).get("emerging_topics"):
        emerging = insights["topic_analysis"]["emerging_topics"]
        predictions["topic_predictions"] = [
            f"Continued discussion about: {', '.join(emerging[:3])}"
        ]
    
    return predictions


def _generate_message_recommendations(insights: Dict, predictions: Optional[Dict]) -> List[Dict[str, str]]:
    """Generate actionable recommendations based on analysis."""
    
    recommendations = []
    
    # Based on conversation quality
    quality_score = insights.get("conversation_quality", {}).get("quality_score", 50)
    if quality_score < 40:
        recommendations.append({
            "action": "Deepen conversations with open-ended questions",
            "reason": "Current exchanges are primarily brief",
            "priority": "medium",
            "impact": "Build stronger connection through meaningful dialogue"
        })
    
    # Based on emotional dynamics
    emotional_tone = insights.get("emotional_dynamics", {}).get("emotional_tone", "neutral")
    if emotional_tone == "concerning":
        recommendations.append({
            "action": "Offer support and check in on wellbeing",
            "reason": "Negative emotional patterns detected",
            "priority": "high",
            "impact": "Show care and strengthen relationship"
        })
    elif emotional_tone == "very_positive":
        recommendations.append({
            "action": "Maintain positive momentum with shared activities",
            "reason": "Relationship is in a great place",
            "priority": "low",
            "impact": "Further strengthen already positive dynamic"
        })
    
    # Based on response patterns
    response_style = insights.get("communication_patterns", {}).get("response_behavior", {}).get("response_style", "")
    if response_style == "delayed":
        recommendations.append({
            "action": "Set regular check-in times",
            "reason": "Communication is sporadic",
            "priority": "medium",
            "impact": "Create consistent connection points"
        })
    
    # Based on predictions
    if predictions and predictions.get("next_interaction", {}).get("likelihood") == "overdue":
        recommendations.append({
            "action": "Send a message now",
            "reason": predictions["next_interaction"]["explanation"],
            "priority": "high",
            "impact": "Maintain communication rhythm"
        })
    
    # Sort by priority
    priority_order = {"high": 0, "medium": 1, "low": 2}
    recommendations.sort(key=lambda x: priority_order.get(x["priority"], 3))
    
    return recommendations[:4]  # Return top 4 recommendations
