"""
Topic analysis tools for the iMessage Advanced Insights server.

This module provides tools for extracting and analyzing topics from iMessage
conversations, including topic extraction, sentiment analysis, and trends.
"""

import logging
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from ..database import get_database
from ..exceptions import DatabaseError, ToolExecutionError
from ..utils.responses import error_response, paginated_response, success_response
from ..utils.sanitization import sanitize_contact_info, sanitize_analysis_result
from .registry import register_tool

logger = logging.getLogger(__name__)


@register_tool(
    name="analyze_conversation_topics",
    description="Extract and analyze topics from conversations with a contact",
)
async def analyze_conversation_topics_tool(
    contact_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    topic_count: int = 5,
    include_sentiment: bool = True,
) -> Dict[str, Any]:
    """
    Extract and analyze topics from conversations with a contact.
    
    Args:
        contact_id: Contact identifier (phone number, email, or handle)
        start_date: Start date for analysis (format: YYYY-MM-DD or "X days/weeks/months ago")
        end_date: End date for analysis (format: YYYY-MM-DD or "X days/weeks/months ago")
        topic_count: Number of top topics to return
        include_sentiment: Whether to include sentiment analysis per topic
        
    Returns:
        Dictionary with topic analysis results
    """
    try:
        # Validate parameters
        if not contact_id:
            return error_response("Contact identifier is required")
        
        if topic_count < 1 or topic_count > 20:
            return error_response("Topic count must be between 1 and 20")
        
        # Get database connection
        db = await get_database()
        
        # Parse dates using our utility function
        from ..utils.decorators import parse_date
        start_date_obj = parse_date(start_date)
        end_date_obj = parse_date(end_date)
        
        # Get contact info
        contact_info = await db.get_contact_info(contact_id=contact_id)
        sanitized_contact = sanitize_contact_info(contact_info)
        
        # Get topic analysis
        topics = await db.get_conversation_topics(
            contact_id=contact_id,
            start_date=start_date_obj,
            end_date=end_date_obj,
            topic_count=topic_count,
            include_sentiment=include_sentiment,
        )
        
        # Sanitize and format the results
        sanitized_topics = []
        for topic in topics.get("topics", []):
            sanitized_topic = {
                "topic": topic.get("topic"),
                "keywords": topic.get("keywords", []),
                "message_count": topic.get("message_count", 0),
                "percentage": topic.get("percentage", 0),
            }
            
            # Add sentiment if included
            if include_sentiment:
                sanitized_topic["sentiment"] = {
                    "score": topic.get("sentiment", {}).get("score", 0),
                    "breakdown": {
                        "positive": topic.get("sentiment", {}).get("positive", 0),
                        "neutral": topic.get("sentiment", {}).get("neutral", 0),
                        "negative": topic.get("sentiment", {}).get("negative", 0),
                    },
                }
            
            sanitized_topics.append(sanitized_topic)
        
        # Prepare the result
        result = {
            "contact": sanitized_contact,
            "date_range": {
                "start": start_date_obj.isoformat() if start_date_obj else None,
                "end": end_date_obj.isoformat() if end_date_obj else None,
            },
            "topics": sanitized_topics,
            "total_messages_analyzed": topics.get("total_messages", 0),
            "uncategorized_percentage": topics.get("uncategorized_percentage", 0),
        }
        
        # Add topic relationships if available
        if "topic_relationships" in topics:
            result["topic_relationships"] = topics["topic_relationships"]
        
        # Sanitize the overall result
        sanitized_result = sanitize_analysis_result(result)
        
        return success_response(sanitized_result)
    except DatabaseError as e:
        logger.error(f"Database error in analyze_conversation_topics_tool: {e}")
        return error_response(e)
    except Exception as e:
        logger.error(f"Unexpected error in analyze_conversation_topics_tool: {e}")
        logger.error(traceback.format_exc())
        return error_response(ToolExecutionError(f"Failed to analyze conversation topics: {str(e)}"))


@register_tool(
    name="get_topic_trends",
    description="Analyze topic trends over time in conversations",
)
async def get_topic_trends_tool(
    contact_id: Optional[str] = None,
    topic: Optional[str] = None,
    interval: str = "month",  # day, week, month
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    topic_count: int = 5,
) -> Dict[str, Any]:
    """
    Analyze topic trends over time in conversations.
    
    Args:
        contact_id: Optional contact identifier (if None, analyzes all conversations)
        topic: Optional specific topic to analyze (if None, returns top topics)
        interval: Time interval for trend analysis (day, week, month)
        start_date: Start date for analysis (format: YYYY-MM-DD or "X days/weeks/months ago")
        end_date: End date for analysis (format: YYYY-MM-DD or "X days/weeks/months ago")
        topic_count: Number of top topics to return when topic is None
        
    Returns:
        Dictionary with topic trend analysis
    """
    try:
        # Validate interval
        if interval not in ["day", "week", "month"]:
            return error_response("Interval must be 'day', 'week', or 'month'")
        
        if topic_count < 1 or topic_count > 20:
            return error_response("Topic count must be between 1 and 20")
        
        # Get database connection
        db = await get_database()
        
        # Parse dates using our utility function
        from ..utils.decorators import parse_date
        start_date_obj = parse_date(start_date)
        end_date_obj = parse_date(end_date)
        
        # Get contact info if contact_id is provided
        contact_data = None
        if contact_id:
            contact_info = await db.get_contact_info(contact_id=contact_id)
            contact_data = sanitize_contact_info(contact_info)
        
        # Get topic trends
        trend_data = await db.get_topic_trends(
            contact_id=contact_id,
            topic=topic,
            interval=interval,
            start_date=start_date_obj,
            end_date=end_date_obj,
            topic_count=topic_count,
        )
        
        # Format the trend data
        formatted_trends = []
        for trend_point in trend_data.get("trends", []):
            formatted_point = {
                "date": trend_point.get("date").isoformat() if isinstance(trend_point.get("date"), datetime) else trend_point.get("date"),
                "topics": {},
            }
            
            # Add topic data for each topic in this time interval
            for topic_name, topic_data in trend_point.get("topics", {}).items():
                formatted_point["topics"][topic_name] = {
                    "message_count": topic_data.get("message_count", 0),
                    "percentage": topic_data.get("percentage", 0),
                    "sentiment_score": topic_data.get("sentiment_score", 0),
                }
            
            formatted_trends.append(formatted_point)
        
        # Prepare the result
        result = {
            "date_range": {
                "start": start_date_obj.isoformat() if start_date_obj else None,
                "end": end_date_obj.isoformat() if end_date_obj else None,
            },
            "interval": interval,
            "trend_data": formatted_trends,
            "total_periods": len(formatted_trends),
            "total_messages_analyzed": trend_data.get("total_messages", 0),
        }
        
        # Add contact info if provided
        if contact_data:
            result["contact"] = contact_data
        
        # Add specific topic info if provided
        if topic:
            result["topic"] = topic
            
            # Add topic emergence data if available
            if "topic_emergence" in trend_data:
                result["topic_emergence"] = {
                    "first_appearance": trend_data["topic_emergence"].get("first_appearance"),
                    "peak_date": trend_data["topic_emergence"].get("peak_date"),
                    "peak_percentage": trend_data["topic_emergence"].get("peak_percentage", 0),
                    "growth_rate": trend_data["topic_emergence"].get("growth_rate", 0),
                }
        else:
            # Add overall topic ranking if no specific topic
            result["topic_ranking"] = trend_data.get("topic_ranking", [])
        
        # Sanitize the overall result
        sanitized_result = sanitize_analysis_result(result)
        
        return success_response(sanitized_result)
    except DatabaseError as e:
        logger.error(f"Database error in get_topic_trends_tool: {e}")
        return error_response(e)
    except Exception as e:
        logger.error(f"Unexpected error in get_topic_trends_tool: {e}")
        logger.error(traceback.format_exc())
        return error_response(ToolExecutionError(f"Failed to get topic trends: {str(e)}"))


@register_tool(
    name="analyze_topic_sentiment",
    description="Analyze sentiment for specific topics across conversations",
)
async def analyze_topic_sentiment_tool(
    topic: str,
    contact_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze sentiment for specific topics across conversations.
    
    Args:
        topic: The topic to analyze sentiment for
        contact_id: Optional contact identifier (if None, analyzes all conversations)
        start_date: Start date for analysis (format: YYYY-MM-DD or "X days/weeks/months ago")
        end_date: End date for analysis (format: YYYY-MM-DD or "X days/weeks/months ago")
        
    Returns:
        Dictionary with sentiment analysis for the specified topic
    """
    try:
        # Validate parameters
        if not topic:
            return error_response("Topic is required")
        
        # Get database connection
        db = await get_database()
        
        # Parse dates using our utility function
        from ..utils.decorators import parse_date
        start_date_obj = parse_date(start_date)
        end_date_obj = parse_date(end_date)
        
        # Get contact info if contact_id is provided
        contact_data = None
        if contact_id:
            contact_info = await db.get_contact_info(contact_id=contact_id)
            contact_data = sanitize_contact_info(contact_info)
        
        # Get topic sentiment analysis
        sentiment_data = await db.get_topic_sentiment(
            topic=topic,
            contact_id=contact_id,
            start_date=start_date_obj,
            end_date=end_date_obj,
        )
        
        # Format the sentiment trend data
        sentiment_trend = []
        for trend_point in sentiment_data.get("sentiment_trend", []):
            sentiment_trend.append({
                "date": trend_point.get("date").isoformat() if isinstance(trend_point.get("date"), datetime) else trend_point.get("date"),
                "sentiment_score": trend_point.get("sentiment_score", 0),
                "positive_percentage": trend_point.get("positive_percentage", 0),
                "neutral_percentage": trend_point.get("neutral_percentage", 0),
                "negative_percentage": trend_point.get("negative_percentage", 0),
                "message_count": trend_point.get("message_count", 0),
            })
        
        # Prepare the result
        result = {
            "topic": topic,
            "date_range": {
                "start": start_date_obj.isoformat() if start_date_obj else None,
                "end": end_date_obj.isoformat() if end_date_obj else None,
            },
            "overall_sentiment": {
                "score": sentiment_data.get("overall_sentiment", {}).get("score", 0),
                "positive_percentage": sentiment_data.get("overall_sentiment", {}).get("positive_percentage", 0),
                "neutral_percentage": sentiment_data.get("overall_sentiment", {}).get("neutral_percentage", 0),
                "negative_percentage": sentiment_data.get("overall_sentiment", {}).get("negative_percentage", 0),
            },
            "sentiment_trend": sentiment_trend,
            "message_samples": {
                "positive": sentiment_data.get("message_samples", {}).get("positive", []),
                "neutral": sentiment_data.get("message_samples", {}).get("neutral", []),
                "negative": sentiment_data.get("message_samples", {}).get("negative", []),
            },
            "total_messages_analyzed": sentiment_data.get("total_messages", 0),
        }
        
        # Add contact info if provided
        if contact_data:
            result["contact"] = contact_data
            
            # Add comparison to overall sentiment if contact-specific
            result["comparison_to_overall"] = sentiment_data.get("comparison_to_overall", {})
        else:
            # Add contact-specific sentiment breakdown if analyzing all conversations
            result["contact_sentiment"] = sentiment_data.get("contact_sentiment", [])
        
        # Sanitize the overall result
        sanitized_result = sanitize_analysis_result(result)
        
        return success_response(sanitized_result)
    except DatabaseError as e:
        logger.error(f"Database error in analyze_topic_sentiment_tool: {e}")
        return error_response(e)
    except Exception as e:
        logger.error(f"Unexpected error in analyze_topic_sentiment_tool: {e}")
        logger.error(traceback.format_exc())
        return error_response(ToolExecutionError(f"Failed to analyze topic sentiment: {str(e)}"))


@register_tool(
    name="analyze_conversation_threading",
    description="Analyze conversation threading and topic connections",
)
async def analyze_conversation_threading_tool(
    contact_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze conversation threading and topic connections.
    
    Args:
        contact_id: Contact identifier (phone number, email, or handle)
        start_date: Start date for analysis (format: YYYY-MM-DD or "X days/weeks/months ago")
        end_date: End date for analysis (format: YYYY-MM-DD or "X days/weeks/months ago")
        
    Returns:
        Dictionary with conversation threading analysis
    """
    try:
        # Validate parameters
        if not contact_id:
            return error_response("Contact identifier is required")
        
        # Get database connection
        db = await get_database()
        
        # Parse dates using our utility function
        from ..utils.decorators import parse_date
        start_date_obj = parse_date(start_date)
        end_date_obj = parse_date(end_date)
        
        # Get contact info
        contact_info = await db.get_contact_info(contact_id=contact_id)
        sanitized_contact = sanitize_contact_info(contact_info)
        
        # Get conversation threading analysis
        threading_data = await db.get_conversation_threading(
            contact_id=contact_id,
            start_date=start_date_obj,
            end_date=end_date_obj,
        )
        
        # Format the conversation threads
        formatted_threads = []
        for thread in threading_data.get("threads", []):
            formatted_thread = {
                "thread_id": thread.get("thread_id"),
                "start_time": thread.get("start_time").isoformat() if isinstance(thread.get("start_time"), datetime) else thread.get("start_time"),
                "end_time": thread.get("end_time").isoformat() if isinstance(thread.get("end_time"), datetime) else thread.get("end_time"),
                "duration_minutes": thread.get("duration_minutes", 0),
                "message_count": thread.get("message_count", 0),
                "main_topic": thread.get("main_topic"),
                "subtopics": thread.get("subtopics", []),
                "sentiment_score": thread.get("sentiment_score", 0),
                "topic_shifts": thread.get("topic_shifts", []),
            }
            
            formatted_threads.append(formatted_thread)
        
        # Format topic connections
        formatted_connections = []
        for connection in threading_data.get("topic_connections", []):
            formatted_connections.append({
                "topic1": connection.get("topic1"),
                "topic2": connection.get("topic2"),
                "connection_strength": connection.get("connection_strength", 0),
                "co_occurrence_count": connection.get("co_occurrence_count", 0),
            })
        
        # Prepare the result
        result = {
            "contact": sanitized_contact,
            "date_range": {
                "start": start_date_obj.isoformat() if start_date_obj else None,
                "end": end_date_obj.isoformat() if end_date_obj else None,
            },
            "conversation_threads": formatted_threads,
            "topic_connections": formatted_connections,
            "metrics": {
                "average_thread_duration_minutes": threading_data.get("metrics", {}).get("avg_thread_duration", 0),
                "average_messages_per_thread": threading_data.get("metrics", {}).get("avg_messages_per_thread", 0),
                "average_topic_shifts_per_thread": threading_data.get("metrics", {}).get("avg_topic_shifts", 0),
                "topic_coherence_score": threading_data.get("metrics", {}).get("topic_coherence", 0),
            },
            "total_threads": len(threading_data.get("threads", [])),
            "total_messages_analyzed": threading_data.get("total_messages", 0),
        }
        
        # Sanitize the overall result
        sanitized_result = sanitize_analysis_result(result)
        
        return success_response(sanitized_result)
    except DatabaseError as e:
        logger.error(f"Database error in analyze_conversation_threading_tool: {e}")
        return error_response(e)
    except Exception as e:
        logger.error(f"Unexpected error in analyze_conversation_threading_tool: {e}")
        logger.error(traceback.format_exc())
        return error_response(ToolExecutionError(f"Failed to analyze conversation threading: {str(e)}"))


@register_tool(
    name="get_communication_style",
    description="Analyze communication style and language patterns in conversations",
)
async def get_communication_style_tool(
    contact_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze communication style and language patterns in conversations.
    
    Args:
        contact_id: Contact identifier (phone number, email, or handle)
        start_date: Start date for analysis (format: YYYY-MM-DD or "X days/weeks/months ago")
        end_date: End date for analysis (format: YYYY-MM-DD or "X days/weeks/months ago")
        
    Returns:
        Dictionary with communication style analysis
    """
    try:
        # Validate parameters
        if not contact_id:
            return error_response("Contact identifier is required")
        
        # Get database connection
        db = await get_database()
        
        # Parse dates using our utility function
        from ..utils.decorators import parse_date
        start_date_obj = parse_date(start_date)
        end_date_obj = parse_date(end_date)
        
        # Get contact info
        contact_info = await db.get_contact_info(contact_id=contact_id)
        sanitized_contact = sanitize_contact_info(contact_info)
        
        # Get communication style analysis
        style_data = await db.get_communication_style(
            contact_id=contact_id,
            start_date=start_date_obj,
            end_date=end_date_obj,
        )
        
        # Prepare the result
        result = {
            "contact": sanitized_contact,
            "date_range": {
                "start": start_date_obj.isoformat() if start_date_obj else None,
                "end": end_date_obj.isoformat() if end_date_obj else None,
            },
            "your_style": {
                "formality_level": style_data.get("your_style", {}).get("formality_level", 0),
                "vocabulary_richness": style_data.get("your_style", {}).get("vocabulary_richness", 0),
                "average_message_length": style_data.get("your_style", {}).get("avg_message_length", 0),
                "emoji_usage_rate": style_data.get("your_style", {}).get("emoji_usage_rate", 0),
                "response_time_seconds": style_data.get("your_style", {}).get("response_time", 0),
                "conversation_initiation_rate": style_data.get("your_style", {}).get("initiation_rate", 0),
                "most_used_phrases": style_data.get("your_style", {}).get("most_used_phrases", []),
            },
            "their_style": {
                "formality_level": style_data.get("their_style", {}).get("formality_level", 0),
                "vocabulary_richness": style_data.get("their_style", {}).get("vocabulary_richness", 0),
                "average_message_length": style_data.get("their_style", {}).get("avg_message_length", 0),
                "emoji_usage_rate": style_data.get("their_style", {}).get("emoji_usage_rate", 0),
                "response_time_seconds": style_data.get("their_style", {}).get("response_time", 0),
                "conversation_initiation_rate": style_data.get("their_style", {}).get("initiation_rate", 0),
                "most_used_phrases": style_data.get("their_style", {}).get("most_used_phrases", []),
            },
            "style_comparison": {
                "vocabulary_overlap_percentage": style_data.get("comparison", {}).get("vocabulary_overlap", 0),
                "emoji_similarity_score": style_data.get("comparison", {}).get("emoji_similarity", 0),
                "formality_difference": style_data.get("comparison", {}).get("formality_difference", 0),
                "language_accommodation_score": style_data.get("comparison", {}).get("language_accommodation", 0),
            },
            "communication_patterns": {
                "conversation_flow_score": style_data.get("patterns", {}).get("conversation_flow", 0),
                "back_and_forth_rate": style_data.get("patterns", {}).get("back_and_forth_rate", 0),
                "question_answer_rate": style_data.get("patterns", {}).get("question_answer_rate", 0),
                "conversation_depth": style_data.get("patterns", {}).get("conversation_depth", 0),
            },
            "total_messages_analyzed": style_data.get("total_messages", 0),
        }
        
        # Sanitize the overall result
        sanitized_result = sanitize_analysis_result(result)
        
        return success_response(sanitized_result)
    except DatabaseError as e:
        logger.error(f"Database error in get_communication_style_tool: {e}")
        return error_response(e)
    except Exception as e:
        logger.error(f"Unexpected error in get_communication_style_tool: {e}")
        logger.error(traceback.format_exc())
        return error_response(ToolExecutionError(f"Failed to get communication style: {str(e)}"))
