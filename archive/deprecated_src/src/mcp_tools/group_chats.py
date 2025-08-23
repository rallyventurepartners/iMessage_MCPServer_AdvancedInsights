"""
Group chat analysis tools for the iMessage Advanced Insights server.

This module provides tools for analyzing group chat data and communication patterns
from the iMessage database.
"""

import logging
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from ..database import get_database
from ..exceptions import DatabaseError, ToolExecutionError
from ..utils.responses import error_response, paginated_response, success_response
from ..utils.sanitization import sanitize_contact_info, sanitize_group_chat_data
from .registry import register_tool

logger = logging.getLogger(__name__)


@register_tool(
    name="get_group_chats",
    description="Get a list of group chats with basic metrics",
)
async def get_group_chats_tool(
    search_term: Optional[str] = None,
    active_since: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
) -> Dict[str, Any]:
    """
    Get a list of group chats with basic metrics.
    
    Args:
        search_term: Optional search term to filter group chats
        active_since: Only show group chats active since this date (format: YYYY-MM-DD or "X days/weeks/months ago")
        page: Page number for paginated results
        page_size: Number of group chats per page
        
    Returns:
        Paginated list of group chats with basic metrics
    """
    try:
        # Validate parameters
        if page < 1:
            return error_response("Page number must be at least 1")
        
        if page_size < 1 or page_size > 100:
            return error_response("Page size must be between 1 and 100")
        
        # Get database connection
        db = await get_database()
        
        # Parse dates using our utility function
        from ..utils.decorators import parse_date
        active_since_obj = parse_date(active_since)
        
        # Calculate offset from page number
        offset = (page - 1) * page_size
        
        # Get group chats
        # Note: The database method doesn't support search_term or active_since filtering
        result = await db.get_group_chats(
            limit=page_size,
            offset=offset
        )
        
        # Get group chats from the result
        group_chats = result.get("group_chats", [])
        total_count = result.get("total", 0)
        
        # Sanitize group chats
        sanitized_group_chats = [sanitize_group_chat_data(chat) for chat in group_chats]
        
        # Create pagination info
        return paginated_response(
            items=sanitized_group_chats,
            page=page,
            page_size=page_size,
            total_items=total_count,
            additional_data={
                "search_term": search_term,
                "active_since": active_since_obj.isoformat() if active_since_obj else None,
            },
        )
    except DatabaseError as e:
        logger.error(f"Database error in get_group_chats_tool: {e}")
        return error_response(e)
    except Exception as e:
        logger.error(f"Unexpected error in get_group_chats_tool: {e}")
        logger.error(traceback.format_exc())
        return error_response(ToolExecutionError(f"Failed to get group chats: {str(e)}"))


@register_tool(
    name="get_group_chat_analytics",
    description="Get detailed analytics for a specific group chat",
)
async def get_group_chat_analytics_tool(
    chat_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get detailed analytics for a specific group chat.
    
    Args:
        chat_id: Group chat identifier
        start_date: Start date for analysis (format: YYYY-MM-DD or "X days/weeks/months ago")
        end_date: End date for analysis (format: YYYY-MM-DD or "X days/weeks/months ago")
        
    Returns:
        Dictionary with detailed group chat analytics
    """
    try:
        # Validate parameters
        if not chat_id:
            return error_response("Group chat identifier is required")
        
        # Get database connection
        db = await get_database()
        
        # Parse dates using our utility function
        from ..utils.decorators import parse_date
        start_date_obj = parse_date(start_date)
        end_date_obj = parse_date(end_date)
        
        # Get group chat info
        chat_info = await db.get_group_chat_info(chat_id=chat_id)
        sanitized_chat = sanitize_group_chat_data(chat_info)
        
        # Get group chat analytics
        analytics = await db.get_group_chat_analytics(
            chat_id=chat_id,
            start_date=start_date_obj,
            end_date=end_date_obj,
        )
        
        # Get member participation
        member_participation = await db.get_group_chat_member_participation(
            chat_id=chat_id,
            start_date=start_date_obj,
            end_date=end_date_obj,
        )
        
        # Sanitize member participation
        sanitized_members = []
        # The method returns a dict with 'members' key containing the list
        members_list = member_participation.get("members", [])
        
        for member in members_list:
            # The member data is already in the correct format from the database
            sanitized_member = {
                "contact": sanitize_contact_info({"identifier": member.get("member_id", "")}),
                "message_count": member.get("message_count", 0),
                "participation_percentage": member.get("participation_percentage", 0),
                "average_message_length": member.get("avg_message_length", 0),
                "first_message_date": member.get("first_message_date"),
                "last_message_date": member.get("last_message_date"),
                "attachment_count": member.get("attachment_count", 0),
                "is_me": member.get("is_me", False)
            }
            sanitized_members.append(sanitized_member)
        
        # Process and sanitize analytics
        analytics_result = {
            "group_chat": sanitized_chat,
            "date_range": {
                "start": start_date_obj.isoformat() if start_date_obj else None,
                "end": end_date_obj.isoformat() if end_date_obj else None,
            },
            "message_counts": {
                "total": analytics.get("total_messages", 0),
                "average_per_day": analytics.get("avg_messages_per_day", 0),
                "attachments": analytics.get("attachments", 0),
            },
            "activity_patterns": {
                "most_active_day": analytics.get("most_active_day"),
                "most_active_hour": analytics.get("most_active_hour"),
                "least_active_day": analytics.get("least_active_day"),
                "average_conversation_gap_hours": analytics.get("avg_conversation_gap", 0),
            },
            "conversation_metrics": {
                "conversation_count": analytics.get("conversation_count", 0),
                "average_messages_per_conversation": analytics.get("avg_messages_per_conversation", 0),
                "longest_conversation_messages": analytics.get("longest_conversation", 0),
                "average_conversation_duration_minutes": analytics.get("avg_conversation_duration", 0),
            },
            "member_participation": sanitized_members,
            "sentiment_analysis": analytics.get("sentiment", {
                "overall": analytics.get("overall_sentiment", 0),
                "trend": analytics.get("sentiment_trend", "stable"),
                "most_positive_day": analytics.get("most_positive_day"),
                "most_negative_day": analytics.get("most_negative_day"),
            }),
        }
        
        return success_response(analytics_result)
    except DatabaseError as e:
        logger.error(f"Database error in get_group_chat_analytics_tool: {e}")
        return error_response(e)
    except Exception as e:
        logger.error(f"Unexpected error in get_group_chat_analytics_tool: {e}")
        logger.error(traceback.format_exc())
        return error_response(ToolExecutionError(f"Failed to get group chat analytics: {str(e)}"))


@register_tool(
    name="get_group_chat_topics",
    description="Analyze topic distribution in a group chat",
)
async def get_group_chat_topics_tool(
    chat_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    topic_count: int = 5,
) -> Dict[str, Any]:
    """
    Analyze topic distribution in a group chat.
    
    Args:
        chat_id: Group chat identifier
        start_date: Start date for analysis (format: YYYY-MM-DD or "X days/weeks/months ago")
        end_date: End date for analysis (format: YYYY-MM-DD or "X days/weeks/months ago")
        topic_count: Number of top topics to return
        
    Returns:
        Dictionary with topic analysis for the group chat
    """
    try:
        # Validate parameters
        if not chat_id:
            return error_response("Group chat identifier is required")
        
        if topic_count < 1 or topic_count > 20:
            return error_response("Topic count must be between 1 and 20")
        
        # Get database connection
        db = await get_database()
        
        # Parse dates using our utility function
        from ..utils.decorators import parse_date
        start_date_obj = parse_date(start_date)
        end_date_obj = parse_date(end_date)
        
        # Get group chat info
        chat_info = await db.get_group_chat_info(chat_id=chat_id)
        sanitized_chat = sanitize_group_chat_data(chat_info)
        
        # Get topic analysis
        topics = await db.get_group_chat_topics(
            chat_id=chat_id,
            start_date=start_date_obj,
            end_date=end_date_obj,
            topic_count=topic_count,
        )
        
        # Get per-member topic participation
        member_topics = await db.get_group_chat_member_topics(
            chat_id=chat_id,
            start_date=start_date_obj,
            end_date=end_date_obj,
            topic_count=topic_count,
        )
        
        # Sanitize member topic participation
        sanitized_member_topics = []
        for member in member_topics:
            contact_info = member.get("contact", {})
            topics_data = member.get("topics", [])
            
            sanitized_member_topics.append({
                "contact": sanitize_contact_info(contact_info),
                "topics": topics_data,
            })
        
        # Format result
        result = {
            "group_chat": sanitized_chat,
            "date_range": {
                "start": start_date_obj.isoformat() if start_date_obj else None,
                "end": end_date_obj.isoformat() if end_date_obj else None,
            },
            "topics": topics,
            "member_topics": sanitized_member_topics,
            "topic_sentiment": topics.get("topic_sentiment", {}),
            "topic_trends": topics.get("topic_trends", {}),
        }
        
        return success_response(result)
    except DatabaseError as e:
        logger.error(f"Database error in get_group_chat_topics_tool: {e}")
        return error_response(e)
    except Exception as e:
        logger.error(f"Unexpected error in get_group_chat_topics_tool: {e}")
        logger.error(traceback.format_exc())
        return error_response(ToolExecutionError(f"Failed to get group chat topics: {str(e)}"))


@register_tool(
    name="get_group_chat_member_dynamics",
    description="Analyze interaction dynamics between group chat members",
)
async def get_group_chat_member_dynamics_tool(
    chat_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze interaction dynamics between group chat members.
    
    Args:
        chat_id: Group chat identifier
        start_date: Start date for analysis (format: YYYY-MM-DD or "X days/weeks/months ago")
        end_date: End date for analysis (format: YYYY-MM-DD or "X days/weeks/months ago")
        
    Returns:
        Dictionary with member interaction dynamics for the group chat
    """
    try:
        # Validate parameters
        if not chat_id:
            return error_response("Group chat identifier is required")
        
        # Get database connection
        db = await get_database()
        
        # Parse dates using our utility function
        from ..utils.decorators import parse_date
        start_date_obj = parse_date(start_date)
        end_date_obj = parse_date(end_date)
        
        # Get group chat info
        chat_info = await db.get_group_chat_info(chat_id=chat_id)
        sanitized_chat = sanitize_group_chat_data(chat_info)
        
        # Get member interaction dynamics
        dynamics = await db.get_group_chat_member_dynamics(
            chat_id=chat_id,
            start_date=start_date_obj,
            end_date=end_date_obj,
        )
        
        # Sanitize interaction data (this contains pairwise interactions)
        sanitized_interactions = []
        for interaction in dynamics.get("interactions", []):
            contact1 = sanitize_contact_info(interaction.get("contact1", {}))
            contact2 = sanitize_contact_info(interaction.get("contact2", {}))
            
            sanitized_interactions.append({
                "contact1": contact1,
                "contact2": contact2,
                "interaction_count": interaction.get("interaction_count", 0),
                "average_response_time_seconds": interaction.get("avg_response_time", 0),
                "sentiment": interaction.get("sentiment", 0),
                "topics": interaction.get("topics", []),
            })
        
        # Sanitize member roles data
        sanitized_member_roles = []
        for member in dynamics.get("member_roles", []):
            contact_info = member.get("contact", {})
            roles_data = member.get("roles", {})
            
            sanitized_member_roles.append({
                "contact": sanitize_contact_info(contact_info),
                "roles": {
                    "conversation_starter": roles_data.get("conversation_starter", 0),
                    "mediator": roles_data.get("mediator", 0),
                    "active_participant": roles_data.get("active_participant", 0),
                    "information_provider": roles_data.get("information_provider", 0),
                    "emotional_support": roles_data.get("emotional_support", 0),
                    "humor_provider": roles_data.get("humor_provider", 0),
                },
                "centrality": roles_data.get("centrality", 0),
                "influence_score": roles_data.get("influence_score", 0),
            })
        
        # Format result
        result = {
            "group_chat": sanitized_chat,
            "date_range": {
                "start": start_date_obj.isoformat() if start_date_obj else None,
                "end": end_date_obj.isoformat() if end_date_obj else None,
            },
            "member_interactions": sanitized_interactions,
            "member_roles": sanitized_member_roles,
            "group_cohesion": dynamics.get("group_cohesion", 0),
            "subgroup_detection": dynamics.get("subgroups", []),
        }
        
        return success_response(result)
    except DatabaseError as e:
        logger.error(f"Database error in get_group_chat_member_dynamics_tool: {e}")
        return error_response(e)
    except Exception as e:
        logger.error(f"Unexpected error in get_group_chat_member_dynamics_tool: {e}")
        logger.error(traceback.format_exc())
        return error_response(ToolExecutionError(f"Failed to get group chat member dynamics: {str(e)}"))


@register_tool(
    name="get_group_chat_language_patterns",
    description="Analyze language patterns in a group chat",
)
async def get_group_chat_language_patterns_tool(
    chat_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze language patterns in a group chat.
    
    Args:
        chat_id: Group chat identifier
        start_date: Start date for analysis (format: YYYY-MM-DD or "X days/weeks/months ago")
        end_date: End date for analysis (format: YYYY-MM-DD or "X days/weeks/months ago")
        
    Returns:
        Dictionary with language pattern analysis for the group chat
    """
    try:
        # Validate parameters
        if not chat_id:
            return error_response("Group chat identifier is required")
        
        # Get database connection
        db = await get_database()
        
        # Parse dates using our utility function
        from ..utils.decorators import parse_date
        start_date_obj = parse_date(start_date)
        end_date_obj = parse_date(end_date)
        
        # Get group chat info
        chat_info = await db.get_group_chat_info(chat_id=chat_id)
        sanitized_chat = sanitize_group_chat_data(chat_info)
        
        # Get language patterns
        patterns = await db.get_group_chat_language_patterns(
            chat_id=chat_id,
            start_date=start_date_obj,
            end_date=end_date_obj,
        )
        
        # Get per-member language patterns
        member_patterns = await db.get_group_chat_member_language_patterns(
            chat_id=chat_id,
            start_date=start_date_obj,
            end_date=end_date_obj,
        )
        
        # Sanitize member language patterns
        sanitized_member_patterns = []
        for member in member_patterns:
            contact_info = member.get("contact", {})
            patterns_data = member.get("patterns", {})
            
            sanitized_member_patterns.append({
                "contact": sanitize_contact_info(contact_info),
                "patterns": {
                    "average_message_length": patterns_data.get("avg_message_length", 0),
                    "unique_word_count": patterns_data.get("unique_word_count", 0),
                    "emoji_usage": patterns_data.get("emoji_usage", {}),
                    "vocabulary_richness": patterns_data.get("vocabulary_richness", 0),
                    "formality_score": patterns_data.get("formality_score", 0),
                    "most_used_phrases": patterns_data.get("most_used_phrases", []),
                    "slang_usage": patterns_data.get("slang_usage", {}),
                },
                "conversation_style": patterns_data.get("conversation_style", {}),
            })
        
        # Format result
        result = {
            "group_chat": sanitized_chat,
            "date_range": {
                "start": start_date_obj.isoformat() if start_date_obj else None,
                "end": end_date_obj.isoformat() if end_date_obj else None,
            },
            "overall_patterns": {
                "average_message_length": patterns.get("avg_message_length", 0),
                "vocabulary_size": patterns.get("vocabulary_size", 0),
                "most_used_words": patterns.get("most_used_words", []),
                "most_used_phrases": patterns.get("most_used_phrases", []),
                "emoji_usage": patterns.get("emoji_usage", {}),
                "slang_usage": patterns.get("slang_usage", {}),
                "formality_level": patterns.get("formality_level", 0),
            },
            "member_patterns": sanitized_member_patterns,
            "language_convergence": patterns.get("language_convergence", 0),
            "chat_vocabulary_overlap": patterns.get("vocabulary_overlap", {}),
        }
        
        return success_response(result)
    except DatabaseError as e:
        logger.error(f"Database error in get_group_chat_language_patterns_tool: {e}")
        return error_response(e)
    except Exception as e:
        logger.error(f"Unexpected error in get_group_chat_language_patterns_tool: {e}")
        logger.error(traceback.format_exc())
        return error_response(ToolExecutionError(f"Failed to get group chat language patterns: {str(e)}"))
