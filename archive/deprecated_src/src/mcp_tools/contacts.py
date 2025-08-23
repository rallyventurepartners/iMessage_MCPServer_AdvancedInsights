"""
Contact analytics tools for the iMessage Advanced Insights server.

This module provides tools for analyzing contact data and communication patterns
from the iMessage database.
"""

import logging
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from ..database import get_database
from ..exceptions import DatabaseError, ToolExecutionError
from ..utils.responses import error_response, paginated_response, success_response
from ..utils.sanitization import sanitize_contact_info
from .registry import register_tool

logger = logging.getLogger(__name__)


@register_tool(
    name="get_contacts",
    description="Get a list of contacts with activity statistics",
)
async def get_contacts_tool(
    search_term: Optional[str] = None,
    active_since: Optional[str] = None,
    page: int = 1,
    page_size: int = 30,
) -> Dict[str, Any]:
    """
    Get a list of contacts with messaging activity statistics.
    
    Args:
        search_term: Optional search term to filter contacts
        active_since: Only show contacts active since this date (format: YYYY-MM-DD or "X days/weeks/months ago")
        page: Page number for paginated results
        page_size: Number of contacts per page
        
    Returns:
        Paginated list of contacts with activity statistics
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
        
        # Get contacts with statistics
        # Note: The database method doesn't support search_term or active_since filtering
        # We'll need to get all contacts and filter them in memory
        result = await db.get_contacts(
            limit=page_size,
            offset=offset,
            minimal=False  # Get full data for filtering
        )
        
        # Get contacts from the result
        contacts = result.get("contacts", [])
        total_count = result.get("total", 0)
        
        # Sanitize contacts to protect sensitive information
        sanitized_contacts = [sanitize_contact_info(contact) for contact in contacts]
        
        # Create pagination info
        return paginated_response(
            items=sanitized_contacts,
            page=page,
            page_size=page_size,
            total_items=total_count,
            additional_data={
                "search_term": search_term,
                "active_since": active_since_obj.isoformat() if active_since_obj else None,
            },
        )
    except DatabaseError as e:
        logger.error(f"Database error in get_contacts_tool: {e}")
        return error_response(e)
    except Exception as e:
        logger.error(f"Unexpected error in get_contacts_tool: {e}")
        logger.error(traceback.format_exc())
        return error_response(ToolExecutionError(f"Failed to get contacts: {str(e)}"))


@register_tool(
    name="get_contact_analytics",
    description="Get detailed analytics for a specific contact",
)
async def get_contact_analytics_tool(
    contact_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get detailed analytics for a specific contact.
    
    Args:
        contact_id: Contact identifier (phone number, email, or handle)
        start_date: Start date for analysis (format: YYYY-MM-DD or "X days/weeks/months ago")
        end_date: End date for analysis (format: YYYY-MM-DD or "X days/weeks/months ago")
        
    Returns:
        Dictionary with detailed contact analytics
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
        
        # Get contact analytics
        analytics = await db.get_contact_analytics(
            contact_id=contact_id,
            start_date=start_date_obj,
            end_date=end_date_obj,
        )
        
        # Get contact info
        contact_info = await db.get_contact_info(contact_id=contact_id)
        sanitized_contact = sanitize_contact_info(contact_info)
        
        # Process and sanitize analytics
        analytics_result = {
            "contact": sanitized_contact,
            "date_range": {
                "start": start_date_obj.isoformat() if start_date_obj else None,
                "end": end_date_obj.isoformat() if end_date_obj else None,
            },
            "message_counts": {
                "total": analytics.get("total_messages", 0),
                "sent": analytics.get("sent_messages", 0),
                "received": analytics.get("received_messages", 0),
                "attachments": analytics.get("attachments", 0),
            },
            "activity_patterns": {
                "most_active_day": analytics.get("most_active_day"),
                "most_active_hour": analytics.get("most_active_hour"),
                "average_response_time_seconds": analytics.get("avg_response_time"),
                "conversation_starters": analytics.get("conversation_starters", 0),
            },
            "conversation_metrics": {
                "conversation_count": analytics.get("conversation_count", 0),
                "average_messages_per_conversation": analytics.get("avg_messages_per_conversation", 0),
                "longest_conversation_messages": analytics.get("longest_conversation", 0),
                "average_conversation_duration_minutes": analytics.get("avg_conversation_duration", 0),
            },
            "content_metrics": {
                "average_message_length": analytics.get("avg_message_length", 0),
                "emoji_count": analytics.get("emoji_count", 0),
                "most_used_emojis": analytics.get("most_used_emojis", []),
                "link_count": analytics.get("link_count", 0),
            },
        }
        
        return success_response(analytics_result)
    except DatabaseError as e:
        logger.error(f"Database error in get_contact_analytics_tool: {e}")
        return error_response(e)
    except Exception as e:
        logger.error(f"Unexpected error in get_contact_analytics_tool: {e}")
        logger.error(traceback.format_exc())
        return error_response(ToolExecutionError(f"Failed to get contact analytics: {str(e)}"))


@register_tool(
    name="get_communication_timeline",
    description="Get a timeline of communication frequency with a contact",
)
async def get_communication_timeline_tool(
    contact_id: str,
    interval: str = "day",  # day, week, month
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get a timeline of communication frequency with a contact.
    
    Args:
        contact_id: Contact identifier (phone number, email, or handle)
        interval: Time interval for grouping (day, week, month)
        start_date: Start date for timeline (format: YYYY-MM-DD or "X days/weeks/months ago")
        end_date: End date for timeline (format: YYYY-MM-DD or "X days/weeks/months ago")
        
    Returns:
        Dictionary with communication timeline data
    """
    try:
        # Validate parameters
        if not contact_id:
            return error_response("Contact identifier is required")
        
        # Validate interval
        if interval not in ["day", "week", "month"]:
            return error_response("Interval must be 'day', 'week', or 'month'")
        
        # Get database connection
        db = await get_database()
        
        # Parse dates using our utility function
        from ..utils.decorators import parse_date
        start_date_obj = parse_date(start_date)
        end_date_obj = parse_date(end_date)
        
        # Get communication timeline
        timeline = await db.get_communication_timeline(
            contact_id=contact_id,
            interval=interval,
            start_date=start_date_obj,
            end_date=end_date_obj,
        )
        
        # Get contact info
        contact_info = await db.get_contact_info(contact_id=contact_id)
        sanitized_contact = sanitize_contact_info(contact_info)
        
        # Format the timeline data
        formatted_timeline = []
        for entry in timeline:
            formatted_timeline.append({
                "date": entry.get("date").isoformat() if isinstance(entry.get("date"), datetime) else entry.get("date"),
                "total_messages": entry.get("total_messages", 0),
                "sent_messages": entry.get("sent_messages", 0),
                "received_messages": entry.get("received_messages", 0),
            })
        
        # Calculate some summary statistics
        total_messages = sum(entry.get("total_messages", 0) for entry in timeline)
        total_sent = sum(entry.get("sent_messages", 0) for entry in timeline)
        total_received = sum(entry.get("received_messages", 0) for entry in timeline)
        
        result = {
            "contact": sanitized_contact,
            "date_range": {
                "start": start_date_obj.isoformat() if start_date_obj else None,
                "end": end_date_obj.isoformat() if end_date_obj else None,
            },
            "interval": interval,
            "timeline": formatted_timeline,
            "summary": {
                "total_messages": total_messages,
                "total_sent": total_sent,
                "total_received": total_received,
                "period_count": len(timeline),
                "average_messages_per_period": total_messages / len(timeline) if timeline else 0,
            },
        }
        
        return success_response(result)
    except DatabaseError as e:
        logger.error(f"Database error in get_communication_timeline_tool: {e}")
        return error_response(e)
    except Exception as e:
        logger.error(f"Unexpected error in get_communication_timeline_tool: {e}")
        logger.error(traceback.format_exc())
        return error_response(ToolExecutionError(f"Failed to get communication timeline: {str(e)}"))


@register_tool(
    name="get_relationship_strength",
    description="Calculate relationship strength indicators for contacts",
)
async def get_relationship_strength_tool(
    contact_id: Optional[str] = None,
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Calculate relationship strength indicators for one or more contacts.
    
    Args:
        contact_id: Optional contact identifier (if None, returns top contacts by strength)
        limit: Number of contacts to return when contact_id is None
        
    Returns:
        Dictionary with relationship strength indicators
    """
    try:
        # Get database connection
        db = await get_database()
        
        # If contact_id is provided, get relationship strength for that contact
        if contact_id:
            # Get contact info
            contact_info = await db.get_contact_info(contact_id=contact_id)
            sanitized_contact = sanitize_contact_info(contact_info)
            
            # Get relationship strength
            strength = await db.get_relationship_strength(contact_id=contact_id)
            
            result = {
                "contact": sanitized_contact,
                "strength_score": strength.get("strength_score", 0),
                "indicators": {
                    "message_frequency": strength.get("message_frequency", 0),
                    "response_rate": strength.get("response_rate", 0),
                    "response_time": strength.get("response_time", 0),
                    "conversation_depth": strength.get("conversation_depth", 0),
                    "initiation_balance": strength.get("initiation_balance", 0),
                    "recent_activity": strength.get("recent_activity", 0),
                },
                "ranking": strength.get("ranking"),
                "percentile": strength.get("percentile"),
            }
            
            return success_response(result)
        
        # If no contact_id, get top contacts by relationship strength
        else:
            # Validate limit
            if limit < 1 or limit > 100:
                return error_response("Limit must be between 1 and 100")
            
            # Get top contacts by relationship strength
            top_contacts = await db.get_top_contacts_by_strength(limit=limit)
            
            # Sanitize contacts
            sanitized_contacts = []
            for contact in top_contacts:
                contact_info = contact.get("contact", {})
                strength_data = contact.get("strength", {})
                
                sanitized_contacts.append({
                    "contact": sanitize_contact_info(contact_info),
                    "strength_score": strength_data.get("strength_score", 0),
                    "ranking": strength_data.get("ranking"),
                    "percentile": strength_data.get("percentile"),
                })
            
            result = {
                "contacts": sanitized_contacts,
                "total_contacts_analyzed": top_contacts[0].get("strength", {}).get("total_contacts", 0) if top_contacts else 0,
            }
            
            return success_response(result)
    except DatabaseError as e:
        logger.error(f"Database error in get_relationship_strength_tool: {e}")
        return error_response(e)
    except Exception as e:
        logger.error(f"Unexpected error in get_relationship_strength_tool: {e}")
        logger.error(traceback.format_exc())
        return error_response(ToolExecutionError(f"Failed to get relationship strength: {str(e)}"))
