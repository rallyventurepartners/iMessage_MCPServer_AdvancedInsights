"""
Visualization tools for the iMessage Advanced Insights server.

This module provides tools for generating and accessing visualizations of
message data, such as network graphs, timeline visualizations, and topic
distributions.
"""

import logging
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from ..database import get_database
from ..exceptions import DatabaseError, ToolExecutionError
from ..utils.responses import error_response, success_response
from ..utils.sanitization import sanitize_contact_info
from .registry import register_tool

logger = logging.getLogger(__name__)


@register_tool(
    name="visualize_message_network",
    description="Generate a visualization of the messaging network",
)
async def visualize_message_network_tool(
    time_period: str = "1 year",
    min_message_count: int = 10,
    format: str = "svg",
    include_group_chats: bool = True,
    layout: str = "force",
) -> Dict[str, Any]:
    """
    Generate a visualization of the messaging network.
    
    Args:
        time_period: Time period to analyze (e.g., "1 year", "6 months")
        min_message_count: Minimum number of messages to include a contact
        format: Output format (svg, png, json)
        include_group_chats: Whether to include group chats in the network
        layout: Network layout algorithm (force, circular, hierarchical)
        
    Returns:
        Dictionary with network visualization data or image
    """
    try:
        # Validate parameters
        if format not in ["svg", "png", "json"]:
            return error_response("Format must be 'svg', 'png', or 'json'")
        
        if layout not in ["force", "circular", "hierarchical"]:
            return error_response("Layout must be 'force', 'circular', or 'hierarchical'")
        
        # Get database connection
        db = await get_database()
        
        # Parse time period to get start date
        from ..utils.decorators import parse_relative_date
        start_date = parse_relative_date(time_period)
        
        # Get network data
        network_data = await db.get_message_network(
            start_date=start_date,
            min_message_count=min_message_count,
            include_group_chats=include_group_chats,
        )
        
        # Sanitize contact info in nodes
        sanitized_nodes = []
        for node in network_data.get("nodes", []):
            if node.get("type") == "contact":
                contact_info = node.get("data", {})
                sanitized_contact = sanitize_contact_info(contact_info)
                node["data"] = sanitized_contact
            
            sanitized_nodes.append(node)
        
        # Generate the visualization
        visualization = await db.generate_network_visualization(
            nodes=sanitized_nodes,
            edges=network_data.get("edges", []),
            layout=layout,
            format=format,
        )
        
        # Prepare the result based on format
        if format == "json":
            # Return the network data for client-side visualization
            result = {
                "time_period": time_period,
                "data": {
                    "nodes": sanitized_nodes,
                    "edges": network_data.get("edges", []),
                },
                "metrics": {
                    "total_nodes": len(sanitized_nodes),
                    "total_edges": len(network_data.get("edges", [])),
                    "density": network_data.get("metrics", {}).get("density", 0),
                    "average_degree": network_data.get("metrics", {}).get("avg_degree", 0),
                },
            }
        else:
            # Return the visualization image (encoded as base64)
            result = {
                "time_period": time_period,
                "format": format,
                "image_data": visualization.get("image_data"),
                "mime_type": f"image/{format}",
                "metrics": {
                    "total_nodes": len(sanitized_nodes),
                    "total_edges": len(network_data.get("edges", [])),
                },
            }
        
        return success_response(result)
    except DatabaseError as e:
        logger.error(f"Database error in visualize_message_network_tool: {e}")
        return error_response(e)
    except Exception as e:
        logger.error(f"Unexpected error in visualize_message_network_tool: {e}")
        logger.error(traceback.format_exc())
        return error_response(ToolExecutionError(f"Failed to visualize message network: {str(e)}"))


@register_tool(
    name="visualize_contact_timeline",
    description="Generate a timeline visualization of communication with a contact",
)
async def visualize_contact_timeline_tool(
    contact_id: str,
    interval: str = "day",  # hour, day, week, month
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    include_sentiment: bool = True,
    format: str = "svg",
) -> Dict[str, Any]:
    """
    Generate a timeline visualization of communication with a contact.
    
    Args:
        contact_id: Contact identifier (phone number, email, or handle)
        interval: Time interval for aggregation (hour, day, week, month)
        start_date: Start date for visualization (format: YYYY-MM-DD or "X days/weeks/months ago")
        end_date: End date for visualization (format: YYYY-MM-DD or "X days/weeks/months ago")
        include_sentiment: Whether to include sentiment analysis in the visualization
        format: Output format (svg, png, json)
        
    Returns:
        Dictionary with timeline visualization data or image
    """
    try:
        # Validate parameters
        if not contact_id:
            return error_response("Contact identifier is required")
        
        if interval not in ["hour", "day", "week", "month"]:
            return error_response("Interval must be 'hour', 'day', 'week', or 'month'")
        
        if format not in ["svg", "png", "json"]:
            return error_response("Format must be 'svg', 'png', or 'json'")
        
        # Get database connection
        db = await get_database()
        
        # Parse dates using our utility function
        from ..utils.decorators import parse_date
        start_date_obj = parse_date(start_date)
        end_date_obj = parse_date(end_date)
        
        # Get contact info
        contact_info = await db.get_contact_info(contact_id=contact_id)
        sanitized_contact = sanitize_contact_info(contact_info)
        
        # Get timeline data
        timeline_data = await db.get_contact_timeline(
            contact_id=contact_id,
            interval=interval,
            start_date=start_date_obj,
            end_date=end_date_obj,
            include_sentiment=include_sentiment,
        )
        
        # Generate the visualization if not returning JSON
        visualization = None
        if format != "json":
            visualization = await db.generate_timeline_visualization(
                timeline_data=timeline_data.get("timeline", []),
                contact_info=sanitized_contact,
                interval=interval,
                include_sentiment=include_sentiment,
                format=format,
            )
        
        # Prepare the result based on format
        if format == "json":
            # Return the timeline data for client-side visualization
            result = {
                "contact": sanitized_contact,
                "date_range": {
                    "start": start_date_obj.isoformat() if start_date_obj else None,
                    "end": end_date_obj.isoformat() if end_date_obj else None,
                },
                "interval": interval,
                "timeline": timeline_data.get("timeline", []),
                "metrics": {
                    "total_messages": timeline_data.get("total_messages", 0),
                    "sent_messages": timeline_data.get("sent_messages", 0),
                    "received_messages": timeline_data.get("received_messages", 0),
                    "average_messages_per_period": timeline_data.get("avg_per_period", 0),
                },
            }
            
            if include_sentiment:
                result["sentiment"] = timeline_data.get("sentiment", {})
        else:
            # Return the visualization image (encoded as base64)
            result = {
                "contact": sanitized_contact,
                "date_range": {
                    "start": start_date_obj.isoformat() if start_date_obj else None,
                    "end": end_date_obj.isoformat() if end_date_obj else None,
                },
                "interval": interval,
                "format": format,
                "image_data": visualization.get("image_data"),
                "mime_type": f"image/{format}",
            }
        
        return success_response(result)
    except DatabaseError as e:
        logger.error(f"Database error in visualize_contact_timeline_tool: {e}")
        return error_response(e)
    except Exception as e:
        logger.error(f"Unexpected error in visualize_contact_timeline_tool: {e}")
        logger.error(traceback.format_exc())
        return error_response(ToolExecutionError(f"Failed to visualize contact timeline: {str(e)}"))


@register_tool(
    name="visualize_topic_distribution",
    description="Generate a visualization of topic distribution in conversations",
)
async def visualize_topic_distribution_tool(
    contact_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    topic_count: int = 10,
    visualization_type: str = "pie",
    format: str = "svg",
) -> Dict[str, Any]:
    """
    Generate a visualization of topic distribution in conversations.
    
    Args:
        contact_id: Optional contact identifier (if None, analyzes all conversations)
        start_date: Start date for visualization (format: YYYY-MM-DD or "X days/weeks/months ago")
        end_date: End date for visualization (format: YYYY-MM-DD or "X days/weeks/months ago")
        topic_count: Number of top topics to include
        visualization_type: Type of visualization (pie, bar, treemap)
        format: Output format (svg, png, json)
        
    Returns:
        Dictionary with topic distribution visualization data or image
    """
    try:
        # Validate parameters
        if topic_count < 1 or topic_count > 20:
            return error_response("Topic count must be between 1 and 20")
        
        if visualization_type not in ["pie", "bar", "treemap"]:
            return error_response("Visualization type must be 'pie', 'bar', or 'treemap'")
        
        if format not in ["svg", "png", "json"]:
            return error_response("Format must be 'svg', 'png', or 'json'")
        
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
        
        # Get topic distribution data
        topics_data = await db.get_topic_distribution(
            contact_id=contact_id,
            start_date=start_date_obj,
            end_date=end_date_obj,
            topic_count=topic_count,
        )
        
        # Generate the visualization if not returning JSON
        visualization = None
        if format != "json":
            visualization = await db.generate_topic_visualization(
                topics=topics_data.get("topics", []),
                contact_info=contact_data,
                visualization_type=visualization_type,
                format=format,
            )
        
        # Prepare the result based on format
        if format == "json":
            # Return the topic data for client-side visualization
            result = {
                "date_range": {
                    "start": start_date_obj.isoformat() if start_date_obj else None,
                    "end": end_date_obj.isoformat() if end_date_obj else None,
                },
                "topics": topics_data.get("topics", []),
                "total_messages_analyzed": topics_data.get("total_messages", 0),
                "uncategorized_percentage": topics_data.get("uncategorized_percentage", 0),
            }
            
            if contact_data:
                result["contact"] = contact_data
        else:
            # Return the visualization image (encoded as base64)
            result = {
                "date_range": {
                    "start": start_date_obj.isoformat() if start_date_obj else None,
                    "end": end_date_obj.isoformat() if end_date_obj else None,
                },
                "visualization_type": visualization_type,
                "format": format,
                "image_data": visualization.get("image_data"),
                "mime_type": f"image/{format}",
            }
            
            if contact_data:
                result["contact"] = contact_data
        
        return success_response(result)
    except DatabaseError as e:
        logger.error(f"Database error in visualize_topic_distribution_tool: {e}")
        return error_response(e)
    except Exception as e:
        logger.error(f"Unexpected error in visualize_topic_distribution_tool: {e}")
        logger.error(traceback.format_exc())
        return error_response(ToolExecutionError(f"Failed to visualize topic distribution: {str(e)}"))


@register_tool(
    name="visualize_sentiment_trends",
    description="Generate a visualization of sentiment trends over time",
)
async def visualize_sentiment_trends_tool(
    contact_id: Optional[str] = None,
    topic: Optional[str] = None,
    interval: str = "week",  # day, week, month
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    format: str = "svg",
) -> Dict[str, Any]:
    """
    Generate a visualization of sentiment trends over time.
    
    Args:
        contact_id: Optional contact identifier (if None, analyzes all conversations)
        topic: Optional topic to filter by
        interval: Time interval for aggregation (day, week, month)
        start_date: Start date for visualization (format: YYYY-MM-DD or "X days/weeks/months ago")
        end_date: End date for visualization (format: YYYY-MM-DD or "X days/weeks/months ago")
        format: Output format (svg, png, json)
        
    Returns:
        Dictionary with sentiment trends visualization data or image
    """
    try:
        # Validate parameters
        if interval not in ["day", "week", "month"]:
            return error_response("Interval must be 'day', 'week', or 'month'")
        
        if format not in ["svg", "png", "json"]:
            return error_response("Format must be 'svg', 'png', or 'json'")
        
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
        
        # Get sentiment trend data
        sentiment_data = await db.get_sentiment_trends(
            contact_id=contact_id,
            topic=topic,
            interval=interval,
            start_date=start_date_obj,
            end_date=end_date_obj,
        )
        
        # Generate the visualization if not returning JSON
        visualization = None
        if format != "json":
            visualization = await db.generate_sentiment_visualization(
                sentiment_data=sentiment_data.get("trends", []),
                contact_info=contact_data,
                topic=topic,
                interval=interval,
                format=format,
            )
        
        # Prepare the result based on format
        if format == "json":
            # Return the sentiment data for client-side visualization
            result = {
                "date_range": {
                    "start": start_date_obj.isoformat() if start_date_obj else None,
                    "end": end_date_obj.isoformat() if end_date_obj else None,
                },
                "interval": interval,
                "trends": sentiment_data.get("trends", []),
                "overall_sentiment": sentiment_data.get("overall_sentiment", 0),
                "total_messages_analyzed": sentiment_data.get("total_messages", 0),
            }
            
            if contact_data:
                result["contact"] = contact_data
            
            if topic:
                result["topic"] = topic
        else:
            # Return the visualization image (encoded as base64)
            result = {
                "date_range": {
                    "start": start_date_obj.isoformat() if start_date_obj else None,
                    "end": end_date_obj.isoformat() if end_date_obj else None,
                },
                "interval": interval,
                "format": format,
                "image_data": visualization.get("image_data"),
                "mime_type": f"image/{format}",
            }
            
            if contact_data:
                result["contact"] = contact_data
            
            if topic:
                result["topic"] = topic
        
        return success_response(result)
    except DatabaseError as e:
        logger.error(f"Database error in visualize_sentiment_trends_tool: {e}")
        return error_response(e)
    except Exception as e:
        logger.error(f"Unexpected error in visualize_sentiment_trends_tool: {e}")
        logger.error(traceback.format_exc())
        return error_response(ToolExecutionError(f"Failed to visualize sentiment trends: {str(e)}"))
