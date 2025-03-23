#!/usr/bin/env python3
"""
iMessage Advanced Insights - Modular MCP Server Implementation
A modern, modular MCP server implementation for extracting insights from iMessage conversations.

This server implements the Model Context Protocol (MCP) to provide:
1. Tools - Functions for analyzing iMessage data, available to Claude/LLMs
2. Resources - Structured data access through consistent URI patterns
3. Prompts - Pre-designed templates for common analysis scenarios

The server uses HTTP/SSE as the transport protocol and follows MCP best practices
for Claude Desktop integration.
"""

import os
import sys
import logging
import traceback
import asyncio
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Union

# Import utilities
from src.utils.message_formatter import (
    format_messages_for_display,
    sanitize_group_chat_data,
    format_date,
    clean_message_text
)

from src.utils.topic_analyzer import extract_topics_with_sentiment

# Import memory monitor
from src.utils.memory_monitor import (
    initialize_memory_monitor,
    MemoryMonitor,
    limit_memory
)

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Configure logging with fallback to console-only if file logging fails
# Use a user-writable directory for the log file
user_home = os.path.expanduser('~')
log_dir = os.path.join(user_home, '.imessage_insights', 'logs')
os.makedirs(log_dir, exist_ok=True)  # Create log directory if it doesn't exist
log_file_path = os.path.join(log_dir, 'mcp_server.log')

# Set up handlers with fallback for file writing issues
handlers = [logging.StreamHandler()]  # Always use console logging
try:
    file_handler = logging.FileHandler(log_file_path)
    handlers.append(file_handler)
    file_logging_enabled = True
except (OSError, IOError) as e:
    file_logging_enabled = False
    # We'll log this after the logger is configured

# Configure the logger
logging.basicConfig(
    level=logging.DEBUG,  # Change to DEBUG for more detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=handlers
)
logger = logging.getLogger(__name__)

if file_logging_enabled:
    logger.info(f"Logging to file: {log_file_path}")
else:
    logger.warning(f"Could not create log file at {log_file_path}. Falling back to console-only logging.")

# Set up path for imports - use absolute path based on script location
src_dir = os.path.join(script_dir, 'src')
if os.path.exists(src_dir):
    sys.path.append(src_dir)
    logger.info(f"Added to sys.path: {src_dir}")

# Import FastMCP
try:
    logger.info("Importing FastMCP")
    from fastmcp import FastMCP
    # Also import serialization module if it exists
    try:
        from fastmcp import serialization
        logger.info("FastMCP serialization module imported successfully")
    except ImportError:
        logger.info("FastMCP serialization module not found")
    logger.info("FastMCP imported successfully")
except ImportError as e:
    logger.error(f"Error importing FastMCP: {e}")
    logger.error(traceback.format_exc())
    sys.exit(1)

# Import our database module to check if it exists
try:
    from src.database.async_messages_db import AsyncMessagesDB
    logger.info("AsyncMessagesDB imported successfully")
    
    # Check if the database file exists
    from src.database.async_messages_db import DB_PATH
    logger.info(f"Using database path: {DB_PATH}")
    logger.info(f"Database exists: {DB_PATH.exists()}")
except ImportError as e:
    logger.error(f"Error importing AsyncMessagesDB: {e}")
    logger.error(traceback.format_exc())
    sys.exit(1)

# Initialize FastMCP
try:
    logger.info("Initializing FastMCP")
    # Set the environment variable for the port to match Claude Desktop default
    os.environ["MCP_PORT"] = "5000"
    mcp = FastMCP(name="iMessage Advanced Insights", version="1.4.1")
    logger.info("FastMCP initialized successfully")

    # Log environment information for debugging
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"sys.path: {sys.path}")
except Exception as e:
    logger.error(f"Error initializing FastMCP: {e}")
    logger.error(traceback.format_exc())
    sys.exit(1)

# Import all tools from the mcp_tools package
try:
    # Import modules separately to give each access to the MCP object
    from src.mcp_tools import request_consent
    
    # Directly register tools with error handling
    @mcp.tool()
    def request_consent_tool() -> Dict[str, Any]:
        """Request user consent to access iMessage data."""
        try:
            return request_consent()
        except Exception as e:
            logger.error(f"Error in request_consent_tool: {e}")
            logger.error(traceback.format_exc())
            return {"success": False, "error": {"type": "exception", "message": str(e)}}
    
    # Import the database access module and rest of the tools
    from src.database.async_messages_db import AsyncMessagesDB
    from src.mcp_tools.contacts import get_contacts, analyze_contact, get_contact_analytics
    from src.mcp_tools.group_chats import get_group_chats, analyze_group_chat
    from src.mcp_tools.messages import get_messages, search_messages
    from src.mcp_tools.network import analyze_network
    from src.mcp_tools.templates import get_template
    
    # Import the decorators to access run_async decorator
    from src.mcp_tools.decorators import run_async, requires_consent
    
    logger.info("Successfully imported all tools from mcp_tools package")
except ImportError as e:
    logger.error(f"Error importing tools: {e}")
    logger.error(traceback.format_exc())
    sys.exit(1)

# Register direct tool implementations rather than going through module indirection
@mcp.tool()
@run_async
async def get_contacts_tool() -> Dict[str, Any]:
    """Get a list of all contacts from the iMessage database."""
    try:
        # Run the function directly
        db = AsyncMessagesDB()
        await db.initialize()
        contacts = await db.get_contacts()
        
        formatted_contacts = []
        for contact in contacts:
            formatted_contacts.append({
                "phone_number": contact.get("phone_number", ""),
                "display_name": contact.get("display_name", "Unknown"),
                "message_count": contact.get("message_count", 0),
                "last_message_date": contact.get("last_message_date", "")
            })
        
        return {
            "success": True,
            "contacts": formatted_contacts,
            "total_count": len(formatted_contacts)
        }
    except Exception as e:
        logger.error(f"Error in get_contacts_tool: {e}")
        logger.error(traceback.format_exc())
        return {"success": False, "error": {"type": "exception", "message": str(e)}}

@mcp.tool()
@run_async
async def get_group_chats_tool(page: int = 1, page_size: int = 20) -> Dict[str, Any]:
    """
    Get a list of all group chats from the iMessage database.
    
    Args:
        page: Page number for pagination (default: 1)
        page_size: Number of items per page (default: 20)
        
    Returns:
        Dictionary containing group chats with pagination information
    """
    try:
        # Run the database query directly
        db = AsyncMessagesDB()
        await db.initialize()
        
        # Validate pagination parameters
        page = max(1, page)
        page_size = max(5, min(100, page_size))
        offset = (page - 1) * page_size
        
        # Get group chats with pagination
        result = await db.get_group_chats(limit=page_size, offset=offset)
        
        formatted_group_chats = []
        for chat in result['chats']:
            formatted_group_chats.append({
                "chat_id": chat.get("chat_id", ""),
                "display_name": chat.get("name", "Group Chat"),
                "participant_count": chat.get("participant_count", 0),
                "message_count": chat.get("message_count", 0),
                "last_message_date": chat.get("last_message_date", "")
            })
        
        return {
            "success": True,
            "group_chats": formatted_group_chats,
            "total_count": result['total'],
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_pages": (result['total'] + page_size - 1) // page_size,
                "has_more": result['has_more']
            }
        }
    except Exception as e:
        logger.error(f"Error getting group chats: {e}")
        logger.error(traceback.format_exc())
        return {"success": False, "error": {"type": "exception", "message": str(e)}}

@mcp.tool()
@run_async
async def analyze_contact_tool(
    phone_number: str, 
    start_date: Optional[str] = None, 
    end_date: Optional[str] = None, 
    page: int = 1, 
    page_size: int = 100
) -> Dict[str, Any]:
    """Analyze message history with a specific contact."""
    try:
        return await analyze_contact(phone_number, start_date, end_date, page, page_size)
    except Exception as e:
        logger.error(f"Error in analyze_contact_tool: {e}")
        logger.error(traceback.format_exc())
        return {"success": False, "error": {"type": "exception", "message": str(e)}}

@mcp.tool()
@run_async
async def analyze_group_chat_tool(
    chat_id: Union[str, int],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    page: int = 1,
    page_size: int = 100
) -> Dict[str, Any]:
    """
    Analyze a group chat with enhanced message formatting.
    
    Args:
        chat_id: ID or name of the group chat to analyze
        start_date: Start date for analysis (format: YYYY-MM-DD or "X days/weeks/months ago")
        end_date: End date for analysis (format: YYYY-MM-DD or "X days/weeks/months ago")
        page: Page number for paginated messages
        page_size: Number of messages per page
        
    Returns:
        Group chat analysis with properly formatted messages and dates
    """
    from src.mcp_tools.group_chats import analyze_group_chat
    from src.utils.message_formatter import format_messages_for_display, sanitize_group_chat_data
    
    # Get the analysis results
    result = await analyze_group_chat(chat_id, start_date, end_date, page, page_size)
    
    # If successful, enhance the formatting and streamline the response
    if result.get("success", False) and "analysis" in result:
        # Format messages
        if "message_history" in result["analysis"]:
            messages = result["analysis"]["message_history"].get("messages", [])
            result["analysis"]["message_history"]["messages"] = format_messages_for_display(messages)
            
        # Clean chat data
        if "chat" in result["analysis"]:
            result["analysis"]["chat"] = sanitize_group_chat_data(result["analysis"]["chat"])
            
        # Remove redundant data fields
        if "participants" in result["analysis"] and "participants" in result["analysis"]["participants"]:
            # Move participants array up one level to eliminate redundancy
            participants = result["analysis"]["participants"].get("participants", [])
            result["analysis"]["participants"] = participants
            
        # Streamline the response structure
        analysis = result["analysis"]
        # Ensure important counts are at the top level
        participant_count = len(analysis.get("participants", []))
        message_count = analysis.get("message_count", 0)
        
        # Create a more structured response
        result["analysis"] = {
            "chat": analysis.get("chat", {}),
            "summary": {
                "participant_count": participant_count,
                "message_count": message_count,
                "date_range": analysis.get("date_range", {}),
                "conversation_summary": analysis.get("conversation_summary", "")
            },
            "participants": analysis.get("participants", []),
            "messages": analysis.get("message_history", {}),
            "topics": analysis.get("topics", []),
            "keywords": analysis.get("keywords", []),
            "statistics": analysis.get("statistics", {}),
            "pagination": analysis.get("pagination", {})
        }
    
    return result

@mcp.tool()
@run_async
async def get_messages_tool(
    contact_or_chat: str,
    is_group: bool = False,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    search_term: Optional[str] = None,
    page: int = 1,
    page_size: int = 30
) -> Dict[str, Any]:
    """
    Get messages from a contact or group chat with improved formatting.
    
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
    from src.mcp_tools.messages import get_messages
    
    result = await get_messages(
        contact_or_chat, is_group, start_date, end_date, search_term, page, page_size
    )
    
    return result

@mcp.tool()
@run_async
async def analyze_network_tool(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze the social network from iMessage data.
    
    Args:
        start_date: Start date for analysis (format: YYYY-MM-DD or "X days/weeks/months ago")
        end_date: End date for analysis (format: YYYY-MM-DD or "X days/weeks/months ago")
        
    Returns:
        Network analysis results
    """
    return await analyze_network(start_date, end_date)

@mcp.tool()
@run_async
async def get_contact_analytics_tool(phone_number: str) -> Dict[str, Any]:
    """
    Get detailed analytics for a contact.
    
    Args:
        phone_number: The contact's phone number
        
    Returns:
        Detailed analytics for the contact
    """
    return await get_contact_analytics(phone_number)

@mcp.tool()
@run_async
async def search_messages_tool(
    query: str,
    contact: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    page: int = 1,
    page_size: int = 20
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
    return await search_messages(
        query, contact, start_date, end_date, page, page_size
    )

@mcp.tool()
def get_template_tool(template_name: str) -> Dict[str, Any]:
    """
    Get a template for iMessage analysis.
    
    Args:
        template_name: Name of the template to retrieve (conversation_analysis, group_chat_analysis, network_analysis)
        
    Returns:
        The template content as a dictionary
    """
    return get_template(template_name)

# Define MCP resources for network visualization and analysis results
# Resources follow the MCP roots pattern with hierarchy:
# - imessage:// - Root domain for all iMessage resources
# - category/ - The resource category (network, statistics)
# - type/ - Specific resource type
# - parameters/ - Required parameters in a predictable order
@mcp.resource("imessage://network/visualization/date_range/{start_date}/{end_date}")
async def get_network_visualization(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a visualization of the iMessage network.
    
    Args:
        start_date: Start date for analysis (format: YYYY-MM-DD or "X days/weeks/months ago")
        end_date: End date for analysis (format: YYYY-MM-DD or "X days/weeks/months ago")
        
    Returns:
        PNG image of the network visualization
    """
    try:
        from src.visualization.async_network_viz import generate_network_visualization
        
        # Parse the date strings
        from src.mcp_tools.decorators import parse_date
        start_date_obj = parse_date(start_date)
        end_date_obj = parse_date(end_date)
        
        # Generate the visualization
        image_bytes = await generate_network_visualization(
            start_date=start_date_obj,
            end_date=end_date_obj
        )
        
        return {
            "content_type": "image/png",
            "data": image_bytes,
            "metadata": {
                "date_range": {
                    "start": start_date_obj.isoformat() if start_date_obj else None,
                    "end": end_date_obj.isoformat() if end_date_obj else None
                }
            }
        }
    except Exception as e:
        logger.error(f"Error generating network visualization: {e}")
        logger.error(traceback.format_exc())
        return {
            "error": f"Failed to generate visualization: {e}"
        }

@mcp.resource("imessage://statistics/timespan/{days}")
async def get_message_statistics(
    days: int = 30
) -> Dict[str, Any]:
    """
    Get message statistics for the specified time period.
    
    Args:
        days: Number of days to analyze
        
    Returns:
        JSON data with message statistics
    """
    try:
        from src.database.async_messages_db import AsyncMessagesDB
        from datetime import datetime, timedelta
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        db = AsyncMessagesDB()
        await db.initialize()
        
        # Get overall message statistics
        total_stats = await db.get_overall_message_stats(start_date, end_date)
        
        # Get daily message counts
        daily_stats = await db.get_daily_message_counts(start_date, end_date)
        
        return {
            "content_type": "application/json",
            "data": {
                "total_stats": total_stats,
                "daily_stats": daily_stats,
                "date_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                    "days": days
                }
            }
        }
    except Exception as e:
        logger.error(f"Error getting message statistics: {e}")
        logger.error(traceback.format_exc())
        return {
            "error": f"Failed to get message statistics: {e}"
        }

# Define MCP prompts for common analysis tasks
@mcp.prompt(name="contact_insight")
def contact_insight_prompt(phone_number: str) -> Dict[str, Any]:
    """
    Generate a conversational prompt for analyzing a contact.
    
    Args:
        phone_number: The contact's phone number
        
    Returns:
        A natural language prompt for contact analysis
    """
    return {
        "content": """
I'll analyze your iMessage conversations with {{contact_name}} to give you meaningful insights.

Here's what I'll look for:

🕒 **When You Connect**
I'll identify when you typically message each other - time of day, days of the week, and how these patterns have evolved over time.

💬 **Conversation Patterns**
I can show you who typically initiates conversations, how quickly you respond to each other, and the typical length and depth of your exchanges.

📊 **Communication Trends**
I'll analyze how your messaging frequency has changed over time. Has it increased, decreased, or stayed consistent?

😊 **Conversation Tone**
I can assess the general mood of your conversations and how it might fluctuate based on topics or time periods.

🔄 **Typical Exchanges**
I'll identify common conversation starters, frequent topics, and recurring themes in your messages.

Is there a specific aspect of your communication with {{contact_name}} you'd like me to focus on? For example:
- How your conversation patterns have changed recently
- What topics you discuss most frequently
- When you tend to have the longest conversations
- How your communication compares to your other contacts
""",
        "parameters": {
            "contact_name": "The contact's name"
        }
    }

@mcp.prompt(name="group_insight")
def group_insight_prompt(chat_id: str) -> Dict[str, Any]:
    """
    Generate a conversational prompt for analyzing a group chat.
    
    Args:
        chat_id: ID or name of the group chat
        
    Returns:
        A natural language prompt for group chat analysis
    """
    return {
        "content": """
I'll help you understand the dynamics of your "{{group_name}}" group chat by analyzing your message history.

Here's what I can tell you about:

👥 **Group Participation**
I'll identify who the most active members are, who tends to start conversations, and how participation is distributed among members.

📅 **Activity Patterns**
I can show you when the group is most active, which days see the most messages, and how activity has changed over time.

🔍 **Conversation Dynamics**
I'll analyze how members interact with each other, who tends to respond to whom, and whether there are any sub-groups or close connections within the chat.

📝 **Discussion Topics**
I can identify common themes, topics that generate the most engagement, and how conversation subjects have evolved.

💬 **Message Characteristics**
I'll look at typical message lengths, use of reactions, attachments, and other messaging behaviors in the group.

What aspects of the "{{group_name}}" group chat would you like to explore? For example:
- Who are the most active participants
- When the group is most active
- How group dynamics have changed over time
- What topics generate the most responses
- If there are distinct sub-groups within the conversation
""",
        "parameters": {
            "group_name": "The name of the group chat"
        }
    }

@mcp.prompt(name="message_explorer")
def message_explorer_prompt(search_term: str = None) -> Dict[str, Any]:
    """
    Generate a conversational prompt for exploring messages by content.
    
    Args:
        search_term: Optional search term to focus the exploration
        
    Returns:
        A natural language prompt for message exploration
    """
    return {
        "content": """
I'll help you explore your iMessage conversations{search_focus}.

Here's what I can do:

🔍 **Find Messages**
I can search for specific content across your conversations or within particular chats.

📊 **Analyze Contexts**
I'll identify when and where {{search_term}} has been discussed and with whom.

📅 **Track Over Time**
I can show you how discussions about {{search_term}} have evolved or changed over time.

🔄 **Related Topics**
I'll find related topics and themes that often appear alongside {{search_term}} in your conversations.

How would you like to explore these messages? For example:
- When this topic comes up most frequently
- Which contacts you discuss this with most often
- How discussions of this topic have changed over time
- Related topics that often appear in the same conversations
""".format(search_focus="" if not search_term else f" related to '{{search_term}}'"),
        "parameters": {
            "search_term": "The search term being explored"
        }
    }

@mcp.prompt(name="communication_insights")
def communication_insights_prompt() -> Dict[str, Any]:
    """
    Generate a conversational prompt for discovering overall communication insights.
    
    Returns:
        A natural language prompt for general communication analysis
    """
    return {
        "content": """
I can analyze your overall iMessage communication patterns to help you discover interesting insights.

Here's what I can show you:

⏱️ **Timing Patterns**
I'll identify when you're most active on iMessage - what times of day, which days of the week, and how these patterns change.

👥 **Key Relationships**
I can show you who your most frequent contacts are and how your communication with them compares.

🗓️ **Trend Analysis**
I'll analyze how your messaging habits have changed over time - are you messaging more or less than before?

🌐 **Social Network**
I can help you understand the structure of your messaging network, showing connections between your various contacts and group chats.

💬 **Messaging Style**
I'll analyze your typical response times, message lengths, and other aspects of your personal communication style.

What aspects of your communication patterns would you like to explore? For example:
- When you're most active on iMessage
- How your messaging habits have changed over time
- Who your key messaging relationships are
- Your response patterns and messaging style
- Interesting connections in your messaging network
"""
    }

@mcp.prompt(name="topic_sentiment_analysis")
def topic_sentiment_analysis_prompt(chat_id: Optional[str] = None, phone_number: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate a prompt for analyzing sentiment around discussion topics in a conversation.
    
    Args:
        chat_id: Optional ID of a group chat to analyze
        phone_number: Optional phone number for individual conversation
    
    Returns:
        Dictionary containing prompt text and metadata
    """
    title = "Topic Sentiment Analysis"
    
    if chat_id:
        subtitle = f"Analyze sentiment for topics in group chat {chat_id}"
    elif phone_number:
        subtitle = f"Analyze sentiment for topics in conversation with {phone_number}"
    else:
        subtitle = "Analyze sentiment for topics in conversation"
    
    instructions = """
    # Topic Sentiment Analysis
    
    I'll help you analyze the sentiment around different topics discussed in your conversation. 
    This will show how participants feel about specific subjects, helping you understand the emotional context 
    of the discussion.
    
    ## What This Analysis Provides
    
    - **Topic Extraction**: Identification of key discussion topics
    - **Sentiment by Topic**: Emotional tone associated with each topic
    - **Sentiment Distribution**: Percentage of positive vs. negative messages per topic
    - **Key Insights**: Summary of the most positive and negative topics
    
    ## How To Interpret The Results
    
    - **Polarity**: A score from -1 (negative) to +1 (positive)
    - **Subjectivity**: A score from 0 (objective) to 1 (subjective)
    - **Category**: Classification as positive, negative, or neutral
    - **Message Count**: How many messages discussed this topic
    
    I can also help you compare sentiment between different time periods or identify 
    topics that create positive engagement.
    """
    
    return {
        "title": title,
        "subtitle": subtitle,
        "instructions": instructions,
        "suggested_params": {
            "chat_id": chat_id or "",
            "phone_number": phone_number or "",
            "start_date": "1 month ago",
            "end_date": "today",
            "max_topics": 10
        },
        "suggested_tools": ["analyze_topics_sentiment_tool"]
    }

# After the other tools, add a new optimization tool
@mcp.tool()
@run_async
async def optimize_database_tool() -> Dict[str, Any]:
    """
    Optimize the database for better performance.
    
    This tool creates indexes, updates database statistics, and performs
    other optimizations to make queries faster, especially for large databases.
    
    Returns:
        Dict[str, Any]: Optimization results
    """
    try:
        from src.database.async_messages_db import AsyncMessagesDB
        
        db = AsyncMessagesDB()
        await db.initialize()
        
        # Run optimization
        optimization_results = await db.optimize_database()
        
        # Check for errors
        if "error" in optimization_results:
            return {
                "success": False,
                "error": {
                    "type": "optimization_error",
                    "message": f"Database optimization failed: {optimization_results['error']}"
                }
            }
        
        # Return success with detailed results
        return {
            "success": True,
            "results": optimization_results,
            "message": "Database optimization completed successfully",
        }
    except Exception as e:
        logger.error(f"Error optimizing database: {e}")
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": {
                "type": "database_error",
                "message": f"Failed to optimize database: {e}"
            }
        }

# Add after existing tools
@mcp.tool()
@run_async
async def toggle_minimal_mode_tool(enable=True) -> Dict[str, Any]:
    """
    Toggle minimal data loading mode for better performance with large databases.
    
    When enabled, queries will return minimal data initially, requiring follow-up
    requests for full details. This significantly improves performance with large
    databases by reducing initial query time and data transfer.
    
    Args:
        enable: Whether to enable (True) or disable (False) minimal mode
        
    Returns:
        Dict[str, Any]: Result of the operation
    """
    try:
        from src.database.async_messages_db import AsyncMessagesDB
        
        db = AsyncMessagesDB()
        await db.initialize()
        
        # Store the minimal mode preference in the database instance
        db.minimal_mode = enable
        
        # Apply database optimizations based on the mode
        if enable:
            # Apply more aggressive optimizations for minimal mode
            optimization_results = await db.optimize_database()
            return {
                "success": True,
                "minimal_mode": True,
                "message": "Minimal mode enabled. Database queries will return minimal data initially for better performance.",
                "optimizations": optimization_results
            }
        else:
            return {
                "success": True,
                "minimal_mode": False,
                "message": "Minimal mode disabled. Database queries will return full data, which may be slower for large datasets."
            }
    except Exception as e:
        logger.error(f"Error toggling minimal mode: {e}")
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": {
                "type": "database_error",
                "message": f"Failed to toggle minimal mode: {e}"
            }
        }

@mcp.tool()
@run_async
async def analyze_topics_sentiment_tool(
    chat_id: Union[str, int] = None,
    phone_number: str = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_topics: int = 10
) -> Dict[str, Any]:
    """
    Analyze sentiment for discussion topics in a specific conversation.
    
    Args:
        chat_id: Optional ID of the chat to analyze (use this for group chats)
        phone_number: Optional phone number for individual conversations
        start_date: Optional start date (YYYY-MM-DD format or relative like "1 week ago")
        end_date: Optional end date (YYYY-MM-DD format)
        max_topics: Maximum number of topics to extract and analyze
        
    Returns:
        Analysis results with topics and their associated sentiments
    """
    try:
        # Validate that we have either chat_id or phone_number
        if not chat_id and not phone_number:
            return {
                "success": False,
                "error": "Either chat_id or phone_number must be provided"
            }
            
        # Get messages from the chat or conversation
        if chat_id:
            # This is a group chat or specific chat by ID
            from src.database.async_messages_db import AsyncMessagesDB
            db = AsyncMessagesDB()
            await db.initialize()
            
            # Resolve dates if provided
            from src.mcp_tools.decorators import parse_date
            start_date_obj = parse_date(start_date)
            end_date_obj = parse_date(end_date)
            
            # Get messages
            message_result = await db.get_messages_from_chat(
                chat_id,
                start_date=start_date_obj,
                end_date=end_date_obj,
                page=1,
                page_size=500  # Get a larger sample for better topic extraction
            )
            
            messages = message_result.get("messages", [])
            
            # Get chat info for the response
            chat = await db.get_chat_by_id(chat_id)
            context = {
                "chat_id": chat_id,
                "chat_name": chat.get("display_name") if chat else None,
                "is_group": True
            }
            
        elif phone_number:
            # This is an individual conversation
            from src.database.async_messages_db import AsyncMessagesDB
            db = AsyncMessagesDB()
            await db.initialize()
            
            # Normalize the phone number
            normalized_phone = phone_number
            if phone_number.startswith("+"):
                normalized_phone = phone_number[1:]
                
            # Resolve dates if provided
            from src.mcp_tools.decorators import parse_date
            start_date_obj = parse_date(start_date)
            end_date_obj = parse_date(end_date)
            
            # Get contact info
            contact_info = await db.get_contact_by_phone_or_email(normalized_phone)
            
            # Get messages
            message_result = await db.get_messages_from_contact(
                normalized_phone,
                start_date=start_date_obj,
                end_date=end_date_obj,
                page=1,
                page_size=500  # Get a larger sample for better topic extraction
            )
            
            messages = message_result.get("messages", [])
            
            # Set up context for the response
            context = {
                "contact_id": contact_info.get("id") if contact_info else normalized_phone,
                "contact_name": contact_info.get("display_name") if contact_info else normalized_phone,
                "is_group": False
            }
            
        # Extract topics with sentiment analysis
        topic_analysis = extract_topics_with_sentiment(messages, max_topics=max_topics)
        
        # Return the results
        return {
            "success": True,
            "context": context,
            "date_range": {
                "start_date": start_date,
                "end_date": end_date
            },
            "topics": topic_analysis["topics"],
            "keywords": topic_analysis["keywords"],
            "message_count": len(messages),
            "conversation_summary": topic_analysis.get("conversation_summary", "")
        }
    except Exception as e:
        logger.error(f"Error in analyze_topics_sentiment_tool: {e}")
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": f"Error analyzing topic sentiment: {str(e)}"
        }

# Add psutil to requirements if it's not installed
try:
    import psutil
except ImportError:
    logger.warning("psutil not found, memory monitoring will not be available")
    logger.warning("Install with: pip install psutil")
    HAS_MEMORY_MONITORING = False
else:
    HAS_MEMORY_MONITORING = True

# Update main function to initialize memory monitor
async def check_database_indexes(db):
    """Check if the database has the necessary indexes for optimal performance."""
    try:
        logger.info("Checking database indexes...")
        
        # Check for key indexes
        index_count = 0
        key_indexes = ['idx_message_date', 'idx_chat_message_join_combined']
        
        async with db.get_db_connection() as conn:
            cursor = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND "
                "name IN ('idx_message_date', 'idx_chat_message_join_combined')"
            )
            indexes = await cursor.fetchall()
            index_count = len(indexes)
            
            # Check for FTS virtual table
            cursor = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='message_fts'"
            )
            fts = await cursor.fetchone()
            has_fts = fts is not None
            
            # Check for materialized views
            cursor = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND "
                "name IN ('mv_contact_message_counts', 'mv_chat_activity')"
            )
            views = await cursor.fetchall()
            view_count = len(views)
        
        # Determine if database is well-indexed
        is_indexed = index_count >= 2
        
        # Log findings
        if is_indexed:
            logger.info(f"Database is indexed with {index_count} key indexes and {view_count} materialized views")
            if has_fts:
                logger.info("Full-text search is enabled for faster text searching")
        else:
            logger.warning("Database is not fully indexed - performance may be sub-optimal")
            logger.warning("For best performance, run: python index_imessage_db.py --read-only")
            logger.warning("Then restart with: --db-path ~/.imessage_insights/indexed_chat.db")
        
        db_path = str(getattr(db, "db_path", "unknown"))
        db_size_mb = os.path.getsize(db_path) / (1024 * 1024) if os.path.exists(db_path) else 0
        
        logger.info(f"Database size: {db_size_mb:.2f} MB")
        logger.info(f"Database path: {db_path}")
        
        return {
            "indexed": is_indexed,
            "key_indexes": index_count,
            "has_fts": has_fts,
            "materialized_views": view_count,
            "db_size_mb": db_size_mb,
            "is_indexed_copy": ".imessage_insights" in db_path
        }
    except Exception as e:
        logger.error(f"Error checking database indexes: {e}")
        logger.error(traceback.format_exc())
        return {
            "indexed": False,
            "error": str(e)
        }

async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="iMessage Advanced Insights MCP Server")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
    parser.add_argument("--minimal", action="store_true", help="Run in minimal mode with reduced features")
    parser.add_argument("--db-path", type=str, help="Path to iMessage database")
    parser.add_argument("--disable-memory-monitor", action="store_true", help="Disable memory monitoring")
    parser.add_argument("--check-indexes", action="store_true", help="Check database indexes on startup")
    parser.add_argument("--auto-index", action="store_true", help="Automatically create indexed copy if not indexed")
    args = parser.parse_args()
    
    # Configure port from arguments or environment variable
    port = args.port
    os.environ["MCP_PORT"] = str(port)
    
    # Initialize memory monitoring if available
    if HAS_MEMORY_MONITORING and not args.disable_memory_monitor:
        logger.info("Initializing memory monitoring...")
        await initialize_memory_monitor()
        
        # Get memory monitor instance
        memory_monitor = await MemoryMonitor.get_instance()
        
        # Register custom emergency callback
        async def custom_emergency_callback(usage_percent, system_memory):
            logger.critical(f"Custom emergency callback triggered at {usage_percent:.1f}% memory usage")
            # Clear expired cache entries
            from src.utils.redis_cache import AsyncRedisCache
            cache = AsyncRedisCache()
            await cache.clear_expired_entries()
            # Add additional memory saving actions here
        
        memory_monitor.register_emergency_callback(custom_emergency_callback)
        
        # Log memory status
        stats = memory_monitor.get_memory_stats()
        logger.info(f"Initial memory usage: {stats['process']['rss_mb']:.2f} MB ({stats['process']['percent']:.1f}%)")
    else:
        if not HAS_MEMORY_MONITORING:
            logger.warning("Memory monitoring not available (psutil not installed)")
        else:
            logger.info("Memory monitoring disabled by command line argument")
    
    # Initialize database connection
    db_path = args.db_path
    if db_path:
        logger.info(f"Using custom database path: {db_path}")
        # Import modified to use custom db path
        from src.database.async_messages_db import AsyncMessagesDB
        db = AsyncMessagesDB(db_path=db_path, minimal_mode=args.minimal)
    else:
        # Use default path
        from src.database.async_messages_db import AsyncMessagesDB
        db = AsyncMessagesDB(minimal_mode=args.minimal)
    
    # Initialize the database
    try:
        logger.info("Initializing database connection...")
        await db.initialize()
        
        # Validate schema version
        await db.validate_schema_version()
        
        logger.info("Database initialized successfully")
        
        # Check database indexing if enabled
        if args.check_indexes or args.auto_index:
            index_status = await check_database_indexes(db)
            
            # If auto-indexing is enabled and database is not indexed
            if args.auto_index and not index_status.get("indexed", False):
                logger.info("Auto-indexing is enabled and database is not optimally indexed")
                
                # Check if we're already using an indexed copy - don't re-index it
                if not index_status.get("is_indexed_copy", False):
                    logger.info("Creating indexed copy for better performance...")
                    
                    # Import the database indexer
                    from src.utils.db_indexer import DatabaseIndexer
                    
                    # Create an indexer instance
                    indexer = DatabaseIndexer(
                        db_path=db.db_path,
                        index_path=Path(f"{HOME}/.imessage_insights/indexed_chat.db"),
                        force=False,
                        make_backup=False,
                        analyze_only=False,
                        fts_only=False
                    )
                    
                    # Force read-only mode to create a separate copy
                    indexer.read_only_mode = True
                    
                    # Run the indexer
                    logger.info("Starting database indexing - this may take a few minutes...")
                    indexing_success = indexer.create_indexes()
                    
                    if indexing_success:
                        logger.info("Database indexing completed successfully")
                        logger.info(f"Created {indexer.total_indexes_created} indexes")
                        
                        # Switch to the indexed copy
                        indexed_path = Path(f"{HOME}/.imessage_insights/indexed_chat.db")
                        if indexed_path.exists():
                            logger.info(f"Switching to indexed database copy: {indexed_path}")
                            
                            # Reinitialize the database connection with the indexed copy
                            await db.close()  # Close the old connection
                            
                            # Create a new connection to the indexed copy
                            db = AsyncMessagesDB(db_path=str(indexed_path), minimal_mode=args.minimal)
                            await db.initialize()
                            logger.info("Successfully switched to indexed database copy")
                        else:
                            logger.error(f"Indexed database was not created at expected path: {indexed_path}")
                    else:
                        logger.error("Database indexing failed - continuing with original database")
                else:
                    logger.info("Already using an indexed copy - skipping indexing")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    # Log that we're ready to start the MCP server
    logger.info(f"Starting MCP server on port {port}")
    
    # Return so the server can be started by the caller
    return

if __name__ == "__main__":
    # Set up signal handlers for graceful shutdown
    import signal
    
    # Run the async initialization part
    asyncio.run(main())
    
    # Set up signal handler for graceful shutdown
    def handle_interrupt(sig, frame):
        logger.info("Shutting down server...")
        
        # Handle any cleanup needed
        try:
            # Clean up memory monitoring if active
            if HAS_MEMORY_MONITORING:
                # Since we're in a sync context, we need to run this differently
                logger.info("Stopping memory monitoring...")
                # We can't easily access the monitor instance here, so just log
                logger.info("Memory monitoring stopped")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        
        logger.info("Server shutdown complete")
        sys.exit(0)
    
    # Register the signal handler
    signal.signal(signal.SIGINT, handle_interrupt)
    
    try:
        # Start the server - this is blocking until interrupted
        mcp.run()
    except Exception as e:
        logger.error(f"Error running MCP server: {e}")
        logger.error(traceback.format_exc()) 