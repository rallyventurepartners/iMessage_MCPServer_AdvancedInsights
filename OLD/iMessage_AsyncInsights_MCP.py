#!/usr/bin/env python3
"""
iMessage_AsyncInsights_MCP.py - Asynchronous FastMCP Implementation of iMessage Analytics

This module implements an asynchronous version of the iMessage Analytics toolset using FastMCP.
It provides various tools for analyzing iMessage data, including contact networks, sentiment analysis,
and message visualization, all optimized with async/await patterns for improved performance.

Usage:
    - Direct execution: python3 iMessage_AsyncInsights_MCP.py
    - FastMCP dev mode: fastmcp dev iMessage_AsyncInsights_MCP.py
    - Integration with Claude: fastmcp install iMessage_AsyncInsights_MCP.py

Requirements:
    - FastMCP
    - aiosqlite
    - asyncio
    - Other packages listed in requirements.txt

Note: 
    This implementation provides significant performance improvements over the synchronous version,
    especially for large message databases and complex analysis operations.
"""

import os
import sys
import logging
import traceback
import threading
import re
import asyncio
import inspect
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import dateutil.relativedelta
from functools import wraps
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

logger.info(f"Current directory: {os.getcwd()}")

try:
    logger.info("Importing FastMCP")
    from fastmcp import FastMCP
    logger.info("Importing datetime modules")
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error(traceback.format_exc())
    sys.exit(1)

# Import our async database module
try:
    from database.async_messages_db import AsyncMessagesDB
except ImportError:
    # Create a symlink for imports to work
    src_dir = Path('src')
    if src_dir.exists():
        sys.path.append(str(src_dir.absolute()))
        try:
            from database.async_messages_db import AsyncMessagesDB
        except ImportError as e:
            logger.error(f"Cannot import AsyncMessagesDB: {e}")
            logger.error(traceback.format_exc())
            sys.exit(1)

try:
    # Initialize FastMCP server
    logger.info("Initializing FastMCP")
    mcp = FastMCP("iMessage Async Query", dependencies=[
        "phonenumbers", 
        "python-dateutil",
        "aiosqlite",
        "aiohttp",
        "asyncio"
    ])
    logger.info("FastMCP initialized successfully")
except Exception as e:
    logger.error(f"Error initializing FastMCP: {e}")
    logger.error(traceback.format_exc())
    sys.exit(1)

# Default to Messages database in user's Library
DEFAULT_DB_PATH = Path.home() / "Library" / "Messages" / "chat.db"
DB_PATH = Path(os.environ.get('SQLITE_DB_PATH', DEFAULT_DB_PATH))

logger.info(f"Using database path: {DB_PATH}")
logger.info(f"Database exists: {DB_PATH.exists()}")

try:
    class AnalysisError(Exception):
        """Exception raised when message analysis fails."""
        pass
except Exception as e:
    logger.error(f"Error defining AnalysisError class: {e}")
    logger.error(traceback.format_exc())

# Initialize AsyncMessagesDB
try:
    db = AsyncMessagesDB(DB_PATH)
    # We need to explicitly initialize the DB in an async context
    # This will be handled in each tool function that uses the DB
except Exception as e:
    logger.error(f"Error initializing AsyncMessagesDB: {e}")
    logger.error(traceback.format_exc())
    sys.exit(1)

# Helper for running async functions in MCP tools
def run_async(func):
    """
    Decorator to run an async function in a synchronous context.
    This allows us to use async functions with FastMCP tools.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(func(*args, **kwargs))
            return result
        finally:
            loop.close()
    return wrapper

def error_response(error_type: str, message: str) -> Dict[str, Any]:
    """Create a standardized error response."""
    return {
        "status": "error",
        "error": {
            "type": error_type,
            "message": message
        }
    }

@mcp.tool()
@run_async
async def get_contacts() -> Dict[str, Any]:
    """Get all contacts you have messaged.
    
    Returns:
        List of contact dictionaries with phone numbers and names.
    """
    try:
        logger.info("Getting all contacts using async implementation")
        # Initialize the DB if needed
        await db.initialize()
        contacts = await db.get_contacts()
        return {"contacts": contacts}
    except Exception as e:
        logger.error(f"Error retrieving contacts: {e}")
        logger.error(traceback.format_exc())
        return error_response("DATABASE_ERROR", f"Error retrieving contacts: {str(e)}")

@mcp.tool()
@run_async
async def get_group_chats() -> Dict[str, Any]:
    """Get all group chats from the database.
    
    Returns:
        Dictionary containing list of group chats with metadata.
    """
    try:
        logger.info("Getting all group chats using async implementation")
        # Initialize the DB if needed
        await db.initialize()
        group_chats = await db.get_group_chats()
        return {"group_chats": group_chats}
    except Exception as e:
        logger.error(f"Error retrieving group chats: {e}")
        logger.error(traceback.format_exc())
        return error_response("DATABASE_ERROR", f"Error retrieving group chats: {str(e)}")

@mcp.tool()
@run_async
async def analyze_contact(
    phone_number: str,
    start_date: str = None,
    end_date: str = None
) -> Dict[str, Any]:
    """Analyze messages with a specific contact.
    
    Args:
        phone_number: The phone number or email of the contact
        start_date: Optional start date in ISO format (YYYY-MM-DD)
        end_date: Optional end date in ISO format (YYYY-MM-DD)
    
    Returns:
        Dictionary containing contact analysis data
    """
    try:
        logger.info(f"Analyzing contact {phone_number} using async implementation")
        if not phone_number:
            return error_response("PARAMETER_ERROR", "Phone number is required")
        
        # Initialize the DB if needed
        await db.initialize()
        # Perform the analysis asynchronously
        result = await db.analyze_contact(phone_number, start_date, end_date)
        return result
    except Exception as e:
        logger.error(f"Error analyzing contact: {e}")
        logger.error(traceback.format_exc())
        return error_response("ANALYSIS_ERROR", f"Error analyzing contact: {str(e)}")

@mcp.tool()
@run_async
async def analyze_group_chat(
    chat_id: Union[str, int],
    start_date: str = None,
    end_date: str = None
) -> Dict[str, Any]:
    """Analyze messages in a group chat.
    
    Args:
        chat_id: The ID, name, or identifier of the group chat
        start_date: Optional start date in ISO format (YYYY-MM-DD)
        end_date: Optional end date in ISO format (YYYY-MM-DD)
    
    Returns:
        Dictionary containing group chat analysis data
    """
    try:
        logger.info(f"Analyzing group chat {chat_id} using async implementation")
        if not chat_id:
            return error_response("PARAMETER_ERROR", "Chat ID is required")
            
        # Initialize the DB if needed
        await db.initialize()
        # Perform the analysis asynchronously
        result = await db.analyze_group_chat(chat_id, start_date, end_date)
        return result
    except Exception as e:
        logger.error(f"Error analyzing group chat: {e}")
        logger.error(traceback.format_exc())
        return error_response("ANALYSIS_ERROR", f"Error analyzing group chat: {str(e)}")

@mcp.tool()
@run_async
async def analyze_contact_network(
    start_date: str = None,
    end_date: str = None,
    min_shared_chats: int = 1
) -> Dict[str, Any]:
    """Analyze the network of contacts based on group chat participation.
    
    Args:
        start_date: Optional start date in ISO format (YYYY-MM-DD)
        end_date: Optional end date in ISO format (YYYY-MM-DD)
        min_shared_chats: Minimum number of shared chats to consider a connection
    
    Returns:
        Dictionary containing the contact network analysis
    """
    try:
        logger.info(f"Analyzing contact network using async implementation")
        
        # Initialize the DB if needed
        await db.initialize()
        
        # Import analysis module
        from analysis.async_network_analysis import analyze_contact_network_async
        
        # Perform the analysis asynchronously
        result = await analyze_contact_network_async(start_date, end_date, min_shared_chats)
        return result
    except Exception as e:
        logger.error(f"Error analyzing contact network: {e}")
        logger.error(traceback.format_exc())
        return error_response("ANALYSIS_ERROR", f"Error analyzing contact network: {str(e)}")

@mcp.tool()
@run_async
async def visualize_network(
    start_date: str = None,
    end_date: str = None,
    min_shared_chats: int = 1,
    layout: str = "spring"
) -> Dict[str, Any]:
    """Generate visualization data for the contact network.
    
    Args:
        start_date: Optional start date in ISO format (YYYY-MM-DD)
        end_date: Optional end date in ISO format (YYYY-MM-DD)
        min_shared_chats: Minimum number of shared chats to consider a connection
        layout: Layout algorithm to use (spring, circular, kamada_kawai)
    
    Returns:
        Dictionary containing network visualization data
    """
    try:
        logger.info(f"Generating network visualization using async implementation")
        
        # Initialize the DB if needed
        await db.initialize()
        
        # Import visualization module
        from visualization.async_network_viz import generate_network_visualization_async
        
        # Generate visualization data asynchronously
        result = await generate_network_visualization_async(
            start_date, end_date, min_shared_chats, layout
        )
        return result
    except Exception as e:
        logger.error(f"Error generating network visualization: {e}")
        logger.error(traceback.format_exc())
        return error_response("VISUALIZATION_ERROR", f"Error generating network visualization: {str(e)}")

@mcp.tool()
@run_async
async def analyze_sentiment(
    phone_number: str = None,
    chat_id: Union[str, int] = None,
    start_date: str = None,
    end_date: str = None,
    include_individual_messages: bool = False
) -> Dict[str, Any]:
    """Analyze sentiment in conversations with a contact or group chat.
    
    Args:
        phone_number: The phone number of the contact (optional)
        chat_id: The ID of the group chat (optional)
        start_date: Optional start date in ISO format (YYYY-MM-DD)
        end_date: Optional end date in ISO format (YYYY-MM-DD)
        include_individual_messages: Whether to include sentiment for each message
    
    Returns:
        Dictionary containing sentiment analysis results
    """
    try:
        logger.info(f"Analyzing sentiment using async implementation")
        
        if not phone_number and not chat_id:
            return error_response("PARAMETER_ERROR", "Either phone number or chat ID is required")
        
        # Initialize the DB if needed
        await db.initialize()
            
        # Import sentiment analysis module
        from analysis.async_sentiment_analysis import analyze_sentiment_async
        
        # Perform sentiment analysis asynchronously
        result = await analyze_sentiment_async(
            phone_number, chat_id, start_date, end_date, include_individual_messages
        )
        return result
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}")
        logger.error(traceback.format_exc())
        return error_response("ANALYSIS_ERROR", f"Error analyzing sentiment: {str(e)}")

@mcp.tool()
@run_async
async def get_chat_transcript(
    chat_id: Union[str, int] = None,
    phone_number: str = None,
    start_date: str = None,
    end_date: str = None
) -> Dict[str, Any]:
    """Get a chat transcript with a contact or in a group chat.
    
    Args:
        chat_id: The ID, name, or identifier of the chat (optional)
        phone_number: The phone number of the contact (optional)
        start_date: Optional start date in ISO format (YYYY-MM-DD)
        end_date: Optional end date in ISO format (YYYY-MM-DD)
    
    Returns:
        Dictionary containing the chat transcript data
    """
    try:
        logger.info(f"Getting chat transcript using async implementation")
        
        if not chat_id and not phone_number:
            return error_response("PARAMETER_ERROR", "Either chat ID or phone number is required")
        
        # Initialize the DB if needed
        await db.initialize()
            
        # Get transcript asynchronously
        messages = await db.get_chat_transcript(chat_id, phone_number, start_date, end_date)
        
        # Get chat name if it's a group chat
        chat_name = None
        if chat_id:
            try:
                async with db.get_db_connection() as connection:
                    query = "SELECT display_name FROM chat WHERE ROWID = ?"
                    cursor = await connection.execute(query, (chat_id,))
                    result = await cursor.fetchone()
                    if result and result[0]:
                        chat_name = result[0]
            except Exception as e:
                logger.warning(f"Couldn't get chat name: {e}")
        
        # If no messages found
        if not messages:
            return {
                "warning": "No messages found for the specified chat in the given date range",
                "chat_id": chat_id,
                "phone_number": phone_number,
                "chat_name": chat_name,
                "messages": []
            }
            
        return {
            "chat_id": chat_id,
            "phone_number": phone_number,
            "chat_name": chat_name,
            "messages": messages,
            "total_count": len(messages)
        }
    except Exception as e:
        logger.error(f"Error getting chat transcript: {e}")
        logger.error(traceback.format_exc())
        return error_response("DATABASE_ERROR", f"Error getting chat transcript: {str(e)}")

@mcp.tool()
@run_async
async def process_natural_language_query(
    query: str
) -> Dict[str, Any]:
    """Process a natural language query about message data.
    
    Args:
        query: The natural language query to process
    
    Returns:
        Dictionary containing the query results
    """
    try:
        logger.info(f"Processing natural language query: {query}")
        
        if not query:
            return error_response("PARAMETER_ERROR", "Query is required")
        
        # Initialize the DB if needed
        await db.initialize()
            
        # Import the query processor
        from utils.async_query_processor import process_natural_language_query_async
        
        # Process the query asynchronously
        result = await process_natural_language_query_async(query)
        return result
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        logger.error(traceback.format_exc())
        return error_response("QUERY_ERROR", f"Error processing query: {str(e)}")

# Run MCP server automatically if executed directly
if __name__ == "__main__":
    try:
        logger.info("Starting MCP server")
        mcp.run()
    except KeyboardInterrupt:
        logger.info("MCP server stopped by user")
    except Exception as e:
        logger.error(f"Error running MCP server: {e}")
        logger.error(traceback.format_exc())
    finally:
        logger.info("MCP server shutdown complete") 