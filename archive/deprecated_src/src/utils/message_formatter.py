#!/usr/bin/env python3
"""
Message Formatter Module

This module provides utilities for formatting and sanitizing message data
for display and analysis. It handles date formatting, text cleaning, and
privacy-sensitive data management.
"""

import logging
import re
import html
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

# Configure logging
logger = logging.getLogger(__name__)

def clean_message_text(text: Optional[str]) -> str:
    """
    Clean and normalize message text for display or analysis.
    
    Args:
        text: Raw message text which may contain control characters or encoding issues
        
    Returns:
        Cleaned text ready for display or analysis
    """
    if text is None:
        return ""
        
    try:
        # Convert to string if not already
        text = str(text)
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Replace control characters
        text = re.sub(r'[\x01-\x1F\x7F]', '', text)
        
        # Handle HTML entities
        text = html.unescape(text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    except Exception as e:
        logger.error(f"Error cleaning message text: {e}")
        return "" if text is None else str(text)

def format_date(timestamp: Union[int, float, str, datetime], format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format a timestamp into a human-readable date string.
    
    Args:
        timestamp: Timestamp to format (Apple timestamp, Unix timestamp, or datetime object)
        format_str: strftime format string
        
    Returns:
        Formatted date string
    """
    try:
        if isinstance(timestamp, datetime):
            # Already a datetime object
            dt = timestamp
        elif isinstance(timestamp, (int, float)):
            # Check if this is an Apple timestamp (nanoseconds since 2001-01-01)
            if timestamp > 1000000000000:  # Likely an Apple timestamp
                # Convert Apple timestamp to Unix timestamp
                seconds_since_2001 = timestamp / 1e9
                epoch_2001 = 978307200  # 2001-01-01 in Unix epoch seconds
                unix_timestamp = epoch_2001 + seconds_since_2001
                dt = datetime.fromtimestamp(unix_timestamp)
            else:
                # Standard Unix timestamp (seconds since 1970-01-01)
                dt = datetime.fromtimestamp(timestamp)
        elif isinstance(timestamp, str):
            # Try to parse as ISO format
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        else:
            logger.error(f"Unsupported timestamp type: {type(timestamp)}")
            return str(timestamp)
            
        return dt.strftime(format_str)
    except Exception as e:
        logger.error(f"Error formatting date: {e}")
        return str(timestamp)

def format_messages_for_display(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Format a list of messages for display, ensuring all fields are properly formatted.
    
    Args:
        messages: List of message objects from the database
        
    Returns:
        Formatted messages ready for display
    """
    formatted_messages = []
    
    for message in messages:
        if not isinstance(message, dict):
            continue
            
        formatted_message = message.copy()
        
        # Clean message text
        if "text" in formatted_message:
            formatted_message["text"] = clean_message_text(formatted_message["text"])
            
        # Format date
        if "date" in formatted_message:
            try:
                # Convert date to ISO format string
                if isinstance(formatted_message["date"], (int, float)):
                    formatted_message["date_display"] = format_date(formatted_message["date"])
                    # Keep original timestamp but also add ISO format
                    formatted_message["date_iso"] = format_date(formatted_message["date"], "%Y-%m-%dT%H:%M:%S")
            except Exception as e:
                logger.error(f"Error formatting message date: {e}")
                formatted_message["date_display"] = str(formatted_message["date"])
        
        # Ensure "is_from_me" is a boolean
        if "is_from_me" in formatted_message:
            formatted_message["is_from_me"] = bool(formatted_message["is_from_me"])
            
        # Add a sender_type field for easy filtering
        formatted_message["sender_type"] = "me" if formatted_message.get("is_from_me") else "them"
        
        formatted_messages.append(formatted_message)
        
    return formatted_messages

def sanitize_group_chat_data(chat_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize group chat data to ensure privacy and proper formatting.
    
    Args:
        chat_data: Raw chat data from the database
        
    Returns:
        Sanitized chat data ready for display
    """
    if not isinstance(chat_data, dict):
        return {}
        
    # Create a copy to avoid modifying the original
    sanitized_data = chat_data.copy()
    
    # Ensure display name is set
    if not sanitized_data.get("display_name"):
        sanitized_data["display_name"] = f"Group Chat {sanitized_data.get('chat_id', '')}"
        
    # Remove any sensitive fields
    sensitive_fields = ["chat_identifier", "account_login"]
    for field in sensitive_fields:
        if field in sanitized_data:
            del sanitized_data[field]
            
    # Format dates if present
    date_fields = ["created_date", "last_message_date"]
    for field in date_fields:
        if field in sanitized_data and sanitized_data[field]:
            try:
                sanitized_data[f"{field}_display"] = format_date(sanitized_data[field])
            except Exception as e:
                logger.error(f"Error formatting chat date: {e}")
    
    return sanitized_data
