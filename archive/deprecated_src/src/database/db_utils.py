#!/usr/bin/env python3
"""
Database Utility Functions

This module provides common utility functions used by the database
implementations, helping to reduce code duplication.
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union

# Configure logging
logger = logging.getLogger(__name__)


def convert_apple_timestamp(timestamp: Optional[int]) -> Optional[datetime]:
    """
    Convert Apple's timestamp to a Python datetime.
    
    Apple uses nanoseconds since 2001-01-01 as their timestamp format.
    
    Args:
        timestamp: Apple timestamp (nanoseconds since 2001-01-01)
        
    Returns:
        datetime: Python datetime object or None if conversion fails
    """
    if not timestamp:
        return None
        
    try:
        # Apple timestamp is nanoseconds since 2001-01-01
        seconds_since_2001 = timestamp / 1e9
        epoch_2001 = 978307200  # 2001-01-01 in Unix epoch seconds
        unix_timestamp = epoch_2001 + seconds_since_2001
        return datetime.fromtimestamp(unix_timestamp)
    except Exception as e:
        logger.error(f"Error converting Apple timestamp {timestamp}: {e}")
        return None


def datetime_to_apple_timestamp(dt: Optional[datetime]) -> Optional[int]:
    """
    Convert a Python datetime to Apple's timestamp format.
    
    Args:
        dt: Python datetime object
        
    Returns:
        int: Apple timestamp (nanoseconds since 2001-01-01) or None if conversion fails
    """
    if not dt:
        return None
        
    try:
        # Calculate seconds since 2001-01-01
        epoch_2001 = 978307200  # 2001-01-01 in Unix epoch seconds
        unix_timestamp = dt.timestamp()
        seconds_since_2001 = unix_timestamp - epoch_2001
        
        # Convert to nanoseconds
        return int(seconds_since_2001 * 1e9)
    except Exception as e:
        logger.error(f"Error converting datetime {dt} to Apple timestamp: {e}")
        return None


def parse_date_string(date_str: Optional[str]) -> Optional[datetime]:
    """
    Parse a date string into a datetime object.
    
    Supports:
    - ISO format dates (e.g., "2023-04-01")
    - ISO format datetimes (e.g., "2023-04-01T12:34:56")
    - Relative dates (e.g., "1 week ago", "3 days ago", "2 months ago")
    - Special values (e.g., "today", "yesterday")
    
    Args:
        date_str: Date string to parse
        
    Returns:
        datetime: Python datetime object or None if parsing failed
    """
    if not date_str:
        return None
    
    # Handle special values
    date_str = date_str.strip().lower()
    
    if date_str == "today":
        return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    if date_str == "yesterday":
        return (datetime.now() - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        
    # Try to parse relative dates
    ago_match = re.match(r"^(\d+)\s+(day|days|week|weeks|month|months|year|years)\s+ago$", date_str)
    if ago_match:
        amount = int(ago_match.group(1))
        unit = ago_match.group(2)
        now = datetime.now()
        
        if unit in ("day", "days"):
            return now - timedelta(days=amount)
        elif unit in ("week", "weeks"):
            return now - timedelta(days=amount * 7)
        elif unit in ("month", "months"):
            # Approximate month as 30 days
            return now - timedelta(days=amount * 30)
        elif unit in ("year", "years"):
            # Approximate year as 365 days
            return now - timedelta(days=amount * 365)
    
    # Try ISO format
    try:
        # Handle "Z" for UTC time
        cleaned_date_str = date_str.replace('Z', '+00:00')
        return datetime.fromisoformat(cleaned_date_str)
    except ValueError:
        pass
    
    # Try other common date formats
    formats = [
        "%Y-%m-%d",       # 2023-04-01
        "%m/%d/%Y",       # 04/01/2023
        "%d/%m/%Y",       # 01/04/2023
        "%b %d, %Y",      # Apr 01, 2023
        "%B %d, %Y",      # April 01, 2023
        "%Y-%m-%d %H:%M:%S",  # 2023-04-01 12:34:56
        "%m/%d/%Y %H:%M:%S",  # 04/01/2023 12:34:56
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    # Could not parse the date
    logger.warning(f"Could not parse date string: {date_str}")
    return None


def format_date_for_display(dt: Optional[datetime]) -> str:
    """
    Format a datetime object for display.
    
    Args:
        dt: Datetime object to format
        
    Returns:
        str: Formatted date string or empty string if dt is None
    """
    if not dt:
        return ""
    
    # Today
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    # Yesterday
    yesterday = today - timedelta(days=1)
    
    # Format date based on how recent it is
    if dt.date() == datetime.now().date():
        # Today - show time only
        return dt.strftime("Today at %I:%M %p")
    elif dt.date() == yesterday.date():
        # Yesterday - show as "Yesterday at HH:MM"
        return dt.strftime("Yesterday at %I:%M %p")
    elif (datetime.now() - dt).days < 7:
        # Within a week - show as "Monday at HH:MM"
        return dt.strftime("%A at %I:%M %p")
    elif dt.year == datetime.now().year:
        # This year - omit year
        return dt.strftime("%b %d at %I:%M %p")
    else:
        # Different year - include year
        return dt.strftime("%b %d, %Y at %I:%M %p")


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to a specified maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length for the text
        
    Returns:
        str: Truncated text with ellipsis if necessary
    """
    if not text:
        return ""
    
    if len(text) <= max_length:
        return text
    
    return text[:max_length - 3] + "..."


def sanitize_query_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize SQL query parameters to prevent SQL injection.
    
    Args:
        params: Dictionary of query parameters
        
    Returns:
        Dict[str, Any]: Sanitized parameters
    """
    sanitized = {}
    
    for key, value in params.items():
        if isinstance(value, str):
            # Remove any SQL injection attempts
            sanitized[key] = re.sub(r'[\'";\-\\]', '', value)
        else:
            sanitized[key] = value
    
    return sanitized


def format_messages_for_display(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Format a list of messages for display.
    
    Cleans up fields and formats dates for display.
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        List[Dict[str, Any]]: Formatted messages
    """
    formatted_messages = []
    
    for message in messages:
        # Create a copy to avoid modifying the original
        formatted = message.copy()
        
        # Format date
        if "date" in formatted:
            date_obj = None
            if "date_obj" in formatted:
                date_obj = formatted["date_obj"]
            else:
                # Try to parse from string
                try:
                    date_obj = parse_date_string(formatted["date"])
                except:
                    pass
            
            if date_obj:
                formatted["formatted_date"] = format_date_for_display(date_obj)
        
        # Clean text
        if "text" in formatted and formatted["text"]:
            formatted["text"] = clean_message_text(formatted["text"])
        
        formatted_messages.append(formatted)
    
    return formatted_messages


def clean_message_text(text: Optional[str]) -> str:
    """
    Clean up message text by removing control characters and normalizing whitespace.
    
    Args:
        text: Text to clean
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Convert None to empty string
    if text is None:
        return ""
    
    # Remove control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # Replace multiple whitespace with a single space, preserve newlines
    text = re.sub(r' +', ' ', text)
    
    # Replace multiple newlines with at most two
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def build_pagination_info(
    total_count: int, 
    page: int, 
    page_size: int
) -> Dict[str, Any]:
    """
    Build pagination information for query results.
    
    Args:
        total_count: Total number of items
        page: Current page number
        page_size: Number of items per page
        
    Returns:
        Dict[str, Any]: Pagination information
    """
    # Ensure sensible values
    page = max(1, page)
    page_size = max(1, page_size)
    
    # Calculate pagination info
    total_pages = (total_count + page_size - 1) // page_size if page_size > 0 else 1
    has_next = page < total_pages
    has_prev = page > 1
    
    return {
        "page": page,
        "page_size": page_size,
        "total_count": total_count,
        "total_pages": total_pages,
        "has_next": has_next,
        "has_prev": has_prev,
        "next_page": page + 1 if has_next else None,
        "prev_page": page - 1 if has_prev else None,
    }


def build_date_filters_sql(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Build SQL WHERE clause for date filtering.
    
    Args:
        start_date: Optional start date for filtering
        end_date: Optional end date for filtering
        
    Returns:
        Tuple[str, Dict[str, Any]]: SQL fragment and parameters
    """
    where_clauses = []
    params = {}
    
    if start_date:
        # Convert to Apple timestamp
        start_timestamp = datetime_to_apple_timestamp(start_date)
        if start_timestamp:
            where_clauses.append("message.date >= :start_date")
            params["start_date"] = start_timestamp
    
    if end_date:
        # Convert to Apple timestamp
        end_timestamp = datetime_to_apple_timestamp(end_date)
        if end_timestamp:
            where_clauses.append("message.date <= :end_date")
            params["end_date"] = end_timestamp
    
    # Combine clauses
    if where_clauses:
        return " AND ".join(where_clauses), params
    else:
        return "", {}
