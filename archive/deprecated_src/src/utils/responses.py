"""
Response utilities for the MCP Server application.

This module provides standardized response formatting for API responses,
ensuring a consistent structure across all endpoints.
"""

import logging
import traceback
from typing import Any, Dict, List, Optional, Union

from ..exceptions import MCPServerError

logger = logging.getLogger(__name__)


def success_response(data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create a standardized success response.
    
    Args:
        data: Optional data to include in the response
        
    Returns:
        Dictionary with success flag and optional data
    """
    response = {"success": True}
    
    if data is not None:
        # Put data in a 'data' key for consistency
        response["data"] = data
    
    return response


def error_response(
    error: Union[str, Exception],
    details: Optional[Dict[str, Any]] = None,
    include_traceback: bool = False,
) -> Dict[str, Any]:
    """
    Create a standardized error response.
    
    Args:
        error: Error message or exception
        details: Additional details about the error
        include_traceback: Whether to include a stack trace (for debug only)
        
    Returns:
        Dictionary with error information
    """
    if isinstance(error, MCPServerError):
        # Use the built-in to_dict method for custom exceptions
        error_dict = error.to_dict()
        
        # Add any additional details
        if details:
            error_dict["details"].update(details)
    elif isinstance(error, Exception):
        # Convert standard exceptions to a consistent format
        error_dict = {
            "type": error.__class__.__name__,
            "message": str(error),
            "details": details or {}
        }
    else:
        # Handle string error messages
        error_dict = {
            "type": "Error",
            "message": str(error),
            "details": details or {}
        }
    
    # Add traceback for debugging if requested
    if include_traceback and isinstance(error, Exception):
        error_dict["traceback"] = traceback.format_exc()
    
    return {
        "success": False,
        "error": error_dict
    }


def paginated_response(
    items: List[Any],
    page: int = 1,
    page_size: int = 20,
    total_items: Optional[int] = None,
    additional_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a standardized paginated response.
    
    Args:
        items: List of items for the current page
        page: Current page number (1-based)
        page_size: Number of items per page
        total_items: Total number of items across all pages
        additional_data: Additional data to include in the response
        
    Returns:
        Dictionary with paginated data
    """
    # Ensure page is at least 1
    page = max(1, page)
    page_size = max(1, page_size)
    
    # Calculate total items if not provided
    if total_items is None:
        total_items = len(items)
    
    # Calculate total pages
    total_pages = (total_items + page_size - 1) // page_size if total_items > 0 else 1
    
    # Determine if there are more pages
    has_more = page < total_pages
    
    # Create pagination information
    pagination = {
        "page": page,
        "page_size": page_size,
        "total_items": total_items,
        "total_pages": total_pages,
        "has_more": has_more
    }
    
    # Create the response with data in a consistent structure
    data = {
        "items": items,
        "pagination": pagination
    }
    
    # Add any additional data
    if additional_data:
        for key, value in additional_data.items():
            if key not in data:
                data[key] = value
    
    return {
        "success": True,
        "data": data
    }


def stream_response(
    generator,
    total_items: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a response for streaming data.
    
    This is used for endpoints that return a generator or streaming response,
    providing metadata about the stream upfront.
    
    Args:
        generator: Async generator that produces data
        total_items: Total number of items in the stream (if known)
        metadata: Additional metadata about the stream
        
    Returns:
        Dictionary with stream information
    """
    response = {
        "success": True,
        "stream": True,
        "metadata": metadata or {}
    }
    
    if total_items is not None:
        response["metadata"]["total_items"] = total_items
    
    # The actual generator will be handled by the connection
    # This is just metadata about the stream
    response["generator"] = generator
    
    return response
