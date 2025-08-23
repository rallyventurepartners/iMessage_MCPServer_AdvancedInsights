"""
Decorators for the MCP Server application.

This module provides various decorators used throughout the application for:
- Error handling
- Authentication and authorization
- Performance monitoring
- Asynchronous execution
- Consent validation
"""

import asyncio
import functools
import logging
import time
import traceback
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

from ..exceptions import ConsentError, DatabaseError, MCPServerError, ToolExecutionError

logger = logging.getLogger(__name__)
F = TypeVar('F', bound=Callable)


def handle_tool_errors(func: F) -> F:
    """Decorator to standardize error handling in MCP tools.
    
    This decorator catches all exceptions that occur in MCP tool functions,
    logs them appropriately, and returns a standardized error response.
    
    Args:
        func: The async function to decorate
        
    Returns:
        Decorated function with standardized error handling
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> Dict[str, Any]:
        try:
            result = await func(*args, **kwargs)
            # If result is already a dict with success flag, return it
            if isinstance(result, dict) and "success" in result:
                return result
                
            # Otherwise, wrap the result in a success response
            return {"success": True, "result": result}
        except MCPServerError as e:
            # Log custom exception with its specialized type
            logger.error(f"{e.__class__.__name__} in {func.__name__}: {e}")
            logger.debug(traceback.format_exc())
            
            return {
                "success": False,
                "error": e.to_dict()
            }
        except Exception as e:
            # Log unexpected exceptions with full stack trace
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            
            # Convert to a ToolExecutionError for consistent error handling
            tool_error = ToolExecutionError(
                f"Unexpected error during tool execution: {str(e)}",
                {
                    "original_type": e.__class__.__name__,
                    "function": func.__name__,
                    "original_message": str(e)
                }
            )
            
            return {
                "success": False,
                "error": tool_error.to_dict()
            }
    return wrapper  # type: ignore


def requires_consent(func: F) -> F:
    """Decorator to verify user consent before accessing user data.
    
    This decorator checks if the user has granted consent for data access
    before executing the decorated function. If consent has not been granted,
    it raises a ConsentError.
    
    Args:
        func: The async function to decorate
        
    Returns:
        Decorated function with consent verification
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        # Import here to avoid circular imports
        from ..mcp_tools.consent import has_consent, is_consent_expired, log_access
        
        # Check if consent has been granted
        if not await has_consent():
            raise ConsentError(
                "User consent is required to access iMessage data. "
                "Please use the request_consent tool first.",
                {"tool": func.__name__}
            )
        
        # Check if consent is still valid (not expired)
        if await is_consent_expired():
            raise ConsentError(
                "User consent has expired. Please request fresh consent.",
                {"tool": func.__name__}
            )
        
        # Log the access
        await log_access(func.__name__)
        
        # Proceed with the function execution
        return await func(*args, **kwargs)
    return wrapper  # type: ignore


def run_async(func: F) -> F:
    """Decorator to ensure a function runs in the event loop.
    
    This decorator ensures that the decorated function is always executed
    within an asyncio event loop, even if called from synchronous code.
    
    Args:
        func: The async function to decorate
        
    Returns:
        Decorated function that is safe to call from both async and sync contexts
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        # If we're already in an event loop, just call the function
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                return asyncio.create_task(func(*args, **kwargs))
        except RuntimeError:
            # No event loop running, create one
            pass
        
        # No event loop running, create one and run the function
        return asyncio.run(func(*args, **kwargs))
    return wrapper  # type: ignore


def retry_async(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: List[Type[Exception]] = [DatabaseError]
) -> Callable[[F], F]:
    """Decorator for retrying async functions with exponential backoff.
    
    This decorator retries the decorated function if it raises one of the
    specified exceptions, with an exponential backoff between retries.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_factor: Factor by which the delay increases after each retry
        exceptions: List of exception types that trigger a retry
        
    Returns:
        Decorator function that adds retry logic to the decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            retry_count = 0
            current_delay = delay
            
            while True:
                try:
                    return await func(*args, **kwargs)
                except tuple(exceptions) as e:
                    retry_count += 1
                    
                    if retry_count > max_retries:
                        logger.warning(
                            f"Maximum retries ({max_retries}) exceeded for {func.__name__}"
                        )
                        # Re-raise the last exception
                        raise
                    
                    logger.info(
                        f"Retry {retry_count}/{max_retries} for {func.__name__} "
                        f"after error: {e}"
                    )
                    
                    # Wait before retrying
                    await asyncio.sleep(current_delay)
                    
                    # Increase delay for next retry
                    current_delay *= backoff_factor
        
        return wrapper  # type: ignore
    return decorator


def performance_monitor(func: F) -> F:
    """Decorator to monitor and log function performance.
    
    This decorator measures the execution time of the decorated function
    and logs it at the DEBUG level. It also logs warnings for slow operations.
    
    Args:
        func: The async function to decorate
        
    Returns:
        Decorated function with performance monitoring
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        
        # Execute the function
        result = await func(*args, **kwargs)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Log execution time
        logger.debug(f"{func.__name__} executed in {execution_time:.4f} seconds")
        
        # Log warning for slow operations (over 1 second)
        if execution_time > 1.0:
            logger.warning(
                f"Slow operation detected: {func.__name__} took {execution_time:.4f} seconds"
            )
        
        return result
    return wrapper  # type: ignore


def parse_date(date_string: Optional[str]) -> Optional[datetime]:
    """Parse a date string into a datetime object.
    
    Supports both absolute dates (YYYY-MM-DD) and relative dates
    like "1 day ago", "2 weeks ago", etc.
    
    Args:
        date_string: The date string to parse
        
    Returns:
        A datetime object, or None if date_string is None
    """
    if not date_string:
        return None
        
    try:
        from datetime import datetime, timedelta
        
        # Try to parse as an absolute date (YYYY-MM-DD)
        try:
            return datetime.strptime(date_string, "%Y-%m-%d")
        except ValueError:
            pass
        
        # Try to parse as a relative date
        parts = date_string.lower().split()
        
        # Handle "X days ago" format
        if "ago" in date_string.lower():
            if len(parts) >= 3 and parts[2] == "ago" and parts[0].isdigit():
                amount = int(parts[0])
                unit = parts[1]
                
                # Calculate the date
                now = datetime.now()
                
                if "day" in unit:
                    return now - timedelta(days=amount)
                elif "week" in unit:
                    return now - timedelta(weeks=amount)
                elif "month" in unit:
                    # Approximate months as 30 days
                    return now - timedelta(days=amount * 30)
                elif "year" in unit:
                    # Approximate years as 365 days
                    return now - timedelta(days=amount * 365)
        
        # Handle "X days" format (without "ago") - interpret as X days ago
        elif len(parts) == 2 and parts[0].isdigit():
            amount = int(parts[0])
            unit = parts[1]
            
            # Calculate the date
            now = datetime.now()
            
            if "day" in unit:
                return now - timedelta(days=amount)
            elif "week" in unit:
                return now - timedelta(weeks=amount)
            elif "month" in unit:
                # Approximate months as 30 days
                return now - timedelta(days=amount * 30)
            elif "year" in unit:
                # Approximate years as 365 days
                return now - timedelta(days=amount * 365)
        
        # Handle "yesterday", "today", etc.
        if date_string.lower() == "today":
            return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        elif date_string.lower() == "yesterday":
            return (datetime.now() - timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        
        # If we can't parse it, log a warning and return None
        logger.warning(f"Could not parse date string: {date_string}")
        return None
    except Exception as e:
        logger.error(f"Error parsing date string '{date_string}': {e}")
        return None
