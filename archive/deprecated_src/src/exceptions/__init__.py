"""
Custom exception hierarchy for the MCP Server application.

This module defines a hierarchy of exceptions specific to the MCP Server,
allowing for more precise error handling and better error reporting to clients.
Each exception type corresponds to a specific category of error that can occur
in the application.
"""

from typing import Any, Dict, Optional


class MCPServerError(Exception):
    """Base exception for all MCP server errors.
    
    All custom exceptions in the application should inherit from this class.
    This allows catching all application-specific errors with a single except clause.
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize the exception with a message and optional details.
        
        Args:
            message: Human-readable error message
            details: Additional context about the error (optional)
        """
        super().__init__(message)
        self.details = details or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the exception to a dictionary for API responses.
        
        Returns:
            Dict containing error type, message, and details
        """
        return {
            "type": self.__class__.__name__,
            "message": str(self),
            "details": self.details
        }


class DatabaseError(MCPServerError):
    """Raised when database operations fail.
    
    This exception is used for all database-related errors, including:
    - Connection failures
    - Query execution errors
    - Schema validation errors
    - Data integrity issues
    """
    pass


class ConfigurationError(MCPServerError):
    """Raised when there are issues with application configuration.
    
    This exception is used for:
    - Missing required configuration
    - Invalid configuration values
    - Configuration conflicts
    """
    pass


class ConsentError(MCPServerError):
    """Raised when operations are attempted without proper user consent.
    
    This exception is used when:
    - Attempting to access user data without explicit consent
    - Consent has expired or been revoked
    - Requested operation exceeds the granted consent scope
    """
    pass


class ResourceError(MCPServerError):
    """Raised when MCP resource operations fail.
    
    This exception is used for errors related to MCP resources:
    - Resource not found
    - Invalid resource URI
    - Resource access denied
    """
    pass


class ToolExecutionError(MCPServerError):
    """Raised when MCP tool execution fails.
    
    This exception is used for errors during tool execution:
    - Invalid tool parameters
    - Tool execution timeout
    - Internal tool processing errors
    """
    pass


class SecurityError(MCPServerError):
    """Raised when security checks fail.
    
    This exception is used for security-related issues:
    - Permission checks fail
    - Secure storage is unavailable
    - File permission issues
    """
    pass


class DataError(MCPServerError):
    """Raised when there are issues with data processing.
    
    This exception is used for data-related errors:
    - Data validation failures
    - Data format errors
    - Data conversion errors
    """
    pass


class MemoryLimitError(MCPServerError):
    """Raised when operations exceed memory limits.
    
    This exception is used when:
    - Memory usage exceeds configured thresholds
    - Large query results would consume too much memory
    - Batch processing needs to be applied due to memory constraints
    """
    pass
