"""
Base functionality for MCP tools.

This module provides common utilities and base classes for tool implementations.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from ..consent import ConsentManager

logger = logging.getLogger(__name__)


async def check_tool_consent(tool_name: str, consent_manager: Optional["ConsentManager"]) -> bool:
    """
    Check if user has active consent for a tool.

    Args:
        tool_name: Name of the tool requesting access
        consent_manager: Consent manager instance

    Returns:
        bool: True if consent is granted, False otherwise
    """
    if consent_manager is None:
        return False

    has_consent = await consent_manager.has_consent()
    if has_consent:
        await consent_manager.log_access(tool_name)
    return has_consent


def create_error_response(error: Exception, error_type: str = "unknown_error") -> Dict[str, Any]:
    """
    Create a standardized error response.

    Args:
        error: The exception that occurred
        error_type: Type of error for categorization

    Returns:
        Dict containing error information
    """
    return {"error": str(error), "error_type": error_type}


def create_consent_error() -> Dict[str, Any]:
    """Create a standard consent required error response."""
    return {"error": "No active consent", "error_type": "consent_required"}
