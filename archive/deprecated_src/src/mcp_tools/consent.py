"""
Consent management tools for the MCP Server application.

This module provides MCP tools for managing user consent to access
iMessage data, including requesting consent, checking consent status,
and revoking consent.
"""

import logging
from typing import Any, Dict, List, Optional

from ..exceptions import ConsentError
from ..utils.consent_manager import ConsentManager
from ..utils.responses import error_response, success_response
from .registry import register_tool

logger = logging.getLogger(__name__)


@register_tool(
    name="request_consent",
    description="Request user consent to access iMessage data",
    requires_user_consent=False,  # This tool is exempt from consent requirement
)
async def request_consent_tool(
    expiry_hours: int = 24
) -> Dict[str, Any]:
    """
    Request user consent to access iMessage data.
    
    This tool presents information about what data will be accessed and
    how it will be used, then records the user's consent if granted.
    
    Args:
        expiry_hours: Number of hours until consent expires
    
    Returns:
        Dictionary with consent status
    """
    try:
        consent_manager = ConsentManager.get_instance()
        
        # Verify valid expiry time
        if expiry_hours < 1 or expiry_hours > 720:  # Max 30 days
            return error_response(
                "Invalid expiry time. Please choose between 1 and 720 hours (30 days)."
            )
        
        # Check if consent is already granted
        if await consent_manager.has_consent():
            expiration = await consent_manager.get_consent_expiration()
            
            return success_response({
                "consent": True,
                "message": "Consent already granted",
                "expires_at": expiration.isoformat() if expiration else None,
            })
        
        # Request consent
        await consent_manager.grant_consent(expires_hours=expiry_hours)
        
        expiration = await consent_manager.get_consent_expiration()
        
        return success_response({
            "consent": True,
            "message": "Consent granted successfully",
            "expires_at": expiration.isoformat() if expiration else None,
        })
    except ConsentError as e:
        return error_response(e)
    except Exception as e:
        logger.error(f"Unexpected error in request_consent_tool: {e}")
        return error_response(f"Failed to process consent request: {str(e)}")


@register_tool(
    name="revoke_consent",
    description="Revoke user consent to access iMessage data",
    requires_user_consent=False,  # This tool is exempt from consent requirement
)
async def revoke_consent_tool() -> Dict[str, Any]:
    """
    Revoke user consent to access iMessage data.
    
    This tool allows the user to revoke previously granted consent
    for accessing their iMessage data.
    
    Returns:
        Dictionary with consent status
    """
    try:
        consent_manager = ConsentManager.get_instance()
        
        # Check if consent is already revoked
        if not await consent_manager.has_consent():
            return success_response({
                "consent": False,
                "message": "Consent already revoked or expired",
            })
        
        # Revoke consent
        await consent_manager.revoke_consent()
        
        return success_response({
            "consent": False,
            "message": "Consent revoked successfully",
        })
    except ConsentError as e:
        return error_response(e)
    except Exception as e:
        logger.error(f"Unexpected error in revoke_consent_tool: {e}")
        return error_response(f"Failed to revoke consent: {str(e)}")


@register_tool(
    name="check_consent",
    description="Check current consent status",
    requires_user_consent=False,  # This tool is exempt from consent requirement
)
async def check_consent_tool() -> Dict[str, Any]:
    """
    Check the current consent status.
    
    This tool allows checking whether consent has been granted
    and when it expires.
    
    Returns:
        Dictionary with consent status
    """
    try:
        consent_manager = ConsentManager.get_instance()
        
        # Check consent status
        has_consent = await consent_manager.has_consent()
        expiration = await consent_manager.get_consent_expiration()
        
        return success_response({
            "consent": has_consent,
            "expires_at": expiration.isoformat() if expiration else None,
            "message": "Consent is active" if has_consent else "No active consent",
        })
    except ConsentError as e:
        return error_response(e)
    except Exception as e:
        logger.error(f"Unexpected error in check_consent_tool: {e}")
        return error_response(f"Failed to check consent status: {str(e)}")


@register_tool(
    name="get_access_log",
    description="Get log of iMessage data access",
    requires_user_consent=True,  # This requires consent as it shows access history
)
async def get_access_log_tool(
    limit: int = 50,
    offset: int = 0
) -> Dict[str, Any]:
    """
    Get a log of iMessage data access.
    
    This tool provides an audit trail of when iMessage data was
    accessed, including which tools were used.
    
    Args:
        limit: Maximum number of log entries to return
        offset: Offset for pagination
    
    Returns:
        Dictionary with access log
    """
    try:
        consent_manager = ConsentManager.get_instance()
        
        # Validate parameters
        if limit < 1 or limit > 1000:
            return error_response("Limit must be between 1 and 1000")
        
        if offset < 0:
            return error_response("Offset must be non-negative")
        
        # Get access log
        log_entries, total_count = await consent_manager.get_access_log(
            limit=limit, offset=offset
        )
        
        # Calculate pagination info
        total_pages = (total_count + limit - 1) // limit if total_count > 0 else 1
        current_page = offset // limit + 1
        
        return success_response({
            "log": log_entries,
            "pagination": {
                "total": total_count,
                "limit": limit,
                "offset": offset,
                "current_page": current_page,
                "total_pages": total_pages,
                "has_more": current_page < total_pages,
            },
        })
    except ConsentError as e:
        return error_response(e)
    except Exception as e:
        logger.error(f"Unexpected error in get_access_log_tool: {e}")
        return error_response(f"Failed to get access log: {str(e)}")


# Expose these functions for use in other modules
async def has_consent() -> bool:
    """
    Check if the user has granted consent.
    
    Returns:
        True if consent has been granted and is not expired
    """
    consent_manager = ConsentManager.get_instance()
    return await consent_manager.has_consent()


async def is_consent_expired() -> bool:
    """
    Check if consent has expired.
    
    Returns:
        True if consent has expired, False otherwise
    """
    consent_manager = ConsentManager.get_instance()
    return await consent_manager.is_consent_expired()


async def log_access(tool_name: str) -> bool:
    """
    Log access to iMessage data for auditing.
    
    Args:
        tool_name: Name of the tool accessing the data
        
    Returns:
        True if access was successfully logged
    """
    consent_manager = ConsentManager.get_instance()
    return await consent_manager.log_access(tool_name)
