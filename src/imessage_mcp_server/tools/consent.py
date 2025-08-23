"""
Consent management tools for iMessage Advanced Insights.

These tools handle user consent for accessing iMessage data,
ensuring privacy-first operation.
"""

from datetime import datetime
from typing import Any, Dict

from imessage_mcp_server.consent import ConsentManager


async def check_tool_consent(consent_manager: ConsentManager, tool_name: str) -> bool:
    """
    Check if user has active consent for a tool.

    Args:
        consent_manager: The consent manager instance
        tool_name: Name of the tool to check consent for

    Returns:
        bool: True if user has active consent, False otherwise
    """
    if consent_manager is None:
        return False

    has_consent = await consent_manager.has_consent()
    if has_consent:
        await consent_manager.log_access(tool_name)
    return has_consent


async def request_consent_tool(
    consent_manager: ConsentManager, expiry_hours: int = 24
) -> Dict[str, Any]:
    """
    Request user consent to access iMessage data.

    This tool must be called before any data access tools can be used.
    Consent expires after the specified duration.

    Args:
        consent_manager: The consent manager instance
        expiry_hours: Hours until consent expires (1-720)

    Returns:
        Dict containing consent status and expiration time

    Raises:
        ValueError: If expiry_hours is out of valid range
    """
    try:
        if expiry_hours < 1 or expiry_hours > 720:
            return {
                "error": "Invalid expiry time. Choose 1-720 hours.",
                "error_type": "validation_error",
            }

        await consent_manager.grant_consent(expiry_hours)
        expiration = await consent_manager.get_consent_expiration()

        return {
            "consent": True,
            "message": f"Consent granted for {expiry_hours} hours",
            "expires_at": expiration.isoformat() if expiration else None,
            "important": "Consent is required for all data access. It will expire automatically.",
        }

    except Exception as e:
        return {"error": f"Failed to grant consent: {str(e)}", "error_type": "consent_error"}


async def check_consent_tool(consent_manager: ConsentManager) -> Dict[str, Any]:
    """
    Check current consent status.

    Args:
        consent_manager: The consent manager instance

    Returns:
        Dict containing current consent status and expiration
    """
    try:
        has_consent = await consent_manager.has_consent()
        expiration = await consent_manager.get_consent_expiration()

        if has_consent and expiration:
            remaining = expiration - datetime.now()
            hours_left = int(remaining.total_seconds() / 3600)

            return {
                "consent": True,
                "expires_at": expiration.isoformat(),
                "hours_remaining": hours_left,
                "message": f"Consent active for {hours_left} more hours",
            }
        else:
            return {
                "consent": False,
                "message": "No active consent. Use 'request_consent' to grant access.",
            }

    except Exception as e:
        return {"error": f"Failed to check consent: {str(e)}", "error_type": "consent_error"}


async def revoke_consent_tool(consent_manager: ConsentManager) -> Dict[str, Any]:
    """
    Revoke user consent to access iMessage data.

    This immediately revokes all access to iMessage data.

    Args:
        consent_manager: The consent manager instance

    Returns:
        Dict confirming consent revocation
    """
    try:
        if not await consent_manager.has_consent():
            return {"consent": False, "message": "Consent already revoked or expired"}

        await consent_manager.revoke_consent()

        return {
            "consent": False,
            "message": "Consent revoked successfully. All data access disabled.",
            "note": "Use 'request_consent' to re-enable access.",
        }

    except Exception as e:
        return {"error": f"Failed to revoke consent: {str(e)}", "error_type": "consent_error"}
