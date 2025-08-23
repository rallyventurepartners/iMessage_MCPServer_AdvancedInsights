"""
Consent management for the MCP Server application.

This module handles user consent for accessing iMessage data, tracking
when consent was granted, and enforcing consent expiration.
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..exceptions import ConsentError

logger = logging.getLogger(__name__)


class ConsentManager:
    """
    Manages user consent for accessing iMessage data.
    
    This class implements the Singleton pattern to ensure only one
    instance exists throughout the application. It handles recording
    consent grants, checking consent validity, and logging access.
    """
    
    # Singleton instance
    _instance = None
    
    @classmethod
    def get_instance(cls) -> 'ConsentManager':
        """Get the singleton instance of the ConsentManager."""
        if cls._instance is None:
            cls._instance = ConsentManager()
        return cls._instance
    
    def __init__(self):
        """Initialize the ConsentManager."""
        # Prevent multiple instances
        if ConsentManager._instance is not None:
            raise RuntimeError("ConsentManager is a singleton. Use get_instance() instead.")
        
        # Set up the consent file path
        home_dir = os.path.expanduser("~")
        self.consent_dir = Path(home_dir) / ".imessage_insights" / "consent"
        self.consent_file = self.consent_dir / "consent.json"
        
        # Create the consent directory if it doesn't exist
        os.makedirs(self.consent_dir, exist_ok=True)
        
        # Load existing consent data
        self.consent_data = self._load_consent_data()
        
        # Save as singleton instance
        ConsentManager._instance = self
    
    def _load_consent_data(self) -> Dict[str, Any]:
        """
        Load consent data from file.
        
        Returns:
            Dict with consent data or empty dict if file doesn't exist
        """
        if not self.consent_file.exists():
            return {
                "has_consent": False,
                "granted_at": None,
                "expires_at": None,
                "access_log": []
            }
        
        try:
            with open(self.consent_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading consent data: {e}")
            return {
                "has_consent": False,
                "granted_at": None,
                "expires_at": None,
                "access_log": []
            }
    
    def _save_consent_data(self) -> bool:
        """
        Save consent data to file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(self.consent_file, "w") as f:
                json.dump(self.consent_data, f, indent=2)
            return True
        except IOError as e:
            logger.error(f"Error saving consent data: {e}")
            return False
    
    async def grant_consent(self, expires_hours: int = 24) -> bool:
        """
        Grant consent for accessing iMessage data.
        
        Args:
            expires_hours: Number of hours until consent expires
            
        Returns:
            True if consent was successfully granted
            
        Raises:
            ConsentError: If there's an error granting consent
        """
        try:
            now = datetime.now()
            expires = now + timedelta(hours=expires_hours)
            
            self.consent_data = {
                "has_consent": True,
                "granted_at": now.isoformat(),
                "expires_at": expires.isoformat(),
                "access_log": self.consent_data.get("access_log", [])
            }
            
            if not self._save_consent_data():
                raise ConsentError(
                    "Failed to save consent data. Check file permissions.",
                    {"consent_file": str(self.consent_file)}
                )
            
            logger.info(f"Consent granted. Expires at {expires.isoformat()}")
            return True
        except Exception as e:
            logger.error(f"Error granting consent: {e}")
            raise ConsentError(
                f"Failed to grant consent: {str(e)}",
                {"original_error": str(e)}
            )
    
    async def revoke_consent(self) -> bool:
        """
        Revoke consent for accessing iMessage data.
        
        Returns:
            True if consent was successfully revoked
            
        Raises:
            ConsentError: If there's an error revoking consent
        """
        try:
            self.consent_data["has_consent"] = False
            self.consent_data["expires_at"] = None
            
            if not self._save_consent_data():
                raise ConsentError(
                    "Failed to save consent data after revocation.",
                    {"consent_file": str(self.consent_file)}
                )
            
            logger.info("Consent revoked")
            return True
        except Exception as e:
            logger.error(f"Error revoking consent: {e}")
            raise ConsentError(
                f"Failed to revoke consent: {str(e)}",
                {"original_error": str(e)}
            )
    
    async def has_consent(self) -> bool:
        """
        Check if the user has granted consent.
        
        Returns:
            True if consent has been granted and is not expired
        """
        # Reload consent data from file to catch external changes
        self.consent_data = self._load_consent_data()
        
        # Check if consent has been granted
        if not self.consent_data.get("has_consent", False):
            return False
        
        # Check if consent has expired
        if await self.is_consent_expired():
            logger.info("Consent has expired")
            return False
        
        return True
    
    async def is_consent_expired(self) -> bool:
        """
        Check if consent has expired.
        
        Returns:
            True if consent has expired, False otherwise
        """
        expires_at = self.consent_data.get("expires_at")
        
        if not expires_at:
            return True
        
        try:
            expires_datetime = datetime.fromisoformat(expires_at)
            return datetime.now() > expires_datetime
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing consent expiration date: {e}")
            return True
    
    async def get_consent_expiration(self) -> Optional[datetime]:
        """
        Get the datetime when consent expires.
        
        Returns:
            Datetime when consent expires, or None if no consent
        """
        expires_at = self.consent_data.get("expires_at")
        
        if not expires_at:
            return None
        
        try:
            return datetime.fromisoformat(expires_at)
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing consent expiration date: {e}")
            return None
    
    async def log_access(self, tool_name: str) -> bool:
        """
        Log access to iMessage data for auditing.
        
        Args:
            tool_name: Name of the tool accessing the data
            
        Returns:
            True if access was successfully logged
        """
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "tool": tool_name
            }
            
            # Add the log entry
            access_log = self.consent_data.get("access_log", [])
            access_log.append(log_entry)
            
            # Limit log size to 1000 entries to prevent unbounded growth
            if len(access_log) > 1000:
                access_log = access_log[-1000:]
            
            self.consent_data["access_log"] = access_log
            
            # Save the updated log
            if not self._save_consent_data():
                logger.warning("Failed to save access log entry")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error logging access: {e}")
            return False
    
    async def get_access_log(
        self, limit: int = 100, offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Get the access log.
        
        Args:
            limit: Maximum number of log entries to return
            offset: Offset for pagination
            
        Returns:
            Tuple of (log entries, total count)
        """
        access_log = self.consent_data.get("access_log", [])
        total_count = len(access_log)
        
        # Sort by timestamp (newest first)
        sorted_log = sorted(
            access_log,
            key=lambda entry: entry.get("timestamp", ""),
            reverse=True
        )
        
        # Apply pagination
        paginated_log = sorted_log[offset:offset + limit]
        
        return paginated_log, total_count


# Convenience function for modules that don't want to use the singleton
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
