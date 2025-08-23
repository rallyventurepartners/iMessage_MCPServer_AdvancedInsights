"""
Consent management for the iMessage MCP Server.

This module handles user consent for accessing iMessage data, tracking
when consent was granted, and enforcing consent expiration.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ConsentManager:
    """Manages user consent for accessing iMessage data."""

    def __init__(self):
        """Initialize the ConsentManager."""
        # Set up the consent file path
        home_dir = os.path.expanduser("~")
        self.consent_dir = Path(home_dir) / ".imessage_insights" / "consent"
        self.consent_file = self.consent_dir / "consent.json"

        # Create the consent directory if it doesn't exist
        os.makedirs(self.consent_dir, exist_ok=True)

        # Load existing consent data
        self.consent_data = self._load_consent_data()

    def _load_consent_data(self) -> Dict[str, Any]:
        """Load consent data from file."""
        if not self.consent_file.exists():
            return {"has_consent": False, "granted_at": None, "expires_at": None, "access_log": []}

        try:
            with open(self.consent_file) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Error loading consent data: {e}")
            return {"has_consent": False, "granted_at": None, "expires_at": None, "access_log": []}

    def _save_consent_data(self) -> bool:
        """Save consent data to file."""
        try:
            with open(self.consent_file, "w") as f:
                json.dump(self.consent_data, f, indent=2, default=str)
            return True
        except OSError as e:
            logger.error(f"Error saving consent data: {e}")
            return False

    async def has_consent(self) -> bool:
        """Check if user has valid consent."""
        if not self.consent_data.get("has_consent", False):
            return False

        # Check if consent has expired
        expires_at = self.consent_data.get("expires_at")
        if expires_at:
            expiry_time = datetime.fromisoformat(expires_at)
            if datetime.now() > expiry_time:
                # Consent has expired
                self.consent_data["has_consent"] = False
                self._save_consent_data()
                return False

        return True

    async def grant_consent(self, expires_hours: int = 24) -> None:
        """Grant consent for the specified duration."""
        now = datetime.now()
        expires_at = now + timedelta(hours=expires_hours)

        self.consent_data.update(
            {
                "has_consent": True,
                "granted_at": now.isoformat(),
                "expires_at": expires_at.isoformat(),
                "grant_duration_hours": expires_hours,
            }
        )

        # Add to consent history
        if "consent_history" not in self.consent_data:
            self.consent_data["consent_history"] = []

        self.consent_data["consent_history"].append(
            {
                "granted_at": now.isoformat(),
                "expires_at": expires_at.isoformat(),
                "duration_hours": expires_hours,
            }
        )

        self._save_consent_data()
        logger.info(f"Consent granted until {expires_at}")

    async def revoke_consent(self) -> None:
        """Revoke consent immediately."""
        self.consent_data["has_consent"] = False
        self.consent_data["revoked_at"] = datetime.now().isoformat()
        self._save_consent_data()
        logger.info("Consent revoked")

    async def get_consent_expiration(self) -> Optional[datetime]:
        """Get when current consent expires."""
        if not await self.has_consent():
            return None

        expires_at = self.consent_data.get("expires_at")
        if expires_at:
            return datetime.fromisoformat(expires_at)

        return None

    async def log_access(self, tool_name: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Log tool access for audit trail."""
        if "access_log" not in self.consent_data:
            self.consent_data["access_log"] = []

        # Keep only last 1000 entries
        if len(self.consent_data["access_log"]) > 1000:
            self.consent_data["access_log"] = self.consent_data["access_log"][-900:]

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "tool": tool_name,
            "details": details or {},
        }

        self.consent_data["access_log"].append(log_entry)
        self._save_consent_data()

    async def get_access_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent access log entries."""
        access_log = self.consent_data.get("access_log", [])
        return access_log[-limit:]

    async def clear_expired_data(self) -> None:
        """Clear old consent history and access logs."""
        now = datetime.now()

        # Clear old consent history (keep last 30 days)
        if "consent_history" in self.consent_data:
            cutoff = now - timedelta(days=30)
            self.consent_data["consent_history"] = [
                entry
                for entry in self.consent_data["consent_history"]
                if datetime.fromisoformat(entry["granted_at"]) > cutoff
            ]

        # Clear old access logs (keep last 7 days)
        if "access_log" in self.consent_data:
            cutoff = now - timedelta(days=7)
            self.consent_data["access_log"] = [
                entry
                for entry in self.consent_data["access_log"]
                if datetime.fromisoformat(entry["timestamp"]) > cutoff
            ]

        self._save_consent_data()
