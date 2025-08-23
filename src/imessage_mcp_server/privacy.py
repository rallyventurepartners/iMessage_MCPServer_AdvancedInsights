"""
Privacy utilities for the iMessage MCP Server.

This module implements contact ID hashing, PII redaction, and other
privacy-preserving functions.
"""

import hashlib
import re
from typing import Any, Dict, List, Optional

from .config import get_config

# PII regex patterns
PATTERNS = {
    "credit_card": re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phone": re.compile(r"\b(?:\+\d{1,2}\s?)?(?:\(\d{3}\)\s?|\d{3}[-.\s]?)\d{3}[-.\s]?\d{4}\b"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "address": re.compile(
        r"\b\d+\s+[A-Za-z0-9\s,]+(?:Avenue|Lane|Road|Boulevard|Drive|Street|Ave|Dr|Rd|Blvd|Ln|St)\.?\b"
    ),
    "amount": re.compile(r"\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?"),
}


def hash_contact_id(contact_id: str, salt: Optional[bytes] = None) -> str:
    """
    Hash a contact ID using BLAKE2b with per-session salt.

    Args:
        contact_id: The contact identifier to hash
        salt: Optional salt (uses session salt if not provided)

    Returns:
        Hashed contact ID in format "hash:xxxxxxxx"
    """
    if not contact_id:
        return contact_id

    # Skip if already hashed
    if contact_id.startswith("hash:"):
        return contact_id

    # Get salt from config if not provided
    if salt is None:
        config = get_config()
        salt = config.session_salt

    # Use BLAKE2b for hashing (16 bytes = 32 hex chars)
    h = hashlib.blake2b(contact_id.encode("utf-8"), salt=salt, digest_size=16)

    return f"hash:{h.hexdigest()[:8]}"  # Use first 8 chars for brevity


def unhash_contact_id(hashed_id: str, contact_map: Dict[str, str]) -> Optional[str]:
    """
    Attempt to reverse a hashed contact ID using a pre-built mapping.

    Args:
        hashed_id: The hashed contact ID
        contact_map: Mapping of hashed IDs to original IDs

    Returns:
        Original contact ID if found, None otherwise
    """
    if not hashed_id.startswith("hash:"):
        return hashed_id

    return contact_map.get(hashed_id)


def redact_pii(text: Optional[str]) -> Optional[str]:
    """
    Redact personally identifiable information from text.

    Args:
        text: Text to redact

    Returns:
        Redacted text
    """
    if not text:
        return text

    result = text

    # Apply redaction patterns
    result = PATTERNS["credit_card"].sub("[CREDIT CARD REDACTED]", result)
    result = PATTERNS["ssn"].sub("[SSN REDACTED]", result)
    result = PATTERNS["amount"].sub("[AMOUNT REDACTED]", result)
    result = PATTERNS["address"].sub("[ADDRESS REDACTED]", result)

    # Partial redaction for phones
    def redact_phone(match):
        phone = match.group(0)
        if len(phone) > 4:
            return phone[:2] + "X" * (len(phone) - 4) + phone[-2:]
        return phone

    result = PATTERNS["phone"].sub(redact_phone, result)

    # Partial redaction for emails
    def redact_email(match):
        email = match.group(0)
        parts = email.split("@")
        if len(parts) == 2:
            username, domain = parts
            if len(username) > 2:
                redacted = username[0] + "X" * (len(username) - 2) + username[-1]
            else:
                redacted = "X" * len(username)
            return f"{redacted}@{domain}"
        return email

    result = PATTERNS["email"].sub(redact_email, result)

    return result


def apply_preview_caps(
    messages: List[Dict[str, Any]], max_messages: int = 20, max_chars: int = 160
) -> List[Dict[str, Any]]:
    """
    Apply preview caps to limit message exposure.

    Args:
        messages: List of message dictionaries
        max_messages: Maximum number of messages
        max_chars: Maximum characters per message

    Returns:
        Capped list of messages
    """
    config = get_config()

    # Use config values if caps are enabled
    if config.privacy.preview_caps.get("enabled", True):
        max_messages = min(max_messages, config.privacy.preview_caps.get("max_messages", 20))
        max_chars = min(max_chars, config.privacy.preview_caps.get("max_chars", 160))

    # Limit number of messages
    result = messages[:max_messages]

    # Truncate message text
    for msg in result:
        if "text" in msg and msg["text"] and len(msg["text"]) > max_chars:
            msg["text"] = msg["text"][: max_chars - 3] + "..."

    return result


def sanitize_contact(
    contact: Dict[str, Any], redact: bool = True, hash_ids: bool = True
) -> Dict[str, Any]:
    """
    Sanitize contact information.

    Args:
        contact: Contact dictionary
        redact: Whether to apply redaction
        hash_ids: Whether to hash identifiers

    Returns:
        Sanitized contact
    """
    config = get_config()
    result = contact.copy()

    # Determine if we should hash/redact
    should_hash = hash_ids and config.privacy.hash_identifiers
    should_redact = redact and config.should_redact(redact)

    # Hash contact ID
    if should_hash and "id" in result:
        result["id"] = hash_contact_id(result["id"])

    if should_hash and "contact_id" in result:
        result["contact_id"] = hash_contact_id(result["contact_id"])

    # Hash handle/identifier
    if should_hash and "handle" in result:
        result["handle"] = hash_contact_id(result["handle"])

    # Redact sensitive fields
    if should_redact:
        # Remove address completely
        if "address" in result:
            result["address"] = "[ADDRESS REDACTED]"

        # Remove notes completely
        if "notes" in result:
            result["notes"] = "[NOTES REDACTED]"

        # Redact display name if requested
        if "display_name" in result and should_redact:
            result["display_name"] = None

    return result


def sanitize_message(
    message: Dict[str, Any], redact: bool = True, hash_ids: bool = True
) -> Dict[str, Any]:
    """
    Sanitize a message dictionary.

    Args:
        message: Message dictionary
        redact: Whether to apply redaction
        hash_ids: Whether to hash identifiers

    Returns:
        Sanitized message
    """
    config = get_config()
    result = message.copy()

    # Determine if we should hash/redact
    should_redact = redact and config.should_redact(redact)
    should_hash = hash_ids and config.privacy.hash_identifiers

    # Redact message text
    if should_redact and "text" in result:
        result["text"] = redact_pii(result["text"])

    # Hash sender/recipient IDs
    if should_hash:
        if "handle_id" in result:
            result["handle_id"] = hash_contact_id(result["handle_id"])

        if "contact_id" in result:
            result["contact_id"] = hash_contact_id(result["contact_id"])

        if "sender_id" in result:
            result["sender_id"] = hash_contact_id(result["sender_id"])

    return result


def build_contact_map(contacts: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Build a mapping of hashed IDs to original IDs.

    Args:
        contacts: List of contacts with original IDs

    Returns:
        Mapping of hashed to original IDs
    """
    config = get_config()
    salt = config.session_salt

    mapping = {}
    for contact in contacts:
        if "id" in contact:
            hashed = hash_contact_id(contact["id"], salt)
            mapping[hashed] = contact["id"]

        if "handle" in contact:
            hashed = hash_contact_id(contact["handle"], salt)
            mapping[hashed] = contact["handle"]

    return mapping


def apply_privacy_filters(data: Any) -> Any:
    """
    Apply privacy filters to data structures.

    This is a pass-through function that returns data as-is since
    privacy filtering is handled at the field level by other functions.

    Args:
        data: Data to filter

    Returns:
        The data unchanged (filtering is done elsewhere)
    """
    return data
