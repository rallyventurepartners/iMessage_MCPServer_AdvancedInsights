"""
Data sanitization utilities for the MCP Server application.

This module provides functions for sanitizing sensitive information 
from messages, contacts, and other data before sending it to clients.
It helps protect privacy and comply with data protection requirements.
"""

import re
from typing import Any, Dict, List, Optional, Union

# Regular expression patterns for sensitive data
CREDIT_CARD_PATTERN = r'\b(?:\d{4}[- ]?){3}\d{4}\b'
SSN_PATTERN = r'\b\d{3}-\d{2}-\d{4}\b'
PHONE_NUMBER_PATTERN = r'\b(?:\+\d{1,2}\s?)?(?:\(\d{3}\)\s?|\d{3}[-.\s]?)\d{3}[-.\s]?\d{4}\b'
EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
ADDRESS_PATTERN = r'\b\d+\s+[A-Za-z0-9\s,]+(?:Avenue|Lane|Road|Boulevard|Drive|Street|Ave|Dr|Rd|Blvd|Ln|St)\.?\b'
FINANCIAL_AMOUNT_PATTERN = r'\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?'


def sanitize_message_content(content: Optional[str]) -> Optional[str]:
    """
    Sanitize sensitive information from message content.
    
    Args:
        content: The message content to sanitize
        
    Returns:
        Sanitized message content
    """
    if not content:
        return content
    
    # Replace credit card numbers
    content = re.sub(
        CREDIT_CARD_PATTERN,
        '[CREDIT CARD REDACTED]',
        content
    )
    
    # Replace SSNs
    content = re.sub(
        SSN_PATTERN,
        '[SSN REDACTED]',
        content
    )
    
    # Partially redact phone numbers (keep first and last 2 digits)
    def redact_phone(match):
        phone = match.group(0)
        if len(phone) > 4:
            return phone[:2] + 'X' * (len(phone) - 4) + phone[-2:]
        return phone
    
    content = re.sub(
        PHONE_NUMBER_PATTERN,
        redact_phone,
        content
    )
    
    # Partially redact email addresses (keep domain)
    def redact_email(match):
        email = match.group(0)
        parts = email.split('@')
        if len(parts) == 2:
            username, domain = parts
            if len(username) > 2:
                redacted_username = username[0] + 'X' * (len(username) - 2) + username[-1]
            else:
                redacted_username = 'X' * len(username)
            return f"{redacted_username}@{domain}"
        return email
    
    content = re.sub(
        EMAIL_PATTERN,
        redact_email,
        content
    )
    
    # Redact addresses
    content = re.sub(
        ADDRESS_PATTERN,
        '[ADDRESS REDACTED]',
        content
    )

    # Redact financial amounts
    content = re.sub(
        FINANCIAL_AMOUNT_PATTERN,
        '[AMOUNT REDACTED]',
        content
    )
    
    return content


def sanitize_contact_info(
    contact: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    Sanitize sensitive information from contact data.
    
    Args:
        contact: Contact information dictionary
        
    Returns:
        Sanitized contact information
    """
    if not contact:
        return contact
    
    # Create a copy to avoid modifying the original
    result = contact.copy()
    
    # Keep phone numbers and emails intact for authenticated users
    # Only redact truly sensitive information like addresses and notes
    
    # Redact address completely
    if 'address' in result:
        result['address'] = '[ADDRESS REDACTED]'
    
    # Redact notes completely (may contain sensitive information)
    if 'notes' in result:
        result['notes'] = '[NOTES REDACTED]'
    
    # Keep identifier field if present (new unified field)
    if 'identifier' in result:
        # Ensure it's properly formatted but not masked
        result['identifier'] = result['identifier']
    
    return result


def sanitize_message(
    message: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    Sanitize a single message object.
    
    Args:
        message: Message dictionary
        
    Returns:
        Sanitized message
    """
    if not message:
        return message
    
    # Create a copy to avoid modifying the original
    result = message.copy()
    
    # Sanitize text content
    if 'text' in result:
        result['text'] = sanitize_message_content(result['text'])
    
    # Sanitize subject (if present)
    if 'subject' in result:
        result['subject'] = sanitize_message_content(result['subject'])
    
    # Sanitize sender information
    if 'sender' in result and isinstance(result['sender'], dict):
        result['sender'] = sanitize_contact_info(result['sender'])
    
    # Sanitize recipient information
    if 'recipients' in result and isinstance(result['recipients'], list):
        result['recipients'] = [
            sanitize_contact_info(recipient) 
            for recipient in result['recipients']
        ]
    
    # Sanitize any attachment descriptions
    if 'attachments' in result and isinstance(result['attachments'], list):
        for attachment in result['attachments']:
            if isinstance(attachment, dict) and 'filename' in attachment:
                # Don't sanitize filenames - they're needed for retrieval
                pass
    
    return result


def sanitize_messages(
    messages: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Sanitize a list of message objects.
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        List of sanitized messages
    """
    return [sanitize_message(msg) for msg in messages]


def sanitize_group_chat_data(
    chat_data: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    Sanitize group chat data.
    
    Args:
        chat_data: Group chat dictionary
        
    Returns:
        Sanitized group chat data
    """
    if not chat_data:
        return chat_data
    
    # Create a copy to avoid modifying the original
    result = chat_data.copy()
    
    # Redact display name if it contains sensitive information
    if 'display_name' in result:
        # Usually, group chat names are not sensitive, but check for patterns
        display_name = result['display_name']
        
        # Check if the display name contains phone numbers, emails, etc.
        if display_name and (re.search(PHONE_NUMBER_PATTERN, display_name) or
                            re.search(EMAIL_PATTERN, display_name) or
                            re.search(ADDRESS_PATTERN, display_name)):
            # Replace with generic name if sensitive information found
            result['display_name'] = f"Group Chat {result.get('chat_id', '')}"
    
    # Sanitize participants
    if 'participants' in result and isinstance(result['participants'], list):
        result['participants'] = [
            sanitize_contact_info(participant)
            for participant in result['participants']
        ]
    
    return result


def sanitize_analysis_result(
    analysis: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Sanitize analysis results that might contain sensitive data.
    
    Args:
        analysis: Analysis result dictionary
        
    Returns:
        Sanitized analysis result
    """
    # Create a copy to avoid modifying the original
    result = analysis.copy()
    
    # Sanitize any message content in the analysis
    if 'messages' in result:
        result['messages'] = sanitize_messages(result['messages'])
    
    if 'message_history' in result and 'messages' in result['message_history']:
        result['message_history']['messages'] = sanitize_messages(
            result['message_history']['messages']
        )
    
    # Sanitize contact information
    if 'contact' in result:
        result['contact'] = sanitize_contact_info(result['contact'])
    
    # Sanitize chat data
    if 'chat' in result:
        result['chat'] = sanitize_group_chat_data(result['chat'])
    
    # Sanitize participants
    if 'participants' in result:
        if isinstance(result['participants'], list):
            result['participants'] = [
                sanitize_contact_info(participant)
                for participant in result['participants']
            ]
        elif isinstance(result['participants'], dict) and 'participants' in result['participants']:
            result['participants']['participants'] = [
                sanitize_contact_info(participant)
                for participant in result['participants']['participants']
            ]
    
    # Sanitize conversation summary (might contain quotes with sensitive data)
    if 'conversation_summary' in result:
        result['conversation_summary'] = sanitize_message_content(
            result['conversation_summary']
        )
    
    # Sanitize topic keywords
    if 'topics' in result:
        # Topics themselves usually don't contain PII, but example quotes might
        topics = result['topics']
        for topic in topics:
            if isinstance(topic, dict) and 'example_messages' in topic:
                topic['example_messages'] = [
                    sanitize_message_content(msg) 
                    for msg in topic['example_messages']
                ]
    
    return result
