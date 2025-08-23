"""
Utility modules for the iMessage Advanced Insights application.

This package contains utility modules used throughout the application.
"""

# Import key utilities for easy access
from .config import get_config, is_feature_enabled, init_config
from .logging_config import configure_logging, get_logger
from .consent_manager import ConsentManager
from .memory_monitor import initialize_memory_monitor, limit_memory
from .responses import success_response, error_response, paginated_response
from .sanitization import (
    sanitize_message_content, 
    sanitize_contact_info,
    sanitize_message,
    sanitize_messages,
    sanitize_group_chat_data,
    sanitize_analysis_result
)

__all__ = [
    # Configuration
    "get_config",
    "is_feature_enabled",
    "init_config",
    
    # Logging
    "configure_logging",
    "get_logger",
    
    # Consent management
    "ConsentManager",
    
    # Memory monitoring
    "initialize_memory_monitor",
    "limit_memory",
    
    # Response formatting
    "success_response",
    "error_response",
    "paginated_response",
    
    # Data sanitization
    "sanitize_message_content",
    "sanitize_contact_info",
    "sanitize_message",
    "sanitize_messages",
    "sanitize_group_chat_data",
    "sanitize_analysis_result",
]
