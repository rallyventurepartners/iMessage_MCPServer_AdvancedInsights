"""
Logging configuration for the MCP Server application.

This module provides utilities for configuring and using logging
throughout the application.
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Dict, Optional

from .config import ServerConfig

# Default log format
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def configure_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """
    Configure logging for the application.
    
    This function configures logging based on the application configuration
    and any overrides provided. It sets up both console and file logging
    according to the configuration.
    
    Args:
        log_level: Override for log level (e.g., "DEBUG", "INFO")
        log_file: Override for log file path
        log_format: Override for log format
        
    Returns:
        Root logger configured according to settings
    """
    # Get configuration
    config = ServerConfig.get_instance()
    logging_config = config.get_logging_config()
    
    # Determine log level
    if log_level is None:
        log_level = config.get("server", "log_level", "INFO")
    
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Determine log format
    if log_format is None:
        log_format = DEFAULT_LOG_FORMAT
    
    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Configure formatter
    formatter = logging.Formatter(log_format)
    
    # Add console handler if enabled
    if logging_config.get("console_logging", True):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Add file handler if enabled
    if logging_config.get("file_logging", True):
        # Determine log file path
        if log_file is None:
            log_dir = os.path.expanduser(logging_config.get(
                "log_directory", "~/.imessage_insights/logs"
            ))
            log_filename = logging_config.get("log_filename", "mcp_server.log")
            log_file = os.path.join(log_dir, log_filename)
        
        # Create log directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        try:
            # Determine rotation settings
            max_bytes = logging_config.get("max_log_size", 10 * 1024 * 1024)  # 10MB default
            backup_count = logging_config.get("backup_count", 3)
            
            # Create rotating file handler
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=backup_count
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            
            root_logger.info(f"Log file configured at: {log_file}")
        except (IOError, PermissionError) as e:
            # Don't fail if file logging can't be configured
            root_logger.warning(f"Could not configure file logging: {e}")
            root_logger.warning("Continuing with console logging only")
    
    # Return the root logger
    root_logger.info(f"Logging initialized with level {log_level}")
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    This is a convenience function for getting a logger with the
    appropriate name based on the module hierarchy.
    
    Args:
        name: Name of the logger
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LoggingContext:
    """
    Context manager for temporarily changing log level.
    
    This allows temporarily changing the log level for a specific
    block of code, useful for debugging.
    
    Example:
        with LoggingContext("my_module", level=logging.DEBUG):
            # Code that needs DEBUG logging
    """
    
    def __init__(self, logger_name: str, level: int):
        """
        Initialize the context manager.
        
        Args:
            logger_name: Name of the logger to modify
            level: New log level to set temporarily
        """
        self.logger = logging.getLogger(logger_name)
        self.level = level
        self.old_level = self.logger.level
    
    def __enter__(self):
        """Set the temporary log level."""
        self.old_level = self.logger.level
        self.logger.setLevel(self.level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore the original log level."""
        self.logger.setLevel(self.old_level)


def get_log_levels() -> Dict[str, int]:
    """
    Get a dictionary of available log levels.
    
    Returns:
        Dictionary mapping level names to numeric values
    """
    return {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
    }
