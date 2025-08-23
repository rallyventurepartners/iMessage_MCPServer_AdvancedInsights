"""
Configuration management for the MCP Server application.

This module provides utilities for loading, validating, and accessing
configuration settings throughout the application.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from ..exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class ServerConfig:
    """
    Server configuration management.
    
    This class implements the Singleton pattern to ensure only one
    instance exists throughout the application. It handles loading
    configuration from various sources, validating settings, and
    providing access to configuration values.
    """
    
    # Singleton instance
    _instance = None
    
    @classmethod
    def get_instance(cls) -> 'ServerConfig':
        """Get the singleton instance of the ServerConfig."""
        if cls._instance is None:
            cls._instance = ServerConfig()
        return cls._instance
    
    def __init__(self):
        """Initialize the ServerConfig."""
        # Prevent multiple instances
        if ServerConfig._instance is not None:
            raise RuntimeError("ServerConfig is a singleton. Use get_instance() instead.")
        
        # Default configuration values
        self.config: Dict[str, Any] = {
            # Server settings
            "server": {
                "name": "iMessage Advanced Insights",
                "version": "2.4.0",
                "port": 5000,
                "log_level": "INFO",
            },
            
            # Database settings
            "database": {
                "use_sharding": None,  # Auto-detect based on size
                "minimal_mode": False,
                "pool_size": 10,
                "shard_size_months": 6,
                "auto_optimize": True,
            },
            
            # Memory monitoring
            "memory": {
                "enable_monitoring": True,
                "limit_mb": 2048,  # 2GB default limit
                "high_threshold": 0.75,
                "low_threshold": 0.5,
                "monitor_interval": 5,  # seconds
            },
            
            # Logging
            "logging": {
                "file_logging": True,
                "console_logging": True,
                "log_directory": "~/.imessage_insights/logs",
                "log_filename": "mcp_server.log",
                "max_log_size": 10485760,  # 10MB
                "backup_count": 3,
            },
            
            # Privacy
            "privacy": {
                "sanitize_messages": True,
                "sanitize_contacts": True,
                "consent_expiry_hours": 24,
            },
            
            # Features
            "features": {
                "enable_network_visualization": True,
                "enable_topic_analysis": True,
                "enable_sentiment_analysis": True,
                "enable_message_search": True,
            },
        }
        
        # Set as singleton instance
        ServerConfig._instance = self
    
    def load_from_file(self, file_path: str) -> bool:
        """
        Load configuration from a JSON file.
        
        Args:
            file_path: Path to the configuration file
            
        Returns:
            True if configuration was successfully loaded
            
        Raises:
            ConfigurationError: If there's an error loading the configuration
        """
        try:
            file_path = os.path.expanduser(file_path)
            
            if not os.path.exists(file_path):
                logger.warning(f"Config file not found: {file_path}")
                return False
            
            with open(file_path, "r") as f:
                file_config = json.load(f)
            
            # Recursively update the configuration
            self._update_config_recursive(self.config, file_config)
            
            logger.info(f"Configuration loaded from {file_path}")
            return True
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing config file: {e}")
            raise ConfigurationError(
                f"Invalid JSON in configuration file: {str(e)}",
                {"file_path": file_path}
            )
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise ConfigurationError(
                f"Failed to load configuration: {str(e)}",
                {"file_path": file_path}
            )
    
    def _update_config_recursive(
        self, target: Dict[str, Any], source: Dict[str, Any]
    ) -> None:
        """
        Recursively update configuration dictionary.
        
        This preserves the structure of the target dictionary while
        updating values from the source dictionary.
        
        Args:
            target: Target dictionary to update
            source: Source dictionary with new values
        """
        for key, value in source.items():
            if (
                key in target and
                isinstance(target[key], dict) and
                isinstance(value, dict)
            ):
                # Recursively update nested dictionaries
                self._update_config_recursive(target[key], value)
            else:
                # Set or replace value
                target[key] = value
    
    def load_from_env(self, prefix: str = "MCP_") -> None:
        """
        Load configuration from environment variables.
        
        Environment variables should be in the format:
        PREFIX_SECTION_KEY=value
        
        For example:
        MCP_SERVER_PORT=5000
        MCP_DATABASE_USE_SHARDING=true
        
        Args:
            prefix: Prefix for environment variables to consider
        """
        try:
            for env_name, env_value in os.environ.items():
                if not env_name.startswith(prefix):
                    continue
                
                # Remove prefix and split into parts
                name_parts = env_name[len(prefix):].lower().split("_")
                
                if len(name_parts) < 2:
                    logger.warning(f"Skipping invalid environment variable: {env_name}")
                    continue
                
                # Extract section and key
                section = name_parts[0]
                key = "_".join(name_parts[1:])
                
                # Skip if section doesn't exist in config
                if section not in self.config:
                    logger.warning(f"Unknown configuration section: {section}")
                    continue
                
                # Skip if key doesn't exist in section
                if key not in self.config[section]:
                    logger.warning(f"Unknown configuration key: {section}.{key}")
                    continue
                
                # Convert value to appropriate type based on existing value
                existing_value = self.config[section][key]
                converted_value = self._convert_env_value(env_value, type(existing_value))
                
                # Update configuration
                self.config[section][key] = converted_value
                logger.debug(f"Updated configuration from environment: {section}.{key}")
        except Exception as e:
            logger.error(f"Error loading configuration from environment: {e}")
    
    def _convert_env_value(self, value: str, target_type: type) -> Any:
        """
        Convert string value from environment to the appropriate type.
        
        Args:
            value: String value from environment variable
            target_type: Target type to convert to
            
        Returns:
            Converted value of the appropriate type
        """
        if target_type == bool:
            return value.lower() in ("true", "yes", "1", "t", "y")
        elif target_type == int:
            return int(value)
        elif target_type == float:
            return float(value)
        elif target_type == list:
            return [item.strip() for item in value.split(",")]
        else:
            return value
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        if section not in self.config:
            return default
        
        if key not in self.config[section]:
            return default
        
        return self.config[section][key]
    
    def set(self, section: str, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key
            value: New value
        """
        if section not in self.config:
            self.config[section] = {}
        
        self.config[section][key] = value
    
    def get_database_config(self) -> Dict[str, Any]:
        """
        Get database configuration.
        
        Returns:
            Dictionary with database configuration
        """
        return self.config.get("database", {})
    
    def get_memory_config(self) -> Dict[str, Any]:
        """
        Get memory monitoring configuration.
        
        Returns:
            Dictionary with memory monitoring configuration
        """
        return self.config.get("memory", {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """
        Get logging configuration.
        
        Returns:
            Dictionary with logging configuration
        """
        return self.config.get("logging", {})
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """
        Check if a feature is enabled.
        
        Args:
            feature_name: Name of the feature to check
            
        Returns:
            True if the feature is enabled
        """
        features = self.config.get("features", {})
        key = f"enable_{feature_name}"
        
        if key in features:
            return features[key]
        
        # Legacy format support
        return features.get(feature_name, False)


# Convenience functions for accessing configuration

def get_config(section: str, key: str, default: Any = None) -> Any:
    """
    Get a configuration value.
    
    Args:
        section: Configuration section
        key: Configuration key
        default: Default value if key is not found
        
    Returns:
        Configuration value or default
    """
    config = ServerConfig.get_instance()
    return config.get(section, key, default)


def is_feature_enabled(feature_name: str) -> bool:
    """
    Check if a feature is enabled.
    
    Args:
        feature_name: Name of the feature to check
        
    Returns:
        True if the feature is enabled
    """
    config = ServerConfig.get_instance()
    return config.is_feature_enabled(feature_name)


def init_config(config_file: Optional[str] = None) -> None:
    """
    Initialize configuration from file and environment.
    
    Args:
        config_file: Optional path to configuration file
    """
    config = ServerConfig.get_instance()
    
    # Load from file if provided
    if config_file:
        try:
            config.load_from_file(config_file)
        except ConfigurationError as e:
            logger.warning(f"Error loading configuration file: {e}")
    
    # Load from environment
    config.load_from_env()
