"""
Configuration management for the iMessage MCP Server.

This module handles all configuration settings including paths, consent windows,
redaction flags, and feature toggles.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class PrivacyConfig:
    """Privacy-related configuration."""

    redact_by_default: bool = True
    hash_identifiers: bool = True
    preview_caps: Dict[str, int] = field(
        default_factory=lambda: {"enabled": True, "max_messages": 20, "max_chars": 160}
    )
    audit_logging: bool = True


@dataclass
class ConsentConfig:
    """Consent management configuration."""

    default_duration_hours: int = 24
    max_duration_hours: int = 720  # 30 days
    require_explicit: bool = True


@dataclass
class DatabaseConfig:
    """Database configuration."""

    path: str = "~/Library/Messages/chat.db"
    read_only: bool = True
    timeout_seconds: int = 30
    use_sharding: bool = False
    shards_dir: Optional[str] = None
    auto_detect_sharding: bool = True


@dataclass
class PerformanceConfig:
    """Performance-related configuration."""

    memory_limit_mb: int = 250
    query_timeout_s: int = 30
    max_concurrent_queries: int = 5
    cache_ttl_seconds: int = 300


@dataclass
class FeatureFlags:
    """Feature toggles."""

    allow_exports: bool = False
    enable_network_egress: bool = False
    use_transformer_nlp: bool = False
    enable_memory_monitoring: bool = True


@dataclass
class Config:
    """Main configuration class."""

    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    consent: ConsentConfig = field(default_factory=ConsentConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    features: FeatureFlags = field(default_factory=FeatureFlags)

    # Session-specific settings (not persisted)
    session_salt: Optional[bytes] = None

    def __post_init__(self):
        """Generate session salt on initialization."""
        self.session_salt = os.urandom(16)  # BLAKE2b max salt length

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create config from dictionary."""
        return cls(
            privacy=PrivacyConfig(**data.get("privacy", {})),
            consent=ConsentConfig(**data.get("consent", {})),
            database=DatabaseConfig(**data.get("database", {})),
            performance=PerformanceConfig(**data.get("performance", {})),
            features=FeatureFlags(**data.get("features", {})),
        )

    @classmethod
    def from_file(cls, path: Path) -> "Config":
        """Load config from JSON file."""
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                return cls.from_dict(data)
        return cls()

    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables."""
        config = cls()

        # Privacy settings
        if env_val := os.getenv("IMSG_REDACT_DEFAULT"):
            config.privacy.redact_by_default = env_val.lower() == "true"

        if env_val := os.getenv("IMSG_HASH_IDENTIFIERS"):
            config.privacy.hash_identifiers = env_val.lower() == "true"

        # Consent settings
        if env_val := os.getenv("IMSG_CONSENT_WINDOW_HOURS"):
            config.consent.default_duration_hours = int(env_val)

        # Database settings
        if env_val := os.getenv("IMSG_DATABASE_PATH"):
            config.database.path = env_val

        if env_val := os.getenv("IMSG_USE_SHARDING"):
            config.database.use_sharding = env_val.lower() == "true"

        # Performance settings
        if env_val := os.getenv("IMSG_MEMORY_LIMIT_MB"):
            config.performance.memory_limit_mb = int(env_val)

        # Feature flags
        if env_val := os.getenv("IMSG_ALLOW_EXPORTS"):
            config.features.allow_exports = env_val.lower() == "true"

        if env_val := os.getenv("IMSG_USE_TRANSFORMER"):
            config.features.use_transformer_nlp = env_val.lower() == "true"

        return config

    def get_db_path(self) -> Path:
        """Get expanded database path."""
        return Path(self.database.path).expanduser()

    def should_redact(self, explicit_redact: Optional[bool] = None) -> bool:
        """Determine if redaction should be applied."""
        if explicit_redact is not None:
            return explicit_redact
        return self.privacy.redact_by_default

    @property
    def consent_db_path(self) -> Path:
        """Get path to consent database."""
        return Path("~/.imessage-mcp/consent.db").expanduser()

    @property
    def db_path(self) -> str:
        """Get database path as string."""
        return self.database.path


def load_config() -> Config:
    """Load configuration from file or environment."""
    # Check for config file in standard locations
    config_paths = [
        Path("config.json"),
        Path("~/.imessage-mcp/config.json").expanduser(),
        Path("/etc/imessage-mcp/config.json"),
    ]

    for path in config_paths:
        if path.exists():
            return Config.from_file(path)

    # Fall back to environment variables
    return Config.from_env()


# Global config instance (lazy loaded)
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
