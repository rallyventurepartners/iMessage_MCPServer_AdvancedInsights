"""
Database access factory for the MCP Server application.

This module provides a unified interface for creating and accessing database instances,
handling both regular and sharded database configurations. It abstracts the underlying
implementation details to provide a consistent interface for database access.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union, TYPE_CHECKING

# Import for type checking only to avoid circular imports
if TYPE_CHECKING:
    from .async_messages_db_new import AsyncMessagesDB
    from .sharded_async_messages_db import ShardedAsyncMessagesDB

from ..exceptions import ConfigurationError, DatabaseError
from ..utils.decorators import retry_async

logger = logging.getLogger(__name__)


async def get_database(
    db_path: Optional[str] = None,
    minimal_mode: bool = False,
    use_shards: Optional[bool] = None,
    shards_dir: Optional[str] = None,
) -> 'AsyncMessagesDBBase':
    """
    Factory function to get the appropriate database instance.
    
    This function determines whether to use a regular database or a sharded one
    based on the configuration and database size. It handles initialization and
    fallback mechanisms automatically.
    
    Args:
        db_path: Path to the database file (optional)
        minimal_mode: Whether to enable minimal mode for better performance
        use_shards: Force use of sharding (None = auto-detect based on size)
        shards_dir: Directory for database shards
        
    Returns:
        AsyncMessagesDBBase: Database instance (normal or sharded)
        
    Raises:
        ConfigurationError: If configuration is invalid
        DatabaseError: If database initialization fails
    """
    # Determine database path
    if not db_path:
        home = os.path.expanduser("~")
        primary_db_path = str(Path(home) / "Library" / "Messages" / "chat.db")
        
        # Check for fallback indexed copy if main database isn't accessible
        fallback_indexed_path = str(
            Path(home) / ".imessage_insights" / "indexed_chat.db"
        )
        
        # Try the primary path first
        if os.path.exists(primary_db_path) and os.access(primary_db_path, os.R_OK):
            db_path = primary_db_path
            logger.info(f"Using primary database path: {db_path}")
        # If primary not accessible, try fallback indexed copy
        elif os.path.exists(fallback_indexed_path) and os.access(fallback_indexed_path, os.R_OK):
            db_path = fallback_indexed_path
            logger.info(f"Primary database not accessible, using indexed copy: {db_path}")
        else:
            # No accessible database, but continue anyway to show error in tools
            logger.warning(f"No accessible database found at {primary_db_path} or {fallback_indexed_path}")
            raise DatabaseError(
                "No accessible database found. Please ensure that the Terminal app has "
                "Full Disk Access permission in System Settings > Privacy & Security.",
                {
                    "primary_path": primary_db_path,
                    "fallback_path": fallback_indexed_path,
                    "error_type": "file_not_accessible"
                }
            )
    
    # Validate database path
    if not os.path.exists(db_path):
        raise DatabaseError(
            f"Database file not found: {db_path}",
            {"error_type": "file_not_found"}
        )
    
    if not os.access(db_path, os.R_OK):
        raise DatabaseError(
            f"Database file not readable: {db_path}",
            {"error_type": "permission_denied"}
        )
    
    # Get database size
    db_size_gb = 0
    if os.path.exists(db_path) and os.access(db_path, os.R_OK):
        db_size_bytes = os.path.getsize(db_path)
        db_size_gb = db_size_bytes / (1024 * 1024 * 1024)
        logger.info(f"Database size: {db_size_gb:.2f} GB")
    
    # Determine if sharding should be used
    if use_shards is None:
        # Auto-detect based on size (10GB+ typically benefits from sharding)
        use_shards = db_size_gb >= 10
        logger.info(f"Auto-detected sharding mode: {'enabled' if use_shards else 'disabled'}")
    
    # Get shards directory
    if not shards_dir:
        home = os.path.expanduser("~")
        shards_dir = str(Path(home) / ".imessage_insights" / "shards")
        
        # Ensure directory exists
        os.makedirs(shards_dir, exist_ok=True)
    
    # Initialize appropriate database type
    if use_shards:
        return await _get_sharded_database(
            db_path=db_path,
            shards_dir=shards_dir,
            minimal_mode=minimal_mode
        )
    else:
        return await _get_standard_database(
            db_path=db_path,
            minimal_mode=minimal_mode
        )


@retry_async(max_retries=2)
async def _get_standard_database(
    db_path: str,
    minimal_mode: bool = False
) -> 'AsyncMessagesDB':
    """
    Get a standard (non-sharded) database instance.
    
    Args:
        db_path: Path to the database file
        minimal_mode: Whether to enable minimal mode for better performance
        
    Returns:
        AsyncMessagesDB: Standard database instance
        
    Raises:
        DatabaseError: If database initialization fails
    """
    logger.info(f"Initializing standard database: {db_path}")
    
    try:
        # Import here to avoid circular imports
        from .async_messages_db_new import AsyncMessagesDB
        
        # Create and initialize database
        db = AsyncMessagesDB(db_path=db_path, minimal_mode=minimal_mode)
        await db.initialize()
        
        # Validate schema version
        await db.validate_schema_version()
        
        logger.info("Standard database initialized successfully")
        return db
    except Exception as e:
        logger.error(f"Error initializing standard database: {e}")
        raise DatabaseError(f"Failed to initialize database: {str(e)}")


@retry_async(max_retries=2)
async def _get_sharded_database(
    db_path: str,
    shards_dir: str,
    minimal_mode: bool = False,
    shard_size_months: int = 6
) -> 'ShardedAsyncMessagesDB':
    """
    Get a sharded database instance for large databases.
    
    Args:
        db_path: Path to the source database file
        shards_dir: Directory to store database shards
        minimal_mode: Whether to enable minimal mode for better performance
        shard_size_months: Size of each shard in months
        
    Returns:
        ShardedAsyncMessagesDB: Sharded database instance
        
    Raises:
        DatabaseError: If database initialization fails
    """
    logger.info(f"Initializing sharded database with source: {db_path}")
    logger.info(f"Using shards directory: {shards_dir}")
    
    try:
        # Import here to avoid circular imports
        from .large_db_handler import DatabaseShard, LargeDatabaseManager
        from .sharded_async_messages_db import ShardedAsyncMessagesDB
        
        # Initialize the shard manager
        shard_manager = LargeDatabaseManager(
            source_db_path=db_path,
            shards_dir=shards_dir,
            shard_size_months=shard_size_months
        )
        
        # Initialize shards
        logger.info("Initializing database shards...")
        shard_init_success = await shard_manager.initialize()
        
        if not shard_init_success:
            logger.error("Failed to initialize database shards")
            # Fall back to normal database mode
            logger.warning("Falling back to normal database mode")
            return await _get_standard_database(db_path, minimal_mode)
        
        # Mark environment as using shards for other modules
        os.environ["USING_DATABASE_SHARDS"] = "1"
        
        # Create and initialize sharded database
        db = ShardedAsyncMessagesDB(
            source_db_path=db_path,
            shard_manager=shard_manager,
            minimal_mode=minimal_mode
        )
        
        await db.initialize()
        logger.info("Sharded database initialized successfully")
        
        return db
    except Exception as e:
        logger.error(f"Error initializing sharded database: {e}")
        # Try to fall back to standard database
        logger.warning("Attempting to fall back to standard database")
        return await _get_standard_database(db_path, minimal_mode)


# Abstract base class stub for type checking
# The real implementation would be in db_base.py
class AsyncMessagesDBBase:
    """Base class for all database implementations."""
    
    @property
    def initialized(self) -> bool:
        """Whether the database has been initialized."""
        raise NotImplementedError
    
    async def initialize(self) -> None:
        """Initialize the database connection."""
        raise NotImplementedError
    
    async def close(self) -> None:
        """Close the database connection."""
        raise NotImplementedError
    
    async def validate_schema_version(self) -> bool:
        """Validate that the database schema is compatible."""
        raise NotImplementedError
