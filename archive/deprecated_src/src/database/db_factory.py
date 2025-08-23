#!/usr/bin/env python3
"""
Database Factory

This module provides a factory class for creating database instances.
The factory intelligently selects the appropriate implementation based on
configuration and database characteristics.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Union, Dict, Any

from .db_base import AsyncMessagesDBBase, DatabaseError

# Configure logging
logger = logging.getLogger(__name__)


class DatabaseFactory:
    """Factory class for creating database instances."""
    
    @staticmethod
    async def create_database(
        db_path: Optional[Union[str, Path]] = None,
        use_sharding: bool = False,
        shards_dir: Optional[str] = None,
        minimal_mode: bool = False,
        pool_size: int = 10,
        auto_detect_sharding: bool = True
    ) -> AsyncMessagesDBBase:
        """
        Create a database instance based on configuration.
        
        This factory method intelligently selects the appropriate database implementation
        based on the provided configuration and database characteristics. It can automatically
        detect when sharding would be beneficial for large databases.
        
        Args:
            db_path: Path to the database file
            use_sharding: Whether to use database sharding
            shards_dir: Directory for database shards
            minimal_mode: Whether to use minimal mode for better performance
            pool_size: Size of the connection pool
            auto_detect_sharding: Whether to automatically detect if sharding would be beneficial
            
        Returns:
            AsyncMessagesDBBase: An instance of a database class
            
        Raises:
            DatabaseError: If there's an error creating the database instance
        """
        logger.info(f"Creating database instance with path: {db_path}")
        logger.info(f"Configuration: sharding={use_sharding}, minimal_mode={minimal_mode}, pool_size={pool_size}")

        # First try to find default database path for macOS if not specified
        if not db_path:
            # Default database path for macOS
            HOME = os.path.expanduser("~")
            default_path = Path(f"{HOME}/Library/Messages/chat.db")
            
            # Check for indexed copy as fallback
            indexed_path = Path(f"{HOME}/.imessage_insights/indexed_chat.db")
            
            # Use default if exists and is readable
            if default_path.exists() and os.access(default_path, os.R_OK):
                db_path = default_path
                logger.info(f"Using default macOS database path: {db_path}")
            # Otherwise try indexed copy
            elif indexed_path.exists() and os.access(indexed_path, os.R_OK):
                db_path = indexed_path
                logger.info(f"Using indexed copy of database: {db_path}")
            else:
                logger.warning("Could not find a readable iMessage database")
                db_path = default_path  # Use default path so error is clear in logs
        
        # Normalize path to Path object
        if db_path and not isinstance(db_path, Path):
            db_path = Path(db_path)
        
        # Normalize shards_dir
        if not shards_dir and use_sharding:
            HOME = os.path.expanduser("~")
            shards_dir = str(Path(f"{HOME}/.imessage_insights/shards"))
            logger.info(f"Using default shards directory: {shards_dir}")
            
        # Determine if we should use sharding
        should_use_sharding = use_sharding
        
        # If sharding is explicitly requested, use it
        if use_sharding:
            logger.info("Database sharding explicitly enabled")
            should_use_sharding = True
        
        # Create appropriate database instance
        try:
            if should_use_sharding:
                # Import here to avoid circular imports
                from .sharded_async_messages_db import ShardedAsyncMessagesDB
                
                logger.info("Creating sharded database instance")
                db = ShardedAsyncMessagesDB(
                    db_path=str(db_path) if db_path else None,
                    shards_dir=shards_dir,
                    minimal_mode=minimal_mode
                )
                await db.initialize()
                return db
            else:
                # Import here to avoid circular imports
                from .async_messages_db_new import AsyncMessagesDB
                
                logger.info("Creating standard database instance")
                db = AsyncMessagesDB(
                    db_path=str(db_path) if db_path else None, 
                    minimal_mode=minimal_mode,
                    pool_size=pool_size
                )
                
                # Initialize the database
                await db.initialize()
                
                # Check if database is large enough to benefit from sharding
                if auto_detect_sharding and await db.should_use_sharding():
                    logger.info("Database detected as large, switching to sharded implementation")
                    
                    # Close the standard database
                    await db.close()
                    
                    # Import and create sharded implementation
                    from .sharded_async_messages_db import ShardedAsyncMessagesDB
                    
                    sharded_db = ShardedAsyncMessagesDB(
                        db_path=str(db_path) if db_path else None,
                        shards_dir=shards_dir,
                        minimal_mode=minimal_mode
                    )
                    await sharded_db.initialize()
                    return sharded_db
                
                # Return the standard database
                return db
                
        except ImportError as e:
            logger.error(f"Error importing database implementation: {e}")
            raise DatabaseError(f"Failed to import database implementation: {e}")
        except Exception as e:
            logger.error(f"Error creating database instance: {e}")
            raise DatabaseError(f"Failed to create database instance: {e}")
    
    @staticmethod
    async def get_database_info(db_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Get information about a database without fully initializing it.
        
        This method provides basic information about the database file,
        such as its size, whether it's indexed, etc.
        
        Args:
            db_path: Path to the database file
            
        Returns:
            Dict[str, Any]: Information about the database
        """
        # Determine database path
        if not db_path:
            HOME = os.path.expanduser("~")
            db_path = Path(f"{HOME}/Library/Messages/chat.db")
        
        # Normalize path
        if not isinstance(db_path, Path):
            db_path = Path(db_path)
        
        # Get information
        info = {
            "path": str(db_path),
            "exists": db_path.exists(),
            "readable": os.access(db_path, os.R_OK) if db_path.exists() else False,
            "size_bytes": 0,
            "size_mb": 0,
            "size_gb": 0,
            "is_indexed": False,
            "should_use_sharding": False,
        }
        
        # If file exists, get more information
        if info["exists"] and info["readable"]:
            size_bytes = db_path.stat().st_size
            info["size_bytes"] = size_bytes
            info["size_mb"] = size_bytes / (1024 * 1024)
            info["size_gb"] = size_bytes / (1024 * 1024 * 1024)
            
            # Determine if this is an indexed copy
            info["is_indexed"] = ".imessage_insights" in str(db_path)
            
            # Determine if sharding would be beneficial
            info["should_use_sharding"] = info["size_gb"] >= 5
        
        return info
