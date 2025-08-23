#!/usr/bin/env python3
"""
Sharded Async Messages Database

This module provides a database interface for extremely large iMessage databases,
using time-based database sharding to improve performance and resource usage.
"""

import asyncio
import json
import logging
import os
import re
import shutil
import stat
import time
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import (Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple,
                    Union)

import aiosqlite

# Import base classes and utilities
from src.database.db_base import AsyncMessagesDBBase, DatabaseError
from src.database.db_utils import (
    build_date_filters_sql,
    build_pagination_info,
    clean_message_text,
    convert_apple_timestamp,
    datetime_to_apple_timestamp,
    format_date_for_display,
    parse_date_string,
    sanitize_query_params,
)
# Import for shard connections
from src.database.async_messages_db_new import AsyncMessagesDB

# Configure logging
logger = logging.getLogger(__name__)


class ShardConfig:
    """Configuration for a database shard."""

    def __init__(
        self,
        shard_id: str,
        shard_path: Path,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        description: str = "",
        size_mb: float = 0.0,
    ):
        """Initialize a shard configuration.

        Args:
            shard_id: Unique identifier for this shard
            shard_path: Path to the shard database file
            start_date: Start date for messages in this shard
            end_date: End date for messages in this shard
            description: Human-readable description of this shard
            size_mb: Size of the shard in megabytes
        """
        self.shard_id = shard_id
        self.shard_path = shard_path
        self.start_date = start_date
        self.end_date = end_date
        self.description = description
        self.size_mb = size_mb
        self.message_count = 0
        self.is_active = True
        self.db_instance = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of this shard config
        """
        return {
            "shard_id": self.shard_id,
            "shard_path": str(self.shard_path),
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "description": self.description,
            "size_mb": self.size_mb,
            "message_count": self.message_count,
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShardConfig":
        """Create a ShardConfig from a dictionary.

        Args:
            data: Dictionary with shard configuration

        Returns:
            ShardConfig instance
        """
        start_date = None
        if data.get("start_date"):
            try:
                start_date = datetime.fromisoformat(data["start_date"])
            except:
                pass

        end_date = None
        if data.get("end_date"):
            try:
                end_date = datetime.fromisoformat(data["end_date"])
            except:
                pass

        config = cls(
            shard_id=data["shard_id"],
            shard_path=Path(data["shard_path"]),
            start_date=start_date,
            end_date=end_date,
            description=data.get("description", ""),
            size_mb=data.get("size_mb", 0.0),
        )

        config.message_count = data.get("message_count", 0)
        config.is_active = data.get("is_active", True)

        return config


class ShardedAsyncMessagesDB(AsyncMessagesDBBase):
    """A database manager for extremely large iMessage databases using sharding.
    
    This implementation divides the database into time-based shards for
    improved performance with very large message histories.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        shards_dir: Optional[str] = None,
        minimal_mode: bool = False,
        shard_manager=None,
        source_db_path=None,
    ):
        """Initialize the sharded database manager.

        Args:
            db_path: Path to the main database file (used as a fallback)
            shards_dir: Directory containing the database shards
            minimal_mode: Whether to return minimal data for better performance
            shard_manager: Optional LargeDatabaseManager instance
            source_db_path: Optional path to the source database for shard creation
        """
        # Main database path
        self.db_path = Path(db_path) if db_path else None
        self.source_db_path = Path(source_db_path) if source_db_path else self.db_path

        # Shards directory
        self.shards_dir = (
            Path(shards_dir)
            if shards_dir
            else Path(os.path.expanduser("~/.imessage_insights/shards"))
        )

        # Create shards directory if it doesn't exist
        os.makedirs(self.shards_dir, exist_ok=True)

        # List of shard configurations
        self.shards: List[ShardConfig] = []

        # Main database instance (used as fallback)
        self.main_db: Optional[AsyncMessagesDB] = None

        # External shard manager if provided
        self.shard_manager = shard_manager
        
        # Whether we're using shards
        self.using_shards = shard_manager is not None

        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        # Cache for optimizing shard selection
        self.shard_cache = {}
        self._shard_cache_lock = asyncio.Lock()
        self._max_shard_cache_size = 100

        # Whether the manager is initialized
        self.initialized = False

        # Performance configuration
        self.minimal_mode = minimal_mode

        # Cache
        
    async def should_use_sharding(self) -> bool:
        """Determine if database sharding should be used based on database size.
        
        Returns:
            bool: True if sharding should be used, False otherwise
        """
        # If we're already using shards, return True
        if self.using_shards and self.shard_manager and self.shard_manager.shards:
            return True
            
        # If no database path is provided, can't determine
        if not self.db_path or not self.db_path.exists():
            return False
            
        # Check database size
        try:
            db_size = self.db_path.stat().st_size
            db_size_gb = db_size / (1024 * 1024 * 1024)
            
            # If database is larger than 10GB, recommend sharding
            if db_size_gb >= 10:
                return True
        except Exception as e:
            logger.error(f"Error checking database size: {e}")
            
        return False
        
    async def get_shards_info(self) -> Dict[str, Any]:
        """Get information about the current shards.
        
        Returns:
            Dict with information about shards
        """
        if not self.initialized:
            raise DatabaseError("Database not initialized. Call initialize() first.")
            
        result = {
            "using_shards": self.using_shards,
            "shards_dir": str(self.shards_dir),
            "shard_count": 0,
            "total_size_mb": 0,
            "total_size_gb": 0,
            "shards": []
        }
        
        # If not using shards, return basic info
        if not self.using_shards or not self.shard_manager:
            return result
            
        # Get information about each shard
        shards_info = []
        total_size_bytes = 0
        
        for shard in self.shards:
            shard_info = {
                "shard_id": shard.shard_id,
                "path": str(shard.shard_path),
                "start_date": shard.start_date.isoformat() if shard.start_date else None,
                "end_date": shard.end_date.isoformat() if shard.end_date else None,
                "description": shard.description,
                "message_count": shard.message_count,
                "size_mb": shard.size_mb
            }
            
            # Update size information
            if shard.shard_path.exists():
                size_bytes = shard.shard_path.stat().st_size
                shard_info["size_mb"] = size_bytes / (1024 * 1024)
                total_size_bytes += size_bytes
                
            shards_info.append(shard_info)
            
        # Calculate total size
        total_size_mb = total_size_bytes / (1024 * 1024)
        total_size_gb = total_size_bytes / (1024 * 1024 * 1024)
        
        # Update result
        result["shard_count"] = len(self.shards)
        result["shards"] = shards_info
        result["total_size_mb"] = total_size_mb
        result["total_size_gb"] = total_size_gb
        result["shard_size_months"] = 6  # Default shard size
        
        return result
    
    def _determine_date_range(self, start_date: Optional[str], end_date: Optional[str]) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Convert string dates to datetime objects for shard selection.
        
        Args:
            start_date: Start date string in ISO format
            end_date: End date string in ISO format
            
        Returns:
            Tuple of start and end datetime objects
        """
        start_dt = None
        end_dt = None
        
        # Convert start date
        if start_date:
            try:
                start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            except ValueError:
                logger.warning(f"Invalid start date format: {start_date}")
                
        # Convert end date
        if end_date:
            try:
                end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            except ValueError:
                logger.warning(f"Invalid end date format: {end_date}")
                
        return start_dt, end_dt

    async def initialize(self):
        """Initialize the sharded database manager.

        This loads shard configurations and initializes connections.
        """
        if self.initialized:
            return

        async with self._lock:
            if self.initialized:
                return


            # Load shard configurations
            try:
                await self._load_shard_configurations()
            except Exception as e:
                logger.error(f"Error loading shard configurations: {e}")
                logger.error(traceback.format_exc())

                # If no shards were loaded and we have a main database path,
                # initialize it as a fallback
                if not self.shards and self.db_path:
                    logger.warning("No shards found. Using main database as fallback.")
                    self.main_db = AsyncMessagesDB(
                        db_path=self.db_path, minimal_mode=self.minimal_mode
                    )
                    await self.main_db.initialize()
                else:
                    raise DatabaseError(f"Failed to initialize sharded database: {e}")

            self.initialized = True
            logger.info(
                f"Sharded database manager initialized with {len(self.shards)} shards"
            )

    async def close(self):
        """Close all database connections."""
        if not self.initialized:
            return

        async with self._lock:
            # Close main database connection
            if self.main_db:
                await self.main_db.close()

            # Close shard connections
            for shard in self.shards:
                if shard.db_instance:
                    await shard.db_instance.close()

            self.initialized = False
            logger.info("All database connections closed")

    async def _load_shard_configurations(self):
        """Load shard configurations from the shards directory."""
        # Check if config file exists
        config_path = self.shards_dir / "shards_config.json"

        if config_path.exists():
            # Load from config file
            try:
                with open(config_path, "r") as f:
                    config_data = json.load(f)

                # Create shard configs
                self.shards = [
                    ShardConfig.from_dict(shard_data)
                    for shard_data in config_data.get("shards", [])
                ]

                # Filter out shards that don't exist
                self.shards = [
                    shard for shard in self.shards if os.path.exists(shard.shard_path)
                ]

                logger.info(
                    f"Loaded {len(self.shards)} shard configurations from {config_path}"
                )

            except Exception as e:
                logger.error(
                    f"Error loading shard configurations from {config_path}: {e}"
                )
                logger.error(traceback.format_exc())
                self.shards = []
        else:
            # Scan directory for shard files
            shard_files = list(self.shards_dir.glob("shard_*.db"))

            if not shard_files:
                logger.warning(f"No shard files found in {self.shards_dir}")
                return

            # Create shard configs from files
            for shard_file in shard_files:
                # Extract shard ID from filename
                match = re.search(r"shard_(\w+)\.db", shard_file.name)
                if not match:
                    continue

                shard_id = match.group(1)

                # Create shard config
                shard_config = ShardConfig(
                    shard_id=shard_id,
                    shard_path=shard_file,
                    description=f"Auto-detected shard {shard_id}",
                    size_mb=os.path.getsize(shard_file) / (1024 * 1024),
                )

                self.shards.append(shard_config)

            logger.info(f"Detected {len(self.shards)} shards in {self.shards_dir}")

            # Save the config
            await self._save_shard_configurations()

    async def _save_shard_configurations(self):
        """Save shard configurations to the config file."""
        try:
            config_path = self.shards_dir / "shards_config.json"

            config_data = {
                "shards": [shard.to_dict() for shard in self.shards],
                "updated_at": datetime.now().isoformat(),
            }

            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=2)

            logger.info(
                f"Saved {len(self.shards)} shard configurations to {config_path}"
            )

        except Exception as e:
            logger.error(f"Error saving shard configurations: {e}")
            logger.error(traceback.format_exc())

    async def _get_shard_for_date(
        self, date: Optional[datetime]
    ) -> Optional[ShardConfig]:
        """Get the shard that contains the specified date.

        Args:
            date: Date to find shard for

        Returns:
            ShardConfig for the shard containing the date, or None if no match
        """
        if not date:
            return None

        for shard in self.shards:
            if (shard.start_date is None or date >= shard.start_date) and (
                shard.end_date is None or date <= shard.end_date
            ):
                return shard

        return None

    async def _get_shards_for_date_range(
        self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
    ) -> List[ShardConfig]:
        """Get all shards that overlap with the specified date range.

        Args:
            start_date: Start date of the range
            end_date: End date of the range

        Returns:
            List of ShardConfig objects that overlap with the date range
        """
        if not start_date and not end_date:
            # No date range specified, return all shards
            return self.shards.copy()

        matching_shards = []

        for shard in self.shards:
            # Check if shard overlaps with the specified date range
            if start_date and shard.end_date and start_date > shard.end_date:
                # Shard ends before the start date
                continue

            if end_date and shard.start_date and end_date < shard.start_date:
                # Shard starts after the end date
                continue

            matching_shards.append(shard)

        return matching_shards

    async def _initialize_shard_db(self, shard: ShardConfig) -> AsyncMessagesDB:
        """Initialize a database connection for a shard.

        Args:
            shard: ShardConfig for the shard to initialize

        Returns:
            AsyncMessagesDB instance for the shard
        """
        if shard.db_instance and shard.db_instance.initialized:
            return shard.db_instance

        # Create database instance
        db = AsyncMessagesDB(
            db_path=str(shard.shard_path), minimal_mode=self.minimal_mode
        )
        await db.initialize()

        # Store in shard config
        shard.db_instance = db

        return db

    async def get_contacts(self, limit=100, offset=0, minimal=None) -> Dict[str, Any]:
        """Get a list of all contacts from all shards.

        Args:
            limit: Maximum number of contacts to return
            offset: Number of contacts to skip
            minimal: Whether to return minimal data (overrides instance setting)

        Returns:
            Dictionary with contacts list and metadata
        """
        if not self.initialized:
            raise DatabaseError("Database not initialized. Call initialize() first.")

        # If no shards are configured, use the main database
        if not self.shards and self.main_db:
            return await self.main_db.get_contacts(
                limit=limit, offset=offset, minimal=minimal
            )

        # Use minimal mode if specified or set at instance level
        minimal = self.minimal_mode if minimal is None else minimal

        # Get contacts from all shards
        all_contacts = {}
        total_count = 0

        # Process shards in parallel
        tasks = []
        for shard in self.shards:
            # Initialize shard database
            db = await self._initialize_shard_db(shard)

            # Create task to get contacts
            task = asyncio.create_task(
                db.get_contacts(limit=1000, offset=0, minimal=minimal)
            )
            tasks.append((shard, task))

        # Wait for all tasks to complete
        for shard, task in tasks:
            try:
                result = await task

                # Process contacts from this shard
                contacts = result.get("contacts", [])

                for contact in contacts:
                    phone_number = contact["phone_number"]

                    if phone_number not in all_contacts:
                        # New contact
                        all_contacts[phone_number] = contact
                    else:
                        # Existing contact, merge data
                        existing = all_contacts[phone_number]

                        # Update message count if not in minimal mode
                        if not minimal and "message_count" in contact:
                            existing["message_count"] = existing.get(
                                "message_count", 0
                            ) + contact.get("message_count", 0)

                        # Update last message date if newer
                        if (
                            "last_message_date" in contact
                            and contact["last_message_date"]
                        ):
                            if (
                                not existing.get("last_message_date")
                                or contact["last_message_date"]
                                > existing["last_message_date"]
                            ):
                                existing["last_message_date"] = contact[
                                    "last_message_date"
                                ]

                # Update total count
                total_count += result.get("total", 0)

            except Exception as e:
                logger.error(f"Error getting contacts from shard {shard.shard_id}: {e}")
                logger.error(traceback.format_exc())

        # Convert contacts to list
        contacts_list = list(all_contacts.values())

        # Sort by name
        contacts_list.sort(key=lambda c: c.get("display_name", "").lower())

        # Apply pagination
        paginated_contacts = contacts_list[offset : offset + limit]

        # Calculate pagination
        has_more = len(contacts_list) > (offset + limit)
        total_pages = (len(contacts_list) + limit - 1) // limit if limit > 0 else 1
        current_page = (offset // limit) + 1 if limit > 0 else 1

        return {
            "contacts": paginated_contacts,
            "total": len(contacts_list),
            "pagination": {
                "limit": limit,
                "offset": offset,
                "has_more": has_more,
                "total_pages": total_pages,
                "current_page": current_page,
            },
        }

    async def get_group_chats(self, limit=20, offset=0) -> Dict[str, Any]:
        """Get a list of all group chats from all shards.

        Args:
            limit: Maximum number of group chats to return
            offset: Number of group chats to skip

        Returns:
            Dictionary with group chats list and metadata
        """
        if not self.initialized:
            raise DatabaseError("Database not initialized. Call initialize() first.")

        # If no shards are configured, use the main database
        if not self.shards and self.main_db:
            return await self.main_db.get_group_chats(limit=limit, offset=offset)

        # Get group chats from all shards
        all_chats = {}

        # Process shards in parallel
        tasks = []
        for shard in self.shards:
            # Initialize shard database
            db = await self._initialize_shard_db(shard)

            # Create task to get group chats
            task = asyncio.create_task(db.get_group_chats(limit=1000, offset=0))
            tasks.append((shard, task))

        # Wait for all tasks to complete
        for shard, task in tasks:
            try:
                result = await task

                # Process chats from this shard
                chats = result.get("chats", [])

                for chat in chats:
                    chat_id = chat["chat_id"]

                    if chat_id not in all_chats:
                        # New chat
                        all_chats[chat_id] = chat
                    else:
                        # Existing chat, merge data
                        existing = all_chats[chat_id]

                        # Update message count
                        if "message_count" in chat:
                            existing["message_count"] = existing.get(
                                "message_count", 0
                            ) + chat.get("message_count", 0)

                        # Update last message date if newer
                        if "last_message_date" in chat and chat["last_message_date"]:
                            if (
                                not existing.get("last_message_date")
                                or chat["last_message_date"]
                                > existing["last_message_date"]
                            ):
                                existing["last_message_date"] = chat[
                                    "last_message_date"
                                ]

            except Exception as e:
                logger.error(
                    f"Error getting group chats from shard {shard.shard_id}: {e}"
                )
                logger.error(traceback.format_exc())

        # Convert chats to list
        chats_list = list(all_chats.values())

        # Sort by last message date (recent first)
        chats_list.sort(key=lambda c: c.get("last_message_date", ""), reverse=True)

        # Apply pagination
        paginated_chats = chats_list[offset : offset + limit]

        # Calculate pagination
        has_more = len(chats_list) > (offset + limit)
        total_pages = (len(chats_list) + limit - 1) // limit if limit > 0 else 1
        current_page = (offset // limit) + 1 if limit > 0 else 1

        return {
            "chats": paginated_chats,
            "total": len(chats_list),
            "has_more": has_more,
            "pagination": {
                "limit": limit,
                "offset": offset,
                "total_pages": total_pages,
                "current_page": current_page,
            },
        }

    async def validate_schema_version(self) -> bool:
        """Check if the database schema is compatible with this application.

        Returns:
            True if schema is compatible, False otherwise
        """
        if not self.initialized:
            raise DatabaseError("Database not initialized. Call initialize() first.")

        # If no shards are configured, use the main database
        if not self.shards and self.main_db:
            return await self.main_db.validate_schema_version()

        # Check the first shard (all shards should have the same schema)
        if not self.shards:
            logger.error("No shards configured for validation")
            return False

        try:
            # Initialize the first shard
            shard = self.shards[0]
            db = await self._initialize_shard_db(shard)

            # Validate schema
            return await db.validate_schema_version()

        except Exception as e:
            logger.error(f"Error validating schema: {e}")
            logger.error(traceback.format_exc())
            return False

    async def get_chat_by_id(self, chat_id: Union[str, int]) -> Dict[str, Any]:
        """Get information about a specific chat by its ID.

        Args:
            chat_id: The chat ID

        Returns:
            Dictionary with chat information
        """
        if not self.initialized:
            raise DatabaseError("Database not initialized. Call initialize() first.")

        # If no shards are configured, use the main database
        if not self.shards and self.main_db:
            return await self.main_db.get_chat_by_id(chat_id)

        # Try to find the chat in each shard
        for shard in self.shards:
            try:
                # Initialize shard database
                db = await self._initialize_shard_db(shard)

                # Try to get chat
                chat = await db.get_chat_by_id(chat_id)

                if chat:
                    # Found the chat in this shard
                    return chat

            except Exception as e:
                logger.error(
                    f"Error getting chat by ID {chat_id} from shard {shard.shard_id}: {e}"
                )
                logger.error(traceback.format_exc())

        # Chat not found in any shard
        return {}

    async def optimize_database(self) -> Dict[str, Any]:
        """Perform database optimizations.

        Returns:
            Dictionary with optimization results
        """
        logger.warning("Database optimization requested for sharded database.")
        logger.warning("Sharded databases are already optimized by design.")

        return {
            "success": True,
            "message": "Sharded databases are already optimized by design",
            "shards": len(self.shards),
        }
        
    async def should_use_sharding(self) -> bool:
        """Determine if database sharding should be used based on database size.
        
        Returns:
            True if sharding is recommended, False otherwise
        """
        # If we're already using shards, return True
        if self.using_shards and self.shard_manager and self.shard_manager.shards:
            return True
            
        # If no source database, can't determine
        if not self.source_db_path or not self.source_db_path.exists():
            return False
            
        try:
            # Get database size
            db_size_bytes = os.path.getsize(self.source_db_path)
            db_size_gb = db_size_bytes / (1024 ** 3)
            
            # If database is larger than 10GB, recommend sharding
            if db_size_gb >= 10:
                logger.info(f"Database size is {db_size_gb:.2f} GB, sharding is recommended")
                return True
                
            # If database is between 5-10GB, recommend sharding for better performance
            if db_size_gb >= 5:
                logger.info(f"Database size is {db_size_gb:.2f} GB, sharding is recommended for better performance")
                return True
                
            # Database is small enough to use without sharding
            logger.info(f"Database size is {db_size_gb:.2f} GB, sharding is not necessary")
            return False
            
        except Exception as e:
            logger.error(f"Error determining if sharding should be used: {e}")
            return False
            
    async def get_shards_info(self) -> Dict[str, Any]:
        """Get information about the database shards.
        
        Returns:
            Dictionary with shard information
        """
        if not self.initialized:
            raise DatabaseError("Database not initialized. Call initialize() first.")
            
        # If using external shard manager
        if self.shard_manager:
            try:
                # Get shard status from the manager
                status = await self.shard_manager.get_shard_status()
                
                return {
                    "using_shards": True,
                    "shard_count": len(self.shard_manager.shards),
                    "shards": status["shards"],
                    "source_db": str(self.shard_manager.source_db_path),
                    "shards_dir": str(self.shard_manager.shards_dir),
                }
            except Exception as e:
                logger.error(f"Error getting shard info from manager: {e}")
                return {
                    "using_shards": True,
                    "shard_count": len(self.shard_manager.shards),
                    "error": str(e)
                }
        
        # If using internal shard configs
        if self.shards:
            return {
                "using_shards": True,
                "shard_count": len(self.shards),
                "shards": [
                    {
                        "id": shard.shard_id,
                        "path": str(shard.shard_path),
                        "start_date": shard.start_date.isoformat() if shard.start_date else None,
                        "end_date": shard.end_date.isoformat() if shard.end_date else None,
                        "size_mb": shard.size_mb,
                        "message_count": shard.message_count,
                    }
                    for shard in self.shards
                ],
                "source_db": str(self.source_db_path) if self.source_db_path else None,
                "shards_dir": str(self.shards_dir),
            }
            
        # Not using shards
        return {
            "using_shards": False,
            "shard_count": 0,
            "shards": [],
            "source_db": str(self.db_path) if self.db_path else None,
        }
        
    async def _determine_date_range(
        self, chat_id: Optional[Union[str, int]] = None, 
        contact_id: Optional[Union[str, int]] = None
    ) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Determine the date range for messages based on parameters.
        
        Args:
            chat_id: Optional chat ID to get date range for
            contact_id: Optional contact ID to get date range for
            
        Returns:
            Tuple of (start_date, end_date) or (None, None) if unknown
        """
        # Check cache first
        cache_key = None
        if chat_id:
            cache_key = f"chat_date_range:{chat_id}"
        elif contact_id:
            cache_key = f"contact_date_range:{contact_id}"
            
        if cache_key:
            async with self._shard_cache_lock:
                if cache_key in self.shard_cache:
                    return self.shard_cache[cache_key]
        
        # No parameters, cannot determine date range
        if not chat_id and not contact_id:
            return None, None
            
        try:
            # Try to find date range from a single shard to avoid loading all shards
            for shard in self.shards:
                try:
                    # Initialize shard database
                    db = await self._initialize_shard_db(shard)
                    
                    if chat_id:
                        # Get messages for chat with limit 1 to find earliest
                        earliest = await db.get_messages_from_chat(
                            chat_id, limit=1, offset=0, order_by="date ASC"
                        )
                        
                        # Get messages for chat with limit 1 to find latest
                        latest = await db.get_messages_from_chat(
                            chat_id, limit=1, offset=0, order_by="date DESC"
                        )
                    elif contact_id:
                        # Get messages for contact with limit 1 to find earliest
                        earliest = await db.get_messages_from_contact(
                            contact_id, limit=1, offset=0, order_by="date ASC"
                        )
                        
                        # Get messages for contact with limit 1 to find latest
                        latest = await db.get_messages_from_contact(
                            contact_id, limit=1, offset=0, order_by="date DESC"
                        )
                    
                    # Check if we got results
                    if (earliest and earliest.get("messages") and 
                        latest and latest.get("messages")):
                        # Extract dates
                        start_date = earliest["messages"][0].get("date_obj")
                        end_date = latest["messages"][0].get("date_obj")
                        
                        # Cache the result
                        if cache_key:
                            async with self._shard_cache_lock:
                                self.shard_cache[cache_key] = (start_date, end_date)
                                
                                # Manage cache size
                                if len(self.shard_cache) > self._max_shard_cache_size:
                                    # Remove oldest entries (first items in dict)
                                    keys_to_remove = list(self.shard_cache.keys())[
                                        :len(self.shard_cache) - self._max_shard_cache_size
                                    ]
                                    for key in keys_to_remove:
                                        del self.shard_cache[key]
                        
                        return start_date, end_date
                
                except Exception as e:
                    logger.error(f"Error determining date range from shard {shard.shard_id}: {e}")
                    continue
            
            # If we reach here, need to search across all shards
            # But this would be expensive, so we'll return None for now
            return None, None
                
        except Exception as e:
            logger.error(f"Error determining date range: {e}")
            return None, None
            
    async def cleanup(self):
        """Close all database connections and clean up resources."""
        # Close database connections
        await self.close()
