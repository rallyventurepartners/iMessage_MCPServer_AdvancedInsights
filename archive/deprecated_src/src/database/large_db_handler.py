#!/usr/bin/env python3
"""
Database sharding implementation for extremely large iMessage databases.

This module provides classes to manage, create, and access sharded databases,
which divides a large database into smaller, more manageable chunks based on time periods.
"""

import asyncio
import json
import logging
import os
import re
import shutil
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import aiosqlite

logger = logging.getLogger(__name__)


class DatabaseShard:
    """
    Represents a time-bounded portion of the message database.
    
    A shard is a separate SQLite database file containing messages and related data 
    from a specific time period. This class provides methods to access and manipulate the shard.
    """

    def __init__(
        self,
        shard_path: Path,
        start_date: datetime,
        end_date: datetime,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a database shard.
        
        Args:
            shard_path: Path to the shard database file
            start_date: Start date of messages in this shard (inclusive)
            end_date: End date of messages in this shard (exclusive)
            metadata: Optional metadata for the shard
        """
        self.shard_path = shard_path
        self.start_date = start_date
        self.end_date = end_date
        self.metadata = metadata or {}
        self.initialized = False
        self.size_bytes = 0
        self.message_count = 0
        self.attachment_count = 0
        self.tables = []
        self.indexes = []
        self._connection_pool = []
        self._max_connections = min(os.cpu_count() or 1, 4)  # Max 4 connections per shard
        self._connection_semaphore = asyncio.Semaphore(self._max_connections)

    def __str__(self) -> str:
        """Return string representation of the shard."""
        return f"Shard {self.shard_path.name} ({self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')})"

    def contains_date(self, date: datetime) -> bool:
        """
        Check if the shard contains the given date.
        
        Args:
            date: The date to check
            
        Returns:
            True if the date is within the shard's time range
        """
        return self.start_date <= date < self.end_date

    def overlaps_range(self, start_date: Optional[datetime], end_date: Optional[datetime]) -> bool:
        """
        Check if the shard overlaps with the given date range.
        
        Args:
            start_date: Start date of the range (or None for no lower bound)
            end_date: End date of the range (or None for no upper bound)
            
        Returns:
            True if the shard overlaps with the given date range
        """
        # If no dates provided, consider it a match
        if start_date is None and end_date is None:
            return True

        # If only start_date provided, check if shard end is after the start date
        if start_date is not None and end_date is None:
            return self.end_date > start_date

        # If only end_date provided, check if shard start is before the end date
        if start_date is None and end_date is not None:
            return self.start_date < end_date

        # If both dates provided, check for overlap
        # (start_date < shard.end_date) and (end_date > shard.start_date)
        if start_date is not None and end_date is not None:
            return start_date < self.end_date and end_date > self.start_date

        return False

    async def initialize(self) -> bool:
        """
        Initialize the shard: gather metadata, run diagnostics.
        
        Returns:
            True if initialization succeeded
        """
        if not self.shard_path.exists():
            logger.error(f"Shard file not found: {self.shard_path}")
            return False

        try:
            # Get file size
            self.size_bytes = os.path.getsize(self.shard_path)

            # Connect to the database
            async with aiosqlite.connect(self.shard_path) as conn:
                # Get table list
                async with conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                ) as cursor:
                    self.tables = [row[0] for row in await cursor.fetchall()]

                # Get index list
                async with conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
                ) as cursor:
                    self.indexes = [row[0] for row in await cursor.fetchall()]

                # Count messages
                if "message" in self.tables:
                    async with conn.execute("SELECT COUNT(*) FROM message") as cursor:
                        self.message_count = (await cursor.fetchone())[0]

                # Count attachments if the table exists
                if "attachment" in self.tables:
                    async with conn.execute("SELECT COUNT(*) FROM attachment") as cursor:
                        self.attachment_count = (await cursor.fetchone())[0]

            self.initialized = True
            logger.info(
                f"Initialized shard: {self.shard_path.name} with {self.message_count} messages "
                f"({(self.size_bytes / (1024 * 1024)):.2f} MB)"
            )
            return True

        except (sqlite3.Error, aiosqlite.Error) as e:
            logger.error(f"Error initializing shard {self.shard_path.name}: {e}")
            return False

    async def get_metadata(self) -> Dict[str, Any]:
        """
        Get detailed metadata about the shard.
        
        Returns:
            Dictionary containing shard metadata
        """
        # Make sure the shard is initialized
        if not self.initialized:
            await self.initialize()

        # Gather statistics
        return {
            "path": str(self.shard_path),
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "date_range": f"{self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}",
            "message_count": self.message_count,
            "attachment_count": self.attachment_count,
            "size_bytes": self.size_bytes,
            "size_mb": round(self.size_bytes / (1024 * 1024), 2),
            "tables": self.tables,
            "table_count": len(self.tables),
            "indexes": self.indexes,
            "index_count": len(self.indexes),
            "name": self.shard_path.name,
            "custom_metadata": self.metadata,
        }

    async def get_connection(self) -> aiosqlite.Connection:
        """
        Get a connection to the shard database, using a connection pool for efficiency.
        
        Returns:
            A database connection
        """
        async with self._connection_semaphore:
            # Create a new connection
            conn = await aiosqlite.connect(self.shard_path)
            
            # Apply pragmas for optimal performance
            await conn.execute("PRAGMA journal_mode = WAL")
            await conn.execute("PRAGMA synchronous = NORMAL")
            await conn.execute("PRAGMA cache_size = 10000")
            await conn.execute("PRAGMA temp_store = MEMORY")
            await conn.execute("PRAGMA mmap_size = 30000000000")  # ~30GB
            await conn.execute("PRAGMA foreign_keys = OFF")  # Turn off for better performance
            
            # Set output to be a dictionary
            conn.row_factory = aiosqlite.Row
            
            return conn

    async def release_connection(self, conn: aiosqlite.Connection) -> None:
        """
        Release a database connection when done using it.
        
        Args:
            conn: The connection to release
        """
        await conn.close()
        self._connection_semaphore.release()

    async def execute(
        self,
        query: str,
        params: Optional[Tuple[Any, ...]] = None,
        fetchall: bool = True,
        timeout: Optional[float] = 30.0,
    ) -> List[Dict[str, Any]]:
        """
        Execute a SQL query on the shard and return the results.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            fetchall: Whether to fetch all results
            timeout: Query timeout in seconds
            
        Returns:
            List of results as dictionaries
        """
        if not self.initialized:
            await self.initialize()

        try:
            # Get a connection from the pool
            conn = await self.get_connection()
            try:
                # Execute the query with a timeout
                cursor = await asyncio.wait_for(
                    conn.execute(query, params or ()), timeout
                )
                
                # Fetch the results
                if fetchall:
                    rows = await cursor.fetchall()
                    # Convert rows to dictionaries
                    result = [dict(row) for row in rows]
                else:
                    row = await cursor.fetchone()
                    result = [dict(row)] if row else []
                
                # Close the cursor
                await cursor.close()
                return result
            finally:
                # Release the connection
                await self.release_connection(conn)
        
        except asyncio.TimeoutError:
            logger.error(f"Query timeout after {timeout}s in shard {self.shard_path.name}: {query}")
            return []
        except Exception as e:
            logger.error(f"Error executing query in shard {self.shard_path.name}: {e}")
            logger.error(f"Query: {query}, Params: {params}")
            return []

    async def cleanup(self) -> None:
        """Clean up resources used by the shard."""
        # No cleanup needed at this time
        pass


class LargeDatabaseManager:
    """
    Manager class for creating and accessing sharded databases.
    
    This class handles:
    1. Creating time-based database shards
    2. Selecting the appropriate shards for queries based on date ranges
    3. Executing queries across multiple shards and combining results
    4. Managing shard metadata and configuration
    """

    def __init__(
        self,
        source_db_path: Union[str, Path],
        shards_dir: Union[str, Path],
        shard_size_months: int = 6,
        max_shards: int = 0,
    ):
        """
        Initialize the large database manager.
        
        Args:
            source_db_path: Path to the source iMessage database
            shards_dir: Directory to store shard databases
            shard_size_months: Size of each shard in months
            max_shards: Maximum number of shards to create (0 for unlimited)
        """
        self.source_db_path = Path(source_db_path)
        self.shards_dir = Path(shards_dir)
        self.shard_size_months = shard_size_months
        self.max_shards = max_shards
        self.shards: List[DatabaseShard] = []
        self.metadata_path = self.shards_dir / "shard_metadata.json"
        self.initialized = False
        
        # For limiting concurrent database operations
        self._connection_semaphore = asyncio.Semaphore(min(os.cpu_count() or 1, 4))

    async def initialize(self) -> bool:
        """
        Initialize the database manager.
        
        Returns:
            True if initialization succeeded
        """
        # Create shards directory if it doesn't exist
        os.makedirs(self.shards_dir, exist_ok=True)
        
        # Check if source database exists
        if not self.source_db_path.exists():
            logger.error(f"Source database not found: {self.source_db_path}")
            return False
        
        # Load existing shards if available
        if await self._load_shards():
            self.initialized = True
            logger.info(f"Loaded {len(self.shards)} existing shards")
            return True
        
        # No existing shards, we'll need to create them
        self.initialized = True
        logger.info("No existing shards found, use create_shards() to create them")
        return True

    async def _load_shards(self) -> bool:
        """
        Load existing shards from metadata file.
        
        Returns:
            True if shards were loaded successfully
        """
        if not self.metadata_path.exists():
            return False
        
        try:
            # Read metadata file
            with open(self.metadata_path, "r") as f:
                metadata = json.load(f)
            
            # Check file format
            if not metadata or "shards" not in metadata:
                logger.warning(f"Invalid metadata file: {self.metadata_path}")
                return False
            
            # Create shard objects
            for shard_data in metadata["shards"]:
                shard_path = Path(shard_data["path"])
                start_date = datetime.fromisoformat(shard_data["start_date"])
                end_date = datetime.fromisoformat(shard_data["end_date"])
                
                # Create the shard object
                shard = DatabaseShard(
                    shard_path=shard_path,
                    start_date=start_date,
                    end_date=end_date,
                    metadata=shard_data.get("custom_metadata", {}),
                )
                
                # Only add if the file exists
                if shard_path.exists():
                    # Initialize the shard
                    await shard.initialize()
                    self.shards.append(shard)
                else:
                    logger.warning(f"Shard file not found: {shard_path}")
            
            # Sort shards by start date
            self.shards.sort(key=lambda s: s.start_date)
            
            return len(self.shards) > 0
        
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Error loading shards metadata: {e}")
            return False

    async def _save_metadata(self) -> bool:
        """
        Save shards metadata to file.
        
        Returns:
            True if metadata was saved successfully
        """
        try:
            # Get metadata for all shards
            shards_data = []
            for shard in self.shards:
                metadata = await shard.get_metadata()
                shards_data.append(metadata)
            
            # Create the metadata object
            metadata = {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "source_db": str(self.source_db_path),
                "shard_size_months": self.shard_size_months,
                "shard_count": len(self.shards),
                "shards": shards_data,
            }
            
            # Write to file
            with open(self.metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            return True
        
        except Exception as e:
            logger.error(f"Error saving shards metadata: {e}")
            return False

    async def create_shards(self) -> bool:
        """
        Create time-based shards from the source database.
        
        Returns:
            True if shards were created successfully
        """
        # Make sure we're initialized
        if not self.initialized:
            await self.initialize()
        
        try:
            # Get date range of messages in the source database
            start_date, end_date = await self._get_date_range_from_source_db()
            
            if start_date is None or end_date is None:
                logger.error("Could not determine date range from source database")
                return False
            
            # Round to first of the month
            start_date = datetime(start_date.year, start_date.month, 1)
            # Round to first of next month
            end_date = datetime(end_date.year, end_date.month, 1) + timedelta(days=32)
            end_date = datetime(end_date.year, end_date.month, 1)
            
            logger.info(f"Creating shards for date range: {start_date} to {end_date}")
            
            # Create shards
            current_date = start_date
            shard_number = 1
            
            while current_date < end_date:
                # Calculate end date for this shard
                shard_end_date = current_date
                for _ in range(self.shard_size_months):
                    # Advance to first of next month
                    month = shard_end_date.month + 1
                    year = shard_end_date.year
                    if month > 12:
                        month = 1
                        year += 1
                    shard_end_date = datetime(year, month, 1)
                
                # Set the last shard end date to the overall end date
                if shard_end_date > end_date:
                    shard_end_date = end_date
                
                # Skip if no messages in this date range
                if not await self._has_messages_in_range(current_date, shard_end_date):
                    current_date = shard_end_date
                    continue
                
                # Create the shard file path
                shard_name = f"shard_{current_date.strftime('%Y%m%d')}_{shard_end_date.strftime('%Y%m%d')}.db"
                shard_path = self.shards_dir / shard_name
                
                # Check if shard already exists
                if shard_path.exists():
                    logger.info(f"Shard already exists: {shard_path}")
                    # Just add it to our list
                    shard = DatabaseShard(
                        shard_path=shard_path,
                        start_date=current_date,
                        end_date=shard_end_date,
                    )
                    await shard.initialize()
                    self.shards.append(shard)
                else:
                    # Create the shard
                    success = await self._create_shard(
                        shard_path, current_date, shard_end_date
                    )
                    if success:
                        logger.info(f"Created shard: {shard_path}")
                        # Add to our list
                        shard = DatabaseShard(
                            shard_path=shard_path,
                            start_date=current_date,
                            end_date=shard_end_date,
                        )
                        await shard.initialize()
                        self.shards.append(shard)
                    else:
                        logger.error(f"Failed to create shard: {shard_path}")
                
                # Move to next shard
                current_date = shard_end_date
                shard_number += 1
                
                # Stop if we've reached max shards
                if self.max_shards > 0 and len(self.shards) >= self.max_shards:
                    logger.info(f"Reached maximum number of shards: {self.max_shards}")
                    break
            
            # Sort shards by start date
            self.shards.sort(key=lambda s: s.start_date)
            
            # Save metadata
            await self._save_metadata()
            
            logger.info(f"Created {len(self.shards)} shards")
            return True
        
        except Exception as e:
            logger.error(f"Error creating shards: {e}")
            logger.error(traceback.format_exc())
            return False

    async def _get_date_range_from_source_db(self) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Get the date range of messages in the source database.
        
        Returns:
            Tuple of (start_date, end_date)
        """
        try:
            # Apple epoch (2001-01-01)
            apple_epoch = datetime(2001, 1, 1)
            
            async with aiosqlite.connect(self.source_db_path) as conn:
                # Get min and max dates
                async with conn.execute("SELECT MIN(date), MAX(date) FROM message") as cursor:
                    row = await cursor.fetchone()
                    if row and row[0] and row[1]:
                        # Convert from Apple time (nanoseconds since 2001-01-01) to datetime
                        min_date = apple_epoch + timedelta(seconds=row[0] / 1e9)
                        max_date = apple_epoch + timedelta(seconds=row[1] / 1e9)
                        return min_date, max_date
                    else:
                        logger.warning("No messages found in database")
                        return None, None
        
        except Exception as e:
            logger.error(f"Error getting date range from source database: {e}")
            return None, None

    async def _has_messages_in_range(
        self, start_date: datetime, end_date: datetime
    ) -> bool:
        """
        Check if there are messages in the given date range.
        
        Args:
            start_date: Start date of the range
            end_date: End date of the range
            
        Returns:
            True if there are messages in the range
        """
        try:
            # Convert to Apple time (nanoseconds since 2001-01-01)
            apple_epoch = datetime(2001, 1, 1)
            start_time = int((start_date - apple_epoch).total_seconds() * 1e9)
            end_time = int((end_date - apple_epoch).total_seconds() * 1e9)
            
            async with aiosqlite.connect(self.source_db_path) as conn:
                # Count messages in the date range
                async with conn.execute(
                    "SELECT COUNT(*) FROM message WHERE date >= ? AND date < ?",
                    (start_time, end_time),
                ) as cursor:
                    count = (await cursor.fetchone())[0]
                    return count > 0
        
        except Exception as e:
            logger.error(f"Error checking for messages in range: {e}")
            return False

    async def _create_shard(
        self, shard_path: Path, start_date: datetime, end_date: datetime
    ) -> bool:
        """
        Create a shard for the specified date range.
        
        Args:
            shard_path: Path where the shard will be saved
            start_date: Start date for messages in this shard
            end_date: End date for messages in this shard
            
        Returns:
            True if shard was created successfully
        """
        try:
            # Convert to Apple time (nanoseconds since 2001-01-01)
            apple_epoch = datetime(2001, 1, 1)
            start_time = int((start_date - apple_epoch).total_seconds() * 1e9)
            end_time = int((end_date - apple_epoch).total_seconds() * 1e9)
            
            # Create the database structure with the necessary tables
            conn = sqlite3.connect(shard_path)
            cursor = conn.cursor()
            
            # Create tables with the same schema as the source database
            source_conn = sqlite3.connect(self.source_db_path)
            source_cursor = source_conn.cursor()
            
            # Get the table schemas we need
            tables_to_copy = [
                "message",
                "chat",
                "chat_message_join",
                "handle",
                "attachment",
                "message_attachment_join",
            ]
            
            # Create tables
            for table in tables_to_copy:
                try:
                    source_cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table}'")
                    create_sql = source_cursor.fetchone()
                    if create_sql and create_sql[0]:
                        cursor.execute(create_sql[0])
                except Exception as e:
                    logger.warning(f"Error creating table {table}: {e}")
            
            conn.commit()
            
            # Copy relevant messages for this date range
            source_cursor.execute(
                "SELECT * FROM message WHERE date >= ? AND date < ?", (start_time, end_time)
            )
            message_rows = source_cursor.fetchall()
            
            if not message_rows:
                logger.warning(f"No messages found for shard {shard_path.name}")
                # Clean up and return False
                conn.close()
                source_conn.close()
                return False
            
            # Get column names
            source_cursor.execute("PRAGMA table_info(message)")
            column_info = source_cursor.fetchall()
            column_names = [col[1] for col in column_info]
            
            # Insert messages
            placeholders = ", ".join(["?"] * len(column_names))
            insert_sql = f"INSERT INTO message ({', '.join(column_names)}) VALUES ({placeholders})"
            
            message_ids = []
            for row in message_rows:
                cursor.execute(insert_sql, row)
                message_ids.append(row[0])  # Assuming ROWID/message.ROWID is the first column
            
            conn.commit()
            
            # Copy related chat_message_join entries
            if message_ids:
                placeholders = ", ".join(["?"] * len(message_ids))
                source_cursor.execute(
                    f"SELECT * FROM chat_message_join WHERE message_id IN ({placeholders})",
                    message_ids,
                )
                join_rows = source_cursor.fetchall()
                
                # Get column names
                source_cursor.execute("PRAGMA table_info(chat_message_join)")
                column_info = source_cursor.fetchall()
                column_names = [col[1] for col in column_info]
                
                # Insert chat_message_join
                if join_rows:
                    placeholders = ", ".join(["?"] * len(column_names))
                    insert_sql = f"INSERT INTO chat_message_join ({', '.join(column_names)}) VALUES ({placeholders})"
                    
                    chat_ids = set()
                    for row in join_rows:
                        cursor.execute(insert_sql, row)
                        chat_ids.add(row[0])  # Assuming chat_id is the first column
                    
                    conn.commit()
                    
                    # Copy related chat entries
                    if chat_ids:
                        placeholders = ", ".join(["?"] * len(chat_ids))
                        source_cursor.execute(
                            f"SELECT * FROM chat WHERE ROWID IN ({placeholders})",
                            list(chat_ids),
                        )
                        chat_rows = source_cursor.fetchall()
                        
                        # Get column names
                        source_cursor.execute("PRAGMA table_info(chat)")
                        column_info = source_cursor.fetchall()
                        column_names = [col[1] for col in column_info]
                        
                        # Insert chats
                        if chat_rows:
                            placeholders = ", ".join(["?"] * len(column_names))
                            insert_sql = f"INSERT INTO chat ({', '.join(column_names)}) VALUES ({placeholders})"
                            
                            for row in chat_rows:
                                cursor.execute(insert_sql, row)
                            
                            conn.commit()
            
            # Copy handles
            source_cursor.execute(
                """
                SELECT DISTINCT h.* 
                FROM handle h
                JOIN message m ON h.ROWID = m.handle_id
                WHERE m.date >= ? AND m.date < ?
                """,
                (start_time, end_time),
            )
            handle_rows = source_cursor.fetchall()
            
            if handle_rows:
                # Get column names
                source_cursor.execute("PRAGMA table_info(handle)")
                column_info = source_cursor.fetchall()
                column_names = [col[1] for col in column_info]
                
                # Insert handles
                placeholders = ", ".join(["?"] * len(column_names))
                insert_sql = f"INSERT INTO handle ({', '.join(column_names)}) VALUES ({placeholders})"
                
                for row in handle_rows:
                    cursor.execute(insert_sql, row)
                
                conn.commit()
            
            # Copy attachments
            source_cursor.execute(
                """
                SELECT DISTINCT a.* 
                FROM attachment a
                JOIN message_attachment_join maj ON a.ROWID = maj.attachment_id
                JOIN message m ON maj.message_id = m.ROWID
                WHERE m.date >= ? AND m.date < ?
                """,
                (start_time, end_time),
            )
            attachment_rows = source_cursor.fetchall()
            
            if attachment_rows:
                # Get column names
                source_cursor.execute("PRAGMA table_info(attachment)")
                column_info = source_cursor.fetchall()
                column_names = [col[1] for col in column_info]
                
                # Insert attachments
                placeholders = ", ".join(["?"] * len(column_names))
                insert_sql = f"INSERT INTO attachment ({', '.join(column_names)}) VALUES ({placeholders})"
                
                attachment_ids = []
                for row in attachment_rows:
                    cursor.execute(insert_sql, row)
                    attachment_ids.append(row[0])  # Assuming ROWID is the first column
                
                conn.commit()
                
                # Copy message_attachment_join
                if attachment_ids and message_ids:
                    placeholders_a = ", ".join(["?"] * len(attachment_ids))
                    placeholders_m = ", ".join(["?"] * len(message_ids))
                    
                    source_cursor.execute(
                        f"""
                        SELECT * FROM message_attachment_join 
                        WHERE attachment_id IN ({placeholders_a})
                        AND message_id IN ({placeholders_m})
                        """,
                        attachment_ids + message_ids,
                    )
                    join_rows = source_cursor.fetchall()
                    
                    # Get column names
                    source_cursor.execute("PRAGMA table_info(message_attachment_join)")
                    column_info = source_cursor.fetchall()
                    column_names = [col[1] for col in column_info]
                    
                    # Insert joins
                    if join_rows:
                        placeholders = ", ".join(["?"] * len(column_names))
                        insert_sql = f"INSERT INTO message_attachment_join ({', '.join(column_names)}) VALUES ({placeholders})"
                        
                        for row in join_rows:
                            cursor.execute(insert_sql, row)
                        
                        conn.commit()
            
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_message_date ON message(date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_message_handle ON message(handle_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_message_join_chat ON chat_message_join(chat_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_message_join_message ON chat_message_join(message_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_message_attachment_join_message ON message_attachment_join(message_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_message_attachment_join_attachment ON message_attachment_join(attachment_id)")
            
            # Create combined index for common joins
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_message_join_combined ON chat_message_join(chat_id, message_id)")
            
            # Create FTS virtual table for full-text search
            try:
                # Check if FTS5 is available
                cursor.execute("SELECT sqlite_compileoption_used('ENABLE_FTS5')")
                has_fts5 = cursor.fetchone()[0]
                
                if has_fts5:
                    # Create FTS5 table
                    cursor.execute(
                        """
                        CREATE VIRTUAL TABLE IF NOT EXISTS message_fts USING fts5(
                            text,
                            content='message',
                            content_rowid='ROWID'
                        )
                        """
                    )
                    
                    # Populate FTS table
                    cursor.execute(
                        """
                        INSERT INTO message_fts(rowid, text)
                        SELECT ROWID, text FROM message WHERE text IS NOT NULL
                        """
                    )
                else:
                    # Fallback to FTS4
                    cursor.execute(
                        """
                        CREATE VIRTUAL TABLE IF NOT EXISTS message_fts USING fts4(
                            text,
                            content='message',
                            notindexed='ROWID'
                        )
                        """
                    )
                    
                    # Populate FTS table
                    cursor.execute(
                        """
                        INSERT INTO message_fts(docid, text)
                        SELECT ROWID, text FROM message WHERE text IS NOT NULL
                        """
                    )
            except sqlite3.Error as e:
                logger.warning(f"Error creating FTS table: {e}")
            
            # Create metadata table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS shard_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
                """
            )
            
            # Store metadata
            metadata = {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "created_at": datetime.now().isoformat(),
                "source_db": str(self.source_db_path),
                "message_count": len(message_rows),
            }
            
            for key, value in metadata.items():
                cursor.execute(
                    "INSERT INTO shard_metadata (key, value) VALUES (?, ?)",
                    (key, json.dumps(value)),
                )
            
            conn.commit()
            
            # Close connections
            cursor.close()
            conn.close()
            source_cursor.close()
            source_conn.close()
            
            return True
        
        except Exception as e:
            logger.error(f"Error creating shard {shard_path.name}: {e}")
            logger.error(traceback.format_exc())
            
            # Clean up partial shard
            if shard_path.exists():
                try:
                    os.unlink(shard_path)
                except Exception:
                    pass
            
            return False

    async def get_shards_for_date_range(
        self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
    ) -> List[DatabaseShard]:
        """
        Get shards that overlap with the given date range.
        
        Args:
            start_date: Optional start date of the range
            end_date: Optional end date of the range
            
        Returns:
            List of shards that overlap with the range
        """
        # If no shards, return empty list
        if not self.shards:
            return []
        
        # If no dates, return all shards
        if start_date is None and end_date is None:
            return self.shards
        
        # Find shards that overlap with the date range
        matching_shards = []
        for shard in self.shards:
            if shard.overlaps_range(start_date, end_date):
                matching_shards.append(shard)
        
        return matching_shards

    async def execute_across_shards(
        self,
        query: str,
        params: Optional[Tuple[Any, ...]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        fetchall: bool = True,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        timeout: Optional[float] = 30.0,
    ) -> List[Dict[str, Any]]:
        """
        Execute a query across shards and return combined results.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            start_date: Optional start date for filtering shards
            end_date: Optional end date for filtering shards
            fetchall: Whether to fetch all results
            limit: Maximum number of results to return
            offset: Number of results to skip
            timeout: Query timeout in seconds
            
        Returns:
            Combined results from all shards
        """
        # Get matching shards
        shards = await self.get_shards_for_date_range(start_date, end_date)
        
        if not shards:
            logger.warning("No matching shards found for query")
            return []
        
        # Check if query has ORDER BY
        has_order_by = bool(re.search(r"\bORDER\s+BY\b", query, re.IGNORECASE))
        
        # Check if query has LIMIT/OFFSET
        has_limit = bool(re.search(r"\bLIMIT\b", query, re.IGNORECASE))
        has_offset = bool(re.search(r"\bOFFSET\b", query, re.IGNORECASE))
        
        # Modify query if we need to handle limit/offset ourselves
        if (limit or offset) and not (has_limit and has_offset):
            # Remove existing LIMIT/OFFSET if present
            query = re.sub(r"\bLIMIT\s+\d+\b", "", query, flags=re.IGNORECASE)
            query = re.sub(r"\bOFFSET\s+\d+\b", "", query, flags=re.IGNORECASE)
        
        # Execute query on each shard in parallel
        tasks = []
        for shard in shards:
            tasks.append(
                shard.execute(
                    query=query,
                    params=params,
                    fetchall=fetchall,
                    timeout=timeout,
                )
            )
        
        # Run tasks in parallel
        try:
            results = await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error executing query across shards: {e}")
            return []
        
        # Flatten results
        flat_results = []
        for shard_results in results:
            flat_results.extend(shard_results)
        
        # Sort results if required
        if has_order_by:
            # This is a bit simplistic - we should parse the ORDER BY clause properly
            # This assumes results have keys that match the ORDER BY field names
            sort_columns = []
            sort_directions = []
            
            # Extract column names from ORDER BY clause
            match = re.search(
                r"\bORDER\s+BY\s+(.+?)(?:\s+LIMIT|\s+OFFSET|$)",
                query,
                re.IGNORECASE,
            )
            if match:
                order_by_clause = match.group(1)
                for column_spec in order_by_clause.split(","):
                    column_spec = column_spec.strip()
                    if " DESC" in column_spec.upper():
                        col = column_spec.split()[0].strip('"').strip("`'")
                        sort_columns.append(col)
                        sort_directions.append(-1)
                    else:
                        col = column_spec.split()[0].strip('"').strip("`'")
                        sort_columns.append(col)
                        sort_directions.append(1)
            
            # Sort results
            if sort_columns:
                for col_idx in range(len(sort_columns) - 1, -1, -1):
                    col = sort_columns[col_idx]
                    direction = sort_directions[col_idx]
                    # Try different column name variations
                    if col in flat_results[0] if flat_results else {}:
                        flat_results.sort(key=lambda x: x.get(col), reverse=direction < 0)
                    elif col.lower() in flat_results[0] if flat_results else {}:
                        flat_results.sort(
                            key=lambda x: x.get(col.lower()), reverse=direction < 0
                        )
        
        # Apply limit and offset if needed
        if offset:
            flat_results = flat_results[offset:]
        
        if limit:
            flat_results = flat_results[:limit]
        
        return flat_results

    async def execute_aggregation_across_shards(
        self,
        query: str,
        aggregation_columns: List[str],
        group_by_columns: List[str],
        params: Optional[Tuple[Any, ...]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        timeout: Optional[float] = 30.0,
    ) -> List[Dict[str, Any]]:
        """
        Execute an aggregation query across shards and combine the results.
        
        Args:
            query: SQL query with aggregation (COUNT, SUM, etc.)
            aggregation_columns: List of column names that have aggregation functions
            group_by_columns: List of column names in the GROUP BY clause
            params: Query parameters
            start_date: Optional start date for filtering shards
            end_date: Optional end date for filtering shards
            timeout: Query timeout in seconds
            
        Returns:
            Combined and properly aggregated results
        """
        # Get matching shards
        shards = await self.get_shards_for_date_range(start_date, end_date)
        
        if not shards:
            logger.warning("No matching shards found for aggregation query")
            return []
        
        # Execute query on each shard in parallel
        tasks = []
        for shard in shards:
            tasks.append(
                shard.execute(
                    query=query,
                    params=params,
                    fetchall=True,
                    timeout=timeout,
                )
            )
        
        # Run tasks in parallel
        try:
            results = await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error executing aggregation query across shards: {e}")
            return []
        
        # Flatten results
        flat_results = []
        for shard_results in results:
            flat_results.extend(shard_results)
        
        # If there are no results, return empty list
        if not flat_results:
            return []
        
        # If no GROUP BY, just sum the aggregation columns
        if not group_by_columns:
            combined = {}
            for column in aggregation_columns:
                combined[column] = sum(float(r.get(column, 0)) for r in flat_results)
            return [combined]
        
        # For queries with GROUP BY, combine results by group
        grouped_results = {}
        for row in flat_results:
            # Create a key from the group by columns
            group_key = tuple(str(row.get(col, "")) for col in group_by_columns)
            
            if group_key not in grouped_results:
                # Initialize new group with group by values and zero aggregates
                grouped_results[group_key] = {}
                for col in group_by_columns:
                    grouped_results[group_key][col] = row.get(col)
                for col in aggregation_columns:
                    grouped_results[group_key][col] = 0
            
            # Sum the aggregation columns
            for col in aggregation_columns:
                grouped_results[group_key][col] += float(row.get(col, 0))
        
        # Convert back to list
        return list(grouped_results.values())

    async def get_shard_status(self) -> Dict[str, Any]:
        """
        Get status information about all shards.
        
        Returns:
            Dictionary with shard status information
        """
        shards_info = []
        for shard in self.shards:
            try:
                metadata = await shard.get_metadata()
                shards_info.append(metadata)
            except Exception as e:
                logger.error(f"Error getting metadata for shard {shard.shard_path.name}: {e}")
                shards_info.append(
                    {
                        "path": str(shard.shard_path),
                        "error": str(e),
                    }
                )
        
        return {
            "shards_count": len(self.shards),
            "shards": shards_info,
            "source_db": str(self.source_db_path),
            "shards_dir": str(self.shards_dir),
            "shard_size_months": self.shard_size_months,
        }

    async def cleanup(self):
        """Clean up resources used by the manager and all shards."""
        for shard in self.shards:
            await shard.cleanup()


import traceback  # For error reporting