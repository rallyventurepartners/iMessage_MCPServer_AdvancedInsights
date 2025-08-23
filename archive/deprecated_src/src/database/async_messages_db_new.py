#!/usr/bin/env python3
"""
Async Messages Database - New Modular Version

This is an enhanced version of the AsyncMessagesDB with improved connection pooling,
better cache management, and more comprehensive error handling.
"""

import asyncio
import json
import logging
import os
import re
import stat
import time
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import (Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple,
                    Union)
from datetime import datetime

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

# Import the contact resolver
from src.utils.contact_resolver import ContactResolverFactory

# Configure logging
logger = logging.getLogger(__name__)

# Default database path for macOS
HOME = os.path.expanduser("~")
DB_PATH = Path(f"{HOME}/Library/Messages/chat.db")


class AsyncMessagesDB(AsyncMessagesDBBase):
    """
    An asynchronous class for handling message database operations.

    This is an enhanced version with improved connection pooling,
    better cache management, and more comprehensive error handling.
    """

    def __init__(self, db_path=None, minimal_mode=False, pool_size=10):
        """Initialize the database connection.

        Args:
            db_path: Path to the SQLite database file. If None, uses the default macOS path.
            minimal_mode: Start in minimal mode for faster performance with less data.
            pool_size: Size of the connection pool.
        """
        self.db_path = Path(db_path) if db_path else DB_PATH
        self._lock = asyncio.Lock()
        self._connection_pool = []
        self._busy_connections = set()
        self.initialized = False

        # Connection pool configuration
        self._max_connections = pool_size
        self._connection_timeout = 30.0  # Seconds to wait for a connection
        self._connection_ttl = 300  # Seconds to keep a connection before refreshing
        self._connection_timestamps = {}  # Track when connections were created
        self._connection_waiters = []  # Queue of waiters for connections

        # Performance configuration
        self.minimal_mode = minimal_mode
        self._has_fts5 = False
        self.is_indexed_db = False

        # Cache initialization

        # Contact resolver
        self.contact_resolver = None

    def check_database_file(self):
        """Check if the database file is accessible and log its properties.

        Returns:
            bool: True if the file exists and is readable, False otherwise
        """
        if not os.path.exists(self.db_path):
            logger.error(f"Database file does not exist: {self.db_path}")
            return False

        # Check read permissions
        if not os.access(self.db_path, os.R_OK):
            logger.error(f"No read permission for database file: {self.db_path}")
            return False

        # Check file size
        file_size = os.path.getsize(self.db_path)
        logger.info(f"iMessage database size: {file_size / (1024*1024):.2f} MB")

        # Check if this is a possibly indexed database (based on path)
        indexed_indicators = [".imessage_insights", "indexed", "index"]
        if any(
            indicator in str(self.db_path).lower() for indicator in indexed_indicators
        ):
            logger.info(f"This appears to be an indexed database: {self.db_path}")
            self.is_indexed_db = True

        return True

    async def initialize(self):
        """Initialize the database connection pool asynchronously.

        This method must be called before any other async methods.

        Raises:
            FileNotFoundError: If the database file doesn't exist
            DatabaseError: If there's an error connecting to the database
        """
        if self.initialized:
            return

        async with self._lock:
            if self.initialized:
                return

            # Check if database exists and is accessible
            if not self.check_database_file():
                raise FileNotFoundError(
                    f"Database file not found or not accessible at {self.db_path}"
                )


            # Initialize at least one connection
            try:
                # Use URI connection string with read-only mode
                uri = f"file:{self.db_path}?mode=ro&immutable=1"

                initial_conn = await aiosqlite.connect(uri, uri=True)
                initial_conn.row_factory = aiosqlite.Row
                
                # Apply performance optimizations
                await initial_conn.execute("PRAGMA cache_size = 10000")  # ~40MB cache
                await initial_conn.execute("PRAGMA temp_store = MEMORY")
                await initial_conn.execute("PRAGMA mmap_size = 268435456")  # 256MB memory map
                await initial_conn.execute("PRAGMA journal_mode = WAL")  # Write-ahead logging
                await initial_conn.execute("PRAGMA synchronous = NORMAL")
                await initial_conn.execute("PRAGMA page_size = 4096")  # Optimal page size
                await initial_conn.execute("PRAGMA busy_timeout = 5000")  # 5 second timeout

                # Check for FTS5 support
                try:
                    cursor = await initial_conn.execute(
                        "SELECT sqlite_compileoption_used('ENABLE_FTS5')"
                    )
                    row = await cursor.fetchone()
                    self._has_fts5 = bool(row and row[0] == 1)
                    logger.info(f"SQLite FTS5 support available: {self._has_fts5}")
                except Exception as e:
                    logger.warning(f"Could not determine FTS5 support: {e}")
                    self._has_fts5 = False

                # Add connection to the pool
                self._connection_pool.append(initial_conn)
                self._connection_timestamps[id(initial_conn)] = time.time()

                # Initialize contact resolver
                self.contact_resolver = (
                    await ContactResolverFactory.create_async_resolver()
                )

                self.initialized = True
                logger.info(
                    f"AsyncMessagesDB initialized with database: {self.db_path}"
                )

            except Exception as e:
                logger.error(f"Error initializing database: {e}")
                logger.error(traceback.format_exc())
                raise DatabaseError(f"Failed to initialize database: {e}")

    @asynccontextmanager
    async def get_db_connection(self):
        """Get a database connection from the pool.

        Returns:
            A database connection from the pool

        Raises:
            DatabaseError: If unable to get a connection
        """
        if not self.initialized:
            raise DatabaseError("Database not initialized. Call initialize() first.")

        conn = None
        waiter = None
        start_time = time.time()

        try:
            while True:
                # Try to get a connection from the pool
                async with self._lock:
                    if self._connection_pool:
                        # Get the oldest connection from the pool
                        conn = self._connection_pool.pop(0)
                        self._busy_connections.add(conn)
                        break
                    elif len(self._busy_connections) < self._max_connections:
                        # Create a new connection if pool is empty but under max connections
                        uri = f"file:{self.db_path}?mode=ro&immutable=1"
                        conn = await aiosqlite.connect(uri, uri=True)
                        conn.row_factory = aiosqlite.Row
                        
                        # Apply performance optimizations
                        await conn.execute("PRAGMA cache_size = 10000")
                        await conn.execute("PRAGMA temp_store = MEMORY")
                        await conn.execute("PRAGMA mmap_size = 268435456")
                        await conn.execute("PRAGMA journal_mode = WAL")
                        await conn.execute("PRAGMA synchronous = NORMAL")
                        await conn.execute("PRAGMA page_size = 4096")
                        await conn.execute("PRAGMA busy_timeout = 5000")
                        
                        self._busy_connections.add(conn)
                        self._connection_timestamps[id(conn)] = time.time()
                        break
                    else:
                        # No available connections and at max capacity - add to waiters
                        if waiter is None:
                            waiter = asyncio.Future()
                            self._connection_waiters.append(waiter)
                
                # If we need to wait, do so outside the lock
                if waiter and not conn:
                    # Check timeout
                    elapsed = time.time() - start_time
                    if elapsed >= self._connection_timeout:
                        # Remove from waiters
                        async with self._lock:
                            if waiter in self._connection_waiters:
                                self._connection_waiters.remove(waiter)
                        raise DatabaseError(
                            f"Timeout waiting for database connection after {self._connection_timeout}s"
                        )
                    
                    # Wait for a connection to become available
                    try:
                        await asyncio.wait_for(
                            waiter, 
                            timeout=self._connection_timeout - elapsed
                        )
                        # Waiter was resolved, loop again to get the connection
                        waiter = None
                    except asyncio.TimeoutError:
                        # Remove from waiters
                        async with self._lock:
                            if waiter in self._connection_waiters:
                                self._connection_waiters.remove(waiter)
                        raise DatabaseError(
                            f"Timeout waiting for database connection after {self._connection_timeout}s"
                        )

            # Check connection health
            try:
                async with conn.execute("SELECT 1") as cursor:
                    await cursor.fetchone()
            except Exception as e:
                logger.error(f"Detected unhealthy connection: {e}")

                # Remove from busy connections
                self._busy_connections.discard(conn)

                try:
                    await conn.close()
                except:
                    pass

                # Create new connection
                uri = f"file:{self.db_path}?mode=ro&immutable=1"
                conn = await aiosqlite.connect(uri, uri=True)
                conn.row_factory = aiosqlite.Row
                
                # Apply performance optimizations
                await conn.execute("PRAGMA cache_size = 10000")
                await conn.execute("PRAGMA temp_store = MEMORY")
                await conn.execute("PRAGMA mmap_size = 268435456")
                await conn.execute("PRAGMA journal_mode = WAL")
                await conn.execute("PRAGMA synchronous = NORMAL")
                await conn.execute("PRAGMA page_size = 4096")
                await conn.execute("PRAGMA busy_timeout = 5000")
                
                self._busy_connections.add(conn)
                self._connection_timestamps[id(conn)] = time.time()

            yield conn

        finally:
            # Return connection to the pool
            if conn is not None:
                async with self._lock:
                    if conn in self._busy_connections:
                        self._busy_connections.remove(conn)
                        
                        # Check if there are waiters
                        if self._connection_waiters:
                            # Give connection to first waiter
                            waiter = self._connection_waiters.pop(0)
                            if not waiter.done():
                                waiter.set_result(None)
                            return

                        # Check if connection is too old and should be refreshed
                        conn_age = time.time() - self._connection_timestamps.get(
                            id(conn), 0
                        )
                        if conn_age > self._connection_ttl:
                            # Close old connection
                            try:
                                await conn.close()
                            except:
                                pass

                            # Create fresh connection
                            try:
                                uri = f"file:{self.db_path}?mode=ro&immutable=1"
                                fresh_conn = await aiosqlite.connect(uri, uri=True)
                                fresh_conn.row_factory = aiosqlite.Row
                                
                                # Apply performance optimizations
                                await fresh_conn.execute("PRAGMA cache_size = 10000")
                                await fresh_conn.execute("PRAGMA temp_store = MEMORY")
                                await fresh_conn.execute("PRAGMA mmap_size = 268435456")
                                await fresh_conn.execute("PRAGMA journal_mode = WAL")
                                await fresh_conn.execute("PRAGMA synchronous = NORMAL")
                                await fresh_conn.execute("PRAGMA page_size = 4096")
                                await fresh_conn.execute("PRAGMA busy_timeout = 5000")
                                
                                self._connection_pool.append(fresh_conn)
                                self._connection_timestamps[id(fresh_conn)] = (
                                    time.time()
                                )
                            except Exception as e:
                                logger.error(f"Error creating fresh connection: {e}")
                        else:
                            # Return healthy connection to the pool
                            self._connection_pool.append(conn)

    @asynccontextmanager
    async def transaction(self):
        """Get a database connection with transaction support.

        This is a convenience wrapper around get_db_connection
        that adds transaction management.

        Returns:
            Database connection with transaction support

        Raises:
            DatabaseError: If unable to get a connection
        """
        async with self.get_db_connection() as conn:
            await conn.execute("BEGIN")
            try:
                yield conn
                await conn.execute("COMMIT")
            except Exception as e:
                try:
                    await conn.execute("ROLLBACK")
                except:
                    pass
                raise

    async def close(self):
        """Close all database connections."""
        if not self.initialized:
            return

        async with self._lock:
            # Close pool connections
            for conn in self._connection_pool:
                try:
                    await conn.close()
                except:
                    pass

            # Close busy connections
            for conn in self._busy_connections:
                try:
                    await conn.close()
                except:
                    pass

            # Clear collections
            self._connection_pool.clear()
            self._busy_connections.clear()
            self._connection_timestamps.clear()
            self.initialized = False

            logger.info("All database connections closed")

    async def get_contacts(self, limit=100, offset=0, minimal=None) -> Dict[str, Any]:
        """Get a list of all contacts from the iMessage database.

        Args:
            limit: Maximum number of contacts to return
            offset: Number of contacts to skip
            minimal: Whether to return minimal data (overrides instance setting)

        Returns:
            Dictionary with contacts list and metadata
        """
        minimal = self.minimal_mode if minimal is None else minimal

        try:
            query = """
            SELECT DISTINCT
                handle.id AS phone_number,
                handle.ROWID AS handle_id,
                handle.service AS service
            FROM 
                handle
            JOIN 
                message ON handle.ROWID = message.handle_id
            ORDER BY 
                handle.id
            LIMIT ? OFFSET ?
            """

            # Get total count (separate query)
            count_query = """
            SELECT COUNT(DISTINCT handle.id) AS count
            FROM handle
            JOIN message ON handle.ROWID = message.handle_id
            """

            async with self.get_db_connection() as conn:
                # Get total count
                cursor = await conn.execute(count_query)
                row = await cursor.fetchone()
                total_count = row[0] if row else 0

                # Get contacts
                cursor = await conn.execute(query, (limit, offset))
                rows = await cursor.fetchall()

                contacts = []
                for row in rows:
                    phone_number = row["phone_number"]
                    handle_id = row["handle_id"]
                    service = row["service"]

                    # Get contact info with name resolution
                    contact_info = (
                        await self.contact_resolver.get_contact_by_identifier(
                            phone_number
                        )
                    )
                    display_name = contact_info.get("display_name", phone_number)
                    first_name = contact_info.get("first_name")
                    last_name = contact_info.get("last_name")
                    organization = contact_info.get("organization")

                    # Get message count for this contact (only if not in minimal mode)
                    message_count = 0
                    last_message_date = None

                    if not minimal:
                        # Get message count
                        msg_count_query = """
                        SELECT COUNT(*) AS count
                        FROM message
                        WHERE handle_id = ?
                        """
                        cursor = await conn.execute(msg_count_query, (handle_id,))
                        row = await cursor.fetchone()
                        message_count = row[0] if row else 0

                        # Get last message date
                        last_msg_query = """
                        SELECT date
                        FROM message
                        WHERE handle_id = ?
                        ORDER BY date DESC
                        LIMIT 1
                        """
                        cursor = await conn.execute(last_msg_query, (handle_id,))
                        row = await cursor.fetchone()
                        if row:
                            # Convert Apple's timestamp to datetime (epoch + nanoseconds since 2001-01-01)
                            date_val = row[0]
                            if date_val:
                                # Apple timestamp is nanoseconds since 2001-01-01
                                seconds_since_2001 = date_val / 1e9
                                epoch_2001 = (
                                    978307200  # 2001-01-01 in Unix epoch seconds
                                )
                                unix_timestamp = epoch_2001 + seconds_since_2001
                                dt = datetime.fromtimestamp(unix_timestamp)
                                last_message_date = dt.isoformat()

                    contacts.append(
                        {
                            # Use normalized identifier instead of raw phone_number
                            "identifier": contact_info.get("identifier", phone_number),
                            "phone_number": phone_number,  # Keep for backwards compatibility
                            "display_name": display_name,
                            "first_name": first_name,
                            "last_name": last_name,
                            "organization": organization,
                            "service": service,
                            "handle_id": handle_id,
                            "identifier_type": "email" if "@" in phone_number else "phone",
                            "message_count": message_count,
                            "last_message_date": last_message_date,
                        }
                    )

                # Calculate pagination
                has_more = total_count > (offset + limit)
                total_pages = (total_count + limit - 1) // limit if limit > 0 else 1
                current_page = (offset // limit) + 1 if limit > 0 else 1

                return {
                    "contacts": contacts,
                    "total": total_count,
                    "pagination": {
                        "limit": limit,
                        "offset": offset,
                        "has_more": has_more,
                        "total_pages": total_pages,
                        "current_page": current_page,
                    },
                }

        except Exception as e:
            logger.error(f"Error getting contacts: {e}")
            logger.error(traceback.format_exc())
            return {"error": str(e), "contacts": [], "total": 0}

    async def get_group_chats(self, limit=20, offset=0) -> Dict[str, Any]:
        """Get a list of all group chats.

        Args:
            limit: Maximum number of group chats to return
            offset: Number of group chats to skip

        Returns:
            Dictionary with group chats list and metadata
        """
        try:
            query = """
            SELECT 
                chat.ROWID as chat_id,
                chat.display_name as name,
                chat.guid
            FROM 
                chat
            WHERE 
                chat.chat_identifier LIKE 'chat%'
            ORDER BY 
                chat.ROWID
            LIMIT ? OFFSET ?
            """

            # Get total count
            count_query = """
            SELECT COUNT(*) as count
            FROM chat
            WHERE chat.chat_identifier LIKE 'chat%'
            """

            async with self.get_db_connection() as conn:
                # Get total count
                cursor = await conn.execute(count_query)
                row = await cursor.fetchone()
                total_count = row[0] if row else 0

                # Get chats
                cursor = await conn.execute(query, (limit, offset))
                rows = await cursor.fetchall()

                chats = []
                for row in rows:
                    chat_id = row["chat_id"]
                    display_name = row["name"]
                    guid = row["guid"]

                    # Get participants for better name generation
                    participants_query = """
                    SELECT handle.id as identifier
                    FROM chat_handle_join
                    JOIN handle ON chat_handle_join.handle_id = handle.ROWID
                    WHERE chat_handle_join.chat_id = ?
                    LIMIT 5
                    """
                    cursor = await conn.execute(participants_query, (chat_id,))
                    participant_rows = await cursor.fetchall()
                    participant_count = len(participant_rows)
                    
                    # Generate better group name if none exists
                    if not display_name and participant_rows:
                        # Get names for participants
                        participant_names = []
                        for p_row in participant_rows[:3]:  # First 3 participants
                            contact_info = await self.contact_resolver.get_contact_by_identifier(p_row["identifier"])
                            name = contact_info.get("display_name", p_row["identifier"])
                            if name:
                                participant_names.append(name.split()[0] if ' ' in name else name)  # First name only
                        
                        if participant_names:
                            if len(participant_rows) > 3:
                                display_name = f"{', '.join(participant_names)} +{len(participant_rows) - 3}"
                            else:
                                display_name = ', '.join(participant_names)
                    
                    if not display_name:
                        display_name = f"Group Chat {chat_id}"

                    # Count messages
                    message_count_query = """
                    SELECT COUNT(*) as count
                    FROM chat_message_join
                    WHERE chat_id = ?
                    """
                    cursor = await conn.execute(message_count_query, (chat_id,))
                    m_row = await cursor.fetchone()
                    message_count = m_row[0] if m_row else 0

                    # Get last message date
                    last_msg_query = """
                    SELECT message.date
                    FROM message
                    JOIN chat_message_join ON message.ROWID = chat_message_join.message_id
                    WHERE chat_message_join.chat_id = ?
                    ORDER BY message.date DESC
                    LIMIT 1
                    """
                    cursor = await conn.execute(last_msg_query, (chat_id,))
                    d_row = await cursor.fetchone()
                    last_message_date = None
                    if d_row and d_row[0]:
                        # Convert Apple's timestamp to datetime
                        date_val = d_row[0]
                        seconds_since_2001 = date_val / 1e9
                        epoch_2001 = 978307200
                        unix_timestamp = epoch_2001 + seconds_since_2001
                        dt = datetime.fromtimestamp(unix_timestamp)
                        last_message_date = dt.isoformat()

                    chats.append(
                        {
                            "chat_id": chat_id,
                            "chat_identifier": guid,  # The actual identifier used in database
                            "display_name": display_name,
                            "name": display_name,  # Keep for backwards compatibility
                            "guid": guid,
                            "participant_count": participant_count,
                            "message_count": message_count,
                            "last_message_date": last_message_date,
                        }
                    )

                # Calculate pagination
                has_more = total_count > (offset + limit)
                total_pages = (total_count + limit - 1) // limit if limit > 0 else 1
                current_page = (offset // limit) + 1 if limit > 0 else 1

                return {
                    "chats": chats,
                    "total": total_count,
                    "has_more": has_more,
                    "pagination": {
                        "limit": limit,
                        "offset": offset,
                        "total_pages": total_pages,
                        "current_page": current_page,
                    },
                }

        except Exception as e:
            logger.error(f"Error getting group chats: {e}")
            logger.error(traceback.format_exc())
            return {"error": str(e), "chats": [], "total": 0}

    async def validate_schema_version(self) -> bool:
        """Check if the database schema is compatible with this application.

        Returns:
            True if schema is compatible, False otherwise
        """
        try:
            async with self.get_db_connection() as conn:
                # Check for key tables
                tables_query = """
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name IN ('message', 'handle', 'chat', 'chat_message_join')
                """
                cursor = await conn.execute(tables_query)
                rows = await cursor.fetchall()

                if len(rows) < 4:
                    missing_tables = set(
                        ["message", "handle", "chat", "chat_message_join"]
                    ) - set(r[0] for r in rows)
                    logger.error(f"Missing required tables: {missing_tables}")
                    return False

                # Check for key columns in message table
                columns_query = "PRAGMA table_info(message)"
                cursor = await conn.execute(columns_query)
                columns = await cursor.fetchall()
                column_names = [col[1] for col in columns]

                required_columns = [
                    "ROWID",
                    "guid",
                    "text",
                    "handle_id",
                    "date",
                    "is_from_me",
                ]
                missing_columns = set(required_columns) - set(column_names)

                if missing_columns:
                    logger.error(
                        f"Missing required columns in message table: {missing_columns}"
                    )
                    return False

                logger.info("Database schema validated successfully")
                return True

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
        try:
            # Normalize chat_id to integer if possible
            if isinstance(chat_id, str) and chat_id.isdigit():
                chat_id = int(chat_id)

            # Use different lookup based on type
            if isinstance(chat_id, int):
                query = "SELECT ROWID as chat_id, display_name, guid FROM chat WHERE ROWID = ?"
                params = (chat_id,)
            else:
                # Try to use chat_id as GUID
                query = "SELECT ROWID as chat_id, display_name, guid FROM chat WHERE guid = ?"
                params = (chat_id,)

            async with self.get_db_connection() as conn:
                cursor = await conn.execute(query, params)
                row = await cursor.fetchone()

                if not row:
                    # Try looking up by name if not found
                    query = "SELECT ROWID as chat_id, display_name, guid FROM chat WHERE display_name = ?"
                    cursor = await conn.execute(query, (chat_id,))
                    row = await cursor.fetchone()

                if row:
                    chat_dict = dict(row)

                    # Get participants
                    participants_query = """
                    SELECT handle.id 
                    FROM handle
                    JOIN chat_handle_join ON handle.ROWID = chat_handle_join.handle_id
                    WHERE chat_handle_join.chat_id = ?
                    """
                    cursor = await conn.execute(
                        participants_query, (chat_dict["chat_id"],)
                    )
                    participant_rows = await cursor.fetchall()
                    participants = [p[0] for p in participant_rows]

                    chat_dict["participants"] = participants
                    chat_dict["participant_count"] = len(participants)

                    # Ensure we have a display name
                    if not chat_dict.get("display_name"):
                        chat_dict["display_name"] = f"Group Chat {chat_dict['chat_id']}"

                    return chat_dict
                else:
                    return {}

        except Exception as e:
            logger.error(f"Error getting chat by ID {chat_id}: {e}")
            logger.error(traceback.format_exc())
            return {}

    async def optimize_database(self) -> Dict[str, Any]:
        """Perform database optimizations.

        This method is a placeholder that logs the fact that we can't optimize
        the database in read-only mode.

        Returns:
            Dictionary with optimization results
        """
        logger.warning(
            "Database optimization requested, but database is in read-only mode."
        )
        logger.warning("To optimize the database, use the index_imessage_db.py script.")

        return {
            "success": False,
            "reason": "Database is in read-only mode",
            "suggested_action": "Use index_imessage_db.py script to create an optimized copy",
        }
        
    async def should_use_sharding(self) -> bool:
        """
        Determine if database sharding should be used based on database size.
        
        Returns:
            True if sharding is recommended, False otherwise
        """
        # If no database path, can't determine
        if not self.db_path or not os.path.exists(self.db_path):
            logger.warning("Cannot determine if sharding is needed - database path not accessible")
            return False
            
        try:
            # Get database size
            db_size_bytes = os.path.getsize(self.db_path)
            db_size_gb = db_size_bytes / (1024 ** 3)
            
            # If database is larger than 5GB, recommend sharding
            if db_size_gb >= 5:
                logger.info(f"Database size is {db_size_gb:.2f} GB, sharding is recommended")
                return True
                
            logger.info(f"Database size is {db_size_gb:.2f} GB, sharding is not necessary")
            return False
        except Exception as e:
            logger.error(f"Error determining if sharding should be used: {e}")
            logger.error(traceback.format_exc())
            return False
    
    # Implementation of missing abstract methods
    async def get_contact_by_phone_or_email(self, identifier: str) -> Dict[str, Any]:
        """Get contact information by phone number or email."""
        from .async_messages_db_implementations import get_contact_by_phone_or_email_impl
        return await get_contact_by_phone_or_email_impl(self, identifier)
    
    async def get_messages_from_chat(
        self, 
        chat_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Get messages from a specific chat/conversation."""
        from .async_messages_db_implementations import get_messages_from_chat_impl
        return await get_messages_from_chat_impl(self, chat_id, start_date, end_date, limit, offset)
    
    async def get_messages_from_contact(
        self, 
        phone_number: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        page: int = 1,
        page_size: int = 50
    ) -> Dict[str, Any]:
        """Get messages exchanged with a specific contact."""
        from .async_messages_db_implementations import get_messages_from_contact_impl
        return await get_messages_from_contact_impl(self, phone_number, start_date, end_date, page, page_size)
    
    async def search_messages(
        self, 
        query: str,
        contact_id: Optional[str] = None,
        chat_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Search messages by content."""
        from .async_messages_db_implementations import search_messages_impl
        return await search_messages_impl(self, query, contact_id, chat_id, start_date, end_date, limit, offset)
    
    async def get_contact_info(self, contact_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific contact."""
        # Find the contact in our contacts list
        contacts_result = await self.get_contacts(limit=1000, offset=0)
        
        for contact in contacts_result.get("contacts", []):
            if (contact.get("phone_number") == contact_id or 
                contact.get("handle_id") == contact_id or
                contact.get("id") == contact_id):
                return contact
        
        # If not found, return a basic structure
        return {
            "id": contact_id,
            "phone_number": contact_id,
            "handle_id": contact_id,
            "display_name": contact_id,
            "message_count": 0,
            "last_message_date": None
        }

    async def _normalize_contact_identifier(self, contact_id: str) -> str:
        """Normalize contact identifier to standard format."""
        if self.contact_resolver:
            return self.contact_resolver.normalize_identifier(contact_id)
        return contact_id
    
    async def get_contact_analytics(
        self,
        contact_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get detailed analytics for a specific contact."""
        try:
            # Normalize the contact identifier
            normalized_id = await self._normalize_contact_identifier(contact_id)
            
            # Build date filter
            date_filters_sql, date_params = build_date_filters_sql(start_date, end_date)
            
            # Build WHERE clause with date filters
            where_clause = "(handle.id = ? OR handle.id LIKE ? OR chat.chat_identifier = ?)"
            if date_filters_sql:
                where_clause += f" AND {date_filters_sql}"
            
            # Get basic message counts
            query = f"""
            SELECT 
                COUNT(*) as total_messages,
                SUM(CASE WHEN message.is_from_me = 1 THEN 1 ELSE 0 END) as sent_messages,
                SUM(CASE WHEN message.is_from_me = 0 THEN 1 ELSE 0 END) as received_messages,
                SUM(CASE WHEN message.cache_has_attachments = 1 THEN 1 ELSE 0 END) as attachments,
                AVG(LENGTH(message.text)) as avg_message_length,
                COUNT(DISTINCT DATE(datetime(message.date/1000000000 + 978307200, 'unixepoch'))) as active_days
            FROM message
            JOIN chat_message_join ON message.ROWID = chat_message_join.message_id
            JOIN chat ON chat_message_join.chat_id = chat.ROWID
            JOIN handle ON message.handle_id = handle.ROWID
            WHERE {where_clause}
            """
            
            # Combine parameters - use normalized ID and also try with wildcards for flexible matching
            params = [normalized_id, f"%{normalized_id}%", normalized_id]
            if date_params:
                params.extend(date_params.values())
            
            async with self.get_db_connection() as conn:
                result = await conn.execute(query, params)
                row = await result.fetchone()
                
                analytics = {
                    "total_messages": row[0] or 0,
                    "sent_messages": row[1] or 0,
                    "received_messages": row[2] or 0,
                    "attachments": row[3] or 0,
                    "avg_message_length": round(row[4] or 0, 2),
                    "active_days": row[5] or 0
                }
                
                # Get activity patterns
                hour_query = f"""
                SELECT 
                    strftime('%H', datetime(message.date/1000000000 + 978307200, 'unixepoch')) as hour,
                    COUNT(*) as count
                FROM message
                JOIN chat_message_join ON message.ROWID = chat_message_join.message_id
                JOIN chat ON chat_message_join.chat_id = chat.ROWID
                JOIN handle ON message.handle_id = handle.ROWID
                WHERE (handle.id = ? OR chat.chat_identifier = ?)
                {" AND " + date_filters_sql if date_filters_sql else ""}
                GROUP BY hour
                ORDER BY count DESC
                LIMIT 1
                """
                
                result = await conn.execute(hour_query, params)
                hour_row = await result.fetchone()
                if hour_row:
                    analytics["most_active_hour"] = int(hour_row[0])
                
                # Get day of week patterns
                dow_query = f"""
                SELECT 
                    CASE strftime('%w', datetime(message.date/1000000000 + 978307200, 'unixepoch'))
                        WHEN '0' THEN 'Sunday'
                        WHEN '1' THEN 'Monday'
                        WHEN '2' THEN 'Tuesday'
                        WHEN '3' THEN 'Wednesday'
                        WHEN '4' THEN 'Thursday'
                        WHEN '5' THEN 'Friday'
                        WHEN '6' THEN 'Saturday'
                    END as day_name,
                    COUNT(*) as count
                FROM message
                JOIN chat_message_join ON message.ROWID = chat_message_join.message_id
                JOIN chat ON chat_message_join.chat_id = chat.ROWID
                JOIN handle ON message.handle_id = handle.ROWID
                WHERE (handle.id = ? OR chat.chat_identifier = ?)
                {" AND " + date_filters_sql if date_filters_sql else ""}
                GROUP BY day_name
                ORDER BY count DESC
                LIMIT 1
                """
                
                result = await conn.execute(dow_query, params)
                dow_row = await result.fetchone()
                if dow_row:
                    analytics["most_active_day"] = dow_row[0]
                
                # Calculate conversation metrics (simplified)
                analytics["conversation_count"] = analytics["active_days"]  # Approximate
                analytics["avg_messages_per_conversation"] = round(analytics["total_messages"] / max(analytics["active_days"], 1), 2)
                
                return analytics
                
        except Exception as e:
            logger.error(f"Error getting contact analytics: {e}")
            return {
                "total_messages": 0,
                "sent_messages": 0,
                "received_messages": 0,
                "attachments": 0,
                "avg_message_length": 0,
                "active_days": 0
            }

    async def get_conversation_topics(
        self,
        contact_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        topic_count: int = 10,
        include_sentiment: bool = True
    ) -> Dict[str, Any]:
        """Extract conversation topics using simple keyword analysis."""
        try:
            # Normalize the contact identifier
            normalized_id = await self._normalize_contact_identifier(contact_id)
            
            # Build date filter
            date_filters_sql, date_params = build_date_filters_sql(start_date, end_date)
            
            # Get messages for analysis
            query = f"""
            SELECT message.text
            FROM message
            JOIN chat_message_join ON message.ROWID = chat_message_join.message_id
            JOIN chat ON chat_message_join.chat_id = chat.ROWID
            JOIN handle ON message.handle_id = handle.ROWID
            WHERE (handle.id = ? OR handle.id LIKE ? OR chat.chat_identifier = ?)
            AND message.text IS NOT NULL AND message.text != ''
            {" AND " + date_filters_sql if date_filters_sql else ""}
            ORDER BY message.date DESC
            LIMIT 1000
            """
            
            params = [normalized_id, f"%{normalized_id}%", normalized_id]
            if date_params:
                params.extend(date_params.values())
            
            async with self.get_db_connection() as conn:
                result = await conn.execute(query, params)
                messages = await result.fetchall()
                
                # Simple topic extraction based on common words
                word_freq = {}
                total_messages = len(messages)
                
                # Common words to ignore
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
                            'before', 'after', 'above', 'below', 'between', 'under', 'again',
                            'further', 'then', 'once', 'i', 'me', 'my', 'myself', 'we', 'our',
                            'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
                            'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
                            'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
                            'am', 'is', 'are', 'was', 'were', 'been', 'being', 'have', 'has',
                            'had', 'having', 'do', 'does', 'did', 'doing', 'will', 'would',
                            'should', 'could', 'ought', 'im', 'youre', 'hes', 'shes', 'its',
                            'were', 'theyre', 'ive', 'youve', 'weve', 'theyve', 'id', 'youd',
                            'hed', 'shed', 'wed', 'theyd', 'ill', 'youll', 'hell', 'shell',
                            'well', 'theyll', 'isnt', 'arent', 'wasnt', 'werent', 'hasnt',
                            'havent', 'hadnt', 'doesnt', 'dont', 'didnt', 'wont', 'wouldnt',
                            'shant', 'shouldnt', 'cant', 'cannot', 'couldnt', 'mustnt', 'lets',
                            'thats', 'whos', 'whats', 'heres', 'theres', 'wheres', 'whens',
                            'whys', 'hows', 'because', 'as', 'until', 'while', 'of', 'at',
                            'by', 'for', 'with', 'about', 'against', 'between', 'into',
                            'through', 'during', 'before', 'after', 'above', 'below', 'to',
                            'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
                            'again', 'further', 'then', 'once'}
                
                for msg_row in messages:
                    if msg_row[0]:
                        # Clean and tokenize
                        words = re.findall(r'\b[a-z]+\b', msg_row[0].lower())
                        for word in words:
                            if len(word) > 3 and word not in stop_words:
                                word_freq[word] = word_freq.get(word, 0) + 1
                
                # Get top words as topics
                sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
                top_words = sorted_words[:topic_count]
                
                topics = []
                for word, count in top_words:
                    topic = {
                        "topic": word.title(),
                        "keywords": [word],
                        "message_count": count,
                        "percentage": round((count / total_messages) * 100, 2) if total_messages > 0 else 0
                    }
                    
                    if include_sentiment:
                        # Simple sentiment (placeholder - would need proper sentiment analysis)
                        topic["sentiment"] = {
                            "score": 0.5,
                            "positive": 0.33,
                            "neutral": 0.34,
                            "negative": 0.33
                        }
                    
                    topics.append(topic)
                
                return {
                    "topics": topics,
                    "total_messages": total_messages,
                    "analysis_period": {
                        "start": start_date.isoformat() if start_date else None,
                        "end": end_date.isoformat() if end_date else None
                    }
                }
                
        except Exception as e:
            logger.error(f"Error getting conversation topics: {e}")
            return {"topics": [], "total_messages": 0}

    async def get_communication_style(
        self,
        contact_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Analyze communication style patterns."""
        try:
            # Normalize the contact identifier
            normalized_id = await self._normalize_contact_identifier(contact_id)
            
            # Build date filter
            date_filters_sql, date_params = build_date_filters_sql(start_date, end_date)
            
            # Get messages for style analysis
            query = f"""
            SELECT 
                message.text,
                message.is_from_me,
                LENGTH(message.text) as length
            FROM message
            JOIN chat_message_join ON message.ROWID = chat_message_join.message_id
            JOIN chat ON chat_message_join.chat_id = chat.ROWID
            JOIN handle ON message.handle_id = handle.ROWID
            WHERE (handle.id = ? OR handle.id LIKE ? OR chat.chat_identifier = ?)
            AND message.text IS NOT NULL AND message.text != ''
            {" AND " + date_filters_sql if date_filters_sql else ""}
            ORDER BY message.date DESC
            LIMIT 500
            """
            
            params = [normalized_id, f"%{normalized_id}%", normalized_id]
            if date_params:
                params.extend(date_params.values())
            
            async with self.get_db_connection() as conn:
                result = await conn.execute(query, params)
                messages = await result.fetchall()
                
                # Analyze style metrics
                my_messages = []
                their_messages = []
                
                for text, is_from_me, length in messages:
                    if is_from_me:
                        my_messages.append((text, length))
                    else:
                        their_messages.append((text, length))
                
                # Calculate basic metrics
                def analyze_style(messages):
                    if not messages:
                        return {
                            "avg_length": 0,
                            "emoji_usage": 0,
                            "punctuation_usage": 0,
                            "capitalization": 0
                        }
                    
                    total_length = sum(length for _, length in messages)
                    emoji_count = 0
                    punctuation_count = 0
                    caps_count = 0
                    
                    for text, _ in messages:
                        # Count emojis (simplified)
                        emoji_count += len(re.findall(r'[😀-🙏🏻-🏿]+', text))
                        # Count punctuation
                        punctuation_count += len(re.findall(r'[!?.,:;]', text))
                        # Count capitalized words
                        caps_count += len(re.findall(r'\b[A-Z][a-z]+\b', text))
                    
                    return {
                        "avg_length": round(total_length / len(messages), 2),
                        "emoji_usage": round(emoji_count / len(messages), 2),
                        "punctuation_usage": round(punctuation_count / len(messages), 2),
                        "capitalization": round(caps_count / len(messages), 2)
                    }
                
                my_style = analyze_style(my_messages)
                their_style = analyze_style(their_messages)
                
                # Determine formality based on metrics
                def get_formality(style):
                    score = 0
                    if style["avg_length"] > 50:
                        score += 2
                    if style["punctuation_usage"] > 2:
                        score += 1
                    if style["emoji_usage"] < 0.5:
                        score += 2
                    if style["capitalization"] > 1:
                        score += 1
                    
                    if score >= 4:
                        return "formal"
                    elif score >= 2:
                        return "semi-formal"
                    else:
                        return "casual"
                
                return {
                    "my_style": {
                        **my_style,
                        "formality": get_formality(my_style),
                        "message_count": len(my_messages)
                    },
                    "their_style": {
                        **their_style,
                        "formality": get_formality(their_style),
                        "message_count": len(their_messages)
                    },
                    "compatibility": {
                        "length_difference": abs(my_style["avg_length"] - their_style["avg_length"]),
                        "emoji_difference": abs(my_style["emoji_usage"] - their_style["emoji_usage"]),
                        "formality_match": my_style.get("formality") == their_style.get("formality")
                    }
                }
                
        except Exception as e:
            logger.error(f"Error analyzing communication style: {e}")
            return {
                "my_style": {},
                "their_style": {},
                "compatibility": {}
            }

    async def get_group_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Get information about a specific group chat."""
        try:
            query = """
            SELECT 
                chat.ROWID,
                chat.chat_identifier,
                chat.display_name,
                chat.group_id,
                COUNT(DISTINCT handle.id) as member_count,
                COUNT(DISTINCT message.ROWID) as message_count,
                MIN(message.date) as first_message_date,
                MAX(message.date) as last_message_date
            FROM chat
            LEFT JOIN chat_message_join ON chat.ROWID = chat_message_join.chat_id
            LEFT JOIN message ON chat_message_join.message_id = message.ROWID
            LEFT JOIN chat_handle_join ON chat.ROWID = chat_handle_join.chat_id
            LEFT JOIN handle ON chat_handle_join.handle_id = handle.ROWID
            WHERE chat.chat_identifier = ? OR chat.ROWID = ?
            GROUP BY chat.ROWID
            """
            
            async with self.get_db_connection() as conn:
                result = await conn.execute(query, (chat_id, chat_id))
                row = await result.fetchone()
                
                if not row:
                    return {
                        "id": chat_id,
                        "chat_identifier": chat_id,
                        "display_name": None,
                        "member_count": 0,
                        "message_count": 0,
                        "first_message_date": None,
                        "last_message_date": None
                    }
                
                return {
                    "id": row[0],
                    "chat_identifier": row[1],
                    "display_name": row[2],
                    "group_id": row[3],
                    "member_count": row[4] or 0,
                    "message_count": row[5] or 0,
                    "first_message_date": convert_apple_timestamp(row[6]) if row[6] else None,
                    "last_message_date": convert_apple_timestamp(row[7]) if row[7] else None
                }
                
        except Exception as e:
            logger.error(f"Error getting group chat info: {e}")
            return {
                "id": chat_id,
                "chat_identifier": chat_id,
                "display_name": None,
                "member_count": 0,
                "message_count": 0
            }

    async def get_group_chat_analytics(
        self,
        chat_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get analytics for a group chat."""
        try:
            # Build date filter
            date_filters_sql, date_params = build_date_filters_sql(start_date, end_date)
            
            # Build WHERE clause with date filters
            where_clause = "(chat.chat_identifier = ? OR chat.ROWID = ?)"
            if date_filters_sql:
                where_clause += f" AND {date_filters_sql}"
            
            # Get message counts and activity
            query = f"""
            SELECT 
                COUNT(DISTINCT message.ROWID) as total_messages,
                COUNT(DISTINCT handle.id) as active_members,
                COUNT(DISTINCT DATE(datetime(message.date/1000000000 + 978307200, 'unixepoch'))) as active_days,
                MIN(message.date) as first_message,
                MAX(message.date) as last_message,
                AVG(LENGTH(message.text)) as avg_message_length
            FROM chat
            JOIN chat_message_join ON chat.ROWID = chat_message_join.chat_id
            JOIN message ON chat_message_join.message_id = message.ROWID
            LEFT JOIN handle ON message.handle_id = handle.ROWID
            WHERE {where_clause}
            """
            
            params = [chat_id, chat_id]
            if date_params:
                params.extend(date_params.values())
            
            async with self.get_db_connection() as conn:
                result = await conn.execute(query, params)
                row = await result.fetchone()
                
                analytics = {
                    "total_messages": row[0] or 0,
                    "active_members": row[1] or 0,
                    "active_days": row[2] or 0,
                    "first_message_date": convert_apple_timestamp(row[3]) if row[3] else None,
                    "last_message_date": convert_apple_timestamp(row[4]) if row[4] else None,
                    "avg_message_length": round(row[5] or 0, 2)
                }
                
                # Get member activity breakdown
                member_query = f"""
                SELECT 
                    handle.id,
                    COUNT(message.ROWID) as message_count,
                    SUM(CASE WHEN message.is_from_me = 1 THEN 1 ELSE 0 END) as sent_count
                FROM chat
                JOIN chat_message_join ON chat.ROWID = chat_message_join.chat_id
                JOIN message ON chat_message_join.message_id = message.ROWID
                LEFT JOIN handle ON message.handle_id = handle.ROWID
                WHERE {where_clause}
                GROUP BY handle.id
                ORDER BY message_count DESC
                """
                
                result = await conn.execute(member_query, params)
                members = await result.fetchall()
                
                member_activity = []
                for member_row in members:
                    if member_row[0]:  # Skip None handles
                        member_activity.append({
                            "member_id": member_row[0],
                            "message_count": member_row[1],
                            "sent_count": member_row[2]
                        })
                
                analytics["member_activity"] = member_activity
                
                return analytics
                
        except Exception as e:
            logger.error(f"Error getting group chat analytics: {e}")
            return {
                "total_messages": 0,
                "active_members": 0,
                "active_days": 0,
                "member_activity": []
            }

    async def get_group_chat_member_participation(
        self,
        chat_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get detailed member participation statistics for a group chat."""
        try:
            # Build date filter
            date_filters_sql, date_params = build_date_filters_sql(start_date, end_date)
            
            # Build WHERE clause with date filters
            where_clause = "(chat.chat_identifier = ? OR chat.ROWID = ?)"
            if date_filters_sql:
                where_clause += f" AND {date_filters_sql}"
            
            # Get member participation details
            query = f"""
            SELECT 
                COALESCE(handle.id, 'Me') as member_id,
                COUNT(message.ROWID) as message_count,
                SUM(CASE WHEN message.is_from_me = 1 THEN 1 ELSE 0 END) as messages_from_me,
                AVG(LENGTH(message.text)) as avg_message_length,
                MIN(message.date) as first_message_date,
                MAX(message.date) as last_message_date,
                SUM(CASE WHEN message.cache_has_attachments = 1 THEN 1 ELSE 0 END) as attachment_count
            FROM chat
            JOIN chat_message_join ON chat.ROWID = chat_message_join.chat_id
            JOIN message ON chat_message_join.message_id = message.ROWID
            LEFT JOIN handle ON message.handle_id = handle.ROWID
            WHERE {where_clause}
            GROUP BY COALESCE(handle.id, 'Me')
            ORDER BY message_count DESC
            """
            
            params = [chat_id, chat_id]
            if date_params:
                params.extend(date_params.values())
            
            async with self.get_db_connection() as conn:
                result = await conn.execute(query, params)
                members = await result.fetchall()
                
                member_data = []
                total_messages = 0
                
                for row in members:
                    member_info = {
                        "member_id": row[0],
                        "message_count": row[1] or 0,
                        "is_me": row[2] > 0,  # If they have messages_from_me > 0, it's the user
                        "avg_message_length": round(row[3] or 0, 2),
                        "first_message_date": convert_apple_timestamp(row[4]) if row[4] else None,
                        "last_message_date": convert_apple_timestamp(row[5]) if row[5] else None,
                        "attachment_count": row[6] or 0
                    }
                    
                    # Calculate participation percentage later
                    total_messages += member_info["message_count"]
                    member_data.append(member_info)
                
                # Add participation percentages
                for member in member_data:
                    member["participation_percentage"] = round(
                        (member["message_count"] / total_messages * 100) if total_messages > 0 else 0, 
                        2
                    )
                
                # Calculate engagement metrics
                active_members = len([m for m in member_data if m["message_count"] > 0])
                
                return {
                    "members": member_data,
                    "total_members": len(member_data),
                    "active_members": active_members,
                    "total_messages": total_messages,
                    "engagement_metrics": {
                        "participation_ratio": round(active_members / len(member_data), 2) if member_data else 0,
                        "messages_per_member": round(total_messages / len(member_data), 2) if member_data else 0
                    }
                }
                
        except Exception as e:
            logger.error(f"Error getting group chat member participation: {e}")
            return {
                "members": [],
                "total_members": 0,
                "active_members": 0,
                "total_messages": 0,
                "engagement_metrics": {}
            }
