#!/usr/bin/env python3
"""
Database Implementation Sample

This file demonstrates how to refactor the AsyncMessagesDB class to 
inherit from the new abstract base class. It is intended as a reference
for the actual implementation and not as production code.
"""

import asyncio
import logging
import os
import traceback
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import aiosqlite

# Import the base class and error type
from src.database.db_base import AsyncMessagesDBBase, DatabaseError

# Import utility functions from db_utils
from src.database.db_utils import (
    build_pagination_info,
    clean_message_text,
    convert_apple_timestamp,
    datetime_to_apple_timestamp,
    format_date_for_display,
)

# Import the Redis cache
# Import the contact resolver
from src.utils.contact_resolver import ContactResolverFactory

# Configure logging
logger = logging.getLogger(__name__)

# Default database path for macOS
HOME = os.path.expanduser("~")
DEFAULT_DB_PATH = Path(f"{HOME}/Library/Messages/chat.db")


class AsyncMessagesDB(AsyncMessagesDBBase):
    """
    An asynchronous class for handling message database operations.
    
    This implementation uses connection pooling, caching, and comprehensive
    error handling for optimal performance.
    """

    def __init__(self, db_path=None, minimal_mode=False, pool_size=10):
        """
        Initialize the database connection.

        Args:
            db_path: Path to the SQLite database file. If None, uses the default macOS path.
            minimal_mode: Start in minimal mode for faster performance with less data.
            pool_size: Size of the connection pool.
        """
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self._lock = asyncio.Lock()
        self._connection_pool = []
        self._busy_connections = set()
        self.initialized = False

        # Connection pool configuration
        self._max_connections = pool_size
        self._connection_timeout = 5.0  # Seconds to wait for a connection
        self._connection_ttl = 300  # Seconds to keep a connection before refreshing
        self._connection_timestamps = {}  # Track when connections were created

        # Performance configuration
        self.minimal_mode = minimal_mode
        self._has_fts5 = False
        self.is_indexed_db = False

        # Cache initialization

        # Contact resolver
        self.contact_resolver = None

    def check_database_file(self):
        """
        Check if the database file is accessible and log its properties.

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
        """
        Initialize the database connection pool asynchronously.

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
                self._connection_timestamps[id(initial_conn)] = asyncio.get_event_loop().time()

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

    @asynccontextmanager
    async def get_db_connection(self):
        """
        Get a database connection from the pool.

        Returns:
            A database connection from the pool

        Raises:
            DatabaseError: If unable to get a connection
        """
        if not self.initialized:
            raise DatabaseError("Database not initialized. Call initialize() first.")

        conn = None

        try:
            # Try to get a connection from the pool
            async with self._lock:
                if self._connection_pool:
                    # Get the oldest connection from the pool
                    conn = self._connection_pool.pop(0)
                    self._busy_connections.add(conn)
                elif len(self._busy_connections) < self._max_connections:
                    # Create a new connection if pool is empty but under max connections
                    uri = f"file:{self.db_path}?mode=ro&immutable=1"
                    conn = await aiosqlite.connect(uri, uri=True)
                    conn.row_factory = aiosqlite.Row
                    self._busy_connections.add(conn)
                    self._connection_timestamps[id(conn)] = asyncio.get_event_loop().time()
                else:
                    # No available connections and at max capacity
                    raise DatabaseError(
                        "No database connections available and at maximum capacity"
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
                self._busy_connections.add(conn)
                self._connection_timestamps[id(conn)] = asyncio.get_event_loop().time()

            yield conn

        finally:
            # Return connection to the pool
            if conn is not None:
                async with self._lock:
                    if conn in self._busy_connections:
                        self._busy_connections.remove(conn)

                        # Check if connection is too old and should be refreshed
                        conn_age = asyncio.get_event_loop().time() - self._connection_timestamps.get(
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
                                self._connection_pool.append(fresh_conn)
                                self._connection_timestamps[id(fresh_conn)] = (
                                    asyncio.get_event_loop().time()
                                )
                            except Exception as e:
                                logger.error(f"Error creating fresh connection: {e}")
                        else:
                            # Return healthy connection to the pool
                            self._connection_pool.append(conn)

    async def get_contacts(self, limit=100, offset=0, minimal=None) -> Dict[str, Any]:
        """
        Get a list of all contacts from the iMessage database.

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

                    # Get contact info
                    contact_info = (
                        await self.contact_resolver.get_contact_by_identifier(
                            phone_number
                        )
                    )
                    display_name = (
                        contact_info.get("name") if contact_info else phone_number
                    )

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
                            # Convert Apple's timestamp to datetime
                            date_val = row[0]
                            if date_val:
                                date_obj = convert_apple_timestamp(date_val)
                                last_message_date = date_obj.isoformat() if date_obj else None

                    contacts.append(
                        {
                            "phone_number": phone_number,
                            "display_name": display_name,
                            "service": service,
                            "handle_id": handle_id,
                            "message_count": message_count,
                            "last_message_date": last_message_date,
                        }
                    )

                # Calculate pagination information
                pagination = build_pagination_info(total_count, offset // limit + 1, limit)

                return {
                    "contacts": contacts,
                    "total": total_count,
                    "pagination": pagination,
                }

        except Exception as e:
            logger.error(f"Error getting contacts: {e}")
            logger.error(traceback.format_exc())
            return {"error": str(e), "contacts": [], "total": 0}

    async def get_group_chats(self, limit=20, offset=0) -> Dict[str, Any]:
        """
        Get a list of all group chats.

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
                    name = row["name"] or f"Group Chat {chat_id}"
                    guid = row["guid"]

                    # Count participants
                    participants_query = """
                    SELECT COUNT(*) as count
                    FROM chat_handle_join
                    WHERE chat_id = ?
                    """
                    cursor = await conn.execute(participants_query, (chat_id,))
                    p_row = await cursor.fetchone()
                    participant_count = p_row[0] if p_row else 0

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
                        date_obj = convert_apple_timestamp(d_row[0])
                        last_message_date = date_obj.isoformat() if date_obj else None

                    chats.append(
                        {
                            "chat_id": chat_id,
                            "name": name,
                            "guid": guid,
                            "participant_count": participant_count,
                            "message_count": message_count,
                            "last_message_date": last_message_date,
                        }
                    )

                # Calculate pagination information
                pagination = build_pagination_info(total_count, offset // limit + 1, limit)

                return {
                    "chats": chats,
                    "total": total_count,
                    "pagination": pagination,
                }

        except Exception as e:
            logger.error(f"Error getting group chats: {e}")
            logger.error(traceback.format_exc())
            return {"error": str(e), "chats": [], "total": 0}

    async def get_chat_by_id(self, chat_id: Union[str, int]) -> Dict[str, Any]:
        """
        Get information about a specific chat by its ID.

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

    async def validate_schema_version(self) -> bool:
        """
        Check if the database schema is compatible with this application.

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

    async def optimize_database(self) -> Dict[str, Any]:
        """
        Perform database optimizations.

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

    async def get_messages_from_chat(
        self,
        chat_id: Union[str, int],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        search_term: Optional[str] = None,
        page: int = 1,
        page_size: int = 50,
        order_by: str = "date DESC"
    ) -> Dict[str, Any]:
        """
        Get messages from a specific chat.
        
        Args:
            chat_id: The ID of the chat
            start_date: Optional start date for filtering
            end_date: Optional end date for filtering
            search_term: Optional search term for filtering
            page: Page number for pagination
            page_size: Number of messages per page
            order_by: Field and direction to sort by
            
        Returns:
            Dictionary with messages and metadata
        """
        try:
            # Calculate the offset based on page and page_size
            offset = (page - 1) * page_size

            # Build query base
            query = """
            SELECT 
                message.ROWID as message_id,
                message.guid,
                message.text,
                message.date,
                message.is_from_me,
                message.handle_id,
                handle.id as phone_number,
                handle.service
            FROM 
                message
            JOIN 
                chat_message_join ON message.ROWID = chat_message_join.message_id
            LEFT JOIN
                handle ON message.handle_id = handle.ROWID
            WHERE 
                chat_message_join.chat_id = ?
            """

            params = [chat_id]

            # Add date filters if specified
            if start_date:
                start_timestamp = datetime_to_apple_timestamp(start_date)
                if start_timestamp:
                    query += " AND message.date >= ?"
                    params.append(start_timestamp)

            if end_date:
                end_timestamp = datetime_to_apple_timestamp(end_date)
                if end_timestamp:
                    query += " AND message.date <= ?"
                    params.append(end_timestamp)

            # Add search term filter if specified
            if search_term:
                if self._has_fts5:
                    # Use FTS5 if available
                    query += """
                    AND message.ROWID IN (
                        SELECT rowid FROM message_fts
                        WHERE message_fts MATCH ?
                    )
                    """
                    params.append(search_term)
                else:
                    # Fallback to LIKE search
                    query += " AND message.text LIKE ?"
                    params.append(f"%{search_term}%")

            # Add order by clause
            query += f" ORDER BY message.{order_by}"

            # Add limit and offset
            query += " LIMIT ? OFFSET ?"
            params.extend([page_size, offset])

            # Count query for total results
            count_query = """
            SELECT COUNT(*) as count
            FROM message
            JOIN chat_message_join ON message.ROWID = chat_message_join.message_id
            WHERE chat_message_join.chat_id = ?
            """

            count_params = [chat_id]

            # Add the same filters to count query
            if start_date:
                start_timestamp = datetime_to_apple_timestamp(start_date)
                if start_timestamp:
                    count_query += " AND message.date >= ?"
                    count_params.append(start_timestamp)

            if end_date:
                end_timestamp = datetime_to_apple_timestamp(end_date)
                if end_timestamp:
                    count_query += " AND message.date <= ?"
                    count_params.append(end_timestamp)

            if search_term:
                if self._has_fts5:
                    count_query += """
                    AND message.ROWID IN (
                        SELECT rowid FROM message_fts
                        WHERE message_fts MATCH ?
                    )
                    """
                    count_params.append(search_term)
                else:
                    count_query += " AND message.text LIKE ?"
                    count_params.append(f"%{search_term}%")

            # Execute queries
            async with self.get_db_connection() as conn:
                # Get total count
                cursor = await conn.execute(count_query, count_params)
                row = await cursor.fetchone()
                total_count = row[0] if row else 0

                # Get messages
                cursor = await conn.execute(query, params)
                rows = await cursor.fetchall()

                messages = []
                for row in rows:
                    # Convert Apple timestamp to datetime
                    date_obj = convert_apple_timestamp(row["date"])

                    # Clean message text
                    text = clean_message_text(row["text"])

                    # Get contact display name
                    phone_number = row["phone_number"]
                    display_name = phone_number

                    # Only look up contact info if not from me
                    if not row["is_from_me"] and phone_number:
                        contact_info = await self.contact_resolver.get_contact_by_identifier(phone_number)
                        display_name = contact_info.get("name", phone_number) if contact_info else phone_number

                    messages.append({
                        "message_id": row["message_id"],
                        "guid": row["guid"],
                        "text": text,
                        "date": date_obj.isoformat() if date_obj else None,
                        "date_obj": date_obj,
                        "is_from_me": bool(row["is_from_me"]),
                        "phone_number": phone_number,
                        "display_name": display_name,
                        "service": row["service"],
                        "formatted_date": format_date_for_display(date_obj) if date_obj else None,
                    })

                # Get chat information
                chat_info = await self.get_chat_by_id(chat_id)

                # Calculate pagination information
                pagination = build_pagination_info(total_count, page, page_size)

                return {
                    "chat": chat_info,
                    "messages": messages,
                    "total_count": total_count,
                    "pagination": pagination,
                    "filters": {
                        "start_date": start_date.isoformat() if start_date else None,
                        "end_date": end_date.isoformat() if end_date else None,
                        "search_term": search_term,
                    },
                }

        except Exception as e:
            logger.error(f"Error getting messages from chat {chat_id}: {e}")
            logger.error(traceback.format_exc())
            return {
                "error": str(e),
                "messages": [],
                "total_count": 0,
                "chat": {},
            }

    async def get_messages_from_contact(
        self,
        contact_id: Union[str, int],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        search_term: Optional[str] = None,
        page: int = 1,
        page_size: int = 50,
        order_by: str = "date DESC"
    ) -> Dict[str, Any]:
        """
        Get messages from a specific contact.
        
        Args:
            contact_id: The ID or phone number of the contact
            start_date: Optional start date for filtering
            end_date: Optional end date for filtering
            search_term: Optional search term for filtering
            page: Page number for pagination
            page_size: Number of messages per page
            order_by: Field and direction to sort by
            
        Returns:
            Dictionary with messages and metadata
        """
        try:
            # Calculate the offset based on page and page_size
            offset = (page - 1) * page_size

            # Handle case where contact_id could be a phone number or a handle_id
            handle_query = "SELECT ROWID, id, service FROM handle WHERE "
            handle_params = []

            if isinstance(contact_id, int) or (isinstance(contact_id, str) and contact_id.isdigit()):
                # This is a handle_id
                handle_query += "ROWID = ?"
                handle_params.append(contact_id)
            else:
                # This is a phone number or email
                handle_query += "id = ?"
                handle_params.append(contact_id)

            # Build query base
            query = """
            SELECT 
                message.ROWID as message_id,
                message.guid,
                message.text,
                message.date,
                message.is_from_me,
                message.handle_id,
                handle.id as phone_number,
                handle.service
            FROM 
                message
            JOIN 
                handle ON message.handle_id = handle.ROWID
            WHERE 
                handle.id = ?
            """

            params = [contact_id]

            # Add date filters if specified
            if start_date:
                start_timestamp = datetime_to_apple_timestamp(start_date)
                if start_timestamp:
                    query += " AND message.date >= ?"
                    params.append(start_timestamp)

            if end_date:
                end_timestamp = datetime_to_apple_timestamp(end_date)
                if end_timestamp:
                    query += " AND message.date <= ?"
                    params.append(end_timestamp)

            # Add search term filter if specified
            if search_term:
                if self._has_fts5:
                    # Use FTS5 if available
                    query += """
                    AND message.ROWID IN (
                        SELECT rowid FROM message_fts
                        WHERE message_fts MATCH ?
                    )
                    """
                    params.append(search_term)
                else:
                    # Fallback to LIKE search
                    query += " AND message.text LIKE ?"
                    params.append(f"%{search_term}%")

            # Add order by clause
            query += f" ORDER BY message.{order_by}"

            # Add limit and offset
            query += " LIMIT ? OFFSET ?"
            params.extend([page_size, offset])

            # Count query for total results
            count_query = """
            SELECT COUNT(*) as count
            FROM message
            JOIN handle ON message.handle_id = handle.ROWID
            WHERE handle.id = ?
            """

            count_params = [contact_id]

            # Add the same filters to count query
            if start_date:
                start_timestamp = datetime_to_apple_timestamp(start_date)
                if start_timestamp:
                    count_query += " AND message.date >= ?"
                    count_params.append(start_timestamp)

            if end_date:
                end_timestamp = datetime_to_apple_timestamp(end_date)
                if end_timestamp:
                    count_query += " AND message.date <= ?"
                    count_params.append(end_timestamp)

            if search_term:
                if self._has_fts5:
                    count_query += """
                    AND message.ROWID IN (
                        SELECT rowid FROM message_fts
                        WHERE message_fts MATCH ?
                    )
                    """
                    count_params.append(search_term)
                else:
                    count_query += " AND message.text LIKE ?"
                    count_params.append(f"%{search_term}%")

            # Execute queries
            async with self.get_db_connection() as conn:
                # First get the handle
                cursor = await conn.execute(handle_query, handle_params)
                handle_row = await cursor.fetchone()

                if not handle_row:
                    return {
                        "error": f"Contact with ID {contact_id} not found",
                        "messages": [],
                        "total_count": 0,
                        "contact": {},
                    }

                handle_id = handle_row["ROWID"]
                contact_info = {
                    "id": handle_row["id"],
                    "service": handle_row["service"],
                    "handle_id": handle_id,
                }

                # Update queries to use handle_id
                query = query.replace("handle.id = ?", "message.handle_id = ?")
                count_query = count_query.replace("handle.id = ?", "message.handle_id = ?")
                params[0] = handle_id
                count_params[0] = handle_id

                # Get total count
                cursor = await conn.execute(count_query, count_params)
                row = await cursor.fetchone()
                total_count = row[0] if row else 0

                # Get messages
                cursor = await conn.execute(query, params)
                rows = await cursor.fetchall()

                # Get contact display name
                contact_name = contact_info["id"]
                resolved_contact = await self.contact_resolver.get_contact_by_identifier(contact_info["id"])
                if resolved_contact:
                    contact_name = resolved_contact.get("name", contact_info["id"])

                contact_info["display_name"] = contact_name

                messages = []
                for row in rows:
                    # Convert Apple timestamp to datetime
                    date_obj = convert_apple_timestamp(row["date"])

                    # Clean message text
                    text = clean_message_text(row["text"])

                    messages.append({
                        "message_id": row["message_id"],
                        "guid": row["guid"],
                        "text": text,
                        "date": date_obj.isoformat() if date_obj else None,
                        "date_obj": date_obj,
                        "is_from_me": bool(row["is_from_me"]),
                        "phone_number": row["phone_number"],
                        "display_name": contact_name,
                        "service": row["service"],
                        "formatted_date": format_date_for_display(date_obj) if date_obj else None,
                    })

                # Calculate pagination information
                pagination = build_pagination_info(total_count, page, page_size)

                return {
                    "contact": contact_info,
                    "messages": messages,
                    "total_count": total_count,
                    "pagination": pagination,
                    "filters": {
                        "start_date": start_date.isoformat() if start_date else None,
                        "end_date": end_date.isoformat() if end_date else None,
                        "search_term": search_term,
                    },
                }

        except Exception as e:
            logger.error(f"Error getting messages from contact {contact_id}: {e}")
            logger.error(traceback.format_exc())
            return {
                "error": str(e),
                "messages": [],
                "total_count": 0,
                "contact": {},
            }

    async def search_messages(
        self,
        query: str,
        contact_id: Optional[Union[str, int]] = None,
        chat_id: Optional[Union[str, int]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        page: int = 1,
        page_size: int = 20
    ) -> Dict[str, Any]:
        """
        Search messages across all conversations or a specific contact/chat.
        
        Args:
            query: Search query
            contact_id: Optional contact to restrict search to
            chat_id: Optional chat to restrict search to
            start_date: Optional start date for filtering
            end_date: Optional end date for filtering
            page: Page number for pagination
            page_size: Number of messages per page
            
        Returns:
            Dictionary with search results
        """
        try:
            # Calculate the offset based on page and page_size
            offset = (page - 1) * page_size

            # Build query base
            base_query = """
            SELECT 
                message.ROWID as message_id,
                message.guid,
                message.text,
                message.date,
                message.is_from_me,
                message.handle_id,
                handle.id as phone_number,
                handle.service
            FROM 
                message
            LEFT JOIN
                handle ON message.handle_id = handle.ROWID
            """

            # Add joins and where clauses based on filters
            where_clauses = []
            params = []

            # Add search term filter
            if self._has_fts5:
                # Use FTS5 if available
                where_clauses.append("""
                    message.ROWID IN (
                        SELECT rowid FROM message_fts
                        WHERE message_fts MATCH ?
                    )
                """)
                params.append(query)
            else:
                # Fallback to LIKE search
                where_clauses.append("message.text LIKE ?")
                params.append(f"%{query}%")

            # Add contact filter if specified
            if contact_id:
                if isinstance(contact_id, int) or (isinstance(contact_id, str) and contact_id.isdigit()):
                    # This is a handle_id
                    where_clauses.append("message.handle_id = ?")
                    params.append(contact_id)
                else:
                    # This is a phone number or email
                    base_query += "JOIN handle h2 ON message.handle_id = h2.ROWID"
                    where_clauses.append("h2.id = ?")
                    params.append(contact_id)

            # Add chat filter if specified
            if chat_id:
                base_query += "JOIN chat_message_join ON message.ROWID = chat_message_join.message_id"
                where_clauses.append("chat_message_join.chat_id = ?")
                params.append(chat_id)

            # Add date filters if specified
            if start_date:
                start_timestamp = datetime_to_apple_timestamp(start_date)
                if start_timestamp:
                    where_clauses.append("message.date >= ?")
                    params.append(start_timestamp)

            if end_date:
                end_timestamp = datetime_to_apple_timestamp(end_date)
                if end_timestamp:
                    where_clauses.append("message.date <= ?")
                    params.append(end_timestamp)

            # Combine where clauses
            if where_clauses:
                base_query += " WHERE " + " AND ".join(where_clauses)

            # Finalize query
            query = base_query + " ORDER BY message.date DESC LIMIT ? OFFSET ?"
            params.extend([page_size, offset])

            # Count query for total results
            count_query = "SELECT COUNT(*) as count FROM (" + base_query + ")"

            # Execute queries
            async with self.get_db_connection() as conn:
                # Get total count
                cursor = await conn.execute(count_query, params[:-2] if params else [])
                row = await cursor.fetchone()
                total_count = row[0] if row else 0

                # Get messages
                cursor = await conn.execute(query, params)
                rows = await cursor.fetchall()

                messages = []
                for row in rows:
                    # Convert Apple timestamp to datetime
                    date_obj = convert_apple_timestamp(row["date"])

                    # Clean message text
                    text = clean_message_text(row["text"])

                    # Get contact display name
                    phone_number = row["phone_number"]
                    display_name = phone_number

                    if not row["is_from_me"] and phone_number:
                        contact_info = await self.contact_resolver.get_contact_by_identifier(phone_number)
                        display_name = contact_info.get("name", phone_number) if contact_info else phone_number

                    messages.append({
                        "message_id": row["message_id"],
                        "guid": row["guid"],
                        "text": text,
                        "date": date_obj.isoformat() if date_obj else None,
                        "date_obj": date_obj,
                        "is_from_me": bool(row["is_from_me"]),
                        "phone_number": phone_number,
                        "display_name": display_name,
                        "service": row["service"],
                        "formatted_date": format_date_for_display(date_obj) if date_obj else None,
                        "highlighted_text": text.replace(
                            query, f"**{query}**"
                        ) if query in text else text,
                    })

                # Calculate pagination information
                pagination = build_pagination_info(total_count, page, page_size)

                return {
                    "messages": messages,
                    "total_count": total_count,
                    "pagination": pagination,
                    "query": query,
                    "filters": {
                        "contact_id": contact_id,
                        "chat_id": chat_id,
                        "start_date": start_date.isoformat() if start_date else None,
                        "end_date": end_date.isoformat() if end_date else None,
                    },
                }

        except Exception as e:
            logger.error(f"Error searching messages: {e}")
            logger.error(traceback.format_exc())
            return {
                "error": str(e),
                "messages": [],
                "total_count": 0,
                "query": query,
            }

    async def get_contact_by_phone_or_email(self, identifier: str) -> Dict[str, Any]:
        """
        Get contact information by phone number or email.
        
        Args:
            identifier: Phone number or email to look up
            
        Returns:
            Dictionary with contact information
        """
        try:
            query = """
            SELECT 
                handle.ROWID as handle_id,
                handle.id,
                handle.service
            FROM 
                handle
            WHERE 
                handle.id = ?
            """

            async with self.get_db_connection() as conn:
                cursor = await conn.execute(query, (identifier,))
                row = await cursor.fetchone()

                if not row:
                    return {}

                contact_dict = dict(row)

                # Get message count
                count_query = """
                SELECT COUNT(*) as count
                FROM message
                WHERE handle_id = ?
                """
                cursor = await conn.execute(count_query, (contact_dict["handle_id"],))
                count_row = await cursor.fetchone()
                contact_dict["message_count"] = count_row[0] if count_row else 0

                # Get last message date
                date_query = """
                SELECT date
                FROM message
                WHERE handle_id = ?
                ORDER BY date DESC
                LIMIT 1
                """
                cursor = await conn.execute(date_query, (contact_dict["handle_id"],))
                date_row = await cursor.fetchone()
                if date_row and date_row[0]:
                    date_obj = convert_apple_timestamp(date_row[0])
                    contact_dict["last_message_date"] = date_obj.isoformat() if date_obj else None

                # Get contact info from resolver
                resolved = await self.contact_resolver.get_contact_by_identifier(identifier)
                if resolved:
                    contact_dict.update(resolved)

                return contact_dict

        except Exception as e:
            logger.error(f"Error getting contact by identifier {identifier}: {e}")
            logger.error(traceback.format_exc())
            return {}

    async def should_use_sharding(self) -> bool:
        """
        Determine if database sharding should be used based on database size.
        
        Returns:
            True if sharding is recommended, False otherwise
        """
        # If no database path, can't determine
        if not self.db_path or not os.path.exists(self.db_path):
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
            return False
