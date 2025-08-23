"""
Database access layer with read-only enforcement and performance optimizations.

This module provides safe, read-only access to the iMessage database with
proper error handling, query optimization, and index validation.
"""

import asyncio
import logging
import sqlite3
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Base exception for database operations."""

    pass


class ReadOnlyDatabase:
    """Read-only database connection manager for iMessage data."""

    def __init__(self, db_path: Union[str, Path], timeout: int = 30):
        """Initialize database connection."""
        self.db_path = Path(db_path).expanduser()
        self.timeout = timeout
        self._connection: Optional[sqlite3.Connection] = None
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize database connection with read-only mode."""
        if not self.db_path.exists():
            raise DatabaseError(f"Database not found: {self.db_path}")

        try:
            # For testing with temp databases, we need to handle non-readonly
            if str(self.db_path).startswith("/var/folders/") or str(self.db_path).startswith(
                "/tmp/"
            ):
                # Temporary database - open normally
                self._connection = sqlite3.connect(
                    str(self.db_path),
                    timeout=self.timeout,
                    check_same_thread=False,
                )
            else:
                # Open in read-only mode using URI
                self._connection = sqlite3.connect(
                    f"file:{self.db_path}?mode=ro",
                    uri=True,
                    timeout=self.timeout,
                    check_same_thread=False,
                )

            # Enable read-only pragma
            self._connection.execute("PRAGMA query_only = ON")

            # Optimize for read performance
            self._connection.execute("PRAGMA cache_size = 10000")
            self._connection.execute("PRAGMA temp_store = MEMORY")

            # Verify read-only status
            cursor = self._connection.execute("PRAGMA query_only")
            if cursor.fetchone()[0] != 1:
                raise DatabaseError("Failed to enforce read-only mode")

            logger.info(f"Database initialized in read-only mode: {self.db_path}")

        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to initialize database: {e}")

    async def close(self) -> None:
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
            logger.info("Database connection closed")

    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[sqlite3.Connection, None]:
        """Get database connection with async context manager."""
        async with self._lock:
            if not self._connection:
                await self.initialize()
            yield self._connection

    async def execute_query(
        self, query: str, params: Optional[Tuple[Any, ...]] = None, fetch_all: bool = True
    ) -> Union[List[Dict[str, Any]], None]:
        """Execute a read-only query with proper error handling."""
        async with self.get_connection() as conn:
            try:
                cursor = conn.execute(query, params or ())

                if fetch_all:
                    rows = cursor.fetchall()
                    # Convert to dictionaries
                    columns = [desc[0] for desc in cursor.description] if cursor.description else []
                    return [dict(zip(columns, row)) for row in rows]
                else:
                    return None

            except sqlite3.Error as e:
                logger.error(f"Query execution failed: {e}")
                raise DatabaseError(f"Query failed: {e}")

    async def execute(self, query: str, params: Optional[List[Any]] = None):
        """Execute query and return cursor-like object for compatibility."""

        class CursorResult:
            def __init__(self, data):
                self._data = data
                self._index = 0

            async def fetchall(self):
                # Convert dict results back to tuples for compatibility
                if self._data and isinstance(self._data[0], dict):
                    keys = list(self._data[0].keys())
                    return [tuple(row[k] for k in keys) for row in self._data]
                return self._data

            async def fetchone(self):
                if self._index < len(self._data):
                    row = self._data[self._index]
                    self._index += 1
                    if isinstance(row, dict):
                        return tuple(row.values())
                    return row
                return None

        result = await self.execute_query(query, tuple(params) if params else None, fetch_all=True)
        return CursorResult(result or [])

    async def check_schema(self) -> Dict[str, Any]:
        """Validate database schema and return available tables."""
        query = """
        SELECT name, type 
        FROM sqlite_master 
        WHERE type IN ('table', 'view') 
        AND name NOT LIKE 'sqlite_%'
        ORDER BY type, name
        """

        tables = await self.execute_query(query)

        # Check for required tables
        table_names = {t["name"] for t in tables if t["type"] == "table"}
        required_tables = {"message", "handle", "chat", "chat_message_join"}
        missing = required_tables - table_names

        return {
            "tables": list(table_names),
            "views": [t["name"] for t in tables if t["type"] == "view"],
            "missing_required": list(missing),
            "schema_valid": len(missing) == 0,
        }

    async def check_indices(self) -> Dict[str, Any]:
        """Check database indices for performance optimization."""
        query = """
        SELECT name, tbl_name, sql 
        FROM sqlite_master 
        WHERE type = 'index' 
        AND name NOT LIKE 'sqlite_%'
        """

        indices = await self.execute_query(query)

        # Check for important indices
        important_indices = {
            "message": ["date", "handle_id"],
            "chat_message_join": ["chat_id", "message_id"],
            "handle": ["id"],
        }

        index_map = {}
        for idx in indices:
            table = idx["tbl_name"]
            if table not in index_map:
                index_map[table] = []
            index_map[table].append(idx["name"])

        recommendations = []
        for table, columns in important_indices.items():
            if table not in index_map:
                recommendations.append(f"Consider adding indices on {table} for columns: {columns}")

        return {
            "total_indices": len(indices),
            "by_table": index_map,
            "recommendations": recommendations,
        }

    async def get_db_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats = {}

        # Get SQLite version
        version_query = "SELECT sqlite_version()"
        version_result = await self.execute_query(version_query)
        stats["sqlite_version"] = (
            version_result[0]["sqlite_version()"] if version_result else "unknown"
        )

        # Get database size
        stats["size_bytes"] = self.db_path.stat().st_size if self.db_path.exists() else 0
        stats["size_mb"] = round(stats["size_bytes"] / (1024 * 1024), 2)

        # Get message count
        try:
            count_query = "SELECT COUNT(*) as count FROM message"
            count_result = await self.execute_query(count_query)
            stats["message_count"] = count_result[0]["count"] if count_result else 0
        except DatabaseError:
            stats["message_count"] = "unknown"

        # Get date range
        try:
            date_query = """
            SELECT 
                MIN(date/1000000000 + 978307200) as min_date,
                MAX(date/1000000000 + 978307200) as max_date
            FROM message
            WHERE date IS NOT NULL
            """
            date_result = await self.execute_query(date_query)
            if date_result and date_result[0]["min_date"]:
                stats["date_range"] = {
                    "start": datetime.fromtimestamp(date_result[0]["min_date"]).isoformat(),
                    "end": datetime.fromtimestamp(date_result[0]["max_date"]).isoformat(),
                }
        except DatabaseError:
            stats["date_range"] = None

        return stats


# Query helpers for common operations


async def get_contact_messages_query(
    contact_id: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = 100,
) -> str:
    """Build query for getting messages from a contact."""
    # Convert dates to iMessage format (nanoseconds since 2001-01-01)
    params = []
    where_clauses = ["h.id = ?"]
    params.append(contact_id)

    if start_date:
        timestamp = int((start_date.timestamp() - 978307200) * 1e9)
        where_clauses.append("m.date >= ?")
        params.append(timestamp)

    if end_date:
        timestamp = int((end_date.timestamp() - 978307200) * 1e9)
        where_clauses.append("m.date <= ?")
        params.append(timestamp)

    query = f"""
    SELECT 
        m.ROWID as message_id,
        m.text,
        m.date/1000000000 + 978307200 as timestamp,
        m.is_from_me,
        h.id as handle_id
    FROM message m
    JOIN handle h ON m.handle_id = h.ROWID
    WHERE {' AND '.join(where_clauses)}
    ORDER BY m.date DESC
    LIMIT {limit}
    """

    return query, params


async def get_group_messages_query(
    chat_id: Union[str, int],
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = 100,
) -> Tuple[str, List[Any]]:
    """Build query for getting messages from a group chat."""
    params = []
    where_clauses = ["c.ROWID = ?"]
    params.append(chat_id)

    if start_date:
        timestamp = int((start_date.timestamp() - 978307200) * 1e9)
        where_clauses.append("m.date >= ?")
        params.append(timestamp)

    if end_date:
        timestamp = int((end_date.timestamp() - 978307200) * 1e9)
        where_clauses.append("m.date <= ?")
        params.append(timestamp)

    query = f"""
    SELECT 
        m.ROWID as message_id,
        m.text,
        m.date/1000000000 + 978307200 as timestamp,
        m.is_from_me,
        h.id as handle_id,
        c.display_name as chat_name
    FROM message m
    JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
    JOIN chat c ON cmj.chat_id = c.ROWID
    LEFT JOIN handle h ON m.handle_id = h.ROWID
    WHERE {' AND '.join(where_clauses)}
    ORDER BY m.date DESC
    LIMIT {limit}
    """

    return query, params


# Global database instance
_db: Optional[ReadOnlyDatabase] = None


async def get_database(db_path: Optional[str] = None) -> ReadOnlyDatabase:
    """Get or create database instance."""
    global _db

    # If a specific path is provided, create a new instance
    if db_path:
        db = ReadOnlyDatabase(db_path, timeout=30)
        await db.initialize()
        return db

    # Otherwise use the global instance
    if _db is None:
        from .config import get_config

        config = get_config()
        path = config.database.path
        _db = ReadOnlyDatabase(path, timeout=config.database.timeout_seconds)
        await _db.initialize()

    return _db


async def close_database() -> None:
    """Close database connection."""
    global _db
    if _db:
        await _db.close()
        _db = None
