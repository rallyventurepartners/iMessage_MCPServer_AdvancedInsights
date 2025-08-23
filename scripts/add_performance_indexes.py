#!/usr/bin/env python3
"""
Add performance indexes to iMessage database.

This script adds missing indexes that significantly improve query performance,
particularly for the is_from_me field which is frequently used in filters.

Usage:
    python add_performance_indexes.py  # Add indexes to default database
    python add_performance_indexes.py --db-path /path/to/chat.db  # Specific database
    python add_performance_indexes.py --dry-run  # Show what would be done
"""

import argparse
import logging
import os
import sqlite3
import sys
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default paths
HOME = os.path.expanduser("~")
DEFAULT_DB_PATH = os.path.join(HOME, "Library/Messages/chat.db")

# Indexes to create
PERFORMANCE_INDEXES = [
    {
        "name": "idx_message_is_from_me",
        "table": "message",
        "columns": "(is_from_me)",
        "description": "Index on is_from_me for filtering sent/received messages"
    },
    {
        "name": "idx_message_is_from_me_date",
        "table": "message",
        "columns": "(is_from_me, date DESC)",
        "description": "Composite index for filtering by sender and sorting by date"
    },
    {
        "name": "idx_message_text_prefix",
        "table": "message",
        "columns": "(text)",
        "where": "WHERE text IS NOT NULL",
        "description": "Prefix index for text searches (first 100 chars)"
    },
    {
        "name": "idx_handle_service",
        "table": "handle",
        "columns": "(service)",
        "description": "Index on service type (iMessage, SMS)"
    },
    {
        "name": "idx_chat_guid",
        "table": "chat",
        "columns": "(guid)",
        "description": "Index on chat GUID for faster lookups"
    }
]


def check_existing_indexes(conn):
    """Check which indexes already exist."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
    )
    existing = {row[0] for row in cursor.fetchall()}
    cursor.close()
    return existing


def analyze_table_sizes(conn):
    """Get row counts for tables to understand database size."""
    cursor = conn.cursor()
    tables = ["message", "handle", "chat", "attachment"]
    sizes = {}

    for table in tables:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            sizes[table] = count
            logger.info(f"Table {table}: {count:,} rows")
        except sqlite3.Error:
            logger.warning(f"Could not count rows in table {table}")

    cursor.close()
    return sizes


def create_index(conn, index_info, dry_run=False):
    """Create a single index."""
    name = index_info["name"]
    table = index_info["table"]
    columns = index_info["columns"]
    where = index_info.get("where", "")
    description = index_info["description"]

    sql = f"CREATE INDEX IF NOT EXISTS {name} ON {table} {columns} {where}"

    if dry_run:
        logger.info(f"Would create: {sql}")
        return True

    try:
        logger.info(f"Creating index {name}: {description}")
        start_time = time.time()

        cursor = conn.cursor()
        cursor.execute(sql)
        conn.commit()
        cursor.close()

        elapsed = time.time() - start_time
        logger.info(f"Created index {name} in {elapsed:.2f} seconds")
        return True

    except sqlite3.Error as e:
        logger.error(f"Error creating index {name}: {e}")
        return False


def optimize_database(conn, dry_run=False):
    """Run ANALYZE to update SQLite statistics."""
    if dry_run:
        logger.info("Would run: ANALYZE")
        return

    try:
        logger.info("Running ANALYZE to update database statistics...")
        start_time = time.time()

        cursor = conn.cursor()
        cursor.execute("ANALYZE")
        conn.commit()
        cursor.close()

        elapsed = time.time() - start_time
        logger.info(f"ANALYZE completed in {elapsed:.2f} seconds")

    except sqlite3.Error as e:
        logger.error(f"Error running ANALYZE: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Add performance indexes to iMessage database"
    )
    parser.add_argument(
        "--db-path",
        default=DEFAULT_DB_PATH,
        help="Path to the chat.db file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--skip-analyze",
        action="store_true",
        help="Skip running ANALYZE after creating indexes"
    )

    args = parser.parse_args()

    # Check if database exists
    if not os.path.exists(args.db_path):
        logger.error(f"Database not found: {args.db_path}")
        sys.exit(1)

    # Check if it's the system database
    if args.db_path == DEFAULT_DB_PATH and not args.dry_run:
        logger.warning("You're about to modify the system iMessage database.")
        logger.warning("It's recommended to work on a copy instead.")
        response = input("Continue anyway? (yes/no): ")
        if response.lower() != "yes":
            logger.info("Aborted.")
            sys.exit(0)

    # Connect to database
    try:
        conn = sqlite3.connect(args.db_path)
        logger.info(f"Connected to database: {args.db_path}")
    except sqlite3.Error as e:
        logger.error(f"Error connecting to database: {e}")
        sys.exit(1)

    try:
        # Analyze current state
        logger.info("\nAnalyzing database...")
        sizes = analyze_table_sizes(conn)
        existing_indexes = check_existing_indexes(conn)

        logger.info(f"\nFound {len(existing_indexes)} existing indexes")

        # Determine which indexes to create
        indexes_to_create = []
        for index in PERFORMANCE_INDEXES:
            if index["name"] not in existing_indexes:
                indexes_to_create.append(index)
            else:
                logger.info(f"Index {index['name']} already exists")

        if not indexes_to_create:
            logger.info("\nAll performance indexes already exist!")
            return

        logger.info(f"\nWill create {len(indexes_to_create)} new indexes")

        # Create indexes
        created = 0
        for index in indexes_to_create:
            if create_index(conn, index, args.dry_run):
                created += 1

        logger.info(f"\nCreated {created} indexes")

        # Run ANALYZE
        if not args.skip_analyze and created > 0:
            optimize_database(conn, args.dry_run)

        logger.info("\nIndex creation complete!")

        # Show final state
        if not args.dry_run:
            existing_indexes = check_existing_indexes(conn)
            logger.info(f"Total indexes now: {len(existing_indexes)}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
