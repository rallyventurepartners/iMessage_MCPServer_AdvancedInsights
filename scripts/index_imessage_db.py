#!/usr/bin/env python3
"""
iMessage Database Indexing Tool

This tool creates an optimized copy of the iMessage database with additional indexes
for significantly improved query performance. It supports various optimization options
including:

1. Creating standard indexes for faster queries
2. Adding FTS (Full-Text Search) virtual tables for text search
3. Creating materialized views for common analytical queries
4. Optimizing database settings (journal mode, etc.)

Usage:
    python index_imessage_db.py  # Default: create indexed copy with all optimizations
    python index_imessage_db.py --read-only  # Create indexed copy, but set read-only
    python index_imessage_db.py --no-fts  # Skip FTS indexes (faster creation)
    python index_imessage_db.py --analyze  # Only analyze, don't create copy
"""

import argparse
import logging
import os
import shutil
import sqlite3
import sys
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default paths
HOME = os.path.expanduser("~")
DEFAULT_DB_PATH = os.path.join(HOME, "Library/Messages/chat.db")
DEFAULT_OUTPUT_DIR = os.path.join(HOME, ".imessage_insights")
DEFAULT_OUTPUT_NAME = "indexed_chat.db"


def analyze_database(db_path):
    """
    Analyze a database to determine if indexing is needed and what optimizations to apply.

    Args:
        db_path: Path to the database to analyze
    """
    if not os.path.exists(db_path):
        logger.error(f"Database file not found: {db_path}")
        return

    # Get file size
    size_bytes = os.path.getsize(db_path)
    size_mb = size_bytes / (1024 * 1024)
    logger.info(f"Database size: {size_mb:.2f} MB ({size_bytes:,} bytes)")

    # Check if it's a very large database
    if size_bytes > 10 * 1024 * 1024 * 1024:  # > 10 GB
        logger.warning(
            "Database is extremely large (>10 GB). Consider using database sharding instead."
        )
        logger.warning("Run: python shard_large_database.py --analyze")

    # Connect to database and analyze
    try:
        conn = sqlite3.connect(db_path)

        # Check existing indexes
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND "
            "name NOT LIKE 'sqlite_%'"
        )
        existing_indexes = [row[0] for row in cursor.fetchall()]
        logger.info(f"Found {len(existing_indexes)} existing indexes:")
        for idx in existing_indexes:
            logger.info(f"  - {idx}")

        # Check if key indexes exist
        recommended_indexes = [
            "idx_message_date",
            "idx_chat_message_join_chat_id",
            "idx_message_handle_id",
            "idx_attachment_message_join",
        ]

        missing_indexes = [
            idx for idx in recommended_indexes if idx not in existing_indexes
        ]
        if missing_indexes:
            logger.info(f"Missing recommended indexes: {', '.join(missing_indexes)}")

        # Check for FTS
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='message_fts'"
        )
        has_fts = cursor.fetchone() is not None
        logger.info(
            f"FTS (Full-Text Search) for messages: {'Present' if has_fts else 'Missing'}"
        )

        # Check for materialized views
        material_views = ["mv_contact_message_counts", "mv_chat_activity"]
        cursor = conn.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND "
            f"name IN ('{material_views[0]}', '{material_views[1]}')"
        )
        existing_views = [row[0] for row in cursor.fetchall()]
        if existing_views:
            logger.info(f"Found materialized views: {', '.join(existing_views)}")
        else:
            logger.info("No materialized views found")

        # Get basic statistics
        cursor = conn.execute("SELECT COUNT(*) FROM message")
        message_count = cursor.fetchone()[0]
        logger.info(f"Total messages: {message_count:,}")

        cursor = conn.execute("SELECT COUNT(*) FROM chat")
        chat_count = cursor.fetchone()[0]
        logger.info(f"Total chats: {chat_count:,}")

        cursor = conn.execute("SELECT COUNT(*) FROM handle")
        handle_count = cursor.fetchone()[0]
        logger.info(f"Total handles: {handle_count:,}")

        # Check for journal mode
        cursor = conn.execute("PRAGMA journal_mode")
        journal_mode = cursor.fetchone()[0]
        logger.info(f"Journal mode: {journal_mode}")

        # Recommendations
        logger.info("\nRecommendations:")

        if size_bytes > 10 * 1024 * 1024 * 1024:
            logger.info(
                "- Use database sharding for best performance with this large database"
            )
            logger.info("  Run: python shard_large_database.py --interactive")
        else:
            if missing_indexes or not has_fts or not existing_views:
                logger.info(
                    "- Create an optimized copy with all recommended indexes and features:"
                )
                logger.info("  python index_imessage_db.py --read-only")

            if not has_fts and message_count > 10000:
                logger.info("- Enable FTS for faster text searching:")
                logger.info("  python index_imessage_db.py --enable-fts")

        conn.close()

    except Exception as e:
        logger.error(f"Error analyzing database: {e}")


def create_indexed_copy(args):
    """
    Create an optimized copy of the database with indexes.

    Args:
        args: Command-line arguments
    """
    source_path = args.source_db
    output_dir = args.output_dir
    output_path = os.path.join(output_dir, args.output_name)

    logger.info("Creating indexed copy of database:")
    logger.info(f"  Source: {source_path}")
    logger.info(f"  Destination: {output_path}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Check if destination already exists
    if os.path.exists(output_path):
        if args.force:
            logger.warning("Destination file exists, overwriting due to --force flag")
            try:
                os.remove(output_path)
            except Exception as e:
                logger.error(f"Could not remove existing file: {e}")
                return False
        else:
            logger.error("Destination file already exists. Use --force to overwrite")
            return False

    # Copy the database file
    try:
        logger.info(
            "Copying database file (this may take a while for large databases)..."
        )
        start_time = time.time()
        shutil.copy2(source_path, output_path)
        elapsed = time.time() - start_time
        logger.info(f"Database copy completed in {elapsed:.1f} seconds")
    except Exception as e:
        logger.error(f"Error copying database: {e}")
        return False

    # Connect to the copy and create indexes
    try:
        logger.info("Creating indexes and optimizations...")
        conn = sqlite3.connect(output_path)

        # Set optimal pragmas
        logger.info("Setting optimal database pragmas...")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA cache_size = 10000")
        conn.execute("PRAGMA temp_store = MEMORY")

        # Begin transaction for faster processing
        conn.execute("BEGIN TRANSACTION")

        # Create message date index
        logger.info("Creating date index...")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_message_date ON message(date)")

        # Create chat_message_join indexes
        logger.info("Creating chat message join indexes...")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_message_join_chat_id ON chat_message_join(chat_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_message_join_message_id ON chat_message_join(message_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_message_join_combined ON chat_message_join(chat_id, message_id)"
        )

        # Create handle indexes
        logger.info("Creating handle indexes...")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_message_handle_id ON message(handle_id)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_handle_id ON handle(id)")

        # Create attachment indexes
        logger.info("Creating attachment indexes...")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_attachment_message_join ON message_attachment_join(message_id, attachment_id)"
        )

        # Create FTS table if enabled
        if args.enable_fts:
            logger.info("Creating FTS (Full-Text Search) virtual table...")
            try:
                # Check if SQLite has FTS5 support
                cursor = conn.execute("SELECT sqlite_compileoption_used('ENABLE_FTS5')")
                has_fts5 = cursor.fetchone()[0] == 1

                if has_fts5:
                    # Create FTS5 virtual table
                    conn.execute(
                        "CREATE VIRTUAL TABLE IF NOT EXISTS message_fts USING fts5(text, content='message', content_rowid='ROWID')"
                    )

                    # Populate the FTS table
                    logger.info("Populating FTS table (this may take a while)...")
                    conn.execute(
                        "INSERT INTO message_fts(rowid, text) SELECT ROWID, text FROM message WHERE text IS NOT NULL"
                    )
                else:
                    logger.warning(
                        "SQLite FTS5 module not available. Skipping FTS index creation."
                    )
            except Exception as e:
                logger.error(f"Error creating FTS table: {e}")
                # Continue with other optimizations

        # Create materialized views if enabled
        if args.create_views:
            logger.info("Creating materialized views...")

            # Message counts by contact
            conn.execute(
                """
            CREATE TABLE IF NOT EXISTS mv_contact_message_counts AS
            SELECT 
                handle.ROWID as handle_id,
                handle.id as contact_id,
                COUNT(message.ROWID) as message_count,
                MIN(message.date) as first_message_date,
                MAX(message.date) as last_message_date,
                SUM(CASE WHEN message.is_from_me = 1 THEN 1 ELSE 0 END) as sent_count,
                SUM(CASE WHEN message.is_from_me = 0 THEN 1 ELSE 0 END) as received_count
            FROM message
            JOIN handle ON message.handle_id = handle.ROWID
            GROUP BY handle.ROWID
            """
            )

            # Chat activity stats
            conn.execute(
                """
            CREATE TABLE IF NOT EXISTS mv_chat_activity AS
            SELECT 
                chat.ROWID as chat_id,
                chat.display_name,
                COUNT(message.ROWID) as message_count,
                MIN(message.date) as first_message_date,
                MAX(message.date) as last_message_date,
                COUNT(DISTINCT handle.ROWID) as participant_count,
                COUNT(DISTINCT date(message.date/1000000000 + 978307200, 'unixepoch')) as active_days
            FROM chat
            JOIN chat_message_join ON chat.ROWID = chat_message_join.chat_id
            JOIN message ON chat_message_join.message_id = message.ROWID
            LEFT JOIN handle ON message.handle_id = handle.ROWID
            GROUP BY chat.ROWID
            """
            )

        # Commit changes
        conn.commit()

        # Run ANALYZE to gather statistics for query planner
        logger.info("Running ANALYZE to gather statistics...")
        conn.execute("ANALYZE")

        # Make read-only if requested
        if args.read_only:
            logger.info("Setting database to read-only mode...")
            conn.close()
            os.chmod(output_path, 0o444)  # Read-only permission

            # Reopen to check journal mode
            conn = sqlite3.connect(f"file:{output_path}?mode=ro", uri=True)

        # Final statistics
        cursor = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='index'")
        index_count = cursor.fetchone()[0]

        cursor = conn.execute("PRAGMA journal_mode")
        journal_mode = cursor.fetchone()[0]

        # Close connection
        conn.close()

        # Success message
        logger.info("\nOptimized database creation successful!")
        logger.info(f"Total indexes: {index_count}")
        logger.info(f"Journal mode: {journal_mode}")
        logger.info(f"FTS enabled: {args.enable_fts}")
        logger.info(f"Materialized views: {args.create_views}")
        logger.info(f"Read-only: {args.read_only}")

        # Provide command line for using the new database
        logger.info("\nTo use the optimized database with the MCP server, run:")
        logger.info(f'  DB_PATH="{output_path}" python mcp_server_modular.py')

        return True

    except Exception as e:
        logger.error(f"Error creating indexed copy: {e}")
        logger.error("Cleaning up incomplete indexed copy...")
        try:
            os.remove(output_path)
        except:
            pass
        return False


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Create an optimized indexed copy of the iMessage database"
    )
    parser.add_argument(
        "--source-db",
        help="Path to the source iMessage database",
        default=DEFAULT_DB_PATH,
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to store the indexed copy",
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--output-name",
        help="Name of the output database file",
        default=DEFAULT_OUTPUT_NAME,
    )
    parser.add_argument(
        "--force", help="Overwrite existing output file", action="store_true"
    )
    parser.add_argument(
        "--read-only", help="Make the output database read-only", action="store_true"
    )
    parser.add_argument(
        "--enable-fts",
        help="Create Full-Text Search index",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-fts",
        help="Skip Full-Text Search index",
        action="store_false",
        dest="enable_fts",
    )
    parser.add_argument(
        "--create-views",
        help="Create materialized views for common queries",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-views",
        help="Skip materialized views",
        action="store_false",
        dest="create_views",
    )
    parser.add_argument(
        "--analyze",
        help="Only analyze the database, don't create a copy",
        action="store_true",
    )

    args = parser.parse_args()

    # Expand user paths
    args.source_db = os.path.expanduser(args.source_db)
    args.output_dir = os.path.expanduser(args.output_dir)

    # Check if source exists
    if not os.path.exists(args.source_db):
        logger.error(f"Source database not found: {args.source_db}")
        return 1

    # Run in appropriate mode
    if args.analyze:
        analyze_database(args.source_db)
    else:
        success = create_indexed_copy(args)
        return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
