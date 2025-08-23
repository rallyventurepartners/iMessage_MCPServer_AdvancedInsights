#!/usr/bin/env python3
"""
Database Sharding Utility

This script creates time-based shards for extremely large iMessage databases
to improve performance and reduce memory usage.

Usage:
    python shard_large_database.py [--source DB_PATH] [--shards-dir DIR_PATH] [--months MONTHS]

Options:
    --source     Path to the source iMessage database (default: ~/Library/Messages/chat.db)
    --shards-dir Directory to store database shards (default: ~/.imessage_insights/shards)
    --months     Number of months per shard (default: 6)
    --force      Force sharding even for small databases
    --analyze    Analyze database first without creating shards
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add the script directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create time-based shards for large iMessage databases"
    )

    # Default paths
    home = os.path.expanduser("~")
    default_db_path = os.path.join(home, "Library", "Messages", "chat.db")
    default_shards_dir = os.path.join(home, ".imessage_insights", "shards")

    # Add arguments
    parser.add_argument(
        "--source",
        default=default_db_path,
        help=f"Path to the source iMessage database (default: {default_db_path})",
    )
    parser.add_argument(
        "--shards-dir",
        default=default_shards_dir,
        help=f"Directory to store database shards (default: {default_shards_dir})",
    )
    parser.add_argument(
        "--months", type=int, default=6, help="Number of months per shard (default: 6)"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force sharding even for small databases"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze database first without creating shards",
    )

    return parser.parse_args()


async def analyze_database(db_path, shards_dir, shard_size_months):
    """
    Analyze the database and provide recommendations.

    Args:
        db_path: Path to the source database
        shards_dir: Directory to store shards
        shard_size_months: Number of months per shard
    """
    # Import manager
    from src.database.large_db_handler import LargeDatabaseManager

    try:
        # Check if database exists
        if not os.path.exists(db_path):
            logger.error(f"Source database not found: {db_path}")
            return False

        # Get database size
        size_bytes = os.path.getsize(db_path)
        size_gb = size_bytes / (1024 * 1024 * 1024)

        logger.info("Database analysis:")
        logger.info(f"  Path: {db_path}")
        logger.info(f"  Size: {size_bytes:,} bytes ({size_gb:.2f} GB)")

        # Determine if sharding is recommended
        sharding_recommended = size_gb >= 10.0

        if sharding_recommended:
            logger.info("  Recommendation: Database sharding RECOMMENDED")
        else:
            logger.info("  Recommendation: Database sharding NOT NEEDED")
            logger.info("  (Use --force to shard anyway)")

        # Initialize manager but don't create shards yet
        manager = LargeDatabaseManager(
            source_db_path=db_path,
            shards_dir=shards_dir,
            shard_size_months=shard_size_months,
        )

        # Connect to database to get date range
        import sqlite3

        import aiosqlite

        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row

            # Get message count
            cursor = await conn.execute("SELECT COUNT(*) as count FROM message")
            row = await cursor.fetchone()
            message_count = row["count"] if row else 0

            # Get date range
            cursor = await conn.execute(
                "SELECT MIN(date) as min_date, MAX(date) as max_date FROM message"
            )
            row = await cursor.fetchone()

            if row and row["min_date"] and row["max_date"]:
                # Convert Apple timestamp to datetime
                apple_epoch = datetime(2001, 1, 1)
                min_date = apple_epoch + await asyncio.to_thread(
                    lambda: sqlite3.TimeDelta(microseconds=row["min_date"] // 1000)
                )
                max_date = apple_epoch + await asyncio.to_thread(
                    lambda: sqlite3.TimeDelta(microseconds=row["max_date"] // 1000)
                )

                date_range_months = (
                    (max_date.year - min_date.year) * 12
                    + max_date.month
                    - min_date.month
                )
                estimated_shards = (date_range_months // shard_size_months) + 1

                logger.info(f"  Messages: {message_count:,}")
                logger.info(
                    f"  Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
                )
                logger.info(f"  Time span: {date_range_months} months")
                logger.info(
                    f"  Estimated shards: {estimated_shards} ({shard_size_months} months per shard)"
                )

                if estimated_shards > 20:
                    logger.warning(
                        "  Warning: Large number of shards. Consider increasing --months value."
                    )
            else:
                logger.warning("  Unable to determine date range")

        return True

    except Exception as e:
        logger.error(f"Error analyzing database: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False


async def create_shards(db_path, shards_dir, shard_size_months, force=False):
    """
    Create database shards.

    Args:
        db_path: Path to the source database
        shards_dir: Directory to store shards
        shard_size_months: Number of months per shard
        force: Force sharding even for small databases
    """
    # Import database classes
    from src.database.large_db_handler import LargeDatabaseManager

    try:
        # Check if database exists
        if not os.path.exists(db_path):
            logger.error(f"Source database not found: {db_path}")
            return False

        # Get database size
        size_bytes = os.path.getsize(db_path)
        size_gb = size_bytes / (1024 * 1024 * 1024)

        # Check if sharding is needed
        sharding_needed = size_gb >= 10.0 or force

        if not sharding_needed:
            logger.info(f"Database size is {size_gb:.2f} GB, below 10 GB threshold")
            logger.info("Sharding not needed. Use --force to shard anyway.")
            return False

        # Create shards directory
        os.makedirs(shards_dir, exist_ok=True)

        # Initialize manager
        logger.info("Initializing shard manager...")
        manager = LargeDatabaseManager(
            source_db_path=db_path,
            shards_dir=shards_dir,
            shard_size_months=shard_size_months,
        )

        # Create shards
        logger.info("Creating shards (this may take a while)...")
        await manager.initialize()
        success = await manager.create_shards()

        if success:
            logger.info(f"Successfully created {len(manager.shards)} shards:")
            for i, shard in enumerate(manager.shards):
                date_range = "unknown"
                if shard.start_date and shard.end_date:
                    date_range = f"{shard.start_date.strftime('%Y-%m-%d')} to {shard.end_date.strftime('%Y-%m-%d')}"

                logger.info(f"  Shard {i+1}: {shard.shard_path.name}")
                logger.info(f"    Date range: {date_range}")
                logger.info(f"    Messages: {shard.message_count:,}")
                logger.info(f"    Size: {shard.size_bytes / (1024 * 1024):.2f} MB")
        else:
            logger.error("Failed to create shards. Check logs for details.")

        return success

    except Exception as e:
        logger.error(f"Error creating shards: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False


async def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()

    # Setup paths
    db_path = os.path.abspath(os.path.expanduser(args.source))
    shards_dir = os.path.abspath(os.path.expanduser(args.shards_dir))

    # Analyze database
    if args.analyze:
        logger.info("Analyzing database...")
        await analyze_database(db_path, shards_dir, args.months)
        return

    # Create shards
    logger.info("Starting database sharding...")
    logger.info(f"  Source: {db_path}")
    logger.info(f"  Shards directory: {shards_dir}")
    logger.info(f"  Months per shard: {args.months}")

    success = await create_shards(db_path, shards_dir, args.months, args.force)

    if success:
        logger.info("Sharding completed successfully")
        logger.info("To use the sharded database, set the following in your config:")
        logger.info(f"  - db_path: {db_path}")
        logger.info(f"  - shards_dir: {shards_dir}")
    else:
        logger.info("Sharding process not completed")


if __name__ == "__main__":
    asyncio.run(main())
