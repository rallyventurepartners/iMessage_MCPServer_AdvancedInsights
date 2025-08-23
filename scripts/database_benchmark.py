#!/usr/bin/env python3
"""
Database Performance Benchmark Tool

This script benchmarks the performance of different database access methods,
comparing standard AsyncMessagesDB with ShardedAsyncMessagesDB.

Usage:
    python database_benchmark.py [--db-path PATH] [--shards-dir DIR] [--tests TESTS]

Options:
    --db-path    Path to the iMessage database
    --shards-dir Directory containing database shards
    --tests      Comma-separated list of tests to run (all, chat, search, recent, count)
"""

import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
import time
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
    parser = argparse.ArgumentParser(description="Benchmark database performance")

    # Default paths
    home = os.path.expanduser("~")
    default_db_path = os.path.join(home, "Library", "Messages", "chat.db")
    default_shards_dir = os.path.join(home, ".imessage_insights", "shards")

    # Add arguments
    parser.add_argument(
        "--db-path",
        default=default_db_path,
        help=f"Path to the iMessage database (default: {default_db_path})",
    )
    parser.add_argument(
        "--shards-dir",
        default=default_shards_dir,
        help=f"Directory containing database shards (default: {default_shards_dir})",
    )
    parser.add_argument(
        "--tests",
        default="all",
        help="Comma-separated list of tests to run (all, chat, search, recent, count)",
    )

    return parser.parse_args()


class BenchmarkResult:
    """Class to store and analyze benchmark results."""

    def __init__(self, name, db_type):
        """Initialize benchmark result."""
        self.name = name
        self.db_type = db_type
        self.durations = []
        self.result_sizes = []
        self.errors = []

    def add_run(self, duration, result_size=None, error=None):
        """Add a benchmark run result."""
        self.durations.append(duration)

        if result_size is not None:
            self.result_sizes.append(result_size)

        if error:
            self.errors.append(error)

    def get_stats(self):
        """Get statistics for the benchmark runs."""
        if not self.durations:
            return {
                "name": self.name,
                "db_type": self.db_type,
                "runs": 0,
                "errors": len(self.errors),
                "avg_duration": 0,
                "min_duration": 0,
                "max_duration": 0,
                "median_duration": 0,
                "avg_result_size": 0,
            }

        # Calculate statistics
        avg_duration = sum(self.durations) / len(self.durations)
        min_duration = min(self.durations)
        max_duration = max(self.durations)

        # Calculate median
        median_duration = statistics.median(self.durations)

        # Calculate average result size if available
        avg_result_size = 0
        if self.result_sizes:
            avg_result_size = sum(self.result_sizes) / len(self.result_sizes)

        return {
            "name": self.name,
            "db_type": self.db_type,
            "runs": len(self.durations),
            "errors": len(self.errors),
            "avg_duration": avg_duration,
            "min_duration": min_duration,
            "max_duration": max_duration,
            "median_duration": median_duration,
            "avg_result_size": avg_result_size,
        }

    def __str__(self):
        """Return string representation of benchmark results."""
        stats = self.get_stats()
        return (
            f"{stats['name']} ({stats['db_type']}): "
            f"avg={stats['avg_duration']:.4f}s, "
            f"min={stats['min_duration']:.4f}s, "
            f"max={stats['max_duration']:.4f}s, "
            f"median={stats['median_duration']:.4f}s, "
            f"runs={stats['runs']}, "
            f"errors={stats['errors']}, "
            f"avg_size={stats['avg_result_size']:.1f}"
        )


class DatabaseBenchmark:
    """Benchmark different database access methods."""

    def __init__(self, db_path, shards_dir):
        """Initialize benchmark."""
        self.db_path = os.path.abspath(os.path.expanduser(db_path))
        self.shards_dir = os.path.abspath(os.path.expanduser(shards_dir))
        self.results = []

        # Check if database exists
        if not os.path.exists(self.db_path):
            logger.error(f"Database not found: {self.db_path}")
            sys.exit(1)

    async def _init_standard_db(self):
        """Initialize standard database."""
        from src.database.async_messages_db import AsyncMessagesDB

        db = AsyncMessagesDB(db_path=self.db_path)
        await db.initialize()
        return db

    async def _init_sharded_db(self):
        """Initialize sharded database."""
        from src.database.sharded_async_messages_db import ShardedAsyncMessagesDB

        # Check if shards directory exists
        if not os.path.exists(self.shards_dir):
            logger.warning(f"Shards directory not found: {self.shards_dir}")
            logger.warning("Run shard_large_database.py first to create shards")
            return None

        db = ShardedAsyncMessagesDB(db_path=self.db_path, shards_dir=self.shards_dir)
        await db.initialize()

        # Check if shards are available
        if not db.using_shards:
            logger.warning("No shards found or sharding not initialized")
            return None

        return db

    async def run_get_chats_benchmark(self, iterations=3):
        """Benchmark getting chat list."""
        logger.info("Benchmarking get_chats query...")

        # Initialize databases
        standard_db = await self._init_standard_db()
        sharded_db = await self._init_sharded_db()

        # Create results
        standard_result = BenchmarkResult("get_chats", "standard")
        sharded_result = BenchmarkResult("get_chats", "sharded")

        # Run standard benchmark
        logger.info("  Running standard database benchmark...")
        for i in range(iterations):
            try:
                start_time = time.time()
                chats = await standard_db.get_chats()
                end_time = time.time()

                standard_result.add_run(
                    duration=end_time - start_time, result_size=len(chats)
                )
            except Exception as e:
                logger.error(f"Error in standard benchmark (iteration {i+1}): {e}")
                standard_result.add_run(0, error=str(e))

        # Run sharded benchmark if available
        if sharded_db:
            logger.info("  Running sharded database benchmark...")
            for i in range(iterations):
                try:
                    start_time = time.time()
                    chats = await sharded_db.get_chats()
                    end_time = time.time()

                    sharded_result.add_run(
                        duration=end_time - start_time, result_size=len(chats)
                    )
                except Exception as e:
                    logger.error(f"Error in sharded benchmark (iteration {i+1}): {e}")
                    sharded_result.add_run(0, error=str(e))

        # Clean up
        await standard_db.cleanup()
        if sharded_db:
            await sharded_db.cleanup()

        # Add results
        self.results.append(standard_result)
        if sharded_db:
            self.results.append(sharded_result)

        return standard_result, sharded_result

    async def run_search_messages_benchmark(self, iterations=3):
        """Benchmark searching messages."""
        logger.info("Benchmarking search_messages query...")

        # Initialize databases
        standard_db = await self._init_standard_db()
        sharded_db = await self._init_sharded_db()

        # Create results
        standard_result = BenchmarkResult("search_messages", "standard")
        sharded_result = BenchmarkResult("search_messages", "sharded")

        # Common search terms
        search_terms = ["hello", "thanks", "weekend", "meeting"]

        # Run standard benchmark
        logger.info("  Running standard database benchmark...")
        for i in range(iterations):
            # Use different search terms in rotation
            search_term = search_terms[i % len(search_terms)]

            try:
                start_time = time.time()
                messages = await standard_db.search_messages(search_term, limit=50)
                end_time = time.time()

                standard_result.add_run(
                    duration=end_time - start_time, result_size=len(messages)
                )
            except Exception as e:
                logger.error(f"Error in standard benchmark (iteration {i+1}): {e}")
                standard_result.add_run(0, error=str(e))

        # Run sharded benchmark if available
        if sharded_db:
            logger.info("  Running sharded database benchmark...")
            for i in range(iterations):
                # Use same search terms in rotation
                search_term = search_terms[i % len(search_terms)]

                try:
                    start_time = time.time()
                    messages = await sharded_db.search_messages(search_term, limit=50)
                    end_time = time.time()

                    sharded_result.add_run(
                        duration=end_time - start_time, result_size=len(messages)
                    )
                except Exception as e:
                    logger.error(f"Error in sharded benchmark (iteration {i+1}): {e}")
                    sharded_result.add_run(0, error=str(e))

        # Clean up
        await standard_db.cleanup()
        if sharded_db:
            await sharded_db.cleanup()

        # Add results
        self.results.append(standard_result)
        if sharded_db:
            self.results.append(sharded_result)

        return standard_result, sharded_result

    async def run_get_recent_messages_benchmark(self, iterations=3):
        """Benchmark getting recent messages."""
        logger.info("Benchmarking get_recent_messages query...")

        # Initialize databases
        standard_db = await self._init_standard_db()
        sharded_db = await self._init_sharded_db()

        # Create results
        standard_result = BenchmarkResult("get_recent_messages", "standard")
        sharded_result = BenchmarkResult("get_recent_messages", "sharded")

        # Run standard benchmark
        logger.info("  Running standard database benchmark...")
        for i in range(iterations):
            try:
                start_time = time.time()
                messages = await standard_db.get_recent_messages(limit=100)
                end_time = time.time()

                standard_result.add_run(
                    duration=end_time - start_time, result_size=len(messages)
                )
            except Exception as e:
                logger.error(f"Error in standard benchmark (iteration {i+1}): {e}")
                standard_result.add_run(0, error=str(e))

        # Run sharded benchmark if available
        if sharded_db:
            logger.info("  Running sharded database benchmark...")
            for i in range(iterations):
                try:
                    start_time = time.time()
                    messages = await sharded_db.get_recent_messages(limit=100)
                    end_time = time.time()

                    sharded_result.add_run(
                        duration=end_time - start_time, result_size=len(messages)
                    )
                except Exception as e:
                    logger.error(f"Error in sharded benchmark (iteration {i+1}): {e}")
                    sharded_result.add_run(0, error=str(e))

        # Clean up
        await standard_db.cleanup()
        if sharded_db:
            await sharded_db.cleanup()

        # Add results
        self.results.append(standard_result)
        if sharded_db:
            self.results.append(sharded_result)

        return standard_result, sharded_result

    async def run_get_message_count_benchmark(self, iterations=3):
        """Benchmark getting message count."""
        logger.info("Benchmarking get_message_count query...")

        # Initialize databases
        standard_db = await self._init_standard_db()
        sharded_db = await self._init_sharded_db()

        # Create results
        standard_result = BenchmarkResult("get_message_count", "standard")
        sharded_result = BenchmarkResult("get_message_count", "sharded")

        # Run standard benchmark
        logger.info("  Running standard database benchmark...")
        for i in range(iterations):
            try:
                start_time = time.time()
                count = await standard_db.get_message_count()
                end_time = time.time()

                standard_result.add_run(
                    duration=end_time - start_time, result_size=count
                )
            except Exception as e:
                logger.error(f"Error in standard benchmark (iteration {i+1}): {e}")
                standard_result.add_run(0, error=str(e))

        # Run sharded benchmark if available
        if sharded_db:
            logger.info("  Running sharded database benchmark...")
            for i in range(iterations):
                try:
                    start_time = time.time()
                    count = await sharded_db.get_message_count()
                    end_time = time.time()

                    sharded_result.add_run(
                        duration=end_time - start_time, result_size=count
                    )
                except Exception as e:
                    logger.error(f"Error in sharded benchmark (iteration {i+1}): {e}")
                    sharded_result.add_run(0, error=str(e))

        # Clean up
        await standard_db.cleanup()
        if sharded_db:
            await sharded_db.cleanup()

        # Add results
        self.results.append(standard_result)
        if sharded_db:
            self.results.append(sharded_result)

        return standard_result, sharded_result

    async def run_all_benchmarks(self, iterations=3):
        """Run all benchmarks."""
        await self.run_get_chats_benchmark(iterations)
        await self.run_search_messages_benchmark(iterations)
        await self.run_get_recent_messages_benchmark(iterations)
        await self.run_get_message_count_benchmark(iterations)

    def print_results(self):
        """Print benchmark results."""
        logger.info("\nBenchmark Results:")
        logger.info("=================")

        # Group results by test
        grouped_results = {}
        for result in self.results:
            if result.name not in grouped_results:
                grouped_results[result.name] = []
            grouped_results[result.name].append(result)

        # Print results by test
        for test_name, results in grouped_results.items():
            logger.info(f"\n{test_name}:")

            # Get standard and sharded results
            standard_result = next(
                (r for r in results if r.db_type == "standard"), None
            )
            sharded_result = next((r for r in results if r.db_type == "sharded"), None)

            # Print individual results
            if standard_result:
                logger.info(f"  Standard: {standard_result}")

            if sharded_result:
                logger.info(f"  Sharded:  {sharded_result}")

            # Calculate improvement if both results are available
            if standard_result and sharded_result:
                std_stats = standard_result.get_stats()
                shd_stats = sharded_result.get_stats()

                if std_stats["avg_duration"] > 0:
                    improvement = (
                        (std_stats["avg_duration"] - shd_stats["avg_duration"])
                        / std_stats["avg_duration"]
                        * 100
                    )
                    logger.info(f"  Improvement: {improvement:.2f}%")

    def save_results(self, filename="benchmark_results.json"):
        """Save benchmark results to a file."""
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "db_path": self.db_path,
            "shards_dir": self.shards_dir,
            "results": [r.get_stats() for r in self.results],
        }

        with open(filename, "w") as f:
            json.dump(results_data, f, indent=2)

        logger.info(f"Results saved to {filename}")


async def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()

    # Setup paths
    db_path = args.db_path
    shards_dir = args.shards_dir

    # Parse tests to run
    tests = args.tests.lower().split(",")
    run_all = "all" in tests

    # Create benchmark
    benchmark = DatabaseBenchmark(db_path, shards_dir)

    # Run selected benchmarks
    if run_all or "chat" in tests:
        await benchmark.run_get_chats_benchmark()

    if run_all or "search" in tests:
        await benchmark.run_search_messages_benchmark()

    if run_all or "recent" in tests:
        await benchmark.run_get_recent_messages_benchmark()

    if run_all or "count" in tests:
        await benchmark.run_get_message_count_benchmark()

    # Print results
    benchmark.print_results()

    # Save results
    results_file = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    benchmark.save_results(results_file)


if __name__ == "__main__":
    asyncio.run(main())
