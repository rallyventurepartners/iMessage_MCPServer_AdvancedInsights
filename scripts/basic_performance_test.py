#!/usr/bin/env python3
"""
Basic Performance Testing Script for iMessage MCP Server

This script runs performance tests on the database layer without requiring
all dependencies, making it suitable for initial performance testing.
"""

import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

import psutil

# Add parent directory to path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('basic_performance_test_results.log')
    ]
)
logger = logging.getLogger(__name__)

# Global test results storage
test_results = {
    "database_info": {},
    "query_performance": {},
    "memory_usage": {},
    "recommendations": []
}

class MemoryMonitor:
    """Simple memory usage monitor"""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.monitoring = False
        self.measurements = {}

    def start(self):
        """Start monitoring memory usage"""
        self.monitoring = True
        self.measurements = {}

    def stop(self):
        """Stop monitoring memory usage"""
        self.monitoring = False

    def mark(self, label):
        """Record memory usage at a specific point"""
        if not self.monitoring:
            return

        self.measurements[label] = self.process.memory_info().rss / (1024 * 1024)  # MB

    def get_profile(self):
        """Get the recorded memory profile"""
        return self.measurements

async def get_database_info():
    """Get basic information about the iMessage database"""
    # Import locally to avoid circular imports

    try:
        # Use direct file access to get database size
        home = os.path.expanduser("~")
        db_path = Path(f"{home}/Library/Messages/chat.db")

        if not os.path.exists(db_path):
            logger.error(f"iMessage database not found at: {db_path}")
            return {
                "exists": False,
                "error": f"Database file not found at {db_path}"
            }

        # Get file size and permissions
        db_size = os.path.getsize(db_path)
        readable = os.access(db_path, os.R_OK)

        # Try to connect to verify access
        try:
            # Import locally to avoid early import issues
            import aiosqlite

            uri = f"file:{db_path}?mode=ro&immutable=1"
            conn = await aiosqlite.connect(uri, uri=True)

            # Get basic stats
            cursor = await conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
            row = await cursor.fetchone()
            table_count = row[0] if row else 0

            # Check for key tables
            cursor = await conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('message', 'handle', 'chat')")
            rows = await cursor.fetchall()
            key_tables = [r[0] for r in rows]

            # Try to get message and contact counts
            message_count = 0
            contact_count = 0

            try:
                cursor = await conn.execute("SELECT COUNT(*) FROM message")
                row = await cursor.fetchone()
                message_count = row[0] if row else 0
            except Exception as e:
                logger.warning(f"Couldn't get message count: {e}")

            try:
                cursor = await conn.execute("SELECT COUNT(DISTINCT id) FROM handle")
                row = await cursor.fetchone()
                contact_count = row[0] if row else 0
            except Exception as e:
                logger.warning(f"Couldn't get contact count: {e}")

            await conn.close()

            return {
                "exists": True,
                "readable": readable,
                "size_bytes": db_size,
                "size_mb": db_size / (1024 * 1024),
                "table_count": table_count,
                "has_key_tables": len(key_tables) == 3,
                "key_tables": key_tables,
                "message_count": message_count,
                "contact_count": contact_count,
                "path": str(db_path)
            }

        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            return {
                "exists": True,
                "readable": readable,
                "size_bytes": db_size,
                "size_mb": db_size / (1024 * 1024),
                "error": str(e),
                "path": str(db_path)
            }

    except Exception as e:
        logger.error(f"Error getting database info: {e}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}

async def test_basic_query_performance():
    """Test the performance of basic database queries"""
    # Import locally to avoid circular imports
    import aiosqlite

    # Get database path
    home = os.path.expanduser("~")
    db_path = Path(f"{home}/Library/Messages/chat.db")

    if not os.path.exists(db_path):
        logger.error(f"Database file not found at {db_path}")
        return {"error": "Database file not found"}

    results = {}

    try:
        # Connect to database
        uri = f"file:{db_path}?mode=ro&immutable=1"
        conn = await aiosqlite.connect(uri, uri=True)
        conn.row_factory = aiosqlite.Row

        # Define test queries
        test_queries = {
            "count_messages": "SELECT COUNT(*) FROM message",
            "count_handles": "SELECT COUNT(*) FROM handle",
            "count_chats": "SELECT COUNT(*) FROM chat",
            "recent_messages": "SELECT message.ROWID, message.text, message.date, message.is_from_me, handle.id FROM message LEFT JOIN handle ON message.handle_id = handle.ROWID ORDER BY message.date DESC LIMIT 100",
            "search_messages": "SELECT message.ROWID, message.text FROM message WHERE message.text LIKE ? LIMIT 100",
            "message_by_date": "SELECT message.ROWID, message.text FROM message WHERE message.date > ? AND message.date < ? LIMIT 100",
        }

        # Run each test query
        for query_name, query in test_queries.items():
            query_results = {
                "execution_times_ms": [],
                "memory_before_mb": 0,
                "memory_after_mb": 0,
                "row_count": 0
            }

            # Get starting memory usage
            mem_before = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
            query_results["memory_before_mb"] = mem_before

            # Run query multiple times to get average performance
            iterations = 3

            for i in range(iterations):
                start_time = time.time()

                try:
                    if query_name == "search_messages":
                        cursor = await conn.execute(query, ("%meeting%",))
                    elif query_name == "message_by_date":
                        # Get timestamps for last 30 days
                        now = int(time.time())
                        thirty_days_ago = now - (30 * 24 * 60 * 60)
                        # Convert to Apple's timestamp (nanoseconds since 2001-01-01)
                        epoch_2001 = 978307200  # 2001-01-01 in Unix epoch seconds
                        start_date = (thirty_days_ago - epoch_2001) * 1e9
                        end_date = (now - epoch_2001) * 1e9
                        cursor = await conn.execute(query, (start_date, end_date))
                    else:
                        cursor = await conn.execute(query)

                    rows = await cursor.fetchall()
                    query_results["row_count"] = len(rows)

                    end_time = time.time()
                    execution_time = (end_time - start_time) * 1000  # Convert to ms
                    query_results["execution_times_ms"].append(execution_time)

                except Exception as e:
                    logger.error(f"Error executing query {query_name}: {e}")
                    query_results["error"] = str(e)
                    break

            # Get ending memory usage
            mem_after = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
            query_results["memory_after_mb"] = mem_after
            query_results["memory_increase_mb"] = mem_after - mem_before

            # Calculate average execution time
            if query_results["execution_times_ms"]:
                query_results["avg_execution_time_ms"] = statistics.mean(query_results["execution_times_ms"])

            results[query_name] = query_results

        # Close the connection
        await conn.close()

    except Exception as e:
        logger.error(f"Error in query performance test: {e}")
        logger.error(traceback.format_exc())
        results["error"] = str(e)

    return results

async def test_memory_usage():
    """Test memory usage for different database operations"""
    # Import locally to avoid circular imports
    import aiosqlite

    # Get database path
    home = os.path.expanduser("~")
    db_path = Path(f"{home}/Library/Messages/chat.db")

    if not os.path.exists(db_path):
        logger.error(f"Database file not found at {db_path}")
        return {"error": "Database file not found"}

    # Initialize memory monitor
    monitor = MemoryMonitor()
    monitor.start()

    results = {}

    try:
        # Connect to database
        uri = f"file:{db_path}?mode=ro&immutable=1"
        conn = await aiosqlite.connect(uri, uri=True)
        conn.row_factory = aiosqlite.Row

        # Baseline memory usage
        monitor.mark("baseline")

        # Test 1: Retrieve and store small result set
        monitor.mark("before_small_query")
        cursor = await conn.execute("SELECT * FROM message LIMIT 10")
        rows = await cursor.fetchall()
        small_result = [dict(row) for row in rows]
        monitor.mark("after_small_query")

        # Test 2: Retrieve and store medium result set
        monitor.mark("before_medium_query")
        cursor = await conn.execute("SELECT * FROM message LIMIT 100")
        rows = await cursor.fetchall()
        medium_result = [dict(row) for row in rows]
        monitor.mark("after_medium_query")

        # Test 3: Retrieve and store large result set
        monitor.mark("before_large_query")
        cursor = await conn.execute("SELECT * FROM message LIMIT 1000")
        rows = await cursor.fetchall()
        large_result = [dict(row) for row in rows]
        monitor.mark("after_large_query")

        # Close the connection
        await conn.close()

        # Store memory profile
        memory_profile = monitor.get_profile()

        # Calculate deltas
        results["memory_profile"] = memory_profile
        results["memory_deltas"] = {
            "small_query": memory_profile.get("after_small_query", 0) - memory_profile.get("before_small_query", 0),
            "medium_query": memory_profile.get("after_medium_query", 0) - memory_profile.get("before_medium_query", 0),
            "large_query": memory_profile.get("after_large_query", 0) - memory_profile.get("before_large_query", 0)
        }

        # Memory usage per record
        results["memory_per_record"] = {
            "small_query": results["memory_deltas"]["small_query"] / len(small_result) if small_result else 0,
            "medium_query": results["memory_deltas"]["medium_query"] / len(medium_result) if medium_result else 0,
            "large_query": results["memory_deltas"]["large_query"] / len(large_result) if large_result else 0
        }

    except Exception as e:
        logger.error(f"Error in memory usage test: {e}")
        logger.error(traceback.format_exc())
        results["error"] = str(e)

    finally:
        monitor.stop()

    return results

def analyze_and_recommend():
    """Analyze test results and generate recommendations"""
    recommendations = []

    # Check database size
    db_info = test_results.get("database_info", {})
    db_size_mb = db_info.get("size_mb", 0)
    message_count = db_info.get("message_count", 0)

    # 1. Database size recommendations
    if db_size_mb > 1000:  # Larger than 1GB
        recommendations.append({
            "area": "Database Size",
            "issue": f"Large database (${db_size_mb:.2f} MB)",
            "recommendation": "Consider enabling database sharding for better performance",
            "priority": "High"
        })

        # If more than 1 million messages, strongly recommend sharding
        if message_count > 1000000:
            recommendations.append({
                "area": "Message Volume",
                "issue": f"Very large message count ({message_count:,})",
                "recommendation": "Enable database sharding and implement aggressive query limits",
                "priority": "Critical"
            })

    # 2. Query performance recommendations
    query_results = test_results.get("query_performance", {})

    # Check if query_results contains an error
    if isinstance(query_results, dict) and not query_results.get("error"):
        slow_queries = []
        for query_name, result in query_results.items():
            if isinstance(result, dict) and "avg_execution_time_ms" in result:
                avg_time = result.get("avg_execution_time_ms", 0)
                if avg_time > 1000:  # Slower than 1 second
                    slow_queries.append((query_name, avg_time))

        if slow_queries:
            for query_name, time in slow_queries:
                recommendations.append({
                    "area": "Query Performance",
                    "issue": f"Slow query: {query_name} ({time:.2f} ms)",
                    "recommendation": "Consider adding indexes or optimizing this query",
                    "priority": "Medium"
                })
    elif isinstance(query_results, dict) and query_results.get("error"):
        # Handle database access error
        recommendations.append({
            "area": "Database Access",
            "issue": f"Database access error: {query_results.get('error')}",
            "recommendation": "Grant Full Disk Access permission to Terminal/VS Code in System Preferences > Security & Privacy > Privacy",
            "priority": "Critical"
        })

    # Note: We already handle slow queries in the block above

    # 3. Memory usage recommendations
    memory_results = test_results.get("memory_usage", {})
    memory_deltas = memory_results.get("memory_deltas", {})

    large_query_delta = memory_deltas.get("large_query", 0)
    if large_query_delta > 100:  # More than 100MB for 1000 records
        recommendations.append({
            "area": "Memory Usage",
            "issue": f"High memory usage for large queries ({large_query_delta:.2f} MB)",
            "recommendation": "Implement pagination and result limiting to control memory usage",
            "priority": "Medium"
        })

    # Add the recommendations to the test results
    test_results["recommendations"] = recommendations

    # Log recommendations
    logger.info("Performance test recommendations:")
    for rec in recommendations:
        logger.info(f"  [{rec['priority']}] {rec['area']}: {rec['recommendation']}")

async def run_tests():
    """Run all performance tests"""
    logger.info("Starting basic performance tests")

    # Get database info
    logger.info("Getting database information...")
    test_results["database_info"] = await get_database_info()

    # Log database info
    db_info = test_results["database_info"]
    if db_info.get("exists", False):
        logger.info(f"Database found at: {db_info.get('path')}")
        logger.info(f"Database size: {db_info.get('size_mb', 0):.2f} MB")
        logger.info(f"Message count: {db_info.get('message_count', 0):,}")
        logger.info(f"Contact count: {db_info.get('contact_count', 0):,}")
    else:
        logger.error(f"Database not found or error: {db_info.get('error')}")

    # Run query performance tests
    logger.info("Testing query performance...")
    test_results["query_performance"] = await test_basic_query_performance()

    # Run memory usage tests
    logger.info("Testing memory usage...")
    test_results["memory_usage"] = await test_memory_usage()

    # Analyze results and make recommendations
    analyze_and_recommend()

    # Save results to file
    with open("basic_performance_test_results.json", "w") as f:
        json.dump(test_results, f, indent=2, default=str)

    logger.info("Performance test results saved to basic_performance_test_results.json")

    # Print summary
    print("\n" + "="*80)
    print("PERFORMANCE TEST SUMMARY")
    print("="*80)

    # Database info
    db_info = test_results["database_info"]
    print(f"Database size: {db_info.get('size_mb', 0):.2f} MB")
    print(f"Messages: {db_info.get('message_count', 0):,}")
    print(f"Contacts: {db_info.get('contact_count', 0):,}")

    # Query performance
    print("\nQuery Performance:")
    for query_name, result in test_results.get("query_performance", {}).items():
        if isinstance(result, dict) and "avg_execution_time_ms" in result:
            print(f"  {query_name}: {result['avg_execution_time_ms']:.2f} ms")

    # Memory usage
    print("\nMemory Usage:")
    memory_deltas = test_results.get("memory_usage", {}).get("memory_deltas", {})
    for query_type, delta in memory_deltas.items():
        print(f"  {query_type}: {delta:.2f} MB")

    # Recommendations
    if test_results.get("recommendations"):
        print("\nRecommendations:")
        for rec in test_results["recommendations"]:
            print(f"  [{rec['priority']}] {rec['area']}: {rec['recommendation']}")

    print("="*80)

def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description="Run basic performance tests for iMessage MCP Server")
    args = parser.parse_args()

    # Run tests
    asyncio.run(run_tests())

if __name__ == "__main__":
    main()
