#!/usr/bin/env python3
"""
Performance testing script for iMessage MCP Server.

This script tests performance with large databases, measures response times,
profiles memory usage, and validates query optimizations.
"""

import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
import time
import tracemalloc
from datetime import datetime, timedelta
from typing import Any, Dict

import psutil

# Add parent directory to path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.database import get_database
from src.utils.concurrency import init_concurrency
from src.utils.memory_monitor import MemoryMonitor
from src.utils.query_cache import get_cache_stats, invalidate_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('performance_test_results.log')
    ]
)
logger = logging.getLogger(__name__)

# Global test results storage
test_results = {
    "database_size": 0,
    "contact_count": 0,
    "message_count": 0,
    "tests": {},
    "memory": {},
    "optimizations": {},
    "recommendations": [],
}


async def get_database_stats() -> Dict[str, int]:
    """Get basic statistics about the database size."""
    db = await get_database()

    try:
        # Get total message count
        message_count = await db.get_message_count()

        # Get contact count
        contact_count = await db.get_contact_count()

        # Get database file size
        db_path = db.get_database_path()
        db_size = os.path.getsize(db_path) if os.path.exists(db_path) else 0

        return {
            "message_count": message_count,
            "contact_count": contact_count,
            "database_size_bytes": db_size,
        }
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        return {
            "message_count": 0,
            "contact_count": 0,
            "database_size_bytes": 0,
        }


async def test_query_performance(
    test_name: str,
    query_func,
    iterations: int = 5,
    **query_params
) -> Dict[str, Any]:
    """
    Test the performance of a database query function.
    
    Args:
        test_name: Name of the test
        query_func: Async function to test
        iterations: Number of test iterations
        **query_params: Parameters to pass to the query function
        
    Returns:
        Dictionary with test results
    """
    logger.info(f"Running performance test: {test_name}")

    # Initialize test results
    results = {
        "name": test_name,
        "iterations": iterations,
        "params": query_params,
        "times_ms": [],
        "memory_usage_mb": [],
        "with_cache": [],
        "without_cache": [],
    }

    # Get database
    db = await get_database()

    # First, test without cache to get baseline
    await invalidate_cache()
    logger.info("  Testing without cache...")

    for i in range(iterations):
        # Start memory tracking
        tracemalloc.start()
        mem_before = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

        # Time the query
        start_time = time.time()
        try:
            result = await query_func(**query_params)
        except Exception as e:
            logger.error(f"  Error in iteration {i+1}: {e}")
            continue

        end_time = time.time()
        execution_time = (end_time - start_time) * 1000  # Convert to ms

        # Record memory usage
        mem_after = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        mem_diff = mem_after - mem_before
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Log and store results
        logger.info(f"  Iteration {i+1} (no cache): {execution_time:.2f}ms, {mem_diff:.2f}MB")
        results["without_cache"].append({
            "time_ms": execution_time,
            "memory_mb": mem_diff,
            "peak_memory_mb": peak / 1024 / 1024,
        })

    # Calculate average without cache
    if results["without_cache"]:
        avg_time_no_cache = statistics.mean([r["time_ms"] for r in results["without_cache"]])
        avg_mem_no_cache = statistics.mean([r["memory_mb"] for r in results["without_cache"]])
        results["avg_time_no_cache_ms"] = avg_time_no_cache
        results["avg_memory_no_cache_mb"] = avg_mem_no_cache

    # Now test with cache enabled
    logger.info("  Testing with cache...")

    for i in range(iterations):
        # Start memory tracking
        tracemalloc.start()
        mem_before = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

        # Time the query
        start_time = time.time()
        try:
            result = await query_func(**query_params)
        except Exception as e:
            logger.error(f"  Error in iteration {i+1}: {e}")
            continue

        end_time = time.time()
        execution_time = (end_time - start_time) * 1000  # Convert to ms

        # Record memory usage
        mem_after = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        mem_diff = mem_after - mem_before
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Log and store results
        logger.info(f"  Iteration {i+1} (with cache): {execution_time:.2f}ms, {mem_diff:.2f}MB")
        results["with_cache"].append({
            "time_ms": execution_time,
            "memory_mb": mem_diff,
            "peak_memory_mb": peak / 1024 / 1024,
        })

    # Calculate average with cache
    if results["with_cache"]:
        avg_time_with_cache = statistics.mean([r["time_ms"] for r in results["with_cache"]])
        avg_mem_with_cache = statistics.mean([r["memory_mb"] for r in results["with_cache"]])
        results["avg_time_with_cache_ms"] = avg_time_with_cache
        results["avg_memory_with_cache_mb"] = avg_mem_with_cache

        # Calculate cache improvement
        if "avg_time_no_cache_ms" in results:
            time_improvement = (results["avg_time_no_cache_ms"] - avg_time_with_cache) / results["avg_time_no_cache_ms"] * 100
            results["cache_time_improvement_pct"] = time_improvement

    # Get cache stats
    cache_stats = await get_cache_stats()
    results["cache_stats"] = cache_stats

    logger.info(f"  Completed test: {test_name}")
    return results


async def test_large_database_performance():
    """Run performance tests specifically for large database scenarios."""
    logger.info("Starting large database performance tests")

    # Get database instance
    db = await get_database()

    # Test getting message counts
    message_count_results = await test_query_performance(
        "message_count",
        db.get_message_count,
        iterations=3
    )
    test_results["tests"]["message_count"] = message_count_results

    # Test getting recent messages
    recent_messages_results = await test_query_performance(
        "recent_messages",
        db.get_messages,
        iterations=3,
        limit=100,
        offset=0,
        order_by="date",
        order_direction="DESC"
    )
    test_results["tests"]["recent_messages"] = recent_messages_results

    # Test searching messages
    search_messages_results = await test_query_performance(
        "search_messages",
        db.search_messages,
        iterations=3,
        search_term="test",
        limit=50
    )
    test_results["tests"]["search_messages"] = search_messages_results

    # Test getting contact list
    contacts_results = await test_query_performance(
        "get_contacts",
        db.get_contacts,
        iterations=3,
        page=1,
        page_size=30
    )
    test_results["tests"]["get_contacts"] = contacts_results

    # Test getting single contact analytics
    if test_results.get("contact_count", 0) > 0:
        # Get a random contact ID
        contacts = (await db.get_contacts(page=1, page_size=1)).get("contacts", [])
        if contacts:
            contact_id = contacts[0].get("id")

            contact_analytics_results = await test_query_performance(
                "contact_analytics",
                db.get_contact_analytics,
                iterations=3,
                contact_id=contact_id,
                start_date=datetime.now() - timedelta(days=30),
                end_date=datetime.now()
            )
            test_results["tests"]["contact_analytics"] = contact_analytics_results

    # Test topic analysis
    if test_results.get("message_count", 0) > 1000:
        topic_analysis_results = await test_query_performance(
            "topic_analysis",
            db.get_conversation_topics,
            iterations=2,  # Reduced iterations as this can be resource-intensive
            topic_count=5,
            include_sentiment=True
        )
        test_results["tests"]["topic_analysis"] = topic_analysis_results

    logger.info("Completed large database performance tests")


async def measure_response_times():
    """Measure response times for various operations."""
    logger.info("Measuring response times for typical operations")

    # Get database instance
    db = await get_database()

    # Define test cases
    test_cases = [
        {
            "name": "get_messages_small",
            "func": db.get_messages,
            "params": {"limit": 20, "offset": 0},
        },
        {
            "name": "get_messages_medium",
            "func": db.get_messages,
            "params": {"limit": 100, "offset": 0},
        },
        {
            "name": "get_messages_large",
            "func": db.get_messages,
            "params": {"limit": 500, "offset": 0},
        },
        {
            "name": "search_messages_simple",
            "func": db.search_messages,
            "params": {"search_term": "hello", "limit": 50},
        },
        {
            "name": "search_messages_complex",
            "func": db.search_messages,
            "params": {
                "search_term": "important meeting",
                "limit": 50,
                "start_date": datetime.now() - timedelta(days=90),
                "end_date": datetime.now(),
            },
        },
    ]

    # Run tests
    response_times = {}
    for test_case in test_cases:
        logger.info(f"Testing response time for: {test_case['name']}")

        results = await test_query_performance(
            test_case["name"],
            test_case["func"],
            iterations=3,
            **test_case["params"]
        )

        response_times[test_case["name"]] = results

    test_results["response_times"] = response_times
    logger.info("Completed response time measurements")


async def profile_memory_usage():
    """Profile memory usage during various operations."""
    logger.info("Profiling memory usage")

    # Set up memory monitor
    memory_monitor = MemoryMonitor()
    memory_monitor.start()

    try:
        # Get database instance
        db = await get_database()

        # Run a series of progressively more memory-intensive operations

        # 1. Basic operation - getting messages
        logger.info("Memory profile: Basic message retrieval")
        memory_monitor.mark("before_basic_messages")
        await db.get_messages(limit=100, offset=0)
        memory_monitor.mark("after_basic_messages")

        # 2. Medium operation - searching messages with filters
        logger.info("Memory profile: Searching messages")
        memory_monitor.mark("before_search")
        await db.search_messages(
            search_term="meeting",
            limit=100,
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now()
        )
        memory_monitor.mark("after_search")

        # 3. Heavy operation - contact analytics
        logger.info("Memory profile: Contact analytics")
        memory_monitor.mark("before_contact_analytics")
        contacts = (await db.get_contacts(page=1, page_size=1)).get("contacts", [])
        if contacts:
            contact_id = contacts[0].get("id")
            await db.get_contact_analytics(
                contact_id=contact_id,
                start_date=datetime.now() - timedelta(days=90),
                end_date=datetime.now()
            )
        memory_monitor.mark("after_contact_analytics")

        # 4. Very heavy operation - topic analysis
        logger.info("Memory profile: Topic analysis")
        memory_monitor.mark("before_topic_analysis")
        if test_results.get("message_count", 0) > 1000:
            await db.get_conversation_topics(
                topic_count=5,
                include_sentiment=True
            )
        memory_monitor.mark("after_topic_analysis")

        # Get memory profile
        memory_profile = memory_monitor.get_profile()
        test_results["memory"]["profile"] = memory_profile

        # Calculate deltas
        test_results["memory"]["deltas"] = {
            "basic_messages": memory_profile.get("after_basic_messages", 0) - memory_profile.get("before_basic_messages", 0),
            "search": memory_profile.get("after_search", 0) - memory_profile.get("before_search", 0),
            "contact_analytics": memory_profile.get("after_contact_analytics", 0) - memory_profile.get("before_contact_analytics", 0),
            "topic_analysis": memory_profile.get("after_topic_analysis", 0) - memory_profile.get("before_topic_analysis", 0),
        }

        # Log memory usage summary
        logger.info("Memory usage summary:")
        for op, delta in test_results["memory"]["deltas"].items():
            logger.info(f"  {op}: {delta:.2f} MB")

    finally:
        # Stop memory monitoring
        memory_monitor.stop()
        logger.info("Completed memory profiling")


async def evaluate_query_optimizations():
    """Evaluate the effectiveness of database query optimizations."""
    logger.info("Evaluating query optimizations")

    # Get database instance
    db = await get_database()

    # Test 1: Query caching
    logger.info("Testing query caching effectiveness")

    # First, get cache stats (before tests)
    before_stats = await get_cache_stats()

    # Run a cacheable query multiple times
    for i in range(5):
        await db.get_messages(limit=50, offset=0)

    # Get cache stats after tests
    after_stats = await get_cache_stats()

    # Calculate cache hit rate
    cache_hit_rate = 0
    if after_stats.get("total_hits", 0) + after_stats.get("total_misses", 0) > 0:
        cache_hit_rate = after_stats.get("total_hits", 0) / (after_stats.get("total_hits", 0) + after_stats.get("total_misses", 0)) * 100

    test_results["optimizations"]["cache_hit_rate"] = cache_hit_rate

    # Test 2: Connection pooling
    logger.info("Testing connection pooling")

    # Run multiple concurrent operations to test connection pool
    tasks = []
    for i in range(10):
        tasks.append(db.get_messages(limit=20, offset=i*20))

    start_time = time.time()
    await asyncio.gather(*tasks)
    end_time = time.time()

    concurrent_execution_time = (end_time - start_time) * 1000  # Convert to ms
    test_results["optimizations"]["concurrent_execution_time_ms"] = concurrent_execution_time

    # Test 3: Batch processing
    logger.info("Testing batch processing")

    # Use the batch processor class if available, otherwise simulate
    try:
        from src.utils.concurrency import BatchProcessor

        async def process_batch(items):
            # Simulate batch processing
            return [f"Processed {item}" for item in items]

        processor = BatchProcessor(process_batch, max_batch_size=5, max_wait_time=0.01)

        start_time = time.time()
        batch_tasks = []
        for i in range(20):
            batch_tasks.append(processor.submit(f"item{i}"))

        await asyncio.gather(*batch_tasks)
        end_time = time.time()

        batch_execution_time = (end_time - start_time) * 1000  # Convert to ms
        test_results["optimizations"]["batch_execution_time_ms"] = batch_execution_time
    except ImportError:
        logger.warning("BatchProcessor not available, skipping batch processing test")

    logger.info("Completed query optimization evaluation")


def analyze_results_and_recommend():
    """Analyze test results and make recommendations."""
    logger.info("Analyzing results and generating recommendations")

    recommendations = []

    # Check if we have a large database
    is_large_db = test_results.get("message_count", 0) > 100000 or test_results.get("database_size", 0) > 100_000_000  # 100MB


    # Analyze query cache effectiveness
    cache_hit_rate = test_results.get("optimizations", {}).get("cache_hit_rate", 0)
    if cache_hit_rate < 50:
        recommendations.append({
            "area": "Caching",
            "issue": "Low cache hit rate",
            "recommendation": "Adjust cache TTL settings or review cache invalidation strategy.",
            "priority": "Medium",
        })

    # Analyze memory usage
    memory_deltas = test_results.get("memory", {}).get("deltas", {})
    max_operation = max(memory_deltas.items(), key=lambda x: x[1], default=(None, 0))

    if max_operation[1] > 500:  # If any operation uses more than 500MB
        recommendations.append({
            "area": "Memory Management",
            "issue": f"High memory usage in {max_operation[0]} operation",
            "recommendation": "Consider implementing pagination or data streaming for large result sets.",
            "priority": "High",
        })

    # Analyze response times
    slow_operations = []
    for test_name, test_data in test_results.get("tests", {}).items():
        if test_data.get("avg_time_no_cache_ms", 0) > 1000:  # More than 1 second
            slow_operations.append((test_name, test_data.get("avg_time_no_cache_ms", 0)))

    if slow_operations:
        for op_name, time_ms in slow_operations:
            recommendations.append({
                "area": "Query Optimization",
                "issue": f"Slow operation: {op_name} ({time_ms:.2f}ms)",
                "recommendation": f"Review SQL execution plan and add indexes to speed up {op_name} operation.",
                "priority": "High",
            })

    # Database sharding recommendation for large DBs
    if is_large_db:
        recommendations.append({
            "area": "Database Sharding",
            "issue": "Large database size may impact performance",
            "recommendation": "Consider enabling database sharding for better performance with large datasets.",
            "priority": "Medium",
        })

    # Add recommendations to results
    test_results["recommendations"] = recommendations

    # Log recommendations
    logger.info("Performance test recommendations:")
    for rec in recommendations:
        logger.info(f"  [{rec['priority']}] {rec['area']}: {rec['recommendation']}")


async def run_tests(args):
    """Run performance tests based on command line arguments."""
    logger.info("Starting performance testing")

    # Initialize modules
    init_concurrency()

    # Get basic database stats
    db_stats = await get_database_stats()
    test_results.update(db_stats)

    logger.info(f"Database stats: {db_stats['message_count']} messages, {db_stats['contact_count']} contacts, {db_stats['database_size_bytes'] / 1024 / 1024:.2f} MB")

    # Run selected tests
    if args.all or args.large_db:
        await test_large_database_performance()

    if args.all or args.response_times:
        await measure_response_times()

    if args.all or args.memory:
        await profile_memory_usage()

    if args.all or args.optimization:
        await evaluate_query_optimizations()

    # Analyze results and make recommendations
    analyze_results_and_recommend()

    # Save results to file
    with open(args.output, 'w') as f:
        json.dump(test_results, f, indent=2, default=str)

    logger.info(f"Performance test results saved to {args.output}")

    # Print summary
    print("\n" + "="*80)
    print("PERFORMANCE TEST SUMMARY")
    print("="*80)
    print(f"Database size: {db_stats['database_size_bytes'] / 1024 / 1024:.2f} MB")
    print(f"Messages: {db_stats['message_count']}")
    print(f"Contacts: {db_stats['contact_count']}")
    print("\nKey performance metrics:")

    # Response times
    for test_name, test_data in test_results.get("tests", {}).items():
        if "avg_time_no_cache_ms" in test_data and "avg_time_with_cache_ms" in test_data:
            print(f"- {test_name}: {test_data['avg_time_no_cache_ms']:.2f}ms → {test_data['avg_time_with_cache_ms']:.2f}ms with cache ({test_data.get('cache_time_improvement_pct', 0):.1f}% improvement)")

    # Cache effectiveness
    if "optimizations" in test_results and "cache_hit_rate" in test_results["optimizations"]:
        print(f"\nCache hit rate: {test_results['optimizations']['cache_hit_rate']:.1f}%")

    # Memory usage
    if "memory" in test_results and "deltas" in test_results["memory"]:
        print("\nMemory usage by operation:")
        for op, delta in test_results["memory"]["deltas"].items():
            print(f"- {op}: {delta:.2f} MB")

    # Recommendations
    if test_results.get("recommendations"):
        print("\nRecommendations:")
        for rec in test_results["recommendations"]:
            print(f"- [{rec['priority']}] {rec['area']}: {rec['recommendation']}")

    print("="*80)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Run performance tests for iMessage MCP Server")
    parser.add_argument("--large-db", action="store_true", help="Run large database performance tests")
    parser.add_argument("--response-times", action="store_true", help="Measure response times")
    parser.add_argument("--memory", action="store_true", help="Profile memory usage")
    parser.add_argument("--optimization", action="store_true", help="Evaluate query optimizations")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--output", default="performance_test_results.json", help="Output file for test results")

    args = parser.parse_args()

    # If no specific tests selected, run all
    if not (args.large_db or args.response_times or args.memory or args.optimization):
        args.all = True

    # Run tests
    asyncio.run(run_tests(args))


if __name__ == "__main__":
    main()
