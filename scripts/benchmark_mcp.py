#!/usr/bin/env python3
"""
Performance benchmarking script for iMessage MCP Server.

This script measures the performance of various MCP tools to ensure
they meet the p95 < 1.5s target.
"""

import asyncio
import json
import logging
import statistics
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_server.config import load_config
from mcp_server.consent import ConsentManager
from mcp_server.db import get_database
from mcp_server.privacy import hash_contact_id, redact_pii

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PerformanceBenchmark:
    """Benchmark tool performance."""

    def __init__(self):
        self.results = {}
        self.config = None
        self.db = None

    async def setup(self):
        """Set up benchmark environment."""
        logger.info("Setting up benchmark environment...")

        # Load config
        self.config = load_config()

        # Get database
        try:
            self.db = await get_database()
            logger.info("Database connected")
        except Exception as e:
            logger.warning(f"Could not connect to database: {e}")
            self.db = None

        # Grant consent for testing
        consent = ConsentManager()
        await consent.grant_consent(expires_hours=1)
        logger.info("Consent granted for testing")

    async def cleanup(self):
        """Clean up after benchmarks."""
        if self.db:
            await self.db.close()

        # Revoke consent
        consent = ConsentManager()
        await consent.revoke_consent()

    async def measure_operation(self, name: str, operation, iterations: int = 10) -> Dict[str, float]:
        """Measure the performance of an operation."""
        times = []
        errors = 0

        for i in range(iterations):
            try:
                start = time.time()
                await operation()
                end = time.time()
                times.append(end - start)
            except Exception as e:
                logger.error(f"Error in {name}: {e}")
                errors += 1

        if not times:
            return {
                "error_rate": 1.0,
                "errors": errors
            }

        return {
            "min": min(times),
            "max": max(times),
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "p50": statistics.quantiles(times, n=100)[49] if len(times) > 2 else times[0],
            "p95": statistics.quantiles(times, n=100)[94] if len(times) > 2 else times[-1],
            "p99": statistics.quantiles(times, n=100)[98] if len(times) > 2 else times[-1],
            "iterations": len(times),
            "error_rate": errors / iterations,
            "errors": errors
        }

    async def benchmark_hashing(self):
        """Benchmark contact ID hashing."""
        test_ids = [
            "+1-555-123-4567",
            "user@example.com",
            "some_apple_id",
            "+44-20-1234-5678",
            "another.email@domain.co.uk"
        ]

        async def hash_operation():
            for test_id in test_ids:
                hash_contact_id(test_id, self.config.session_salt)

        return await self.measure_operation("Contact ID Hashing", hash_operation, 100)

    async def benchmark_redaction(self):
        """Benchmark PII redaction."""
        test_texts = [
            "Call me at 555-123-4567",
            "My email is john.doe@example.com",
            "Send $1,234.56 to account 1234-5678-9012-3456",
            "I live at 123 Main Street, Anytown, USA",
            "My SSN is 123-45-6789 and I need help"
        ]

        async def redaction_operation():
            for text in test_texts:
                redact_pii(text)

        return await self.measure_operation("PII Redaction", redaction_operation, 100)

    async def benchmark_db_query(self):
        """Benchmark database queries."""
        if not self.db:
            return {"skipped": True, "reason": "No database connection"}

        async def query_operation():
            # Simple count query
            await self.db.execute_query("SELECT COUNT(*) FROM message LIMIT 1")

        return await self.measure_operation("Simple DB Query", query_operation, 50)

    async def benchmark_contact_stats(self):
        """Benchmark contact statistics query."""
        if not self.db:
            return {"skipped": True, "reason": "No database connection"}

        async def stats_operation():
            query = """
            SELECT 
                h.id,
                COUNT(*) as message_count
            FROM message m
            JOIN handle h ON m.handle_id = h.ROWID
            GROUP BY h.id
            ORDER BY message_count DESC
            LIMIT 10
            """
            await self.db.execute_query(query)

        return await self.measure_operation("Contact Stats Query", stats_operation, 10)

    async def benchmark_date_range_query(self):
        """Benchmark date range queries."""
        if not self.db:
            return {"skipped": True, "reason": "No database connection"}

        # Calculate timestamp for 30 days ago
        start_date = datetime.now() - timedelta(days=30)
        start_timestamp = int((start_date.timestamp() - 978307200) * 1e9)

        async def date_query_operation():
            query = """
            SELECT COUNT(*) as count
            FROM message
            WHERE date >= ?
            """
            await self.db.execute_query(query, (start_timestamp,))

        return await self.measure_operation("Date Range Query", date_query_operation, 10)

    async def benchmark_memory_usage(self):
        """Measure memory usage."""
        try:
            import psutil
            process = psutil.Process()

            # Get memory info
            mem_info = process.memory_info()

            return {
                "rss_mb": mem_info.rss / (1024 * 1024),
                "vms_mb": mem_info.vms / (1024 * 1024),
                "available": True
            }
        except ImportError:
            return {"available": False, "reason": "psutil not installed"}

    async def run_all_benchmarks(self):
        """Run all benchmarks."""
        benchmarks = [
            ("Hashing Performance", self.benchmark_hashing),
            ("PII Redaction", self.benchmark_redaction),
            ("Simple DB Query", self.benchmark_db_query),
            ("Contact Stats Query", self.benchmark_contact_stats),
            ("Date Range Query", self.benchmark_date_range_query),
        ]

        for name, benchmark_func in benchmarks:
            logger.info(f"Running benchmark: {name}")
            result = await benchmark_func()
            self.results[name] = result

        # Check memory usage
        self.results["Memory Usage"] = await self.benchmark_memory_usage()

    def print_results(self):
        """Print benchmark results."""
        print("\n" + "=" * 60)
        print("PERFORMANCE BENCHMARK RESULTS")
        print("=" * 60)

        # Performance targets
        p95_target = 1.5  # seconds
        memory_target = 250  # MB

        all_passed = True

        for name, result in self.results.items():
            print(f"\n{name}:")
            print("-" * 40)

            if "skipped" in result and result["skipped"]:
                print(f"  SKIPPED: {result['reason']}")
                continue

            if name == "Memory Usage":
                if result["available"]:
                    rss_mb = result["rss_mb"]
                    print(f"  RSS Memory: {rss_mb:.1f} MB")
                    print(f"  VMS Memory: {result['vms_mb']:.1f} MB")

                    if rss_mb > memory_target:
                        print(f"  ❌ FAILED: Memory usage ({rss_mb:.1f} MB) exceeds target ({memory_target} MB)")
                        all_passed = False
                    else:
                        print("  ✅ PASSED: Memory usage within target")
                else:
                    print(f"  SKIPPED: {result['reason']}")
            else:
                # Performance metrics
                if "error_rate" in result and result["error_rate"] == 1.0:
                    print("  ❌ FAILED: All operations failed")
                    all_passed = False
                    continue

                print(f"  Iterations: {result.get('iterations', 0)}")
                print(f"  Error rate: {result.get('error_rate', 0):.1%}")

                if "p95" in result:
                    p95_ms = result["p95"] * 1000
                    print(f"  p50: {result['p50']*1000:.1f} ms")
                    print(f"  p95: {p95_ms:.1f} ms")
                    print(f"  p99: {result['p99']*1000:.1f} ms")
                    print(f"  Min: {result['min']*1000:.1f} ms")
                    print(f"  Max: {result['max']*1000:.1f} ms")

                    # Check against target
                    if result["p95"] > p95_target:
                        print(f"  ❌ FAILED: p95 ({p95_ms:.1f} ms) exceeds target ({p95_target*1000:.1f} ms)")
                        all_passed = False
                    else:
                        print("  ✅ PASSED: p95 within target")

        print("\n" + "=" * 60)
        if all_passed:
            print("✅ ALL PERFORMANCE TARGETS MET")
        else:
            print("❌ SOME PERFORMANCE TARGETS FAILED")
        print("=" * 60)

        return all_passed


async def main():
    """Run performance benchmarks."""
    benchmark = PerformanceBenchmark()

    try:
        await benchmark.setup()
        await benchmark.run_all_benchmarks()
        all_passed = benchmark.print_results()

        # Save results to file
        results_file = Path("benchmark_results.json")
        with open(results_file, "w") as f:
            json.dump(benchmark.results, f, indent=2, default=str)
        print(f"\nResults saved to: {results_file}")

        return 0 if all_passed else 1

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1
    finally:
        await benchmark.cleanup()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
