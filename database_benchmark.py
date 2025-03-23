#!/usr/bin/env python3
"""
iMessage Database Performance Benchmark Tool

This utility measures the performance of common database operations to quantify optimization improvements.
It can run in two modes: standard mode using the original database, and optimized mode using the indexed copy.

Usage:
    python database_benchmark.py [options]

Options:
    --db-path PATH     Path to the database to benchmark (default: ~/Library/Messages/chat.db)
    --indexed-db PATH  Path to the indexed database (default: ~/.imessage_insights/indexed_chat.db)
    --output FILE      Path to save benchmark results (default: benchmark_results.json)
    --iterations N     Number of iterations for each test (default: 5)
    --compare          Run benchmarks on both databases and compare results
    --quiet            Suppress detailed progress output
"""

import os
import sys
import time
import json
import sqlite3
import asyncio
import argparse
import logging
import statistics
import aiosqlite
from pathlib import Path
from datetime import datetime
from tabulate import tabulate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Default paths
HOME = os.path.expanduser("~")
DEFAULT_DB_PATH = Path(f"{HOME}/Library/Messages/chat.db")
DEFAULT_INDEXED_PATH = Path(f"{HOME}/.imessage_insights/indexed_chat.db")
DEFAULT_OUTPUT_PATH = Path("benchmark_results.json")

class DatabaseBenchmark:
    """Utility to benchmark database performance."""
    
    def __init__(self, db_path=None, indexed_db_path=None, output_path=None, iterations=5, quiet=False):
        """Initialize the benchmark tool with paths and options.
        
        Args:
            db_path: Path to the original database
            indexed_db_path: Path to the indexed database
            output_path: Path to save benchmark results
            iterations: Number of iterations for each test
            quiet: Suppress detailed progress output
        """
        self.db_path = Path(db_path or DEFAULT_DB_PATH)
        self.indexed_db_path = Path(indexed_db_path or DEFAULT_INDEXED_PATH)
        self.output_path = Path(output_path or DEFAULT_OUTPUT_PATH)
        self.iterations = iterations
        self.quiet = quiet
        self.results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "db_path": str(self.db_path),
                "indexed_db_path": str(self.indexed_db_path),
                "iterations": self.iterations,
            },
            "standard_db": {},
            "indexed_db": {},
            "comparison": {}
        }
        
    def log(self, message, level=logging.INFO):
        """Log a message if not in quiet mode."""
        if not self.quiet:
            logger.log(level, message)
    
    async def run_query_benchmark(self, conn, query, params=None, description=None):
        """Run a benchmark for a single query.
        
        Args:
            conn: Database connection
            query: SQL query to benchmark
            params: Query parameters
            description: Description of the test
            
        Returns:
            Dict with benchmark results
        """
        if params is None:
            params = {}
            
        results = []
        
        for i in range(self.iterations):
            self.log(f"Running iteration {i+1}/{self.iterations} for: {description}")
            
            # Clear cache by running PRAGMA
            await conn.execute("PRAGMA cache_size = 0")
            await conn.execute("PRAGMA cache_size = 10000")
            
            # Measure query execution time
            start_time = time.time()
            cursor = await conn.execute(query, params)
            rows = await cursor.fetchall()
            duration = time.time() - start_time
            
            results.append({
                "duration": duration,
                "row_count": len(rows)
            })
            
            # Wait a bit between iterations to reduce system load effects
            await asyncio.sleep(0.1)
        
        # Calculate statistics
        durations = [r["duration"] for r in results]
        
        return {
            "description": description,
            "query": query,
            "min_time": min(durations),
            "max_time": max(durations),
            "avg_time": statistics.mean(durations),
            "median_time": statistics.median(durations),
            "iterations": self.iterations,
            "row_count": results[0]["row_count"]
        }
    
    async def benchmark_database(self, db_path, db_type="standard_db"):
        """Run benchmarks on a database.
        
        Args:
            db_path: Path to the database
            db_type: Type of database ('standard_db' or 'indexed_db')
            
        Returns:
            Dict with benchmark results
        """
        self.log(f"Benchmarking {db_type} at {db_path}")
        
        # Define benchmark queries
        benchmarks = [
            {
                "name": "recent_messages",
                "description": "Recent messages (100)",
                "query": "SELECT message.ROWID, message.date, message.text FROM message ORDER BY message.date DESC LIMIT 100"
            },
            {
                "name": "chat_messages",
                "description": "Messages in a specific chat",
                "query": """
                    SELECT message.ROWID, message.date, message.text 
                    FROM message 
                    JOIN chat_message_join ON message.ROWID = chat_message_join.message_id 
                    WHERE chat_message_join.chat_id = ? 
                    ORDER BY message.date DESC LIMIT 100
                """,
                "params": (1,)  # Use the first chat ID
            },
            {
                "name": "text_search",
                "description": "Text search (basic)",
                "query": "SELECT message.ROWID, message.date, message.text FROM message WHERE message.text LIKE ? LIMIT 100",
                "params": ("%the%",)
            },
            {
                "name": "fts_search",
                "description": "Full-text search (if available)",
                "query": """
                    SELECT message.ROWID, message.date, message.text 
                    FROM message_fts 
                    JOIN message ON message_fts.rowid = message.ROWID 
                    WHERE message_fts MATCH ? 
                    ORDER BY rank 
                    LIMIT 100
                """,
                "params": ("the",)
            },
            {
                "name": "date_range",
                "description": "Messages in date range",
                "query": """
                    SELECT message.ROWID, message.date, message.text 
                    FROM message 
                    WHERE message.date BETWEEN ? AND ? 
                    ORDER BY message.date DESC LIMIT 100
                """,
                "params": (600000000000000, 700000000000000)
            },
            {
                "name": "contact_messages",
                "description": "Messages from a contact",
                "query": """
                    SELECT message.ROWID, message.date, message.text 
                    FROM message 
                    JOIN handle ON message.handle_id = handle.ROWID 
                    WHERE handle.id LIKE ? 
                    ORDER BY message.date DESC LIMIT 100
                """,
                "params": ("%@%",)  # Match email addresses
            },
            {
                "name": "message_count",
                "description": "Count messages per contact",
                "query": """
                    SELECT handle.id, COUNT(message.ROWID) as msg_count 
                    FROM message 
                    JOIN handle ON message.handle_id = handle.ROWID 
                    GROUP BY handle.id 
                    ORDER BY msg_count DESC 
                    LIMIT 50
                """
            },
            {
                "name": "complex_join",
                "description": "Complex multi-table join",
                "query": """
                    SELECT chat.display_name, handle.id, message.text, message.date 
                    FROM chat 
                    JOIN chat_message_join ON chat.ROWID = chat_message_join.chat_id 
                    JOIN message ON chat_message_join.message_id = message.ROWID 
                    JOIN chat_handle_join ON chat.ROWID = chat_handle_join.chat_id 
                    JOIN handle ON chat_handle_join.handle_id = handle.ROWID 
                    WHERE message.is_from_me = 0 
                    ORDER BY message.date DESC 
                    LIMIT 50
                """
            }
        ]
        
        try:
            # Connect to the database
            conn = await aiosqlite.connect(db_path)
            conn.row_factory = aiosqlite.Row
            
            # Check if FTS is available
            has_fts = False
            try:
                cursor = await conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='message_fts'")
                result = await cursor.fetchone()
                has_fts = bool(result)
                self.log(f"FTS support: {has_fts}")
            except Exception as e:
                self.log(f"Error checking FTS support: {e}", level=logging.WARNING)
                
            # Run benchmarks
            results = {}
            for benchmark in benchmarks:
                name = benchmark["name"]
                
                # Skip FTS test if not available
                if name == "fts_search" and not has_fts:
                    self.log("Skipping FTS test as it's not available", level=logging.WARNING)
                    continue
                    
                self.log(f"Running benchmark: {benchmark['description']}")
                try:
                    result = await self.run_query_benchmark(
                        conn, 
                        benchmark["query"], 
                        benchmark.get("params"),
                        benchmark["description"]
                    )
                    results[name] = result
                except Exception as e:
                    self.log(f"Error running benchmark {name}: {e}", level=logging.ERROR)
                    results[name] = {"error": str(e)}
            
            # Close connection
            await conn.close()
            
            # Store results
            self.results[db_type] = results
            
            return results
            
        except Exception as e:
            self.log(f"Error benchmarking database: {e}", level=logging.ERROR)
            return {"error": str(e)}
            
    def compare_results(self):
        """Compare performance between standard and indexed databases."""
        if not self.results["standard_db"] or not self.results["indexed_db"]:
            self.log("Missing results for comparison", level=logging.ERROR)
            return
            
        comparison = {}
        
        for test_name in self.results["standard_db"]:
            if test_name in self.results["indexed_db"]:
                standard = self.results["standard_db"][test_name]
                indexed = self.results["indexed_db"][test_name]
                
                # Skip if there was an error
                if "error" in standard or "error" in indexed:
                    comparison[test_name] = {
                        "description": standard.get("description", "Unknown"),
                        "error": "Error occurred during benchmarking"
                    }
                    continue
                
                # Calculate improvement
                standard_time = standard["avg_time"]
                indexed_time = indexed["avg_time"]
                
                if standard_time > 0:
                    improvement = ((standard_time - indexed_time) / standard_time) * 100
                else:
                    improvement = 0
                    
                comparison[test_name] = {
                    "description": standard["description"],
                    "standard_time": standard_time,
                    "indexed_time": indexed_time,
                    "improvement_pct": improvement,
                    "improvement_factor": standard_time / indexed_time if indexed_time > 0 else float('inf')
                }
        
        self.results["comparison"] = comparison
        return comparison
        
    def print_comparison_table(self):
        """Print a formatted table of benchmark comparisons."""
        if not self.results["comparison"]:
            self.log("No comparison data available", level=logging.ERROR)
            return
            
        table_data = []
        for test_name, data in self.results["comparison"].items():
            if "error" in data:
                continue
                
            table_data.append([
                data["description"],
                f"{data['standard_time']*1000:.2f}ms",
                f"{data['indexed_time']*1000:.2f}ms",
                f"{data['improvement_pct']:.1f}%",
                f"{data['improvement_factor']:.1f}x"
            ])
            
        # Sort by improvement percentage
        table_data.sort(key=lambda x: float(x[3].replace('%', '')), reverse=True)
        
        print("\nBenchmark Results Comparison:")
        print(tabulate(
            table_data,
            headers=["Test", "Standard DB", "Indexed DB", "Improvement", "Factor"],
            tablefmt="grid"
        ))
        
    def save_results(self):
        """Save benchmark results to a JSON file."""
        try:
            # Create directory if it doesn't exist
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.output_path, 'w') as f:
                json.dump(self.results, f, indent=2)
                
            self.log(f"Benchmark results saved to {self.output_path}")
            return True
        except Exception as e:
            self.log(f"Error saving benchmark results: {e}", level=logging.ERROR)
            return False
            
    async def run_benchmarks(self, compare=False):
        """Run the benchmarks.
        
        Args:
            compare: Whether to run benchmarks on both databases
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if databases exist
            if not self.db_path.exists():
                self.log(f"Database not found: {self.db_path}", level=logging.ERROR)
                return False
                
            if compare and not self.indexed_db_path.exists():
                self.log(f"Indexed database not found: {self.indexed_db_path}", level=logging.ERROR)
                return False
                
            # Run benchmarks on standard database
            self.log(f"Starting benchmarks on standard database")
            await self.benchmark_database(self.db_path, "standard_db")
            
            # Run benchmarks on indexed database if requested
            if compare:
                self.log(f"Starting benchmarks on indexed database")
                await self.benchmark_database(self.indexed_db_path, "indexed_db")
                
                # Compare results
                self.log("Comparing results")
                self.compare_results()
                
                # Print comparison table
                self.print_comparison_table()
            
            # Save results
            self.save_results()
            
            return True
            
        except Exception as e:
            self.log(f"Error running benchmarks: {e}", level=logging.ERROR)
            return False

async def main():
    """Main entry point for the benchmark utility."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="iMessage Database Performance Benchmark Tool")
    parser.add_argument("--db-path", type=str, help=f"Path to the database to benchmark (default: {DEFAULT_DB_PATH})")
    parser.add_argument("--indexed-db", type=str, help=f"Path to the indexed database (default: {DEFAULT_INDEXED_PATH})")
    parser.add_argument("--output", type=str, help=f"Path to save benchmark results (default: {DEFAULT_OUTPUT_PATH})")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations for each test (default: 5)")
    parser.add_argument("--compare", action="store_true", help="Run benchmarks on both databases and compare results")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed progress output")
    
    args = parser.parse_args()
    
    # Create and run the benchmark
    benchmark = DatabaseBenchmark(
        db_path=args.db_path,
        indexed_db_path=args.indexed_db,
        output_path=args.output,
        iterations=args.iterations,
        quiet=args.quiet
    )
    
    success = await benchmark.run_benchmarks(compare=args.compare)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))