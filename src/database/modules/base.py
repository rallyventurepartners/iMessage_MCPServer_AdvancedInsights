"""
Base database module that provides core functionality for database operations.

This module contains the foundational components for database access including:
- Connection pool management
- Initialization and cleanup
- Core query execution
- Transaction handling
- Error handling and recovery
"""

import os
import logging
import traceback
import asyncio
import aiosqlite
import sqlite3
from pathlib import Path
import stat
import time
import psutil
import math
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

# Import the Redis cache
from src.utils.redis_cache import AsyncRedisCache

# Configure logging
logger = logging.getLogger(__name__)

# Default database path for macOS
HOME = os.path.expanduser("~")
DB_PATH = Path(f"{HOME}/Library/Messages/chat.db")

class DatabaseError(Exception):
    """Custom exception for database-related errors."""
    pass

class ConnectionHealthStatus:
    """Status enumeration for connection health."""
    HEALTHY = "healthy"
    SLOW = "slow"
    UNRESPONSIVE = "unresponsive"
    ERROR = "error"

class ConnectionStats:
    """Class to track connection statistics and health."""
    
    def __init__(self, conn_id):
        self.conn_id = conn_id
        self.created_at = time.time()
        self.last_used_at = time.time()
        self.total_queries = 0
        self.total_query_time = 0.0
        self.errors = 0
        self.consecutive_errors = 0
        self.health_status = ConnectionHealthStatus.HEALTHY
        self.slow_queries = 0
        self.query_times = []  # Track recent query times for averaging
        self.max_query_times = 50  # Keep only the last 50 query times
    
    def record_query(self, query_time):
        """Record statistics for a completed query."""
        self.total_queries += 1
        self.total_query_time += query_time
        self.last_used_at = time.time()
        self.consecutive_errors = 0
        
        # Add to recent query times
        self.query_times.append(query_time)
        if len(self.query_times) > self.max_query_times:
            self.query_times.pop(0)  # Remove oldest
        
        # Check if this was a slow query
        if query_time > 1.0:  # More than 1 second is considered slow
            self.slow_queries += 1
        
        # Update health status
        self._update_health()
    
    def record_error(self):
        """Record a query error."""
        self.errors += 1
        self.consecutive_errors += 1
        self.last_used_at = time.time()
        
        # Update health status
        self._update_health()
    
    def _update_health(self):
        """Update the connection health status based on metrics."""
        # Check consecutive errors
        if self.consecutive_errors >= 3:
            self.health_status = ConnectionHealthStatus.ERROR
            return
        
        # Check average query time if we have enough data
        if len(self.query_times) >= 5:
            avg_query_time = sum(self.query_times[-5:]) / 5
            if avg_query_time > 2.0:
                self.health_status = ConnectionHealthStatus.SLOW
                return
        
        # Check if this connection hasn't been used for a long time
        idle_time = time.time() - self.last_used_at
        if idle_time > 300:  # 5 minutes idle
            self.health_status = ConnectionHealthStatus.UNRESPONSIVE
            return
        
        # Connection appears healthy
        self.health_status = ConnectionHealthStatus.HEALTHY

    def get_average_query_time(self):
        """Get the average query time for recent queries."""
        if not self.query_times:
            return 0.0
        return sum(self.query_times) / len(self.query_times)
    
    def get_age(self):
        """Get the age of this connection in seconds."""
        return time.time() - self.created_at

class AsyncDatabaseBase:
    """Base class for asynchronous database operations.
    
    This class provides the foundation for all database operations including
    connection pool management, initialization, and core query functionality.
    """
    
    _instance = None
    _lock = asyncio.Lock()
    _connection_pool = []
    _min_connections = 2
    _max_connections = 15
    _init_complete = False
    
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
    
    def __init__(self, db_path=None, minimal_mode=False):
        """Initialize the database connection.
        
        Args:
            db_path: Path to the SQLite database file. If None, uses the default macOS path.
            minimal_mode: Start in minimal mode for faster performance with less data.
        """
        self.db_path = db_path or DB_PATH
        self._lock = asyncio.Lock()
        self._connection_pool = []
        self._busy_connections = set()
        self.initialized = False
        
        # Connection pool configuration
        self._min_connections = 2  # Minimum number of connections to maintain
        self._initial_connections = 3  # Initial connections to create
        self._max_connections = 15  # Maximum number of connections in the pool
        self._connection_timeout = 5.0  # Seconds to wait for a connection
        self._connection_ttl = 600  # Seconds to keep a connection before refreshing (10 minutes)
        self._connection_timestamps = {}  # Track when connections were created
        self._connection_acquisition_timeout = 10.0  # Timeout for acquiring a connection
        self._connection_monitor_task = None  # Task for monitoring connection pool health
        self._connection_scaling_task = None  # Task for dynamic sizing of the pool
        self._connection_usage_stats = {}  # Track usage statistics per connection
        self._dynamic_scaling_enabled = True  # Enable/disable dynamic scaling
        self._load_metrics = {}  # Track load metrics for the connection pool
        self._last_health_check = time.time()
        self._connection_ids = 0  # Counter for connection IDs
        self._slow_query_threshold = 1.0  # Seconds
        self._query_history = []  # Track recent queries for analysis
        self._max_query_history = 100  # Maximum query history to keep
        
        # Performance metrics
        self._usage_count = 0  # Count of connection uses
        self._peak_usage = 0  # Peak number of connections used simultaneously
        self._waiting_count = 0  # Count of waiters for connections
        self._peak_waiting = 0  # Peak number of waiters
        self._total_wait_time = 0.0  # Total time spent waiting for connections
        self._total_query_time = 0.0  # Total time spent in queries
        self._total_queries = 0  # Total number of queries executed
        self._slow_queries = 0  # Count of slow queries
        self._connection_acquisition_times = []  # Track times to acquire connections
        
        # Prepared statement cache
        self._prepared_statements = {}
        self._statement_usage_count = {}
        self._max_prepared_statements = 100
        
        # Initialize cache
        self.cache = AsyncRedisCache()
        
        # Set minimal mode based on parameter
        self.minimal_mode = minimal_mode
        
        # Track if the database has FTS capabilities
        self._has_fts5 = False
        
        # Check if this is an indexed database
        self.is_indexed_db = False
        
        # State for rate limiting and backpressure
        self._query_rate_limiter = asyncio.Semaphore(25)  # Max concurrent queries
        self._backpressure_enabled = True  # Whether to enable backpressure
        self._backpressure_high_threshold = 0.9  # High watermark (90% of max connections)
        self._backpressure_low_threshold = 0.7  # Low watermark (70% of max connections)
        self._backpressure_active = False  # Whether backpressure is currently active
    
    def check_database_file(self, db_path):
        """Check if the database file is accessible and log its properties.
        
        Args:
            db_path: Path to the database file
            
        Returns:
            bool: True if the file exists and is readable, False otherwise
        """
        if not os.path.exists(db_path):
            logger.error(f"Database file does not exist: {db_path}")
            return False
            
        # Check read permissions
        if not os.access(db_path, os.R_OK):
            logger.error(f"No read permission for database file: {db_path}")
            return False
            
        # Check file size
        file_size = os.path.getsize(db_path)
        logger.info(f"iMessage database size: {file_size / (1024*1024):.2f} MB")
        
        # Check file permissions
        permissions = stat.S_IMODE(os.stat(db_path).st_mode)
        logger.info(f"Database file permissions: {oct(permissions)}")
        
        # Check if this is a possibly indexed database (based on path)
        indexed_indicators = [".imessage_insights", "indexed", "index"]
        if any(indicator in str(db_path).lower() for indicator in indexed_indicators):
            logger.info(f"This appears to be an indexed database: {db_path}")
            self.is_indexed_db = True
        
        return True
    
    async def initialize(self):
        """Initialize the database connection pool asynchronously.
        
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
            if not self.check_database_file(self.db_path):
                raise FileNotFoundError(f"Database file not found or not accessible at {self.db_path}")
            
            # Dynamically adjust connection pool size based on available system resources
            self._adjust_pool_size_from_system()
            
            # Initialize connections
            try:
                # Create initial pool connections
                created_count = 0
                for i in range(self._initial_connections):
                    try:
                        await self._create_new_connection()
                        created_count += 1
                    except Exception as e:
                        logger.error(f"Error creating pool connection {i}: {e}")
                
                if created_count == 0:
                    raise DatabaseError("Failed to create any initial database connections")
                
                # Start connection pool monitor
                if self._connection_monitor_task is None:
                    self._connection_monitor_task = asyncio.create_task(
                        self._monitor_connection_pool()
                    )
                    self._connection_monitor_task.set_name("db_connection_monitor")
                
                # Start connection scaling task if dynamic scaling is enabled
                if self._dynamic_scaling_enabled and self._connection_scaling_task is None:
                    self._connection_scaling_task = asyncio.create_task(
                        self._dynamic_connection_scaling()
                    )
                    self._connection_scaling_task.set_name("db_connection_scaling")
                
                self.initialized = True
                logger.info(f"Database initialized successfully: {self.db_path}")
                logger.info(f"Connection pool: min={self._min_connections}, "
                           f"initial={self._initial_connections}, max={self._max_connections}")
                
            except aiosqlite.Error as e:
                raise DatabaseError(f"Error connecting to database: {e}")
                
            except Exception as e:
                logger.error(f"Error during database initialization: {e}")
                logger.error(traceback.format_exc())
                raise DatabaseError(f"Unexpected error during database initialization: {e}")
    
    def _adjust_pool_size_from_system(self):
        """Adjust connection pool size based on system resources."""
        try:
            # Get available system memory
            mem = psutil.virtual_memory()
            available_mem_gb = mem.available / (1024 * 1024 * 1024)
            
            # Get CPU count
            cpu_count = psutil.cpu_count(logical=True)
            
            # Calculate appropriate connection pool size
            # Each SQLite connection can use ~10-30MB of memory depending on cache settings
            # We'll aim to use at most 10% of available memory for connections
            max_by_memory = min(math.floor(available_mem_gb * 100), 50)  # Max 50 connections
            
            # Consider CPU count - typically 1-2 connections per CPU is reasonable for read ops
            max_by_cpu = cpu_count * 2
            
            # Take the smaller of the two limits
            new_max = min(max_by_memory, max_by_cpu)
            
            # Ensure reasonable minimum
            self._max_connections = max(5, min(new_max, 30))  # Between 5 and 30
            self._min_connections = max(2, self._max_connections // 5)  # At least 2, or 20% of max
            self._initial_connections = min(self._max_connections // 2, 5)  # 50% of max, up to 5
            
            logger.info(f"Adjusted connection pool based on system resources: "
                      f"min={self._min_connections}, max={self._max_connections}, "
                      f"initial={self._initial_connections}")
            
        except Exception as e:
            logger.warning(f"Error adjusting connection pool size from system resources: {e}")
            # Use default values if adjustment fails
    
    async def _create_new_connection(self):
        """Create a new database connection and add it to the pool.
        
        Returns:
            The newly created connection
            
        Raises:
            DatabaseError: If there's an error creating the connection
        """
        try:
            # Use URI connection string with read-only mode
            uri = f"file:{self.db_path}?mode=ro&immutable=1"
            
            # Create new connection
            connection = await aiosqlite.connect(uri, uri=True)
            
            # Configure connection
            await connection.execute("PRAGMA temp_store = MEMORY")
            await connection.execute("PRAGMA journal_mode = OFF")
            await connection.execute("PRAGMA synchronous = OFF")
            await connection.execute("PRAGMA cache_size = 10000")
            await connection.execute("PRAGMA busy_timeout = 5000")  # 5 second busy timeout
            
            # Set row factory for named columns
            connection.row_factory = aiosqlite.Row
            
            # Generate a unique connection ID for tracking
            self._connection_ids += 1
            conn_id = self._connection_ids
            
            # Store connection metadata
            connection._id = conn_id  # Attach ID to connection object
            connection._created_at = time.time()
            
            # Initialize connection statistics
            self._connection_usage_stats[conn_id] = ConnectionStats(conn_id)
            
            # Check if the database has FTS5 support (only on first connection)
            if len(self._connection_pool) == 0:
                try:
                    cursor = await connection.execute("SELECT sqlite_version()")
                    version = await cursor.fetchone()
                    logger.info(f"SQLite version: {version[0]}")
                    
                    # Check for FTS5 module
                    cursor = await connection.execute("SELECT name FROM pragma_module_list WHERE name='fts5'")
                    result = await cursor.fetchone()
                    self._has_fts5 = bool(result)
                    logger.info(f"FTS5 support: {self._has_fts5}")
                except Exception as e:
                    logger.warning(f"Error checking SQLite version or FTS5 support: {e}")
                    self._has_fts5 = False
            
            # Add to pool
            self._connection_pool.append(connection)
            self._connection_timestamps[connection] = time.time()
            
            logger.debug(f"Created new database connection (id={conn_id}, pool size={len(self._connection_pool)})")
            return connection
            
        except Exception as e:
            logger.error(f"Error creating new database connection: {e}")
            raise DatabaseError(f"Failed to create new database connection: {e}")
    
    async def _monitor_connection_pool(self):
        """Monitor the connection pool for health and cleanup.
        
        This task runs in the background and:
        1. Closes old connections past their TTL
        2. Checks for leaked connections
        3. Maintains pool size within limits
        4. Monitors connection health and replaces unhealthy connections
        """
        try:
            health_check_interval = 60  # Check every minute
            leak_check_interval = 300  # Check for leaks every 5 minutes
            metrics_log_interval = 600  # Log metrics every 10 minutes
            
            last_leak_check = time.time()
            last_metrics_log = time.time()
            
            while True:
                try:
                    # Wait for a while before checking
                    await asyncio.sleep(health_check_interval)
                    
                    current_time = time.time()
                    
                    # Perform regular health check
                    await self._perform_connection_health_check()
                    
                    # Check for leaked connections at longer intervals
                    if current_time - last_leak_check >= leak_check_interval:
                        await self._check_for_connection_leaks()
                        last_leak_check = current_time
                    
                    # Log metrics at longer intervals
                    if current_time - last_metrics_log >= metrics_log_interval:
                        self._log_connection_pool_metrics()
                        last_metrics_log = current_time
                    
                except asyncio.CancelledError:
                    # Allow cancellation to propagate
                    raise
                    
                except Exception as e:
                    logger.error(f"Error in connection pool monitor: {e}")
                    logger.error(traceback.format_exc())
                
        except asyncio.CancelledError:
            logger.info("Connection pool monitor task cancelled")
        
        except Exception as e:
            logger.error(f"Fatal error in connection pool monitor: {e}")
            logger.error(traceback.format_exc())
    
    async def _perform_connection_health_check(self):
        """Check the health of all connections and replace unhealthy ones."""
        connections_to_close = []
        connections_to_test = []
        
        # First gather connections to check and ones to close by age
        async with self._lock:
            current_time = time.time()
            
            # Identify old connections to close or test
            for conn in list(self._connection_pool):
                # Skip busy connections
                if conn in self._busy_connections:
                    continue
                    
                # Get connection timestamp and stats
                timestamp = self._connection_timestamps.get(conn, 0)
                conn_id = getattr(conn, '_id', None)
                conn_stats = self._connection_usage_stats.get(conn_id)
                
                # Check TTL - close connections older than TTL
                if conn_id and conn_stats and current_time - timestamp > self._connection_ttl:
                    # Don't close if this is the only idle connection
                    idle_connections = [c for c in self._connection_pool if c not in self._busy_connections]
                    if len(idle_connections) > 1:
                        connections_to_close.append(conn)
                        continue
                
                # Add old or questionable connections to check list
                if (conn_stats and conn_stats.health_status != ConnectionHealthStatus.HEALTHY) or \
                   (current_time - timestamp > self._connection_ttl / 2):
                    connections_to_test.append(conn)
        
        # Test connections that need health check
        for conn in connections_to_test:
            try:
                # Run a simple query to test the connection
                start_time = time.time()
                await conn.execute("SELECT 1")
                response_time = time.time() - start_time
                
                # Get connection ID and update stats
                conn_id = getattr(conn, '_id', None)
                if conn_id and conn_id in self._connection_usage_stats:
                    stats = self._connection_usage_stats[conn_id]
                    stats.record_query(response_time)
                    
                    # If connection is still not healthy, add to close list
                    if stats.health_status != ConnectionHealthStatus.HEALTHY:
                        connections_to_close.append(conn)
                        logger.warning(f"Unhealthy connection detected (id={conn_id}, "
                                      f"status={stats.health_status})")
                    
            except Exception as e:
                # Connection is unhealthy, close it
                connections_to_close.append(conn)
                
                # Update error count in stats
                conn_id = getattr(conn, '_id', None)
                if conn_id and conn_id in self._connection_usage_stats:
                    self._connection_usage_stats[conn_id].record_error()
                
                logger.warning(f"Failed connection health check: {e}")
        
        # Close unhealthy connections
        for conn in connections_to_close:
            await self._close_connection(conn)
        
        # Create new connections if needed to maintain minimum pool size
        await self._maintain_minimum_pool_size()
    
    async def _check_for_connection_leaks(self):
        """Check for potential connection leaks (connections held too long)."""
        async with self._lock:
            current_time = time.time()
            
            # Check for connections held too long (potential leaks)
            for conn in list(self._busy_connections):
                conn_id = getattr(conn, '_id', None)
                if conn_id in self._connection_usage_stats:
                    stats = self._connection_usage_stats[conn_id]
                    busy_time = current_time - stats.last_used_at
                    
                    # If a connection has been busy for more than 5 minutes, log a warning
                    if busy_time > 300:  # 5 minutes
                        logger.warning(f"Potential connection leak detected: connection {conn_id} "
                                      f"has been busy for {busy_time:.1f} seconds")
            
            # Log current pool state
            self._log_connection_pool_state()
    
    def _log_connection_pool_state(self):
        """Log the current state of the connection pool."""
        total_connections = len(self._connection_pool)
        busy_connections = len(self._busy_connections)
        free_connections = total_connections - busy_connections
        
        logger.info(f"Connection pool state: {total_connections} total, "
                   f"{busy_connections} busy, {free_connections} free")
        
        # Log connection ages
        if self._connection_pool:
            current_time = time.time()
            ages = []
            for conn in self._connection_pool:
                created_at = getattr(conn, '_created_at', current_time)
                ages.append(current_time - created_at)
            
            avg_age = sum(ages) / len(ages) if ages else 0
            max_age = max(ages) if ages else 0
            
            logger.info(f"Connection ages: avg={avg_age:.1f}s, max={max_age:.1f}s")
    
    def _log_connection_pool_metrics(self):
        """Log detailed metrics about the connection pool usage."""
        # Calculate metrics
        total_queries = self._total_queries
        slow_query_pct = (self._slow_queries / total_queries * 100) if total_queries > 0 else 0
        avg_query_time = (self._total_query_time / total_queries) if total_queries > 0 else 0
        
        # Log main metrics
        logger.info(f"Connection pool metrics:"
                   f" queries={total_queries},"
                   f" slow_queries={self._slow_queries} ({slow_query_pct:.1f}%),"
                   f" avg_query_time={avg_query_time*1000:.1f}ms,"
                   f" peak_connections={self._peak_usage}/{self._max_connections}")
        
        # Log connection acquisition metrics
        if len(self._connection_acquisition_times) > 0:
            avg_wait = sum(self._connection_acquisition_times) / len(self._connection_acquisition_times)
            max_wait = max(self._connection_acquisition_times) if self._connection_acquisition_times else 0
            logger.info(f"Connection wait metrics:"
                       f" avg_wait={avg_wait*1000:.1f}ms,"
                       f" max_wait={max_wait*1000:.1f}ms,"
                       f" peak_waiters={self._peak_waiting}")
        
        # Reset some counters
        self._connection_acquisition_times = []
    
    async def _dynamic_connection_scaling(self):
        """Dynamically scale the connection pool based on usage patterns."""
        try:
            while True:
                try:
                    # Wait before checking
                    await asyncio.sleep(30)  # Check every 30 seconds
                    
                    if not self._dynamic_scaling_enabled:
                        continue
                    
                    async with self._lock:
                        # Calculate pool usage ratio
                        total_connections = len(self._connection_pool)
                        busy_connections = len(self._busy_connections)
                        usage_ratio = busy_connections / total_connections if total_connections > 0 else 0
                        
                        # Update load metrics
                        self._load_metrics['usage_ratio'] = usage_ratio
                        self._load_metrics['total_connections'] = total_connections
                        self._load_metrics['busy_connections'] = busy_connections
                        
                        # Scale up: If pool is more than 70% busy and below max, add connections
                        if usage_ratio > 0.7 and total_connections < self._max_connections:
                            # Calculate how many connections to add (25% of current size, at least 1)
                            to_add = max(1, total_connections // 4)
                            # Don't exceed max connections
                            to_add = min(to_add, self._max_connections - total_connections)
                            
                            logger.info(f"Scaling up connection pool: adding {to_add} connections "
                                       f"(usage ratio: {usage_ratio:.2f})")
                            
                            # Add new connections
                            for _ in range(to_add):
                                try:
                                    await self._create_new_connection()
                                except Exception as e:
                                    logger.error(f"Error creating connection during scale-up: {e}")
                        
                        # Scale down: If pool is less than 30% busy for a while and above min, remove connections
                        elif usage_ratio < 0.3 and total_connections > self._min_connections:
                            # Only scale down if the ratio has been low for a while
                            # (this is tracked by consecutive low ratio readings)
                            if self._load_metrics.get('low_usage_count', 0) >= 3:  # 3 consecutive readings
                                # Calculate how many connections to remove (max 25% of current size, at least 1)
                                to_remove = max(1, total_connections // 4)
                                # Don't go below min connections
                                to_remove = min(to_remove, total_connections - self._min_connections)
                                
                                logger.info(f"Scaling down connection pool: removing {to_remove} connections "
                                           f"(usage ratio: {usage_ratio:.2f})")
                                
                                # Remove idle connections
                                removed = 0
                                for conn in list(self._connection_pool):
                                    if conn not in self._busy_connections:
                                        await self._close_connection(conn)
                                        removed += 1
                                        if removed >= to_remove:
                                            break
                                
                                # Reset low usage counter
                                self._load_metrics['low_usage_count'] = 0
                            else:
                                # Increment low usage counter
                                self._load_metrics['low_usage_count'] = self._load_metrics.get('low_usage_count', 0) + 1
                        else:
                            # Reset low usage counter if usage is not low
                            if usage_ratio >= 0.3:
                                self._load_metrics['low_usage_count'] = 0
                        
                        # Check if we need to activate backpressure
                        if self._backpressure_enabled:
                            pool_usage_ratio = busy_connections / self._max_connections
                            
                            # Activate backpressure if usage is above high threshold
                            if not self._backpressure_active and pool_usage_ratio >= self._backpressure_high_threshold:
                                self._backpressure_active = True
                                logger.warning(f"Activating database query backpressure (usage ratio: {pool_usage_ratio:.2f})")
                            
                            # Deactivate backpressure if usage falls below low threshold
                            elif self._backpressure_active and pool_usage_ratio <= self._backpressure_low_threshold:
                                self._backpressure_active = False
                                logger.info(f"Deactivating database query backpressure (usage ratio: {pool_usage_ratio:.2f})")
                    
                except asyncio.CancelledError:
                    raise
                    
                except Exception as e:
                    logger.error(f"Error in connection pool scaling: {e}")
                    logger.error(traceback.format_exc())
                
        except asyncio.CancelledError:
            logger.info("Connection pool scaling task cancelled")
        
        except Exception as e:
            logger.error(f"Fatal error in connection pool scaling: {e}")
            logger.error(traceback.format_exc())
    
    async def _maintain_minimum_pool_size(self):
        """Ensure the pool has at least the minimum number of connections."""
        async with self._lock:
            current_pool_size = len(self._connection_pool)
            
            # If we're below the minimum, create new connections
            if current_pool_size < self._min_connections:
                to_add = self._min_connections - current_pool_size
                logger.info(f"Adding {to_add} connections to maintain minimum pool size")
                
                for _ in range(to_add):
                    try:
                        await self._create_new_connection()
                    except Exception as e:
                        logger.error(f"Error creating connection during maintenance: {e}")
    
    async def _close_connection(self, conn):
        """Close a database connection and remove it from the pool.
        
        Args:
            conn: The connection to close
        """
        async with self._lock:
            try:
                # Remove from pool and tracking
                if conn in self._connection_pool:
                    self._connection_pool.remove(conn)
                if conn in self._connection_timestamps:
                    del self._connection_timestamps[conn]
                if conn in self._busy_connections:
                    self._busy_connections.remove(conn)
                
                # Remove from stats tracking
                conn_id = getattr(conn, '_id', None)
                if conn_id and conn_id in self._connection_usage_stats:
                    del self._connection_usage_stats[conn_id]
                
                # Close the connection
                await conn.close()
                logger.debug(f"Closed database connection (id={conn_id}, pool size={len(self._connection_pool)})")
                
            except Exception as e:
                logger.error(f"Error closing database connection: {e}")
    
    async def cleanup(self):
        """Clean up resources when shutting down.
        
        This method should be called when the application is shutting down
        to properly release all database connections and resources.
        """
        logger.info("Cleaning up database resources...")
        
        # Cancel the monitoring task if it's running
        if self._connection_monitor_task and not self._connection_monitor_task.done():
            try:
                self._connection_monitor_task.cancel()
                await asyncio.wait([self._connection_monitor_task], timeout=5)
            except Exception as e:
                logger.warning(f"Error cancelling monitor task: {e}")
        
        # Cancel the scaling task if it's running
        if self._connection_scaling_task and not self._connection_scaling_task.done():
            try:
                self._connection_scaling_task.cancel()
                await asyncio.wait([self._connection_scaling_task], timeout=5)
            except Exception as e:
                logger.warning(f"Error cancelling scaling task: {e}")
        
        # Close all connections in the pool
        async with self._lock:
            for conn in self._connection_pool:
                try:
                    await conn.close()
                except:
                    pass
            
            self._connection_pool = []
            self._connection_timestamps = {}
            self._busy_connections = set()
            self._connection_usage_stats = {}
        
        self.initialized = False
        logger.info("Database resources cleaned up")
    
    async def _get_connection(self):
        """Get a connection from the pool or create a new one if needed.
        
        Returns:
            aiosqlite.Connection: A database connection

        Raises:
            TimeoutError: If no connection could be acquired within the timeout
            DatabaseError: If there's an error creating a new connection
        """
        # Make sure we're initialized
        if not self.initialized:
            await self.initialize()
        
        # Track metrics
        start_time = time.time()
        self._usage_count += 1
        
        # Check for backpressure if enabled
        if self._backpressure_active and self._backpressure_enabled:
            # Apply rate limiting if backpressure is active
            try:
                # Add jitter to avoid thundering herd when backpressure is released
                jitter = random.uniform(0, 0.1)
                await asyncio.sleep(jitter)
                
                # Try to acquire rate limiter permit with short timeout
                async with asyncio.timeout(0.5):
                    await self._query_rate_limiter.acquire()
            except asyncio.TimeoutError:
                # If we can't get a permit quickly, it means we're under high load
                logger.warning("Database query rate limited due to backpressure")
                raise DatabaseError("Database query rejected due to high load (backpressure active)")
            finally:
                # Always release the permit
                self._query_rate_limiter.release()
        
        # Try to get a connection from the pool
        try:
            async with self._lock:
                # Track waiting statistics
                self._waiting_count += 1
                self._peak_waiting = max(self._peak_waiting, self._waiting_count)
                
                # Check if we have any free connections in the pool
                free_connections = [conn for conn in self._connection_pool if conn not in self._busy_connections]
                
                if free_connections:
                    # Use an existing connection (prioritize newer connections)
                    # Sort by creation timestamp (newest first)
                    free_connections.sort(
                        key=lambda c: self._connection_timestamps.get(c, 0),
                        reverse=True
                    )
                    
                    connection = free_connections[0]
                    self._busy_connections.add(connection)
                    
                    # Update metrics
                    self._waiting_count -= 1
                    self._peak_usage = max(self._peak_usage, len(self._busy_connections))
                    
                    # Record acquisition time
                    acquisition_time = time.time() - start_time
                    self._connection_acquisition_times.append(acquisition_time)
                    
                    return connection
                
                # If we're below max connections, create a new one
                if len(self._connection_pool) < self._max_connections:
                    try:
                        connection = await self._create_new_connection()
                        self._busy_connections.add(connection)
                        
                        # Update metrics
                        self._waiting_count -= 1
                        self._peak_usage = max(self._peak_usage, len(self._busy_connections))
                        
                        # Record acquisition time
                        acquisition_time = time.time() - start_time
                        self._connection_acquisition_times.append(acquisition_time)
                        
                        return connection
                        
                    except Exception as e:
                        logger.error(f"Error creating new database connection: {e}")
                        self._waiting_count -= 1  # Update waiting count on error
                        raise DatabaseError(f"Failed to create new database connection: {e}")
                
                # Update metrics - we're still waiting
                self._waiting_count -= 1
        
            # If we get here, no free connections and at max capacity
            # Wait for a connection to become available
            try:
                # Use a future to wait for a connection
                future = asyncio.Future()
                wait_start = time.time()
                
                async def wait_for_connection():
                    while True:
                        async with self._lock:
                            free_connections = [conn for conn in self._connection_pool if conn not in self._busy_connections]
                            if free_connections:
                                # Prioritize newer connections
                                free_connections.sort(
                                    key=lambda c: self._connection_timestamps.get(c, 0),
                                    reverse=True
                                )
                                connection = free_connections[0]
                                self._busy_connections.add(connection)
                                
                                # Update metrics
                                self._peak_usage = max(self._peak_usage, len(self._busy_connections))
                                
                                future.set_result(connection)
                                return
                        
                        # Wait a bit before checking again
                        await asyncio.sleep(0.1)
                
                # Start waiting task
                wait_task = asyncio.create_task(wait_for_connection())
                
                # Wait with timeout
                connection = await asyncio.wait_for(future, timeout=self._connection_acquisition_timeout)
                
                # Cancel task if still running
                if not wait_task.done():
                    wait_task.cancel()
                    
                # Record acquisition time
                acquisition_time = time.time() - start_time
                self._connection_acquisition_times.append(acquisition_time)
                self._total_wait_time += (time.time() - wait_start)
                    
                return connection
                
            except asyncio.TimeoutError:
                logger.error(f"Timeout waiting for database connection (current pool size: {len(self._connection_pool)})")
                raise TimeoutError("Timeout waiting for database connection")
                
            except Exception as e:
                logger.error(f"Error waiting for database connection: {e}")
                raise DatabaseError(f"Failed to get database connection: {e}")
        except Exception as e:
            # If we get here, something went wrong with the whole connection acquisition process
            logger.error(f"Error acquiring database connection: {e}")
            raise DatabaseError(f"Failed to acquire database connection: {e}")
    
    async def _return_connection(self, conn):
        """Return a connection to the pool.
        
        Args:
            conn: The connection to return
        """
        if not conn:
            return
        
        async with self._lock:
            if conn in self._busy_connections:
                self._busy_connections.remove(conn)
            
            # Update timestamp
            self._connection_timestamps[conn] = time.time()
    
    @asynccontextmanager
    async def get_db_connection(self, query_timeout=15):
        """Get a database connection from the pool.
        
        This is an async context manager that automatically returns
        the connection to the pool when the context exits.
        
        Args:
            query_timeout: Timeout in seconds for query execution
            
        Yields:
            aiosqlite.Connection: A database connection
            
        Raises:
            TimeoutError: If no connection could be acquired within the timeout
            DatabaseError: If there's an error with the database
        """
        conn = None
        try:
            conn = await self._get_connection()
            
            # Set timeout for this connection
            await conn.execute(f"PRAGMA busy_timeout = {query_timeout * 1000}")
            
            yield conn
            
        finally:
            if conn:
                await self._return_connection(conn)
                
    async def execute_optimized_query(self, query_or_conn, query=None, params=None, 
                                     fetch_all=True, explain=False, timeout=15.0, 
                                     cache_key=None, cache_ttl=None):
        """Execute an optimized database query with connection handling and performance tracking.
        
        This method provides a flexible way to execute queries:
        1. You can pass a connection and query separately
        2. You can pass just the query and it will get a connection for you
        
        Args:
            query_or_conn: Either a SQL query string or a database connection
            query: SQL query string (if query_or_conn is a connection)
            params: Query parameters
            fetch_all: Whether to fetch all results or just one
            explain: Whether to run EXPLAIN QUERY PLAN
            timeout: Query timeout in seconds
            cache_key: Optional key for caching results
            cache_ttl: Optional TTL for cache in seconds
            
        Returns:
            Query results
            
        Raises:
            TimeoutError: If the query times out
            DatabaseError: For other database errors
        """
        # Check if we should use cache
        if cache_key:
            cached_result = await self.cache.get(cache_key)
            if cached_result is not None:
                return cached_result
        
        # Determine if we got a connection or a query
        if isinstance(query_or_conn, str):
            # We got a query string
            sql_query = query_or_conn
            connection_provided = False
        else:
            # We got a connection
            connection = query_or_conn
            sql_query = query
            connection_provided = True
            
        # Validate query
        if not sql_query:
            raise ValueError("No SQL query provided")
            
        # Normalize params
        if params is None:
            params = ()
            
        # Get connection ID for tracking
        conn_id = None
        if connection_provided:
            conn_id = getattr(connection, '_id', None)
        
        # Start timing
        start_time = time.time()
        
        try:
            # If explain is requested, run the explain query
            if explain:
                explain_sql = f"EXPLAIN QUERY PLAN {sql_query}"
                
                if connection_provided:
                    # Use the provided connection
                    cursor = await connection.execute(explain_sql, params)
                    plan = await cursor.fetchall()
                    logger.debug(f"Query plan: {plan}")
                else:
                    # Get a connection from the pool
                    async with self.get_db_connection(query_timeout=timeout) as conn:
                        conn_id = getattr(conn, '_id', None)
                        cursor = await conn.execute(explain_sql, params)
                        plan = await cursor.fetchall()
                        logger.debug(f"Query plan: {plan}")
                        
                        # Check for better plans if this one seems inefficient
                        self._check_query_plan_efficiency(sql_query, plan)
            
            # Execute the actual query
            if connection_provided:
                # Use the provided connection
                cursor = await connection.execute(sql_query, params)
                if fetch_all:
                    result = await cursor.fetchall()
                else:
                    result = await cursor.fetchone()
            else:
                # Get a connection from the pool
                async with self.get_db_connection(query_timeout=timeout) as conn:
                    conn_id = getattr(conn, '_id', None)
                    cursor = await conn.execute(sql_query, params)
                    if fetch_all:
                        result = await cursor.fetchall()
                    else:
                        result = await cursor.fetchone()
            
            # Calculate query time and update metrics
            query_time = time.time() - start_time
            self._total_query_time += query_time
            self._total_queries += 1
            
            # Track slow queries
            if query_time > self._slow_query_threshold:
                self._slow_queries += 1
                logger.warning(f"Slow query detected ({query_time:.2f}s): {sql_query[:100]}...")
                
                # Add to query history for analysis
                self._add_to_query_history(sql_query, params, query_time, conn_id)
            
            # Update connection stats
            if conn_id and conn_id in self._connection_usage_stats:
                self._connection_usage_stats[conn_id].record_query(query_time)
            
            # Cache result if requested
            if cache_key and result:
                await self.cache.set(cache_key, result, ttl=cache_ttl)
                    
            return result
                
        except asyncio.TimeoutError:
            logger.error(f"Query timeout: {sql_query[:100]}...")
            
            # Update connection stats
            if conn_id and conn_id in self._connection_usage_stats:
                self._connection_usage_stats[conn_id].record_error()
                
            raise TimeoutError(f"Query execution timed out after {timeout} seconds")
            
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            logger.error(f"Query: {sql_query[:100]}...")
            
            # Update connection stats
            if conn_id and conn_id in self._connection_usage_stats:
                self._connection_usage_stats[conn_id].record_error()
                
            raise DatabaseError(f"Error executing query: {e}")
    
    def _check_query_plan_efficiency(self, query, plan):
        """Check if the query plan is efficient and log warnings if not."""
        # Look for full table scans in large tables
        full_scans = []
        for step in plan:
            plan_detail = step[-1] if len(step) > 0 else ""
            if "SCAN TABLE" in plan_detail and "USING INDEX" not in plan_detail:
                table_name = plan_detail.split("SCAN TABLE")[1].strip().split(" ")[0]
                full_scans.append(table_name)
        
        if full_scans:
            logger.warning(f"Inefficient query plan detected - full table scan on: {', '.join(full_scans)}")
            logger.warning(f"Consider adding indexes for query: {query[:100]}...")
            
            # For more detailed warnings, analyze joins and where clauses
            if "JOIN" in query.upper() and len(full_scans) > 1:
                logger.warning("Multi-table join without proper indexes detected")
                
            # Log specific suggestions
            if "chat_message_join" in full_scans:
                logger.warning("Missing index on chat_message_join table - consider adding indexes on chat_id and message_id")
            if "message" in full_scans and "WHERE" in query.upper():
                logger.warning("Missing index on message table filters - consider adding indexes on date, handle_id, or is_from_me")
    
    def _add_to_query_history(self, query, params, execution_time, conn_id):
        """Add a query to the history for analysis."""
        # Create history entry
        entry = {
            'query': query[:500],  # Truncate long queries
            'params': str(params)[:100],  # Truncate params
            'execution_time': execution_time,
            'timestamp': time.time(),
            'conn_id': conn_id
        }
        
        # Add to history
        self._query_history.append(entry)
        
        # Trim history if too long
        if len(self._query_history) > self._max_query_history:
            self._query_history = self._query_history[-self._max_query_history:]
            
    async def safe_db_operation(self, operation_func, *args, **kwargs):
        """Execute a database operation with error handling and retries.
        
        Args:
            operation_func: Async function to execute
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Result of the operation
            
        Raises:
            DatabaseError: If the operation fails after retries
        """
        max_retries = kwargs.pop('max_retries', 3)
        retry_delay = kwargs.pop('retry_delay', 1.0)
        
        for attempt in range(max_retries):
            try:
                # Execute the operation
                result = await operation_func(*args, **kwargs)
                return result
                
            except asyncio.TimeoutError:
                # Don't retry timeouts
                logger.error(f"Timeout in database operation (attempt {attempt+1}/{max_retries})")
                if attempt == max_retries - 1:
                    raise DatabaseError("Database operation timed out")
                await asyncio.sleep(retry_delay)
                
            except sqlite3.OperationalError as e:
                # SQLite operational errors (locked, busy, etc.)
                error_str = str(e).lower()
                
                # Check for database locked errors
                if 'database is locked' in error_str or 'busy' in error_str:
                    logger.warning(f"Database locked (attempt {attempt+1}/{max_retries}): {e}")
                    if attempt == max_retries - 1:
                        raise DatabaseError(f"Database is locked: {e}")
                    await asyncio.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    # For other operational errors, don't retry
                    logger.error(f"SQLite operational error: {e}")
                    raise DatabaseError(f"SQLite operational error: {e}")
                    
            except Exception as e:
                # Handle other exceptions
                logger.error(f"Error in database operation: {e}")
                logger.error(traceback.format_exc())
                raise DatabaseError(f"Error in database operation: {e}")
    
    # For methods that modify data, we need to invalidate the cache
    async def invalidate_cache(self, key_prefix=None):
        """Invalidate the cache for a specific key prefix or all keys.
        
        Args:
            key_prefix: The key prefix to invalidate, or None for all keys
        """
        from src.utils.redis_cache import invalidate_by_pattern
        
        if key_prefix:
            await invalidate_by_pattern(key_prefix)
        else:
            # Invalidate all cache entries
            await invalidate_by_pattern("*")
            
        logger.info("Cache invalidated")
    
    async def get_pool_stats(self):
        """Get statistics about the connection pool.
        
        Returns:
            Dictionary with pool statistics
        """
        async with self._lock:
            stats = {
                'pool_size': len(self._connection_pool),
                'busy_connections': len(self._busy_connections),
                'free_connections': len(self._connection_pool) - len(self._busy_connections),
                'max_connections': self._max_connections,
                'min_connections': self._min_connections,
                'usage_ratio': len(self._busy_connections) / len(self._connection_pool) if self._connection_pool else 0,
                'peak_usage': self._peak_usage,
                'total_queries': self._total_queries,
                'slow_queries': self._slow_queries,
                'slow_query_pct': (self._slow_queries / self._total_queries * 100) if self._total_queries > 0 else 0,
                'avg_query_time': (self._total_query_time / self._total_queries) if self._total_queries > 0 else 0,
                'total_wait_time': self._total_wait_time,
                'backpressure_active': self._backpressure_active
            }
            
            # Add connection age statistics
            if self._connection_pool:
                current_time = time.time()
                ages = []
                for conn in self._connection_pool:
                    created_at = getattr(conn, '_created_at', current_time)
                    ages.append(current_time - created_at)
                
                stats['avg_connection_age'] = sum(ages) / len(ages) if ages else 0
                stats['max_connection_age'] = max(ages) if ages else 0
            
            return stats

    async def get_db_connection(self):
        """Get a database connection from the pool.
        
        Returns:
            An aiosqlite connection from the pool
        """
        if not self.initialized:
            await self.initialize()
            
        return await self._get_connection()

    async def close(self):
        """Close all database connections and clean up resources.
        
        This method should be called when shutting down the database to ensure
        all connections are properly closed and resources are freed.
        """
        async with self._lock:
            # Close all connections in the pool
            for conn in self._connection_pool:
                try:
                    await conn.close()
                except Exception as e:
                    logger.warning(f"Error closing connection: {e}")
            
            # Clear the connection pool
            self._connection_pool = []
            self._busy_connections = set()
            self._connection_timestamps = {}
            self._connection_usage_stats = {}
            
            # Cancel any connection monitor task
            if self._connection_monitor_task and not self._connection_monitor_task.done():
                self._connection_monitor_task.cancel()
                try:
                    await self._connection_monitor_task
                except asyncio.CancelledError:
                    pass
            
            # Cancel any connection scaling task
            if self._connection_scaling_task and not self._connection_scaling_task.done():
                self._connection_scaling_task.cancel()
                try:
                    await self._connection_scaling_task
                except asyncio.CancelledError:
                    pass
            
            # Clean up prepared statements
            self._prepared_statements = {}
            self._statement_usage_count = {}
            
            logger.info("Database connections closed and resources cleaned up")
            self.initialized = False