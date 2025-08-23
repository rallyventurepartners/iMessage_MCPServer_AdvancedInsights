"""
Concurrency and thread safety utilities for database operations.

This module provides utilities for managing concurrent database access,
connection pooling, and ensuring thread safety in asynchronous operations.
"""

import logging
import asyncio
import time
import traceback
from typing import Any, Dict, List, Optional, Set, Callable, Awaitable, TypeVar, Generic
from functools import wraps
from contextlib import asynccontextmanager

from .config import get_config
from ..exceptions import DatabaseError

logger = logging.getLogger(__name__)

# Type variables for generic functions
T = TypeVar('T')
R = TypeVar('R')

# Global connection pool
_connection_pool: Dict[str, List[Any]] = {}
_active_connections: Dict[str, Set[Any]] = {}

# Semaphores for controlling concurrent access
_db_semaphore: Optional[asyncio.Semaphore] = None
_operation_semaphores: Dict[str, asyncio.Semaphore] = {}


class AsyncLock:
    """
    A reentrant lock for asynchronous operations.
    
    This lock can be acquired multiple times by the same task without
    causing a deadlock. Each acquire() must be matched with a release().
    """
    
    def __init__(self):
        """Initialize the lock with internal structures."""
        self._lock = asyncio.Lock()
        self._owner = None
        self._count = 0
    
    async def acquire(self):
        """
        Acquire the lock. If the lock is already held by the current task,
        increment the recursion counter.
        
        Returns:
            True when the lock is acquired
        """
        task = asyncio.current_task()
        
        if self._owner == task:
            # Already owned by this task, just increment the counter
            self._count += 1
            return True
        
        # Wait to acquire the lock
        await self._lock.acquire()
        
        # Set ownership and counter
        self._owner = task
        self._count = 1
        return True
    
    def release(self):
        """
        Release the lock. If this is the last release for this task,
        release the actual lock.
        
        Raises:
            RuntimeError: If the lock is not owned by the current task
        """
        task = asyncio.current_task()
        
        if self._owner != task:
            raise RuntimeError("Cannot release a lock not owned by the current task")
        
        self._count -= 1
        if self._count == 0:
            # Last release, clear ownership and release the lock
            self._owner = None
            self._lock.release()
    
    @asynccontextmanager
    async def __call__(self):
        """Context manager for using the lock with 'async with'."""
        await self.acquire()
        try:
            yield
        finally:
            self.release()


class DatabaseConnectionPool:
    """
    A pool of database connections for reuse across operations.
    
    This reduces the overhead of creating new connections for every operation
    and helps manage concurrent connections efficiently.
    """
    
    def __init__(self, pool_name: str, min_size: int = 5, max_size: int = 20, 
                 connection_timeout: float = 30.0, idle_timeout: float = 300.0):
        """
        Initialize the connection pool.
        
        Args:
            pool_name: Unique name for this connection pool
            min_size: Minimum number of connections to maintain
            max_size: Maximum number of connections allowed
            connection_timeout: Timeout in seconds for connection acquisition
            idle_timeout: Timeout in seconds for idle connections
        """
        self.pool_name = pool_name
        self.min_size = min_size
        self.max_size = max_size
        self.connection_timeout = connection_timeout
        self.idle_timeout = idle_timeout
        
        # Initialize pool structures if not already created
        if pool_name not in _connection_pool:
            _connection_pool[pool_name] = []
        
        if pool_name not in _active_connections:
            _active_connections[pool_name] = set()
        
        # Locks for pool management
        self._pool_lock = AsyncLock()
        
        # Connection factory and cleanup handlers
        self._connection_factory = None
        self._connection_cleanup = None
        
        # Start maintenance task
        asyncio.create_task(self._maintenance_task())
    
    def set_connection_handlers(self, factory: Callable[[], Awaitable[Any]], 
                                cleanup: Callable[[Any], Awaitable[None]]):
        """
        Set handlers for creating and cleaning up connections.
        
        Args:
            factory: Async function that creates and returns a new connection
            cleanup: Async function that properly closes a connection
        """
        self._connection_factory = factory
        self._connection_cleanup = cleanup
    
    async def initialize(self):
        """Initialize the connection pool with minimum connections."""
        if not self._connection_factory:
            raise ValueError("Connection factory must be set before initialization")
        
        async with self._pool_lock():
            # Create minimum number of connections
            while len(_connection_pool[self.pool_name]) < self.min_size:
                try:
                    conn = await self._connection_factory()
                    conn._last_used = time.time()  # Add timestamp
                    _connection_pool[self.pool_name].append(conn)
                    logger.debug(f"Added connection to {self.pool_name} pool (init)")
                except Exception as e:
                    logger.error(f"Error initializing connection pool: {e}")
                    break
    
    @asynccontextmanager
    async def connection(self):
        """
        Acquire a connection from the pool.
        
        Yields:
            A database connection
        """
        if not self._connection_factory or not self._connection_cleanup:
            raise ValueError("Connection handlers must be set before using the pool")
        
        conn = None
        is_new = False
        
        # Acquire connection with timeout
        start_time = time.time()
        while conn is None:
            if time.time() - start_time > self.connection_timeout:
                raise DatabaseError(f"Timeout waiting for database connection from {self.pool_name} pool")
            
            async with self._pool_lock():
                # Check if there are available connections in the pool
                if _connection_pool[self.pool_name]:
                    conn = _connection_pool[self.pool_name].pop()
                    logger.debug(f"Got connection from {self.pool_name} pool")
                # If under max size, create a new connection
                elif len(_active_connections[self.pool_name]) < self.max_size:
                    try:
                        conn = await self._connection_factory()
                        is_new = True
                        logger.debug(f"Created new connection for {self.pool_name} pool")
                    except Exception as e:
                        logger.error(f"Error creating database connection: {e}")
                        await asyncio.sleep(0.5)  # Short delay before retry
                else:
                    # Wait a bit and retry
                    logger.warning(f"Connection pool {self.pool_name} exhausted, waiting...")
                    await asyncio.sleep(0.5)
            
            # If we got a connection, mark it as active
            if conn:
                _active_connections[self.pool_name].add(conn)
                conn._last_used = time.time()
        
        try:
            # Yield the connection for use
            yield conn
        finally:
            # Return connection to pool or close if there was an error
            async with self._pool_lock():
                if conn in _active_connections[self.pool_name]:
                    _active_connections[self.pool_name].remove(conn)
                    
                    # Update last used time
                    conn._last_used = time.time()
                    
                    # Return to pool if not at max size
                    if len(_connection_pool[self.pool_name]) < self.max_size:
                        _connection_pool[self.pool_name].append(conn)
                        logger.debug(f"Returned connection to {self.pool_name} pool")
                    else:
                        # Close excess connection
                        try:
                            await self._connection_cleanup(conn)
                            logger.debug(f"Closed excess connection from {self.pool_name} pool")
                        except Exception as e:
                            logger.error(f"Error closing database connection: {e}")
    
    async def close(self):
        """Close all connections in the pool."""
        if not self._connection_cleanup:
            logger.warning("No connection cleanup handler set for pool")
            return
        
        async with self._pool_lock():
            # Close pooled connections
            for conn in _connection_pool[self.pool_name]:
                try:
                    await self._connection_cleanup(conn)
                except Exception as e:
                    logger.error(f"Error closing pooled connection: {e}")
            
            # Log warning for any active connections
            active_count = len(_active_connections[self.pool_name])
            if active_count > 0:
                logger.warning(f"{active_count} connections from {self.pool_name} pool still active during shutdown")
            
            # Clear pool
            _connection_pool[self.pool_name] = []
    
    async def _maintenance_task(self):
        """
        Background task for pool maintenance.
        
        - Removes idle connections beyond min_size
        - Ensures minimum pool size is maintained
        """
        while True:
            try:
                await asyncio.sleep(60)  # Run maintenance every minute
                
                if not self._connection_factory or not self._connection_cleanup:
                    continue
                
                async with self._pool_lock():
                    current_time = time.time()
                    
                    # Close idle connections beyond min_size
                    if len(_connection_pool[self.pool_name]) > self.min_size:
                        to_remove = []
                        
                        for i, conn in enumerate(_connection_pool[self.pool_name]):
                            if current_time - conn._last_used > self.idle_timeout:
                                to_remove.append((i, conn))
                                
                                # Stop if we would go below min_size
                                if len(_connection_pool[self.pool_name]) - len(to_remove) < self.min_size:
                                    break
                        
                        # Remove from end to avoid index shifting
                        for i, conn in reversed(to_remove):
                            try:
                                await self._connection_cleanup(conn)
                                del _connection_pool[self.pool_name][i]
                                logger.debug(f"Closed idle connection from {self.pool_name} pool")
                            except Exception as e:
                                logger.error(f"Error closing idle connection: {e}")
                    
                    # Ensure minimum pool size
                    while len(_connection_pool[self.pool_name]) < self.min_size:
                        try:
                            conn = await self._connection_factory()
                            conn._last_used = time.time()
                            _connection_pool[self.pool_name].append(conn)
                            logger.debug(f"Added connection to {self.pool_name} pool (maintenance)")
                        except Exception as e:
                            logger.error(f"Error creating connection during maintenance: {e}")
                            break
            
            except Exception as e:
                logger.error(f"Error in connection pool maintenance: {e}")
                logger.error(traceback.format_exc())


def init_concurrency():
    """Initialize concurrency structures based on configuration."""
    config = get_config()
    
    # Initialize database semaphore
    global _db_semaphore
    max_concurrent_operations = config.get("database", {}).get("concurrency", {}).get("max_concurrent_operations", 20)
    _db_semaphore = asyncio.Semaphore(max_concurrent_operations)
    
    # Initialize operation-specific semaphores
    operation_limits = config.get("database", {}).get("concurrency", {}).get("operation_limits", {})
    for operation, limit in operation_limits.items():
        _operation_semaphores[operation] = asyncio.Semaphore(limit)


def limit_concurrency(max_concurrent: Optional[int] = None, operation_type: Optional[str] = None):
    """
    Decorator to limit concurrent executions of a function.
    
    Args:
        max_concurrent: Maximum number of concurrent executions (overrides config)
        operation_type: Operation type for using specific semaphores
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Determine which semaphore to use
            semaphore = None
            
            if operation_type and operation_type in _operation_semaphores:
                semaphore = _operation_semaphores[operation_type]
            elif max_concurrent:
                # Create a new semaphore for this function if specified
                if not hasattr(wrapper, '_semaphore'):
                    wrapper._semaphore = asyncio.Semaphore(max_concurrent)
                semaphore = wrapper._semaphore
            else:
                # Use the global database semaphore
                semaphore = _db_semaphore
            
            # Use the semaphore to limit concurrency
            async with semaphore:
                return await func(*args, **kwargs)
        
        return wrapper
    
    return decorator


class BatchProcessor(Generic[T, R]):
    """
    Batch processor for database operations.
    
    This allows batching multiple small operations into fewer larger ones,
    which can be more efficient for certain types of database queries.
    """
    
    def __init__(self, processor: Callable[[List[T]], Awaitable[List[R]]], 
                 max_batch_size: int = 100, 
                 max_wait_time: float = 0.05):
        """
        Initialize the batch processor.
        
        Args:
            processor: Async function that processes a batch of items
            max_batch_size: Maximum number of items in a batch
            max_wait_time: Maximum time to wait for a batch to fill
        """
        self.processor = processor
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        
        self._batch: List[T] = []
        self._results: Dict[int, asyncio.Future[R]] = {}
        self._batch_lock = AsyncLock()
        self._next_id = 0
        self._batch_task: Optional[asyncio.Task] = None
    
    async def submit(self, item: T) -> R:
        """
        Submit an item for batch processing.
        
        Args:
            item: The item to process
            
        Returns:
            The result for this specific item
        """
        # Create a future for this item's result
        result_future: asyncio.Future[R] = asyncio.Future()
        
        async with self._batch_lock():
            # Assign an ID to this item
            item_id = self._next_id
            self._next_id += 1
            
            # Add to current batch
            self._batch.append(item)
            self._results[item_id] = result_future
            
            # Start batch processing if needed
            if len(self._batch) >= self.max_batch_size:
                # Process immediately if batch is full
                if self._batch_task is None or self._batch_task.done():
                    self._batch_task = asyncio.create_task(self._process_batch())
            elif self._batch_task is None or self._batch_task.done():
                # Start delay timer for batch processing
                self._batch_task = asyncio.create_task(self._delayed_process())
        
        # Wait for this item's result
        return await result_future
    
    async def _delayed_process(self):
        """Wait before processing to allow more items to accumulate."""
        await asyncio.sleep(self.max_wait_time)
        await self._process_batch()
    
    async def _process_batch(self):
        """Process the current batch of items."""
        current_batch = []
        result_futures = {}
        
        async with self._batch_lock():
            # Get the current batch
            current_batch = self._batch.copy()
            self._batch = []
            
            # Get the futures for this batch
            result_futures = self._results.copy()
            self._results = {}
        
        if not current_batch:
            return
        
        try:
            # Process the batch
            results = await self.processor(current_batch)
            
            # Set results for individual items
            for i, result in enumerate(results):
                item_id = list(result_futures.keys())[i]
                if item_id in result_futures:
                    result_futures[item_id].set_result(result)
        except Exception as e:
            # Set exception for all futures in this batch
            for future in result_futures.values():
                if not future.done():
                    future.set_exception(e)
