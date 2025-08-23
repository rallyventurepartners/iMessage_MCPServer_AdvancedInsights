"""
Query caching system for optimizing database operations.

This module provides utilities to cache the results of expensive database queries
to improve performance and reduce database load.
"""

import logging
import time
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from functools import wraps

from .config import get_config
from ..exceptions import DatabaseError

logger = logging.getLogger(__name__)

# In-memory cache storage
_query_cache: Dict[str, Dict[str, Any]] = {}
_cache_stats: Dict[str, Dict[str, int]] = {
    "hits": {},
    "misses": {},
    "expirations": {},
}

# Cache configuration
DEFAULT_CACHE_TTL = 300  # 5 minutes
DEFAULT_CACHE_SIZE = 1000  # Max cache entries


def get_cache_config():
    """Get cache configuration from global config."""
    config = get_config()
    cache_config = {
        "enabled": config.get("database", {}).get("query_cache", {}).get("enabled", True),
        "ttl": config.get("database", {}).get("query_cache", {}).get("ttl", DEFAULT_CACHE_TTL),
        "max_size": config.get("database", {}).get("query_cache", {}).get("max_size", DEFAULT_CACHE_SIZE),
        "stats_enabled": config.get("database", {}).get("query_cache", {}).get("stats_enabled", True),
    }
    return cache_config


def generate_cache_key(prefix: str, *args, **kwargs) -> str:
    """
    Generate a unique cache key based on function arguments.
    
    Args:
        prefix: Prefix for the cache key (usually function name)
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        A unique string key
    """
    # Convert args to string representations
    arg_parts = [str(arg) for arg in args]
    
    # Convert kwargs to sorted string representations
    kwarg_parts = [f"{k}={kwargs[k]}" for k in sorted(kwargs.keys())]
    
    # Combine all parts
    key_parts = arg_parts + kwarg_parts
    
    # Join everything with a separator
    return f"{prefix}:{':'.join(key_parts)}"


async def cache_query_result(
    key: str, 
    query_func: Callable[[], Any], 
    ttl: Optional[int] = None
) -> Any:
    """
    Cache the result of a database query.
    
    Args:
        key: Unique cache key
        query_func: Async function that performs the actual query
        ttl: Time-to-live in seconds for this cache entry
        
    Returns:
        Query result (either from cache or fresh query)
    """
    cache_config = get_cache_config()
    
    if not cache_config["enabled"]:
        # Cache disabled, execute query directly
        return await query_func()
    
    current_time = time.time()
    ttl = ttl or cache_config["ttl"]
    
    # Check if we have a valid cache entry
    if key in _query_cache:
        cache_entry = _query_cache[key]
        if current_time < cache_entry["expiry"]:
            # Cache hit
            if cache_config["stats_enabled"]:
                _cache_stats["hits"][key] = _cache_stats["hits"].get(key, 0) + 1
            logger.debug(f"Cache hit for key: {key}")
            return cache_entry["result"]
        else:
            # Cache expired
            if cache_config["stats_enabled"]:
                _cache_stats["expirations"][key] = _cache_stats["expirations"].get(key, 0) + 1
            logger.debug(f"Cache expired for key: {key}")
    else:
        # Cache miss
        if cache_config["stats_enabled"]:
            _cache_stats["misses"][key] = _cache_stats["misses"].get(key, 0) + 1
        logger.debug(f"Cache miss for key: {key}")
    
    # Execute the query
    result = await query_func()
    
    # Store in cache
    _query_cache[key] = {
        "result": result,
        "expiry": current_time + ttl,
        "created": current_time,
    }
    
    # Check cache size and evict if necessary
    if len(_query_cache) > cache_config["max_size"]:
        await evict_cache_entries(int(cache_config["max_size"] * 0.2))  # Evict 20%
    
    return result


async def evict_cache_entries(count: int = 1):
    """
    Evict the oldest cache entries.
    
    Args:
        count: Number of entries to evict
    """
    if not _query_cache:
        return
    
    # Sort by creation time
    sorted_entries = sorted(_query_cache.items(), key=lambda x: x[1]["created"])
    
    # Evict the oldest entries
    for i in range(min(count, len(sorted_entries))):
        key = sorted_entries[i][0]
        logger.debug(f"Evicting cache entry: {key}")
        del _query_cache[key]


def cached_query(ttl: Optional[int] = None):
    """
    Decorator for caching database query methods.
    
    Args:
        ttl: Time-to-live in seconds for cache entries
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key based on function name and arguments
            cache_key = generate_cache_key(func.__name__, *args, **kwargs)
            
            # Create query function that captures the original function call
            async def query_func():
                return await func(*args, **kwargs)
            
            # Use cache system
            return await cache_query_result(cache_key, query_func, ttl)
        
        return wrapper
    
    return decorator


async def invalidate_cache(prefix: Optional[str] = None):
    """
    Invalidate cache entries.
    
    Args:
        prefix: Optional prefix to selectively invalidate entries
    """
    global _query_cache
    
    if prefix:
        # Selective invalidation
        keys_to_remove = [k for k in _query_cache.keys() if k.startswith(prefix)]
        for key in keys_to_remove:
            del _query_cache[key]
        logger.info(f"Invalidated {len(keys_to_remove)} cache entries with prefix '{prefix}'")
    else:
        # Full invalidation
        count = len(_query_cache)
        _query_cache = {}
        logger.info(f"Invalidated all {count} cache entries")


async def get_cache_stats() -> Dict[str, Any]:
    """
    Get statistics about the query cache.
    
    Returns:
        Dictionary with cache statistics
    """
    total_hits = sum(_cache_stats["hits"].values())
    total_misses = sum(_cache_stats["misses"].values())
    total_expirations = sum(_cache_stats["expirations"].values())
    total_queries = total_hits + total_misses
    
    hit_rate = (total_hits / total_queries) * 100 if total_queries > 0 else 0
    
    # Calculate memory usage (approximate)
    import sys
    memory_usage = sys.getsizeof(_query_cache)
    for key, value in _query_cache.items():
        memory_usage += sys.getsizeof(key)
        memory_usage += sys.getsizeof(value["result"])
    
    # Get current cache entries count and details
    current_time = time.time()
    active_entries = 0
    expired_entries = 0
    for entry in _query_cache.values():
        if current_time < entry["expiry"]:
            active_entries += 1
        else:
            expired_entries += 1
    
    return {
        "total_entries": len(_query_cache),
        "active_entries": active_entries,
        "expired_entries": expired_entries,
        "total_hits": total_hits,
        "total_misses": total_misses,
        "total_expirations": total_expirations,
        "hit_rate_percentage": round(hit_rate, 2),
        "memory_usage_bytes": memory_usage,
        "top_hits": sorted(_cache_stats["hits"].items(), key=lambda x: x[1], reverse=True)[:10],
        "top_misses": sorted(_cache_stats["misses"].items(), key=lambda x: x[1], reverse=True)[:10],
    }


async def start_cache_maintenance():
    """Start background task for cache maintenance."""
    cache_config = get_config().get("database", {}).get("query_cache", {})
    
    # If maintenance is disabled, don't start the task
    if not cache_config.get("maintenance_enabled", True):
        return
    
    maintenance_interval = cache_config.get("maintenance_interval", 300)  # 5 minutes
    
    while True:
        try:
            # Cleanup expired entries
            current_time = time.time()
            keys_to_remove = [
                k for k, v in _query_cache.items() 
                if current_time >= v["expiry"]
            ]
            
            for key in keys_to_remove:
                del _query_cache[key]
                
            if keys_to_remove:
                logger.debug(f"Cache maintenance removed {len(keys_to_remove)} expired entries")
                
            # Check cache size and evict if necessary
            max_size = cache_config.get("max_size", DEFAULT_CACHE_SIZE)
            if len(_query_cache) > max_size:
                await evict_cache_entries(int(max_size * 0.2))  # Evict 20%
                
        except Exception as e:
            logger.error(f"Error in cache maintenance: {e}")
            
        # Wait for next maintenance cycle
        await asyncio.sleep(maintenance_interval)
