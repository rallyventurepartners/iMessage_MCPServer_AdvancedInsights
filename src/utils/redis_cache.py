import json
import logging
import asyncio
import hashlib
from functools import wraps
from typing import Any, Callable, Dict, Optional, Union

# Configure logging
logger = logging.getLogger(__name__)

# We'll use a simple in-memory cache as our primary implementation
# No external Redis dependency needed
class AsyncRedisCache:
    """Asynchronous cache manager with in-memory storage."""
    
    _instance = None
    _lock = asyncio.Lock()
    _initialized = False
    _memory_cache = {}  # In-memory cache
    _memory_cache_ttl = {}  # TTL data for in-memory cache
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(AsyncRedisCache, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0", ttl: int = 3600):
        """Initialize the cache manager.
        
        Args:
            redis_url: Ignored, kept for compatibility
            ttl: Default cache TTL in seconds (1 hour default)
        """
        if not self._initialized:
            self.default_ttl = ttl
            self._initialized = True
            logger.info(f"Initialized AsyncRedisCache with in-memory storage")
    
    async def initialize(self):
        """Initialize the cache system."""
        return True
    
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            The cached value, or None if not found
        """
        # Clean expired items in memory cache
        await self._clean_expired()
        
        # Check if key exists
        if key in self._memory_cache:
            return self._memory_cache[key]
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds, or None to use default
            
        Returns:
            True if successful, False otherwise
        """
        if ttl is None:
            ttl = self.default_ttl
            
        try:
            self._memory_cache[key] = value
            self._memory_cache_ttl[key] = asyncio.get_event_loop().time() + ttl
            return True
        except Exception as e:
            logger.warning(f"Error setting key {key} in memory cache: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete a key from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful, False otherwise
        """
        if key in self._memory_cache:
            del self._memory_cache[key]
            if key in self._memory_cache_ttl:
                del self._memory_cache_ttl[key]
            return True
        return False
    
    async def flush(self) -> bool:
        """Flush the entire cache.
        
        Returns:
            True if successful, False otherwise
        """
        self._memory_cache.clear()
        self._memory_cache_ttl.clear()
        return True
    
    async def _clean_expired(self):
        """Clean expired items from the in-memory cache."""
        now = asyncio.get_event_loop().time()
        expired_keys = [k for k, exp in self._memory_cache_ttl.items() if exp <= now]
        
        for key in expired_keys:
            if key in self._memory_cache:
                del self._memory_cache[key]
            if key in self._memory_cache_ttl:
                del self._memory_cache_ttl[key]
    
    @staticmethod
    def generate_key(prefix: str, *args, **kwargs) -> str:
        """Generate a unique cache key based on function arguments.
        
        Args:
            prefix: Key prefix (usually function name)
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            A unique cache key
        """
        # Create a string representation of args and kwargs
        key_parts = [prefix]
        
        # Add positional args
        for arg in args:
            key_parts.append(str(arg))
        
        # Add keyword args (sorted to ensure consistent keys)
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}:{v}")
        
        # Join and hash to create a compact, unique key
        key_str = ":".join(key_parts)
        return f"cache:{prefix}:{hashlib.md5(key_str.encode()).hexdigest()}"

def cached(ttl: Optional[int] = None, key_prefix: Optional[str] = None):
    """Decorator to cache function results.
    
    Args:
        ttl: Cache TTL in seconds, or None to use default
        key_prefix: Custom key prefix, or None to use function name
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get cache instance
            cache = AsyncRedisCache()
            
            # Generate cache key
            prefix = key_prefix or func.__name__
            key = cache.generate_key(prefix, *args, **kwargs)
            
            # Try to get from cache
            cached_result = await cache.get(key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {key}")
                return cached_result
            
            # Cache miss, call the function
            logger.debug(f"Cache miss for {key}")
            result = await func(*args, **kwargs)
            
            # Store in cache
            await cache.set(key, result, ttl)
            
            return result
        return wrapper
    return decorator 