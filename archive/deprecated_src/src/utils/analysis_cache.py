"""
Advanced caching layer for expensive NLP and analysis operations.

This module extends the basic query cache to provide specialized caching
for sentiment analysis, topic modeling, and other computationally expensive
operations with intelligent invalidation strategies.
"""

import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import json

from .query_cache import cached_query, cache_query_result, generate_cache_key

logger = logging.getLogger(__name__)

# Specialized cache TTLs for different analysis types
CACHE_TTL_SETTINGS = {
    "sentiment": 3600,      # 1 hour - sentiment doesn't change for historical messages
    "topic": 7200,          # 2 hours - topics are relatively stable
    "llm_insights": 1800,   # 30 minutes - LLM insights might need fresher data
    "conversation_depth": 3600,  # 1 hour
    "emotional_dynamics": 3600,  # 1 hour
    "network_analysis": 7200,    # 2 hours - social networks change slowly
}


def get_message_hash(messages: List[Dict[str, Any]]) -> str:
    """
    Generate a hash for a list of messages to use as cache key component.
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        SHA256 hash of the messages
    """
    # Sort messages by ID to ensure consistent hashing
    sorted_messages = sorted(messages, key=lambda m: m.get('id', ''))
    
    # Extract relevant fields for hashing
    hash_data = []
    for msg in sorted_messages:
        hash_data.append({
            'id': msg.get('id'),
            'text': msg.get('text', ''),
            'date': msg.get('date', ''),
            'sender': msg.get('sender', '')
        })
    
    # Convert to JSON string and hash
    json_str = json.dumps(hash_data, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


async def cache_sentiment_analysis(
    text: str,
    analysis_func,
    additional_params: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Cache sentiment analysis results for text.
    
    Args:
        text: Text to analyze
        analysis_func: Function that performs sentiment analysis
        additional_params: Additional parameters for cache key
        
    Returns:
        Cached or fresh sentiment analysis results
    """
    # Generate cache key
    text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
    cache_key = f"sentiment:{text_hash}"
    
    if additional_params:
        param_str = ":".join(f"{k}={v}" for k, v in sorted(additional_params.items()))
        cache_key += f":{param_str}"
    
    async def perform_analysis():
        return await analysis_func(text)
    
    return await cache_query_result(
        cache_key,
        perform_analysis,
        ttl=CACHE_TTL_SETTINGS["sentiment"]
    )


async def cache_topic_analysis(
    messages: List[Dict[str, Any]],
    analysis_func,
    time_window: Optional[str] = None
) -> Dict[str, Any]:
    """
    Cache topic analysis results for a set of messages.
    
    Args:
        messages: Messages to analyze
        analysis_func: Function that performs topic analysis
        time_window: Time window for analysis (affects cache key)
        
    Returns:
        Cached or fresh topic analysis results
    """
    # Generate cache key based on messages
    msg_hash = get_message_hash(messages)
    cache_key = f"topics:{msg_hash}"
    
    if time_window:
        cache_key += f":{time_window}"
    
    async def perform_analysis():
        return await analysis_func(messages)
    
    return await cache_query_result(
        cache_key,
        perform_analysis,
        ttl=CACHE_TTL_SETTINGS["topic"]
    )


async def cache_conversation_intelligence(
    contact_id: str,
    time_period: str,
    analysis_depth: str,
    analysis_func
) -> Dict[str, Any]:
    """
    Cache comprehensive conversation intelligence analysis.
    
    Args:
        contact_id: Contact identifier
        time_period: Time period for analysis
        analysis_depth: Depth level of analysis
        analysis_func: Function that performs the analysis
        
    Returns:
        Cached or fresh analysis results
    """
    cache_key = f"conv_intel:{contact_id}:{time_period}:{analysis_depth}"
    
    async def perform_analysis():
        return await analysis_func(contact_id, time_period, analysis_depth)
    
    return await cache_query_result(
        cache_key,
        perform_analysis,
        ttl=CACHE_TTL_SETTINGS["conversation_depth"]
    )


async def cache_llm_insights(
    context_hash: str,
    prompt: str,
    llm_func,
    focus_areas: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Cache LLM-generated insights to avoid redundant API calls.
    
    Args:
        context_hash: Hash of the context/messages being analyzed
        prompt: The prompt being used
        llm_func: Function that calls the LLM
        focus_areas: Optional focus areas for the analysis
        
    Returns:
        Cached or fresh LLM insights
    """
    # Create cache key from prompt hash and context
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
    cache_key = f"llm:{context_hash}:{prompt_hash}"
    
    if focus_areas:
        areas_str = ",".join(sorted(focus_areas))
        cache_key += f":{hashlib.md5(areas_str.encode()).hexdigest()[:8]}"
    
    async def perform_analysis():
        return await llm_func(prompt)
    
    return await cache_query_result(
        cache_key,
        perform_analysis,
        ttl=CACHE_TTL_SETTINGS["llm_insights"]
    )


def invalidate_analysis_cache(contact_id: Optional[str] = None, cache_type: Optional[str] = None):
    """
    Invalidate analysis cache entries.
    
    Args:
        contact_id: Optional contact ID to invalidate caches for
        cache_type: Optional cache type to invalidate (sentiment, topic, etc.)
    """
    from .query_cache import invalidate_cache
    
    if contact_id and cache_type:
        prefix = f"{cache_type}:*{contact_id}*"
    elif cache_type:
        prefix = f"{cache_type}:"
    elif contact_id:
        # Invalidate all analysis types for this contact
        prefixes = ["sentiment:", "topics:", "conv_intel:", "llm:"]
        for prefix in prefixes:
            asyncio.create_task(invalidate_cache(f"{prefix}*{contact_id}*"))
        return
    else:
        prefix = None
    
    asyncio.create_task(invalidate_cache(prefix))


# Decorator for caching analysis functions
def cached_analysis(cache_type: str, key_params: List[str]):
    """
    Decorator for caching analysis functions with custom TTL.
    
    Args:
        cache_type: Type of analysis (determines TTL)
        key_params: List of parameter names to include in cache key
        
    Returns:
        Decorated function
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Build cache key from specified parameters
            cache_key_parts = [cache_type]
            
            # Get function argument names
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Extract key parameters
            for param in key_params:
                if param in bound_args.arguments:
                    value = bound_args.arguments[param]
                    if isinstance(value, list):
                        # For lists (like messages), use hash
                        value = hashlib.md5(str(value).encode()).hexdigest()[:8]
                    cache_key_parts.append(f"{param}={value}")
            
            cache_key = ":".join(cache_key_parts)
            
            # Get TTL for this cache type
            ttl = CACHE_TTL_SETTINGS.get(cache_type, 3600)
            
            async def query_func():
                return await func(*args, **kwargs)
            
            return await cache_query_result(cache_key, query_func, ttl)
        
        return wrapper
    return decorator


# Example usage in sentiment.py:
# @cached_analysis("sentiment", ["text"])
# async def calculate_message_sentiment(text: str) -> Dict[str, Any]:
#     ... existing implementation ...