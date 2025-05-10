"""
Utilities module for the trading system.

This module provides common utilities:
- Logging configuration
- Redis client
- Database client
- Memory caching
- Performance monitoring
"""

import functools
import logging
import time
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast

from .db_client import timescaledb_client
from .logging import get_logger, setup_logger
from .redis_client import redis_client

# Setup module logger
logger = get_logger(__name__)

# Type definitions for type hints
T = TypeVar("T")
CacheFunction = Callable[..., T]


def timed(func: Callable) -> Callable:
    """
    Decorator to measure and log the execution time of a function.

    Args:
        func: The function to be timed

    Returns:
        Wrapped function that logs execution time
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            duration = end_time - start_time
            logger.debug(f"Function {func.__name__} executed in {duration:.6f} seconds")

    return wrapper


def redis_cached(ttl: int = 3600):
    """
    Decorator to cache function results in Redis.

    Args:
        ttl: Time to live in seconds (default: 1 hour)

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a cache key based on function name and arguments
            key_parts = [func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = "cache:" + ":".join(key_parts)

            # Try to get from cache
            cached = redis_client.get(cache_key)
            if cached:
                return cached

            # Execute function and cache result
            result = func(*args, **kwargs)
            redis_client.set(cache_key, result, ex=ttl)
            return result

        return wrapper

    return decorator


def memory_cached(maxsize: int = 128):
    """
    Decorator to cache function results in memory with LRU policy.

    Args:
        maxsize: Maximum cache size (default: 128)

    Returns:
        Decorator function
    """
    return functools.lru_cache(maxsize=maxsize)


__all__ = [
    # Logging
    "setup_logger",
    "get_logger",
    # Clients
    "redis_client",
    "timescaledb_client",
    # Performance utilities
    "timed",
    "redis_cached",
    "memory_cached",
]
