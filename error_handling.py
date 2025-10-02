"""
Error handling utilities with retries and exponential backoff.
"""

import asyncio
import time
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

T = TypeVar('T')


async def retry_async(
    func: Callable[..., Any],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Any:
    """
    Retry an async function with exponential backoff.
    
    Args:
        func: Async function to retry.
        max_retries: Maximum number of retry attempts.
        initial_delay: Initial delay between retries in seconds.
        backoff_factor: Multiplier for delay after each retry.
        exceptions: Tuple of exception types to catch and retry.
    
    Returns:
        Result of the function call.
    
    Raises:
        Last exception if all retries are exhausted.
    """
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except exceptions as e:
            last_exception = e
            if attempt < max_retries:
                print(f"  Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
                delay *= backoff_factor
            else:
                print(f"  All {max_retries + 1} attempts failed")
                raise last_exception


def with_timeout(timeout_seconds: float):
    """
    Decorator to add timeout to async functions.
    
    Args:
        timeout_seconds: Timeout in seconds.
    
    Returns:
        Decorated function that raises TimeoutError if exceeded.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                raise TimeoutError(f"{func.__name__} exceeded timeout of {timeout_seconds}s")
        return wrapper
    return decorator


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, max_calls: int, time_window: float):
        """
        Initialize rate limiter.
        
        Args:
            max_calls: Maximum calls allowed in time window.
            time_window: Time window in seconds.
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
    
    async def acquire(self):
        """Wait if necessary to respect rate limit."""
        now = time.time()
        
        self.calls = [t for t in self.calls if now - t < self.time_window]
        
        if len(self.calls) >= self.max_calls:
            sleep_time = self.time_window - (now - self.calls[0])
            if sleep_time > 0:
                print(f"  Rate limit: waiting {sleep_time:.1f}s...")
                await asyncio.sleep(sleep_time)
                now = time.time()
                self.calls = [t for t in self.calls if now - t < self.time_window]
        
        self.calls.append(now)
