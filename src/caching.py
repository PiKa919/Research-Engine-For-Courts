# src/caching.py

import os
import pickle
import time
import logging
from typing import Any, Dict, Optional
from abc import ABC, abstractmethod

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from src import config

logger = logging.getLogger(__name__)

class CacheBackend(ABC):
    """Abstract base class for cache backends"""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with optional TTL"""
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete value from cache"""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all values from cache"""
        pass

class MemoryCache(CacheBackend):
    """In-memory cache implementation"""

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl

    def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache"""
        if key in self.cache:
            entry = self.cache[key]
            if time.time() < entry['expires_at']:
                return entry['value']
            else:
                # Entry expired, remove it
                del self.cache[key]
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in memory cache"""
        if ttl is None:
            ttl = self.default_ttl

        # Remove expired entries if cache is full
        if len(self.cache) >= self.max_size:
            self._cleanup_expired()

        if len(self.cache) >= self.max_size:
            # Remove oldest entry (simple LRU approximation)
            oldest_key = min(self.cache.keys(),
                           key=lambda k: self.cache[k]['created_at'])
            del self.cache[oldest_key]

        self.cache[key] = {
            'value': value,
            'expires_at': time.time() + ttl,
            'created_at': time.time()
        }

    def delete(self, key: str) -> None:
        """Delete value from memory cache"""
        if key in self.cache:
            del self.cache[key]

    def clear(self) -> None:
        """Clear all values from memory cache"""
        self.cache.clear()

    def _cleanup_expired(self) -> None:
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if current_time >= entry['expires_at']
        ]
        for key in expired_keys:
            del self.cache[key]

class RedisCache(CacheBackend):
    """Redis cache implementation"""

    def __init__(self, url: str = "redis://localhost:6379", default_ttl: int = 3600):
        if not REDIS_AVAILABLE:
            raise ImportError("redis package is required for RedisCache")

        self.redis = redis.from_url(url)
        self.default_ttl = default_ttl

    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        try:
            data = self.redis.get(key)
            if data:
                return pickle.loads(data)
        except Exception as e:
            logger.warning(f"Error getting from Redis cache: {e}")
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in Redis cache"""
        try:
            if ttl is None:
                ttl = self.default_ttl

            data = pickle.dumps(value)
            self.redis.setex(key, ttl, data)
        except Exception as e:
            logger.warning(f"Error setting Redis cache: {e}")

    def delete(self, key: str) -> None:
        """Delete value from Redis cache"""
        try:
            self.redis.delete(key)
        except Exception as e:
            logger.warning(f"Error deleting from Redis cache: {e}")

    def clear(self) -> None:
        """Clear all values from Redis cache"""
        try:
            self.redis.flushdb()
        except Exception as e:
            logger.warning(f"Error clearing Redis cache: {e}")

class FileCache(CacheBackend):
    """File-based cache for large objects"""

    def __init__(self, cache_dir: str = "cache", default_ttl: int = 3600):
        self.cache_dir = cache_dir
        self.default_ttl = default_ttl
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_path(self, key: str) -> str:
        """Get file path for cache key"""
        safe_key = "".join(c for c in key if c.isalnum() or c in ('_', '-')).rstrip()
        return os.path.join(self.cache_dir, f"{safe_key}.pkl")

    def get(self, key: str) -> Optional[Any]:
        """Get value from file cache"""
        cache_path = self._get_cache_path(key)
        if os.path.exists(cache_path):
            try:
                # Check if file is expired
                if time.time() - os.path.getmtime(cache_path) > self.default_ttl:
                    os.remove(cache_path)
                    return None

                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Error reading from file cache: {e}")
                # Remove corrupted file
                if os.path.exists(cache_path):
                    os.remove(cache_path)
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in file cache"""
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.warning(f"Error writing to file cache: {e}")

    def delete(self, key: str) -> None:
        """Delete value from file cache"""
        cache_path = self._get_cache_path(key)
        if os.path.exists(cache_path):
            try:
                os.remove(cache_path)
            except Exception as e:
                logger.warning(f"Error deleting from file cache: {e}")

    def clear(self) -> None:
        """Clear all values from file cache"""
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    os.remove(os.path.join(self.cache_dir, filename))
        except Exception as e:
            logger.warning(f"Error clearing file cache: {e}")

class MultiLevelCache:
    """Multi-level cache with memory, Redis, and file fallbacks"""

    def __init__(self):
        self.caches = []

        # Initialize memory cache (fastest)
        self.memory_cache = MemoryCache(
            max_size=config.CACHE_CONFIG["max_cache_size"],
            default_ttl=config.CACHE_CONFIG["ttl_seconds"]
        )
        self.caches.append(("memory", self.memory_cache))

        # Initialize Redis cache if available
        if REDIS_AVAILABLE:
            try:
                self.redis_cache = RedisCache(
                    url=config.CACHE_CONFIG["redis_url"],
                    default_ttl=config.CACHE_CONFIG["ttl_seconds"]
                )
                self.caches.append(("redis", self.redis_cache))
                logger.info("Redis cache initialized")
            except Exception as e:
                logger.warning(f"Redis cache initialization failed: {e}")

        # Initialize file cache (slowest but persistent)
        self.file_cache = FileCache(
            cache_dir=config.CACHE_CONFIG["cache_dir"],
            default_ttl=config.CACHE_CONFIG["ttl_seconds"]
        )
        self.caches.append(("file", self.file_cache))

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache hierarchy"""
        for cache_name, cache in self.caches:
            value = cache.get(key)
            if value is not None:
                # Update faster caches with the value
                for faster_cache_name, faster_cache in self.caches:
                    if faster_cache_name == cache_name:
                        break
                    try:
                        faster_cache.set(key, value)
                    except Exception:
                        pass  # Ignore errors in cache promotion
                return value
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in all cache levels"""
        for cache_name, cache in self.caches:
            try:
                cache.set(key, value, ttl)
            except Exception as e:
                logger.warning(f"Error setting {cache_name} cache: {e}")

    def delete(self, key: str) -> None:
        """Delete value from all cache levels"""
        for cache_name, cache in self.caches:
            try:
                cache.delete(key)
            except Exception as e:
                logger.warning(f"Error deleting from {cache_name} cache: {e}")

    def clear(self) -> None:
        """Clear all cache levels"""
        for cache_name, cache in self.caches:
            try:
                cache.clear()
            except Exception as e:
                logger.warning(f"Error clearing {cache_name} cache: {e}")

# Global cache instance
_cache_instance = None

def get_cache() -> MultiLevelCache:
    """Get global cache instance"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = MultiLevelCache()
    return _cache_instance

def cached_query(func):
    """Decorator for caching query results"""
    def wrapper(*args, **kwargs):
        # Create cache key from function name and arguments
        key_parts = [func.__name__]
        key_parts.extend(str(arg) for arg in args)
        key_parts.extend(f"{k}:{v}" for k, v in kwargs.items())
        cache_key = "|".join(key_parts)

        cache = get_cache()
        result = cache.get(cache_key)

        if result is None:
            result = func(*args, **kwargs)
            cache.set(cache_key, result)

        return result
    return wrapper