# src/monitoring.py
"""
Performance monitoring and profiling for the Legal Research Engine.

Features:
- Timing decorators for function performance measurement
- Metrics collection and aggregation
- Performance profiling with detailed statistics
- Memory usage tracking
- Query performance analytics
- System resource monitoring
"""

import time
import logging
import psutil
import threading
from functools import wraps
from typing import Dict, List, Any, Optional
from collections import defaultdict
import json
from datetime import datetime, timedelta
from . import config
import asyncio

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Comprehensive performance monitoring system"""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.timers = {}
        self.lock = threading.Lock()
        self.enabled = config.PERFORMANCE_CONFIG.get("monitoring_enabled", True)

    def start_timer(self, name: str):
        """Start a performance timer"""
        if not self.enabled:
            return
        with self.lock:
            self.timers[name] = time.time()

    def stop_timer(self, name: str) -> float:
        """Stop a performance timer and return elapsed time"""
        if not self.enabled:
            return 0.0

        with self.lock:
            if name not in self.timers:
                logger.warning(f"Timer '{name}' was not started")
                return 0.0

            elapsed = time.time() - self.timers[name]
            self.metrics[f"timer_{name}"].append({
                "duration": elapsed,
                "timestamp": datetime.now().isoformat()
            })
            del self.timers[name]
            return elapsed

    def record_metric(self, name: str, value: Any, metadata: Optional[Dict] = None):
        """Record a custom metric"""
        if not self.enabled:
            return

        with self.lock:
            metric_data = {
                "value": value,
                "timestamp": datetime.now().isoformat()
            }
            if metadata:
                metric_data.update(metadata)

            self.metrics[name].append(metric_data)

    def get_system_stats(self) -> Dict:
        """Get current system resource statistics"""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_mb": psutil.virtual_memory().used / 1024 / 1024,
            "memory_available_mb": psutil.virtual_memory().available / 1024 / 1024,
            "disk_usage_percent": psutil.disk_usage('/').percent,
            "timestamp": datetime.now().isoformat()
        }

    def get_performance_summary(self, last_n_minutes: int = 60) -> Dict:
        """Get performance summary for the last N minutes"""
        if not self.enabled:
            return {"monitoring_disabled": True}

        cutoff_time = datetime.now() - timedelta(minutes=last_n_minutes)
        summary = {}

        with self.lock:
            for metric_name, values in self.metrics.items():
                # Filter recent values
                recent_values = [
                    v for v in values
                    if datetime.fromisoformat(v["timestamp"]) > cutoff_time
                ]

                if not recent_values:
                    continue

                if metric_name.startswith("timer_"):
                    durations = [v["duration"] for v in recent_values]
                    summary[metric_name] = {
                        "count": len(durations),
                        "avg_duration": sum(durations) / len(durations),
                        "min_duration": min(durations),
                        "max_duration": max(durations),
                        "total_duration": sum(durations)
                    }
                else:
                    summary[metric_name] = {
                        "count": len(recent_values),
                        "values": recent_values[-10:]  # Last 10 values
                    }

        return summary

    def export_metrics(self, filepath: str):
        """Export metrics to a JSON file"""
        if not self.enabled:
            return

        with self.lock:
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "metrics": dict(self.metrics),
                "system_info": self.get_system_stats()
            }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Metrics exported to {filepath}")

# Global monitor instance
_monitor = PerformanceMonitor()

def get_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance"""
    return _monitor

def timing_decorator(func_name: Optional[str] = None):
    """Decorator to time function execution"""
    def decorator(func):
        name = func_name or f"{func.__module__}.{func.__name__}"

        @wraps(func)
        def wrapper(*args, **kwargs):
            monitor = get_monitor()
            monitor.start_timer(name)

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = monitor.stop_timer(name)
                logger.debug(f"{name} completed in {duration:.3f}s")

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            monitor = get_monitor()
            monitor.start_timer(name)

            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = monitor.stop_timer(name)
                logger.debug(f"{name} completed in {duration:.3f}s")

        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    return decorator

def query_performance_tracker(query_type: str = "general"):
    """Decorator specifically for tracking query performance"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            monitor = get_monitor()
            monitor.start_timer(f"query_{query_type}")

            # Record query metadata
            monitor.record_metric(
                f"query_{query_type}_start",
                1,
                {"function": func.__name__, "args_count": len(args)}
            )

            try:
                result = func(*args, **kwargs)

                # Record successful query
                monitor.record_metric(
                    f"query_{query_type}_success",
                    1,
                    {"function": func.__name__}
                )

                return result
            except Exception as e:
                # Record failed query
                monitor.record_metric(
                    f"query_{query_type}_error",
                    1,
                    {"function": func.__name__, "error": str(e)}
                )
                raise
            finally:
                duration = monitor.stop_timer(f"query_{query_type}")

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            monitor = get_monitor()
            monitor.start_timer(f"query_{query_type}")

            # Record query metadata
            monitor.record_metric(
                f"query_{query_type}_start",
                1,
                {"function": func.__name__, "args_count": len(args)}
            )

            try:
                result = await func(*args, **kwargs)

                # Record successful query
                monitor.record_metric(
                    f"query_{query_type}_success",
                    1,
                    {"function": func.__name__}
                )

                return result
            except Exception as e:
                # Record failed query
                monitor.record_metric(
                    f"query_{query_type}_error",
                    1,
                    {"function": func.__name__, "error": str(e)}
                )
                raise
            finally:
                duration = monitor.stop_timer(f"query_{query_type}")

        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    return decorator

# Convenience functions for common monitoring tasks
def record_memory_usage(label: str = "general"):
    """Record current memory usage"""
    monitor = get_monitor()
    memory_info = psutil.virtual_memory()
    monitor.record_metric(
        f"memory_{label}",
        memory_info.percent,
        {
            "used_mb": memory_info.used / 1024 / 1024,
            "available_mb": memory_info.available / 1024 / 1024
        }
    )

def record_cache_hit(cache_name: str):
    """Record a cache hit"""
    monitor = get_monitor()
    monitor.record_metric(f"cache_{cache_name}_hit", 1)

def record_cache_miss(cache_name: str):
    """Record a cache miss"""
    monitor = get_monitor()
    monitor.record_metric(f"cache_{cache_name}_miss", 1)

def log_performance_summary():
    """Log a summary of current performance metrics"""
    monitor = get_monitor()
    summary = monitor.get_performance_summary()

    if summary:
        logger.info("=== Performance Summary ===")
        for metric_name, data in summary.items():
            if "timer" in metric_name:
                logger.info(f"{metric_name}: {data['count']} calls, "
                          f"avg {data['avg_duration']:.3f}s, "
                          f"total {data['total_duration']:.3f}s")
            else:
                logger.info(f"{metric_name}: {data['count']} events")
        logger.info("==========================")
    else:
        logger.info("No performance metrics available")