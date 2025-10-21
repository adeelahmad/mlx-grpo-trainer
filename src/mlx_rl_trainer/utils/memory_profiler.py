# /src/mlx_rl_trainer/utils/memory_profiler.py
# Revision: 001
# Goal: Memory profiling and leak detection utilities
# Type: New Code
# Description: Comprehensive memory monitoring and optimization tools

import logging
import time
import gc
from typing import Dict, Any, List, Optional, Tuple, Callable
from collections import deque
from dataclasses import dataclass
import numpy as np

try:
    import mlx.core as mx

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Single memory measurement snapshot."""

    timestamp: float
    mlx_allocated_mb: float = 0.0
    mlx_cached_mb: float = 0.0
    mlx_peak_mb: float = 0.0
    system_rss_mb: float = 0.0
    system_available_mb: float = 0.0
    gc_count: int = 0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "mlx_allocated_mb": self.mlx_allocated_mb,
            "mlx_cached_mb": self.mlx_cached_mb,
            "mlx_peak_mb": self.mlx_peak_mb,
            "system_rss_mb": self.system_rss_mb,
            "system_available_mb": self.system_available_mb,
            "gc_count": self.gc_count,
        }


class MemoryProfiler:
    """
    Comprehensive memory profiling and leak detection.

    Features:
    - MLX memory tracking
    - System memory monitoring
    - Memory leak detection
    - Automatic cleanup triggers
    - Memory usage alerts
    """

    def __init__(
        self,
        max_history: int = 1000,
        alert_threshold_mb: float = 8000.0,
        leak_detection_window: int = 50,
    ):
        """
        Initialize memory profiler.

        Args:
            max_history: Maximum snapshots to keep
            alert_threshold_mb: Memory threshold for alerts
            leak_detection_window: Window for leak detection
        """
        self.max_history = max_history
        self.alert_threshold_mb = alert_threshold_mb
        self.leak_detection_window = leak_detection_window

        # Memory history
        self.history: deque = deque(maxlen=max_history)

        # Statistics
        self.peak_memory = 0.0
        self.total_cleanups = 0
        self.total_alerts = 0

        # Check availability
        self.mlx_available = MLX_AVAILABLE
        self.psutil_available = PSUTIL_AVAILABLE

        if not self.mlx_available:
            logger.warning("MLX not available, memory profiling will be limited")

        if not self.psutil_available:
            logger.warning("psutil not available, system memory tracking disabled")

        logger.info("Initialized MemoryProfiler")

    def take_snapshot(self) -> MemorySnapshot:
        """Take a memory snapshot."""
        snapshot = MemorySnapshot(timestamp=time.time())

        # MLX memory
        if self.mlx_available:
            try:
                snapshot.mlx_allocated_mb = mx.metal.get_active_memory() / 1048576
                snapshot.mlx_cached_mb = mx.metal.get_cache_memory() / 1048576
                snapshot.mlx_peak_mb = mx.metal.get_peak_memory() / 1048576
            except Exception as e:
                logger.debug(f"Could not get MLX memory: {e}")

        # System memory
        if self.psutil_available:
            try:
                process = psutil.Process()
                mem_info = process.memory_info()
                snapshot.system_rss_mb = mem_info.rss / 1048576

                system_mem = psutil.virtual_memory()
                snapshot.system_available_mb = system_mem.available / 1048576
            except Exception as e:
                logger.debug(f"Could not get system memory: {e}")

        # Garbage collection
        snapshot.gc_count = len(gc.get_objects())

        # Update peak
        if snapshot.mlx_allocated_mb > self.peak_memory:
            self.peak_memory = snapshot.mlx_allocated_mb

        # Store snapshot
        self.history.append(snapshot)

        return snapshot

    def get_current_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        snapshot = self.take_snapshot()
        return snapshot.to_dict()

    def check_memory_health(self) -> Tuple[bool, str]:
        """
        Check memory health.

        Returns:
            (is_healthy, message)
        """
        snapshot = self.take_snapshot()

        # Check if memory exceeds threshold
        if snapshot.mlx_allocated_mb > self.alert_threshold_mb:
            self.total_alerts += 1
            return (
                False,
                f"Memory usage ({snapshot.mlx_allocated_mb:.1f}MB) exceeds threshold ({self.alert_threshold_mb:.1f}MB)",
            )

        # Check for memory leak
        leak_detected, leak_msg = self.detect_memory_leak()
        if leak_detected:
            self.total_alerts += 1
            return False, leak_msg

        # Check system memory
        if self.psutil_available and snapshot.system_available_mb < 1000:
            self.total_alerts += 1
            return (
                False,
                f"Low system memory: {snapshot.system_available_mb:.1f}MB available",
            )

        return True, "Memory health OK"

    def detect_memory_leak(self) -> Tuple[bool, str]:
        """
        Detect potential memory leaks.

        Returns:
            (leak_detected, description)
        """
        if len(self.history) < self.leak_detection_window:
            return False, "Insufficient data"

        # Get recent memory usage
        recent = list(self.history)[-self.leak_detection_window :]
        older = (
            list(self.history)[
                -self.leak_detection_window * 2 : -self.leak_detection_window
            ]
            if len(self.history) >= self.leak_detection_window * 2
            else None
        )

        if older is None:
            return False, "Insufficient data"

        # Calculate average allocated memory
        recent_avg = np.mean([s.mlx_allocated_mb for s in recent])
        older_avg = np.mean([s.mlx_allocated_mb for s in older])

        # Check for sustained increase
        increase_pct = (recent_avg - older_avg) / max(older_avg, 1) * 100

        if increase_pct > 30:
            return (
                True,
                f"Possible memory leak: {increase_pct:.1f}% increase over last {self.leak_detection_window} steps",
            )

        return False, "No leak detected"

    def get_memory_trend(self) -> str:
        """Get memory usage trend."""
        if len(self.history) < 20:
            return "INSUFFICIENT_DATA"

        recent = [s.mlx_allocated_mb for s in list(self.history)[-10:]]
        older = [s.mlx_allocated_mb for s in list(self.history)[-20:-10]]

        recent_avg = np.mean(recent)
        older_avg = np.mean(older)

        change_pct = (recent_avg - older_avg) / max(older_avg, 1) * 100

        if change_pct > 20:
            return f"INCREASING ({change_pct:+.1f}%)"
        elif change_pct < -20:
            return f"DECREASING ({change_pct:+.1f}%)"
        else:
            return f"STABLE ({change_pct:+.1f}%)"

    def aggressive_cleanup(self) -> Dict[str, float]:
        """Perform aggressive memory cleanup."""
        before = self.take_snapshot()

        # Garbage collection
        gc.collect()

        # MLX cleanup
        if self.mlx_available:
            try:
                mx.metal.clear_cache()
                mx.clear_cache()
            except Exception as e:
                logger.debug(f"MLX cleanup error: {e}")

        # Force another GC
        gc.collect()

        after = self.take_snapshot()

        self.total_cleanups += 1

        # Calculate freed memory
        freed_mb = before.mlx_allocated_mb - after.mlx_allocated_mb

        logger.info(f"Aggressive cleanup freed {freed_mb:.1f}MB")

        return {
            "freed_mb": freed_mb,
            "before_mb": before.mlx_allocated_mb,
            "after_mb": after.mlx_allocated_mb,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get profiler statistics."""
        if not self.history:
            return {}

        allocated = [s.mlx_allocated_mb for s in self.history]

        return {
            "peak_memory_mb": self.peak_memory,
            "current_memory_mb": allocated[-1] if allocated else 0,
            "avg_memory_mb": np.mean(allocated) if allocated else 0,
            "total_cleanups": self.total_cleanups,
            "total_alerts": self.total_alerts,
            "snapshots_taken": len(self.history),
            "memory_trend": self.get_memory_trend(),
        }

    def export_history(self) -> List[Dict[str, float]]:
        """Export memory history."""
        return [s.to_dict() for s in self.history]

    def reset(self) -> None:
        """Reset profiler state."""
        self.history.clear()
        self.peak_memory = 0.0
        self.total_cleanups = 0
        self.total_alerts = 0
        logger.info("Reset memory profiler")


class MemoryMonitor:
    """Context manager for monitoring memory in code blocks."""

    def __init__(self, profiler: MemoryProfiler, name: str):
        """
        Initialize memory monitor.

        Args:
            profiler: MemoryProfiler instance
            name: Name of the monitored block
        """
        self.profiler = profiler
        self.name = name
        self.start_snapshot: Optional[MemorySnapshot] = None
        self.end_snapshot: Optional[MemorySnapshot] = None

    def __enter__(self):
        """Enter context."""
        self.start_snapshot = self.profiler.take_snapshot()
        logger.debug(f"Memory monitor started: {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        self.end_snapshot = self.profiler.take_snapshot()

        # Calculate memory change
        delta_mb = (
            self.end_snapshot.mlx_allocated_mb - self.start_snapshot.mlx_allocated_mb
        )

        logger.debug(
            f"Memory monitor finished: {self.name} " f"(delta: {delta_mb:+.1f}MB)"
        )

        # Alert if large increase
        if delta_mb > 500:
            logger.warning(f"Large memory increase in {self.name}: {delta_mb:.1f}MB")


def memory_profiled(profiler: MemoryProfiler):
    """Decorator for memory profiling functions."""

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            with MemoryMonitor(profiler, func.__name__):
                return func(*args, **kwargs)

        return wrapper

    return decorator


# Dependencies: mlx, psutil, numpy
# Install: pip install mlx psutil numpy
# Usage: profiler = MemoryProfiler()
#        snapshot = profiler.take_snapshot()
#        health, msg = profiler.check_memory_health()
# Status: Complete and commit-ready
