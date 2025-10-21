# /src/mlx_rl_trainer/monitoring/stats_collector.py
# Revision: 001
# Goal: Comprehensive statistics collection system for training metrics
# Type: New Code
# Description: Centralized stats collection with automatic aggregation and persistence

import logging
import time
import json
import threading
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
import numpy as np
import mlx.core as mx

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class MetricStats:
    """Statistics for a single metric over time."""

    name: str
    values: deque = field(default_factory=lambda: deque(maxlen=10000))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=10000))

    def add_value(self, value: float, timestamp: Optional[float] = None) -> None:
        """Add a new value to the metric."""
        self.values.append(float(value))
        self.timestamps.append(timestamp or time.time())

    def get_statistics(self) -> Dict[str, float]:
        """Calculate comprehensive statistics."""
        if not self.values:
            return {}

        values_array = np.array(list(self.values))
        return {
            "mean": float(np.mean(values_array)),
            "std": float(np.std(values_array)),
            "min": float(np.min(values_array)),
            "max": float(np.max(values_array)),
            "median": float(np.median(values_array)),
            "q25": float(np.percentile(values_array, 25)),
            "q75": float(np.percentile(values_array, 75)),
            "last": float(values_array[-1]),
            "count": len(values_array),
        }

    def get_recent(self, n: int = 100) -> List[float]:
        """Get the most recent n values."""
        return list(self.values)[-n:]

    def get_moving_average(self, window: int = 10) -> List[float]:
        """Calculate moving average."""
        if len(self.values) < window:
            return list(self.values)

        values_array = np.array(list(self.values))
        return list(np.convolve(values_array, np.ones(window) / window, mode="valid"))


@dataclass
class TrainingStats:
    """Comprehensive training statistics."""

    # Loss metrics
    total_loss: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    thinking_loss: float = 0.0
    answer_loss: float = 0.0
    sft_loss: float = 0.0

    # Reward metrics
    reward_mean: float = 0.0
    reward_std: float = 0.0
    reward_min: float = 0.0
    reward_max: float = 0.0

    # Learning metrics
    learning_rate: float = 0.0
    grad_norm: float = 0.0
    kl_divergence: float = 0.0

    # Token metrics
    total_tokens: int = 0
    thinking_tokens: int = 0
    answer_tokens: int = 0

    # Memory metrics
    memory_allocated_mb: float = 0.0
    memory_cached_mb: float = 0.0
    memory_peak_mb: float = 0.0

    # Timing metrics
    step_time_s: float = 0.0
    generation_time_s: float = 0.0
    training_time_s: float = 0.0

    # Gradient metrics
    thinking_grad_norm: float = 0.0
    answer_grad_norm: float = 0.0
    gradient_match_rate: float = 0.0

    # Sample quality metrics
    format_reward: float = 0.0
    content_reward: float = 0.0
    mcq_accuracy: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def update_from_dict(self, data: Dict[str, Any]) -> None:
        """Update from dictionary."""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)


class ComprehensiveStatsCollector:
    """
    Comprehensive statistics collector for training.

    Features:
    - Automatic metric aggregation
    - Time-series storage
    - Statistical analysis
    - Memory-efficient storage
    - Thread-safe operations
    """

    def __init__(
        self, output_dir: Path, max_history: int = 10000, aggregation_window: int = 100
    ):
        """
        Initialize stats collector.

        Args:
            output_dir: Directory to save stats
            max_history: Maximum number of historical values to keep
            aggregation_window: Window size for moving averages
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.max_history = max_history
        self.aggregation_window = aggregation_window

        # Metric storage
        self.metrics: Dict[str, MetricStats] = {}
        self.step_stats: List[TrainingStats] = []
        self.current_step = 0

        # Aggregated statistics
        self.aggregated_stats: Dict[str, Dict[str, float]] = {}

        # Threading
        self._lock = threading.Lock()

        # Session info
        self.session_start_time = time.time()
        self.total_steps = 0

        logger.info(f"Initialized ComprehensiveStatsCollector: {self.output_dir}")

    def record_metric(
        self,
        name: str,
        value: Union[float, int, mx.array, np.ndarray],
        step: Optional[int] = None,
    ) -> None:
        """
        Record a metric value.

        Args:
            name: Metric name
            value: Metric value
            step: Training step (optional)
        """
        # Convert value to float
        if isinstance(value, (mx.array, np.ndarray)):
            if value.size == 1:
                value = float(value.item())
            else:
                value = float(np.mean(value))
        else:
            value = float(value)

        with self._lock:
            if name not in self.metrics:
                self.metrics[name] = MetricStats(name=name)

            self.metrics[name].add_value(value, time.time())

    def record_step_stats(self, stats: TrainingStats, step: int) -> None:
        """Record complete step statistics."""
        with self._lock:
            self.step_stats.append(stats)
            self.current_step = step
            self.total_steps += 1

            # Record individual metrics
            for key, value in stats.to_dict().items():
                self.record_metric(f"step/{key}", value, step)

    def record_batch_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Record batch of metrics."""
        for name, value in metrics.items():
            try:
                self.record_metric(name, value, step)
            except Exception as e:
                logger.warning(f"Failed to record metric {name}: {e}")

    def get_metric_stats(self, name: str) -> Optional[Dict[str, float]]:
        """Get statistics for a specific metric."""
        with self._lock:
            if name in self.metrics:
                return self.metrics[name].get_statistics()
            return None

    def get_all_current_stats(self) -> Dict[str, float]:
        """Get current value of all metrics."""
        with self._lock:
            stats = {}
            for name, metric in self.metrics.items():
                if metric.values:
                    stats[name] = metric.values[-1]
            return stats

    def get_aggregated_stats(self) -> Dict[str, Dict[str, float]]:
        """Get aggregated statistics for all metrics."""
        with self._lock:
            aggregated = {}
            for name, metric in self.metrics.items():
                aggregated[name] = metric.get_statistics()
            return aggregated

    def get_moving_averages(self, window: int = 10) -> Dict[str, List[float]]:
        """Get moving averages for all metrics."""
        with self._lock:
            averages = {}
            for name, metric in self.metrics.items():
                averages[name] = metric.get_moving_average(window)
            return averages

    def get_recent_trends(self, n: int = 100) -> Dict[str, List[float]]:
        """Get recent values for all metrics."""
        with self._lock:
            trends = {}
            for name, metric in self.metrics.items():
                trends[name] = metric.get_recent(n)
            return trends

    def calculate_correlations(self, metrics: List[str]) -> np.ndarray:
        """Calculate correlation matrix between metrics."""
        with self._lock:
            data = []
            for metric_name in metrics:
                if metric_name in self.metrics:
                    data.append(list(self.metrics[metric_name].values))

            if not data:
                return np.array([])

            # Ensure all arrays have same length
            min_len = min(len(arr) for arr in data)
            data = [arr[-min_len:] for arr in data]

            return np.corrcoef(data)

    def export_to_json(self, filepath: Optional[Path] = None) -> None:
        """Export all statistics to JSON."""
        if filepath is None:
            filepath = self.output_dir / f"stats_export_{int(time.time())}.json"

        with self._lock:
            export_data = {
                "session_info": {
                    "start_time": self.session_start_time,
                    "total_steps": self.total_steps,
                    "current_step": self.current_step,
                    "duration_hours": (time.time() - self.session_start_time) / 3600,
                },
                "aggregated_stats": self.get_aggregated_stats(),
                "recent_stats": self.get_all_current_stats(),
            }

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Exported stats to {filepath}")

    def export_to_dataframe(self) -> Optional[Any]:
        """Export statistics to pandas DataFrame."""
        if not PANDAS_AVAILABLE:
            logger.warning("Pandas not available, cannot export to DataFrame")
            return None

        with self._lock:
            data = []
            for name, metric in self.metrics.items():
                for value, timestamp in zip(metric.values, metric.timestamps):
                    data.append(
                        {"metric": name, "value": value, "timestamp": timestamp}
                    )

        if not data:
            return None

        return pd.DataFrame(data)

    def get_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive summary report."""
        with self._lock:
            duration = time.time() - self.session_start_time

            report = {
                "session": {
                    "duration_hours": duration / 3600,
                    "total_steps": self.total_steps,
                    "avg_step_time": duration / max(self.total_steps, 1),
                    "metrics_tracked": len(self.metrics),
                },
                "metrics": {},
            }

            # Add statistics for each metric
            for name, metric in self.metrics.items():
                stats = metric.get_statistics()
                if stats:
                    report["metrics"][name] = stats

            return report

    def reset(self) -> None:
        """Reset all statistics."""
        with self._lock:
            self.metrics.clear()
            self.step_stats.clear()
            self.current_step = 0
            self.total_steps = 0
            self.session_start_time = time.time()

        logger.info("Reset all statistics")

    def cleanup_old_data(self, keep_last_n: int = 1000) -> None:
        """Remove old data to free memory."""
        with self._lock:
            for metric in self.metrics.values():
                if len(metric.values) > keep_last_n:
                    metric.values = deque(
                        list(metric.values)[-keep_last_n:], maxlen=10000
                    )
                    metric.timestamps = deque(
                        list(metric.timestamps)[-keep_last_n:], maxlen=10000
                    )

        logger.info(f"Cleaned up old data, keeping last {keep_last_n} entries")


# Dependencies: numpy, pandas (optional)
# Install: pip install numpy pandas
# Usage: stats_collector = ComprehensiveStatsCollector(output_dir=Path("./stats"))
#        stats_collector.record_metric("loss", 0.5, step=100)
# Status: Complete and commit-ready
