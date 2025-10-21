
"""
Advanced Parameter Distribution Tracking for MLX RL Training

This module provides comprehensive parameter distribution tracking and visualization
for MLX-based reinforcement learning training. It implements sophisticated tracking
and analysis of parameter distributions across model layers, enabling deep insights
into training dynamics and potential issues like weight saturation or vanishing weights.

Key Features:
- Layer-wise parameter distribution tracking with temporal analysis
- Parameter magnitude distribution visualization
- Statistical analysis of parameter distributions
- Detection of abnormal parameter distributions
- Tracking of parameter changes over time
- Visualization of parameter distribution evolution
- Correlation analysis between parameter distributions and model performance

Architecture:
- Strategy Pattern: Multiple parameter analysis algorithms
- Observer Pattern: Real-time parameter distribution monitoring
- Repository Pattern: Efficient parameter history storage
- Factory Pattern: Dynamic visualization creation
- Command Pattern: Encapsulated parameter analysis operations
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Union, Tuple, Any, Set, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import math
import statistics
import json
from pathlib import Path

import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten

from ..progress.exceptions import (
    PerformanceMetricError,
    create_error_context,
)

logger = logging.getLogger(__name__)


class ParameterDistributionMetricType(Enum):
    """Types of parameter distribution metrics that can be collected."""
    MEAN = "mean"
    VARIANCE = "variance"
    SKEWNESS = "skewness"
    KURTOSIS = "kurtosis"
    PERCENTILES = "percentiles"
    HISTOGRAM = "histogram"
    RANGE = "range"
    ZERO_FRACTION = "zero_fraction"
    SPARSITY = "sparsity"
    NORM = "norm"


@dataclass
class ParameterDistributionStatistics:
    """
    Comprehensive statistics for parameter distribution analysis.
    
    This class encapsulates various statistical measures for analyzing
    parameter distributions across model layers and training steps.
    
    Attributes:
        layer_means: Dictionary mapping layer names to parameter means
        layer_variances: Dictionary mapping layer names to parameter variances
        layer_skewness: Dictionary mapping layer names to parameter skewness
        layer_kurtosis: Dictionary mapping layer names to parameter kurtosis
        layer_percentiles: Dictionary mapping layer names to parameter percentiles
        layer_histograms: Dictionary mapping layer names to parameter histograms
        layer_ranges: Dictionary mapping layer names to parameter ranges (min, max)
        layer_zero_fractions: Dictionary mapping layer names to fraction of zero parameters
        layer_sparsity: Dictionary mapping layer names to parameter sparsity
        layer_norms: Dictionary mapping layer names to parameter norms
        global_statistics: Dictionary of global statistics across all parameters
        anomaly_score: Overall parameter distribution anomaly score
        timestamp: When these statistics were computed
    """
    layer_means: Dict[str, float]
    layer_variances: Dict[str, float] = field(default_factory=dict)
    layer_skewness: Dict[str, float] = field(default_factory=dict)
    layer_kurtosis: Dict[str, float] = field(default_factory=dict)
    layer_percentiles: Dict[str, Dict[str, float]] = field(default_factory=dict)
    layer_histograms: Dict[str, Tuple[List[float], List[float]]] = field(default_factory=dict)
    layer_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    layer_zero_fractions: Dict[str, float] = field(default_factory=dict)
    layer_sparsity: Dict[str, float] = field(default_factory=dict)
    layer_norms: Dict[str, float] = field(default_factory=dict)
    global_statistics: Dict[str, float] = field(default_factory=dict)
    anomaly_score: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary for serialization."""
        return {
            'layer_means': self.layer_means,
            'layer_variances': self.layer_variances,
            'layer_skewness': self.layer_skewness,
            'layer_kurtosis': self.layer_kurtosis,
            'layer_percentiles': self.layer_percentiles,
            'layer_ranges': self.layer_ranges,
            'layer_zero_fractions': self.layer_zero_fractions,
            'layer_sparsity': self.layer_sparsity,
            'layer_norms': self.layer_norms,
            'global_statistics': self.global_statistics,
            'anomaly_score': self.anomaly_score,
            'timestamp': self.timestamp
        }


class ParameterDistributionTracker:
    """
    Advanced parameter distribution tracking system for deep learning models.
    
    This class provides comprehensive analysis of parameter distributions
    across model layers, enabling detection of training issues like weight
    saturation or vanishing weights, and providing insights into
    training dynamics.
    
    Features:
    - Layer-wise parameter distribution tracking with temporal analysis
    - Parameter magnitude distribution visualization
    - Statistical analysis of parameter distributions
    - Detection of abnormal parameter distributions
    - Tracking of parameter changes over time
    - Visualization of parameter distribution evolution
    - Correlation analysis between parameter distributions and model performance
    """
    
    def __init__(
        self,
        history_size: int = 100,
        anomaly_threshold: float = 3.0,
        compute_histograms: bool = True,
        compute_percentiles: bool = True,
        percentiles: List[float] = None,
        histogram_bins: int = 50,
        enable_anomaly_detection: bool = True
    ):
        """
        Initialize parameter distribution tracker.
        
        Args:
            history_size: Number of historical parameter distributions to retain
            anomaly_threshold: Threshold for anomaly detection (z-score)
            compute_histograms: Whether to compute parameter histograms
            compute_percentiles: Whether to compute parameter percentiles
            percentiles: List of percentiles to compute (default: [0, 1, 5, 25, 50, 75, 95, 99, 100])
            histogram_bins: Number of bins for parameter histograms
            enable_anomaly_detection: Whether to detect anomalies
        """
        self.history_size = history_size
        self.anomaly_threshold = anomaly_threshold
        self.compute_histograms = compute_histograms
        self.compute_percentiles = compute_percentiles
        self.percentiles = percentiles or [0, 1, 5, 25, 50, 75, 95, 99, 100]
        self.histogram_bins = histogram_bins
        self.enable_anomaly_detection = enable_anomaly_detection
        
        # Parameter history
        self.parameter_history: deque = deque(maxlen=history_size)
        self.statistics_history: deque = deque(maxlen=history_size)
        
        # Layer structure tracking
        self.layer_structure: Dict[str, Dict[str, Any]] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance tracking
        self.analysis_count = 0
        self.analysis_times = deque(maxlen=100)
        
        # Observers for real-time updates
        self._observers: Set[Callable[[ParameterDistributionStatistics], None]] = set()
        
        logger.info(
            f"Initialized ParameterDistributionTracker with "
            f"history_size={history_size}, "
            f"anomaly_threshold={anomaly_threshold}, "
            f"compute_histograms={compute_histograms}, "
            f"compute_percentiles={compute_percentiles}"
        )
    
    def analyze_parameters(
        self,
        parameters: Dict[str, mx.array],
        step: Optional[int] = None
    ) -> ParameterDistributionStatistics:
        """
        Perform comprehensive parameter distribution analysis.
        
        Args:
            parameters: Dictionary of parameter arrays
            step: Training step number (optional)
            
        Returns:
            Parameter distribution statistics
            
        Raises:
            PerformanceMetricError: If analysis fails
        """
        start_time = time.perf_counter()
        
        with self._lock:
            try:
                # Store parameters in history
                self.parameter_history.append((parameters, step, time.time()))
                
                # Update layer structure information
                self._update_layer_structure(parameters)
                
                # Compute layer means
                layer_means = self._compute_layer_means(parameters)
                
                # Compute layer variances
                layer_variances = self._compute_layer_variances(parameters)
                
                # Compute layer skewness
                layer_skewness = self._compute_layer_skewness(parameters)
                
                # Compute layer kurtosis
                layer_kurtosis = self._compute_layer_kurtosis(parameters)
                
                # Compute layer percentiles
                layer_percentiles = {}
                if self.compute_percentiles:
                    layer_percentiles = self._compute_layer_percentiles(parameters)
                
                # Compute layer histograms
                layer_histograms = {}
                if self.compute_histograms:
                    layer_histograms = self._compute_layer_histograms(parameters)
                
                # Compute layer ranges
                layer_ranges = self._compute_layer_ranges(parameters)
                
                # Compute layer zero fractions
                layer_zero_fractions = self._compute_layer_zero_fractions(parameters)
                
                # Compute layer sparsity
                layer_sparsity = self._compute_layer_sparsity(parameters)
                
                # Compute layer norms
                layer_norms = self._compute_layer_norms(parameters)
                
                # Compute global statistics
                global_statistics = self._compute_global_statistics(parameters)
                
                # Compute anomaly score
                anomaly_score = 0.0
                if self.enable_anomaly_detection:
                    anomaly_score = self._compute_anomaly_score(
                        layer_means, layer_variances, layer_ranges
                    )
                
                # Create statistics object
                stats = ParameterDistributionStatistics(
                    layer_means=layer_means,
                    layer_variances=layer_variances,
                    layer_skewness=layer_skewness,
                    layer_kurtosis=layer_kurtosis,
                    layer_percentiles=layer_percentiles,
                    layer_histograms=layer_histograms,
                    layer_ranges=layer_ranges,
                    layer_zero_fractions=layer_zero_fractions,
                    layer_sparsity=layer_sparsity,
                    layer_norms=layer_norms,
                    global_statistics=global_statistics,
                    anomaly_score=anomaly_score
                )
                
                # Store statistics in history
                self.statistics_history.append(stats)
                
                # Update performance tracking
                self.analysis_count += 1
                analysis_time = time.perf_counter() - start_time
                self.analysis_times.append(analysis_time)
                
                # Notify observers
                self._notify_observers(stats)
                
                return stats
                
            except Exception as e:
                error_context = {
                    'parameter_count': len(parameters),
                    'analysis_count': self.analysis_count,
                    'step': step
                }
                raise PerformanceMetricError(
                    f"Parameter distribution analysis failed: {e}",
                    context=error_context
                ) from e
    
    def _update_layer_structure(self, parameters: Dict[str, mx.array]) -> None:
        """Update layer structure information."""
        for path, param in parameters.items():
            if path not in self.layer_structure:
                self.layer_structure[path] = {
                    'shape': param.shape,
                    'size': param.size,
                    'dtype': str(param.dtype),
                    'first_seen': self.analysis_count
                }
    
    def _compute_layer_means(self, parameters: Dict[str, mx.array]) -> Dict[str, float]:
        """Compute means for each layer's parameters."""
        layer_means = {}
        
        for path, param in parameters.items():
            try:
                # Compute mean
                mean_value = float(mx.mean(param))
                layer_means[path] = mean_value
                
            except Exception as e:
                logger.warning(f"Failed to compute mean for layer {path}: {e}")
        
        return layer_means
    
    def _compute_layer_variances(self, parameters: Dict[str, mx.array]) -> Dict[str, float]:
        """Compute variances for each layer's parameters."""
        layer_variances = {}
        
        for path, param in parameters.items():
            try:
                # Compute variance
                variance_value = float(mx.var(param))
                layer_variances[path] = variance_value
                
            except Exception as e:
                logger.warning(f"Failed to compute variance for layer {path}: {e}")
        
        return layer_variances
    
    def _compute_layer_skewness(self, parameters: Dict[str, mx.array]) -> Dict[str, float]:
        """Compute skewness for each layer's parameters."""
        layer_skewness = {}
        
        for path, param in parameters.items():
            try:
                # Convert to numpy for skewness calculation
                param_np = param.tolist()
                
                # Compute skewness
                if len(param_np) > 0:
                    mean = statistics.mean(param_np)
                    std = statistics.stdev(param_np) if len(param_np) > 1 else 1.0
                    
                    if std > 0:
                        # Calculate skewness
                        skewness = sum((x - mean) ** 3 for x in param_np) / (len(param_np) * std ** 3)
                        layer_skewness[path] = skewness
                    else:
                        layer_skewness[path] = 0.0
                else:
                    layer_skewness[path] = 0.0
                
            except Exception as e:
                logger.warning(f"Failed to compute skewness for layer {path}: {e}")
                layer_skewness[path] = 0.0
        
        return layer_skewness
    
    def _compute_layer_kurtosis(self, parameters: Dict[str, mx.array]) -> Dict[str, float]:
        """Compute kurtosis for each layer's parameters."""
        layer_kurtosis = {}
        
        for path, param in parameters.items():
            try:
                # Convert to numpy for kurtosis calculation
                param_np = param.tolist()
                
                # Compute kurtosis
                if len(param_np) > 0:
                    mean = statistics.mean(param_np)
                    std = statistics.stdev(param_np) if len(param_np) > 1 else 1.0
                    
                    if std > 0:
                        # Calculate kurtosis
                        kurtosis = sum((x - mean) ** 4 for x in param_np) / (len(param_np) * std ** 4) - 3.0
                        layer_kurtosis[path] = kurtosis
                    else:
                        layer_kurtosis[path] = 0.0
                else:
                    layer_kurtosis[path] = 0.0
                
            except Exception as e:
                logger.warning(f"Failed to compute kurtosis for layer {path}: {e}")
                layer_kurtosis[path] = 0.0
        
        return layer_kurtosis
    
    def _compute_layer_percentiles(self, parameters: Dict[str, mx.array]) -> Dict[str, Dict[str, float]]:
        """Compute percentiles for each layer's parameters."""
        layer_percentiles = {}
        
        for path, param in parameters.items():
            try:
                # Convert to numpy for percentile calculation
                param_np = param.tolist()
                
                # Compute percentiles
                if len(param_np) > 0:
                    percentiles_dict = {}
                    for p in self.percentiles:
                        percentile_value = np.percentile(param_np, p)
                        percentiles_dict[f"p{p}"] = float(percentile_value)
                    
                    layer_percentiles[path] = percentiles_dict
                
            except Exception as e:
                logger.warning(f"Failed to compute percentiles for layer {path}: {e}")
        
        return layer_percentiles
    
    def _compute_layer_histograms(self, parameters: Dict[str, mx.array]) -> Dict[str, Tuple[List[float], List[float]]]:
        """Compute histograms for each layer's parameters."""
        layer_histograms = {}
        
        for path, param in parameters.items():
            try:
                # Convert to numpy for histogram calculation
                param_np = param.tolist()
                
                # Compute histogram
                if len(param_np) > 0:
                    hist, bin_edges = np.histogram(param_np, bins=self.histogram_bins)
                    
                    # Convert to Python lists for serialization
                    hist_list = hist.tolist()
                    bin_edges_list = bin_edges.tolist()
                    
                    layer_histograms[path] = (hist_list, bin_edges_list)
                
            except Exception as e:
                logger.warning(f"Failed to compute histogram for layer {path}: {e}")
        
        return layer_histograms
    
    def _compute_layer_ranges(self, parameters: Dict[str, mx.array]) -> Dict[str, Tuple[float, float]]:
        """Compute ranges (min, max) for each layer's parameters."""
        layer_ranges = {}
        
        for path, param in parameters.items():
            try:
                # Compute min and max
                min_value = float(mx.min(param))
                max_value = float(mx.max(param))
                
                layer_ranges[path] = (min_value, max_value)
                
            except Exception as e:
                logger.warning(f"Failed to compute range for layer {path}: {e}")
        
        return layer_ranges
    
    def _compute_layer_zero_fractions(self, parameters: Dict[str, mx.array]) -> Dict[str, float]:
        """Compute fraction of zero parameters for each layer."""
        layer_zero_fractions = {}
        
        for path, param in parameters.items():
            try:
                # Count zeros
                zero_count = mx.sum(mx.abs(param) < 1e-10)
                total_count = param.size
                
                # Compute fraction
                if total_count > 0:
                    zero_fraction = float(zero_count) / total_count
                else:
                    zero_fraction = 0.0
                
                layer_zero_fractions[path] = zero_fraction
                
            except Exception as e:
                logger.warning(f"Failed to compute zero fraction for layer {path}: {e}")
        
        return layer_zero_fractions
    
    def _compute_layer_sparsity(self, parameters: Dict[str, mx.array]) -> Dict[str, float]:
        """Compute sparsity for each layer's parameters."""
        layer_sparsity = {}
        
        for path, param in parameters.items():
            try:
                # Count non-zeros
                non_zero_count = mx.sum(mx.abs(param) >= 1e-10)
                total_count = param.size
                
                # Compute sparsity (1 - density)
                if total_count > 0:
                    sparsity = 1.0 - float(non_zero_count) / total_count
                else:
                    sparsity = 0.0
                
                layer_sparsity[path] = sparsity
                
            except Exception as e:
                logger.warning(f"Failed to compute sparsity for layer {path}: {e}")
        
        return layer_sparsity
    
    def _compute_layer_norms(self, parameters: Dict[str, mx.array]) -> Dict[str, float]:
        """Compute L2 norms for each layer's parameters."""
        layer_norms = {}
        
        for path, param in parameters.items():
            try:
                # Compute squared norm
                norm_squared = mx.sum(param * param)
                
                # Take square root for final norm
                norm_value = float(mx.sqrt(norm_squared + 1e-8))
                
                layer_norms[path] = norm_value
                
            except Exception as e:
                logger.warning(f"Failed to compute norm for layer {path}: {e}")
        
        return layer_norms
    
    def _compute_global_statistics(self, parameters: Dict[str, mx.array]) -> Dict[str, float]:
        """Compute global statistics across all parameters."""
        global_statistics = {}
        
        try:
            # Flatten all parameters
            all_params = []
            for param in parameters.values():
                all_params.extend(param.reshape(-1).tolist())
            
            if all_params:
                # Compute global statistics
                global_statistics["mean"] = statistics.mean(all_params)
                global_statistics["variance"] = statistics.variance(all_params) if len(all_params) > 1 else 0.0
                global_statistics["min"] = min(all_params)
                global_statistics["max"] = max(all_params)
                global_statistics["median"] = statistics.median(all_params)
                
                # Compute global zero fraction
                zero_count = sum(1 for p in all_params if abs(p) < 1e-10)
                global_statistics["zero_fraction"] = zero_count / len(all_params)
                
                # Compute global sparsity
                global_statistics["sparsity"] = 1.0 - (len(all_params) - zero_count) / len(all_params)
                
                # Compute global norm
                global_norm_squared = sum(p * p for p in all_params)
                global_statistics["norm"] = math.sqrt(global_norm_squared)
        
        except Exception as e:
            logger.warning(f"Failed to compute global statistics: {e}")
        
        return global_statistics
    
    def _compute_anomaly_score(
        self,
        layer_means: Dict[str, float],
        layer_variances: Dict[str, float],
        layer_ranges: Dict[str, Tuple[float, float]]
    ) -> float:
        """
        Compute overall parameter distribution anomaly score.
        
        A higher score indicates more anomalous parameter distributions.
        """
        # Check if we have enough history for anomaly detection
        if len(self.statistics_history) < 5:
            return 0.0
        
        # Get historical values
        historical_means = defaultdict(list)
        historical_variances = defaultdict(list)
        historical_ranges = defaultdict(list)
        
        for stats in self.statistics_history:
            for layer, mean in stats.layer_means.items():
                historical_means[layer].append(mean)
            
            for layer, variance in stats.layer_variances.items():
                historical_variances[layer].append(variance)
            
            for layer, range_tuple in stats.layer_ranges.items():
                historical_ranges[layer].append(range_tuple)
        
        # Compute z-scores for each layer
        mean_z_scores = []
        variance_z_scores = []
        range_z_scores = []
        
        for layer, mean in layer_means.items():
            if layer in historical_means and len(historical_means[layer]) >= 5:
                mean_z_score = self._compute_z_score(mean, historical_means[layer])
                mean_z_scores.append(mean_z_score)
        
        for layer, variance in layer_variances.items():
            if layer in historical_variances and len(historical_variances[layer]) >= 5:
                variance_z_score = self._compute_z_score(variance, historical_variances[layer])
                variance_z_scores.append(variance_z_score)
        
        for layer, range_tuple in layer_ranges.items():
            if layer in historical_ranges and len(historical_ranges[layer]) >= 5:
                # Compute z-score for range (max - min)
                current_range = range_tuple[1] - range_tuple[0]
                historical_range_values = [r[1] - r[0] for r in historical_ranges[layer]]
                range_z_score = self._compute_z_score(current_range, historical_range_values)
                range_z_scores.append(range_z_score)
        
        # Combine z-scores
        all_z_scores = mean_z_scores + variance_z_scores + range_z_scores
        if not all_z_scores:
            return 0.0
        
        max_z_score = max(all_z_scores)
        
        # Threshold for anomaly detection
        return max(0.0, max_z_score - self.anomaly_threshold)
    
    def _compute_z_score(self, value: float, historical_values: List[float]) -> float:
        """Compute z-score of a value relative to historical values."""
        if len(historical_values) < 2:
            return 0.0
        
        mean = statistics.mean(historical_values)
        std = statistics.stdev(historical_values)
        
        if std == 0:
            return 0.0
        
        return abs(value - mean) / std
    
    def add_observer(self, observer: Callable[[ParameterDistributionStatistics], None]) -> None:
        """
        Add an observer for real-time parameter distribution updates.
        
        Args:
            observer: Callback function to receive parameter distribution statistics
        """
        self._observers.add(observer)
    
    def remove_observer(self, observer: Callable[[ParameterDistributionStatistics], None]) -> None:
        """
        Remove an observer.
        
        Args:
            observer: Observer to remove
        """
        self._observers.discard(observer)
    
    def _notify_observers(self, stats: ParameterDistributionStatistics) -> None:
        """Notify registered observers of new parameter distribution statistics."""
        for observer in self._observers.copy():  # Copy to avoid modification during iteration
            try:
                observer(stats)
            except Exception as e:
                logger.warning(f"Observer notification failed: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the tracker."""
        stats = {
            "analysis_count": self.analysis_count,
            "layer_count": len(self.layer_structure),
            "history_size": len(self.parameter_history),
        }
        
        # Analysis performance
        if self.analysis_times:
            times = list(self.analysis_times)
            stats["analysis_performance"] = {
                "avg_time_ms": statistics.mean(times) * 1000,
                "min_time_ms": min(times) * 1000,
                "max_time_ms": max(times) * 1000,
                "p95_time_ms": np.percentile(times, 95) * 1000,
            }
        
        return stats
    
    def get_layer_structure(self) -> Dict[str, Dict[str, Any]]:
        """Get information about the layer structure."""
        return self.layer_structure.copy()
    
    def get_historical_statistics(
        self,
        metric_type: ParameterDistributionMetricType,
        max_points: int = 100
    ) -> Dict[str, List[float]]:
        """
        Get historical values for a specific metric type.
        
        Args:
            metric_type: Type of parameter distribution metric
            max_points: Maximum number of data points to return
            
        Returns:
            Dictionary mapping layer names to lists of historical values
        """
        with self._lock:
            history = list(self.statistics_history)[-max_points:]
            
            if not history:
                return {}
            
            if metric_type == ParameterDistributionMetricType.MEAN:
                # Extract means for each layer
                result = defaultdict(list)
                for stats in history:
                    for layer, mean in stats.layer_means.items():
                        result[layer].append(mean)
                return dict(result)
            
            elif metric_type == ParameterDistributionMetricType.VARIANCE:
                # Extract variances for each layer
                result = defaultdict(list)
                for stats in history:
                    for layer, variance in stats.layer_variances.items():
                        result[layer].append(variance)
                return dict(result)
            
            elif metric_type == ParameterDistributionMetricType.SKEWNESS:
                # Extract skewness for each layer
                result = defaultdict(list)
                for stats in history:
                    for layer, skewness in stats.layer_skewness.items():
                        result[layer].append(skewness)
                return dict(result)
            
            elif metric_type == ParameterDistributionMetricType.KURTOSIS:
                # Extract kurtosis for each layer
                result = defaultdict(list)
                for stats in history:
                    for layer, kurtosis in stats.layer_kurtosis.items():
                        result[layer].append(kurtosis)
                return dict(result)
            
            elif metric_type == ParameterDistributionMetricType.RANGE:
                # Extract ranges for each layer
                result = defaultdict(list)
                for stats in history:
                    for layer, range_tuple in stats.layer_ranges.items():
                        result[layer].append(range_tuple[1] - range_tuple[0])
                return dict(result)
            
            elif metric_type == ParameterDistributionMetricType.ZERO_FRACTION:
                # Extract zero fractions for each layer
                result = defaultdict(list)
                for stats in history:
                    for layer, zero_fraction in stats.layer_zero_fractions.items():
                        result[layer].append(zero_fraction)
                return dict(result)
            
            elif metric_type == ParameterDistributionMetricType.SPARSITY:
                # Extract sparsity for each layer
                result = defaultdict(list)
                for stats in history:
                    for layer, sparsity in stats.layer_sparsity.items():
                        result[layer].append(sparsity)
                return dict(result)
            
            elif metric_type == ParameterDistributionMetricType.NORM:
                # Extract norms for each layer
                result = defaultdict(list)
                for stats in history:
                    for layer, norm in stats.layer_norms.items():
                        result[layer].append(norm)
                return dict(result)
            
            else:
                return {}
    
    def export_statistics(
        self,
        output_path: Path,
        format: str = "json",
        max_points: int = 1000
    ) -> None:
        """
        Export parameter distribution statistics to file.
        
        Args:
            output_path: Path for output file
            format: Export format ("json" or "csv")
            max_points: Maximum number of data points to export
        """
        with self._lock:
            history = list(self.statistics_history)[-max_