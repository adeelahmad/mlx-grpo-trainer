
"""
Advanced Gradient Flow Visualization Metrics for MLX RL Training

This module provides comprehensive gradient flow visualization metrics for MLX-based
reinforcement learning training. It implements sophisticated tracking and analysis
of gradient flow patterns across model layers, enabling deep insights into training
dynamics and potential issues like vanishing or exploding gradients.

Key Features:
- Layer-wise gradient flow tracking with temporal analysis
- Gradient magnitude distribution visualization
- Gradient direction consistency analysis
- Vanishing/exploding gradient detection
- Gradient flow correlation with model performance
- Temporal gradient evolution visualization
- Gradient spectral analysis for frequency components
- Gradient flow anomaly detection

Architecture:
- Strategy Pattern: Multiple gradient analysis algorithms
- Observer Pattern: Real-time gradient flow monitoring
- Repository Pattern: Efficient gradient history storage
- Factory Pattern: Dynamic visualization creation
- Command Pattern: Encapsulated gradient analysis operations
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

from ..progress.gradient_tracker import GradientNormTracker, LayerGradientInfo
from ..progress.exceptions import (
    GradientNormComputationError,
    PerformanceMetricError,
    create_error_context,
)

logger = logging.getLogger(__name__)


class GradientFlowMetricType(Enum):
    """Types of gradient flow metrics that can be collected."""
    LAYER_NORM = "layer_norm"
    LAYER_NORM_RATIO = "layer_norm_ratio"
    LAYER_UPDATE_RATIO = "layer_update_ratio"
    COSINE_SIMILARITY = "cosine_similarity"
    SPECTRAL_NORM = "spectral_norm"
    GRADIENT_SIGNAL_TO_NOISE = "gradient_signal_to_noise"
    GRADIENT_VARIANCE = "gradient_variance"
    GRADIENT_ENTROPY = "gradient_entropy"


@dataclass
class GradientFlowStatistics:
    """
    Comprehensive statistics for gradient flow analysis.
    
    This class encapsulates various statistical measures for analyzing
    gradient flow patterns across model layers and training steps.
    
    Attributes:
        layer_norms: Dictionary mapping layer names to gradient norms
        layer_norm_ratios: Ratios between consecutive layer norms
        max_to_min_norm_ratio: Ratio between max and min gradient norms
        vanishing_gradient_score: Score indicating vanishing gradient severity
        exploding_gradient_score: Score indicating exploding gradient severity
        gradient_variance: Variance of gradient norms across layers
        gradient_entropy: Entropy of gradient distribution
        gradient_signal_to_noise: Signal-to-noise ratio of gradients
        cosine_similarities: Cosine similarities between layer gradients
        spectral_norms: Spectral norms of gradient matrices
        anomaly_score: Overall gradient flow anomaly score
        timestamp: When these statistics were computed
    """
    layer_norms: Dict[str, float]
    layer_norm_ratios: Dict[str, float] = field(default_factory=dict)
    max_to_min_norm_ratio: float = 1.0
    vanishing_gradient_score: float = 0.0
    exploding_gradient_score: float = 0.0
    gradient_variance: float = 0.0
    gradient_entropy: float = 0.0
    gradient_signal_to_noise: float = float('inf')
    cosine_similarities: Dict[str, float] = field(default_factory=dict)
    spectral_norms: Dict[str, float] = field(default_factory=dict)
    anomaly_score: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary for serialization."""
        return {
            'layer_norms': self.layer_norms,
            'layer_norm_ratios': self.layer_norm_ratios,
            'max_to_min_norm_ratio': self.max_to_min_norm_ratio,
            'vanishing_gradient_score': self.vanishing_gradient_score,
            'exploding_gradient_score': self.exploding_gradient_score,
            'gradient_variance': self.gradient_variance,
            'gradient_entropy': self.gradient_entropy,
            'gradient_signal_to_noise': self.gradient_signal_to_noise,
            'cosine_similarities': self.cosine_similarities,
            'spectral_norms': self.spectral_norms,
            'anomaly_score': self.anomaly_score,
            'timestamp': self.timestamp
        }


class GradientFlowAnalyzer:
    """
    Advanced gradient flow analysis system for deep learning models.
    
    This class provides comprehensive analysis of gradient flow patterns
    across model layers, enabling detection of training issues like
    vanishing or exploding gradients, and providing insights into
    training dynamics.
    
    Features:
    - Layer-wise gradient flow tracking with temporal analysis
    - Gradient magnitude distribution visualization
    - Gradient direction consistency analysis
    - Vanishing/exploding gradient detection
    - Gradient flow correlation with model performance
    - Temporal gradient evolution visualization
    - Gradient spectral analysis for frequency components
    - Gradient flow anomaly detection
    """
    
    def __init__(
        self,
        history_size: int = 100,
        vanishing_threshold: float = 0.01,
        exploding_threshold: float = 100.0,
        anomaly_threshold: float = 3.0,
        enable_spectral_analysis: bool = True,
        enable_temporal_analysis: bool = True,
        enable_anomaly_detection: bool = True
    ):
        """
        Initialize gradient flow analyzer.
        
        Args:
            history_size: Number of historical gradient flows to retain
            vanishing_threshold: Threshold for detecting vanishing gradients
            exploding_threshold: Threshold for detecting exploding gradients
            anomaly_threshold: Threshold for anomaly detection (z-score)
            enable_spectral_analysis: Whether to perform spectral analysis
            enable_temporal_analysis: Whether to perform temporal analysis
            enable_anomaly_detection: Whether to detect anomalies
        """
        self.history_size = history_size
        self.vanishing_threshold = vanishing_threshold
        self.exploding_threshold = exploding_threshold
        self.anomaly_threshold = anomaly_threshold
        self.enable_spectral_analysis = enable_spectral_analysis
        self.enable_temporal_analysis = enable_temporal_analysis
        self.enable_anomaly_detection = enable_anomaly_detection
        
        # Gradient history
        self.gradient_history: deque = deque(maxlen=history_size)
        self.statistics_history: deque = deque(maxlen=history_size)
        
        # Layer structure tracking
        self.layer_structure: Dict[str, Dict[str, Any]] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance tracking
        self.analysis_count = 0
        self.analysis_times = deque(maxlen=100)
        
        # Observers for real-time updates
        self._observers: Set[Callable[[GradientFlowStatistics], None]] = set()
        
        logger.info(
            f"Initialized GradientFlowAnalyzer with "
            f"history_size={history_size}, "
            f"vanishing_threshold={vanishing_threshold}, "
            f"exploding_threshold={exploding_threshold}"
        )
    
    def analyze_gradients(
        self,
        gradients: Dict[str, mx.array],
        step: Optional[int] = None
    ) -> GradientFlowStatistics:
        """
        Perform comprehensive gradient flow analysis.
        
        Args:
            gradients: Dictionary of gradient arrays
            step: Training step number (optional)
            
        Returns:
            Gradient flow statistics
            
        Raises:
            GradientNormComputationError: If analysis fails
        """
        start_time = time.perf_counter()
        
        with self._lock:
            try:
                # Store gradients in history
                self.gradient_history.append((gradients, step, time.time()))
                
                # Update layer structure information
                self._update_layer_structure(gradients)
                
                # Compute layer norms
                layer_norms = self._compute_layer_norms(gradients)
                
                # Compute layer norm ratios
                layer_norm_ratios = self._compute_layer_norm_ratios(layer_norms)
                
                # Compute max-to-min norm ratio
                norm_values = list(layer_norms.values())
                max_norm = max(norm_values) if norm_values else 0.0
                min_norm = min(norm_values) if norm_values else 0.0
                max_to_min_ratio = max_norm / max(min_norm, 1e-8)
                
                # Compute vanishing/exploding gradient scores
                vanishing_score = self._compute_vanishing_gradient_score(layer_norms)
                exploding_score = self._compute_exploding_gradient_score(layer_norms)
                
                # Compute gradient variance and entropy
                gradient_variance = statistics.variance(norm_values) if len(norm_values) > 1 else 0.0
                gradient_entropy = self._compute_gradient_entropy(norm_values)
                
                # Compute signal-to-noise ratio
                signal_to_noise = self._compute_signal_to_noise_ratio(gradients)
                
                # Compute cosine similarities between layers
                cosine_similarities = {}
                if len(gradients) > 1:
                    cosine_similarities = self._compute_cosine_similarities(gradients)
                
                # Compute spectral norms
                spectral_norms = {}
                if self.enable_spectral_analysis:
                    spectral_norms = self._compute_spectral_norms(gradients)
                
                # Compute anomaly score
                anomaly_score = 0.0
                if self.enable_anomaly_detection:
                    anomaly_score = self._compute_anomaly_score(
                        layer_norms, max_to_min_ratio, vanishing_score, exploding_score
                    )
                
                # Create statistics object
                stats = GradientFlowStatistics(
                    layer_norms=layer_norms,
                    layer_norm_ratios=layer_norm_ratios,
                    max_to_min_norm_ratio=max_to_min_ratio,
                    vanishing_gradient_score=vanishing_score,
                    exploding_gradient_score=exploding_score,
                    gradient_variance=gradient_variance,
                    gradient_entropy=gradient_entropy,
                    gradient_signal_to_noise=signal_to_noise,
                    cosine_similarities=cosine_similarities,
                    spectral_norms=spectral_norms,
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
                    'gradient_count': len(gradients),
                    'analysis_count': self.analysis_count,
                    'step': step
                }
                raise GradientNormComputationError(
                    f"Gradient flow analysis failed: {e}",
                    gradient_info=error_context
                ) from e
    
    def _update_layer_structure(self, gradients: Dict[str, mx.array]) -> None:
        """Update layer structure information."""
        for path, grad in gradients.items():
            if path not in self.layer_structure:
                self.layer_structure[path] = {
                    'shape': grad.shape,
                    'size': grad.size,
                    'dtype': str(grad.dtype),
                    'first_seen': self.analysis_count
                }
    
    def _compute_layer_norms(self, gradients: Dict[str, mx.array]) -> Dict[str, float]:
        """Compute L2 norms for each layer's gradients."""
        layer_norms = {}
        
        for path, grad in gradients.items():
            try:
                # Compute squared norm
                grad_norm_squared = mx.sum(grad * grad)
                
                # Take square root for final norm
                norm_value = float(mx.sqrt(grad_norm_squared + 1e-8))
                
                layer_norms[path] = norm_value
                
            except Exception as e:
                logger.warning(f"Failed to compute norm for layer {path}: {e}")
        
        return layer_norms
    
    def _compute_layer_norm_ratios(self, layer_norms: Dict[str, float]) -> Dict[str, float]:
        """Compute ratios between consecutive layer norms."""
        layer_paths = sorted(layer_norms.keys())
        layer_norm_ratios = {}
        
        for i in range(1, len(layer_paths)):
            prev_path = layer_paths[i-1]
            curr_path = layer_paths[i]
            
            prev_norm = layer_norms[prev_path]
            curr_norm = layer_norms[curr_path]
            
            if prev_norm > 0:
                ratio = curr_norm / prev_norm
                layer_norm_ratios[f"{prev_path}→{curr_path}"] = ratio
        
        return layer_norm_ratios
    
    def _compute_vanishing_gradient_score(self, layer_norms: Dict[str, float]) -> float:
        """
        Compute a score indicating the severity of vanishing gradients.
        
        A higher score indicates more severe vanishing gradients.
        """
        if not layer_norms:
            return 0.0
        
        norm_values = list(layer_norms.values())
        max_norm = max(norm_values)
        
        # Count layers with very small gradients
        small_gradient_count = sum(1 for norm in norm_values if norm < self.vanishing_threshold * max_norm)
        
        # Compute score as proportion of layers with small gradients
        return small_gradient_count / len(norm_values) if norm_values else 0.0
    
    def _compute_exploding_gradient_score(self, layer_norms: Dict[str, float]) -> float:
        """
        Compute a score indicating the severity of exploding gradients.
        
        A higher score indicates more severe exploding gradients.
        """
        if not layer_norms:
            return 0.0
        
        norm_values = list(layer_norms.values())
        
        # Check for unusually large gradients
        if max(norm_values) > self.exploding_threshold:
            # Count layers with very large gradients
            large_gradient_count = sum(1 for norm in norm_values if norm > self.exploding_threshold)
            
            # Compute score as proportion of layers with large gradients
            return large_gradient_count / len(norm_values)
        
        return 0.0
    
    def _compute_gradient_entropy(self, norm_values: List[float]) -> float:
        """
        Compute entropy of gradient norm distribution.
        
        Higher entropy indicates more uniform distribution of gradient norms.
        """
        if not norm_values:
            return 0.0
        
        # Normalize norms to probabilities
        total_norm = sum(norm_values)
        if total_norm <= 0:
            return 0.0
        
        probabilities = [norm / total_norm for norm in norm_values]
        
        # Compute entropy
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy
    
    def _compute_signal_to_noise_ratio(self, gradients: Dict[str, mx.array]) -> float:
        """
        Compute signal-to-noise ratio of gradients.
        
        Higher values indicate cleaner gradient signal.
        """
        signal = 0.0
        noise = 0.0
        
        for path, grad in gradients.items():
            try:
                # Flatten gradient
                flat_grad = grad.reshape(-1)
                
                # Compute mean (signal) and variance (noise)
                mean = mx.mean(flat_grad)
                var = mx.mean((flat_grad - mean) ** 2)
                
                # Accumulate weighted by parameter count
                signal += abs(float(mean)) * grad.size
                noise += float(var) * grad.size
                
            except Exception as e:
                logger.warning(f"Failed to compute SNR for layer {path}: {e}")
        
        # Compute overall SNR
        return signal / max(noise, 1e-8)
    
    def _compute_cosine_similarities(self, gradients: Dict[str, mx.array]) -> Dict[str, float]:
        """
        Compute cosine similarities between layer gradients.
        
        This measures the directional alignment between gradients.
        """
        similarities = {}
        layer_paths = sorted(gradients.keys())
        
        for i in range(len(layer_paths)):
            for j in range(i+1, len(layer_paths)):
                path_i = layer_paths[i]
                path_j = layer_paths[j]
                
                try:
                    # Get gradients
                    grad_i = gradients[path_i]
                    grad_j = gradients[path_j]
                    
                    # Flatten gradients
                    flat_i = grad_i.reshape(-1)
                    flat_j = grad_j.reshape(-1)
                    
                    # Ensure same length by padding with zeros
                    if flat_i.size < flat_j.size:
                        flat_i = mx.concatenate([flat_i, mx.zeros(flat_j.size - flat_i.size)])
                    elif flat_j.size < flat_i.size:
                        flat_j = mx.concatenate([flat_j, mx.zeros(flat_i.size - flat_j.size)])
                    
                    # Compute cosine similarity
                    dot_product = mx.sum(flat_i * flat_j)
                    norm_i = mx.sqrt(mx.sum(flat_i * flat_i))
                    norm_j = mx.sqrt(mx.sum(flat_j * flat_j))
                    
                    similarity = float(dot_product / (norm_i * norm_j + 1e-8))
                    similarities[f"{path_i}↔{path_j}"] = similarity
                    
                except Exception as e:
                    logger.warning(f"Failed to compute similarity between {path_i} and {path_j}: {e}")
        
        return similarities
    
    def _compute_spectral_norms(self, gradients: Dict[str, mx.array]) -> Dict[str, float]:
        """
        Compute spectral norms of gradient matrices.
        
        The spectral norm is the largest singular value of the matrix.
        """
        spectral_norms = {}
        
        for path, grad in gradients.items():
            try:
                # Only compute for 2D matrices
                if len(grad.shape) == 2:
                    # Use power iteration method to estimate spectral norm
                    u = mx.random.normal(shape=(grad.shape[0], 1))
                    v = mx.random.normal(shape=(grad.shape[1], 1))
                    
                    # Normalize
                    u = u / mx.sqrt(mx.sum(u * u))
                    v = v / mx.sqrt(mx.sum(v * v))
                    
                    # Power iteration (5 iterations)
                    for _ in range(5):
                        v = mx.matmul(grad.T, u)
                        v = v / mx.sqrt(mx.sum(v * v) + 1e-8)
                        
                        u = mx.matmul(grad, v)
                        u = u / mx.sqrt(mx.sum(u * u) + 1e-8)
                    
                    # Compute spectral norm
                    spectral_norm = float(mx.matmul(mx.matmul(u.T, grad), v))
                    spectral_norms[path] = spectral_norm
                    
            except Exception as e:
                logger.warning(f"Failed to compute spectral norm for layer {path}: {e}")
        
        return spectral_norms
    
    def _compute_anomaly_score(
        self,
        layer_norms: Dict[str, float],
        max_to_min_ratio: float,
        vanishing_score: float,
        exploding_score: float
    ) -> float:
        """
        Compute overall gradient flow anomaly score.
        
        A higher score indicates more anomalous gradient flow.
        """
        # Check if we have enough history for anomaly detection
        if len(self.statistics_history) < 5:
            return 0.0
        
        # Get historical values
        historical_ratios = [stats.max_to_min_norm_ratio for stats in self.statistics_history]
        historical_vanishing = [stats.vanishing_gradient_score for stats in self.statistics_history]
        historical_exploding = [stats.exploding_gradient_score for stats in self.statistics_history]
        
        # Compute z-scores
        ratio_z_score = self._compute_z_score(max_to_min_ratio, historical_ratios)
        vanishing_z_score = self._compute_z_score(vanishing_score, historical_vanishing)
        exploding_z_score = self._compute_z_score(exploding_score, historical_exploding)
        
        # Combine z-scores
        combined_z_score = max(ratio_z_score, vanishing_z_score, exploding_z_score)
        
        # Threshold for anomaly detection
        return max(0.0, combined_z_score - self.anomaly_threshold)
    
    def _compute_z_score(self, value: float, historical_values: List[float]) -> float:
        """Compute z-score of a value relative to historical values."""
        if len(historical_values) < 2:
            return 0.0
        
        mean = statistics.mean(historical_values)
        std = statistics.stdev(historical_values)
        
        if std == 0:
            return 0.0
        
        return abs(value - mean) / std
    
    def add_observer(self, observer: Callable[[GradientFlowStatistics], None]) -> None:
        """
        Add an observer for real-time gradient flow updates.
        
        Args:
            observer: Callback function to receive gradient flow statistics
        """
        self._observers.add(observer)
    
    def remove_observer(self, observer: Callable[[GradientFlowStatistics], None]) -> None:
        """
        Remove an observer.
        
        Args:
            observer: Observer to remove
        """
        self._observers.discard(observer)
    
    def _notify_observers(self, stats: GradientFlowStatistics) -> None:
        """Notify registered observers of new gradient flow statistics."""
        for observer in self._observers.copy():  # Copy to avoid modification during iteration
            try:
                observer(stats)
            except Exception as e:
                logger.warning(f"Observer notification failed: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the analyzer."""
        stats = {
            "analysis_count": self.analysis_count,
            "layer_count": len(self.layer_structure),
            "history_size": len(self.gradient_history),
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
        metric_type: GradientFlowMetricType,
        max_points: int = 100
    ) -> Dict[str, List[float]]:
        """
        Get historical values for a specific metric type.
        
        Args:
            metric_type: Type of gradient flow metric
            max_points: Maximum number of data points to return
            
        Returns:
            Dictionary mapping layer names to lists of historical values
        """
        with self._lock:
            history = list(self.statistics_history)[-max_points:]
            
            if not history:
                return {}
            
            if metric_type == GradientFlowMetricType.LAYER_NORM:
                # Extract layer norms for each layer
                result = defaultdict(list)
                for stats in history:
                    for layer, norm in stats.layer_norms.items():
                        result[layer].append(norm)
                return dict(result)
            
            elif metric_type == GradientFlowMetricType.LAYER_NORM_RATIO:
                # Extract layer norm ratios
                result = defaultdict(list)
                for stats in history:
                    for layer_pair, ratio in stats.layer_norm_ratios.items():
                        result[layer_pair].append(ratio)
                return dict(result)
            
            elif metric_type == GradientFlowMetricType.COSINE_SIMILARITY:
                # Extract cosine similarities
                result = defaultdict(list)
                for stats in history:
                    for layer_pair, similarity in stats.cosine_similarities.items():
                        result[layer_pair].append(similarity)
                return dict(result)
            
            elif metric_type == GradientFlowMetricType.SPECTRAL_NORM:
                # Extract spectral norms
                result = defaultdict(list)
                for stats in history:
                    for layer, norm in stats.spectral_norms.items():
                        result[layer].append(norm)
                return dict(result)
            
            elif metric_type == GradientFlowMetricType.GRADIENT_SIGNAL_TO_NOISE:
                # Extract signal-to-noise ratios
                return {"global": [stats.gradient_signal_to_noise for stats in history]}
            
            elif metric_type == GradientFlowMetricType.GRADIENT_VARIANCE:
                # Extract gradient variances
                return {"global": [stats.gradient_variance for stats in history]}
            
            elif metric_type == GradientFlowMetricType.GRADIENT_ENTROPY:
                # Extract gradient entropies
                return {"global": [stats.gradient_entropy for stats in history]}
            
            else:
                return {}
    
    def export_statistics(
        self,
        output_path: Path,
        format: str = "json",
        max_points: int = 1000
    ) -> None:
        """
        Export gradient flow statistics to file.
        
        Args:
            output_path: Path for output file
            format: Export format ("json" or "csv")
            max_points: Maximum number of data points to export
        """
        with self._lock:
            history = list(self.statistics_history)[-max_points:]
            
            if not history:
                logger.warning("No gradient flow statistics to export")
                return
            
            # Create export data
            export_data = {
                "metadata": {
                    "export_timestamp": time.time(),
                    "analysis_count": self.analysis_count,
                    "layer_structure": self.layer_structure,
                },
                "statistics": [stats.to_dict() for stats in history]
            }
            
            # Write to file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == "json":
                with open(output_path, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            
            elif format.lower() == "csv":
                # Export as CSV (simplified format)
                import csv
                with open(output_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    
                    # Write header
                    writer.writerow([
                        "timestamp", "max_to_min_ratio", "vanishing_score",
                        "exploding_score", "gradient_variance", "gradient_entropy",
                        "signal_to_noise", "anomaly_score"
                    ])
                    
                    # Write data
                    for stats in history:
                        writer.writerow([
                            stats.timestamp,
                            stats.max_to_min_norm_ratio,
                            stats.vanishing_gradient_score,
                            stats.exploding_gradient_score,
                            stats.gradient_variance,
                            stats.gradient_entropy,
                            stats.gradient_signal_to_noise,
                            stats.anomaly_score
                        ])
            
            logger.info(f"Exported gradient flow statistics to {output_path}")
    
    def reset(self) -> None:
        """Reset analyzer state."""
        with self._lock:
            self.gradient_history.clear()
            self.statistics_history.clear()
            self.layer_structure.clear()
            self.analysis_count = 0
            self.analysis_times.clear()
            
            logger.info("Reset gradient flow analyzer")


class GradientFlowVisualizer:
    """
    Visualization system for gradient flow metrics.
    
    This class provides various visualization methods for gradient flow
    metrics, enabling deep insights into training dynamics and potential
    issues like vanishing or exploding gradients.
    
    Features:
    - Layer-wise gradient flow visualization
    - Gradient magnitude distribution plots
    - Gradient direction consistency visualization
    - Temporal gradient evolution plots
    - Gradient flow anomaly visualization
    """
    
    def __init__(
        self,
        output_dir: Path,
        analyzer: GradientFlowAnalyzer,
        dpi: int = 150,
        figsize: Tuple[int, int] = (12, 8)
    ):
        """
        Initialize gradient flow visualizer.
        
        Args:
            output_dir: Directory to save visualizations
            analyzer: Gradient flow analyzer to use
            dpi: Image resolution
            figsize: Default figure size
        """
        self.output_dir = output_dir
        self.analyzer = analyzer
        self.dpi = dpi
        self.figsize = figsize
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Import visualization libraries
        try:
            import matplotlib
            matplotlib.use("Agg")  # Non-interactive backend
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            self.plt = plt
            self.sns = sns
            self.visualization_available = True
            
        except ImportError:
            logger.warning("Matplotlib or seaborn not available, visualization disabled")
            self.visualization_available = False
        
        logger.info(f"Initialized GradientFlowVisualizer: {self.output_dir}")
    
    def visualize_layer_norms(
        self,
        filename: str = "gradient_layer_norms.png",
        max_layers: int = 20,
        log_scale: bool = True
    ) -> Optional[Path]:
        """
        Visualize gradient norms across layers.
        
        Args:
            filename: Output filename
            max_layers: Maximum number of layers to display
            log_scale: Whether to use logarithmic scale
            
        Returns:
            Path to the generated visualization, or None if visualization failed
        """
        if not self.visualization_available:
            logger.warning("Visualization libraries not available")
            return None
        
        with self.analyzer._lock:
            # Get latest statistics
            if not self.analyzer.statistics_history:
                logger.warning("No gradient flow statistics available")
                return None
            
            latest_stats = self.analyzer.statistics_history[-1]
            
            # Sort layers by norm
            sorted_layers = sorted(
                latest_stats.layer_norms.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Limit number of layers
            if len(sorted_layers) > max_layers:
                sorted_layers = sorted_layers[:max_layers]
            
            # Create figure
            fig, ax = self.plt.subplots(figsize=self.figsize)
            
            # Plot layer norms
            layer_names = [name for name, _ in sorted_layers]
            layer_norms = [norm for _, norm in sorted_layers]
            
            # Create horizontal bar chart
            bars = ax.barh(range(len(layer_names)), layer_norms)
            
            # Add labels
            ax.set_yticks(range(len(layer_names)))
            ax.set_yticklabels([name.split('.')[-1] for name in layer_names])
            
            # Set log scale if requested
            if log_scale and min(layer_norms) > 0:
                ax.set_xscale('log')
            
            # Add title and labels
            ax.set_title('Gradient Norms Across Layers')
            ax.set_xlabel('Gradient Norm (L2)')
            ax.set_ylabel('Layer')
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Add colorbar
            sm = self.plt.cm.ScalarMappable(
                cmap=self.plt.cm.viridis,
                norm=self.plt.Normalize(vmin=min(layer_norms), vmax=max(layer_norms))
            )
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label('Gradient Norm')
            
            # Add annotations
            for i, (norm, bar) in enumerate(zip(layer_norms, bars)):
                ax.text(
                    norm * 1.05,
                    i,
                    f"{norm:.2e}",
                    va='center'
                )
            
            # Save figure
            output_path = self.output_dir / filename
            fig.tight_layout()
            fig.savefig(output_path, dpi=self.dpi)
            self.plt.close(fig)
            
            return output_path
            
    def visualize_gradient_flow_over_time(
        self,
        filename: str = "gradient_flow_over_time.png",
        max_layers: int = 10,
        max_steps: int = 100
    ) -> Optional[Path]:
        """
        Visualize gradient flow evolution over training steps.
        
        Args:
            filename: Output filename
            max_layers: Maximum number of layers to display
            max_steps: Maximum number of steps to display
            
        Returns:
            Path to the generated visualization, or None if visualization failed
        """
        if not self.visualization_available:
            logger.warning("Visualization libraries not available")
            return None
        
        with self.analyzer._lock:
            # Get historical data
            history = list(self.analyzer.statistics_history)[-max_steps:]
            
            if not history:
                logger.warning("No gradient flow history available")
                return None
            
            # Get top layers by average norm
            all_layers = set()
            for stats in history:
                all_layers.update(stats.layer_norms.keys())
            
            layer_avg_norms = {}
            for layer in all_layers:
                norms = [stats.layer_norms.get(layer, 0.0) for stats in history if layer in stats.layer_norms]
                if norms:
                    layer_avg_norms[layer] = sum(norms) / len(norms)
            
            # Select top layers
            top_layers = sorted(
                layer_avg_norms.items(),
                key=lambda x: x[1],
                reverse=True
            )[:max_layers]
            
            top_layer_names = [layer for layer, _ in top_layers]
            
            # Create figure
            fig, ax = self.plt.subplots(figsize=self.figsize)
            
            # Plot gradient norms over time for each layer
            for layer in top_layer_names:
                steps = []
                norms = []
                
                for i, stats in enumerate(history):
                    if layer in stats.layer_norms:
                        steps.append(i)
                        norms.append(stats.layer_norms[layer])
                
                if steps and norms:
                    ax.plot(steps, norms, label=layer.split('.')[-1], marker='o', markersize=3)
            
            # Add title and labels
            ax.set_title('Gradient Flow Evolution Over Time')
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Gradient Norm (L2)')
            
            # Add grid and legend
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
            
            # Save figure
            output_path = self.output_dir / filename
            fig.tight_layout()
            fig.savefig(output_path, dpi=self.dpi)
            self.plt.close(fig)
            
            return output_path
            
    def visualize_vanishing_exploding_gradients(
        self,
        filename: str = "vanishing_exploding_gradients.png",
        max_steps: int = 100
    ) -> Optional[Path]:
        """
        Visualize vanishing and exploding gradient scores over time.
        
        Args:
            filename: Output filename
            max_steps: Maximum number of steps to display
            
        Returns:
            Path to the generated visualization, or None if visualization failed
        """
        if not self.visualization_available:
            logger.warning("Visualization libraries not available")
            return None
        
        with self.analyzer._lock:
            # Get historical data
            history = list(self.analyzer.statistics_history)[-max_steps:]
            
            if not history:
                logger.warning("No gradient flow history available")
                return None
            
            # Extract scores
            steps = list(range(len(history)))
            vanishing_scores = [stats.vanishing_gradient_score for stats in history]
            exploding_scores = [stats.exploding_gradient_score for stats in history]
            max_to_min_ratios = [stats.max_to_min_norm_ratio for stats in history]
            
            # Create figure with two subplots
            fig, (ax1, ax2) = self.plt.subplots(2, 1, figsize=self.figsize, sharex=True)
            
            # Plot vanishing and exploding gradient scores
            ax1.plot(steps, vanishing_scores, label='Vanishing', color='blue', marker='o', markersize=3)
            ax1.plot(steps, exploding_scores, label='Exploding', color='red', marker='x', markersize=3)
            
            # Plot max-to-min ratio
            ax2.plot(steps, max_to_min_ratios, label='Max/Min Ratio', color='purple')
            ax2.set_yscale('log')
            
            # Add titles and labels
            ax1.set_title('Gradient Health Metrics Over Time')
            ax1.set_ylabel('Gradient Score')
            ax2.set_xlabel('Training Step')
            ax2.set_ylabel('Max/Min Ratio (log scale)')
            
            # Add grid and legend
            ax1.grid(True, alpha=0.3)
            ax2.grid(True, alpha=0.3)
            ax1.legend()
            ax2.legend()
            
            # Add threshold lines
            ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Warning Threshold')
            
            # Add annotations for extreme values
            max_vanishing_idx = vanishing_scores.index(max(vanishing_scores))
            max_exploding_idx = exploding_scores.index(max(exploding_scores))
            
            if vanishing_scores[max_vanishing_idx] > 0.3:
                ax1.annotate(
                    f"Peak: {vanishing_scores[max_vanishing_idx]:.2f}",
                    xy=(max_vanishing_idx, vanishing_scores[max_vanishing_idx]),
                    xytext=(max_vanishing_idx, vanishing_scores[max_vanishing_idx] + 0.1),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                    ha='center'
                )
            
            if exploding_scores[max_exploding_idx] > 0.3:
                ax1.annotate(
                    f"Peak: {exploding_scores[max_exploding_idx]:.2f}",
                    xy=(max_exploding_idx, exploding_scores[max_exploding_idx]),
                    xytext=(max_exploding_idx, exploding_scores[max_exploding_idx] + 0.1),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                    ha='center'
                )
            
            # Save figure
            output_path = self.output_dir / filename
            fig.tight_layout()
            fig.savefig(output_path, dpi=self.dpi)
            self.plt.close(fig)
            
            return output_path
            
    def visualize_gradient_statistics(
        self,
        filename: str = "gradient_statistics.png",
        max_steps: int = 100
    ) -> Optional[Path]:
        """
        Visualize gradient statistics over time.
        
        Args:
            filename: Output filename
            max_steps: Maximum number of steps to display
            
        Returns:
            Path to the generated visualization, or None if visualization failed
        """
        if not self.visualization_available:
            logger.warning("Visualization libraries not available")
            return None
        
        with self.analyzer._lock:
            # Get historical data
            history = list(self.analyzer.statistics_history)[-max_steps:]
            
            if not history:
                logger.warning("No gradient flow history available")
                return None
            
            # Extract statistics
            steps = list(range(len(history)))
            variances = [stats.gradient_variance for stats in history]
            entropies = [stats.gradient_entropy for stats in history]
            snrs = [min(stats.gradient_signal_to_noise, 10.0) for stats in history]  # Cap SNR for better visualization
            anomaly_scores = [stats.anomaly_score for stats in history]
            
            # Create figure with subplots
            fig, axs = self.plt.subplots(2, 2, figsize=self.figsize, sharex=True)
            
            # Plot variance
            axs[0, 0].plot(steps, variances, color='blue')
            axs[0, 0].set_title('Gradient Variance')
            axs[0, 0].set_ylabel('Variance')
            axs[0, 0].grid(True, alpha=0.3)
            
            # Plot entropy
            axs[0, 1].plot(steps, entropies, color='green')
            axs[0, 1].set_title('Gradient Entropy')
            axs[0, 1].set_ylabel('Entropy')
            axs[0, 1].grid(True, alpha=0.3)
            
            # Plot signal-to-noise ratio
            axs[1, 0].plot(steps, snrs, color='purple')
            axs[1, 0].set_title('Signal-to-Noise Ratio')
            axs[1, 0].set_ylabel('SNR')
            axs[1, 0].set_xlabel('Training Step')
            axs[1, 0].grid(True, alpha=0.3)
            
            # Plot anomaly score
            axs[1, 1].plot(steps, anomaly_scores, color='red')
            axs[1, 1].set_title('Anomaly Score')
            axs[1, 1].set_ylabel('Score')
            axs[1, 1].set_xlabel('Training Step')
            axs[1, 1].grid(True, alpha=0.3)
            
            # Add threshold line for anomaly score
            axs[1, 1].axhline(y=self.analyzer.anomaly_threshold, color='r', linestyle='--', alpha=0.5)
            
            # Save figure
            output_path = self.output_dir / filename
            fig.tight_layout()
            fig.savefig(output_path, dpi=self.dpi)
            self.plt.close(fig)
            
            return output_path
            
    def visualize_layer_correlations(
        self,
        filename: str = "layer_correlations.png",
        max_layers: int = 15
    ) -> Optional[Path]:
        """
        Visualize correlations between layer gradients.
        
        Args:
            filename: Output filename
            max_layers: Maximum number of layers to display
            
        Returns:
            Path to the generated visualization, or None if visualization failed
        """
        if not self.visualization_available:
            logger.warning("Visualization libraries not available")
            return None
        
        with self.analyzer._lock:
            # Get latest statistics
            if not self.analyzer.statistics_history:
                logger.warning("No gradient flow statistics available")
                return None
            
            latest_stats = self.analyzer.statistics_history[-1]
            
            # Check if we have cosine similarities
            if not latest_stats.cosine_similarities:
                logger.warning("No cosine similarity data available")
                return None
            
            # Extract layer names from cosine similarities
            layer_names = set()
            for pair in latest_stats.cosine_similarities.keys():
                layers = pair.split('↔')
                layer_names.update(layers)
            
            # Limit number of layers
            if len(layer_names) > max_layers:
                # Sort layers by norm
                sorted_layers = sorted(
                    [(layer, latest_stats.layer_norms.get(layer, 0.0)) for layer in layer_names],
                    key=lambda x: x[1],
                    reverse=True
                )
                layer_names = {layer for layer, _ in sorted_layers[:max_layers]}
            
            # Create correlation matrix
            layer_list = sorted(layer_names)
            n_layers = len(layer_list)
            corr_matrix = np.eye(n_layers)  # Start with identity matrix (self-correlation = 1)
            
            # Fill correlation matrix
            for i in range(n_layers):
                for j in range(i+1, n_layers):
                    layer_i = layer_list[i]
                    layer_j = layer_list[j]
                    
                    # Check both possible orderings of the pair
                    pair_key = f"{layer_i}↔{layer_j}"
                    reverse_pair_key = f"{layer_j}↔{layer_i}"
                    
                    if pair_key in latest_stats.cosine_similarities:
                        similarity = latest_stats.cosine_similarities[pair_key]
                    elif reverse_pair_key in latest_stats.cosine_similarities:
                        similarity = latest_stats.cosine_similarities[reverse_pair_key]
                    else:
                        similarity = 0.0
                    
                    # Set both entries (matrix is symmetric)
                    corr_matrix[i, j] = similarity
                    corr_matrix[j, i] = similarity
            
            # Create figure
            fig, ax = self.plt.subplots(figsize=self.figsize)
            
            # Plot correlation matrix as heatmap
            im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Cosine Similarity')
            
            # Add labels
            ax.set_title('Layer Gradient Correlations')
            ax.set_xticks(range(n_layers))
            ax.set_yticks(range(n_layers))
            
            # Use shortened layer names for readability
            shortened_names = [name.split('.')[-1] for name in layer_list]
            ax.set_xticklabels(shortened_names, rotation=45, ha='right')
            ax.set_yticklabels(shortened_names)
            
            # Add grid
            ax.grid(False)
            
            # Add annotations
            for i in range(n_layers):
                for j in range(n_layers):
                    if i != j:  # Skip diagonal (self-correlations)
                        text_color = 'white' if abs(corr_matrix[i, j]) > 0.5 else 'black'
                        ax.text(
                            j, i, f"{corr_matrix[i, j]:.2f}",
                            ha='center', va='center', color=text_color,
                            fontsize=8
                        )
            
            # Save figure
            output_path = self.output_dir / filename
            fig.tight_layout()
            fig.savefig(output_path, dpi=self.dpi)
            self.plt.close(fig)
            
            return output_path
    
    def create_gradient_flow_dashboard(
        self,
        filename: str = "gradient_flow_dashboard.png"
    ) -> Optional[Path]:
        """
        Create a comprehensive dashboard of gradient flow visualizations.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to the generated dashboard, or None if visualization failed
        """
        if not self.visualization_available:
            logger.warning("Visualization libraries not available")
            return None
        
        with self.analyzer._lock:
            # Check if we have data
            if not self.analyzer.statistics_history:
                logger.warning("No gradient flow statistics available")
                return None
            
            # Create figure with subplots
            fig = self.plt.figure(figsize=(20, 15))
            gs = self.plt.GridSpec(3, 3, figure=fig)
            
            # 1. Layer norms (top left)
            ax1 = fig.add_subplot(gs[0, 0])
            self._plot_layer_norms(ax1, max_layers=10)
            
            # 2. Gradient flow over time (top center and right)
            ax2 = fig.add_subplot(gs[0, 1:])
            self._plot_gradient_flow_over_time(ax2, max_layers=5, max_steps=50)
            
            # 3. Vanishing/exploding gradients (middle left)
            ax3 = fig.add_subplot(gs[1, 0])
            self._plot_vanishing_exploding_scores(ax3, max_steps=50)
            
            # 4. Layer correlations (middle center and right)
            ax4 = fig.add_subplot(gs[1, 1:])
            self._plot_layer_correlations(ax4, max_layers=10)
            
            # 5. Gradient statistics (bottom row)
            ax5 = fig.add_subplot(gs[2, 0])
            self._plot_gradient_variance_entropy(ax5, max_steps=50)
            
            ax6 = fig.add_subplot(gs[2, 1])
            self._plot_signal_to_noise(ax6, max_steps=50)
            
            ax7 = fig.add_subplot(gs[2, 2])
            self._plot_anomaly_score(ax7, max_steps=50)
            
            # Add title
            fig.suptitle('Gradient Flow Analysis Dashboard', fontsize=16, fontweight='bold')
            
            # Save figure
            output_path = self.output_dir / filename
            fig.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for suptitle
            fig.savefig(output_path, dpi=self.dpi)
            self.plt.close(fig)
            
            return output_path
    
    def _plot_layer_norms(self, ax, max_layers: int = 10) -> None:
        """Plot layer norms on the given axes."""
        latest_stats = self.analyzer.statistics_history[-1]
        
        # Sort layers by norm
        sorted_layers = sorted(
            latest_stats.layer_norms.items(),
            key=lambda x: x[1],
            reverse=True
        )[:max_layers]
        
        # Plot layer norms
        layer_names = [name.split('.')[-1] for name, _ in sorted_layers]
        layer_norms = [norm for _, norm in sorted_layers]
        
        # Create horizontal bar chart
        bars = ax.barh(range(len(layer_names)), layer_norms)
        
        # Add labels
        ax.set_yticks(range(len(layer_names)))
        ax.set_yticklabels(layer_names)
        
        # Add title and labels
        ax.set_title('Layer Gradient Norms')
        ax.set_xlabel('Norm (L2)')
        
        # Add grid
        ax.grid(True, alpha=0.3)
    
    def _plot_gradient_flow_over_time(self, ax, max_layers: int = 5, max_steps: int = 50) -> None:
        """Plot gradient flow over time on the given axes."""
        history = list(self.analyzer.statistics_history)[-max_steps:]
        
        # Get top layers by average norm
        all_layers = set()
        for stats in history:
            all_layers.update(stats.layer_norms.keys())
        
        layer_avg_norms = {}
        for layer in all_layers:
            norms = [stats.layer_norms.get(layer, 0.0) for stats in history if layer in stats.layer_norms]
            if norms:
                layer_avg_norms[layer] = sum(norms) / len(norms)
        
        # Select top layers
        top_layers = sorted(
            layer_avg_norms.items(),
            key=lambda x: x[1],
            reverse=True
        )[:max_layers]
        
        top_layer_names = [layer for layer, _ in top_layers]
        
        # Plot gradient norms over time for each layer
        for layer in top_layer_names:
            steps = []
            norms = []
            
            for i, stats in enumerate(history):
                if layer in stats.layer_norms:
                    steps.append(i)
                    norms.append(stats.layer_norms[layer])
            
            if steps and norms:
                ax.plot(steps, norms, label=layer.split('.')[-1], marker='o', markersize=3)
        
        # Add title and labels
        ax.set_title('Gradient Flow Evolution')
        ax.set_xlabel('Step')
        ax.set_ylabel('Gradient Norm')
        
        # Add grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
    
    def _plot_vanishing_exploding_scores(self, ax, max_steps: int = 50) -> None:
        """Plot vanishing and exploding gradient scores on the given axes."""
        history = list(self.analyzer.statistics_history)[-max_steps:]
        
        # Extract scores
        steps = list(range(len(history)))
        vanishing_scores = [stats.vanishing_gradient_score for stats in history]
        exploding_scores = [stats.exploding_gradient_score for stats in history]
        
        # Plot scores
        ax.plot(steps, vanishing_scores, label='Vanishing', color='blue')
        ax.plot(steps, exploding_scores, label='Exploding', color='red')
        
        # Add title and labels
        ax.set_title('Gradient Health Metrics')
        ax.set_xlabel('Step')
        ax.set_ylabel('Score')
        
        # Add grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add threshold line
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    
    def _plot_layer_correlations(self, ax, max_layers: int = 10) -> None:
        """Plot layer correlations on the given axes."""
        latest_stats = self.analyzer.statistics_history[-1]
        
        # Check if we have cosine similarities
        if not latest_stats.cosine_similarities:
            ax.text(0.5, 0.5, "No correlation data available", ha='center', va='center')
            return
        
        # Extract layer names from cosine similarities
        layer_names = set()
        for pair in latest_stats.cosine_similarities.keys():
            layers = pair.split('↔')
            layer_names.update(layers)
        
        # Limit number of layers
        if len(layer_names) > max_layers:
            # Sort layers by norm
            sorted_layers = sorted(
                [(layer, latest_stats.layer_norms.get(layer, 0.0)) for layer in layer_names],
                key=lambda x: x[1],
                reverse=True
            )
            layer_names = {layer for layer, _ in sorted_layers[:max_layers]}
        
        # Create correlation matrix
        layer_list = sorted(layer_names)
        n_layers = len(layer_list)
        corr_matrix = np.eye(n_layers)
        
        # Fill correlation matrix
        for i in range(n_layers):
            for j in range(i+1, n_layers):
                layer_i = layer_list[i]
                layer_j = layer_list[j]
                
                pair_key = f"{layer_i}↔{layer_j}"
                reverse_pair_key = f"{layer_j}↔{layer_i}"
                
                if pair_key in latest_stats.cosine_similarities:
                    similarity = latest_stats.cosine_similarities[pair_key]
                elif reverse_pair_key in latest_stats.cosine_similarities:
                    similarity = latest_stats.cosine_similarities[reverse_pair_key]
                else:
                    similarity = 0.0
                
                corr_matrix[i, j] = similarity
                corr_matrix[j, i] = similarity
        
        # Plot correlation matrix as heatmap
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.set_label('Cosine Similarity')
        
        # Add labels
        ax.set_title('Layer Gradient Correlations')
        ax.set_xticks(range(n_layers))
        ax.set_yticks(range(n_layers))
        
        # Use shortened layer names for readability
        shortened_names = [name.split('.')[-1] for name in layer_list]
        ax.set_xticklabels(shortened_names, rotation=45, ha='right')
        ax.set_yticklabels(shortened_names)
    
    def _plot_gradient_variance_entropy(self, ax, max_steps: int = 50) -> None:
        """Plot gradient variance and entropy on the given axes."""
        history = list(self.analyzer.statistics_history)[-max_steps:]
        
        # Extract statistics
        steps = list(range(len(history)))
        variances = [stats.gradient_variance for stats in history]
        entropies = [stats.gradient_entropy for stats in history]
        
        # Plot statistics
        ax.plot(steps, variances, label='Variance', color='blue')
        ax.plot(steps, entropies, label='Entropy', color='green')
        
        # Add title and labels
        ax.set_title('Gradient Distribution Metrics')
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        
        # Add grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_signal_to_noise(self, ax, max_steps: int = 50) -> None:
        """Plot signal-to-noise ratio on the given axes."""
        history = list(self.analyzer.statistics_history)[-max_steps:]
        
        # Extract statistics
        steps = list(range(len(history)))
        snrs = [min(stats.gradient_signal_to_noise, 10.0) for stats in history]  # Cap SNR for better visualization
        
        # Plot SNR
        ax.plot(steps, snrs, color='purple')
        
        # Add title and labels
        ax.set_title('Signal-to-Noise Ratio')
        ax.set_xlabel('Step')
        ax.set_ylabel('SNR')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add reference line for SNR = 1
        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    
    def _plot_anomaly_score(self, ax, max_steps: int = 50) -> None:
        """Plot anomaly score on the given axes."""
        history = list(self.analyzer.statistics_history)[-max_steps:]
        
        # Extract statistics
        steps = list(range(len(history)))
        anomaly_scores = [stats.anomaly_score for stats in history]
        
        # Plot anomaly score
        ax.plot(steps, anomaly_scores, color='red')
        
        # Add title and labels
        ax.set_title('Gradient Anomaly Score')
        ax.set_xlabel('Step')
        ax.set_ylabel('Score')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add threshold line
        ax.axhline(y=self.analyzer.anomaly_threshold, color='r', linestyle='--', alpha=0.5,
                  label=f'Threshold ({self.analyzer.anomaly_threshold})')
        ax.legend()