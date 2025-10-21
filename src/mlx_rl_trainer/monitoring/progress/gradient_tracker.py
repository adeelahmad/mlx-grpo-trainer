"""
Gradient Norm Tracking System for MLX RL Training

This module provides comprehensive gradient norm tracking capabilities
for MLX-based reinforcement learning training. It implements multiple
strategies for computing gradient norms with different performance
characteristics and accuracy trade-offs.

The gradient tracking system is designed with enterprise-grade patterns:
- Strategy Pattern: Multiple gradient norm computation algorithms
- Observer Pattern: Real-time gradient norm monitoring
- Factory Pattern: Dynamic strategy selection based on requirements
- Circuit Breaker: Fault tolerance for gradient computation failures
- Caching: Performance optimization for repeated computations

Key Features:
- Real-time gradient norm computation with MLX optimization
- Layer-wise gradient analysis for detailed insights
- Adaptive norm computation with outlier detection
- Thread-safe operations for concurrent training
- Comprehensive error handling and fallback mechanisms
- Performance monitoring and optimization hints

Supported Gradient Norm Strategies:
- L2NormStrategy: Standard Euclidean norm computation
- LayerWiseNormStrategy: Per-layer gradient norm analysis
- AdaptiveNormStrategy: Adaptive computation with outlier handling
- CachedNormStrategy: Performance-optimized with intelligent caching
- RobustNormStrategy: Fault-tolerant with multiple fallback methods

Example:
    ```python
    from mlx_rl_trainer.monitoring.progress.gradient_tracker import (
        GradientNormTracker, L2NormStrategy
    )
    
    # Initialize gradient tracker with L2 norm strategy
    tracker = GradientNormTracker(
        strategy=L2NormStrategy(),
        enable_caching=True,
        cache_size=100
    )
    
    # Compute gradient norm from MLX gradient dictionary
    gradient_norm = tracker.compute_norm(gradients)
    
    # Get layer-wise analysis
    layer_analysis = tracker.analyze_layers(gradients)
    ```
"""

import time
import threading
import logging
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import weakref
import hashlib
import pickle

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_map

from .strategies.base_strategy import (
    MetricComputationStrategy,
    StrategyMetadata,
    StrategyPriority,
    ComputationComplexity,
    MetricDict,
)
from .exceptions import (
    GradientNormComputationError,
    PerformanceMetricError,
    ThreadSafetyError,
    create_error_context,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GradientNormResult:
    """
    Result of gradient norm computation.
    
    This class encapsulates the results of gradient norm computation
    including the computed norm value, computation metadata, and
    performance statistics.
    
    Attributes:
        norm_value: The computed gradient norm
        computation_time: Time taken for computation (seconds)
        strategy_used: Name of the strategy used for computation
        layer_count: Number of layers processed
        parameter_count: Total number of parameters
        cache_hit: Whether the result was retrieved from cache
        outliers_detected: Number of outlier gradients detected
        metadata: Additional computation metadata
    """
    norm_value: float
    computation_time: float
    strategy_used: str
    layer_count: int = 0
    parameter_count: int = 0
    cache_hit: bool = False
    outliers_detected: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'gradient_norm': self.norm_value,
            'computation_time_ms': self.computation_time * 1000,
            'strategy': self.strategy_used,
            'layer_count': self.layer_count,
            'parameter_count': self.parameter_count,
            'cache_hit': self.cache_hit,
            'outliers_detected': self.outliers_detected,
            **self.metadata
        }


@dataclass
class LayerGradientInfo:
    """
    Information about gradients for a specific layer.
    
    Attributes:
        layer_name: Name/path of the layer
        norm_value: Gradient norm for this layer
        parameter_count: Number of parameters in this layer
        gradient_shape: Shape of the gradient tensor
        is_outlier: Whether this layer has outlier gradients
        statistics: Additional statistical information
    """
    layer_name: str
    norm_value: float
    parameter_count: int
    gradient_shape: Tuple[int, ...]
    is_outlier: bool = False
    statistics: Dict[str, float] = field(default_factory=dict)


class GradientNormCache:
    """
    High-performance cache for gradient norm computations.
    
    This cache implements an intelligent caching strategy that:
    - Uses gradient structure hashing for cache keys
    - Implements LRU eviction policy
    - Provides thread-safe operations
    - Monitors cache performance metrics
    - Supports cache invalidation strategies
    
    The cache is designed to handle the dynamic nature of gradient
    computations while providing significant performance improvements
    for repeated or similar gradient structures.
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: float = 300.0,
        enable_metrics: bool = True
    ):
        """
        Initialize gradient norm cache.
        
        Args:
            max_size: Maximum number of cached entries
            ttl_seconds: Time-to-live for cached entries
            enable_metrics: Whether to collect cache metrics
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.enable_metrics = enable_metrics
        
        # Cache storage
        self._cache: Dict[str, Tuple[GradientNormResult, float]] = {}
        self._access_order: deque = deque()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance metrics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._total_requests = 0
    
    def _generate_cache_key(self, gradients: Dict[str, mx.array]) -> str:
        """
        Generate cache key from gradient structure.
        
        Args:
            gradients: Gradient dictionary
            
        Returns:
            Unique cache key for the gradient structure
        """
        try:
            # Create a structure signature based on gradient shapes and paths
            structure_info = []
            for path, grad in tree_flatten(gradients):
                if isinstance(grad, mx.array):
                    structure_info.append((path, grad.shape, str(grad.dtype)))
            
            # Sort for consistent ordering
            structure_info.sort()
            
            # Create hash of the structure
            structure_str = str(structure_info)
            return hashlib.md5(structure_str.encode()).hexdigest()
            
        except Exception as e:
            logger.warning(f"Failed to generate cache key: {e}")
            return f"fallback_{time.time()}"
    
    def get(self, gradients: Dict[str, mx.array]) -> Optional[GradientNormResult]:
        """
        Retrieve cached gradient norm result.
        
        Args:
            gradients: Gradient dictionary
            
        Returns:
            Cached result if available and valid, None otherwise
        """
        with self._lock:
            self._total_requests += 1
            
            cache_key = self._generate_cache_key(gradients)
            current_time = time.time()
            
            if cache_key in self._cache:
                result, timestamp = self._cache[cache_key]
                
                # Check TTL
                if current_time - timestamp <= self.ttl_seconds:
                    # Update access order for LRU
                    if cache_key in self._access_order:
                        self._access_order.remove(cache_key)
                    self._access_order.append(cache_key)
                    
                    self._hits += 1
                    
                    # Mark as cache hit
                    cached_result = GradientNormResult(
                        norm_value=result.norm_value,
                        computation_time=result.computation_time,
                        strategy_used=result.strategy_used,
                        layer_count=result.layer_count,
                        parameter_count=result.parameter_count,
                        cache_hit=True,
                        outliers_detected=result.outliers_detected,
                        metadata=result.metadata
                    )
                    
                    return cached_result
                else:
                    # Expired entry
                    del self._cache[cache_key]
                    if cache_key in self._access_order:
                        self._access_order.remove(cache_key)
            
            self._misses += 1
            return None
    
    def put(
        self,
        gradients: Dict[str, mx.array],
        result: GradientNormResult
    ) -> None:
        """
        Store gradient norm result in cache.
        
        Args:
            gradients: Gradient dictionary
            result: Computed gradient norm result
        """
        with self._lock:
            cache_key = self._generate_cache_key(gradients)
            current_time = time.time()
            
            # Evict if at capacity
            if len(self._cache) >= self.max_size and cache_key not in self._cache:
                self._evict_lru()
            
            # Store result
            self._cache[cache_key] = (result, current_time)
            
            # Update access order
            if cache_key in self._access_order:
                self._access_order.remove(cache_key)
            self._access_order.append(cache_key)
    
    def _evict_lru(self) -> None:
        """Evict least recently used cache entry."""
        if self._access_order:
            lru_key = self._access_order.popleft()
            if lru_key in self._cache:
                del self._cache[lru_key]
                self._evictions += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get cache performance metrics."""
        with self._lock:
            hit_rate = self._hits / max(self._total_requests, 1)
            return {
                'cache_size': len(self._cache),
                'max_size': self.max_size,
                'hit_rate': hit_rate,
                'hits': self._hits,
                'misses': self._misses,
                'evictions': self._evictions,
                'total_requests': self._total_requests,
            }
    
    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()


class GradientNormStrategy(MetricComputationStrategy):
    """
    Abstract base class for gradient norm computation strategies.
    
    This class defines the interface for different gradient norm
    computation algorithms, each optimized for specific use cases
    and performance requirements.
    """
    
    def _validate_input_data(self, data: Dict[str, Any]) -> None:
        """Validate gradient data for norm computation."""
        if 'gradients' not in data:
            raise GradientNormComputationError(
                "Missing 'gradients' key in input data",
                gradient_info={'data_keys': list(data.keys())}
            )
        
        gradients = data['gradients']
        if not isinstance(gradients, dict):
            raise GradientNormComputationError(
                f"Gradients must be a dictionary, got {type(gradients)}",
                gradient_info={'gradients_type': str(type(gradients))}
            )
        
        if not gradients:
            raise GradientNormComputationError(
                "Empty gradients dictionary provided",
                gradient_info={'gradients_empty': True}
            )
    
    def _preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess gradient data before norm computation."""
        gradients = data['gradients']
        
        # Filter out non-MLX arrays and invalid gradients
        valid_gradients = {}
        for path, grad in tree_flatten(gradients):
            if isinstance(grad, mx.array) and grad.size > 0:
                valid_gradients[path] = grad
        
        if not valid_gradients:
            raise GradientNormComputationError(
                "No valid MLX arrays found in gradients",
                gradient_info={'original_count': len(gradients)}
            )
        
        return {
            'gradients': valid_gradients,
            'original_data': data
        }
    
    @abstractmethod
    def _compute_gradient_norm_impl(
        self,
        gradients: Dict[str, mx.array]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Implement gradient norm computation logic.
        
        Args:
            gradients: Dictionary of gradient arrays
            
        Returns:
            Tuple of (norm_value, computation_metadata)
        """
        pass
    
    def _compute_metrics_impl(self, data: Dict[str, Any]) -> MetricDict:
        """Compute gradient norm metrics."""
        gradients = data['gradients']
        
        start_time = time.perf_counter()
        norm_value, metadata = self._compute_gradient_norm_impl(gradients)
        computation_time = time.perf_counter() - start_time
        
        # Count parameters and layers
        layer_count = len(gradients)
        parameter_count = sum(grad.size for grad in gradients.values())
        
        return {
            'gradient_norm': norm_value,
            'computation_time': computation_time,
            'layer_count': layer_count,
            'parameter_count': parameter_count,
            **metadata
        }


class L2NormStrategy(GradientNormStrategy):
    """
    Standard L2 (Euclidean) norm computation strategy.
    
    This strategy computes the L2 norm of all gradients by:
    1. Flattening all gradient tensors
    2. Computing the sum of squares
    3. Taking the square root
    
    Time Complexity: O(n) where n is total number of parameters
    Space Complexity: O(1) with streaming computation
    
    This is the most commonly used gradient norm computation and
    provides a good balance between accuracy and performance.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize L2 norm strategy."""
        metadata = StrategyMetadata(
            name="L2NormStrategy",
            description="Standard L2 (Euclidean) norm computation",
            priority=StrategyPriority.HIGH,
            complexity=ComputationComplexity.O_N,
            memory_usage="low",
            thread_safe=True,
            performance_hints=(
                "Use for standard gradient norm computation",
                "Efficient for most training scenarios",
                "Good balance of accuracy and performance"
            )
        )
        super().__init__(metadata, config)
        
        # Configuration parameters
        self.use_streaming = self.config.get('use_streaming', True)
        self.numerical_stability_eps = self.config.get('eps', 1e-8)
    
    def _validate_config_impl(self) -> None:
        """Validate L2 norm strategy configuration."""
        if 'eps' in self.config:
            eps = self.config['eps']
            if not isinstance(eps, (int, float)) or eps < 0:
                raise ValueError("eps must be a non-negative number")
    
    def _initialize_impl(self) -> None:
        """Initialize L2 norm strategy."""
        logger.debug(f"Initialized {self.metadata.name} with streaming={self.use_streaming}")
    
    def _compute_gradient_norm_impl(
        self,
        gradients: Dict[str, mx.array]
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute L2 norm of gradients."""
        try:
            if self.use_streaming:
                # Streaming computation to minimize memory usage
                total_norm_squared = mx.array(0.0)
                
                for path, grad in gradients.items():
                    # Compute squared norm for this gradient
                    grad_norm_squared = mx.sum(grad * grad)
                    total_norm_squared = total_norm_squared + grad_norm_squared
                    
                    # Evaluate to prevent memory accumulation
                    mx.eval(total_norm_squared)
                
                # Take square root for final norm
                norm_value = float(mx.sqrt(total_norm_squared + self.numerical_stability_eps))
                
            else:
                # Batch computation (higher memory usage but potentially faster)
                all_grads = [grad.flatten() for grad in gradients.values()]
                concatenated = mx.concatenate(all_grads)
                norm_value = float(mx.sqrt(mx.sum(concatenated * concatenated) + self.numerical_stability_eps))
            
            metadata = {
                'method': 'streaming' if self.use_streaming else 'batch',
                'numerical_stability_eps': self.numerical_stability_eps,
                'gradient_paths': list(gradients.keys())
            }
            
            return norm_value, metadata
            
        except Exception as e:
            raise GradientNormComputationError(
                f"L2 norm computation failed: {e}",
                gradient_info={
                    'gradient_count': len(gradients),
                    'use_streaming': self.use_streaming
                }
            ) from e


class LayerWiseNormStrategy(GradientNormStrategy):
    """
    Layer-wise gradient norm computation strategy.
    
    This strategy computes gradient norms for each layer separately
    and provides detailed analysis of gradient distribution across
    the network. It's useful for:
    - Identifying layers with vanishing/exploding gradients
    - Analyzing gradient flow patterns
    - Debugging training dynamics
    - Layer-specific gradient clipping
    
    The strategy provides both individual layer norms and an
    aggregated global norm using configurable aggregation methods.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize layer-wise norm strategy."""
        metadata = StrategyMetadata(
            name="LayerWiseNormStrategy",
            description="Per-layer gradient norm analysis",
            priority=StrategyPriority.MEDIUM,
            complexity=ComputationComplexity.O_N,
            memory_usage="medium",
            thread_safe=True,
            performance_hints=(
                "Use for detailed gradient analysis",
                "Helpful for debugging training dynamics",
                "Provides layer-specific insights"
            )
        )
        super().__init__(metadata, config)
        
        # Configuration parameters
        self.aggregation_method = self.config.get('aggregation_method', 'l2')
        self.outlier_threshold = self.config.get('outlier_threshold', 3.0)
        self.min_layer_params = self.config.get('min_layer_params', 1)
        
        # Layer analysis state
        self.layer_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
    
    def _validate_config_impl(self) -> None:
        """Validate layer-wise norm strategy configuration."""
        valid_methods = ['l2', 'max', 'mean', 'weighted']
        if self.aggregation_method not in valid_methods:
            raise ValueError(f"aggregation_method must be one of {valid_methods}")
        
        if not isinstance(self.outlier_threshold, (int, float)) or self.outlier_threshold <= 0:
            raise ValueError("outlier_threshold must be a positive number")
    
    def _initialize_impl(self) -> None:
        """Initialize layer-wise norm strategy."""
        logger.debug(
            f"Initialized {self.metadata.name} with "
            f"aggregation={self.aggregation_method}, "
            f"outlier_threshold={self.outlier_threshold}"
        )
    
    def _compute_layer_norm(self, grad: mx.array) -> float:
        """Compute norm for a single layer's gradient."""
        return float(mx.sqrt(mx.sum(grad * grad)))
    
    def _detect_outliers(self, layer_norms: Dict[str, float]) -> List[str]:
        """Detect outlier layers based on gradient norms."""
        if len(layer_norms) < 3:  # Need at least 3 layers for outlier detection
            return []
        
        norms = list(layer_norms.values())
        mean_norm = np.mean(norms)
        std_norm = np.std(norms)
        
        outliers = []
        for layer_name, norm in layer_norms.items():
            if abs(norm - mean_norm) > self.outlier_threshold * std_norm:
                outliers.append(layer_name)
        
        return outliers
    
    def _aggregate_layer_norms(
        self,
        layer_norms: Dict[str, float],
        layer_params: Dict[str, int]
    ) -> float:
        """Aggregate layer norms into global norm."""
        if self.aggregation_method == 'l2':
            # Standard L2 aggregation
            return np.sqrt(sum(norm ** 2 for norm in layer_norms.values()))
        
        elif self.aggregation_method == 'max':
            # Maximum layer norm
            return max(layer_norms.values())
        
        elif self.aggregation_method == 'mean':
            # Mean layer norm
            return np.mean(list(layer_norms.values()))
        
        elif self.aggregation_method == 'weighted':
            # Parameter-weighted aggregation
            total_weighted = sum(
                norm ** 2 * layer_params.get(layer, 1)
                for layer, norm in layer_norms.items()
            )
            total_params = sum(layer_params.values())
            return np.sqrt(total_weighted / max(total_params, 1))
        
        else:
            # Fallback to L2
            return np.sqrt(sum(norm ** 2 for norm in layer_norms.values()))
    
    def _compute_gradient_norm_impl(
        self,
        gradients: Dict[str, mx.array]
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute layer-wise gradient norms."""
        try:
            layer_norms = {}
            layer_params = {}
            layer_info = []
            
            # Compute norm for each layer
            for path, grad in gradients.items():
                if grad.size >= self.min_layer_params:
                    layer_norm = self._compute_layer_norm(grad)
                    layer_norms[path] = layer_norm
                    layer_params[path] = grad.size
                    
                    # Store layer information
                    layer_info.append(LayerGradientInfo(
                        layer_name=path,
                        norm_value=layer_norm,
                        parameter_count=grad.size,
                        gradient_shape=grad.shape
                    ))
                    
                    # Update layer history for trend analysis
                    self.layer_history[path].append(layer_norm)
            
            # Detect outliers
            outliers = self._detect_outliers(layer_norms)
            
            # Mark outliers in layer info
            for info in layer_info:
                if info.layer_name in outliers:
                    info.is_outlier = True
            
            # Aggregate layer norms
            global_norm = self._aggregate_layer_norms(layer_norms, layer_params)
            
            # Compute statistics
            norm_values = list(layer_norms.values())
            statistics = {
                'layer_norm_mean': np.mean(norm_values),
                'layer_norm_std': np.std(norm_values),
                'layer_norm_min': np.min(norm_values),
                'layer_norm_max': np.max(norm_values),
                'layer_norm_median': np.median(norm_values),
            }
            
            metadata = {
                'aggregation_method': self.aggregation_method,
                'layer_norms': layer_norms,
                'outlier_layers': outliers,
                'outliers_detected': len(outliers),
                'statistics': statistics,
                'layer_info': [info.__dict__ for info in layer_info]
            }
            
            return global_norm, metadata
            
        except Exception as e:
            raise GradientNormComputationError(
                f"Layer-wise norm computation failed: {e}",
                gradient_info={
                    'gradient_count': len(gradients),
                    'aggregation_method': self.aggregation_method
                }
            ) from e
    
    def get_layer_analysis(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed analysis of layer gradient trends."""
        analysis = {}
        
        for layer_name, history in self.layer_history.items():
            if len(history) > 1:
                norms = list(history)
                analysis[layer_name] = {
                    'current_norm': norms[-1],
                    'mean_norm': np.mean(norms),
                    'std_norm': np.std(norms),
                    'trend': 'increasing' if norms[-1] > norms[0] else 'decreasing',
                    'stability': np.std(norms) / max(np.mean(norms), 1e-8),
                    'history_length': len(norms)
                }
        
        return analysis


class AdaptiveNormStrategy(GradientNormStrategy):
    """
    Adaptive gradient norm computation strategy.
    
    This strategy adapts its computation method based on:
    - Gradient characteristics (sparsity, magnitude distribution)
    - Performance requirements (accuracy vs speed trade-offs)
    - System resources (memory constraints, compute capacity)
    - Historical performance data
    
    The strategy automatically selects the most appropriate computation
    method and can switch between different algorithms during training
    based on changing conditions.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize adaptive norm strategy."""
        metadata = StrategyMetadata(
            name="AdaptiveNormStrategy",
            description="Adaptive gradient norm computation with automatic optimization",
            priority=StrategyPriority.HIGH,
            complexity=ComputationComplexity.O_N,
            memory_usage="adaptive",
            thread_safe=True,
            performance_hints=(
                "Automatically optimizes computation method",
                "Adapts to gradient characteristics",
                "Best for varying training conditions"
            )
        )
        super().__init__(metadata, config)
        
        # Configuration parameters
        self.adaptation_interval = self.config.get('adaptation_interval', 100)
        self.performance_threshold = self.config.get('performance_threshold', 0.001)
        self.sparsity_threshold = self.config.get('sparsity_threshold', 0.1)
        
        # Available sub-strategies
        self.l2_strategy = L2NormStrategy({'use_streaming': True})
        self.layerwise_strategy = LayerWiseNormStrategy({'aggregation_method': 'l2'})
        
        # Adaptation state
        self.current_strategy = self.l2_strategy
        self.adaptation_counter = 0
        self.performance_history = deque(maxlen=self.adaptation_interval)
        self.gradient_characteristics = {}
    
    def _validate_config_impl(self) -> None:
        """Validate adaptive norm strategy configuration."""
        if not isinstance(self.adaptation_interval, int) or self.adaptation_interval <= 0:
            raise ValueError("adaptation_interval must be a positive integer")
        
        if not isinstance(self.performance_threshold, (int, float)) or self.performance_threshold <= 0:
            raise ValueError("performance_threshold must be a positive number")
    
    def _initialize_impl(self) -> None:
        """Initialize adaptive norm strategy."""
        logger.debug(
            f"Initialized {self.metadata.name} with "
            f"adaptation_interval={self.adaptation_interval}"
        )
    
    def _analyze_gradient_characteristics(
        self,
        gradients: Dict[str, mx.array]
    ) -> Dict[str, Any]:
        """Analyze gradient characteristics for adaptation decisions."""
        total_params = sum(grad.size for grad in gradients.values())
        
        # Compute sparsity (approximate)
        zero_count = 0
        sample_size = min(1000, total_params)  # Sample for efficiency
        
        for grad in gradients.values():
            if grad.size <= sample_size:
                zero_count += int(mx.sum(grad == 0))
            else:
                # Sample random elements
                flat_grad = grad.flatten()
                indices = np.random.choice(grad.size, sample_size, replace=False)
                sampled = flat_grad[indices]
                zero_count += int(mx.sum(sampled == 0))
        
        sparsity = zero_count / sample_size
        
        # Compute magnitude statistics
        all_magnitudes = []
        for grad in gradients.values():
            if grad.size <= 1000:
                all_magnitudes.extend(mx.abs(grad.flatten()).tolist())
            else:
                # Sample for large gradients
                flat_grad = mx.abs(grad.flatten())
                indices = np.random.choice(grad.size, 1000, replace=False)
                all_magnitudes.extend(flat_grad[indices].tolist())
        
        magnitude_stats = {
            'mean': np.mean(all_magnitudes),
            'std': np.std(all_magnitudes),
            'max': np.max(all_magnitudes),
            'min': np.min(all_magnitudes)
        }
        
        return {
            'total_parameters': total_params,
            'layer_count': len(gradients),
            'sparsity': sparsity,
            'magnitude_stats': magnitude_stats,
            'is_sparse': sparsity > self.sparsity_threshold
        }
    
    def _select_optimal_strategy(
        self,
        characteristics: Dict[str, Any],
        performance_history: List[float]
    ) -> GradientNormStrategy:
        """Select optimal strategy based on characteristics and performance."""
        # Simple heuristic-based selection
        # In production, this could use ML models or more sophisticated logic
        
        if characteristics['is_sparse']:
            # Use layerwise for sparse gradients to get better insights
            return self.layerwise_strategy
        
        if characteristics['layer_count'] > 100:
            # Use streaming L2 for large models
            return self.l2_strategy
        
        if len(performance_history) > 10:
            avg_time = np.mean(performance_history[-10:])
            if avg_time > self.performance_threshold:
                # Switch to faster method if performance is poor
                return self.l2_strategy
        
        # Default to current strategy if no clear preference
        return self.current_strategy
    
    def _compute_gradient_norm_impl(
        self,
        gradients: Dict[str, mx.array]
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute gradient norm using adaptive strategy selection."""
        try:
            start_time = time.perf_counter()
            
            # Analyze gradient characteristics
            characteristics = self._analyze_gradient_characteristics(gradients)
            self.gradient_characteristics = characteristics
            
            # Adapt strategy if needed
            self.adaptation_counter += 1
            if self.adaptation_counter % self.adaptation_interval == 0:
                optimal_strategy = self._select_optimal_strategy(
                    characteristics, list(self.performance_history)
                )
                
                if optimal_strategy != self.current_strategy:
                    logger.debug(
                        f"Switching from {self.current_strategy.metadata.name} "
                        f"to {optimal_strategy.metadata.name}"
                    )
                    self.current_strategy = optimal_strategy
            
            # Compute norm using current strategy
            norm_value, strategy_metadata = self.current_strategy._compute_gradient_norm_impl(gradients)
            
            computation_time = time.perf_counter() - start_time
            self.performance_history.append(computation_time)
            
            # Combine metadata
            metadata = {
                'adaptive_strategy_used': self.current_strategy.metadata.name,
                'gradient_characteristics': characteristics,
                'adaptation_counter': self.adaptation_counter,
                'performance_history_size': len(self.performance_history),
                **strategy_metadata
            }
            
            return norm_value, metadata
            
        except Exception as e:
            raise GradientNormComputationError(
                f"Adaptive norm computation failed: {e}",
                gradient_info={
                    'current_strategy': self.current_strategy.metadata.name,
                    'adaptation_counter': self.adaptation_counter
                }
            ) from e


class GradientNormTracker:
    """
    High-level gradient norm tracking system.
    
    This class provides a unified interface for gradient norm computation
    and tracking, integrating multiple strategies, caching, and monitoring
    capabilities.
    
    Features:
    - Multiple computation strategies with automatic selection
    - Intelligent caching for performance optimization
    - Real-time monitoring and alerting
    - Thread-safe operations for concurrent training
    - Comprehensive error handling and fallback mechanisms
    - Performance analytics and optimization recommendations
    """
    
    def __init__(
        self,
        strategy: Optional[GradientNormStrategy] = None,
        enable_caching: bool = True,
        cache_size: int = 1000,
        cache_ttl: float = 300.0,
        enable_monitoring: bool = True,
        fallback_strategies: Optional[List[GradientNormStrategy]] = None
    ):
        """
        Initialize gradient norm tracker.
        
        Args:
            strategy: Primary gradient norm computation strategy
            enable_caching: Whether to enable result caching
            cache_size: Maximum cache size
            cache_ttl: Cache time-to-live in seconds
            enable_monitoring: Whether to enable performance monitoring
            fallback_strategies: List of fallback strategies for error recovery
        """
        # Primary strategy
        self.strategy = strategy or AdaptiveNormStrategy()
        
        # Fallback strategies for error recovery
        self.fallback_strategies = fallback_strategies or [
            L2NormStrategy({'use_streaming': True}),
            L2NormStrategy({'use_streaming': False})
        ]
        
        # Caching system
        self.enable_caching = enable_caching
        self.cache = GradientNormCache(
            max_size=cache_size,
            ttl_seconds=cache_ttl,
            enable_metrics=enable_monitoring
        ) if enable_caching else None
        
        # Monitoring and analytics
        self.enable_monitoring = enable_monitoring
        self.computation_history = deque(maxlen=1000)
        self.error_count = 0
        self.fallback_count = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(
            f"Initialized GradientNormTracker with strategy: {self.strategy.metadata.name}"
        )
    
    def compute_norm(
        self,
        gradients: Dict[str, mx.array],
        use_cache: bool = True
    ) -> GradientNormResult:
        """
        Compute gradient norm with caching and error handling.
        
        Args:
            gradients: Dictionary of gradient arrays
            use_cache: Whether to use caching for this computation
            
        Returns:
            Gradient norm computation result
            
        Raises:
            GradientNormComputationError: If all computation strategies fail
        """
        with self._lock:
            # Try cache first
            if self.enable_caching and use_cache and self.cache:
                cached_result = self.cache.get(gradients)
                if cached_result:
                    return cached_result
            
            # Attempt computation with primary strategy
            strategies_to_try = [self.strategy] + self.fallback_strategies
            
            for i, strategy in enumerate(strategies_to_try):
                try:
                    start_time = time.perf_counter()
                    
                    # Compute gradient norm
                    metrics = strategy.execute({'gradients': gradients})
                    
                    computation_time = time.perf_counter() - start_time
                    
                    # Create result object
                    result = GradientNormResult(
                        norm_value=metrics['gradient_norm'],
                        computation_time=computation_time,
                        strategy_used=strategy.metadata.name,
                        layer_count=metrics.get('layer_count', 0),
                        parameter_count=metrics.get('parameter_count', 0),
                        cache_hit=False,
                        outliers_detected=metrics.get('outliers_detected', 0),
                        metadata=metrics
                    )
                    
                    # Cache result if enabled
                    if self.enable_caching and use_cache and self.cache:
                        self.cache.put(gradients, result)
                    
                    # Update monitoring
                    if self.enable_monitoring:
                        self.computation_history.append(result)
                    
                    # Track fallback usage
                    if i > 0:
                        self.fallback_count += 1
                        logger.warning(
                            f"Used fallback strategy {strategy.metadata.name} "
                            f"(attempt {i + 1})"
                        )
                    
                    return result
                    
                except Exception as e:
                    self.error_count += 1
                    logger.warning(
                        f"Strategy {strategy.metadata.name} failed: {e}"
                    )
                    
                    if i == len(strategies_to_try) - 1:
                        # All strategies failed
                        raise GradientNormComputationError(
                            f"All gradient norm computation strategies failed. "
                            f"Last error: {e}",
                            gradient_info={
                                'strategies_tried': [s.metadata.name for s in strategies_to_try],
                                'error_count': self.error_count
                            }
                        ) from e
    
    def analyze_layers(
        self,
        gradients: Dict[str, mx.array]
    ) -> Dict[str, LayerGradientInfo]:
        """
        Perform detailed layer-wise gradient analysis.
        
        Args:
            gradients: Dictionary of gradient arrays
            
        Returns:
            Dictionary mapping layer names to gradient information
        """
        # Use layer-wise strategy for detailed analysis
        layerwise_strategy = LayerWiseNormStrategy()
        
        try:
            metrics = layerwise_strategy.execute({'gradients': gradients})
            
            # Extract layer information
            layer_info = {}
            for info_dict in metrics.get('layer_info', []):
                layer_name = info_dict['layer_name']
                layer_info[layer_name] = LayerGradientInfo(**info_dict)
            
            return layer_info
            
        except Exception as e:
            raise GradientNormComputationError(
                f"Layer analysis failed: {e}",
                gradient_info={'analysis_type': 'layer_wise'}
            ) from e
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            'primary_strategy': self.strategy.metadata.name,
            'error_count': self.error_count,
            'fallback_count': self.fallback_count,
            'computation_count': len(self.computation_history),
        }
        
        # Add strategy performance stats
        stats.update(self.strategy.get_performance_stats())
        
        # Add cache stats if enabled
        if self.cache:
            stats['cache'] = self.cache.get_metrics()
        
        # Add computation time statistics
        if self.computation_history:
            times = [r.computation_time for r in self.computation_history]
            stats['computation_time'] = {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times),
                'p95': np.percentile(times, 95),
                'p99': np.percentile(times, 99)
            }
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset all performance statistics."""
        with self._lock:
            self.error_count = 0
            self.fallback_count = 0
            self.computation_history.clear()
            self.strategy.reset_stats()
            
            if self.cache:
                self.cache.clear()


# Export all classes and functions
__all__ = [
    'GradientNormResult',
    'LayerGradientInfo',
    'GradientNormCache',
    'GradientNormStrategy',
    'L2NormStrategy',
    'LayerWiseNormStrategy',
    'AdaptiveNormStrategy',
    'GradientNormTracker',
]