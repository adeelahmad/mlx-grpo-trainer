"""
Unified Metrics Interface for MLX RL Trainer

This module defines a comprehensive interface for metrics collection, validation,
and reporting in the MLX RL Trainer framework. It provides a consistent API
for all metrics operations while ensuring backward compatibility with existing
metrics loggers.

The interface follows the Adapter pattern to provide a unified API across different
metrics implementations, and the Strategy pattern to allow for pluggable validation
and sampling strategies.

Key Features:
- Abstract base classes for metrics collection and reporting
- Pluggable validation strategies for metrics data
- Correlation ID support for tracking context across operations
- Thread-safe operations for concurrent environments
- Intelligent sampling for high-frequency metrics
- Comprehensive error handling and reporting
"""

import abc
import enum
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable, TypeVar, Generic

import numpy as np

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

logger = logging.getLogger(__name__)

# Type aliases for improved readability
MetricName = str
MetricValue = Union[float, int, str, bool, None]
MetricDict = Dict[MetricName, MetricValue]
MetricTimestamp = float
StepNumber = int
CorrelationId = str


class MetricValidationLevel(enum.Enum):
    """Validation levels for metrics data."""
    NONE = 0       # No validation
    BASIC = 1      # Basic type checking
    STANDARD = 2   # Type checking and range validation
    STRICT = 3     # Comprehensive validation with warnings/errors


class MetricSamplingStrategy(enum.Enum):
    """Sampling strategies for high-frequency metrics."""
    NONE = 0           # No sampling, record all metrics
    FIXED_INTERVAL = 1 # Sample at fixed intervals
    ADAPTIVE = 2       # Adaptive sampling based on value changes
    RESERVOIR = 3      # Reservoir sampling for statistical representation
    PRIORITY = 4       # Priority-based sampling for important metrics


class MetricPriority(enum.Enum):
    """Priority levels for metrics."""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class MetricContext:
    """
    Context information for metrics collection.
    
    This class provides contextual information for metrics collection,
    including correlation IDs for tracking related metrics across
    different components and operations.
    
    Attributes:
        correlation_id: Unique identifier for correlating related metrics
        parent_id: Optional parent correlation ID for hierarchical tracking
        timestamp: Time when the context was created
        tags: Optional tags for categorizing metrics
        metadata: Additional contextual information
    """
    correlation_id: CorrelationId = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[CorrelationId] = None
    timestamp: MetricTimestamp = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def create_child_context(self) -> 'MetricContext':
        """Create a child context with this context as parent."""
        return MetricContext(
            parent_id=self.correlation_id,
            tags=self.tags.copy(),
            metadata=self.metadata.copy()
        )
    
    def with_tag(self, key: str, value: str) -> 'MetricContext':
        """Create a new context with an additional tag."""
        new_context = MetricContext(
            correlation_id=self.correlation_id,
            parent_id=self.parent_id,
            timestamp=self.timestamp,
            tags=self.tags.copy(),
            metadata=self.metadata.copy()
        )
        new_context.tags[key] = value
        return new_context
    
    def with_metadata(self, key: str, value: Any) -> 'MetricContext':
        """Create a new context with additional metadata."""
        new_context = MetricContext(
            correlation_id=self.correlation_id,
            parent_id=self.parent_id,
            timestamp=self.timestamp,
            tags=self.tags.copy(),
            metadata=self.metadata.copy()
        )
        new_context.metadata[key] = value
        return new_context


@dataclass
class MetricValidationResult:
    """
    Result of metric validation.
    
    Attributes:
        is_valid: Whether the metric is valid
        level: Validation level applied
        errors: List of validation errors
        warnings: List of validation warnings
    """
    is_valid: bool = True
    level: MetricValidationLevel = MetricValidationLevel.NONE
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ValidationStrategy(abc.ABC):
    """
    Abstract base class for metric validation strategies.
    
    This class defines the interface for metric validation strategies,
    which are responsible for validating metrics data according to
    specific rules and constraints.
    """
    
    @abc.abstractmethod
    def validate(self, name: MetricName, value: MetricValue) -> MetricValidationResult:
        """
        Validate a metric value.
        
        Args:
            name: Name of the metric
            value: Value to validate
            
        Returns:
            Validation result
        """
        pass


class BasicValidationStrategy(ValidationStrategy):
    """
    Basic validation strategy that checks metric types.
    
    This strategy performs basic type checking on metric values,
    ensuring they are of the expected types.
    """
    
    def validate(self, name: MetricName, value: MetricValue) -> MetricValidationResult:
        """
        Validate a metric value using basic type checking.
        
        Args:
            name: Name of the metric
            value: Value to validate
            
        Returns:
            Validation result
        """
        result = MetricValidationResult(level=MetricValidationLevel.BASIC)
        
        # Check for None
        if value is None:
            return result
        
        # Check for valid types
        if not isinstance(value, (int, float, str, bool)):
            if MLX_AVAILABLE and isinstance(value, mx.array):
                # MLX arrays are valid
                pass
            elif isinstance(value, np.ndarray):
                # NumPy arrays are valid
                pass
            else:
                result.is_valid = False
                result.errors.append(
                    f"Invalid type for metric '{name}': {type(value).__name__}"
                )
        
        return result


class StandardValidationStrategy(BasicValidationStrategy):
    """
    Standard validation strategy that checks types and ranges.
    
    This strategy extends basic validation with range checking for
    numeric values, ensuring they fall within expected ranges.
    """
    
    def __init__(self, ranges: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None):
        """
        Initialize standard validation strategy.
        
        Args:
            ranges: Dictionary mapping metric names to (min, max) tuples
        """
        self.ranges = ranges or {}
    
    def validate(self, name: MetricName, value: MetricValue) -> MetricValidationResult:
        """
        Validate a metric value using type and range checking.
        
        Args:
            name: Name of the metric
            value: Value to validate
            
        Returns:
            Validation result
        """
        # First perform basic validation
        result = super().validate(name, value)
        result.level = MetricValidationLevel.STANDARD
        
        # Skip further validation if already invalid
        if not result.is_valid or value is None:
            return result
        
        # Check range for numeric values
        if isinstance(value, (int, float)):
            if name in self.ranges:
                min_val, max_val = self.ranges[name]
                
                if min_val is not None and value < min_val:
                    result.is_valid = False
                    result.errors.append(
                        f"Value {value} for metric '{name}' is below minimum {min_val}"
                    )
                
                if max_val is not None and value > max_val:
                    result.is_valid = False
                    result.errors.append(
                        f"Value {value} for metric '{name}' exceeds maximum {max_val}"
                    )
        
        return result


class StrictValidationStrategy(StandardValidationStrategy):
    """
    Strict validation strategy with comprehensive checks.
    
    This strategy performs comprehensive validation including type checking,
    range validation, and additional custom rules.
    """
    
    def __init__(
        self,
        ranges: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
        required_metrics: Optional[Set[str]] = None,
        custom_validators: Optional[Dict[str, Callable[[Any], Tuple[bool, Optional[str]]]]] = None
    ):
        """
        Initialize strict validation strategy.
        
        Args:
            ranges: Dictionary mapping metric names to (min, max) tuples
            required_metrics: Set of required metric names
            custom_validators: Dictionary mapping metric names to custom validator functions
        """
        super().__init__(ranges)
        self.required_metrics = required_metrics or set()
        self.custom_validators = custom_validators or {}
    
    def validate(self, name: MetricName, value: MetricValue) -> MetricValidationResult:
        """
        Validate a metric value using comprehensive checks.
        
        Args:
            name: Name of the metric
            value: Value to validate
            
        Returns:
            Validation result
        """
        # First perform standard validation
        result = super().validate(name, value)
        result.level = MetricValidationLevel.STRICT
        
        # Skip further validation if already invalid
        if not result.is_valid:
            return result
        
        # Apply custom validators
        if name in self.custom_validators:
            validator = self.custom_validators[name]
            try:
                is_valid, error_msg = validator(value)
                if not is_valid:
                    result.is_valid = False
                    result.errors.append(error_msg or f"Custom validation failed for '{name}'")
            except Exception as e:
                result.is_valid = False
                result.errors.append(f"Custom validator error for '{name}': {str(e)}")
        
        return result
    
    def validate_metrics(self, metrics: MetricDict) -> Dict[str, MetricValidationResult]:
        """
        Validate a dictionary of metrics.
        
        Args:
            metrics: Dictionary of metrics to validate
            
        Returns:
            Dictionary mapping metric names to validation results
        """
        results = {}
        
        # Check for required metrics
        for required in self.required_metrics:
            if required not in metrics:
                results[required] = MetricValidationResult(
                    is_valid=False,
                    level=MetricValidationLevel.STRICT,
                    errors=[f"Required metric '{required}' is missing"]
                )
        
        # Validate each metric
        for name, value in metrics.items():
            results[name] = self.validate(name, value)
        
        return results


class SamplingStrategy(abc.ABC):
    """
    Abstract base class for metric sampling strategies.
    
    This class defines the interface for metric sampling strategies,
    which are responsible for determining when to sample high-frequency
    metrics to reduce storage and processing overhead.
    """
    
    @abc.abstractmethod
    def should_sample(self, name: MetricName, value: MetricValue, context: MetricContext) -> bool:
        """
        Determine whether to sample a metric.
        
        Args:
            name: Name of the metric
            value: Current metric value
            context: Metric context
            
        Returns:
            True if the metric should be sampled, False otherwise
        """
        pass


class FixedIntervalSamplingStrategy(SamplingStrategy):
    """
    Sampling strategy that samples metrics at fixed intervals.
    
    This strategy samples metrics at fixed intervals based on step numbers
    or time intervals, reducing the volume of data collected.
    """
    
    def __init__(self, interval: int = 10, time_based: bool = False):
        """
        Initialize fixed interval sampling strategy.
        
        Args:
            interval: Sampling interval (in steps or seconds)
            time_based: Whether to use time-based intervals
        """
        self.interval = max(1, interval)
        self.time_based = time_based
        self.last_sample_time = {}
        self.last_sample_step = {}
    
    def should_sample(self, name: MetricName, value: MetricValue, context: MetricContext) -> bool:
        """
        Determine whether to sample a metric based on fixed intervals.
        
        Args:
            name: Name of the metric
            value: Current metric value
            context: Metric context
            
        Returns:
            True if the metric should be sampled, False otherwise
        """
        current_time = time.time()
        step = context.metadata.get('step', 0)
        
        if self.time_based:
            # Time-based sampling
            last_time = self.last_sample_time.get(name, 0)
            if current_time - last_time >= self.interval:
                self.last_sample_time[name] = current_time
                return True
            return False
        else:
            # Step-based sampling
            last_step = self.last_sample_step.get(name, -self.interval)
            if step - last_step >= self.interval:
                self.last_sample_step[name] = step
                return True
            return False


class AdaptiveSamplingStrategy(SamplingStrategy):
    """
    Adaptive sampling strategy based on value changes.
    
    This strategy samples metrics based on how much their values have changed,
    focusing on capturing significant changes while reducing redundant data.
    """
    
    def __init__(self, threshold: float = 0.05, min_interval: int = 1, max_interval: int = 100):
        """
        Initialize adaptive sampling strategy.
        
        Args:
            threshold: Change threshold for sampling (as fraction of value)
            min_interval: Minimum sampling interval
            max_interval: Maximum sampling interval
        """
        self.threshold = threshold
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.last_values = {}
        self.last_sample_step = {}
    
    def should_sample(self, name: MetricName, value: MetricValue, context: MetricContext) -> bool:
        """
        Determine whether to sample a metric based on value changes.
        
        Args:
            name: Name of the metric
            value: Current metric value
            context: Metric context
            
        Returns:
            True if the metric should be sampled, False otherwise
        """
        step = context.metadata.get('step', 0)
        
        # Always sample at minimum interval
        last_step = self.last_sample_step.get(name, -self.min_interval)
        if step - last_step < self.min_interval:
            return False
        
        # Always sample at maximum interval
        if step - last_step >= self.max_interval:
            self.last_sample_step[name] = step
            if isinstance(value, (int, float)):
                self.last_values[name] = value
            return True
        
        # Sample based on value change
        if isinstance(value, (int, float)):
            last_value = self.last_values.get(name, value)
            
            # Calculate relative change
            if abs(last_value) > 1e-10:
                relative_change = abs((value - last_value) / last_value)
            else:
                relative_change = abs(value - last_value)
            
            if relative_change >= self.threshold:
                self.last_sample_step[name] = step
                self.last_values[name] = value
                return True
        
        return False


class PrioritySamplingStrategy(SamplingStrategy):
    """
    Priority-based sampling strategy.
    
    This strategy samples metrics based on their priority levels,
    ensuring that high-priority metrics are sampled more frequently.
    """
    
    def __init__(
        self,
        priorities: Dict[str, MetricPriority],
        intervals: Dict[MetricPriority, int]
    ):
        """
        Initialize priority sampling strategy.
        
        Args:
            priorities: Dictionary mapping metric names to priority levels
            intervals: Dictionary mapping priority levels to sampling intervals
        """
        self.priorities = priorities
        self.intervals = intervals
        self.last_sample_step = {}
    
    def should_sample(self, name: MetricName, value: MetricValue, context: MetricContext) -> bool:
        """
        Determine whether to sample a metric based on priority.
        
        Args:
            name: Name of the metric
            value: Current metric value
            context: Metric context
            
        Returns:
            True if the metric should be sampled, False otherwise
        """
        step = context.metadata.get('step', 0)
        
        # Get priority and interval
        priority = self.priorities.get(name, MetricPriority.MEDIUM)
        interval = self.intervals.get(priority, 10)
        
        # Check if it's time to sample
        last_step = self.last_sample_step.get(name, -interval)
        if step - last_step >= interval:
            self.last_sample_step[name] = step
            return True
        
        return False


T = TypeVar('T')

class MetricsInterface(Generic[T], abc.ABC):
    """
    Abstract base class for metrics collection and reporting.
    
    This interface defines the core functionality for metrics operations,
    providing a consistent API across different implementations.
    """
    
    @abc.abstractmethod
    def log_metric(
        self,
        name: MetricName,
        value: MetricValue,
        step: Optional[StepNumber] = None,
        context: Optional[MetricContext] = None
    ) -> T:
        """
        Log a single metric.
        
        Args:
            name: Name of the metric
            value: Metric value
            step: Training step number (optional)
            context: Metric context (optional)
            
        Returns:
            Implementation-specific result
        """
        pass
    
    @abc.abstractmethod
    def log_metrics(
        self,
        metrics: MetricDict,
        step: Optional[StepNumber] = None,
        context: Optional[MetricContext] = None
    ) -> T:
        """
        Log multiple metrics.
        
        Args:
            metrics: Dictionary of metrics
            step: Training step number (optional)
            context: Metric context (optional)
            
        Returns:
            Implementation-specific result
        """
        pass
    
    @abc.abstractmethod
    def get_metric(
        self,
        name: MetricName,
        default: Optional[MetricValue] = None
    ) -> Optional[MetricValue]:
        """
        Get the latest value of a metric.
        
        Args:
            name: Name of the metric
            default: Default value if metric not found
            
        Returns:
            Latest metric value or default
        """
        pass
    
    @abc.abstractmethod
    def get_metrics(
        self,
        names: Optional[List[MetricName]] = None
    ) -> MetricDict:
        """
        Get the latest values of multiple metrics.
        
        Args:
            names: List of metric names (optional, gets all if None)
            
        Returns:
            Dictionary of metric values
        """
        pass


class MetricsLogger(MetricsInterface[None]):
    """
    Base implementation of the metrics interface.
    
    This class provides a thread-safe implementation of the metrics interface
    with support for validation, sampling, and correlation IDs.
    """
    
    def __init__(
        self,
        validation_strategy: Optional[ValidationStrategy] = None,
        sampling_strategy: Optional[SamplingStrategy] = None,
        enable_correlation_ids: bool = True
    ):
        """
        Initialize metrics logger.
        
        Args:
            validation_strategy: Strategy for validating metrics
            sampling_strategy: Strategy for sampling high-frequency metrics
            enable_correlation_ids: Whether to enable correlation ID tracking
        """
        self.validation_strategy = validation_strategy or BasicValidationStrategy()
        self.sampling_strategy = sampling_strategy
        self.enable_correlation_ids = enable_correlation_ids
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Metrics storage
        self._metrics: MetricDict = {}
        self._metrics_history: Dict[MetricName, List[Tuple[MetricValue, MetricTimestamp, Optional[StepNumber]]]] = {}
        
        # Context tracking
        self._context_stack: List[MetricContext] = []
        self._current_context = MetricContext()
    
    def push_context(self, context: Optional[MetricContext] = None) -> MetricContext:
        """
        Push a new context onto the stack.
        
        Args:
            context: Context to push (creates new if None)
            
        Returns:
            The current context
        """
        with self._lock:
            if context is None:
                if self._context_stack:
                    context = self._context_stack[-1].create_child_context()
                else:
                    context = MetricContext()
            
            self._context_stack.append(context)
            self._current_context = context
            return context
    
    def pop_context(self) -> Optional[MetricContext]:
        """
        Pop the current context from the stack.
        
        Returns:
            The popped context, or None if stack is empty
        """
        with self._lock:
            if not self._context_stack:
                return None
            
            popped = self._context_stack.pop()
            self._current_context = self._context_stack[-1] if self._context_stack else MetricContext()
            return popped
    
    def get_current_context(self) -> MetricContext:
        """
        Get the current context.
        
        Returns:
            Current context
        """
        return self._current_context
    
    def log_metric(
        self,
        name: MetricName,
        value: MetricValue,
        step: Optional[StepNumber] = None,
        context: Optional[MetricContext] = None
    ) -> None:
        """
        Log a single metric.
        
        Args:
            name: Name of the metric
            value: Metric value
            step: Training step number (optional)
            context: Metric context (optional)
        """
        # Use current context if none provided
        if context is None:
            context = self._current_context
        
        # Validate metric
        validation_result = self.validation_strategy.validate(name, value)
        if not validation_result.is_valid:
            for error in validation_result.errors:
                logger.error(f"Metric validation error: {error}")
            return
        
        # Check if we should sample this metric
        if self.sampling_strategy and not self.sampling_strategy.should_sample(name, value, context):
            return
        
        # Store metric
        with self._lock:
            self._metrics[name] = value
            
            # Store in history
            if name not in self._metrics_history:
                self._metrics_history[name] = []
            
            self._metrics_history[name].append((value, time.time(), step))
            
            # Limit history size
            if len(self._metrics_history[name]) > 1000:
                self._metrics_history[name] = self._metrics_history[name][-1000:]
    
    def log_metrics(
        self,
        metrics: MetricDict,
        step: Optional[StepNumber] = None,
        context: Optional[MetricContext] = None
    ) -> None:
        """
        Log multiple metrics.
        
        Args:
            metrics: Dictionary of metrics
            step: Training step number (optional)
            context: Metric context (optional)
        """
        # Use current context if none provided
        if context is None:
            context = self._current_context
        
        # Add step to context metadata
        if step is not None:
            context = context.with_metadata('step', step)
        
        # Log each metric
        for name, value in metrics.items():
            self.log_metric(name, value, step, context)
    
    def get_metric(
        self,
        name: MetricName,
        default: Optional[MetricValue] = None
    ) -> Optional[MetricValue]:
        """
        Get the latest value of a metric.
        
        Args:
            name: Name of the metric
            default: Default value if metric not found
            
        Returns:
            Latest metric value or default
        """
        with self._lock:
            return self._metrics.get(name, default)
    
    def get_metrics(
        self,
        names: Optional[List[MetricName]] = None
    ) -> MetricDict:
        """
        Get the latest values of multiple metrics.
        
        Args:
            names: List of metric names (optional, gets all if None)
            
        Returns:
            Dictionary of metric values
        """
        with self._lock:
            if names is None:
                return self._metrics.copy()
            
            return {name: self._metrics.get(name) for name in names if name in self._metrics}
    
    def get_metric_history(
        self,
        name: MetricName,
        max_points: int = 100
    ) -> List[Tuple[MetricValue, MetricTimestamp, Optional[StepNumber]]]:
        """
        Get historical values of a metric.
        
        Args:
            name: Name of the metric
            max_points: Maximum number of points to return
            
        Returns:
            List of (value, timestamp, step) tuples
        """
        with self._lock:
            history = self._metrics_history.get(name, [])
            return history[-max_points:]
    
    def clear_metrics(self) -> None:
        """Clear all metrics data."""
        with self._lock:
            self._metrics.clear()
            self._metrics_history.clear()


# Export all classes and functions
__all__ = [
    'MetricName',
    'MetricValue',
    'MetricDict',
    'MetricTimestamp',
    'StepNumber',
    'CorrelationId',
    'MetricValidationLevel',
    'MetricSamplingStrategy',
    'MetricPriority',
    'MetricContext',
    'MetricValidationResult',
    'ValidationStrategy',
    'BasicValidationStrategy',
    'StandardValidationStrategy',
    'StrictValidationStrategy',
    'SamplingStrategy',
    'FixedIntervalSamplingStrategy',
    'AdaptiveSamplingStrategy',
    'PrioritySamplingStrategy',
    'MetricsInterface',
    'MetricsLogger',
]