"""
Base Strategy Interfaces for Progress Tracking System

This module defines the abstract base classes and interfaces for the Strategy
pattern implementation in the progress tracking system. These interfaces
ensure consistent behavior across different strategy implementations while
allowing for flexible algorithm selection at runtime.

The Strategy pattern is implemented here to provide:
- Interchangeable algorithms for metric computation
- Runtime selection of rendering strategies
- Extensible aggregation methods
- Testable and maintainable code structure

Key Interfaces:
- MetricComputationStrategy: For computing various training metrics
- RenderingStrategy: For different progress bar display formats
- AggregationStrategy: For metric aggregation and smoothing
- ValidationStrategy: For input validation and error checking

Design Principles Applied:
- Single Responsibility: Each strategy handles one specific concern
- Open/Closed: Easy to add new strategies without modifying existing code
- Liskov Substitution: All strategies are interchangeable
- Interface Segregation: Focused interfaces for specific capabilities
- Dependency Inversion: Depend on abstractions, not implementations

Example:
    ```python
    # Define a custom gradient norm strategy
    class CustomGradientNormStrategy(MetricComputationStrategy):
        def compute(self, data: Dict[str, Any]) -> Dict[str, float]:
            # Custom gradient norm computation
            return {"gradient_norm": custom_norm_value}
    
    # Use the strategy
    strategy = CustomGradientNormStrategy()
    result = strategy.compute(gradient_data)
    ```
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple, Generic, TypeVar
from dataclasses import dataclass
from enum import Enum
import time
import threading
from contextlib import contextmanager

from mlx_rl_trainer.monitoring.progress.exceptions import (
    MetricComputationError,
    RendererError,
    ConfigurationError,
    ThreadSafetyError,
    create_error_context,
)

# Type variables for generic strategies
T = TypeVar('T')
MetricValue = Union[int, float, str]
MetricDict = Dict[str, MetricValue]


class StrategyPriority(Enum):
    """Priority levels for strategy selection."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class ComputationComplexity(Enum):
    """Computational complexity levels for strategies."""
    O_1 = "O(1)"          # Constant time
    O_LOG_N = "O(log n)"  # Logarithmic time
    O_N = "O(n)"          # Linear time
    O_N_LOG_N = "O(n log n)"  # Linearithmic time
    O_N2 = "O(n²)"        # Quadratic time
    O_N3 = "O(n³)"        # Cubic time


@dataclass(frozen=True)
class StrategyMetadata:
    """
    Metadata for strategy implementations.
    
    This class provides comprehensive information about a strategy
    including performance characteristics, compatibility, and usage hints.
    
    Attributes:
        name: Human-readable strategy name
        description: Detailed description of the strategy
        version: Strategy version for compatibility tracking
        author: Strategy author/maintainer
        priority: Strategy priority level
        complexity: Computational complexity
        memory_usage: Expected memory usage pattern
        thread_safe: Whether the strategy is thread-safe
        requires_gpu: Whether GPU acceleration is required
        supported_platforms: List of supported platforms
        dependencies: List of required dependencies
        performance_hints: Performance optimization hints
    """
    name: str
    description: str
    version: str = "1.0.0"
    author: str = "MLX RL Trainer Team"
    priority: StrategyPriority = StrategyPriority.MEDIUM
    complexity: ComputationComplexity = ComputationComplexity.O_N
    memory_usage: str = "low"
    thread_safe: bool = True
    requires_gpu: bool = False
    supported_platforms: Tuple[str, ...] = ("linux", "darwin", "win32")
    dependencies: Tuple[str, ...] = ()
    performance_hints: Tuple[str, ...] = ()


class BaseStrategy(ABC, Generic[T]):
    """
    Abstract base class for all strategy implementations.
    
    This class provides the foundation for all strategy implementations
    in the progress tracking system. It enforces a consistent interface
    while providing common functionality like error handling, validation,
    and performance monitoring.
    
    The class implements several enterprise patterns:
    - Template Method: Common execution flow with customizable steps
    - Strategy: Interchangeable algorithm implementations
    - Observer: Notification of strategy events
    - Command: Encapsulation of strategy operations
    
    Attributes:
        metadata: Strategy metadata and information
        is_initialized: Whether the strategy has been initialized
        last_execution_time: Time of last strategy execution
        execution_count: Number of times strategy has been executed
        total_execution_time: Total time spent in strategy execution
        error_count: Number of errors encountered
        _lock: Thread synchronization lock
    """
    
    def __init__(
        self,
        metadata: StrategyMetadata,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize base strategy.
        
        Args:
            metadata: Strategy metadata and information
            config: Optional configuration parameters
        """
        self.metadata = metadata
        self.config = config or {}
        self.is_initialized = False
        self.last_execution_time = 0.0
        self.execution_count = 0
        self.total_execution_time = 0.0
        self.error_count = 0
        
        # Thread safety
        self._lock = threading.RLock() if metadata.thread_safe else None
        
        # Validation
        self._validate_config()
        
        # Initialize strategy-specific components
        self._initialize()
    
    def _validate_config(self) -> None:
        """
        Validate strategy configuration.
        
        Raises:
            ConfigurationError: If configuration is invalid
        """
        try:
            self._validate_config_impl()
        except Exception as e:
            raise ConfigurationError(
                f"Configuration validation failed for strategy '{self.metadata.name}': {e}",
                config_section="strategy_config",
                context=create_error_context("config_validation", config=self.config)
            ) from e
    
    @abstractmethod
    def _validate_config_impl(self) -> None:
        """
        Strategy-specific configuration validation.
        
        Subclasses must implement this method to validate their
        specific configuration parameters.
        """
        pass
    
    def _initialize(self) -> None:
        """
        Initialize strategy-specific components.
        
        This method is called during strategy construction and should
        set up any resources or state required by the strategy.
        """
        try:
            self._initialize_impl()
            self.is_initialized = True
        except Exception as e:
            raise ConfigurationError(
                f"Strategy initialization failed for '{self.metadata.name}': {e}",
                context=create_error_context("strategy_initialization")
            ) from e
    
    @abstractmethod
    def _initialize_impl(self) -> None:
        """
        Strategy-specific initialization logic.
        
        Subclasses must implement this method to perform any
        initialization required for their operation.
        """
        pass
    
    @contextmanager
    def _execution_context(self):
        """
        Context manager for strategy execution.
        
        This context manager handles:
        - Thread synchronization (if required)
        - Performance monitoring
        - Error tracking
        - Resource cleanup
        
        Yields:
            None
        """
        start_time = time.perf_counter()
        
        # Acquire lock if thread safety is required
        if self._lock:
            self._lock.acquire()
        
        try:
            yield
            
            # Update performance metrics
            execution_time = time.perf_counter() - start_time
            self.last_execution_time = execution_time
            self.total_execution_time += execution_time
            self.execution_count += 1
            
        except Exception as e:
            self.error_count += 1
            raise
        finally:
            # Release lock if acquired
            if self._lock:
                self._lock.release()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get strategy performance statistics.
        
        Returns:
            Dictionary containing performance metrics
        """
        avg_execution_time = (
            self.total_execution_time / max(self.execution_count, 1)
        )
        
        return {
            "execution_count": self.execution_count,
            "total_execution_time": self.total_execution_time,
            "average_execution_time": avg_execution_time,
            "last_execution_time": self.last_execution_time,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.execution_count, 1),
            "is_initialized": self.is_initialized,
        }
    
    def reset_stats(self) -> None:
        """Reset performance statistics."""
        with self._execution_context():
            self.execution_count = 0
            self.total_execution_time = 0.0
            self.last_execution_time = 0.0
            self.error_count = 0
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> T:
        """
        Execute the strategy with given parameters.
        
        This is the main entry point for strategy execution.
        Subclasses must implement this method to provide their
        specific functionality.
        
        Args:
            *args: Positional arguments for strategy execution
            **kwargs: Keyword arguments for strategy execution
            
        Returns:
            Strategy execution result
        """
        pass
    
    def __str__(self) -> str:
        """Return string representation of strategy."""
        return f"{self.metadata.name} v{self.metadata.version}"
    
    def __repr__(self) -> str:
        """Return detailed string representation of strategy."""
        return (
            f"{self.__class__.__name__}("
            f"name='{self.metadata.name}', "
            f"version='{self.metadata.version}', "
            f"priority={self.metadata.priority}, "
            f"complexity={self.metadata.complexity})"
        )


class MetricComputationStrategy(BaseStrategy[MetricDict]):
    """
    Abstract base class for metric computation strategies.
    
    This class defines the interface for strategies that compute
    training metrics such as gradient norms, loss values, memory
    usage, and performance indicators.
    
    Metric computation strategies must be:
    - Deterministic: Same input produces same output
    - Efficient: Minimal computational overhead
    - Robust: Handle edge cases gracefully
    - Thread-safe: Support concurrent execution
    
    The strategy follows the Template Method pattern:
    1. Validate input data
    2. Preprocess data if needed
    3. Compute metrics
    4. Postprocess results
    5. Return formatted metrics
    """
    
    @abstractmethod
    def _validate_input_data(self, data: Dict[str, Any]) -> None:
        """
        Validate input data for metric computation.
        
        Args:
            data: Input data dictionary
            
        Raises:
            MetricComputationError: If input data is invalid
        """
        pass
    
    @abstractmethod
    def _preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess input data before metric computation.
        
        Args:
            data: Raw input data
            
        Returns:
            Preprocessed data ready for computation
        """
        pass
    
    @abstractmethod
    def _compute_metrics_impl(self, data: Dict[str, Any]) -> MetricDict:
        """
        Implement the actual metric computation logic.
        
        Args:
            data: Preprocessed input data
            
        Returns:
            Dictionary of computed metrics
        """
        pass
    
    def _postprocess_results(self, metrics: MetricDict) -> MetricDict:
        """
        Postprocess computed metrics.
        
        This method can be overridden to apply transformations,
        filtering, or formatting to the computed metrics.
        
        Args:
            metrics: Raw computed metrics
            
        Returns:
            Postprocessed metrics
        """
        return metrics
    
    def execute(self, data: Dict[str, Any]) -> MetricDict:
        """
        Execute metric computation strategy.
        
        Args:
            data: Input data for metric computation
            
        Returns:
            Dictionary of computed metrics
            
        Raises:
            MetricComputationError: If computation fails
        """
        with self._execution_context():
            try:
                # Validate input data
                self._validate_input_data(data)
                
                # Preprocess data
                processed_data = self._preprocess_data(data)
                
                # Compute metrics
                metrics = self._compute_metrics_impl(processed_data)
                
                # Postprocess results
                final_metrics = self._postprocess_results(metrics)
                
                return final_metrics
                
            except Exception as e:
                raise MetricComputationError(
                    f"Metric computation failed in strategy '{self.metadata.name}': {e}",
                    metric_name=self.metadata.name,
                    computation_type="metric_computation",
                    context=create_error_context(
                        "metric_computation",
                        strategy=self.metadata.name,
                        data_keys=list(data.keys()) if data else []
                    )
                ) from e


class RenderingStrategy(BaseStrategy[str]):
    """
    Abstract base class for progress bar rendering strategies.
    
    This class defines the interface for strategies that render
    progress bars and training information to the terminal or
    other output devices.
    
    Rendering strategies handle:
    - Progress bar formatting and layout
    - Color and styling application
    - Terminal compatibility checks
    - Dynamic content updates
    - Multi-line display management
    
    The rendering process follows these steps:
    1. Check terminal compatibility
    2. Format progress information
    3. Apply styling and colors
    4. Generate output string
    5. Handle terminal-specific optimizations
    """
    
    @abstractmethod
    def _check_terminal_compatibility(self) -> bool:
        """
        Check if the current terminal supports this rendering strategy.
        
        Returns:
            True if terminal is compatible, False otherwise
        """
        pass
    
    @abstractmethod
    def _format_progress_info(
        self,
        current: int,
        total: int,
        metrics: MetricDict,
        elapsed_time: float
    ) -> Dict[str, str]:
        """
        Format progress information for display.
        
        Args:
            current: Current progress value
            total: Total progress value
            metrics: Current training metrics
            elapsed_time: Elapsed time in seconds
            
        Returns:
            Dictionary of formatted display elements
        """
        pass
    
    @abstractmethod
    def _apply_styling(self, formatted_info: Dict[str, str]) -> Dict[str, str]:
        """
        Apply styling and colors to formatted information.
        
        Args:
            formatted_info: Formatted display elements
            
        Returns:
            Styled display elements
        """
        pass
    
    @abstractmethod
    def _generate_output_impl(self, styled_info: Dict[str, str]) -> str:
        """
        Generate the final output string for display.
        
        Args:
            styled_info: Styled display elements
            
        Returns:
            Final output string ready for display
        """
        pass
    
    def execute(
        self,
        current: int,
        total: int,
        metrics: MetricDict,
        elapsed_time: float
    ) -> str:
        """
        Execute rendering strategy to generate progress display.
        
        Args:
            current: Current progress value
            total: Total progress value
            metrics: Current training metrics
            elapsed_time: Elapsed time in seconds
            
        Returns:
            Formatted progress bar string
            
        Raises:
            RendererError: If rendering fails
        """
        with self._execution_context():
            try:
                # Check terminal compatibility
                if not self._check_terminal_compatibility():
                    raise RendererError(
                        f"Terminal not compatible with renderer '{self.metadata.name}'",
                        renderer_type=self.metadata.name
                    )
                
                # Format progress information
                formatted_info = self._format_progress_info(
                    current, total, metrics, elapsed_time
                )
                
                # Apply styling
                styled_info = self._apply_styling(formatted_info)
                
                # Generate output
                output = self._generate_output_impl(styled_info)
                
                return output
                
            except Exception as e:
                raise RendererError(
                    f"Rendering failed in strategy '{self.metadata.name}': {e}",
                    renderer_type=self.metadata.name,
                    context=create_error_context(
                        "progress_rendering",
                        current=current,
                        total=total,
                        metrics_count=len(metrics)
                    )
                ) from e


class AggregationStrategy(BaseStrategy[MetricDict]):
    """
    Abstract base class for metric aggregation strategies.
    
    This class defines the interface for strategies that aggregate
    and smooth metrics over time, providing stable and meaningful
    progress indicators.
    
    Aggregation strategies handle:
    - Moving averages and smoothing
    - Outlier detection and filtering
    - Trend analysis and prediction
    - Statistical aggregation methods
    - Time-based windowing
    
    Common aggregation methods:
    - Simple moving average
    - Exponential moving average
    - Weighted moving average
    - Median filtering
    - Percentile-based aggregation
    """
    
    @abstractmethod
    def _initialize_buffers(self) -> None:
        """
        Initialize internal buffers for metric aggregation.
        
        This method should set up any data structures needed
        for storing historical metric values.
        """
        pass
    
    @abstractmethod
    def _update_buffers(self, metrics: MetricDict) -> None:
        """
        Update internal buffers with new metric values.
        
        Args:
            metrics: New metric values to add to buffers
        """
        pass
    
    @abstractmethod
    def _compute_aggregated_metrics(self) -> MetricDict:
        """
        Compute aggregated metrics from buffered values.
        
        Returns:
            Dictionary of aggregated metrics
        """
        pass
    
    def _cleanup_old_data(self) -> None:
        """
        Clean up old data from buffers to prevent memory growth.
        
        This method should remove old metric values that are
        no longer needed for aggregation.
        """
        pass
    
    def execute(self, metrics: MetricDict) -> MetricDict:
        """
        Execute aggregation strategy to smooth metrics.
        
        Args:
            metrics: New metric values to aggregate
            
        Returns:
            Aggregated and smoothed metrics
        """
        with self._execution_context():
            try:
                # Update buffers with new metrics
                self._update_buffers(metrics)
                
                # Compute aggregated metrics
                aggregated = self._compute_aggregated_metrics()
                
                # Clean up old data
                self._cleanup_old_data()
                
                return aggregated
                
            except Exception as e:
                raise MetricComputationError(
                    f"Metric aggregation failed in strategy '{self.metadata.name}': {e}",
                    metric_name="aggregated_metrics",
                    computation_type="metric_aggregation",
                    context=create_error_context(
                        "metric_aggregation",
                        strategy=self.metadata.name,
                        input_metrics=list(metrics.keys())
                    )
                ) from e


class ValidationStrategy(BaseStrategy[bool]):
    """
    Abstract base class for input validation strategies.
    
    This class defines the interface for strategies that validate
    input data, configuration parameters, and system state before
    processing.
    
    Validation strategies ensure:
    - Data type correctness
    - Value range validation
    - Structural integrity checks
    - Dependency verification
    - Security constraint enforcement
    """
    
    @abstractmethod
    def _validate_data_types(self, data: Any) -> List[str]:
        """
        Validate data types in input.
        
        Args:
            data: Input data to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        pass
    
    @abstractmethod
    def _validate_value_ranges(self, data: Any) -> List[str]:
        """
        Validate value ranges in input.
        
        Args:
            data: Input data to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        pass
    
    @abstractmethod
    def _validate_structure(self, data: Any) -> List[str]:
        """
        Validate data structure and format.
        
        Args:
            data: Input data to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        pass
    
    def execute(self, data: Any) -> bool:
        """
        Execute validation strategy on input data.
        
        Args:
            data: Input data to validate
            
        Returns:
            True if validation passes, False otherwise
            
        Raises:
            ConfigurationError: If validation fails with errors
        """
        with self._execution_context():
            errors = []
            
            # Collect all validation errors
            errors.extend(self._validate_data_types(data))
            errors.extend(self._validate_value_ranges(data))
            errors.extend(self._validate_structure(data))
            
            if errors:
                raise ConfigurationError(
                    f"Validation failed: {'; '.join(errors)}",
                    context=create_error_context(
                        "input_validation",
                        strategy=self.metadata.name,
                        error_count=len(errors)
                    )
                )
            
            return True


# Strategy registry for dynamic strategy selection
class StrategyRegistry:
    """
    Registry for managing strategy implementations.
    
    This class provides a centralized registry for all strategy
    implementations, allowing for dynamic strategy selection and
    configuration at runtime.
    
    Features:
    - Dynamic strategy registration
    - Priority-based strategy selection
    - Compatibility checking
    - Performance monitoring
    - Fallback strategy support
    """
    
    def __init__(self):
        """Initialize strategy registry."""
        self._strategies: Dict[str, Dict[str, BaseStrategy]] = {
            'metric_computation': {},
            'rendering': {},
            'aggregation': {},
            'validation': {},
        }
        self._fallback_strategies: Dict[str, BaseStrategy] = {}
        self._lock = threading.RLock()
    
    def register_strategy(
        self,
        category: str,
        name: str,
        strategy: BaseStrategy,
        is_fallback: bool = False
    ) -> None:
        """
        Register a strategy implementation.
        
        Args:
            category: Strategy category
            name: Strategy name
            strategy: Strategy implementation
            is_fallback: Whether this is a fallback strategy
        """
        with self._lock:
            if category not in self._strategies:
                self._strategies[category] = {}
            
            self._strategies[category][name] = strategy
            
            if is_fallback:
                self._fallback_strategies[category] = strategy
    
    def get_strategy(
        self,
        category: str,
        name: Optional[str] = None,
        priority: Optional[StrategyPriority] = None
    ) -> Optional[BaseStrategy]:
        """
        Get a strategy implementation.
        
        Args:
            category: Strategy category
            name: Specific strategy name (optional)
            priority: Minimum priority level (optional)
            
        Returns:
            Strategy implementation or None if not found
        """
        with self._lock:
            if category not in self._strategies:
                return None
            
            strategies = self._strategies[category]
            
            if name:
                return strategies.get(name)
            
            # Select best strategy by priority
            best_strategy = None
            best_priority = StrategyPriority.LOW
            
            for strategy in strategies.values():
                if priority and strategy.metadata.priority.value < priority.value:
                    continue
                
                if strategy.metadata.priority.value > best_priority.value:
                    best_strategy = strategy
                    best_priority = strategy.metadata.priority
            
            return best_strategy or self._fallback_strategies.get(category)
    
    def list_strategies(self, category: Optional[str] = None) -> Dict[str, List[str]]:
        """
        List all registered strategies.
        
        Args:
            category: Specific category to list (optional)
            
        Returns:
            Dictionary mapping categories to strategy names
        """
        with self._lock:
            if category:
                return {category: list(self._strategies.get(category, {}).keys())}
            
            return {
                cat: list(strategies.keys())
                for cat, strategies in self._strategies.items()
            }


# Global strategy registry instance
strategy_registry = StrategyRegistry()


# Export all base classes and utilities
__all__ = [
    'StrategyPriority',
    'ComputationComplexity',
    'StrategyMetadata',
    'BaseStrategy',
    'MetricComputationStrategy',
    'RenderingStrategy',
    'AggregationStrategy',
    'ValidationStrategy',
    'StrategyRegistry',
    'strategy_registry',
    'MetricValue',
    'MetricDict',
]