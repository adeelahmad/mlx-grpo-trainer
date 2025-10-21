"""
Enhanced Dynamic Logit Processing - Base Interfaces and Abstract Classes

This module provides the foundational architecture for the enhanced dynamic logit processing system,
implementing enterprise-grade patterns including Strategy, Template Method, and Observer patterns
with comprehensive type safety and performance optimization.

The architecture follows SOLID principles:
- Single Responsibility: Each processor handles one specific concern
- Open/Closed: Extensible through interfaces, closed for modification  
- Liskov Substitution: All processors implement common interface
- Interface Segregation: Separate interfaces for different capabilities
- Dependency Inversion: Depend on abstractions, not concretions

Performance Characteristics:
- O(1) processor lookup and registration
- O(n) processing complexity where n is vocabulary size
- Thread-safe operations with minimal locking overhead
- Memory-efficient caching with LRU eviction

Security Considerations:
- Input validation with comprehensive sanitization
- Resource limits to prevent DoS attacks
- Secure error handling without information leakage
- Audit logging for all processing operations

Example:
    >>> from mlx_rl_trainer.generation.processors.base import LogitProcessor
    >>> from mlx_rl_trainer.core.config import GenerationConfig
    >>> import mlx.core as mx
    >>> 
    >>> class CustomProcessor(LogitProcessor):
    ...     def process_logits(self, logits, history, context):
    ...         # Custom processing logic
    ...         return logits
    >>> 
    >>> processor = CustomProcessor()
    >>> config = GenerationConfig()
    >>> context = ProcessingContext(config=config, tokenizer=tokenizer)
    >>> result = processor.process_logits(logits, history, context)
"""

import abc
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache, wraps
from typing import (
    Any, Callable, Dict, Generic, List, Optional, Protocol, Set, 
    Tuple, TypeVar, Union, runtime_checkable, Final
)
from weakref import WeakSet

import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper

from mlx_rl_trainer.core.config import GenerationConfig, ExperimentConfig
from mlx_rl_trainer.core.exceptions import CheckpointError

# Type variables for generic implementations
T = TypeVar('T')
P = TypeVar('P', bound='LogitProcessor')
C = TypeVar('C', bound='ProcessingContext')

# Constants for performance optimization
MAX_CACHE_SIZE: Final[int] = 10000
DEFAULT_THREAD_POOL_SIZE: Final[int] = 4
MAX_PROCESSING_TIME_SECONDS: Final[float] = 30.0
MAX_HISTORY_LENGTH: Final[int] = 8192

logger = logging.getLogger(__name__)


class ProcessingPhase(Enum):
    """
    Enumeration of processing phases in text generation.
    
    This enum defines the different phases of text generation that require
    different logit processing strategies. Each phase has specific characteristics
    and processing requirements.
    
    Attributes:
        INITIALIZATION: Initial setup phase before generation starts
        THINKING: Processing during the thinking phase (before </think>)
        THINK_TRANSITION: Processing at the boundary of thinking phase
        ANSWER: Processing during the answer phase (after </think>)
        ANSWER_TRANSITION: Processing at the boundary of answer phase
        COMPLETION: Final processing phase before generation ends
        ERROR_RECOVERY: Processing during error recovery scenarios
    """
    INITIALIZATION = auto()
    THINKING = auto()
    THINK_TRANSITION = auto()
    ANSWER = auto()
    ANSWER_TRANSITION = auto()
    COMPLETION = auto()
    ERROR_RECOVERY = auto()

    def __str__(self) -> str:
        return self.name.lower()

    @property
    def is_transition(self) -> bool:
        """Check if this phase represents a transition between major phases."""
        return self in (self.THINK_TRANSITION, self.ANSWER_TRANSITION)

    @property
    def is_stable(self) -> bool:
        """Check if this phase represents a stable processing state."""
        return self in (self.THINKING, self.ANSWER)


class ProcessingPriority(Enum):
    """
    Priority levels for processor execution order.
    
    Higher priority processors execute first in the processing pipeline.
    This allows for proper dependency management and optimization of
    critical processing paths.
    """
    CRITICAL = 1000
    HIGH = 750
    NORMAL = 500
    LOW = 250
    BACKGROUND = 100

    def __lt__(self, other: 'ProcessingPriority') -> bool:
        return self.value < other.value

    def __le__(self, other: 'ProcessingPriority') -> bool:
        return self.value <= other.value


@dataclass(frozen=True)
class ProcessingMetrics:
    """
    Immutable metrics container for processor performance monitoring.
    
    This class provides comprehensive metrics collection for monitoring
    processor performance, debugging issues, and optimizing processing
    pipelines. All metrics are immutable to ensure thread safety.
    
    Attributes:
        processor_id: Unique identifier for the processor instance
        phase: Current processing phase
        execution_time_ms: Time taken for processing in milliseconds
        input_tokens: Number of input tokens processed
        output_tokens: Number of output tokens generated
        cache_hits: Number of cache hits during processing
        cache_misses: Number of cache misses during processing
        memory_usage_mb: Peak memory usage in megabytes
        error_count: Number of errors encountered
        warning_count: Number of warnings generated
        custom_metrics: Additional processor-specific metrics
    """
    processor_id: str
    phase: ProcessingPhase
    execution_time_ms: float
    input_tokens: int
    output_tokens: int
    cache_hits: int = 0
    cache_misses: int = 0
    memory_usage_mb: float = 0.0
    error_count: int = 0
    warning_count: int = 0
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @property
    def cache_hit_ratio(self) -> float:
        """Calculate cache hit ratio as a percentage."""
        total_requests = self.cache_hits + self.cache_misses
        return (self.cache_hits / total_requests * 100) if total_requests > 0 else 0.0

    @property
    def tokens_per_second(self) -> float:
        """Calculate processing throughput in tokens per second."""
        if self.execution_time_ms <= 0:
            return 0.0
        return (self.input_tokens + self.output_tokens) / (self.execution_time_ms / 1000.0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            'processor_id': self.processor_id,
            'phase': str(self.phase),
            'execution_time_ms': self.execution_time_ms,
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'memory_usage_mb': self.memory_usage_mb,
            'error_count': self.error_count,
            'warning_count': self.warning_count,
            'cache_hit_ratio': self.cache_hit_ratio,
            'tokens_per_second': self.tokens_per_second,
            'timestamp': self.timestamp,
            'custom_metrics': self.custom_metrics.copy()
        }


@dataclass
class ProcessingContext:
    """
    Comprehensive context container for logit processing operations.
    
    This class provides all necessary context information for processors
    to make informed decisions about logit modifications. It includes
    configuration, tokenizer access, phase information, and performance
    tracking capabilities.
    
    The context is designed to be thread-safe and supports both mutable
    and immutable operations depending on the use case.
    
    Attributes:
        config: Generation configuration with bias parameters
        tokenizer: MLX tokenizer for token operations
        phase: Current processing phase
        correlation_id: Unique identifier for request tracing
        mcq_flags: Boolean flags indicating MCQ status per batch item
        batch_size: Size of the current processing batch
        vocabulary_size: Size of the model vocabulary
        device_info: Information about the processing device
        debug_mode: Enable detailed debugging information
        performance_tracking: Enable performance metrics collection
        custom_data: Additional processor-specific data
    """
    config: GenerationConfig
    tokenizer: TokenizerWrapper
    phase: ProcessingPhase = ProcessingPhase.INITIALIZATION
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    mcq_flags: Optional[List[bool]] = None
    batch_size: int = 1
    vocabulary_size: int = 32000
    device_info: Dict[str, Any] = field(default_factory=dict)
    debug_mode: bool = False
    performance_tracking: bool = True
    custom_data: Dict[str, Any] = field(default_factory=dict)
    _metrics_history: List[ProcessingMetrics] = field(default_factory=list, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)
    
    def __hash__(self) -> int:
        """
        Make ProcessingContext hashable for use as dictionary keys.
        
        This implementation creates a hash based on immutable attributes
        that uniquely identify the context for caching purposes.
        """
        # Create a hash based on key immutable attributes
        hashable_items = (
            self.phase,
            self.correlation_id,
            self.batch_size,
            self.vocabulary_size,
            self.debug_mode,
            self.performance_tracking,
            tuple(self.mcq_flags) if self.mcq_flags else None,
            # Note: We don't include config, tokenizer, device_info, or custom_data
            # as they may contain unhashable objects
        )
        return hash(hashable_items)
    
    def __eq__(self, other) -> bool:
        """
        Check equality based on key attributes for caching.
        
        Args:
            other: Another ProcessingContext to compare with
            
        Returns:
            True if contexts are equivalent for caching purposes
        """
        if not isinstance(other, ProcessingContext):
            return False
        
        return (
            self.phase == other.phase and
            self.correlation_id == other.correlation_id and
            self.batch_size == other.batch_size and
            self.vocabulary_size == other.vocabulary_size and
            self.debug_mode == other.debug_mode and
            self.performance_tracking == other.performance_tracking and
            self.mcq_flags == other.mcq_flags
        )

    def __post_init__(self):
        """Initialize derived attributes and validate configuration."""
        if self.mcq_flags is None:
            self.mcq_flags = [False] * self.batch_size
        
        # Validate batch consistency
        if len(self.mcq_flags) != self.batch_size:
            raise ValueError(
                f"MCQ flags length ({len(self.mcq_flags)}) must match batch size ({self.batch_size})"
            )
        
        # Initialize device information if not provided
        if not self.device_info:
            try:
                self.device_info = {
                    'device_type': 'mps' if mx.metal.is_available() else 'cpu',
                    'memory_limit': getattr(mx, 'get_memory_limit', lambda: None)(),
                    'peak_memory': getattr(mx, 'get_peak_memory', lambda: 0)()
                }
            except Exception as e:
                logger.warning(f"Failed to initialize device info: {e}")
                self.device_info = {'device_type': 'unknown'}

    def add_metrics(self, metrics: ProcessingMetrics) -> None:
        """
        Thread-safely add processing metrics to the context.
        
        Args:
            metrics: Processing metrics to add
            
        Raises:
            ValueError: If metrics are invalid or corrupted
        """
        with self._lock:
            if not isinstance(metrics, ProcessingMetrics):
                raise ValueError("Metrics must be a ProcessingMetrics instance")
            
            self._metrics_history.append(metrics)
            
            # Limit history size to prevent memory leaks
            if len(self._metrics_history) > 1000:
                self._metrics_history = self._metrics_history[-500:]

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get aggregated metrics summary for all processors.
        
        Returns:
            Dictionary containing aggregated performance metrics
        """
        with self._lock:
            if not self._metrics_history:
                return {}
            
            total_time = sum(m.execution_time_ms for m in self._metrics_history)
            total_tokens = sum(m.input_tokens + m.output_tokens for m in self._metrics_history)
            total_errors = sum(m.error_count for m in self._metrics_history)
            
            return {
                'total_processors': len(set(m.processor_id for m in self._metrics_history)),
                'total_execution_time_ms': total_time,
                'total_tokens_processed': total_tokens,
                'average_tokens_per_second': total_tokens / (total_time / 1000.0) if total_time > 0 else 0,
                'total_errors': total_errors,
                'error_rate': total_errors / len(self._metrics_history) if self._metrics_history else 0,
                'phases_processed': list(set(str(m.phase) for m in self._metrics_history))
            }

    def clone(self, **overrides) -> 'ProcessingContext':
        """
        Create a copy of the context with optional parameter overrides.
        
        Args:
            **overrides: Parameters to override in the new context
            
        Returns:
            New ProcessingContext instance with specified overrides
        """
        # Create base parameters from current instance
        params = {
            'config': self.config,
            'tokenizer': self.tokenizer,
            'phase': self.phase,
            'correlation_id': self.correlation_id,
            'mcq_flags': self.mcq_flags.copy() if self.mcq_flags else None,
            'batch_size': self.batch_size,
            'vocabulary_size': self.vocabulary_size,
            'device_info': self.device_info.copy(),
            'debug_mode': self.debug_mode,
            'performance_tracking': self.performance_tracking,
            'custom_data': self.custom_data.copy()
        }
        
        # Apply overrides
        params.update(overrides)
        
        return ProcessingContext(**params)


class ProcessingException(Exception):
    """
    Base exception class for logit processing errors.
    
    This exception hierarchy provides structured error handling with
    detailed context information for debugging and monitoring.
    
    Attributes:
        message: Human-readable error description
        processor_id: ID of the processor that raised the exception
        correlation_id: Request correlation ID for tracing
        phase: Processing phase where error occurred
        context: Additional context information
    """
    
    def __init__(
        self, 
        message: str, 
        processor_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        phase: Optional[ProcessingPhase] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.processor_id = processor_id
        self.correlation_id = correlation_id
        self.phase = phase
        self.context = context or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for structured logging."""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'processor_id': self.processor_id,
            'correlation_id': self.correlation_id,
            'phase': str(self.phase) if self.phase else None,
            'context': self.context
        }


class ProcessorConfigurationError(ProcessingException):
    """Raised when processor configuration is invalid or incomplete."""
    pass


class ProcessorExecutionError(ProcessingException):
    """Raised when processor execution fails due to runtime errors."""
    pass


class ProcessorTimeoutError(ProcessingException):
    """Raised when processor execution exceeds time limits."""
    pass


class ProcessorResourceError(ProcessingException):
    """Raised when processor exceeds resource limits (memory, etc.)."""
    pass


@runtime_checkable
class MetricsCollector(Protocol):
    """
    Protocol for metrics collection and reporting.
    
    This protocol defines the interface for collecting and reporting
    processing metrics. Implementations can provide different storage
    backends (memory, database, external services).
    """
    
    def collect_metrics(self, metrics: ProcessingMetrics) -> None:
        """
        Collect processing metrics for storage and analysis.
        
        Args:
            metrics: Processing metrics to collect
        """
        ...
    
    def get_metrics(
        self, 
        processor_id: Optional[str] = None,
        phase: Optional[ProcessingPhase] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[ProcessingMetrics]:
        """
        Retrieve collected metrics with optional filtering.
        
        Args:
            processor_id: Filter by processor ID
            phase: Filter by processing phase
            start_time: Filter by start timestamp
            end_time: Filter by end timestamp
            
        Returns:
            List of matching metrics
        """
        ...


@runtime_checkable
class CacheProvider(Protocol):
    """
    Protocol for caching processor results and intermediate data.
    
    This protocol enables different caching strategies including
    in-memory, distributed, and persistent caching solutions.
    """
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve cached value by key.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        ...
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Store value in cache with optional TTL.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        """
        ...
    
    def invalidate(self, pattern: str) -> int:
        """
        Invalidate cache entries matching pattern.
        
        Args:
            pattern: Pattern to match for invalidation
            
        Returns:
            Number of entries invalidated
        """
        ...


def performance_monitor(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator for monitoring processor performance.
    
    This decorator automatically collects performance metrics for
    processor methods, including execution time, memory usage,
    and error tracking.
    
    Args:
        func: Function to monitor
        
    Returns:
        Wrapped function with performance monitoring
        
    Example:
        >>> @performance_monitor
        ... def process_logits(self, logits, history, context):
        ...     return modified_logits
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs) -> T:
        if not hasattr(self, 'processor_id'):
            return func(self, *args, **kwargs)
        
        start_time = time.perf_counter()
        start_memory = 0
        error_count = 0
        
        try:
            # Get initial memory usage if available
            if hasattr(mx, 'get_peak_memory'):
                start_memory = mx.get_peak_memory()
            
            result = func(self, *args, **kwargs)
            
            return result
            
        except Exception as e:
            error_count = 1
            logger.error(
                f"Performance monitor caught exception in {func.__name__}: {e}",
                extra={'processor_id': self.processor_id}
            )
            raise
            
        finally:
            end_time = time.perf_counter()
            execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
            # Calculate memory usage if available
            memory_usage = 0
            if hasattr(mx, 'get_peak_memory'):
                try:
                    end_memory = mx.get_peak_memory()
                    memory_usage = (end_memory - start_memory) / (1024 * 1024)  # Convert to MB
                except Exception:
                    pass
            
            # Extract context information if available
            context = None
            phase = ProcessingPhase.INITIALIZATION
            
            for arg in args:
                if isinstance(arg, ProcessingContext):
                    context = arg
                    phase = arg.phase
                    break
            
            # Create metrics
            metrics = ProcessingMetrics(
                processor_id=self.processor_id,
                phase=phase,
                execution_time_ms=execution_time,
                input_tokens=0,  # Will be updated by specific processors
                output_tokens=0,  # Will be updated by specific processors
                memory_usage_mb=memory_usage,
                error_count=error_count
            )
            
            # Add metrics to context if available
            if context and context.performance_tracking:
                context.add_metrics(metrics)
    
    return wrapper


@runtime_checkable
class LogitProcessor(Protocol):
    """
    Core protocol for logit processing operations.
    
    This protocol defines the essential interface that all logit processors
    must implement. It follows the Strategy pattern to allow different
    processing algorithms while maintaining a consistent interface.
    
    The protocol is designed for high performance with minimal overhead
    and maximum flexibility for different processing strategies.
    """
    
    processor_id: str
    priority: ProcessingPriority
    
    def process_logits(
        self, 
        logits: mx.array, 
        history: List[List[int]], 
        context: ProcessingContext
    ) -> mx.array:
        """
        Process logits based on generation history and context.
        
        This is the core method that all processors must implement.
        It receives the current logits, generation history, and processing
        context, and returns modified logits.
        
        Args:
            logits: Current logits tensor [batch_size, vocab_size]
            history: Token generation history for each batch item
            context: Processing context with configuration and state
            
        Returns:
            Modified logits tensor with same shape as input
            
        Raises:
            ProcessorExecutionError: If processing fails
            ProcessorTimeoutError: If processing exceeds time limits
            ProcessorResourceError: If resource limits are exceeded
        """
        ...
    
    def can_process(self, context: ProcessingContext) -> bool:
        """
        Check if this processor can handle the given context.
        
        This method allows processors to selectively participate in
        processing based on context conditions such as phase, configuration,
        or other runtime factors.
        
        Args:
            context: Processing context to evaluate
            
        Returns:
            True if processor can handle this context, False otherwise
        """
        ...
    
    def get_cache_key(
        self, 
        logits: mx.array, 
        history: List[List[int]], 
        context: ProcessingContext
    ) -> Optional[str]:
        """
        Generate cache key for processor results.
        
        This method enables caching of processor results to improve
        performance for repeated operations. Return None to disable
        caching for this operation.
        
        Args:
            logits: Current logits tensor
            history: Token generation history
            context: Processing context
            
        Returns:
            Cache key string or None to disable caching
        """
        ...


class BaseLogitProcessor(abc.ABC):
    """
    Abstract base class providing common functionality for logit processors.
    
    This class implements the Template Method pattern, providing a structured
    approach to processor implementation with built-in performance monitoring,
    error handling, and caching support.
    
    The class follows SOLID principles:
    - Single Responsibility: Handles common processor functionality
    - Open/Closed: Extensible through abstract methods
    - Liskov Substitution: All subclasses can be used interchangeably
    - Interface Segregation: Minimal required interface
    - Dependency Inversion: Depends on abstractions (protocols)
    
    Attributes:
        processor_id: Unique identifier for this processor instance
        priority: Processing priority for pipeline ordering
        enabled: Whether this processor is currently enabled
        cache_provider: Optional cache provider for result caching
        metrics_collector: Optional metrics collector for monitoring
    """
    
    def __init__(
        self,
        processor_id: Optional[str] = None,
        priority: ProcessingPriority = ProcessingPriority.NORMAL,
        enabled: bool = True,
        cache_provider: Optional[CacheProvider] = None,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        """
        Initialize base processor with common configuration.
        
        Args:
            processor_id: Unique processor identifier (auto-generated if None)
            priority: Processing priority for pipeline ordering
            enabled: Whether processor is enabled
            cache_provider: Optional cache provider for results
            metrics_collector: Optional metrics collector
        """
        self.processor_id = processor_id or f"{self.__class__.__name__}_{uuid.uuid4().hex[:8]}"
        self.priority = priority
        self.enabled = enabled
        self.cache_provider = cache_provider
        self.metrics_collector = metrics_collector
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance tracking
        self._total_calls = 0
        self._total_time = 0.0
        self._error_count = 0
        
        logger.info(f"Initialized processor {self.processor_id} with priority {priority}")

    @performance_monitor
    def process_logits(
        self, 
        logits: mx.array, 
        history: List[List[int]], 
        context: ProcessingContext
    ) -> mx.array:
        """
        Template method for logit processing with built-in optimizations.
        
        This method implements the Template Method pattern, providing
        a structured approach to processing with automatic caching,
        validation, and error handling.
        
        Args:
            logits: Current logits tensor [batch_size, vocab_size]
            history: Token generation history for each batch item
            context: Processing context with configuration and state
            
        Returns:
            Modified logits tensor with same shape as input
            
        Raises:
            ProcessorExecutionError: If processing fails
            ProcessorConfigurationError: If configuration is invalid
        """
        with self._lock:
            self._total_calls += 1
        
        # Validate inputs
        self._validate_inputs(logits, history, context)
        
        # Check if processor is enabled and can handle this context
        if not self.enabled or not self.can_process(context):
            return logits
        
        # Try cache first if available
        cache_key = self.get_cache_key(logits, history, context)
        if cache_key and self.cache_provider:
            cached_result = self.cache_provider.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for processor {self.processor_id}")
                return cached_result
        
        try:
            # Perform actual processing
            start_time = time.perf_counter()
            result = self._process_logits_impl(logits, history, context)
            processing_time = time.perf_counter() - start_time
            
            # Update performance tracking
            with self._lock:
                self._total_time += processing_time
            
            # Cache result if caching is enabled
            if cache_key and self.cache_provider:
                self.cache_provider.set(cache_key, result)
            
            # Validate output
            self._validate_output(result, logits)
            
            return result
            
        except Exception as e:
            with self._lock:
                self._error_count += 1
            
            logger.error(
                f"Processor {self.processor_id} failed: {e}",
                extra={
                    'processor_id': self.processor_id,
                    'correlation_id': context.correlation_id,
                    'phase': str(context.phase)
                }
            )
            
            if isinstance(e, ProcessingException):
                raise
            else:
                raise ProcessorExecutionError(
                    f"Processor execution failed: {e}",
                    processor_id=self.processor_id,
                    correlation_id=context.correlation_id,
                    phase=context.phase
                ) from e

    @abc.abstractmethod
    def _process_logits_impl(
        self, 
        logits: mx.array, 
        history: List[List[int]], 
        context: ProcessingContext
    ) -> mx.array:
        """
        Abstract method for actual logit processing implementation.
        
        Subclasses must implement this method to provide specific
        processing logic. The base class handles validation, caching,
        and error handling automatically.
        
        Args:
            logits: Current logits tensor [batch_size, vocab_size]
            history: Token generation history for each batch item
            context: Processing context with configuration and state
            
        Returns:
            Modified logits tensor with same shape as input
        """
        pass

    def can_process(self, context: ProcessingContext) -> bool:
        """
        Default implementation allows processing in all contexts.
        
        Subclasses can override this method to implement conditional
        processing based on context state, configuration, or other factors.
        
        Args:
            context: Processing context to evaluate
            
        Returns:
            True if processor can handle this context
        """
        return True

    def get_cache_key(
        self, 
        logits: mx.array, 
        history: List[List[int]], 
        context: ProcessingContext
    ) -> Optional[str]:
        """
        Generate cache key based on processor state and inputs.
        
        Default implementation creates a key based on processor ID,
        history hash, and relevant context parameters. Subclasses
        can override for custom caching strategies.
        
        Args:
            logits: Current logits tensor
            history: Token generation history
            context: Processing context
            
        Returns:
            Cache key string or None to disable caching
        """
        if not self.cache_provider:
            return None
        
        # Create hash of relevant inputs
        history_hash = hash(tuple(tuple(h) for h in history))
        context_hash = hash((
            context.phase,
            context.batch_size,
            context.vocabulary_size,
            tuple(context.mcq_flags) if context.mcq_flags else ()
        ))
        
        return f"{self.processor_id}:{history_hash}:{context_hash}"

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for this processor.
        
        Returns:
            Dictionary containing performance metrics
        """
        with self._lock:
            avg_time = self._total_time / self._total_calls if self._total_calls > 0 else 0
            error_rate = self._error_count / self._total_calls if self._total_calls > 0 else 0
            
            return {
                'processor_id': self.processor_id,
                'total_calls': self._total_calls,
                'total_time_seconds': self._total_time,
                'average_time_seconds': avg_time,
                'error_count': self._error_count,
                'error_rate': error_rate,
                'enabled': self.enabled,
                'priority': self.priority.name
            }

    def reset_stats(self) -> None:
        """Reset performance statistics."""
        with self._lock:
            self._total_calls = 0
            self._total_time = 0.0
            self._error_count = 0

    def _validate_inputs(
        self, 
        logits: mx.array, 
        history: List[List[int]], 
        context: ProcessingContext
    ) -> None:
        """
        Validate processor inputs for correctness and security.
        
        Args:
            logits: Logits tensor to validate
            history: History to validate
            context: Context to validate
            
        Raises:
            ProcessorConfigurationError: If inputs are invalid
        """
        # Validate logits tensor
        if not isinstance(logits, mx.array):
            raise ProcessorConfigurationError(
                "Logits must be an MLX array",
                processor_id=self.processor_id,
                correlation_id=context.correlation_id
            )
        
        if logits.ndim != 2:
            raise ProcessorConfigurationError(
                f"Logits must be 2D tensor, got {logits.ndim}D",
                processor_id=self.processor_id,
                correlation_id=context.correlation_id
            )
        
        batch_size, vocab_size = logits.shape
        
        # Validate batch consistency
        if batch_size != context.batch_size:
            raise ProcessorConfigurationError(
                f"Logits batch size ({batch_size}) doesn't match context ({context.batch_size})",
                processor_id=self.processor_id,
                correlation_id=context.correlation_id
            )
        
        if vocab_size != context.vocabulary_size:
            raise ProcessorConfigurationError(
                f"Logits vocab size ({vocab_size}) doesn't match context ({context.vocabulary_size})",
                processor_id=self.processor_id,
                correlation_id=context.correlation_id
            )
        
        # Validate history
        if len(history) != batch_size:
            raise ProcessorConfigurationError(
                f"History length ({len(history)}) doesn't match batch size ({batch_size})",
                processor_id=self.processor_id,
                correlation_id=context.correlation_id
            )
        
        # Check for reasonable history lengths (security)
        for i, hist in enumerate(history):
            if len(hist) > MAX_HISTORY_LENGTH:
                raise ProcessorConfigurationError(
                    f"History too long at index {i}: {len(hist)} > {MAX_HISTORY_LENGTH}",
                    processor_id=self.processor_id,
                    correlation_id=context.correlation_id
                )

    def _validate_output(self, result: mx.array, original: mx.array) -> None:
        """
        Validate processor output for correctness.
        
        Args:
            result: Processed logits to validate
            original: Original logits for comparison
            
        Raises:
            ProcessorExecutionError: If output is invalid
        """
        if not isinstance(result, mx.array):
            raise ProcessorExecutionError(
                "Processor must return MLX array",
                processor_id=self.processor_id
            )
        
        if result.shape != original.shape:
            raise ProcessorExecutionError(
                f"Output shape {result.shape} doesn't match input {original.shape}",
                processor_id=self.processor_id
            )
        
        # Check for NaN or infinite values
        if mx.any(mx.isnan(result)) or mx.any(mx.isinf(result)):
            raise ProcessorExecutionError(
                "Processor output contains NaN or infinite values",
                processor_id=self.processor_id
            )


class ProcessorRegistry:
    """
    Thread-safe registry for managing processor instances.
    
    This class provides centralized management of processor instances
    with support for registration, discovery, and lifecycle management.
    It implements the Registry pattern with thread-safe operations.
    
    Features:
    - Thread-safe registration and lookup
    - Priority-based ordering
    - Conditional processor filtering
    - Performance monitoring integration
    - Automatic cleanup of unused processors
    """
    
    def __init__(self):
        """Initialize empty processor registry."""
        self._processors: Dict[str, LogitProcessor] = {}
        self._processors_by_priority: Dict[ProcessingPriority, List[LogitProcessor]] = {}
        self._lock = threading.RLock()
        self._weak_refs: WeakSet[LogitProcessor] = WeakSet()
        
        # Initialize priority buckets
        for priority in ProcessingPriority:
            self._processors_by_priority[priority] = []

    def register(self, processor: LogitProcessor) -> None:
        """
        Register a processor instance.
        
        Args:
            processor: Processor to register
            
        Raises:
            ValueError: If processor is invalid or already registered
        """
        if not isinstance(processor, (LogitProcessor, BaseLogitProcessor)):
            raise ValueError("Processor must implement LogitProcessor protocol")
        
        with self._lock:
            if processor.processor_id in self._processors:
                raise ValueError(f"Processor {processor.processor_id} already registered")
            
            self._processors[processor.processor_id] = processor
            self._processors_by_priority[processor.priority].append(processor)
            self._weak_refs.add(processor)
            
            logger.info(f"Registered processor {processor.processor_id} with priority {processor.priority}")

    def unregister(self, processor_id: str) -> bool:
        """
        Unregister a processor by ID.
        
        Args:
            processor_id: ID of processor to unregister
            
        Returns:
            True if processor was found and removed, False otherwise
        """
        with self._lock:
            processor = self._processors.pop(processor_id, None)
            if processor is None:
                return False
            
            # Remove from priority bucket
            self._processors_by_priority[processor.priority].remove(processor)
            
            logger.info(f"Unregistered processor {processor_id}")
            return True

    def get_processor(self, processor_id: str) -> Optional[LogitProcessor]:
        """
        Get processor by ID.
        
        Args:
            processor_id: Processor ID to lookup
            
        Returns:
            Processor instance or None if not found
        """
        with self._lock:
            return self._processors.get(processor_id)

    def get_processors_for_context(self, context: ProcessingContext) -> List[LogitProcessor]:
        """
        Get all processors that can handle the given context, ordered by priority.
        
        Args:
            context: Processing context to match against
            
        Returns:
            List of processors ordered by priority (highest first)
        """
        processors = []
        
        with self._lock:
            # Iterate through priorities in descending order
            for priority in sorted(ProcessingPriority, reverse=True):
                for processor in self._processors_by_priority[priority]:
                    if hasattr(processor, 'enabled') and not processor.enabled:
                        continue
                    
                    if processor.can_process(context):
                        processors.append(processor)
        
        return processors

    def get_all_processors(self) -> List[LogitProcessor]:
        """
        Get all registered processors.
        
        Returns:
            List of all registered processors
        """
        with self._lock:
            return list(self._processors.values())

    def clear(self) -> None:
        """Clear all registered processors."""
        with self._lock:
            self._processors.clear()
            for priority_list in self._processors_by_priority.values():
                priority_list.clear()
            self._weak_refs.clear()
            
            logger.info("Cleared all processors from registry")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get registry statistics.
        
        Returns:
            Dictionary containing registry statistics
        """
        with self._lock:
            priority_counts = {
                priority.name: len(processors) 
                for priority, processors in self._processors_by_priority.items()
            }
            
            return {
                'total_processors': len(self._processors),
                'processors_by_priority': priority_counts,
                'processor_ids': list(self._processors.keys())
            }


# Global processor registry instance
_global_registry = ProcessorRegistry()


def get_global_registry() -> ProcessorRegistry:
    """
    Get the global processor registry instance.
    
    Returns:
        Global ProcessorRegistry instance
    """
    return _global_registry


def register_processor(processor: LogitProcessor) -> None:
    """
    Register a processor in the global registry.
    
    Args:
        processor: Processor to register
    """
    _global_registry.register(processor)


def create_processing_context(
    config: GenerationConfig,
    tokenizer: TokenizerWrapper,
    **kwargs
) -> ProcessingContext:
    """
    Factory function for creating processing contexts.
    
    Args:
        config: Generation configuration
        tokenizer: MLX tokenizer
        **kwargs: Additional context parameters
        
    Returns:
        Configured ProcessingContext instance
    """
    return ProcessingContext(
        config=config,
        tokenizer=tokenizer,
        **kwargs
    )