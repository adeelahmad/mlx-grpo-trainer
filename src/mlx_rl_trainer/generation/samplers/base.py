"""
Abstract sampler interfaces and protocols for the MLX RL Trainer sampling system.

This module defines the core abstractions and contracts for the sampling system,
implementing enterprise-grade design patterns and SOLID principles for maximum
extensibility and maintainability.

Architecture:
    - SamplerProtocol: Core interface for all samplers
    - AbstractSampler: Base implementation with common functionality
    - SamplerMetrics: Performance and quality metrics collection
    - SamplerRegistry: Registry pattern for sampler discovery and management
    - CircuitBreaker: Fault tolerance for external sampler dependencies

Design Patterns Applied:
    - Protocol Pattern: Type-safe interfaces without inheritance overhead
    - Template Method Pattern: Common sampler workflow with customizable steps
    - Registry Pattern: Dynamic sampler discovery and registration
    - Circuit Breaker Pattern: Fault tolerance and graceful degradation
    - Observer Pattern: Metrics collection and monitoring
    - Strategy Pattern: Different sampling strategies with common interface

SOLID Principles:
    - Single Responsibility: Each class handles one aspect of sampling
    - Open/Closed: Extensible for new sampler types without modification
    - Liskov Substitution: All samplers are interchangeable through protocols
    - Interface Segregation: Separate interfaces for different sampler concerns
    - Dependency Inversion: Depends on abstractions, not concrete implementations

Example:
    >>> from mlx_rl_trainer.generation.samplers.base import SamplerProtocol
    >>> class CustomSampler(AbstractSampler):
    ...     def _sample_token(self, logits, **kwargs):
    ...         return self._apply_temperature_sampling(logits)
    >>> sampler = CustomSampler(temperature=0.7)
    >>> token = sampler.sample(logits, tokenizer)
"""

import logging
import time
import threading
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, Dict, List, Optional, Union, Callable, Protocol, TypeVar, Generic,
    runtime_checkable, ClassVar, Final, Tuple, NamedTuple, Set
)
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, Future
import asyncio
import uuid
import json
import hashlib
from functools import wraps, lru_cache
import inspect

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.tokenizer_utils import TokenizerWrapper

# Type definitions for enhanced type safety
T = TypeVar('T')
SamplerResult = Union[mx.array, int, List[int]]
LogitsArray = mx.array
TokenizerType = TypeVar('TokenizerType', bound=TokenizerWrapper)

logger = logging.getLogger(__name__)


class SamplerState(Enum):
    """
    Enumeration of sampler operational states.
    
    Used for state management and monitoring of sampler instances.
    """
    INITIALIZING = auto()
    READY = auto()
    SAMPLING = auto()
    ERROR = auto()
    DISABLED = auto()
    MAINTENANCE = auto()


class SamplerType(Enum):
    """
    Enumeration of supported sampler types.
    
    Defines the different sampling strategies available in the system.
    """
    TEMPERATURE = "temperature"
    NUCLEUS = "nucleus"  # top_p
    TOP_K = "top_k"
    MIN_P = "min_p"
    XTC = "xtc"
    REPETITION_PENALTY = "repetition_penalty"
    HYBRID = "hybrid"
    CUSTOM = "custom"


class SamplerError(Exception):
    """
    Base exception for sampler-related errors.
    
    Provides structured error reporting with context information
    for debugging and monitoring purposes.
    """
    
    def __init__(
        self,
        message: str,
        error_code: str = "SAMPLER_ERROR",
        sampler_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.sampler_id = sampler_id
        self.context = context or {}
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.timestamp = time.time()
        
        # Log the error with full context
        logger.error(
            f"Sampler Error [{self.error_code}]: {message}",
            extra={
                "correlation_id": self.correlation_id,
                "sampler_id": self.sampler_id,
                "context": self.context,
                "timestamp": self.timestamp
            }
        )


class SamplingError(SamplerError):
    """Specific error for sampling operation failures."""
    
    def __init__(self, operation: str, reason: str, **kwargs):
        message = f"Sampling operation '{operation}' failed: {reason}"
        super().__init__(message, error_code="SAMPLING_ERROR", **kwargs)
        self.operation = operation
        self.reason = reason


class CircuitBreakerError(SamplerError):
    """Error when circuit breaker is open."""
    
    def __init__(self, sampler_id: str, failure_count: int, **kwargs):
        message = f"Circuit breaker open for sampler {sampler_id} after {failure_count} failures"
        super().__init__(message, error_code="CIRCUIT_BREAKER_OPEN", **kwargs)
        self.failure_count = failure_count


@dataclass
class SamplerMetrics:
    """
    Comprehensive metrics collection for sampler performance and quality.
    
    Tracks operational metrics, performance characteristics, and quality indicators
    for monitoring and optimization purposes.
    
    Attributes:
        sampler_id: Unique identifier for the sampler
        total_samples: Total number of samples generated
        successful_samples: Number of successful samples
        failed_samples: Number of failed samples
        average_latency_ms: Average sampling latency in milliseconds
        peak_latency_ms: Peak sampling latency
        cache_hits: Number of cache hits
        cache_misses: Number of cache misses
        error_count: Total error count
        last_error: Last error message
        created_at: Timestamp when metrics were created
        last_updated: Timestamp of last update
    """
    
    sampler_id: str
    total_samples: int = 0
    successful_samples: int = 0
    failed_samples: int = 0
    average_latency_ms: float = 0.0
    peak_latency_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    
    # Performance tracking
    _latency_history: deque = field(default_factory=lambda: deque(maxlen=1000), init=False, repr=False)
    _error_history: deque = field(default_factory=lambda: deque(maxlen=100), init=False, repr=False)
    
    def record_sample(self, latency_ms: float, success: bool, error: Optional[str] = None):
        """
        Record a sampling operation.
        
        Args:
            latency_ms: Latency of the operation in milliseconds
            success: Whether the operation was successful
            error: Error message if operation failed
        """
        self.total_samples += 1
        self.last_updated = time.time()
        
        if success:
            self.successful_samples += 1
        else:
            self.failed_samples += 1
            self.error_count += 1
            if error:
                self.last_error = error
                self._error_history.append((time.time(), error))
        
        # Update latency metrics
        self._latency_history.append(latency_ms)
        if latency_ms > self.peak_latency_ms:
            self.peak_latency_ms = latency_ms
        
        # Recalculate average latency
        if self._latency_history:
            self.average_latency_ms = sum(self._latency_history) / len(self._latency_history)
    
    def record_cache_hit(self):
        """Record a cache hit."""
        self.cache_hits += 1
    
    def record_cache_miss(self):
        """Record a cache miss."""
        self.cache_misses += 1
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as a percentage."""
        if self.total_samples == 0:
            return 0.0
        return (self.successful_samples / self.total_samples) * 100.0
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate as a percentage."""
        total_cache_ops = self.cache_hits + self.cache_misses
        if total_cache_ops == 0:
            return 0.0
        return (self.cache_hits / total_cache_ops) * 100.0
    
    def get_recent_errors(self, limit: int = 10) -> List[Tuple[float, str]]:
        """Get recent errors with timestamps."""
        return list(self._error_history)[-limit:]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            "sampler_id": self.sampler_id,
            "total_samples": self.total_samples,
            "successful_samples": self.successful_samples,
            "failed_samples": self.failed_samples,
            "success_rate": self.success_rate,
            "average_latency_ms": self.average_latency_ms,
            "peak_latency_ms": self.peak_latency_ms,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hit_rate,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
        }


class CircuitBreaker:
    """
    Circuit breaker implementation for sampler fault tolerance.
    
    Implements the Circuit Breaker pattern to provide fault tolerance
    and graceful degradation when samplers fail repeatedly.
    
    States:
        - CLOSED: Normal operation, requests pass through
        - OPEN: Failures detected, requests fail fast
        - HALF_OPEN: Testing if service has recovered
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._state = "CLOSED"
        self._lock = threading.RLock()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function through the circuit breaker.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerError: If circuit breaker is open
        """
        with self._lock:
            if self._state == "OPEN":
                if time.time() - self._last_failure_time < self.recovery_timeout:
                    raise CircuitBreakerError(
                        sampler_id=kwargs.get("sampler_id", "unknown"),
                        failure_count=self._failure_count
                    )
                else:
                    self._state = "HALF_OPEN"
                    self._success_count = 0
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise
    
    def _on_success(self):
        """Handle successful operation."""
        if self._state == "HALF_OPEN":
            self._success_count += 1
            if self._success_count >= self.success_threshold:
                self._state = "CLOSED"
                self._failure_count = 0
        elif self._state == "CLOSED":
            self._failure_count = max(0, self._failure_count - 1)
    
    def _on_failure(self):
        """Handle failed operation."""
        self._failure_count += 1
        self._last_failure_time = time.time()
        
        if self._failure_count >= self.failure_threshold:
            self._state = "OPEN"
    
    @property
    def state(self) -> str:
        """Get current circuit breaker state."""
        return self._state
    
    @property
    def failure_count(self) -> int:
        """Get current failure count."""
        return self._failure_count


@runtime_checkable
class SamplerProtocol(Protocol):
    """
    Protocol defining the interface for all samplers.
    
    This protocol ensures type safety and consistent interfaces across
    all sampler implementations while maintaining flexibility for
    different sampling strategies.
    
    Methods:
        sample: Generate a token sample from logits
        batch_sample: Generate multiple token samples efficiently
        get_metrics: Retrieve performance metrics
        configure: Update sampler configuration
        reset: Reset sampler state
    """
    
    def sample(
        self,
        logits: LogitsArray,
        tokenizer: TokenizerWrapper,
        **kwargs
    ) -> SamplerResult:
        """
        Generate a token sample from logits.
        
        Args:
            logits: Input logits array
            tokenizer: Tokenizer for special token handling
            **kwargs: Additional sampling parameters
            
        Returns:
            Sampled token(s)
        """
        ...
    
    def batch_sample(
        self,
        logits_batch: List[LogitsArray],
        tokenizer: TokenizerWrapper,
        **kwargs
    ) -> List[SamplerResult]:
        """
        Generate multiple token samples efficiently.
        
        Args:
            logits_batch: Batch of logits arrays
            tokenizer: Tokenizer for special token handling
            **kwargs: Additional sampling parameters
            
        Returns:
            List of sampled tokens
        """
        ...
    
    def get_metrics(self) -> SamplerMetrics:
        """
        Retrieve performance metrics for this sampler.
        
        Returns:
            Comprehensive metrics object
        """
        ...
    
    def configure(self, **params) -> None:
        """
        Update sampler configuration.
        
        Args:
            **params: Configuration parameters to update
        """
        ...
    
    def reset(self) -> None:
        """Reset sampler state and metrics."""
        ...


class AbstractSampler(ABC):
    """
    Abstract base class for all samplers with common functionality.
    
    Implements the Template Method pattern to provide a consistent
    sampling workflow while allowing customization of specific steps.
    
    Features:
        - Comprehensive metrics collection
        - Circuit breaker for fault tolerance
        - Performance optimization with caching
        - Thread-safe operations
        - Extensive logging and monitoring
        
    Attributes:
        sampler_id: Unique identifier for this sampler
        sampler_type: Type of sampling strategy
        state: Current operational state
        metrics: Performance metrics collector
        circuit_breaker: Fault tolerance mechanism
    """
    
    def __init__(
        self,
        sampler_id: Optional[str] = None,
        sampler_type: SamplerType = SamplerType.CUSTOM,
        enable_circuit_breaker: bool = True,
        enable_metrics: bool = True,
        **config_params
    ):
        self.sampler_id = sampler_id or f"{sampler_type.value}_{uuid.uuid4().hex[:8]}"
        self.sampler_type = sampler_type
        self.state = SamplerState.INITIALIZING
        self.config_params = config_params
        self.created_at = time.time()
        
        # Initialize metrics collection
        if enable_metrics:
            self.metrics = SamplerMetrics(sampler_id=self.sampler_id)
        else:
            self.metrics = None
        
        # Initialize circuit breaker
        if enable_circuit_breaker:
            self.circuit_breaker = CircuitBreaker()
        else:
            self.circuit_breaker = None
        
        # Thread safety
        self._lock = threading.RLock()
        self._cache: Dict[str, Any] = {}
        self._cache_lock = threading.RLock()
        
        # Performance optimization
        self._last_logits_hash: Optional[str] = None
        self._last_result: Optional[SamplerResult] = None
        
        # Initialize the sampler
        self._initialize()
        self.state = SamplerState.READY
        
        logger.info(
            f"Initialized {self.__class__.__name__} sampler",
            extra={
                "sampler_id": self.sampler_id,
                "sampler_type": self.sampler_type.value,
                "config_params": self.config_params
            }
        )
    
    @abstractmethod
    def _initialize(self) -> None:
        """
        Initialize sampler-specific components.
        
        This method is called during construction and should set up
        any sampler-specific state or resources.
        """
        pass
    
    @abstractmethod
    def _sample_token(
        self,
        logits: LogitsArray,
        tokenizer: TokenizerWrapper,
        **kwargs
    ) -> SamplerResult:
        """
        Core sampling logic to be implemented by subclasses.
        
        Args:
            logits: Input logits array
            tokenizer: Tokenizer for special token handling
            **kwargs: Additional sampling parameters
            
        Returns:
            Sampled token(s)
        """
        pass
    
    def sample(
        self,
        logits: LogitsArray,
        tokenizer: TokenizerWrapper,
        **kwargs
    ) -> SamplerResult:
        """
        Generate a token sample with full error handling and metrics.
        
        This method implements the Template Method pattern, providing
        a consistent workflow with customizable sampling logic.
        
        Args:
            logits: Input logits array
            tokenizer: Tokenizer for special token handling
            **kwargs: Additional sampling parameters
            
        Returns:
            Sampled token(s)
            
        Raises:
            SamplingError: If sampling fails
            CircuitBreakerError: If circuit breaker is open
        """
        start_time = time.time()
        correlation_id = kwargs.get("correlation_id", str(uuid.uuid4()))
        
        try:
            with self._lock:
                # State validation
                if self.state not in [SamplerState.READY, SamplerState.SAMPLING]:
                    raise SamplingError(
                        operation="sample",
                        reason=f"Sampler in invalid state: {self.state}",
                        sampler_id=self.sampler_id,
                        correlation_id=correlation_id
                    )
                
                self.state = SamplerState.SAMPLING
                
                # Input validation
                self._validate_inputs(logits, tokenizer, **kwargs)
                
                # Check cache for identical logits
                logits_hash = self._compute_logits_hash(logits)
                cached_result = self._check_cache(logits_hash, **kwargs)
                if cached_result is not None:
                    if self.metrics:
                        self.metrics.record_cache_hit()
                    return cached_result
                
                if self.metrics:
                    self.metrics.record_cache_miss()
                
                # Perform sampling through circuit breaker
                if self.circuit_breaker:
                    result = self.circuit_breaker.call(
                        self._sample_token,
                        logits,
                        tokenizer,
                        sampler_id=self.sampler_id,
                        **kwargs
                    )
                else:
                    result = self._sample_token(logits, tokenizer, **kwargs)
                
                # Cache the result
                self._cache_result(logits_hash, result, **kwargs)
                
                # Record success metrics
                latency_ms = (time.time() - start_time) * 1000
                if self.metrics:
                    self.metrics.record_sample(latency_ms, True)
                
                self.state = SamplerState.READY
                
                logger.debug(
                    f"Successful sampling operation",
                    extra={
                        "sampler_id": self.sampler_id,
                        "correlation_id": correlation_id,
                        "latency_ms": latency_ms,
                        "result_shape": getattr(result, 'shape', 'scalar')
                    }
                )
                
                return result
                
        except Exception as e:
            # Record failure metrics
            latency_ms = (time.time() - start_time) * 1000
            if self.metrics:
                self.metrics.record_sample(latency_ms, False, str(e))
            
            self.state = SamplerState.ERROR
            
            # Re-raise as SamplingError if not already
            if not isinstance(e, SamplerError):
                raise SamplingError(
                    operation="sample",
                    reason=str(e),
                    sampler_id=self.sampler_id,
                    correlation_id=correlation_id,
                    context={"original_error": str(e)}
                )
            raise
        
        finally:
            # Ensure state is reset
            if self.state == SamplerState.SAMPLING:
                self.state = SamplerState.READY
    
    def batch_sample(
        self,
        logits_batch: List[LogitsArray],
        tokenizer: TokenizerWrapper,
        **kwargs
    ) -> List[SamplerResult]:
        """
        Generate multiple token samples with batch optimization.
        
        Args:
            logits_batch: Batch of logits arrays
            tokenizer: Tokenizer for special token handling
            **kwargs: Additional sampling parameters
            
        Returns:
            List of sampled tokens
        """
        if not logits_batch:
            return []
        
        # Default implementation: sample individually
        # Subclasses can override for true batch optimization
        results = []
        for i, logits in enumerate(logits_batch):
            try:
                result = self.sample(
                    logits, 
                    tokenizer, 
                    batch_index=i,
                    **kwargs
                )
                results.append(result)
            except Exception as e:
                logger.warning(
                    f"Failed to sample batch item {i}: {e}",
                    extra={"sampler_id": self.sampler_id, "batch_index": i}
                )
                # Use fallback sampling for failed items
                results.append(self._fallback_sample(logits, tokenizer))
        
        return results
    
    def _validate_inputs(
        self,
        logits: LogitsArray,
        tokenizer: TokenizerWrapper,
        **kwargs
    ) -> None:
        """
        Validate input parameters for sampling.
        
        Args:
            logits: Input logits array
            tokenizer: Tokenizer instance
            **kwargs: Additional parameters
            
        Raises:
            SamplingError: If inputs are invalid
        """
        if not isinstance(logits, mx.array):
            raise SamplingError(
                operation="input_validation",
                reason=f"Expected mx.array for logits, got {type(logits)}",
                sampler_id=self.sampler_id
            )
        
        if logits.size == 0:
            raise SamplingError(
                operation="input_validation",
                reason="Empty logits array",
                sampler_id=self.sampler_id
            )
        
        if not hasattr(tokenizer, 'decode'):
            raise SamplingError(
                operation="input_validation",
                reason="Invalid tokenizer: missing decode method",
                sampler_id=self.sampler_id
            )
    
    def _compute_logits_hash(self, logits: LogitsArray) -> str:
        """Compute hash of logits for caching."""
        try:
            # Use a subset of logits for hash computation to balance accuracy and performance
            if logits.size > 1000:
                # Sample every nth element for large arrays
                step = logits.size // 1000
                sample_logits = logits.flatten()[::step]
            else:
                sample_logits = logits.flatten()
            
            # Convert to bytes and hash
            logits_bytes = sample_logits.astype(mx.float32).tobytes()
            return hashlib.md5(logits_bytes).hexdigest()
        except Exception:
            # Fallback to timestamp-based hash if computation fails
            return hashlib.md5(str(time.time()).encode()).hexdigest()
    
    def _check_cache(self, logits_hash: str, **kwargs) -> Optional[SamplerResult]:
        """Check cache for previous results."""
        cache_key = f"{logits_hash}_{hash(frozenset(kwargs.items()))}"
        
        with self._cache_lock:
            if cache_key in self._cache:
                result, timestamp = self._cache[cache_key]
                # Cache expires after 60 seconds
                if time.time() - timestamp < 60:
                    return result
                else:
                    del self._cache[cache_key]
        
        return None
    
    def _cache_result(self, logits_hash: str, result: SamplerResult, **kwargs) -> None:
        """Cache sampling result."""
        cache_key = f"{logits_hash}_{hash(frozenset(kwargs.items()))}"
        
        with self._cache_lock:
            # Limit cache size
            if len(self._cache) > 1000:
                # Remove oldest entries
                oldest_keys = sorted(
                    self._cache.keys(),
                    key=lambda k: self._cache[k][1]
                )[:100]
                for key in oldest_keys:
                    del self._cache[key]
            
            self._cache[cache_key] = (result, time.time())
    
    def _fallback_sample(
        self,
        logits: LogitsArray,
        tokenizer: TokenizerWrapper
    ) -> SamplerResult:
        """
        Fallback sampling method for error recovery.
        
        Provides a simple, reliable sampling method when the main
        sampling logic fails.
        """
        try:
            # Simple temperature sampling as fallback
            temperature = 0.7
            scaled_logits = logits / temperature
            probs = nn.softmax(scaled_logits, axis=-1)
            
            # Sample from the distribution
            if probs.ndim == 1:
                return mx.random.categorical(mx.log(probs + 1e-8))
            else:
                # Handle batch dimension
                return mx.random.categorical(mx.log(probs + 1e-8), axis=-1)
                
        except Exception as e:
            logger.error(f"Fallback sampling failed: {e}")
            # Ultimate fallback: return most likely token
            return mx.argmax(logits, axis=-1)
    
    def get_metrics(self) -> Optional[SamplerMetrics]:
        """Get performance metrics for this sampler."""
        return self.metrics
    
    def configure(self, **params) -> None:
        """
        Update sampler configuration.
        
        Args:
            **params: Configuration parameters to update
        """
        with self._lock:
            self.config_params.update(params)
            self._on_configuration_changed()
    
    def _on_configuration_changed(self) -> None:
        """Handle configuration changes."""
        # Clear cache when configuration changes
        with self._cache_lock:
            self._cache.clear()
        
        logger.info(
            f"Configuration updated for sampler {self.sampler_id}",
            extra={
                "sampler_id": self.sampler_id,
                "new_config": self.config_params
            }
        )
    
    def reset(self) -> None:
        """Reset sampler state and metrics."""
        with self._lock:
            if self.metrics:
                self.metrics = SamplerMetrics(sampler_id=self.sampler_id)
            
            with self._cache_lock:
                self._cache.clear()
            
            if self.circuit_breaker:
                self.circuit_breaker._failure_count = 0
                self.circuit_breaker._success_count = 0
                self.circuit_breaker._state = "CLOSED"
            
            self.state = SamplerState.READY
        
        logger.info(f"Reset sampler {self.sampler_id}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self._cleanup()
    
    def _cleanup(self) -> None:
        """Cleanup resources."""
        with self._cache_lock:
            self._cache.clear()
        
        logger.debug(f"Cleaned up sampler {self.sampler_id}")


class SamplerRegistry:
    """
    Registry for managing sampler instances and types.
    
    Implements the Registry pattern for dynamic sampler discovery,
    registration, and lifecycle management.
    
    Features:
        - Thread-safe registration and lookup
        - Automatic cleanup of unused samplers
        - Performance monitoring across all samplers
        - Health checking and circuit breaker management
    """
    
    _instance: Optional['SamplerRegistry'] = None
    _lock: ClassVar[threading.RLock] = threading.RLock()
    
    def __new__(cls) -> 'SamplerRegistry':
        """Singleton pattern implementation."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._samplers: Dict[str, SamplerProtocol] = {}
        self._sampler_types: Dict[str, type] = {}
        self._weak_refs: Set[weakref.ref] = set()
        self._registry_lock = threading.RLock()
        self._cleanup_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="sampler-cleanup")
        self._initialized = True
        
        logger.info("Initialized SamplerRegistry singleton")
    
    def register_sampler_type(
        self,
        name: str,
        sampler_class: type,
        override: bool = False
    ) -> None:
        """
        Register a new sampler type.
        
        Args:
            name: Name for the sampler type
            sampler_class: Sampler class to register
            override: Whether to override existing registration
            
        Raises:
            ValueError: If name already registered and override=False
        """
        with self._registry_lock:
            if name in self._sampler_types and not override:
                raise ValueError(f"Sampler type '{name}' already registered")
            
            # Validate that the class implements SamplerProtocol
            if not issubclass(sampler_class, AbstractSampler):
                raise ValueError(f"Sampler class must inherit from AbstractSampler")
            
            self._sampler_types[name] = sampler_class
            
            logger.info(
                f"Registered sampler type '{name}'",
                extra={
                    "sampler_class": sampler_class.__name__,
                    "override": override
                }
            )
    
    def create_sampler(
        self,
        sampler_type: str,
        sampler_id: Optional[str] = None,
        **config_params
    ) -> SamplerProtocol:
        """
        Create a new sampler instance.
        
        Args:
            sampler_type: Type of sampler to create
            sampler_id: Optional custom ID for the sampler
            **config_params: Configuration parameters
            
        Returns:
            New sampler instance
            
        Raises:
            ValueError: If sampler type not registered
        """
        with self._registry_lock:
            if sampler_type not in self._sampler_types:
                raise ValueError(f"Unknown sampler type: {sampler_type}")
            
            sampler_class = self._sampler_types[sampler_type]
            sampler = sampler_class(
                sampler_id=sampler_id,
                **config_params
            )
            
            # Register the instance
            self._samplers[sampler.sampler_id] = sampler
            
            # Set up weak reference for cleanup
            def cleanup_callback(ref):
                self._cleanup_sampler(sampler.sampler_id)
            
            weak_ref = weakref.ref(sampler, cleanup_callback)
            self._weak_refs.add(weak_ref)
            
            logger.info(
                f"Created sampler instance",
                extra={
                    "sampler_id": sampler.sampler_id,
                    "sampler_type": sampler_type,
                    "config_params": config_params
                }
            )
            
            return sampler
    
    def get_sampler(self, sampler_id: str) -> Optional[SamplerProtocol]:
        """Get a registered sampler by ID."""
        with self._registry_lock:
            return self._samplers.get(sampler_id)
    
    def list_samplers(self) -> List[str]:
        """List all registered sampler IDs."""
        with self._registry_lock:
            return list(self._samplers.keys())
    
    def list_sampler_types(self) -> List[str]:
        """List all registered sampler types."""
        with self._registry_lock:
            return list(self._sampler_types.keys())
    
    def get_global_metrics(self) -> Dict[str, Any]:
        """
        Get aggregated metrics across all samplers.
        
        Returns:
            Dictionary of global metrics
        """
        with self._registry_lock:
            total_samples = 0
            total_successful = 0
            total_failed = 0
            total_errors = 0
            avg_latencies = []
            
            for sampler in self._samplers.values():
                if hasattr(sampler, 'metrics') and sampler.metrics:
                    metrics = sampler.metrics
                    total_samples += metrics.total_samples
                    total_successful += metrics.successful_samples
                    total_failed += metrics.failed_samples
                    total_errors += metrics.error_count
                    if metrics.average_latency_ms > 0:
                        avg_latencies.append(metrics.average_latency_ms)
            
            return {
                "total_samplers": len(self._samplers),
                "total_samples": total_samples,
                "total_successful": total_successful,
                "total_failed": total_failed,
                "total_errors": total_errors,
                "global_success_rate": (total_successful / total_samples * 100) if total_samples > 0 else 0.0,
                "average_latency_ms": sum(avg_latencies) / len(avg_latencies) if avg_latencies else 0.0,
                "active_sampler_types": list(set(
                    sampler.sampler_type.value for sampler in self._samplers.values()
                    if hasattr(sampler, 'sampler_type')
                ))
            }
    
    def _cleanup_sampler(self, sampler_id: str) -> None:
        """Clean up a sampler instance."""
        with self._registry_lock:
            if sampler_id in self._samplers:
                del self._samplers[sampler_id]
                logger.debug(f"Cleaned up sampler {sampler_id}")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all registered samplers.
        
        Returns:
            Health status report
        """
        with self._registry_lock:
            healthy_samplers = 0
            unhealthy_samplers = 0
            sampler_states = defaultdict(int)
            
            for sampler in self._samplers.values():
                if hasattr(sampler, 'state'):
                    state = sampler.state
                    sampler_states[state.name] += 1
                    
                    if state in [SamplerState.READY, SamplerState.SAMPLING]:
                        healthy_samplers += 1
                    else:
                        unhealthy_samplers += 1
                else:
                    unhealthy_samplers += 1
            
            return {
                "timestamp": time.time(),
                "total_samplers": len(self._samplers),
                "healthy_samplers": healthy_samplers,
                "unhealthy_samplers": unhealthy_samplers,
                "sampler_states": dict(sampler_states),
                "health_percentage": (healthy_samplers / len(self._samplers) * 100) if self._samplers else 100.0
            }
    
    def shutdown(self) -> None:
        """Shutdown the registry and cleanup all resources."""
        with self._registry_lock:
            # Cleanup all samplers
            for sampler in self._samplers.values():
                if hasattr(sampler, '_cleanup'):
                    try:
                        sampler._cleanup()
                    except Exception as e:
                        logger.warning(f"Error cleaning up sampler: {e}")
            
            self._samplers.clear()
            self._sampler_types.clear()
            self._weak_refs.clear()
            
            # Shutdown executor
            self._cleanup_executor.shutdown(wait=True)
            
            logger.info("SamplerRegistry shutdown complete")


# Utility functions for sampler operations
def with_metrics(func: Callable) -> Callable:
    """
    Decorator to automatically collect metrics for sampler operations.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function with metrics collection
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, 'metrics') or not self.metrics:
            return func(self, *args, **kwargs)
        
        start_time = time.time()
        try:
            result = func(self, *args, **kwargs)
            latency_ms = (time.time() - start_time) * 1000
            self.metrics.record_sample(latency_ms, True)
            return result
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.metrics.record_sample(latency_ms, False, str(e))
            raise
    
    return wrapper


def validate_sampler_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate sampler configuration parameters.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Temperature validation
    if 'temperature' in config:
        temp = config['temperature']
        if not isinstance(temp, (int, float)) or temp < 0 or temp > 2.0:
            errors.append("Temperature must be a number between 0.0 and 2.0")
    
    # Top-p validation
    if 'top_p' in config:
        top_p = config['top_p']
        if not isinstance(top_p, (int, float)) or top_p < 0 or top_p > 1.0:
            errors.append("top_p must be a number between 0.0 and 1.0")
    
    # Top-k validation
    if 'top_k' in config:
        top_k = config['top_k']
        if not isinstance(top_k, int) or top_k < 0:
            errors.append("top_k must be a non-negative integer")
    
    # Min-p validation
    if 'min_p' in config:
        min_p = config['min_p']
        if not isinstance(min_p, (int, float)) or min_p < 0 or min_p > 1.0:
            errors.append("min_p must be a number between 0.0 and 1.0")
    
    return errors


# Global registry instance
_global_registry: Optional[SamplerRegistry] = None


def get_global_registry() -> SamplerRegistry:
    """
    Get the global sampler registry instance.
    
    Returns:
        Global SamplerRegistry singleton
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = SamplerRegistry()
    return _global_registry


# Export public API
__all__ = [
    # Core protocols and interfaces
    "SamplerProtocol",
    "AbstractSampler",
    
    # Enums and types
    "SamplerState",
    "SamplerType",
    "SamplerResult",
    "LogitsArray",
    
    # Exceptions
    "SamplerError",
    "SamplingError",
    "CircuitBreakerError",
    
    # Metrics and monitoring
    "SamplerMetrics",
    "CircuitBreaker",
    
    # Registry and management
    "SamplerRegistry",
    "get_global_registry",
    
    # Utilities
    "with_metrics",
    "validate_sampler_config",
]