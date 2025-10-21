"""
Sampler factory implementation with comprehensive MLX integration.

This module implements the Factory pattern to create and manage sampler instances
with seamless integration to the MLX framework. It provides a sophisticated
abstraction layer that bridges configuration parameters with MLX's native
sampling capabilities while maintaining enterprise-grade reliability and performance.

Architecture:
    - SamplerFactory: Main factory with strategy-based sampler creation
    - MLXSamplerAdapter: Adapter pattern for MLX framework integration
    - SamplerBuilder: Builder pattern for complex sampler construction
    - SamplerPool: Object pool for performance optimization
    - SamplerMetricsCollector: Comprehensive metrics and monitoring

Design Patterns Applied:
    - Factory Pattern: Centralized sampler creation with type safety
    - Abstract Factory Pattern: Family of related sampler objects
    - Builder Pattern: Step-by-step construction of complex samplers
    - Adapter Pattern: Integration with MLX framework
    - Object Pool Pattern: Performance optimization through reuse
    - Strategy Pattern: Different creation strategies for different contexts
    - Observer Pattern: Metrics collection and event notification

SOLID Principles:
    - Single Responsibility: Each class handles one aspect of sampler creation
    - Open/Closed: Extensible for new sampler types without modification
    - Liskov Substitution: All factories are interchangeable through interfaces
    - Interface Segregation: Separate interfaces for different factory concerns
    - Dependency Inversion: Depends on abstractions for maximum flexibility

Example:
    >>> from mlx_rl_trainer.generation.samplers.factory import SamplerFactory
    >>> from mlx_rl_trainer.generation.config.enhanced_config import EnhancedGenerationConfig
    >>> 
    >>> factory = SamplerFactory()
    >>> config = EnhancedGenerationConfig(...)
    >>> sampler = factory.create_sampler(config, phase="think")
    >>> token = sampler(logits, tokenizer)
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
    runtime_checkable, ClassVar, Final, Tuple, NamedTuple, Set, Type
)
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, Future
import asyncio
import uuid
import json
import hashlib
from functools import wraps, lru_cache, partial
import inspect
import copy
import gc
import sys
from pathlib import Path

# MLX imports
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper

# Import our components
try:
    from .base import (
        SamplerProtocol, AbstractSampler, SamplerError, SamplingError,
        SamplerMetrics, SamplerRegistry, get_global_registry,
        SamplerState, SamplerType, CircuitBreaker
    )
    from ..bridge.config_resolver import (
        ConfigurationResolver, ResolutionContext, ResolutionResult,
        get_global_resolver, resolve_sampling_parameters
    )
    from ..config.enhanced_config import (
        EnhancedGenerationConfig, SamplingPhaseConfig, ConfigurationError
    )
except ImportError:
    # Fallback for development/testing
    from mlx_rl_trainer.generation.samplers.base import (
        SamplerProtocol, AbstractSampler, SamplerError, SamplingError,
        SamplerMetrics, SamplerRegistry, get_global_registry,
        SamplerState, SamplerType, CircuitBreaker
    )
    from mlx_rl_trainer.generation.bridge.config_resolver import (
        ConfigurationResolver, ResolutionContext, ResolutionResult,
        get_global_resolver, resolve_sampling_parameters
    )
    from mlx_rl_trainer.generation.config.enhanced_config import (
        EnhancedGenerationConfig, SamplingPhaseConfig, ConfigurationError
    )

# Import core configuration
try:
    from ...core.config import GenerationConfig, ExperimentConfig
except ImportError:
    from mlx_rl_trainer.core.config import GenerationConfig, ExperimentConfig

# Type definitions for enhanced type safety
T = TypeVar('T')
ConfigType = TypeVar('ConfigType', bound=Union[GenerationConfig, EnhancedGenerationConfig])
MLXSampler = Callable[[mx.array], mx.array]
SamplerInstance = Union[SamplerProtocol, MLXSampler]

logger = logging.getLogger(__name__)


class SamplerCreationStrategy(Enum):
    """
    Enumeration of sampler creation strategies.
    
    Defines different approaches for creating samplers based on
    context, performance requirements, and configuration complexity.
    """
    DIRECT_MLX = "direct_mlx"  # Direct MLX sampler creation
    WRAPPED_MLX = "wrapped_mlx"  # MLX sampler with our wrapper
    CUSTOM_IMPLEMENTATION = "custom_implementation"  # Custom sampler implementation
    POOLED = "pooled"  # Use object pool for performance
    CACHED = "cached"  # Use cached sampler instances
    HYBRID = "hybrid"  # Combination of strategies


class SamplerLifecycle(Enum):
    """
    Enumeration of sampler lifecycle states.
    
    Tracks the lifecycle of sampler instances for proper
    resource management and monitoring.
    """
    CREATING = auto()
    INITIALIZING = auto()
    READY = auto()
    ACTIVE = auto()
    IDLE = auto()
    DISPOSING = auto()
    DISPOSED = auto()
    ERROR = auto()


@dataclass
class SamplerCreationContext:
    """
    Context information for sampler creation.
    
    Provides comprehensive context for the sampler creation process,
    including configuration, performance hints, and lifecycle management.
    
    Attributes:
        config: Configuration object for the sampler
        phase: Sampling phase (think/answer/global)
        tokenizer: Tokenizer instance for special token handling
        creation_strategy: Strategy for sampler creation
        performance_hints: Performance optimization hints
        lifecycle_callbacks: Callbacks for lifecycle events
        correlation_id: Unique identifier for this creation request
        metadata: Additional context metadata
        timeout_seconds: Timeout for sampler creation
        retry_count: Number of creation retries allowed
    """
    
    config: Union[GenerationConfig, EnhancedGenerationConfig]
    phase: str = "global"
    tokenizer: Optional[TokenizerWrapper] = None
    creation_strategy: SamplerCreationStrategy = SamplerCreationStrategy.WRAPPED_MLX
    performance_hints: Dict[str, Any] = field(default_factory=dict)
    lifecycle_callbacks: Dict[str, Callable] = field(default_factory=dict)
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 30.0
    retry_count: int = 3
    
    def __post_init__(self):
        """Validate context consistency."""
        if self.timeout_seconds <= 0:
            raise ValueError("Timeout must be positive")
        if self.retry_count < 0:
            raise ValueError("Retry count must be non-negative")


@dataclass
class SamplerCreationResult:
    """
    Result of sampler creation process.
    
    Contains the created sampler along with comprehensive metadata
    about the creation process for monitoring and debugging.
    
    Attributes:
        sampler: Created sampler instance
        sampler_id: Unique identifier for the sampler
        creation_time_ms: Time taken for creation in milliseconds
        strategy_used: Strategy that was actually used
        parameters_resolved: Resolved parameters used for creation
        warnings: List of warnings generated during creation
        metadata: Creation metadata and statistics
        lifecycle_state: Current lifecycle state
    """
    
    sampler: SamplerInstance
    sampler_id: str
    creation_time_ms: float
    strategy_used: SamplerCreationStrategy
    parameters_resolved: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    lifecycle_state: SamplerLifecycle = SamplerLifecycle.READY
    
    @property
    def success(self) -> bool:
        """Check if creation was successful."""
        return self.sampler is not None and self.lifecycle_state == SamplerLifecycle.READY
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "sampler_id": self.sampler_id,
            "creation_time_ms": self.creation_time_ms,
            "strategy_used": self.strategy_used.value,
            "parameters_resolved": self.parameters_resolved,
            "warnings": self.warnings,
            "metadata": self.metadata,
            "lifecycle_state": self.lifecycle_state.name,
            "success": self.success
        }


class SamplerFactoryError(SamplerError):
    """Base exception for sampler factory errors."""
    
    def __init__(
        self,
        message: str,
        creation_context: Optional[SamplerCreationContext] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.creation_context = creation_context


class SamplerCreationError(SamplerFactoryError):
    """Exception for sampler creation failures."""
    
    def __init__(
        self,
        strategy: SamplerCreationStrategy,
        reason: str,
        **kwargs
    ):
        message = f"Sampler creation failed using strategy '{strategy.value}': {reason}"
        super().__init__(message, **kwargs)
        self.strategy = strategy
        self.reason = reason


class SamplerPoolExhaustedError(SamplerFactoryError):
    """Exception when sampler pool is exhausted."""
    
    def __init__(self, pool_size: int, active_samplers: int, **kwargs):
        message = f"Sampler pool exhausted: {active_samplers}/{pool_size} samplers active"
        super().__init__(message, **kwargs)
        self.pool_size = pool_size
        self.active_samplers = active_samplers


@runtime_checkable
class SamplerFactoryProtocol(Protocol):
    """
    Protocol for sampler factory implementations.
    
    Defines the interface for different sampler factory strategies,
    enabling flexible and extensible sampler creation approaches.
    """
    
    def create_sampler(
        self,
        context: SamplerCreationContext
    ) -> SamplerCreationResult:
        """
        Create a sampler instance.
        
        Args:
            context: Creation context with configuration and hints
            
        Returns:
            Sampler creation result
        """
        ...
    
    def supports_strategy(self, strategy: SamplerCreationStrategy) -> bool:
        """
        Check if factory supports the given strategy.
        
        Args:
            strategy: Creation strategy to check
            
        Returns:
            True if supported, False otherwise
        """
        ...
    
    def get_factory_metrics(self) -> Dict[str, Any]:
        """
        Get factory performance metrics.
        
        Returns:
            Dictionary of factory metrics
        """
        ...


class MLXSamplerAdapter(AbstractSampler):
    """
    Adapter for integrating MLX samplers with our sampler interface.
    
    Implements the Adapter pattern to provide a consistent interface
    for MLX samplers while maintaining full compatibility with the
    MLX framework and adding enterprise-grade features.
    
    Features:
        - Seamless MLX integration with parameter mapping
        - Comprehensive error handling and recovery
        - Performance monitoring and optimization
        - Thread-safe operations with concurrent access support
        - Advanced caching and memoization
    """
    
    def __init__(
        self,
        mlx_sampler: MLXSampler,
        parameters: Dict[str, Any],
        sampler_id: Optional[str] = None,
        **kwargs
    ):
        # Initialize base sampler
        super().__init__(
            sampler_id=sampler_id,
            sampler_type=SamplerType.HYBRID,
            **kwargs
        )
        
        self.mlx_sampler = mlx_sampler
        self.parameters = parameters.copy()
        self.creation_time = time.time()
        
        # Performance optimization
        self._call_count = 0
        self._total_call_time = 0.0
        self._last_logits_shape: Optional[Tuple[int, ...]] = None
        
        # Thread safety for MLX sampler calls
        self._mlx_lock = threading.RLock()
        
        logger.info(
            f"Created MLX sampler adapter",
            extra={
                "sampler_id": self.sampler_id,
                "parameters": self.parameters,
                "mlx_sampler_type": type(self.mlx_sampler).__name__
            }
        )
    
    def _initialize(self) -> None:
        """Initialize MLX sampler adapter."""
        # Validate MLX sampler
        if not callable(self.mlx_sampler):
            raise SamplerCreationError(
                strategy=SamplerCreationStrategy.WRAPPED_MLX,
                reason="MLX sampler is not callable",
                sampler_id=self.sampler_id
            )
        
        # Test MLX sampler with dummy data
        try:
            test_logits = mx.random.normal((10,))
            with self._mlx_lock:
                result = self.mlx_sampler(test_logits)
            
            if not isinstance(result, (mx.array, int)):
                raise SamplerCreationError(
                    strategy=SamplerCreationStrategy.WRAPPED_MLX,
                    reason=f"MLX sampler returned invalid type: {type(result)}",
                    sampler_id=self.sampler_id
                )
                
        except Exception as e:
            raise SamplerCreationError(
                strategy=SamplerCreationStrategy.WRAPPED_MLX,
                reason=f"MLX sampler test failed: {e}",
                sampler_id=self.sampler_id
            )
    
    def _sample_token(
        self,
        logits: mx.array,
        tokenizer: TokenizerWrapper,
        **kwargs
    ) -> mx.array:
        """Core sampling logic using MLX sampler."""
        start_time = time.time()
        
        try:
            # Update call statistics
            self._call_count += 1
            self._last_logits_shape = logits.shape
            
            # Call MLX sampler with thread safety
            with self._mlx_lock:
                result = self.mlx_sampler(logits)
            
            # Validate result
            if not isinstance(result, (mx.array, int)):
                raise SamplingError(
                    operation="mlx_sampling",
                    reason=f"Invalid result type: {type(result)}",
                    sampler_id=self.sampler_id
                )
            
            # Convert to array if needed
            if isinstance(result, int):
                result = mx.array([result])
            elif isinstance(result, mx.array) and result.ndim == 0:
                result = mx.array([result.item()])
            
            # Update performance metrics
            call_time = time.time() - start_time
            self._total_call_time += call_time
            
            logger.debug(
                f"MLX sampling completed",
                extra={
                    "sampler_id": self.sampler_id,
                    "logits_shape": logits.shape,
                    "result_shape": result.shape,
                    "call_time_ms": call_time * 1000,
                    "call_count": self._call_count
                }
            )
            
            return result
            
        except Exception as e:
            call_time = time.time() - start_time
            self._total_call_time += call_time
            
            logger.error(
                f"MLX sampling failed: {e}",
                extra={
                    "sampler_id": self.sampler_id,
                    "logits_shape": logits.shape,
                    "call_time_ms": call_time * 1000,
                    "error_type": type(e).__name__
                }
            )
            
            # Re-raise as SamplingError if not already
            if isinstance(e, SamplingError):
                raise
            else:
                raise SamplingError(
                    operation="mlx_sampling",
                    reason=str(e),
                    sampler_id=self.sampler_id
                )
    
    def get_mlx_sampler(self) -> MLXSampler:
        """Get the underlying MLX sampler."""
        return self.mlx_sampler
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get the parameters used to create this sampler."""
        return self.parameters.copy()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this adapter."""
        avg_call_time = (
            self._total_call_time / self._call_count
            if self._call_count > 0 else 0.0
        )
        
        return {
            "call_count": self._call_count,
            "total_call_time_ms": self._total_call_time * 1000,
            "average_call_time_ms": avg_call_time * 1000,
            "last_logits_shape": self._last_logits_shape,
            "creation_time": self.creation_time,
            "uptime_seconds": time.time() - self.creation_time
        }


class SamplerPool:
    """
    Object pool for sampler instances with advanced lifecycle management.
    
    Implements the Object Pool pattern to optimize performance by reusing
    sampler instances, reducing creation overhead and memory allocation.
    
    Features:
        - Configurable pool size and lifecycle policies
        - Thread-safe operations with concurrent access
        - Health monitoring and automatic cleanup
        - Performance metrics and optimization
        - Graceful degradation under load
    """
    
    def __init__(
        self,
        max_pool_size: int = 50,
        min_pool_size: int = 5,
        max_idle_time: float = 300.0,  # 5 minutes
        cleanup_interval: float = 60.0,  # 1 minute
        health_check_interval: float = 30.0  # 30 seconds
    ):
        self.max_pool_size = max_pool_size
        self.min_pool_size = min_pool_size
        self.max_idle_time = max_idle_time
        self.cleanup_interval = cleanup_interval
        self.health_check_interval = health_check_interval
        
        # Pool storage
        self._available_samplers: deque = deque()
        self._active_samplers: Dict[str, Tuple[SamplerInstance, float]] = {}
        self._sampler_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Thread safety
        self._pool_lock = threading.RLock()
        
        # Performance metrics
        self._pool_hits = 0
        self._pool_misses = 0
        self._total_created = 0
        self._total_destroyed = 0
        
        # Background maintenance
        self._maintenance_executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="sampler-pool"
        )
        self._shutdown_event = threading.Event()
        
        # Start maintenance tasks
        self._start_maintenance_tasks()
        
        logger.info(
            f"Initialized sampler pool",
            extra={
                "max_pool_size": self.max_pool_size,
                "min_pool_size": self.min_pool_size,
                "max_idle_time": self.max_idle_time
            }
        )
    
    def acquire_sampler(
        self,
        context: SamplerCreationContext,
        factory: 'SamplerFactory'
    ) -> SamplerInstance:
        """
        Acquire a sampler from the pool or create a new one.
        
        Args:
            context: Creation context for the sampler
            factory: Factory to create new samplers if needed
            
        Returns:
            Sampler instance
            
        Raises:
            SamplerPoolExhaustedError: If pool is exhausted and cannot create new samplers
        """
        with self._pool_lock:
            # Try to get from pool first
            sampler = self._try_get_from_pool(context)
            if sampler:
                self._pool_hits += 1
                sampler_id = getattr(sampler, 'sampler_id', str(uuid.uuid4()))
                self._active_samplers[sampler_id] = (sampler, time.time())
                
                logger.debug(
                    f"Acquired sampler from pool",
                    extra={
                        "sampler_id": sampler_id,
                        "pool_size": len(self._available_samplers),
                        "active_count": len(self._active_samplers)
                    }
                )
                
                return sampler
            
            # Pool miss - check if we can create new sampler
            if len(self._active_samplers) >= self.max_pool_size:
                raise SamplerPoolExhaustedError(
                    pool_size=self.max_pool_size,
                    active_samplers=len(self._active_samplers),
                    correlation_id=context.correlation_id
                )
            
            # Create new sampler
            self._pool_misses += 1
            result = factory._create_sampler_direct(context)
            sampler = result.sampler
            
            self._total_created += 1
            sampler_id = result.sampler_id
            self._active_samplers[sampler_id] = (sampler, time.time())
            self._sampler_metadata[sampler_id] = {
                "creation_context": context,
                "creation_result": result,
                "acquisition_time": time.time()
            }
            
            logger.debug(
                f"Created new sampler for pool",
                extra={
                    "sampler_id": sampler_id,
                    "pool_size": len(self._available_samplers),
                    "active_count": len(self._active_samplers),
                    "total_created": self._total_created
                }
            )
            
            return sampler
    
    def release_sampler(self, sampler: SamplerInstance) -> None:
        """
        Release a sampler back to the pool.
        
        Args:
            sampler: Sampler instance to release
        """
        with self._pool_lock:
            sampler_id = getattr(sampler, 'sampler_id', None)
            if not sampler_id or sampler_id not in self._active_samplers:
                logger.warning(f"Attempted to release unknown sampler: {sampler_id}")
                return
            
            # Remove from active samplers
            del self._active_samplers[sampler_id]
            
            # Check if sampler is still healthy
            if self._is_sampler_healthy(sampler):
                # Return to pool if there's space
                if len(self._available_samplers) < self.max_pool_size:
                    self._available_samplers.append((sampler, time.time()))
                    
                    logger.debug(
                        f"Released sampler to pool",
                        extra={
                            "sampler_id": sampler_id,
                            "pool_size": len(self._available_samplers),
                            "active_count": len(self._active_samplers)
                        }
                    )
                else:
                    # Pool is full, destroy sampler
                    self._destroy_sampler(sampler)
            else:
                # Sampler is unhealthy, destroy it
                self._destroy_sampler(sampler)
    
    def _try_get_from_pool(self, context: SamplerCreationContext) -> Optional[SamplerInstance]:
        """Try to get a compatible sampler from the pool."""
        # For now, use simple FIFO strategy
        # In a more sophisticated implementation, we could match by configuration
        while self._available_samplers:
            sampler, pool_time = self._available_samplers.popleft()
            
            # Check if sampler is still valid
            if time.time() - pool_time > self.max_idle_time:
                self._destroy_sampler(sampler)
                continue
            
            if self._is_sampler_healthy(sampler):
                return sampler
            else:
                self._destroy_sampler(sampler)
        
        return None
    
    def _is_sampler_healthy(self, sampler: SamplerInstance) -> bool:
        """Check if a sampler is healthy and ready for use."""
        try:
            # Basic health checks
            if hasattr(sampler, 'state'):
                return sampler.state in [SamplerState.READY, SamplerState.SAMPLING]
            
            # For MLX samplers, we assume they're healthy if callable
            return callable(sampler)
            
        except Exception as e:
            logger.warning(f"Sampler health check failed: {e}")
            return False
    
    def _destroy_sampler(self, sampler: SamplerInstance) -> None:
        """Destroy a sampler and clean up resources."""
        try:
            sampler_id = getattr(sampler, 'sampler_id', 'unknown')
            
            # Call cleanup if available
            if hasattr(sampler, '_cleanup'):
                sampler._cleanup()
            
            # Remove metadata
            if sampler_id in self._sampler_metadata:
                del self._sampler_metadata[sampler_id]
            
            self._total_destroyed += 1
            
            logger.debug(f"Destroyed sampler {sampler_id}")
            
        except Exception as e:
            logger.warning(f"Error destroying sampler: {e}")
    
    def _start_maintenance_tasks(self) -> None:
        """Start background maintenance tasks."""
        # Cleanup task
        self._maintenance_executor.submit(self._cleanup_worker)
        
        # Health check task
        self._maintenance_executor.submit(self._health_check_worker)
    
    def _cleanup_worker(self) -> None:
        """Background worker for pool cleanup."""
        while not self._shutdown_event.wait(self.cleanup_interval):
            try:
                with self._pool_lock:
                    current_time = time.time()
                    
                    # Clean up idle samplers
                    cleaned_count = 0
                    new_available = deque()
                    
                    while self._available_samplers:
                        sampler, pool_time = self._available_samplers.popleft()
                        
                        if current_time - pool_time > self.max_idle_time:
                            self._destroy_sampler(sampler)
                            cleaned_count += 1
                        else:
                            new_available.append((sampler, pool_time))
                    
                    self._available_samplers = new_available
                    
                    # Ensure minimum pool size
                    if len(self._available_samplers) < self.min_pool_size:
                        # Could pre-create samplers here if needed
                        pass
                    
                    if cleaned_count > 0:
                        logger.debug(f"Pool cleanup: removed {cleaned_count} idle samplers")
                        
            except Exception as e:
                logger.error(f"Pool cleanup error: {e}")
    
    def _health_check_worker(self) -> None:
        """Background worker for health checks."""
        while not self._shutdown_event.wait(self.health_check_interval):
            try:
                with self._pool_lock:
                    # Check active samplers
                    unhealthy_samplers = []
                    
                    for sampler_id, (sampler, _) in self._active_samplers.items():
                        if not self._is_sampler_healthy(sampler):
                            unhealthy_samplers.append(sampler_id)
                    
                    # Remove unhealthy active samplers
                    for sampler_id in unhealthy_samplers:
                        if sampler_id in self._active_samplers:
                            sampler, _ = self._active_samplers[sampler_id]
                            del self._active_samplers[sampler_id]
                            self._destroy_sampler(sampler)
                    
                    if unhealthy_samplers:
                        logger.warning(f"Removed {len(unhealthy_samplers)} unhealthy active samplers")
                        
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get pool performance statistics."""
        with self._pool_lock:
            total_requests = self._pool_hits + self._pool_misses
            hit_rate = (self._pool_hits / total_requests * 100) if total_requests > 0 else 0.0
            
            return {
                "max_pool_size": self.max_pool_size,
                "min_pool_size": self.min_pool_size,
                "available_samplers": len(self._available_samplers),
                "active_samplers": len(self._active_samplers),
                "pool_utilization": len(self._active_samplers) / self.max_pool_size * 100,
                "pool_hits": self._pool_hits,
                "pool_misses": self._pool_misses,
                "hit_rate": hit_rate,
                "total_created": self._total_created,
                "total_destroyed": self._total_destroyed,
                "net_samplers": self._total_created - self._total_destroyed
            }
    
    def shutdown(self) -> None:
        """Shutdown the pool and cleanup all resources."""
        logger.info("Shutting down sampler pool")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Shutdown executor
        self._maintenance_executor.shutdown(wait=True)
        
        with self._pool_lock:
            # Destroy all samplers
            while self._available_samplers:
                sampler, _ = self._available_samplers.popleft()
                self._destroy_sampler(sampler)
            
            for sampler_id, (sampler, _) in self._active_samplers.items():
                self._destroy_sampler(sampler)
            
            self._active_samplers.clear()
            self._sampler_metadata.clear()
        
        logger.info("Sampler pool shutdown complete")


class SamplerFactory:
    """
    Main sampler factory with comprehensive creation strategies.
    
    Implements the Factory pattern with multiple creation strategies,
    providing a unified interface for creating samplers while supporting
    different optimization approaches and integration patterns.
    
    Features:
        - Multiple creation strategies with automatic selection
        - Configuration resolution and parameter mapping
        - Performance optimization with pooling and caching
        - Comprehensive error handling and recovery
        - Extensive monitoring and observability
        - Thread-safe operations with concurrent access support
    """
    
    def __init__(
        self,
        config_resolver: Optional[ConfigurationResolver] = None,
        sampler_registry: Optional[SamplerRegistry] = None,
        enable_pooling: bool = True,
        pool_config: Optional[Dict[str, Any]] = None,
        default_strategy: SamplerCreationStrategy = SamplerCreationStrategy.WRAPPED_MLX
    ):
        # Initialize components
        self.config_resolver = config_resolver or get_global_resolver()
        self.sampler_registry = sampler_registry or get_global_registry()
        self.default_strategy = default_strategy
        
        # Initialize object pool
        self.enable_pooling = enable_pooling
        if enable_pooling:
            pool_config = pool_config or {}
            self.sampler_pool = SamplerPool(**pool_config)
        else:
            self.sampler_pool = None
        
        # Strategy implementations
        self._strategy_handlers: Dict[SamplerCreationStrategy, Callable] = {
            SamplerCreationStrategy.DIRECT_MLX: self._create_direct_mlx,
            SamplerCreationStrategy.WRAPPED_MLX: self._create_wrapped_mlx,
            SamplerCreationStrategy.CUSTOM_IMPLEMENTATION: self._create_custom_implementation,
            SamplerCreationStrategy.POOLED: self._create_pooled,
            SamplerCreationStrategy.CACHED: self._create_cached,
            SamplerCreationStrategy.HYBRID: self._create_hybrid,
        }
        
        # Performance metrics
        self._creation_count = 0
        self._total_creation_time = 0.0
        self._strategy_usage: Dict[SamplerCreationStrategy, int] = defaultdict(int)
        self._error_count = 0
        
        # Thread safety
        self._factory_lock = threading.RLock()
        
        # Cache for created samplers
        self._sampler_cache: Dict[str, Tuple[SamplerInstance, float]] = {}
        self._cache_lock = threading.RLock()
        
        logger.info(
            f"Initialized SamplerFactory",
            extra={
                "default_strategy": self.default_strategy.value,
                "enable_pooling": self.enable_pooling,
                "available_strategies": list(self._strategy_handlers.keys())
            }
        )
    
    def create_sampler(
        self,
        config: Union[GenerationConfig, EnhancedGenerationConfig, ExperimentConfig],
        phase: str = "global",
        tokenizer: Optional[TokenizerWrapper] = None,
        strategy: Optional[SamplerCreationStrategy] = None,
        **overrides
    ) -> SamplerInstance:
        """
        Create a sampler instance with comprehensive configuration resolution.
        
        Args:
            config: Configuration object for the sampler
            phase: Sampling phase (think/answer/global)
            tokenizer: Optional tokenizer for special token handling
            strategy: Optional creation strategy override
            **overrides: Parameter overrides
            
        Returns:
            Created sampler instance
            
        Raises:
            SamplerCreationError: If sampler creation fails
        """
        start_time = time.time()
        correlation_id = str(uuid.uuid4())
        
        try:
            with self._factory_lock:
                self._creation_count += 1
                
                # Create context
                context = SamplerCreationContext(
                    config=config,
                    phase=phase,
                    tokenizer=tokenizer,
                    creation_strategy=strategy or self.default_strategy,
                    correlation_id=correlation_id,
                    metadata=overrides
                )
                
                # Create sampler using specified strategy
                result = self._create_sampler_internal(context)
                
                # Update metrics
                creation_time = (time.time() - start_time) * 1000
                self._total_creation_time += creation_time
                self._strategy_usage[result.strategy_used] += 1
                
                logger.info(
                    f"Sampler creation completed",
                    extra={
                        "correlation_id": correlation_id,
                        "sampler_id": result.sampler_id,
                        "strategy_used": result.strategy_used.value,
                        "creation_time_ms": creation_time,
                        "phase": phase,
                        "success": result.success
                    }
                )
                
                return result.sampler
                
        except Exception as e:
            creation_time = (time.time() - start_time) * 1000
            self._total_creation_time += creation_time
            self._error_count += 1
            
            logger.error(
                f"Sampler creation failed: {e}",
                extra={
                    "correlation_id": correlation_id,
                    "phase": phase,
                    "strategy": strategy.value if strategy else self.default_strategy.value,
                    "creation_time_ms": creation_time,
                    "error_type": type(e).__name__
                }
            )
            
            if isinstance(e, SamplerFactoryError):
                raise
            else:
                raise SamplerCreationError(
                    strategy=strategy or self.default_strategy,
                    reason=str(e),
                    correlation_id=correlation_id
                )
    
    def _create_sampler_internal(self, context: SamplerCreationContext) -> SamplerCreationResult:
        """Internal sampler creation with strategy dispatch."""
        strategy = context.creation_strategy
        
        if strategy not in self._strategy_handlers:
            raise SamplerCreationError(
                strategy=strategy,
                reason=f"Unsupported creation strategy: {strategy}",
                creation_context=context
            )
        
        handler = self._strategy_handlers[strategy]
        return handler(context)
    
    def _create_direct_mlx(self, context: SamplerCreationContext) -> SamplerCreationResult:
        """Create sampler using direct MLX integration."""
        start_time = time.time()
        
        try:
            # Resolve parameters
            resolution_context = ResolutionContext(
                phase=context.phase,
                correlation_id=context.correlation_id,
                override_parameters=context.metadata
            )
            
            resolution_result = self.config_resolver.resolve_parameters(
                context.config, resolution_context
            )
            
            if not resolution_result.success:
                raise SamplerCreationError(
                    strategy=SamplerCreationStrategy.DIRECT_MLX,
                    reason=f"Parameter resolution failed: {resolution_result.conflicts}",
                    creation_context=context
                )
            
            # Create MLX sampler
            mlx_sampler = make_sampler(**resolution_result.parameters)
            
            # Generate sampler ID
            sampler_id = f"mlx_direct_{uuid.uuid4().hex[:8]}"
            
            creation_time = (time.time() - start_time) * 1000
            
            return SamplerCreationResult(
                sampler=mlx_sampler,
                sampler_id=sampler_id,
                creation_time_ms=creation_time,
                strategy_used=SamplerCreationStrategy.DIRECT_MLX,
                parameters_resolved=resolution_result.parameters,
                metadata={
                    "mlx_sampler_type": type(mlx_sampler).__name__,
                    "resolution_result": resolution_result.to_dict()
                }
            )
            
        except Exception as e:
            raise SamplerCreationError(
                strategy=SamplerCreationStrategy.DIRECT_MLX,
                reason=str(e),
                creation_context=context
            )
    
    def _create_wrapped_mlx(self, context: SamplerCreationContext) -> SamplerCreationResult:
        """Create sampler using wrapped MLX integration."""
        start_time = time.time()
        
        try:
            # First create direct MLX sampler
            direct_result = self._create_direct_mlx(context)
            mlx_sampler = direct_result.sampler
            
            # Wrap with our adapter
            adapter = MLXSamplerAdapter(
                mlx_sampler=mlx_sampler,
                parameters=direct_result.parameters_resolved,
                sampler_id=f"mlx_wrapped_{uuid.uuid4().hex[:8]}"
            )
            
            creation_time = (time.time() - start_time) * 1000
            
            return SamplerCreationResult(
                sampler=adapter,
                sampler_id=adapter.sampler_id,
                creation_time_ms=creation_time,
                strategy_used=SamplerCreationStrategy.WRAPPED_MLX,
                parameters_resolved=direct_result.parameters_resolved,
                metadata={
                    "adapter_type": type(adapter).__name__,
                    "underlying_mlx_type": type(mlx_sampler).__name__,
                    "direct_creation_time_ms": direct_result.creation_time_ms
                }
            )
            
        except Exception as e:
            raise SamplerCreationError(
                strategy=SamplerCreationStrategy.WRAPPED_MLX,
                reason=str(e),
                creation_context=context
            )
    
    def _create_custom_implementation(self, context: SamplerCreationContext) -> SamplerCreationResult:
        """Create sampler using custom implementation."""
        # This would create samplers from the registry
        # For now, fall back to wrapped MLX
        return self._create_wrapped_mlx(context)
    
    def _create_pooled(self, context: SamplerCreationContext) -> SamplerCreationResult:
        """Create sampler using object pool."""
        if not self.enable_pooling or not self.sampler_pool:
            # Fall back to wrapped MLX if pooling not available
            return self._create_wrapped_mlx(context)
        
        start_time = time.time()
        
        try:
            sampler = self.sampler_pool.acquire_sampler(context, self)
            sampler_id = getattr(sampler, 'sampler_id', f"pooled_{uuid.uuid4().hex[:8]}")
            
            creation_time = (time.time() - start_time) * 1000
            
            return SamplerCreationResult(
                sampler=sampler,
                sampler_id=sampler_id,
                creation_time_ms=creation_time,
                strategy_used=SamplerCreationStrategy.POOLED,
                parameters_resolved={},  # Would be filled by pool
                metadata={
                    "pool_stats": self.sampler_pool.get_pool_stats(),
                    "from_pool": True
                }
            )
            
        except Exception as e:
            raise SamplerCreationError(
                strategy=SamplerCreationStrategy.POOLED,
                reason=str(e),
                creation_context=context
            )
    
    def _create_cached(self, context: SamplerCreationContext) -> SamplerCreationResult:
        """Create sampler using cache."""
        start_time = time.time()
        
        try:
            # Compute cache key
            cache_key = self._compute_cache_key(context)
            
            with self._cache_lock:
                # Check cache
                if cache_key in self._sampler_cache:
                    sampler, cache_time = self._sampler_cache[cache_key]
                    
                    # Check if cache entry is still valid (5 minutes TTL)
                    if time.time() - cache_time < 300:
                        creation_time = (time.time() - start_time) * 1000
                        
                        return SamplerCreationResult(
                            sampler=sampler,
                            sampler_id=getattr(sampler, 'sampler_id', f"cached_{uuid.uuid4().hex[:8]}"),
                            creation_time_ms=creation_time,
                            strategy_used=SamplerCreationStrategy.CACHED,
                            parameters_resolved={},
                            metadata={
                                "cache_hit": True,
                                "cache_age_seconds": time.time() - cache_time
                            }
                        )
                    else:
                        # Cache expired
                        del self._sampler_cache[cache_key]
                
                # Cache miss - create new sampler
                result = self._create_wrapped_mlx(context)
                
                # Cache the result
                self._sampler_cache[cache_key] = (result.sampler, time.time())
                
                # Limit cache size
                if len(self._sampler_cache) > 100:
                    # Remove oldest entries
                    oldest_key = min(
                        self._sampler_cache.keys(),
                        key=lambda k: self._sampler_cache[k][1]
                    )
                    del self._sampler_cache[oldest_key]
                
                result.strategy_used = SamplerCreationStrategy.CACHED
                result.metadata["cache_hit"] = False
                
                return result
                
        except Exception as e:
            raise SamplerCreationError(
                strategy=SamplerCreationStrategy.CACHED,
                reason=str(e),
                creation_context=context
            )
    
    def _create_hybrid(self, context: SamplerCreationContext) -> SamplerCreationResult:
        """Create sampler using hybrid strategy."""
        # Intelligent strategy selection based on context
        
        # Use pooling for high-frequency requests
        if self.enable_pooling and context.performance_hints.get("high_frequency", False):
            try:
                return self._create_pooled(context)
            except SamplerPoolExhaustedError:
                # Fall back to cached if pool is exhausted
                return self._create_cached(context)
        
        # Use caching for repeated configurations
        if context.performance_hints.get("repeated_config", False):
            return self._create_cached(context)
        
        # Default to wrapped MLX for general use
        return self._create_wrapped_mlx(context)
    
    def _create_sampler_direct(self, context: SamplerCreationContext) -> SamplerCreationResult:
        """Direct sampler creation without pooling (used by pool)."""
        # Remove pooled strategy to avoid recursion
        original_strategy = context.creation_strategy
        if original_strategy == SamplerCreationStrategy.POOLED:
            context.creation_strategy = SamplerCreationStrategy.WRAPPED_MLX
        
        try:
            return self._create_sampler_internal(context)
        finally:
            context.creation_strategy = original_strategy
    
    def _compute_cache_key(self, context: SamplerCreationContext) -> str:
        """Compute cache key for sampler context."""
        # Create deterministic hash of context
        key_data = {
            "config_type": type(context.config).__name__,
            "phase": context.phase,
            "strategy": context.creation_strategy.value,
            "metadata": sorted(context.metadata.items()) if context.metadata else []
        }
        
        # Add config-specific data
        if hasattr(context.config, '__dict__'):
            config_dict = {k: v for k, v in context.config.__dict__.items() if not k.startswith('_')}
            key_data["config"] = sorted(config_dict.items())
        
        key_str = json.dumps(key_data, default=str, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def release_sampler(self, sampler: SamplerInstance) -> None:
        """
        Release a sampler back to the pool if applicable.
        
        Args:
            sampler: Sampler instance to release
        """
        if self.enable_pooling and self.sampler_pool:
            self.sampler_pool.release_sampler(sampler)
        else:
            # Clean up sampler if not pooled
            if hasattr(sampler, '_cleanup'):
                try:
                    sampler._cleanup()
                except Exception as e:
                    logger.warning(f"Error cleaning up sampler: {e}")
    
    def get_factory_metrics(self) -> Dict[str, Any]:
        """Get comprehensive factory metrics."""
        with self._factory_lock:
            avg_creation_time = (
                self._total_creation_time / self._creation_count
                if self._creation_count > 0 else 0.0
            )
            
            metrics = {
                "total_creations": self._creation_count,
                "total_creation_time_ms": self._total_creation_time,
                "average_creation_time_ms": avg_creation_time,
                "error_count": self._error_count,
                "error_rate": (self._error_count / self._creation_count * 100) if self._creation_count > 0 else 0.0,
                "strategy_usage": {k.value: v for k, v in self._strategy_usage.items()},
                "cache_size": len(self._sampler_cache) if hasattr(self, '_sampler_cache') else 0
            }
            
            # Add pool metrics if available
            if self.enable_pooling and self.sampler_pool:
                metrics["pool_stats"] = self.sampler_pool.get_pool_stats()
            
            return metrics
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on factory components."""
        health_status = {
            "timestamp": time.time(),
            "overall_status": "healthy",
            "components": {}
        }
        
        # Check config resolver
        try:
            resolver_health = self.config_resolver.health_check()
            health_status["components"]["config_resolver"] = resolver_health["overall_status"]
        except Exception as e:
            health_status["components"]["config_resolver"] = f"unhealthy: {e}"
            health_status["overall_status"] = "degraded"
        
        # Check sampler registry
        try:
            registry_health = self.sampler_registry.health_check()
            health_status["components"]["sampler_registry"] = registry_health["overall_status"]
        except Exception as e:
            health_status["components"]["sampler_registry"] = f"unhealthy: {e}"
            health_status["overall_status"] = "degraded"
        
        # Check sampler pool
        if self.enable_pooling and self.sampler_pool:
            try:
                pool_stats = self.sampler_pool.get_pool_stats()
                if pool_stats["pool_utilization"] > 90:
                    health_status["components"]["sampler_pool"] = "warning: high utilization"
                    if health_status["overall_status"] == "healthy":
                        health_status["overall_status"] = "warning"
                else:
                    health_status["components"]["sampler_pool"] = "healthy"
            except Exception as e:
                health_status["components"]["sampler_pool"] = f"unhealthy: {e}"
                health_status["overall_status"] = "degraded"
        
        return health_status
    
    def shutdown(self) -> None:
        """Shutdown the factory and cleanup all resources."""
        logger.info("Shutting down SamplerFactory")
        
        # Shutdown pool
        if self.enable_pooling and self.sampler_pool:
            self.sampler_pool.shutdown()
        
        # Clear cache
        with self._cache_lock:
            for sampler, _ in self._sampler_cache.values():
                if hasattr(sampler, '_cleanup'):
                    try:
                        sampler._cleanup()
                    except Exception as e:
                        logger.warning(f"Error cleaning up cached sampler: {e}")
            
            self._sampler_cache.clear()
        
        logger.info("SamplerFactory shutdown complete")


# Global factory instance for convenience
_global_factory: Optional[SamplerFactory] = None


def get_global_factory() -> SamplerFactory:
    """
    Get the global sampler factory instance.
    
    Returns:
        Global SamplerFactory singleton
    """
    global _global_factory
    if _global_factory is None:
        _global_factory = SamplerFactory()
    return _global_factory


# Utility functions for common operations
def create_sampler(
    config: Union[GenerationConfig, EnhancedGenerationConfig, ExperimentConfig],
    phase: str = "global",
    tokenizer: Optional[TokenizerWrapper] = None,
    **overrides
) -> SamplerInstance:
    """
    Convenience function for creating samplers.
    
    Args:
        config: Configuration object
        phase: Sampling phase (think/answer/global)
        tokenizer: Optional tokenizer
        **overrides: Parameter overrides
        
    Returns:
        Created sampler instance
        
    Raises:
        SamplerCreationError: If creation fails
    """
    factory = get_global_factory()
    return factory.create_sampler(config, phase, tokenizer, **overrides)


def create_mlx_sampler(
    parameters: Dict[str, Any],
    strategy: SamplerCreationStrategy = SamplerCreationStrategy.DIRECT_MLX
) -> MLXSampler:
    """
    Create a direct MLX sampler from parameters.
    
    Args:
        parameters: MLX sampler parameters
        strategy: Creation strategy
        
    Returns:
        MLX sampler instance
    """
    if strategy == SamplerCreationStrategy.DIRECT_MLX:
        return make_sampler(**parameters)
    else:
        raise ValueError(f"Strategy {strategy} not supported for direct MLX creation")


# Export public API
__all__ = [
    # Core classes
    "SamplerFactory",
    "MLXSamplerAdapter",
    "SamplerPool",
    "SamplerCreationContext",
    "SamplerCreationResult",
    
    # Enums
    "SamplerCreationStrategy",
    "SamplerLifecycle",
    
    # Protocols
    "SamplerFactoryProtocol",
    
    # Exceptions
    "SamplerFactoryError",
    "SamplerCreationError",
    "SamplerPoolExhaustedError",
    
    # Utilities
    "get_global_factory",
    "create_sampler",
    "create_mlx_sampler",
]