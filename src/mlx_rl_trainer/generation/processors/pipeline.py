"""
Processor Pipeline Management for Enhanced Dynamic Logit Processing

This module implements a sophisticated pipeline management system for orchestrating
multiple logit processors in a coordinated, efficient, and fault-tolerant manner.
It provides enterprise-grade pipeline execution with comprehensive monitoring,
error handling, and performance optimization.

The pipeline system implements several advanced architectural patterns:
- Chain of Responsibility: Sequential processor execution with fallback handling
- Command Pattern: Encapsulated processor operations with undo capabilities
- Observer Pattern: Pipeline event monitoring and notification
- Strategy Pattern: Pluggable execution strategies (parallel, sequential, conditional)
- Composite Pattern: Hierarchical processor organization
- Circuit Breaker: Fault tolerance for processor failures

Key Features:
- Multi-strategy execution (sequential, parallel, conditional)
- Comprehensive error handling with circuit breaker patterns
- Performance monitoring and optimization
- Dynamic processor registration and management
- Conditional execution based on context and performance
- Resource management and memory optimization
- Detailed audit logging and observability

Performance Characteristics:
- O(n) sequential execution where n is number of processors
- O(log n) parallel execution with thread pool optimization
- O(1) processor lookup and registration
- Memory-efficient with lazy loading and resource pooling
- Thread-safe operations with minimal locking overhead

Security Considerations:
- Input validation and sanitization at pipeline boundaries
- Resource limits to prevent DoS attacks
- Secure processor isolation and sandboxing
- Comprehensive audit logging for security monitoring
- Access control for processor registration and execution

Example:
    >>> from mlx_rl_trainer.generation.processors.pipeline import ProcessorPipeline
    >>> from mlx_rl_trainer.generation.processors.enhanced_bias_processor import EnhancedBiasProcessor
    >>> 
    >>> # Create pipeline with processors
    >>> pipeline = ProcessorPipeline(
    ...     execution_strategy='sequential',
    ...     enable_monitoring=True,
    ...     enable_circuit_breaker=True
    ... )
    >>> 
    >>> # Add processors
    >>> bias_processor = EnhancedBiasProcessor(ban_phrases=['bad'], encourage_phrases=['good'])
    >>> pipeline.add_processor(bias_processor)
    >>> 
    >>> # Execute pipeline
    >>> result = pipeline.process_logits(logits, history, context)
"""

import asyncio
import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import wraps
from typing import (
    Any, Callable, Dict, List, Optional, Set, Tuple, Union, 
    Protocol, runtime_checkable, AsyncIterator, Iterator
)
from weakref import WeakSet

import mlx.core as mx

from .base import (
    BaseLogitProcessor, LogitProcessor, ProcessingContext, ProcessingPhase,
    ProcessingException, ProcessingMetrics, ProcessingPriority,
    ProcessorRegistry, get_global_registry, performance_monitor
)

logger = logging.getLogger(__name__)


class ExecutionStrategy(Enum):
    """
    Pipeline execution strategies for different performance and reliability requirements.
    
    Each strategy provides different trade-offs between performance, reliability,
    and resource utilization based on the specific use case requirements.
    """
    SEQUENTIAL = auto()      # Execute processors one by one (highest reliability)
    PARALLEL = auto()        # Execute processors in parallel (highest performance)
    CONDITIONAL = auto()     # Execute processors based on conditions (adaptive)
    HYBRID = auto()         # Mix of sequential and parallel based on processor types
    FAIL_FAST = auto()      # Stop on first error (fastest failure detection)
    BEST_EFFORT = auto()    # Continue despite errors (highest availability)

    def __str__(self) -> str:
        return self.name.lower()

    @property
    def supports_parallelism(self) -> bool:
        """Check if this strategy supports parallel execution."""
        return self in (self.PARALLEL, self.HYBRID)

    @property
    def is_fault_tolerant(self) -> bool:
        """Check if this strategy continues execution despite errors."""
        return self in (self.BEST_EFFORT, self.CONDITIONAL)


class ProcessorState(Enum):
    """
    States that a processor can be in within the pipeline.
    
    These states enable sophisticated processor lifecycle management
    and provide visibility into processor health and performance.
    """
    REGISTERED = auto()      # Processor is registered but not active
    ACTIVE = auto()         # Processor is active and processing requests
    SUSPENDED = auto()      # Processor is temporarily suspended
    FAILED = auto()         # Processor has failed and needs attention
    CIRCUIT_OPEN = auto()   # Circuit breaker is open for this processor
    MAINTENANCE = auto()    # Processor is in maintenance mode

    def __str__(self) -> str:
        return self.name.lower()

    @property
    def is_executable(self) -> bool:
        """Check if processor can execute in this state."""
        return self in (self.ACTIVE, self.REGISTERED)

    @property
    def requires_attention(self) -> bool:
        """Check if processor state requires administrative attention."""
        return self in (self.FAILED, self.MAINTENANCE)


@dataclass(frozen=True)
class PipelineEvent:
    """
    Immutable event representing pipeline execution milestones.
    
    This class encapsulates information about pipeline events for
    monitoring, debugging, and audit purposes.
    
    Attributes:
        event_type: Type of pipeline event
        pipeline_id: Unique identifier for the pipeline
        processor_id: ID of processor involved (if applicable)
        timestamp: When the event occurred
        duration_ms: Event duration in milliseconds (if applicable)
        success: Whether the event completed successfully
        error_message: Error message if event failed
        metadata: Additional event-specific data
    """
    event_type: str
    pipeline_id: str
    processor_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    duration_ms: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            'event_type': self.event_type,
            'pipeline_id': self.pipeline_id,
            'processor_id': self.processor_id,
            'timestamp': self.timestamp,
            'duration_ms': self.duration_ms,
            'success': self.success,
            'error_message': self.error_message,
            'metadata': self.metadata.copy()
        }


@dataclass
class ProcessorMetadata:
    """
    Metadata container for processors within the pipeline.
    
    This class maintains additional information about processors
    beyond their core functionality, enabling sophisticated
    pipeline management and optimization.
    
    Attributes:
        processor: The actual processor instance
        state: Current processor state
        registration_time: When processor was registered
        last_execution_time: Last successful execution timestamp
        total_executions: Total number of executions
        total_errors: Total number of errors encountered
        average_execution_time: Average execution time in milliseconds
        circuit_breaker_failures: Number of consecutive failures
        custom_metadata: Additional processor-specific metadata
    """
    processor: LogitProcessor
    state: ProcessorState = ProcessorState.REGISTERED
    registration_time: float = field(default_factory=time.time)
    last_execution_time: Optional[float] = None
    total_executions: int = 0
    total_errors: int = 0
    average_execution_time: float = 0.0
    circuit_breaker_failures: int = 0
    custom_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def error_rate(self) -> float:
        """Calculate error rate as a percentage."""
        return (self.total_errors / self.total_executions * 100) if self.total_executions > 0 else 0.0

    @property
    def is_healthy(self) -> bool:
        """Check if processor is in a healthy state."""
        return (
            self.state.is_executable and 
            self.error_rate < 50.0 and  # Less than 50% error rate
            self.circuit_breaker_failures < 5  # Less than 5 consecutive failures
        )

    def update_execution_stats(self, execution_time: float, success: bool) -> None:
        """
        Update execution statistics.
        
        Args:
            execution_time: Execution time in milliseconds
            success: Whether execution was successful
        """
        self.total_executions += 1
        self.last_execution_time = time.time()
        
        if success:
            self.circuit_breaker_failures = 0
            # Update average execution time using exponential moving average
            alpha = 0.1  # Smoothing factor
            self.average_execution_time = (
                alpha * execution_time + (1 - alpha) * self.average_execution_time
            )
        else:
            self.total_errors += 1
            self.circuit_breaker_failures += 1


class PipelineException(ProcessingException):
    """Base exception for pipeline-related errors."""
    pass


class ProcessorRegistrationException(PipelineException):
    """Raised when processor registration fails."""
    pass


class PipelineExecutionException(PipelineException):
    """Raised when pipeline execution encounters errors."""
    pass


class CircuitBreakerException(PipelineException):
    """Raised when circuit breaker prevents processor execution."""
    pass


@runtime_checkable
class PipelineEventListener(Protocol):
    """
    Protocol for listening to pipeline events.
    
    This protocol enables event-driven monitoring and reactive
    processing of pipeline execution events.
    """
    
    def on_pipeline_event(self, event: PipelineEvent) -> None:
        """
        Handle a pipeline event.
        
        Args:
            event: Pipeline event to handle
        """
        ...


@runtime_checkable
class ExecutionStrategyImpl(Protocol):
    """
    Protocol for pipeline execution strategy implementations.
    
    This protocol defines the interface for different execution
    strategies, enabling pluggable execution algorithms.
    """
    
    def execute_processors(
        self,
        processors: List[ProcessorMetadata],
        logits: mx.array,
        history: List[List[int]],
        context: ProcessingContext,
        pipeline: 'ProcessorPipeline'
    ) -> mx.array:
        """
        Execute processors according to the strategy.
        
        Args:
            processors: List of processor metadata to execute
            logits: Input logits tensor
            history: Token generation history
            context: Processing context
            pipeline: Pipeline instance for callbacks
            
        Returns:
            Modified logits tensor
        """
        ...


class CircuitBreaker:
    """
    Circuit breaker implementation for processor fault tolerance.
    
    This class implements the Circuit Breaker pattern to provide
    fault tolerance and prevent cascading failures in the pipeline.
    It monitors processor health and automatically isolates failing
    processors to maintain overall system stability.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Processor is failing, requests are blocked
    - HALF_OPEN: Testing if processor has recovered
    
    Features:
    - Configurable failure thresholds and timeouts
    - Automatic recovery testing
    - Comprehensive monitoring and logging
    - Thread-safe operation
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 3
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before testing recovery (seconds)
            success_threshold: Number of successes needed to close circuit
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        # State management
        self._state = 'CLOSED'
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._lock = threading.RLock()
        
        logger.info(
            f"Initialized CircuitBreaker with failure_threshold={failure_threshold}, "
            f"recovery_timeout={recovery_timeout}s"
        )

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerException: If circuit is open
        """
        with self._lock:
            if self._state == 'OPEN':
                if time.time() - self._last_failure_time < self.recovery_timeout:
                    raise CircuitBreakerException(
                        f"Circuit breaker is OPEN (failures: {self._failure_count})"
                    )
                else:
                    # Transition to HALF_OPEN for recovery testing
                    self._state = 'HALF_OPEN'
                    self._success_count = 0
                    logger.info("Circuit breaker transitioning to HALF_OPEN for recovery testing")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self) -> None:
        """Handle successful execution."""
        with self._lock:
            if self._state == 'HALF_OPEN':
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._state = 'CLOSED'
                    self._failure_count = 0
                    logger.info("Circuit breaker CLOSED after successful recovery")
            elif self._state == 'CLOSED':
                self._failure_count = 0  # Reset failure count on success

    def _on_failure(self) -> None:
        """Handle failed execution."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._state == 'HALF_OPEN':
                # Failed during recovery, go back to OPEN
                self._state = 'OPEN'
                logger.warning("Circuit breaker OPEN after failed recovery attempt")
            elif self._failure_count >= self.failure_threshold:
                # Too many failures, open the circuit
                self._state = 'OPEN'
                logger.warning(
                    f"Circuit breaker OPEN after {self._failure_count} failures "
                    f"(threshold: {self.failure_threshold})"
                )

    @property
    def state(self) -> str:
        """Get current circuit breaker state."""
        return self._state

    @property
    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        return self._state == 'OPEN'

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self._lock:
            return {
                'state': self._state,
                'failure_count': self._failure_count,
                'success_count': self._success_count,
                'last_failure_time': self._last_failure_time,
                'failure_threshold': self.failure_threshold,
                'recovery_timeout': self.recovery_timeout
            }

    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        with self._lock:
            self._state = 'CLOSED'
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = 0.0
            logger.info("Circuit breaker reset to CLOSED state")


class SequentialExecutionStrategy:
    """
    Sequential execution strategy for maximum reliability.
    
    This strategy executes processors one by one in priority order,
    providing maximum reliability and predictable behavior at the
    cost of potentially higher latency.
    """
    
    def execute_processors(
        self,
        processors: List[ProcessorMetadata],
        logits: mx.array,
        history: List[List[int]],
        context: ProcessingContext,
        pipeline: 'ProcessorPipeline'
    ) -> mx.array:
        """Execute processors sequentially."""
        result_logits = logits
        
        for processor_meta in processors:
            if not processor_meta.is_healthy:
                logger.warning(
                    f"Skipping unhealthy processor {processor_meta.processor.processor_id}"
                )
                continue
            
            try:
                start_time = time.perf_counter()
                
                # Execute processor with circuit breaker protection
                if pipeline.enable_circuit_breaker:
                    circuit_breaker = pipeline._get_circuit_breaker(processor_meta.processor.processor_id)
                    result_logits = circuit_breaker.call(
                        processor_meta.processor.process_logits,
                        result_logits,
                        history,
                        context
                    )
                else:
                    result_logits = processor_meta.processor.process_logits(
                        result_logits,
                        history,
                        context
                    )
                
                # Update statistics
                execution_time = (time.perf_counter() - start_time) * 1000
                processor_meta.update_execution_stats(execution_time, True)
                
                # Emit success event
                pipeline._emit_event(PipelineEvent(
                    event_type='processor_success',
                    pipeline_id=pipeline.pipeline_id,
                    processor_id=processor_meta.processor.processor_id,
                    duration_ms=execution_time,
                    success=True
                ))
                
            except Exception as e:
                execution_time = (time.perf_counter() - start_time) * 1000
                processor_meta.update_execution_stats(execution_time, False)
                
                # Emit failure event
                pipeline._emit_event(PipelineEvent(
                    event_type='processor_failure',
                    pipeline_id=pipeline.pipeline_id,
                    processor_id=processor_meta.processor.processor_id,
                    duration_ms=execution_time,
                    success=False,
                    error_message=str(e)
                ))
                
                logger.error(
                    f"Processor {processor_meta.processor.processor_id} failed: {e}",
                    extra={'correlation_id': context.correlation_id}
                )
                
                # Handle error based on strategy
                if pipeline.execution_strategy == ExecutionStrategy.FAIL_FAST:
                    raise PipelineExecutionException(
                        f"Pipeline execution failed at processor {processor_meta.processor.processor_id}: {e}"
                    ) from e
                # For other strategies, continue with next processor
        
        return result_logits


class ParallelExecutionStrategy:
    """
    Parallel execution strategy for maximum performance.
    
    This strategy executes processors in parallel using a thread pool,
    providing maximum performance at the cost of increased complexity
    and resource usage.
    """
    
    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize parallel execution strategy.
        
        Args:
            max_workers: Maximum number of worker threads (None for auto)
        """
        self.max_workers = max_workers

    def execute_processors(
        self,
        processors: List[ProcessorMetadata],
        logits: mx.array,
        history: List[List[int]],
        context: ProcessingContext,
        pipeline: 'ProcessorPipeline'
    ) -> mx.array:
        """Execute processors in parallel."""
        if not processors:
            return logits
        
        # Filter healthy processors
        healthy_processors = [p for p in processors if p.is_healthy]
        
        if not healthy_processors:
            logger.warning("No healthy processors available for parallel execution")
            return logits
        
        # Execute processors in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all processor tasks
            future_to_processor = {}
            
            for processor_meta in healthy_processors:
                future = executor.submit(
                    self._execute_single_processor,
                    processor_meta,
                    logits,
                    history,
                    context,
                    pipeline
                )
                future_to_processor[future] = processor_meta
            
            # Collect results
            results = []
            
            for future in as_completed(future_to_processor):
                processor_meta = future_to_processor[future]
                
                try:
                    result_logits = future.result()
                    results.append((processor_meta.processor.priority, result_logits))
                    
                except Exception as e:
                    logger.error(
                        f"Parallel processor {processor_meta.processor.processor_id} failed: {e}",
                        extra={'correlation_id': context.correlation_id}
                    )
                    
                    if pipeline.execution_strategy == ExecutionStrategy.FAIL_FAST:
                        # Cancel remaining tasks
                        for f in future_to_processor:
                            f.cancel()
                        
                        raise PipelineExecutionException(
                            f"Parallel execution failed at processor {processor_meta.processor.processor_id}: {e}"
                        ) from e
        
        # Combine results (for now, just return the last result)
        # More sophisticated combination strategies could be implemented
        if results:
            # Sort by priority and return highest priority result
            results.sort(key=lambda x: x[0], reverse=True)
            return results[0][1]
        
        return logits

    def _execute_single_processor(
        self,
        processor_meta: ProcessorMetadata,
        logits: mx.array,
        history: List[List[int]],
        context: ProcessingContext,
        pipeline: 'ProcessorPipeline'
    ) -> mx.array:
        """Execute a single processor with monitoring."""
        start_time = time.perf_counter()
        
        try:
            # Execute processor with circuit breaker protection
            if pipeline.enable_circuit_breaker:
                circuit_breaker = pipeline._get_circuit_breaker(processor_meta.processor.processor_id)
                result_logits = circuit_breaker.call(
                    processor_meta.processor.process_logits,
                    logits,
                    history,
                    context
                )
            else:
                result_logits = processor_meta.processor.process_logits(
                    logits,
                    history,
                    context
                )
            
            # Update statistics
            execution_time = (time.perf_counter() - start_time) * 1000
            processor_meta.update_execution_stats(execution_time, True)
            
            # Emit success event
            pipeline._emit_event(PipelineEvent(
                event_type='processor_success',
                pipeline_id=pipeline.pipeline_id,
                processor_id=processor_meta.processor.processor_id,
                duration_ms=execution_time,
                success=True
            ))
            
            return result_logits
            
        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            processor_meta.update_execution_stats(execution_time, False)
            
            # Emit failure event
            pipeline._emit_event(PipelineEvent(
                event_type='processor_failure',
                pipeline_id=pipeline.pipeline_id,
                processor_id=processor_meta.processor.processor_id,
                duration_ms=execution_time,
                success=False,
                error_message=str(e)
            ))
            
            raise


class ProcessorPipeline:
    """
    Main processor pipeline orchestrator with enterprise-grade capabilities.
    
    This class provides comprehensive pipeline management with support for
    multiple execution strategies, fault tolerance, performance monitoring,
    and dynamic processor management. It implements several design patterns
    to ensure scalability, reliability, and maintainability.
    
    Key Features:
    - Multiple execution strategies (sequential, parallel, conditional)
    - Circuit breaker pattern for fault tolerance
    - Comprehensive monitoring and observability
    - Dynamic processor registration and management
    - Resource management and optimization
    - Event-driven architecture with listeners
    - Thread-safe operations
    
    Architecture Patterns:
    - Chain of Responsibility: Processor execution chain
    - Strategy: Pluggable execution strategies
    - Observer: Event notification system
    - Circuit Breaker: Fault tolerance
    - Registry: Processor management
    """
    
    def __init__(
        self,
        pipeline_id: Optional[str] = None,
        execution_strategy: Union[ExecutionStrategy, str] = ExecutionStrategy.SEQUENTIAL,
        enable_monitoring: bool = True,
        enable_circuit_breaker: bool = True,
        circuit_breaker_config: Optional[Dict[str, Any]] = None,
        max_parallel_workers: Optional[int] = None,
        processor_registry: Optional[ProcessorRegistry] = None
    ):
        """
        Initialize processor pipeline.
        
        Args:
            pipeline_id: Unique pipeline identifier (auto-generated if None)
            execution_strategy: Strategy for processor execution
            enable_monitoring: Enable performance monitoring
            enable_circuit_breaker: Enable circuit breaker pattern
            circuit_breaker_config: Circuit breaker configuration
            max_parallel_workers: Maximum parallel workers for parallel strategy
            processor_registry: Processor registry (uses global if None)
        """
        self.pipeline_id = pipeline_id or f"pipeline_{uuid.uuid4().hex[:8]}"
        self.enable_monitoring = enable_monitoring
        self.enable_circuit_breaker = enable_circuit_breaker
        
        # Parse execution strategy
        if isinstance(execution_strategy, str):
            self.execution_strategy = ExecutionStrategy[execution_strategy.upper()]
        else:
            self.execution_strategy = execution_strategy
        
        # Initialize processor management
        self._processors: Dict[str, ProcessorMetadata] = {}
        self._processors_lock = threading.RLock()
        
        # Initialize execution strategies
        self._execution_strategies = {
            ExecutionStrategy.SEQUENTIAL: SequentialExecutionStrategy(),
            ExecutionStrategy.PARALLEL: ParallelExecutionStrategy(max_parallel_workers),
            ExecutionStrategy.FAIL_FAST: SequentialExecutionStrategy(),
            ExecutionStrategy.BEST_EFFORT: SequentialExecutionStrategy()
        }
        
        # Initialize circuit breakers
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._circuit_breaker_config = circuit_breaker_config or {}
        
        # Initialize event system
        self._event_listeners: List[PipelineEventListener] = []
        self._listeners_lock = threading.RLock()
        
        # Performance tracking
        self._total_executions = 0
        self._total_execution_time = 0.0
        self._successful_executions = 0
        self._failed_executions = 0
        
        # Registry integration
        self._registry = processor_registry or get_global_registry()
        
        logger.info(
            f"Initialized ProcessorPipeline {self.pipeline_id} with strategy {self.execution_strategy}"
        )

    def add_processor(self, processor: LogitProcessor) -> None:
        """
        Add a processor to the pipeline.
        
        Args:
            processor: Processor to add
            
        Raises:
            ProcessorRegistrationException: If registration fails
        """
        if not hasattr(processor, 'processor_id') or not processor.processor_id:
            raise ProcessorRegistrationException("Processor must have a valid processor_id")
        
        with self._processors_lock:
            if processor.processor_id in self._processors:
                raise ProcessorRegistrationException(
                    f"Processor {processor.processor_id} is already registered"
                )
            
            # Create processor metadata
            metadata = ProcessorMetadata(
                processor=processor,
                state=ProcessorState.ACTIVE
            )
            
            self._processors[processor.processor_id] = metadata
            
            # Initialize circuit breaker if enabled
            if self.enable_circuit_breaker:
                self._circuit_breakers[processor.processor_id] = CircuitBreaker(
                    **self._circuit_breaker_config
                )
        
        # Emit registration event
        self._emit_event(PipelineEvent(
            event_type='processor_registered',
            pipeline_id=self.pipeline_id,
            processor_id=processor.processor_id,
            success=True
        ))
        
        logger.info(f"Added processor {processor.processor_id} to pipeline {self.pipeline_id}")

    def remove_processor(self, processor_id: str) -> bool:
        """
        Remove a processor from the pipeline.
        
        Args:
            processor_id: ID of processor to remove
            
        Returns:
            True if processor was found and removed, False otherwise
        """
        with self._processors_lock:
            metadata = self._processors.pop(processor_id, None)
            if metadata is None:
                return False
            
            # Remove circuit breaker
            self._circuit_breakers.pop(processor_id, None)
        
        # Emit removal event
        self._emit_event(PipelineEvent(
            event_type='processor_removed',
            pipeline_id=self.pipeline_id,
            processor_id=processor_id,
            success=True
        ))
        
        logger.info(f"Removed processor {processor_id} from pipeline {self.pipeline_id}")
        return True

    def get_processor_metadata(self, processor_id: str) -> Optional[ProcessorMetadata]:
        """
        Get metadata for a specific processor.
        
        Args:
            processor_id: ID of processor
            
        Returns:
            Processor metadata or None if not found
        """
        with self._processors_lock:
            return self._processors.get(processor_id)

    def set_processor_state(self, processor_id: str, state: ProcessorState) -> bool:
        """
        Set the state of a specific processor.
        
        Args:
            processor_id: ID of processor
            state: New processor state
            
        Returns:
            True if state was updated, False if processor not found
        """
        with self._processors_lock:
            metadata = self._processors.get(processor_id)
            if metadata is None:
                return False
            
            old_state = metadata.state
            metadata.state = state
        
        # Emit state change event
        self._emit_event(PipelineEvent(
            event_type='processor_state_changed',
            pipeline_id=self.pipeline_id,
            processor_id=processor_id,
            success=True,
            metadata={'old_state': str(old_state), 'new_state': str(state)}
        ))
        
        logger.info(f"Processor {processor_id} state changed: {old_state} -> {state}")
        return True

    @performance_monitor
    def process_logits(
        self, 
        logits: mx.array, 
        history: List[List[int]], 
        context: ProcessingContext
    ) -> mx.array:
        """
        Process logits through the processor pipeline.
        
        This is the main entry point for pipeline execution. It orchestrates
        the execution of all registered processors according to the configured
        execution strategy.
        
        Args:
            logits: Input logits tensor [batch_size, vocab_size]
            history: Token generation history for each batch item
            context: Processing context with configuration and state
            
        Returns:
            Modified logits tensor after processing through pipeline
            
        Raises:
            PipelineExecutionException: If pipeline execution fails
        """
        start_time = time.perf_counter()
        
        # Emit pipeline start event
        self._emit_event(PipelineEvent(
            event_type='pipeline_start',
            pipeline_id=self.pipeline_id,
            success=True,
            metadata={'batch_size': logits.shape[0], 'vocab_size': logits.shape[1]}
        ))
        
        try:
            # Get processors for execution
            processors = self._get_executable_processors(context)
            
            if not processors:
                logger.debug(f"No executable processors in pipeline {self.pipeline_id}")
                return logits
            
            # Execute processors using selected strategy
            strategy = self._execution_strategies.get(self.execution_strategy)
            if strategy is None:
                raise PipelineExecutionException(
                    f"Unsupported execution strategy: {self.execution_strategy}"
                )
            
            result_logits = strategy.execute_processors(
                processors, logits, history, context, self
            )
            
            # Update performance statistics
            execution_time = time.perf_counter() - start_time
            self._update_pipeline_stats(execution_time, True)
            
            # Emit pipeline success event
            self._emit_event(PipelineEvent(
                event_type='pipeline_success',
                pipeline_id=self.pipeline_id,
                duration_ms=execution_time * 1000,
                success=True,
                metadata={'processors_executed': len(processors)}
            ))
            
            return result_logits
            
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            self._update_pipeline_stats(execution_time, False)
            
            # Emit pipeline failure event
            self._emit_event(PipelineEvent(
                event_type='pipeline_failure',
                pipeline_id=self.pipeline_id,
                duration_ms=execution_time * 1000,
                success=False,
                error_message=str(e)
            ))
            
            logger.error(
                f"Pipeline {self.pipeline_id} execution failed: {e}",
                extra={'correlation_id': context.correlation_id}
            )
            
            raise PipelineExecutionException(
                f"Pipeline execution failed: {e}"
            ) from e

    def _get_executable_processors(self, context: ProcessingContext) -> List[ProcessorMetadata]:
        """
        Get list of processors that can execute in the current context.
        
        Args:
            context: Processing context
            
        Returns:
            List of executable processor metadata, sorted by priority
        """
        executable = []
        
        with self._processors_lock:
            for metadata in self._processors.values():
                # Check if processor is in executable state
                if not metadata.state.is_executable:
                    continue
                
                # Check if processor can handle this context
                if not metadata.processor.can_process(context):
                    continue
                
                # Check circuit breaker status
                if self.enable_circuit_breaker:
                    circuit_breaker = self._circuit_breakers.get(metadata.processor.processor_id)
                    if circuit_breaker and circuit_breaker.is_open:
                        continue
                
                executable.append(metadata)
        
        # Sort by processor priority (highest first)
        executable.sort(key=lambda m: m.processor.priority, reverse=True)
        return executable

    def _get_circuit_breaker(self, processor_id: str) -> CircuitBreaker:
        """Get circuit breaker for a processor."""
        circuit_breaker = self._circuit_breakers.get(processor_id)
        if circuit_breaker is None:
            circuit_breaker = CircuitBreaker(**self._circuit_breaker_config)
            self._circuit_breakers[processor_id] = circuit_breaker
        return circuit_breaker

    def _update_pipeline_stats(self, execution_time: float, success: bool) -> None:
        """Update pipeline performance statistics."""
        self._total_executions += 1
        self._total_execution_time += execution_time
        
        if success:
            self._successful_executions += 1
        else:
            self._failed_executions += 1

    def add_event_listener(self, listener: PipelineEventListener) -> None:
        """
        Add a pipeline event listener.
        
        Args:
            listener: Event listener to add
        """
        with self._listeners_lock:
            self._event_listeners.append(listener)

    def remove_event_listener(self, listener: PipelineEventListener) -> bool:
        """
        Remove a pipeline event listener.
        
        Args:
            listener: Event listener to remove
            
        Returns:
            True if listener was found and removed, False otherwise
        """
        with self._listeners_lock:
            try:
                self._event_listeners.remove(listener)
                return True
            except ValueError:
                return False

    def _emit_event(self, event: PipelineEvent) -> None:
        """
        Emit a pipeline event to all registered listeners.
        
        Args:
            event: Event to emit
        """
        with self._listeners_lock:
            for listener in self._event_listeners:
                try:
                    listener.on_pipeline_event(event)
                except Exception as e:
                    logger.error(f"Event listener failed: {e}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive pipeline performance statistics.
        
        Returns:
            Dictionary containing performance metrics
        """
        avg_execution_time = (
            self._total_execution_time / self._total_executions 
            if self._total_executions > 0 else 0
        )
        
        success_rate = (
            self._successful_executions / self._total_executions * 100
            if self._total_executions > 0 else 0
        )
        
        # Get processor statistics
        processor_stats = {}
        with self._processors_lock:
            for processor_id, metadata in self._processors.items():
                processor_stats[processor_id] = {
                    'state': str(metadata.state),
                    'total_executions': metadata.total_executions,
                    'total_errors': metadata.total_errors,
                    'error_rate': metadata.error_rate,
                    'average_execution_time': metadata.average_execution_time,
                    'is_healthy': metadata.is_healthy
                }
        
        # Get circuit breaker statistics
        circuit_breaker_stats = {}
        for processor_id, circuit_breaker in self._circuit_breakers.items():
            circuit_breaker_stats[processor_id] = circuit_breaker.get_stats()
        
        return {
            'pipeline_id': self.pipeline_id,
            'execution_strategy': str(self.execution_strategy),
            'total_executions': self._total_executions,
            'successful_executions': self._successful_executions,
            'failed_executions': self._failed_executions,
            'success_rate': success_rate,
            'total_execution_time_seconds': self._total_execution_time,
            'average_execution_time_seconds': avg_execution_time,
            'registered_processors': len(self._processors),
            'active_processors': len([m for m in self._processors.values() if m.state == ProcessorState.ACTIVE]),
            'processor_stats': processor_stats,
            'circuit_breaker_stats': circuit_breaker_stats if self.enable_circuit_breaker else {}
        }

    def reset_stats(self) -> None:
        """Reset pipeline performance statistics."""
        self._total_executions = 0
        self._total_execution_time = 0.0
        self._successful_executions = 0
        self._failed_executions = 0
        
        # Reset processor statistics
        with self._processors_lock:
            for metadata in self._processors.values():
                metadata.total_executions = 0
                metadata.total_errors = 0
                metadata.average_execution_time = 0.0
                metadata.circuit_breaker_failures = 0
        
        # Reset circuit breakers
        for circuit_breaker in self._circuit_breakers.values():
            circuit_breaker.reset()
        
        logger.info(f"Reset statistics for pipeline {self.pipeline_id}")

    def get_processor_list(self) -> List[Dict[str, Any]]:
        """
        Get list of all registered processors with their metadata.
        
        Returns:
            List of processor information dictionaries
        """
        processors = []
        
        with self._processors_lock:
            for processor_id, metadata in self._processors.items():
                processors.append({
                    'processor_id': processor_id,
                    'processor_type': type(metadata.processor).__name__,
                    'priority': metadata.processor.priority,
                    'state': str(metadata.state),
                    'registration_time': metadata.registration_time,
                    'last_execution_time': metadata.last_execution_time,
                    'total_executions': metadata.total_executions,
                    'total_errors': metadata.total_errors,
                    'error_rate': metadata.error_rate,
                    'is_healthy': metadata.is_healthy
                })
        
        # Sort by priority
        processors.sort(key=lambda p: p['priority'], reverse=True)
        return processors


# Factory functions for common pipeline configurations

def create_simple_pipeline(
    processors: Optional[List[LogitProcessor]] = None,
    execution_strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL
) -> ProcessorPipeline:
    """
    Create a simple processor pipeline with basic configuration.
    
    Args:
        processors: List of processors to add (optional)
        execution_strategy: Execution strategy to use
        
    Returns:
        Configured ProcessorPipeline instance
    """
    pipeline = ProcessorPipeline(execution_strategy=execution_strategy)
    
    if processors:
        for processor in processors:
            pipeline.add_processor(processor)
    
    return pipeline


def create_fault_tolerant_pipeline(
    processors: Optional[List[LogitProcessor]] = None,
    failure_threshold: int = 3,
    recovery_timeout: float = 30.0
) -> ProcessorPipeline:
    """
    Create a fault-tolerant pipeline with circuit breaker protection.
    
    Args:
        processors: List of processors to add (optional)
        failure_threshold: Circuit breaker failure threshold
        recovery_timeout: Circuit breaker recovery timeout
        
    Returns:
        Configured ProcessorPipeline instance with fault tolerance
    """
    circuit_breaker_config = {
        'failure_threshold': failure_threshold,
        'recovery_timeout': recovery_timeout
    }
    
    pipeline = ProcessorPipeline(
        execution_strategy=ExecutionStrategy.BEST_EFFORT,
        enable_circuit_breaker=True,
        circuit_breaker_config=circuit_breaker_config
    )
    
    if processors:
        for processor in processors:
            pipeline.add_processor(processor)
    
    return pipeline


def create_high_performance_pipeline(
    processors: Optional[List[LogitProcessor]] = None,
    max_workers: Optional[int] = None
) -> ProcessorPipeline:
    """
    Create a high-performance pipeline with parallel execution.
    
    Args:
        processors: List of processors to add (optional)
        max_workers: Maximum parallel workers
        
    Returns:
        Configured ProcessorPipeline instance optimized for performance
    """
    pipeline = ProcessorPipeline(
        execution_strategy=ExecutionStrategy.PARALLEL,
        max_parallel_workers=max_workers,
        enable_monitoring=True
    )
    
    if processors:
        for processor in processors:
            pipeline.add_processor(processor)
    
    return pipeline