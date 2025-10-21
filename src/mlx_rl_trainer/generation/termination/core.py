"""
Core termination system implementation with enterprise-grade architecture.

This module implements the main GenerationTerminator class and supporting
infrastructure for robust text generation termination with comprehensive
EOS token handling, multiple stopping criteria, and advanced monitoring.

Architecture:
    - GenerationTerminator: Main orchestrator with strategy coordination
    - TerminationResult: Immutable result object with comprehensive metadata
    - TerminationState: State management with thread-safe operations
    - Custom exception hierarchy for structured error handling
    - Performance monitoring with detailed metrics collection

Design Patterns:
    - Facade Pattern: Simplified interface for complex termination logic
    - State Pattern: Termination state management with transitions
    - Observer Pattern: Event-driven monitoring and notifications
    - Command Pattern: Encapsulated termination operations
    - Circuit Breaker Pattern: Fault tolerance and graceful degradation

SOLID Principles Implementation:
    - Single Responsibility: Each class has one clear purpose
    - Open/Closed: Extensible through strategy pattern
    - Liskov Substitution: All strategies implement common interface
    - Interface Segregation: Focused interfaces for specific concerns
    - Dependency Inversion: Depends on abstractions, not concretions

Example:
    >>> from mlx_rl_trainer.generation.termination.core import GenerationTerminator
    >>> from mlx_rl_trainer.generation.termination.config import TerminationConfig
    >>> 
    >>> config = TerminationConfig(max_tokens=512, eos_tokens=['</s>'])
    >>> terminator = GenerationTerminator(config)
    >>> result = terminator.should_terminate(tokens, logits, step=100)
    >>> if result.should_stop:
    ...     print(f"Termination reason: {result.reason}")
"""

import logging
import time
import threading
import uuid
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, Dict, List, Optional, Union, Callable, Protocol, TypeVar, Generic,
    runtime_checkable, ClassVar, Final, Tuple, NamedTuple, Set, FrozenSet
)
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, Future
import asyncio
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
from mlx_lm.tokenizer_utils import TokenizerWrapper

# Type definitions for enhanced type safety
T = TypeVar('T')
TokenArray = mx.array
LogitsArray = mx.array
TokenizerType = TypeVar('TokenizerType', bound=TokenizerWrapper)

logger = logging.getLogger(__name__)


class TerminationReason(Enum):
    """
    Enumeration of possible termination reasons.
    
    Provides comprehensive categorization of why generation was terminated,
    enabling detailed analysis and optimization of termination behavior.
    """
    EOS_TOKEN = "eos_token"                    # EOS token detected
    MAX_LENGTH = "max_length"                  # Maximum length reached
    REPETITION = "repetition"                  # Repetitive content detected
    QUALITY_THRESHOLD = "quality_threshold"    # Quality below threshold
    TIMEOUT = "timeout"                        # Generation timeout
    USER_STOP = "user_stop"                    # User-requested stop
    ERROR = "error"                           # Error condition
    CIRCUIT_BREAKER = "circuit_breaker"       # Circuit breaker triggered
    RESOURCE_LIMIT = "resource_limit"         # Resource constraints
    SAFETY_FILTER = "safety_filter"          # Safety filter triggered
    CUSTOM = "custom"                         # Custom termination logic
    UNKNOWN = "unknown"                       # Unknown reason


class TerminationState(Enum):
    """
    Enumeration of termination system states.
    
    Tracks the operational state of the termination system for
    proper lifecycle management and monitoring.
    """
    INITIALIZING = auto()
    READY = auto()
    EVALUATING = auto()
    TERMINATING = auto()
    ERROR = auto()
    DISABLED = auto()
    MAINTENANCE = auto()


class TerminationConfidence(Enum):
    """
    Enumeration of termination confidence levels.
    
    Indicates the confidence level of termination decisions
    for quality assessment and debugging purposes.
    """
    VERY_LOW = 0.1
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.7
    VERY_HIGH = 0.9
    CERTAIN = 1.0


@dataclass(frozen=True)
class TerminationResult:
    """
    Immutable result object for termination decisions.
    
    Contains comprehensive information about termination decisions,
    including reasoning, confidence levels, and diagnostic metadata
    for monitoring and optimization purposes.
    
    Attributes:
        should_stop: Whether generation should be terminated
        reason: Primary reason for termination decision
        confidence: Confidence level of the decision (0.0-1.0)
        metadata: Additional diagnostic information
        timestamp: When the decision was made
        correlation_id: Unique identifier for tracking
        strategy_results: Results from individual strategies
        performance_metrics: Performance data for the decision
        fallback_used: Whether fallback logic was employed
        warnings: List of warnings generated during evaluation
    """
    
    should_stop: bool
    reason: TerminationReason
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    strategy_results: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    fallback_used: bool = False
    warnings: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate result consistency."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        
        if self.should_stop and self.reason == TerminationReason.UNKNOWN:
            object.__setattr__(self, 'warnings', 
                             self.warnings + ["Termination without clear reason"])
    
    @property
    def confidence_level(self) -> TerminationConfidence:
        """Get confidence level enum from numeric confidence."""
        if self.confidence >= 0.9:
            return TerminationConfidence.CERTAIN
        elif self.confidence >= 0.7:
            return TerminationConfidence.VERY_HIGH
        elif self.confidence >= 0.5:
            return TerminationConfidence.HIGH
        elif self.confidence >= 0.3:
            return TerminationConfidence.MEDIUM
        elif self.confidence >= 0.1:
            return TerminationConfidence.LOW
        else:
            return TerminationConfidence.VERY_LOW
    
    @property
    def is_high_confidence(self) -> bool:
        """Check if decision has high confidence."""
        return self.confidence >= TerminationConfidence.HIGH.value
    
    @property
    def execution_time_ms(self) -> Optional[float]:
        """Get execution time in milliseconds if available."""
        return self.performance_metrics.get('execution_time_ms')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'should_stop': self.should_stop,
            'reason': self.reason.value,
            'confidence': self.confidence,
            'confidence_level': self.confidence_level.name,
            'metadata': self.metadata,
            'timestamp': self.timestamp,
            'correlation_id': self.correlation_id,
            'strategy_results': self.strategy_results,
            'performance_metrics': self.performance_metrics,
            'fallback_used': self.fallback_used,
            'warnings': self.warnings,
            'is_high_confidence': self.is_high_confidence
        }
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        status = "STOP" if self.should_stop else "CONTINUE"
        return (f"TerminationResult({status}, reason={self.reason.value}, "
                f"confidence={self.confidence:.2f})")


class TerminationError(Exception):
    """
    Base exception for termination system errors.
    
    Provides structured error reporting with context information
    for debugging and monitoring purposes.
    """
    
    def __init__(
        self,
        message: str,
        error_code: str = "TERMINATION_ERROR",
        terminator_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.terminator_id = terminator_id
        self.context = context or {}
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.timestamp = time.time()
        
        # Log the error with full context
        logger.error(
            f"Termination Error [{self.error_code}]: {message}",
            extra={
                "correlation_id": self.correlation_id,
                "terminator_id": self.terminator_id,
                "context": self.context,
                "timestamp": self.timestamp
            }
        )


class TerminationTimeoutError(TerminationError):
    """Exception for termination evaluation timeouts."""
    
    def __init__(self, timeout_seconds: float, **kwargs):
        message = f"Termination evaluation timed out after {timeout_seconds}s"
        super().__init__(message, error_code="TERMINATION_TIMEOUT", **kwargs)
        self.timeout_seconds = timeout_seconds


class InvalidTerminationConfigError(TerminationError):
    """Exception for invalid termination configuration."""
    
    def __init__(self, config_issue: str, **kwargs):
        message = f"Invalid termination configuration: {config_issue}"
        super().__init__(message, error_code="INVALID_CONFIG", **kwargs)
        self.config_issue = config_issue


class CircuitBreakerError(TerminationError):
    """Exception when circuit breaker is open."""
    
    def __init__(self, failure_count: int, **kwargs):
        message = f"Termination circuit breaker open after {failure_count} failures"
        super().__init__(message, error_code="CIRCUIT_BREAKER_OPEN", **kwargs)
        self.failure_count = failure_count


@runtime_checkable
class TerminationObserver(Protocol):
    """
    Protocol for termination event observers.
    
    Enables event-driven monitoring and notifications for
    termination decisions and system state changes.
    """
    
    def on_termination_decision(
        self,
        result: TerminationResult,
        context: Dict[str, Any]
    ) -> None:
        """
        Called when a termination decision is made.
        
        Args:
            result: The termination decision result
            context: Additional context information
        """
        ...
    
    def on_state_change(
        self,
        old_state: TerminationState,
        new_state: TerminationState,
        context: Dict[str, Any]
    ) -> None:
        """
        Called when termination system state changes.
        
        Args:
            old_state: Previous state
            new_state: New state
            context: Additional context information
        """
        ...
    
    def on_error(
        self,
        error: TerminationError,
        context: Dict[str, Any]
    ) -> None:
        """
        Called when a termination error occurs.
        
        Args:
            error: The termination error
            context: Additional context information
        """
        ...


class CircuitBreaker:
    """
    Circuit breaker implementation for termination fault tolerance.
    
    Implements the Circuit Breaker pattern to provide fault tolerance
    and graceful degradation when termination strategies fail repeatedly.
    
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
                        failure_count=self._failure_count,
                        terminator_id=kwargs.get("terminator_id", "unknown")
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


class GenerationTerminator:
    """
    Main orchestrator for text generation termination decisions.
    
    Implements the Facade pattern to provide a unified interface for
    complex termination logic involving multiple strategies, EOS token
    detection, and comprehensive monitoring capabilities.
    
    Features:
        - Multiple termination strategies with priority ordering
        - Comprehensive EOS token detection with fallback mechanisms
        - Performance monitoring and metrics collection
        - Thread-safe operations with concurrent access support
        - Circuit breaker pattern for fault tolerance
        - Event-driven monitoring and notifications
        - Advanced caching and memoization for performance
        
    Architecture:
        - Strategy Pattern: Pluggable termination strategies
        - Observer Pattern: Event-driven monitoring
        - State Pattern: Termination state management
        - Circuit Breaker Pattern: Fault tolerance
        - Template Method Pattern: Common evaluation workflow
    """
    
    def __init__(
        self,
        config: 'TerminationConfig',
        strategies: Optional[List['TerminationStrategy']] = None,
        observers: Optional[List[TerminationObserver]] = None,
        terminator_id: Optional[str] = None,
        enable_circuit_breaker: bool = True,
        enable_caching: bool = True,
        **kwargs
    ):
        # Core configuration
        self.config = config
        self.terminator_id = terminator_id or f"terminator_{uuid.uuid4().hex[:8]}"
        self.created_at = time.time()
        
        # State management
        self.state = TerminationState.INITIALIZING
        self._state_lock = threading.RLock()
        
        # Strategy management
        self.strategies = strategies or []
        self._strategy_lock = threading.RLock()
        
        # Observer pattern implementation
        self.observers: List[TerminationObserver] = observers or []
        self._observer_lock = threading.RLock()
        
        # Circuit breaker for fault tolerance
        if enable_circuit_breaker:
            self.circuit_breaker = CircuitBreaker(
                failure_threshold=config.circuit_breaker_threshold,
                recovery_timeout=config.circuit_breaker_timeout,
                success_threshold=config.circuit_breaker_success_threshold
            )
        else:
            self.circuit_breaker = None
        
        # Performance optimization
        self.enable_caching = enable_caching
        if enable_caching:
            self._result_cache: Dict[str, Tuple[TerminationResult, float]] = {}
            self._cache_lock = threading.RLock()
        
        # Performance metrics
        self._evaluation_count = 0
        self._total_evaluation_time = 0.0
        self._cache_hits = 0
        self._cache_misses = 0
        self._error_count = 0
        
        # Background maintenance
        self._maintenance_executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix=f"terminator-{self.terminator_id}"
        )
        self._shutdown_event = threading.Event()
        
        # Initialize the terminator
        self._initialize()
        self.state = TerminationState.READY
        
        logger.info(
            f"Initialized GenerationTerminator",
            extra={
                "terminator_id": self.terminator_id,
                "num_strategies": len(self.strategies),
                "num_observers": len(self.observers),
                "enable_circuit_breaker": enable_circuit_breaker,
                "enable_caching": enable_caching
            }
        )
    
    def _initialize(self) -> None:
        """Initialize terminator components."""
        # Import strategies here to avoid circular imports
        from .strategies import TerminationStrategyFactory
        
        # Create default strategies if none provided
        if not self.strategies:
            factory = TerminationStrategyFactory()
            self.strategies = factory.create_default_strategies(self.config)
        
        # Validate configuration
        self._validate_configuration()
        
        # Start background maintenance if needed
        if self.enable_caching:
            self._start_cache_maintenance()
    
    def _validate_configuration(self) -> None:
        """Validate terminator configuration."""
        if not self.config:
            raise InvalidTerminationConfigError("Configuration is required")
        
        if not self.strategies:
            raise InvalidTerminationConfigError("At least one strategy is required")
        
        # Validate strategy compatibility
        for strategy in self.strategies:
            if not hasattr(strategy, 'evaluate'):
                raise InvalidTerminationConfigError(
                    f"Strategy {type(strategy).__name__} missing evaluate method"
                )
    
    def should_terminate(
        self,
        tokens: TokenArray,
        logits: Optional[LogitsArray] = None,
        tokenizer: Optional[TokenizerWrapper] = None,
        step: int = 0,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> TerminationResult:
        """
        Evaluate whether generation should be terminated.
        
        This method implements the main termination evaluation workflow,
        coordinating multiple strategies and providing comprehensive
        monitoring and error handling.
        
        Args:
            tokens: Generated tokens array
            logits: Current logits (optional)
            tokenizer: Tokenizer for text processing
            step: Current generation step
            context: Additional context information
            **kwargs: Additional parameters
            
        Returns:
            Comprehensive termination result
            
        Raises:
            TerminationError: If evaluation fails
            TerminationTimeoutError: If evaluation times out
        """
        start_time = time.time()
        correlation_id = kwargs.get('correlation_id', str(uuid.uuid4()))
        
        try:
            with self._state_lock:
                # State validation
                if self.state not in [TerminationState.READY, TerminationState.EVALUATING]:
                    raise TerminationError(
                        f"Terminator in invalid state: {self.state}",
                        terminator_id=self.terminator_id,
                        correlation_id=correlation_id
                    )
                
                self.state = TerminationState.EVALUATING
                self._evaluation_count += 1
                
                # Check cache first
                if self.enable_caching:
                    cached_result = self._check_cache(tokens, step, context)
                    if cached_result:
                        self._cache_hits += 1
                        self.state = TerminationState.READY
                        return cached_result
                    self._cache_misses += 1
                
                # Perform evaluation through circuit breaker
                if self.circuit_breaker:
                    result = self.circuit_breaker.call(
                        self._evaluate_internal,
                        tokens, logits, tokenizer, step, context,
                        terminator_id=self.terminator_id,
                        correlation_id=correlation_id,
                        **kwargs
                    )
                else:
                    result = self._evaluate_internal(
                        tokens, logits, tokenizer, step, context, **kwargs
                    )
                
                # Cache the result
                if self.enable_caching:
                    self._cache_result(tokens, step, context, result)
                
                # Update metrics
                evaluation_time = (time.time() - start_time) * 1000
                self._total_evaluation_time += evaluation_time
                
                # Add performance metrics to result
                result = self._add_performance_metrics(result, evaluation_time)
                
                # Notify observers
                self._notify_observers_decision(result, context or {})
                
                self.state = TerminationState.READY
                
                logger.debug(
                    f"Termination evaluation completed",
                    extra={
                        "terminator_id": self.terminator_id,
                        "correlation_id": correlation_id,
                        "should_stop": result.should_stop,
                        "reason": result.reason.value,
                        "confidence": result.confidence,
                        "evaluation_time_ms": evaluation_time,
                        "step": step
                    }
                )
                
                return result
                
        except Exception as e:
            # Update error metrics
            evaluation_time = (time.time() - start_time) * 1000
            self._total_evaluation_time += evaluation_time
            self._error_count += 1
            
            self.state = TerminationState.ERROR
            
            # Notify observers of error
            if isinstance(e, TerminationError):
                self._notify_observers_error(e, context or {})
            
            logger.error(
                f"Termination evaluation failed: {e}",
                extra={
                    "terminator_id": self.terminator_id,
                    "correlation_id": correlation_id,
                    "evaluation_time_ms": evaluation_time,
                    "step": step,
                    "error_type": type(e).__name__
                }
            )
            
            # Re-raise as TerminationError if not already
            if not isinstance(e, TerminationError):
                raise TerminationError(
                    f"Evaluation failed: {e}",
                    terminator_id=self.terminator_id,
                    correlation_id=correlation_id,
                    context={"original_error": str(e), "step": step}
                )
            raise
        
        finally:
            # Ensure state is reset
            if self.state == TerminationState.EVALUATING:
                self.state = TerminationState.READY
    
    def _evaluate_internal(
        self,
        tokens: TokenArray,
        logits: Optional[LogitsArray],
        tokenizer: Optional[TokenizerWrapper],
        step: int,
        context: Optional[Dict[str, Any]],
        **kwargs
    ) -> TerminationResult:
        """Internal evaluation logic with strategy coordination."""
        strategy_results = {}
        warnings = []
        max_confidence = 0.0
        primary_reason = TerminationReason.UNKNOWN
        should_stop = False
        fallback_used = False
        
        # Evaluate each strategy
        with self._strategy_lock:
            for i, strategy in enumerate(self.strategies):
                try:
                    strategy_result = strategy.evaluate(
                        tokens=tokens,
                        logits=logits,
                        tokenizer=tokenizer,
                        step=step,
                        context=context,
                        **kwargs
                    )
                    
                    strategy_name = type(strategy).__name__
                    strategy_results[strategy_name] = strategy_result
                    
                    # Update decision based on strategy result
                    if strategy_result.should_stop:
                        should_stop = True
                        if strategy_result.confidence > max_confidence:
                            max_confidence = strategy_result.confidence
                            primary_reason = strategy_result.reason
                    
                    # Collect warnings
                    if hasattr(strategy_result, 'warnings'):
                        warnings.extend(strategy_result.warnings)
                    
                except Exception as e:
                    strategy_name = type(strategy).__name__
                    warning_msg = f"Strategy {strategy_name} failed: {e}"
                    warnings.append(warning_msg)
                    
                    logger.warning(
                        warning_msg,
                        extra={
                            "terminator_id": self.terminator_id,
                            "strategy_name": strategy_name,
                            "step": step
                        }
                    )
                    
                    # Use fallback logic for failed strategies
                    if self._should_use_fallback(strategy, e):
                        fallback_result = self._apply_fallback_logic(
                            tokens, step, strategy, e
                        )
                        if fallback_result:
                            strategy_results[f"{strategy_name}_fallback"] = fallback_result
                            fallback_used = True
        
        # Apply termination logic
        final_confidence = max_confidence if should_stop else 0.0
        
        # Create comprehensive result
        result = TerminationResult(
            should_stop=should_stop,
            reason=primary_reason,
            confidence=final_confidence,
            strategy_results=strategy_results,
            fallback_used=fallback_used,
            warnings=warnings,
            metadata={
                "step": step,
                "num_strategies": len(self.strategies),
                "num_tokens": tokens.size if hasattr(tokens, 'size') else len(tokens),
                "terminator_id": self.terminator_id
            }
        )
        
        return result
    
    def _should_use_fallback(self, strategy: 'TerminationStrategy', error: Exception) -> bool:
        """Determine if fallback logic should be used for a failed strategy."""
        # Use fallback for non-critical errors
        return not isinstance(error, (MemoryError, KeyboardInterrupt))
    
    def _apply_fallback_logic(
        self,
        tokens: TokenArray,
        step: int,
        failed_strategy: 'TerminationStrategy',
        error: Exception
    ) -> Optional[TerminationResult]:
        """Apply fallback termination logic when strategies fail."""
        try:
            # Simple fallback: check basic length limits
            max_tokens = getattr(self.config, 'max_tokens', 512)
            if step >= max_tokens:
                return TerminationResult(
                    should_stop=True,
                    reason=TerminationReason.MAX_LENGTH,
                    confidence=0.8,
                    metadata={"fallback": True, "failed_strategy": type(failed_strategy).__name__}
                )
            
            # Check for obvious EOS patterns in tokens
            if hasattr(tokens, 'tolist'):
                token_list = tokens.tolist()
                if isinstance(token_list, list) and len(token_list) > 0:
                    # Look for common EOS token IDs (this is a simple heuristic)
                    common_eos_ids = [0, 1, 2, 50256, 50257]  # Common EOS token IDs
                    if token_list[-1] in common_eos_ids:
                        return TerminationResult(
                            should_stop=True,
                            reason=TerminationReason.EOS_TOKEN,
                            confidence=0.6,
                            metadata={"fallback": True, "eos_token_id": token_list[-1]}
                        )
            
            return None
            
        except Exception as fallback_error:
            logger.error(f"Fallback logic failed: {fallback_error}")
            return None
    
    def _check_cache(
        self,
        tokens: TokenArray,
        step: int,
        context: Optional[Dict[str, Any]]
    ) -> Optional[TerminationResult]:
        """Check cache for previous evaluation results."""
        if not self.enable_caching:
            return None
        
        cache_key = self._compute_cache_key(tokens, step, context)
        
        with self._cache_lock:
            if cache_key in self._result_cache:
                result, timestamp = self._result_cache[cache_key]
                # Cache expires after 60 seconds
                if time.time() - timestamp < 60:
                    return result
                else:
                    del self._result_cache[cache_key]
        
        return None
    
    def _cache_result(
        self,
        tokens: TokenArray,
        step: int,
        context: Optional[Dict[str, Any]],
        result: TerminationResult
    ) -> None:
        """Cache evaluation result."""
        if not self.enable_caching:
            return
        
        cache_key = self._compute_cache_key(tokens, step, context)
        
        with self._cache_lock:
            # Limit cache size
            if len(self._result_cache) > 1000:
                # Remove oldest entries
                oldest_keys = sorted(
                    self._result_cache.keys(),
                    key=lambda k: self._result_cache[k][1]
                )[:100]
                for key in oldest_keys:
                    del self._result_cache[key]
            
            self._result_cache[cache_key] = (result, time.time())
    
    def _compute_cache_key(
        self,
        tokens: TokenArray,
        step: int,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Compute cache key for evaluation parameters."""
        try:
            # Create deterministic hash of inputs
            key_data = {
                "step": step,
                "num_tokens": tokens.size if hasattr(tokens, 'size') else len(tokens),
                "context": sorted(context.items()) if context else []
            }
            
            # Add token hash (sample for performance)
            if hasattr(tokens, 'tolist'):
                token_list = tokens.tolist()
                if isinstance(token_list, list) and len(token_list) > 0:
                    # Use last few tokens for cache key
                    key_tokens = token_list[-min(10, len(token_list)):]
                    key_data["tokens"] = key_tokens
            
            key_str = json.dumps(key_data, default=str, sort_keys=True)
            return hashlib.md5(key_str.encode()).hexdigest()
            
        except Exception:
            # Fallback to timestamp-based key
            return hashlib.md5(str(time.time()).encode()).hexdigest()
    
    def _add_performance_metrics(
        self,
        result: TerminationResult,
        evaluation_time_ms: float
    ) -> TerminationResult:
        """Add performance metrics to result."""
        performance_metrics = {
            'execution_time_ms': evaluation_time_ms,
            'cache_hit_rate': (
                self._cache_hits / (self._cache_hits + self._cache_misses)
                if (self._cache_hits + self._cache_misses) > 0 else 0.0
            ),
            'total_evaluations': self._evaluation_count,
            'error_rate': (
                self._error_count / self._evaluation_count
                if self._evaluation_count > 0 else 0.0
            )
        }
        
        # Create new result with updated metrics
        return TerminationResult(
            should_stop=result.should_stop,
            reason=result.reason,
            confidence=result.confidence,
            metadata=result.metadata,
            timestamp=result.timestamp,
            correlation_id=result.correlation_id,
            strategy_results=result.strategy_results,
            performance_metrics=performance_metrics,
            fallback_used=result.fallback_used,
            warnings=result.warnings
        )
    
    def add_observer(self, observer: TerminationObserver) -> None:
        """Add a termination observer."""
        with self._observer_lock:
            if observer not in self.observers:
                self.observers.append(observer)
                logger.debug(f"Added observer {type(observer).__name__}")
    
    def remove_observer(self, observer: TerminationObserver) -> None:
        """Remove a termination observer."""
        with self._observer_lock:
            if observer in self.observers:
                self.observers.remove(observer)
                logger.debug(f"Removed observer {type(observer).__name__}")
    
    def _notify_observers_decision(
        self,
        result: TerminationResult,
        context: Dict[str, Any]
    ) -> None:
        """Notify observers of termination decision."""
        with self._observer_lock:
            for observer in self.observers:
                try:
                    observer.on_termination_decision(result, context)
                except Exception as e:
                    logger.warning(f"Observer notification failed: {e}")
    
    def _notify_observers_error(
        self,
        error: TerminationError,
        context: Dict[str, Any]
    ) -> None:
        """Notify observers of termination error."""
        with self._observer_lock:
            for observer in self.observers:
                try:
                    observer.on_error(error, context)
                except Exception as e:
                    logger.warning(f"Observer error notification failed: {e}")
    
    def _start_cache_maintenance(self) -> None:
        """Start background cache maintenance."""
        def cleanup_cache():
            while not self._shutdown_event.wait(300):  # 5 minutes
                try:
                    with self._cache_lock:
                        current_time = time.time()
                        expired_keys = [
                            key for key, (_, timestamp) in self._result_cache.items()
                            if current_time - timestamp > 300  # 5 minutes TTL
                        ]
                        
                        for key in expired_keys:
                            del self._result_cache[key]
                        
                        if expired_keys:
                            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                            
                except Exception as e:
                    logger.error(f"Cache cleanup error: {e}")
        
        self._maintenance_executor.submit(cleanup_cache)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive terminator metrics."""
        avg_evaluation_time = (
            self._total_evaluation_time / self._evaluation_count
            if self._evaluation_count > 0 else 0.0
        )
        
        cache_hit_rate = (
            self._cache_hits / (self._cache_hits + self._cache_misses)
            if (self._cache_hits + self._cache_misses) > 0 else 0.0
        )
        
        return {
            "terminator_id": self.terminator_id,
            "state": self.state.name,
            "total_evaluations": self._evaluation_count,
            "total_evaluation_time_ms": self._total_evaluation_time,
            "average_evaluation_time_ms": avg_evaluation_time,
            "error_count": self._error_count,
            "error_rate": (
                self._error_count / self._evaluation_count
                if self._evaluation_count > 0 else 0.0
            ),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self._result_cache) if self.enable_caching else 0,
            "num_strategies": len(self.strategies),
            "num_observers": len(self.observers),
            "circuit_breaker_state": (
                self.circuit_breaker.state if self.circuit_breaker else "DISABLED"
            ),
            "uptime_seconds": time.time() - self.created_at
        }
    
    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self._evaluation_count = 0
        self._total_evaluation_time = 0.0
        self._cache_hits = 0
        self._cache_misses = 0
        self._error_count = 0
        
        if self.enable_caching:
            with self._cache_lock:
                self._result_cache.clear()
        
        logger.info(f"Reset metrics for terminator {self.terminator_id}")
    
    def shutdown(self) -> None:
        """Shutdown the terminator and cleanup resources."""
        logger.info(f"Shutting down terminator {self.terminator_id}")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Shutdown executor
        self._maintenance_executor.shutdown(wait=True)
        
        # Clear cache
        if self.enable_caching:
            with self._cache_lock:
                self._result_cache.clear()
        
        # Clear observers
        with self._observer_lock:
            self.observers.clear()
        
        self.state = TerminationState.DISABLED
        
        logger.info(f"Terminator {self.terminator_id} shutdown complete")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.shutdown()


# Export public API
__all__ = [
    "GenerationTerminator",
    "TerminationResult",
    "TerminationReason",
    "TerminationState",
    "TerminationConfidence",
    "TerminationError",
    "TerminationTimeoutError",
    "InvalidTerminationConfigError",
    "CircuitBreakerError",
    "TerminationObserver",
    "CircuitBreaker"
]