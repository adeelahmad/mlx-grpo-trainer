"""
Custom Exception Hierarchy for Progress Tracking System

This module defines a comprehensive exception hierarchy for the enhanced
progress bar system, providing specific error types for different failure
scenarios with detailed error information and recovery suggestions.

The exception hierarchy follows enterprise-grade error handling patterns:
- Structured error information with context
- Error codes for programmatic handling
- Recovery suggestions for operational issues
- Correlation IDs for distributed tracing
- Severity levels for monitoring systems

Exception Hierarchy:
    ProgressTrackingError (base)
    ├── MetricComputationError
    │   ├── GradientNormComputationError
    │   ├── MemoryTrackingError
    │   └── PerformanceMetricError
    ├── RendererError
    │   ├── DisplayFormatError
    │   └── TerminalCompatibilityError
    ├── ConfigurationError
    │   ├── InvalidConfigurationError
    │   └── MissingConfigurationError
    └── ThreadSafetyError
        ├── ConcurrentAccessError
        └── DeadlockError

Example:
    ```python
    try:
        gradient_norm = compute_gradient_norm(gradients)
    except GradientNormComputationError as e:
        logger.error(f"Gradient computation failed: {e}")
        # Use fallback value
        gradient_norm = 0.0
    ```
"""

import time
import uuid
from typing import Dict, Any, Optional, List, Union
from enum import Enum
import traceback
import threading


class ErrorSeverity(Enum):
    """Error severity levels for monitoring and alerting."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification and handling."""
    COMPUTATION = "computation"
    RENDERING = "rendering"
    CONFIGURATION = "configuration"
    THREAD_SAFETY = "thread_safety"
    PERFORMANCE = "performance"
    MEMORY = "memory"
    NETWORK = "network"
    VALIDATION = "validation"


class ProgressTrackingError(Exception):
    """
    Base exception for all progress tracking related errors.
    
    This exception provides comprehensive error context including:
    - Unique error ID for tracking
    - Timestamp for temporal analysis
    - Severity level for prioritization
    - Category for classification
    - Context information for debugging
    - Recovery suggestions for operations
    - Correlation ID for distributed tracing
    
    Attributes:
        error_id: Unique identifier for this error instance
        timestamp: When the error occurred
        severity: Error severity level
        category: Error category for classification
        context: Additional context information
        recovery_suggestions: List of suggested recovery actions
        correlation_id: ID for tracing across system boundaries
        thread_id: ID of the thread where error occurred
        stack_trace: Full stack trace for debugging
    """
    
    def __init__(
        self,
        message: str,
        *,
        error_code: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.COMPUTATION,
        context: Optional[Dict[str, Any]] = None,
        recovery_suggestions: Optional[List[str]] = None,
        correlation_id: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        """
        Initialize progress tracking error.
        
        Args:
            message: Human-readable error message
            error_code: Machine-readable error code
            severity: Error severity level
            category: Error category
            context: Additional context information
            recovery_suggestions: List of recovery suggestions
            correlation_id: Correlation ID for tracing
            cause: Original exception that caused this error
        """
        super().__init__(message)
        
        # Core error information
        self.error_id = str(uuid.uuid4())
        self.timestamp = time.time()
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.severity = severity
        self.category = category
        
        # Context and debugging information
        self.context = context or {}
        self.recovery_suggestions = recovery_suggestions or []
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.thread_id = threading.get_ident()
        self.stack_trace = traceback.format_exc()
        self.cause = cause
        
        # Add system context
        self._add_system_context()
    
    def _add_system_context(self) -> None:
        """Add system-level context information."""
        import psutil
        import os
        
        try:
            process = psutil.Process()
            self.context.update({
                'system': {
                    'pid': os.getpid(),
                    'memory_mb': process.memory_info().rss / 1024 / 1024,
                    'cpu_percent': process.cpu_percent(),
                    'thread_count': process.num_threads(),
                    'open_files': len(process.open_files()),
                }
            })
        except Exception:
            # Don't fail if system info unavailable
            pass
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert error to dictionary for serialization.
        
        Returns:
            Dictionary representation of the error
        """
        return {
            'error_id': self.error_id,
            'timestamp': self.timestamp,
            'message': self.message,
            'error_code': self.error_code,
            'severity': self.severity.value,
            'category': self.category.value,
            'context': self.context,
            'recovery_suggestions': self.recovery_suggestions,
            'correlation_id': self.correlation_id,
            'thread_id': self.thread_id,
            'stack_trace': self.stack_trace,
            'cause': str(self.cause) if self.cause else None,
        }
    
    def __str__(self) -> str:
        """Return formatted error message."""
        return (
            f"[{self.error_code}] {self.message} "
            f"(ID: {self.error_id[:8]}, Severity: {self.severity.value})"
        )
    
    def __repr__(self) -> str:
        """Return detailed error representation."""
        return (
            f"{self.__class__.__name__}("
            f"message='{self.message}', "
            f"error_code='{self.error_code}', "
            f"severity={self.severity}, "
            f"category={self.category})"
        )


class MetricComputationError(ProgressTrackingError):
    """
    Exception raised when metric computation fails.
    
    This exception is raised when any metric computation operation fails,
    including gradient norms, memory usage, performance metrics, etc.
    """
    
    def __init__(
        self,
        message: str,
        metric_name: str,
        computation_type: str,
        **kwargs
    ):
        """
        Initialize metric computation error.
        
        Args:
            message: Error message
            metric_name: Name of the metric that failed
            computation_type: Type of computation that failed
            **kwargs: Additional arguments for base class
        """
        super().__init__(
            message,
            category=ErrorCategory.COMPUTATION,
            **kwargs
        )
        self.metric_name = metric_name
        self.computation_type = computation_type
        
        # Add metric-specific context
        self.context.update({
            'metric_name': metric_name,
            'computation_type': computation_type,
        })
        
        # Add recovery suggestions
        self.recovery_suggestions.extend([
            f"Check input data for metric '{metric_name}'",
            f"Verify {computation_type} computation parameters",
            "Use fallback metric value if available",
            "Skip this metric and continue with others",
        ])


class GradientNormComputationError(MetricComputationError):
    """
    Exception raised when gradient norm computation fails.
    
    This is a specialized error for gradient norm computation failures,
    which are critical for training monitoring.
    """
    
    def __init__(
        self,
        message: str,
        gradient_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize gradient norm computation error.
        
        Args:
            message: Error message
            gradient_info: Information about the gradients
            **kwargs: Additional arguments for base class
        """
        super().__init__(
            message,
            metric_name="gradient_norm",
            computation_type="gradient_norm_calculation",
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        
        # Add gradient-specific context
        if gradient_info:
            self.context.update({'gradient_info': gradient_info})
        
        # Add gradient-specific recovery suggestions
        self.recovery_suggestions.extend([
            "Check gradient dictionary structure",
            "Verify MLX array validity",
            "Use L2 norm fallback computation",
            "Skip gradient norm display for this step",
        ])


class MemoryTrackingError(MetricComputationError):
    """
    Exception raised when memory tracking fails.
    
    This error occurs when memory usage computation or tracking fails,
    which can happen due to system limitations or MLX issues.
    """
    
    def __init__(
        self,
        message: str,
        memory_type: str = "unknown",
        **kwargs
    ):
        """
        Initialize memory tracking error.
        
        Args:
            message: Error message
            memory_type: Type of memory being tracked (MLX, system, etc.)
            **kwargs: Additional arguments for base class
        """
        super().__init__(
            message,
            metric_name="memory_usage",
            computation_type="memory_tracking",
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )
        
        self.memory_type = memory_type
        self.context.update({'memory_type': memory_type})
        
        # Add memory-specific recovery suggestions
        self.recovery_suggestions.extend([
            f"Check {memory_type} memory API availability",
            "Use alternative memory tracking method",
            "Disable memory tracking temporarily",
            "Check system permissions for memory access",
        ])


class PerformanceMetricError(MetricComputationError):
    """
    Exception raised when performance metric computation fails.
    
    This error occurs when timing, throughput, or other performance
    metrics cannot be computed correctly.
    """
    
    def __init__(
        self,
        message: str,
        performance_type: str,
        **kwargs
    ):
        """
        Initialize performance metric error.
        
        Args:
            message: Error message
            performance_type: Type of performance metric
            **kwargs: Additional arguments for base class
        """
        super().__init__(
            message,
            metric_name=f"performance_{performance_type}",
            computation_type="performance_measurement",
            **kwargs
        )
        
        self.performance_type = performance_type
        self.context.update({'performance_type': performance_type})


class RendererError(ProgressTrackingError):
    """
    Exception raised when progress bar rendering fails.
    
    This exception covers all rendering-related failures including
    display formatting, terminal compatibility, and output issues.
    """
    
    def __init__(
        self,
        message: str,
        renderer_type: str,
        **kwargs
    ):
        """
        Initialize renderer error.
        
        Args:
            message: Error message
            renderer_type: Type of renderer that failed
            **kwargs: Additional arguments for base class
        """
        super().__init__(
            message,
            category=ErrorCategory.RENDERING,
            **kwargs
        )
        
        self.renderer_type = renderer_type
        self.context.update({'renderer_type': renderer_type})
        
        # Add rendering-specific recovery suggestions
        self.recovery_suggestions.extend([
            f"Switch to alternative {renderer_type} renderer",
            "Use minimal rendering mode",
            "Check terminal compatibility",
            "Disable progress bar display",
        ])


class DisplayFormatError(RendererError):
    """
    Exception raised when display formatting fails.
    
    This error occurs when the progress bar format cannot be rendered
    correctly due to formatting issues or invalid parameters.
    """
    
    def __init__(
        self,
        message: str,
        format_string: str,
        **kwargs
    ):
        """
        Initialize display format error.
        
        Args:
            message: Error message
            format_string: The format string that failed
            **kwargs: Additional arguments for base class
        """
        super().__init__(
            message,
            renderer_type="format_renderer",
            **kwargs
        )
        
        self.format_string = format_string
        self.context.update({'format_string': format_string})


class TerminalCompatibilityError(RendererError):
    """
    Exception raised when terminal compatibility issues occur.
    
    This error occurs when the terminal doesn't support required
    features like colors, unicode, or specific escape sequences.
    """
    
    def __init__(
        self,
        message: str,
        terminal_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize terminal compatibility error.
        
        Args:
            message: Error message
            terminal_info: Information about the terminal
            **kwargs: Additional arguments for base class
        """
        super().__init__(
            message,
            renderer_type="terminal_renderer",
            **kwargs
        )
        
        if terminal_info:
            self.context.update({'terminal_info': terminal_info})


class ConfigurationError(ProgressTrackingError):
    """
    Exception raised when configuration issues occur.
    
    This exception covers all configuration-related failures including
    invalid settings, missing required parameters, and validation errors.
    """
    
    def __init__(
        self,
        message: str,
        config_section: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize configuration error.
        
        Args:
            message: Error message
            config_section: Configuration section that failed
            **kwargs: Additional arguments for base class
        """
        super().__init__(
            message,
            category=ErrorCategory.CONFIGURATION,
            **kwargs
        )
        
        self.config_section = config_section
        if config_section:
            self.context.update({'config_section': config_section})
        
        # Add configuration-specific recovery suggestions
        self.recovery_suggestions.extend([
            "Check configuration file syntax",
            "Verify all required parameters are present",
            "Use default configuration values",
            "Validate configuration against schema",
        ])


class InvalidConfigurationError(ConfigurationError):
    """
    Exception raised when configuration values are invalid.
    
    This error occurs when configuration parameters have invalid
    values, types, or ranges.
    """
    
    def __init__(
        self,
        message: str,
        parameter_name: str,
        parameter_value: Any,
        expected_type: Optional[type] = None,
        **kwargs
    ):
        """
        Initialize invalid configuration error.
        
        Args:
            message: Error message
            parameter_name: Name of the invalid parameter
            parameter_value: The invalid value
            expected_type: Expected type for the parameter
            **kwargs: Additional arguments for base class
        """
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        
        self.parameter_name = parameter_name
        self.parameter_value = parameter_value
        self.expected_type = expected_type
        
        self.context.update({
            'parameter_name': parameter_name,
            'parameter_value': str(parameter_value),
            'parameter_type': type(parameter_value).__name__,
            'expected_type': expected_type.__name__ if expected_type else None,
        })


class MissingConfigurationError(ConfigurationError):
    """
    Exception raised when required configuration is missing.
    
    This error occurs when required configuration parameters
    are not provided or cannot be found.
    """
    
    def __init__(
        self,
        message: str,
        missing_parameters: List[str],
        **kwargs
    ):
        """
        Initialize missing configuration error.
        
        Args:
            message: Error message
            missing_parameters: List of missing parameter names
            **kwargs: Additional arguments for base class
        """
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        
        self.missing_parameters = missing_parameters
        self.context.update({'missing_parameters': missing_parameters})


class ThreadSafetyError(ProgressTrackingError):
    """
    Exception raised when thread safety issues occur.
    
    This exception covers all thread safety related failures including
    concurrent access violations, deadlocks, and race conditions.
    """
    
    def __init__(
        self,
        message: str,
        operation: str,
        **kwargs
    ):
        """
        Initialize thread safety error.
        
        Args:
            message: Error message
            operation: Operation that caused the thread safety issue
            **kwargs: Additional arguments for base class
        """
        super().__init__(
            message,
            category=ErrorCategory.THREAD_SAFETY,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        
        self.operation = operation
        self.context.update({'operation': operation})
        
        # Add thread safety recovery suggestions
        self.recovery_suggestions.extend([
            "Retry operation with proper locking",
            "Use thread-safe alternatives",
            "Serialize access to shared resources",
            "Check for deadlock conditions",
        ])


class ConcurrentAccessError(ThreadSafetyError):
    """
    Exception raised when concurrent access violations occur.
    
    This error occurs when multiple threads attempt to access
    shared resources without proper synchronization.
    """
    
    def __init__(
        self,
        message: str,
        resource_name: str,
        accessing_threads: List[int],
        **kwargs
    ):
        """
        Initialize concurrent access error.
        
        Args:
            message: Error message
            resource_name: Name of the resource being accessed
            accessing_threads: List of thread IDs accessing the resource
            **kwargs: Additional arguments for base class
        """
        super().__init__(
            message,
            operation="concurrent_access",
            **kwargs
        )
        
        self.resource_name = resource_name
        self.accessing_threads = accessing_threads
        
        self.context.update({
            'resource_name': resource_name,
            'accessing_threads': accessing_threads,
            'thread_count': len(accessing_threads),
        })


class DeadlockError(ThreadSafetyError):
    """
    Exception raised when deadlock conditions are detected.
    
    This error occurs when threads are waiting for each other
    in a circular dependency, causing a deadlock.
    """
    
    def __init__(
        self,
        message: str,
        involved_threads: List[int],
        resources: List[str],
        **kwargs
    ):
        """
        Initialize deadlock error.
        
        Args:
            message: Error message
            involved_threads: List of thread IDs involved in deadlock
            resources: List of resources involved in deadlock
            **kwargs: Additional arguments for base class
        """
        super().__init__(
            message,
            operation="resource_locking",
            severity=ErrorSeverity.CRITICAL,
            **kwargs
        )
        
        self.involved_threads = involved_threads
        self.resources = resources
        
        self.context.update({
            'involved_threads': involved_threads,
            'resources': resources,
            'deadlock_size': len(involved_threads),
        })


# Utility functions for error handling

def handle_progress_error(
    error: ProgressTrackingError,
    logger,
    fallback_action: Optional[callable] = None
) -> Any:
    """
    Handle progress tracking errors with logging and optional fallback.
    
    Args:
        error: The progress tracking error to handle
        logger: Logger instance for error reporting
        fallback_action: Optional fallback action to execute
        
    Returns:
        Result of fallback action if provided, None otherwise
    """
    # Log error with appropriate level based on severity
    log_level = {
        ErrorSeverity.LOW: logger.debug,
        ErrorSeverity.MEDIUM: logger.warning,
        ErrorSeverity.HIGH: logger.error,
        ErrorSeverity.CRITICAL: logger.critical,
    }.get(error.severity, logger.error)
    
    log_level(
        f"Progress tracking error: {error.message} "
        f"(ID: {error.error_id}, Category: {error.category.value})"
    )
    
    # Log recovery suggestions
    if error.recovery_suggestions:
        logger.info(f"Recovery suggestions: {', '.join(error.recovery_suggestions)}")
    
    # Execute fallback action if provided
    if fallback_action:
        try:
            return fallback_action()
        except Exception as fallback_error:
            logger.error(f"Fallback action failed: {fallback_error}")
    
    return None


def create_error_context(
    operation: str,
    **additional_context
) -> Dict[str, Any]:
    """
    Create standardized error context dictionary.
    
    Args:
        operation: Name of the operation being performed
        **additional_context: Additional context information
        
    Returns:
        Standardized error context dictionary
    """
    context = {
        'operation': operation,
        'timestamp': time.time(),
        'thread_id': threading.get_ident(),
    }
    context.update(additional_context)
    return context


# Export all exception classes
__all__ = [
    'ErrorSeverity',
    'ErrorCategory',
    'ProgressTrackingError',
    'MetricComputationError',
    'GradientNormComputationError',
    'MemoryTrackingError',
    'PerformanceMetricError',
    'RendererError',
    'DisplayFormatError',
    'TerminalCompatibilityError',
    'ConfigurationError',
    'InvalidConfigurationError',
    'MissingConfigurationError',
    'ThreadSafetyError',
    'ConcurrentAccessError',
    'DeadlockError',
    'handle_progress_error',
    'create_error_context',
]