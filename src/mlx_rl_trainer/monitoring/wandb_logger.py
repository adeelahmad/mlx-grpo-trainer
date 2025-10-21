
"""
Advanced Weights & Biases Integration with Unified Metrics Interface

This module provides an enterprise-grade integration with Weights & Biases for experiment
tracking and visualization. It implements the unified metrics interface with comprehensive
features for experiment comparison, custom panels, intelligent sampling, and memory
optimizations.

Key Features:
- Implementation of the unified metrics interface
- Correlation ID support for tracking context across operations
- Custom WandB panels and visualizations
- Advanced experiment comparison capabilities
- Intelligent sampling for high-frequency metrics
- Memory usage optimizations
- Thread-safe operations for concurrent environments
- Circuit breaker pattern for API resilience
- Comprehensive error handling and logging
"""

import logging
import time
import threading
import json
import uuid
import gc
import os
import traceback
from typing import Dict, Any, List, Optional, Union, Tuple, Set, Callable, TypeVar, Generic, cast
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
from contextlib import contextmanager
import numpy as np
from datetime import datetime

try:
    import wandb
    from wandb.sdk.lib import RunDisabled
    from wandb.errors import Error as WandbError
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    RunDisabled = object  # type: ignore
    WandbError = Exception  # type: ignore

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

from mlx_rl_trainer.monitoring.metrics_interface import (
    MetricsInterface, MetricContext, MetricName, MetricValue, MetricDict,
    StepNumber, CorrelationId, MetricValidationLevel, MetricValidationResult,
    ValidationStrategy, BasicValidationStrategy, StandardValidationStrategy,
    StrictValidationStrategy, SamplingStrategy, FixedIntervalSamplingStrategy,
    AdaptiveSamplingStrategy, PrioritySamplingStrategy, MetricPriority
)

# Configure logger
logger = logging.getLogger(__name__)


class WandbCircuitBreakerState(Enum):
    """Circuit breaker states for WandB API calls."""
    CLOSED = 0      # Normal operation
    OPEN = 1        # Circuit is open, calls will fail fast
    HALF_OPEN = 2   # Testing if service is back


class WandbApiError(Exception):
    """Exception raised for WandB API errors."""
    pass


class WandbCircuitBreaker:
    """
    Circuit breaker for WandB API calls.
    
    This class implements the circuit breaker pattern to prevent cascading failures
    when the WandB API is experiencing issues. It automatically detects failures and
    temporarily disables API calls, periodically testing if the service is back.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        half_open_timeout: float = 30.0
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            reset_timeout: Time in seconds before attempting to close circuit
            half_open_timeout: Time in seconds to wait in half-open state
        """
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_timeout = half_open_timeout
        
        self.state = WandbCircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.last_state_change_time = time.time()
        
        self._lock = threading.RLock()
    
    def record_success(self) -> None:
        """Record a successful API call."""
        with self._lock:
            if self.state == WandbCircuitBreakerState.HALF_OPEN:
                logger.info("WandB API is back online, closing circuit breaker")
                self.state = WandbCircuitBreakerState.CLOSED
                self.failure_count = 0
                self.last_state_change_time = time.time()
    
    def record_failure(self) -> None:
        """Record a failed API call."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == WandbCircuitBreakerState.CLOSED and self.failure_count >= self.failure_threshold:
                logger.warning(
                    f"WandB API failure threshold reached ({self.failure_count} failures), opening circuit breaker"
                )
                self.state = WandbCircuitBreakerState.OPEN
                self.last_state_change_time = time.time()
    
    def can_execute(self) -> bool:
        """
        Check if an API call can be executed.
        
        Returns:
            True if the call can proceed, False otherwise
        """
        with self._lock:
            current_time = time.time()
            
            if self.state == WandbCircuitBreakerState.CLOSED:
                return True
            
            elif self.state == WandbCircuitBreakerState.OPEN:
                # Check if it's time to try again
                if current_time - self.last_state_change_time >= self.reset_timeout:
                    logger.info("Transitioning circuit breaker to half-open state")
                    self.state = WandbCircuitBreakerState.HALF_OPEN
                    self.last_state_change_time = current_time
                    return True
                return False
            
            elif self.state == WandbCircuitBreakerState.HALF_OPEN:
                # Only allow one test call
                if current_time - self.last_state_change_time >= self.half_open_timeout:
                    self.last_state_change_time = current_time
                    return True
                return False
            
            return False


class WandbPanelConfig:
    """
    Configuration for custom WandB panels.
    
    This class defines the configuration for custom panels in the WandB dashboard,
    including layout, visualization type, and data sources.
    """
    
    def __init__(
        self,
        panel_type: str,
        title: str,
        metrics: List[str],
        layout: Optional[Dict[str, Any]] = None,
        visualization_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize panel configuration.
        
        Args:
            panel_type: Type of panel (line, bar, scatter, etc.)
            title: Panel title
            metrics: List of metrics to include
            layout: Panel layout configuration
            visualization_config: Visualization-specific configuration
        """
        self.panel_type = panel_type
        self.title = title
        self.metrics = metrics
        self.layout = layout or {}
        self.visualization_config = visualization_config or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for WandB API."""
        return {
            "panel_type": self.panel_type,
            "title": self.title,
            "metrics": self.metrics,
            "layout": self.layout,
            "visualization_config": self.visualization_config
        }


class WandbCustomPanel:
    """
    Custom panel for WandB dashboards.
    
    This class represents a custom panel in the WandB dashboard, with methods
    for configuring and updating the panel.
    """
    
    def __init__(
        self,
        config: WandbPanelConfig,
        run: Any
    ):
        """
        Initialize custom panel.
        
        Args:
            config: Panel configuration
            run: WandB run object
        """
        self.config = config
        self.run = run
        self.panel_id = f"panel_{uuid.uuid4().hex[:8]}"
    
    def update(self, data: Dict[str, Any]) -> None:
        """
        Update panel with new data.
        
        Args:
            data: New data for the panel
        """
        if not WANDB_AVAILABLE:
            return
        
        try:
            # Create panel data
            panel_data = {
                "id": self.panel_id,
                "config": self.config.to_dict(),
                "data": data
            }
            
            # Log to WandB
            self.run.log({f"custom_panel/{self.panel_id}": panel_data})
        except Exception as e:
            logger.warning(f"Failed to update custom panel: {e}")


class WandbDashboard:
    """
    Custom dashboard for WandB.
    
    This class represents a custom dashboard in WandB, composed of multiple panels
    with different visualizations and data sources.
    """
    
    def __init__(
        self,
        name: str,
        run: Any,
        layout: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize dashboard.
        
        Args:
            name: Dashboard name
            run: WandB run object
            layout: Dashboard layout configuration
        """
        self.name = name
        self.run = run
        self.layout = layout or {"width": 2, "height": 2}
        self.panels: Dict[str, WandbCustomPanel] = {}
    
    def add_panel(self, panel: WandbCustomPanel) -> None:
        """
        Add panel to dashboard.
        
        Args:
            panel: Panel to add
        """
        self.panels[panel.panel_id] = panel
    
    def update(self, data: Dict[str, Any]) -> None:
        """
        Update all panels with new data.
        
        Args:
            data: New data for panels
        """
        for panel in self.panels.values():
            panel_data = {metric: data.get(metric, []) for metric in panel.config.metrics}
            panel.update(panel_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for WandB API."""
        return {
            "name": self.name,
            "layout": self.layout,
            "panels": [panel.panel_id for panel in self.panels.values()]
        }


@dataclass
class MetricBufferEntry:
    """
    Entry in the metric buffer.
    
    This class represents a single entry in the metric buffer, with the metric value,
    timestamp, step number, and context information.
    """
    value: MetricValue
    timestamp: float = field(default_factory=time.time)
    step: Optional[StepNumber] = None
    context: Optional[MetricContext] = None


class MetricBuffer:
    """
    Buffer for metrics before sending to WandB.
    
    This class implements a memory-efficient buffer for metrics, with support for
    batch processing and automatic flushing based on size or time thresholds.
    """
    
    def __init__(
        self,
        max_size: int = 100,
        flush_interval: float = 5.0
    ):
        """
        Initialize metric buffer.
        
        Args:
            max_size: Maximum buffer size before flushing
            flush_interval: Time interval in seconds for automatic flushing
        """
        self.max_size = max_size
        self.flush_interval = flush_interval
        
        self.buffer: Dict[MetricName, List[MetricBufferEntry]] = {}
        self.last_flush_time = time.time()
        
        self._lock = threading.RLock()
    
    def add(
        self,
        name: MetricName,
        value: MetricValue,
        step: Optional[StepNumber] = None,
        context: Optional[MetricContext] = None
    ) -> None:
        """
        Add metric to buffer.
        
        Args:
            name: Metric name
            value: Metric value
            step: Step number
            context: Metric context
        """
        with self._lock:
            if name not in self.buffer:
                self.buffer[name] = []
            
            self.buffer[name].append(MetricBufferEntry(
                value=value,
                step=step,
                context=context
            ))
    
    def should_flush(self) -> bool:
        """
        Check if buffer should be flushed.
        
        Returns:
            True if buffer should be flushed, False otherwise
        """
        with self._lock:
            # Check if any metric has reached max size
            for entries in self.buffer.values():
                if len(entries) >= self.max_size:
                    return True
            
            # Check if flush interval has elapsed
            current_time = time.time()
            if current_time - self.last_flush_time >= self.flush_interval:
                return True
            
            return False
    
    def get_metrics_for_flush(self) -> Dict[MetricName, List[MetricBufferEntry]]:
        """
        Get metrics for flushing.
        
        Returns:
            Dictionary of metrics to flush
        """
        with self._lock:
            # Create a copy of the buffer
            metrics_to_flush = self.buffer.copy()
            
            # Clear buffer
            self.buffer = {}
            
            # Update last flush time
            self.last_flush_time = time.time()
            
            return metrics_to_flush
    
    def clear(self) -> None:
        """Clear buffer."""
        with self._lock:
            self.buffer.clear()
            self.last_flush_time = time.time()


class EnhancedWandBLogger(MetricsInterface[None]):
    """
    Enhanced Weights & Biases logger with unified metrics interface.

    This class provides an enterprise-grade integration with Weights & Biases,
    implementing the unified metrics interface with comprehensive features for
    experiment comparison, custom panels, intelligent sampling, and memory
    optimizations.

    Features:
    - Implementation of the unified metrics interface
    - Correlation ID support for tracking context across operations
    - Custom WandB panels and visualizations
    - Advanced experiment comparison capabilities
    - Intelligent sampling for high-frequency metrics
    - Memory usage optimizations
    - Thread-safe operations for concurrent environments
    - Circuit breaker pattern for API resilience
    - Comprehensive error handling and logging
    """

    def __init__(
        self,
        project: str,
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        validation_strategy: Optional[ValidationStrategy] = None,
        sampling_strategy: Optional[SamplingStrategy] = None,
        enable_correlation_ids: bool = True,
        buffer_size: int = 100,
        flush_interval: float = 5.0,
        offline: bool = False,
        resume: bool = False,
        experiment_group: Optional[str] = None,
    ):
        """
        Initialize enhanced WandB logger.

        Args:
            project: WandB project name
            entity: WandB entity (username/team)
            name: Run name
            config: Configuration dictionary
            tags: Run tags
            notes: Run notes
            validation_strategy: Strategy for validating metrics
            sampling_strategy: Strategy for sampling high-frequency metrics
            enable_correlation_ids: Whether to enable correlation ID tracking
            buffer_size: Maximum buffer size before flushing
            flush_interval: Time interval in seconds for automatic flushing
            offline: Whether to run in offline mode
            resume: Whether to resume a previous run
            experiment_group: Group name for experiment comparison
        """
        if not WANDB_AVAILABLE:
            raise ImportError("wandb is required for EnhancedWandBLogger")

        # Store initialization parameters
        self.project = project
        self.entity = entity
        self.run_name = name
        self.enable_correlation_ids = enable_correlation_ids
        self.experiment_group = experiment_group
        
        # Add experiment group to tags if provided
        if experiment_group and tags:
            tags.append(f"group:{experiment_group}")
        elif experiment_group:
            tags = [f"group:{experiment_group}"]

        # Initialize validation and sampling strategies
        self.validation_strategy = validation_strategy or BasicValidationStrategy()
        self.sampling_strategy = sampling_strategy
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Metrics storage
        self._metrics: MetricDict = {}
        self._metrics_history: Dict[MetricName, List[Tuple[MetricValue, float, Optional[StepNumber]]]] = {}
        
        # Context tracking
        self._context_stack: List[MetricContext] = []
        self._current_context = MetricContext()
        
        # Metric buffer for batch logging
        self.metric_buffer = MetricBuffer(max_size=buffer_size, flush_interval=flush_interval)
        
        # Circuit breaker for API resilience
        self.circuit_breaker = WandbCircuitBreaker()
        
        # Custom panels and dashboards
        self.custom_panels: Dict[str, WandbCustomPanel] = {}
        self.dashboards: Dict[str, WandbDashboard] = {}
        
        # Statistics
        self.api_call_count = 0
        self.api_error_count = 0
        self.last_cleanup_time = time.time()
        
        # Initialize WandB with error handling
        try:
            # Add correlation ID to config if enabled
            if config is None:
                config = {}
            
            if self.enable_correlation_ids:
                config["correlation_id"] = self._current_context.correlation_id
            
            # Add experiment group to config if provided
            if experiment_group:
                config["experiment_group"] = experiment_group
            
            # Initialize WandB
            self.run = wandb.init(
                project=project,
                entity=entity,
                name=name,
                config=config,
                tags=tags,
                notes=notes,
                reinit=False,
                mode="offline" if offline else "online",
                resume=resume,
            )
            
            # Define custom charts
            self._define_custom_charts()
            
            logger.info(
                f"Initialized EnhancedWandBLogger: {self.run.url}",
                extra={"correlation_id": self._current_context.correlation_id}
            )
            
            # Record successful initialization
            self.circuit_breaker.record_success()
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            logger.error(
                f"Failed to initialize WandB: {e}",
                extra={
                    "correlation_id": self._current_context.correlation_id,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
            )
            
            # Create a disabled run for graceful degradation
            self.run = RunDisabled()
    
    def _define_custom_charts(self) -> None:
        """Define custom WandB charts."""
        if not self.circuit_breaker.can_execute():
            return
        
        try:
            # Define metric step
            wandb.define_metric("step")
            wandb.define_metric("*", step_metric="step")
            
            # Group metrics by category
            categories = [
                "loss",
                "reward",
                "gradient",
                "memory",
                "tokens",
                "learning",
                "generation",
                "thinking",
                "answer",
            ]
            
            for category in categories:
                wandb.define_metric(f"{category}/*", step_metric="step")
            
            # Record successful API call
            self.circuit_breaker.record_success()
            self.api_call_count += 1
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            self.api_error_count += 1
            logger.error(
                f"Failed to define custom charts: {e}",
                extra={
                    "correlation_id": self._current_context.correlation_id,
                    "error": str(e)
                }
            )
    
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
            
            logger.debug(
                f"Pushed context: {context.correlation_id}",
                extra={"correlation_id": context.correlation_id}
            )
            
            return context
    
    def pop_context(self) -> Optional[MetricContext]:
        """
        Pop the current context from the stack.
        
        Returns:
            The popped context, or None if stack is empty
        """
        with self._lock:
            if not self._context_stack:
                logger.warning(
                    "Attempted to pop context from empty stack",
                    extra={"correlation_id": self._current_context.correlation_id}
                )
                return None
            
            popped = self._context_stack.pop()
            self._current_context = self._context_stack[-1] if self._context_stack else MetricContext()
            
            logger.debug(
                f"Popped context: {popped.correlation_id}",
                extra={"correlation_id": self._current_context.correlation_id}
            )
            
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
        if self.validation_strategy:
            validation_result = self.validation_strategy.validate(name, value)
            if not validation_result.is_valid:
                for error in validation_result.errors:
                    logger.error(
                        f"Metric validation error: {error}",
                        extra={"correlation_id": context.correlation_id}
                    )
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
        
        # Add to buffer for batch logging
        self.metric_buffer.add(name, value, step, context)
        
        # Flush buffer if needed
        if self.metric_buffer.should_flush():
            self._flush_metrics()
    
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
    ) -> List[Tuple[MetricValue, float, Optional[StepNumber]]]:
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
    
    def _flush_metrics(self) -> None:
        """Flush metrics to WandB."""
        if not self.circuit_breaker.can_execute():
            logger.warning(
                "Circuit breaker open, skipping metrics flush",
                extra={"correlation_id": self._current_context.correlation_id}
            )
            return
        
        metrics_to_flush = self.metric_buffer.get_metrics_for_flush()
        if not metrics_to_flush:
            return
        
        try:
            # Group metrics by step
            metrics_by_step: Dict[Optional[StepNumber], Dict[str, Any]] = {}
            
            for name, entries in metrics_to_flush.items():
                for entry in entries:
                    step = entry.step
                    
                    if step not in metrics_by_step:
                        metrics_by_step[step] = {}
                    
                    # Convert value to appropriate type
                    value = entry.value
                    if isinstance(value, (int, float, np.number)):
                        metrics_by_step[step][name] = float(value)
                    elif isinstance(value, np.ndarray):
                        if value.size == 1:
                            metrics_by_step[step][name] = float(value.item())
            
            # Log metrics by step
            for step, metrics in metrics_by_step.items():
                if metrics:
                    if step is not None:
                        metrics["step"] = step
                    
                    wandb.log(metrics)
            
            # Record successful API call
            self.circuit_breaker.record_success()
            self.api_call_count += 1
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            self.api_error_count += 1
            logger.error(
                f"Failed to flush metrics: {e}",
                extra={
                    "correlation_id": self._current_context.correlation_id,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
            )
    
    def log_training_metrics(
        self,
        loss: float,
        reward: float,
        grad_norm: float,
        learning_rate: float,
        step: int,
        **kwargs,
    ) -> None:
        """
        Log core training metrics.
        
        Args:
            loss: Training loss
            reward: Reward value
            grad_norm: Gradient norm
            learning_rate: Learning rate
            step: Training step
            **kwargs: Additional metrics
        """
        metrics = {
            "loss/total": loss,
            "reward/mean": reward,
            "gradient/norm": grad_norm,
            "learning/rate": learning_rate,
            "step": step,
        }
        
        # Add additional metrics
        metrics.update(kwargs)
        
        # Log metrics
        self.log_metrics(metrics, step=step)
    
    def log_distribution(
        self, 
        name: str, 
        values: List[float], 
        step: Optional[int] = None
    ) -> None:
        """
        Log distribution of values.
        
        Args:
            name: Distribution name
            values: List of values
            step: Training step
        """
        if not values:
            return
        
        if not self.circuit_breaker.can_execute():
            logger.warning(
                "Circuit breaker open, skipping distribution logging",
                extra={"correlation_id": self._current_context.correlation_id}
            )
            return
        
        try:
            values_array = np.array(values)
            
            # Log histogram
            wandb.log({f"distribution/{name}": wandb.Histogram(values_array), "step": step})
            
            # Log statistics
            stats = {
                f"distribution/{name}/mean": float(np.mean(values_array)),
                f"distribution/{name}/std": float(np.std(values_array)),
                f"distribution/{name}/min": float(np.min(values_array)),
                f"distribution/{name}/max": float(np.max(values_array)),
                f"distribution/{name}/median": float(np.median(values_array)),
            }
            
            if step is not None:
                stats["step"] = step
            
            wandb.log(stats)
            
            # Record successful API call
            self.circuit_breaker.record_success()
            self.api_call_count += 1
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            self.api_error_count += 1
            logger.error(
                f"Failed to log distribution: {e}",
                extra={
                    "correlation_id": self._current_context.correlation_id,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
            )
    
    def log_sample_table(
        self,
        prompts: List[str],
        generations: List[str],
        references: List[str],
        rewards: List[float],
        step: int,
        max_samples: int = 10,
        context: Optional[MetricContext] = None,
    ) -> None:
        """
        Log samples as WandB table.
        
        Args:
            prompts: List of prompts
            generations: List of generated texts
            references: List of reference texts
            rewards: List of reward values
            step: Training step
            max_samples: Maximum number of samples to log
            context: Metric context
        """
        if not prompts:
            return
        
        if not self.circuit_breaker.can_execute():
            logger.warning(
                "Circuit breaker open, skipping sample table logging",
                extra={"correlation_id": (context or self._current_context).correlation_id}
            )
            return
        
        try:
            # Limit number of samples
            n = min(len(prompts), max_samples)
            
            # Create table
            columns = ["Step", "Prompt", "Generation", "Reference", "Reward"]
            data = []
            
            for i in range(n):
                data.append(
                    [
                        step,
                        prompts[i][:200],  # Truncate for display
                        generations[i][:500],
                        references[i][:500],
                        rewards[i],
                    ]
                )
            
            # Create and log table
            table = wandb.Table(columns=columns, data=data)
            wandb.log({f"samples/step_{step}": table})
            
            # Record successful API call
            self.circuit_breaker.record_success()
            self.api_call_count += 1
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            self.api_error_count += 1
            logger.error(
                f"Failed to log sample table: {e}",
                extra={
                    "correlation_id": (context or self._current_context).correlation_id,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
            )
    
    def log_gradient_flow(self, layer_gradients: Dict[str, float], step: int) -> None:
        """
        Log gradient flow across layers.
        
        Args:
            layer_gradients: Dictionary of layer_name -> gradient norm
            step: Training step
        """
        if not self.circuit_breaker.can_execute():
            logger.warning(
                "Circuit breaker open, skipping gradient flow logging",
                extra={"correlation_id": self._current_context.correlation_id}
            )
            return
        
        try:
            # Log each layer gradient
            log_dict = {f"gradient/layer_{layer_name}": grad_norm for layer_name, grad_norm in layer_gradients.items()}
            log_dict["step"] = step
            
            wandb.log(log_dict)
            
            # Record successful API call
            self.circuit_breaker.record_success()
            self.api_call_count += 1
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            self.api_error_count += 1
            logger.error(
                f"Failed to log gradient flow: {e}",
                extra={
                    "correlation_id": self._current_context.correlation_id,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
            )
    
    def log_memory_stats(
        self,
        allocated_mb: float,
        cached_mb: float,
        peak_mb: float,
        step: int
    ) -> None:
        """
        Log memory statistics.
        
        Args:
            allocated_mb: Allocated memory in MB
            cached_mb: Cached memory in MB
            peak_mb: Peak memory usage in MB
            step: Training step
        """
        if not self.circuit_breaker.can_execute():
            logger.warning(
                "Circuit breaker open, skipping memory stats logging",
                extra={"correlation_id": self._current_context.correlation_id}
            )
            return
        
        try:
            wandb.log(
                {
                    "memory/allocated_mb": allocated_mb,
                    "memory/cached_mb": cached_mb,
                    "memory/peak_mb": peak_mb,
                    "step": step,
                }
            )
            
            # Record successful API call
            self.circuit_breaker.record_success()
            self.api_call_count += 1
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            self.api_error_count += 1
            logger.error(
                f"Failed to log memory stats: {e}",
                extra={
                    "correlation_id": self._current_context.correlation_id,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
            )
    
    def log_token_stats(
        self,
        thinking_tokens: int,
        answer_tokens: int,
        total_tokens: int,
        step: int
    ) -> None:
        """
        Log token statistics.
        
        Args:
            thinking_tokens: Number of thinking tokens
            answer_tokens: Number of answer tokens
            total_tokens: Total number of tokens
            step: Training step
        """
        if not self.circuit_breaker.can_execute():
            logger.warning(
                "Circuit breaker open, skipping token stats logging",
                extra={"correlation_id": self._current_context.correlation_id}
            )
            return
        
        try:
            wandb.log(
                {
                    "tokens/thinking": thinking_tokens,
                    "tokens/answer": answer_tokens,
                    "tokens/total": total_tokens,
                    "tokens/ratio": thinking_tokens / max(answer_tokens, 1),
                    "step": step,
                }
            )
            
            # Record successful API call
            self.circuit_breaker.record_success()
            self.api_call_count += 1
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            self.api_error_count += 1
            logger.error(
                f"Failed to log token stats: {e}",
                extra={
                    "correlation_id": self._current_context.correlation_id,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
            )
    
    def log_reward_breakdown(
        self,
        reward_components: Dict[str, float],
        step: int
    ) -> None:
        """
        Log breakdown of reward components.
        
        Args:
            reward_components: Dictionary of component_name -> reward value
            step: Training step
        """
        if not self.circuit_breaker.can_execute():
            logger.warning(
                "Circuit breaker open, skipping reward breakdown logging",
                extra={"correlation_id": self._current_context.correlation_id}
            )
            return
        
        try:
            # Log each reward component
            log_dict = {f"reward/component/{component}": value for component, value in reward_components.items()}
            log_dict["step"] = step
            
            wandb.log(log_dict)
            
            # Record successful API call
            self.circuit_breaker.record_success()
            self.api_call_count += 1
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            self.api_error_count += 1
            logger.error(
                f"Failed to log reward breakdown: {e}",
                extra={
                    "correlation_id": self._current_context.correlation_id,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
            )
    
    def log_image(
        self,
        name: str,
        image_path: Path,
        caption: Optional[str] = None,
        step: Optional[int] = None,
    ) -> None:
        """
        Log image to WandB.
        
        Args:
            name: Image name
            image_path: Path to image file
            caption: Optional image caption
            step: Training step
        """
        if not self.circuit_breaker.can_execute():
            logger.warning(
                "Circuit breaker open, skipping image logging",
                extra={"correlation_id": self._current_context.correlation_id}
            )
            return
        
        try:
            # Create log dictionary
            log_dict = {name: wandb.Image(str(image_path), caption=caption)}
            
            if step is not None:
                log_dict["step"] = step
            
            # Log to WandB
            wandb.log(log_dict)
            
            # Record successful API call
            self.circuit_breaker.record_success()
            self.api_call_count += 1
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            self.api_error_count += 1
            logger.error(
                f"Failed to log image: {e}",
                extra={
                    "correlation_id": self._current_context.correlation_id,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
            )
    
    def log_chart(
        self,
        name: str,
        chart_path: Path,
        step: Optional[int] = None
    ) -> None:
        """
        Log chart image.
        
        Args:
            name: Chart name
            chart_path: Path to chart image
            step: Training step
        """
        self.log_image(f"charts/{name}", chart_path, step=step)
    
    def log_artifact(
        self,
        artifact_name: str,
        artifact_type: str,
        artifact_path: Path,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log artifact to WandB.
        
        Args:
            artifact_name: Artifact name
            artifact_type: Artifact type
            artifact_path: Path to artifact file
            metadata: Optional metadata
        """
        if not self.circuit_breaker.can_execute():
            logger.warning(
                "Circuit breaker open, skipping artifact logging",
                extra={"correlation_id": self._current_context.correlation_id}
            )
            return
        
        try:
            # Create artifact
            artifact = wandb.Artifact(
                name=artifact_name,
                type=artifact_type,
                metadata=metadata
            )
            
            # Add file to artifact
            artifact.add_file(str(artifact_path))
            
            # Log artifact
            self.run.log_artifact(artifact)
            
            # Record successful API call
            self.circuit_breaker.record_success()
            self.api_call_count += 1
            
            logger.info(
                f"Logged artifact: {artifact_name}",
                extra={"correlation_id": self._current_context.correlation_id}
            )
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            self.api_error_count += 1
            logger.error(
                f"Failed to log artifact: {e}",
                extra={
                    "correlation_id": self._current_context.correlation_id,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
            )
    
    def log_checkpoint(
        self,
        checkpoint_path: Path,
        step: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log model checkpoint as artifact.
        
        Args:
            checkpoint_path: Path to checkpoint file
            step: Training step
            metadata: Optional metadata
        """
        # Create artifact name
        artifact_name = f"checkpoint_step_{step}"
        
        # Create metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({"step": step, "timestamp": time.time()})
        
        # Log artifact
        self.log_artifact(
            artifact_name=artifact_name,
            artifact_type="model",
            artifact_path=checkpoint_path,
            metadata=metadata,
        )
    
    def log_config_update(self, config: Dict[str, Any]) -> None:
        """
        Update run configuration.
        
        Args:
            config: Configuration dictionary
        """
        if not self.circuit_breaker.can_execute():
            logger.warning(
                "Circuit breaker open, skipping config update",
                extra={"correlation_id": self._current_context.correlation_id}
            )
            return
        
        try:
            # Update config
            wandb.config.update(config)
            
            # Record successful API call
            self.circuit_breaker.record_success()
            self.api_call_count += 1
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            self.api_error_count += 1
            logger.error(
                f"Failed to update config: {e}",
                extra={
                    "correlation_id": self._current_context.correlation_id,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
            )
    
    def alert(
        self,
        title: str,
        text: str,
        level: str = "INFO",
        wait_duration: int = 300
    ) -> None:
        """
        Send WandB alert.
        
        Args:
            title: Alert title
            text: Alert text
            level: Alert level (INFO, WARN, ERROR)
            wait_duration: Wait duration in seconds before sending another alert
        """
        if not self.circuit_breaker.can_execute():
            logger.warning(
                "Circuit breaker open, skipping alert",
                extra={"correlation_id": self._current_context.correlation_id}
            )
            return
        
        try:
            # Send alert
            wandb.alert(
                title=title,
                text=text,
                level=getattr(wandb.AlertLevel, level),
                wait_duration=wait_duration,
            )
            
            # Record successful API call
            self.circuit_breaker.record_success()
            self.api_call_count += 1
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            self.api_error_count += 1
            logger.error(
                f"Failed to send alert: {e}",
                extra={
                    "correlation_id": self._current_context.correlation_id,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
            )
    
    def create_custom_panel(
        self,
        panel_type: str,
        title: str,
        metrics: List[str],
        layout: Optional[Dict[str, Any]] = None,
        visualization_config: Optional[Dict[str, Any]] = None
    ) -> WandbCustomPanel:
        """
        Create a custom panel for WandB dashboard.
        
        Args:
            panel_type: Type of panel (line, bar, scatter, etc.)
            title: Panel title
            metrics: List of metrics to include
            layout: Panel layout configuration
            visualization_config: Visualization-specific configuration
            
        Returns:
            Custom panel object
        """
        # Create panel configuration
        config = WandbPanelConfig(
            panel_type=panel_type,
            title=title,
            metrics=metrics,
            layout=layout,
            visualization_config=visualization_config
        )
        
        # Create panel
        panel = WandbCustomPanel(config, self.run)
        
        # Store panel
        self.custom_panels[panel.panel_id] = panel
        
        return panel
    
    def create_dashboard(
        self,
        name: str,
        layout: Optional[Dict[str, Any]] = None
    ) -> WandbDashboard:
        """
        Create a custom dashboard for WandB.
        
        Args:
            name: Dashboard name
            layout: Dashboard layout configuration
            
        Returns:
            Dashboard object
        """
        # Create dashboard
        dashboard = WandbDashboard(name, self.run, layout)
        
        # Store dashboard
        self.dashboards[name] = dashboard
        
        return dashboard
    
    def compare_experiments(
        self,
        experiment_ids: List[str],
        metrics: List[str],
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple experiments.
        
        Args:
            experiment_ids: List of experiment IDs to compare
            metrics: List of metrics to compare
            output_path: Path to save comparison results
            
        Returns:
            Dictionary of comparison results
        """
        if not self.circuit_breaker.can_execute():
            logger.warning(
                "Circuit breaker open, skipping experiment comparison",
                extra={"correlation_id": self._current_context.correlation_id}
            )
            return {}
        
        try:
            # Create API object
            api = wandb.Api()
            
            # Get runs
            runs = []
            for exp_id in experiment_ids:
                try:
                    run = api.run(f"{self.entity}/{self.project}/{exp_id}")
                    runs.append(run)
                except Exception as e:
                    logger.warning(
                        f"Failed to get run {exp_id}: {e}",
                        extra={"correlation_id": self._current_context.correlation_id}
                    )
            
            if not runs:
                logger.warning(
                    "No valid runs found for comparison",
                    extra={"correlation_id": self._current_context.correlation_id}
                )
                return {}
            
            # Extract metrics
            comparison_data = {}
            for run in runs:
                run_data = {"config": run.config}
                
                for metric in metrics:
                    try:
                        history = run.history(keys=[metric])
                        if not history.empty:
                            run_data[metric] = history[metric].tolist()
                        else:
                            run_data[metric] = []
                    except Exception as e:
                        logger.warning(
                            f"Failed to get metric {metric} for run {run.id}: {e}",
                            extra={"correlation_id": self._current_context.correlation_id}
                        )
                        run_data[metric] = []
                
                comparison_data[run.id] = run_data
            
            # Save comparison results if output path provided
            if output_path:
                with open(output_path, "w") as f:
                    json.dump(comparison_data, f, indent=2)
            
            # Record successful API call
            self.circuit_breaker.record_success()
            self.api_call_count += 1
            
            return comparison_data
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            self.api_error_count += 1
            logger.error(
                f"Failed to compare experiments: {e}",
                extra={
                    "correlation_id": self._current_context.correlation_id,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
            )
            return {}
    
    def get_experiment_group_metrics(
        self,
        group_name: Optional[str] = None,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get metrics for all runs in an experiment group.
        
        Args:
            group_name: Group name (uses current group if None)
            metrics: List of metrics to get (gets all if None)
            
        Returns:
            Dictionary of metrics by run
        """
        if not self.circuit_breaker.can_execute():
            logger.warning(
                "Circuit breaker open, skipping experiment group metrics",
                extra={"correlation_id": self._current_context.correlation_id}
            )
            return {}
        
        try:
            # Use current group if none provided
            if group_name is None:
                group_name = self.experiment_group
            
            if not group_name:
                logger.warning(
                    "No experiment group specified",
                    extra={"correlation_id": self._current_context.correlation_id}
                )
                return {}
            
            # Create API object
            api = wandb.Api()
            
            # Get runs in group
            runs = api.runs(
                f"{self.entity}/{self.project}",
                filters={"tags": {"$in": [f"group:{group_name}"]}}
            )
            
            # Extract metrics
            group_data = {}
            for run in runs:
                run_data = {"config": run.config}
                
                # Get all metrics if none specified
                if metrics is None:
                    summary = run.summary
                    for key, value in summary.items():
                        if isinstance(value, (int, float)):
                            run_data[key] = value
                else:
                    # Get specified metrics
                    for metric in metrics:
                        try:
                            if metric in run.summary:
                                run_data[metric] = run.summary[metric]
                            else:
                                history = run.history(keys=[metric])
                                if not history.empty:
                                    run_data[metric] = history[metric].tolist()
                                else:
                                    run_data[metric] = []
                        except Exception as e:
                            logger.warning(
                                f"Failed to get metric {metric} for run {run.id}: {e}",
                                extra={"correlation_id": self._current_context.correlation_id}
                            )
                            run_data[metric] = []
                
                group_data[run.id] = run_data
            
            # Record successful API call
            self.circuit_breaker.record_success()
            self.api_call_count += 1
            
            return group_data
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            self.api_error_count += 1
            logger.error(
                f"Failed to get experiment group metrics: {e}",
                extra={
                    "correlation_id": self._current_context.correlation_id,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
            )
            return {}
    
    def cleanup(self) -> None:
        """Perform memory cleanup."""
        gc.collect()
        
        if MLX_AVAILABLE:
            try:
                mx.clear_cache()
            except:
                pass
        
        logger.debug(
            "Performed memory cleanup",
            extra={"correlation_id": self._current_context.correlation_id}
        )
        
        self.last_cleanup_time = time.time()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get logger statistics.
        
        Returns:
            Dictionary of logger statistics
        """
        return {
            "api_call_count": self.api_call_count,
            "api_error_count": self.api_error_count,
            "circuit_breaker_state": self.circuit_breaker.state.name,
            "circuit_breaker_failure_count": self.circuit_breaker.failure_count,
            "last_cleanup_time": self.last_cleanup_time,
            "correlation_id": self._current_context.correlation_id,
            "context_stack_depth": len(self._context_stack),
            "custom_panels_count": len(self.custom_panels),
            "dashboards_count": len(self.dashboards)
        }
    
    def finish(self) -> None:
        """Finish WandB run."""
        if not self.circuit_breaker.can_execute():
            logger.warning(
                "Circuit breaker open, skipping run finish",
                extra={"correlation_id": self._current_context.correlation_id}
            )
            return
        
        try:
            # Flush any remaining metrics
            self._flush_metrics()
            
            # Finish run
            if self.run:
                self.run.finish()
                
                logger.info(
                    "Finished WandB run",
                    extra={"correlation_id": self._current_context.correlation_id}
                )
            
        except Exception as e:
            logger.error(
                f"Failed to finish WandB run: {e}",
                extra={
                    "correlation_id": self._current_context.correlation_id,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
            )