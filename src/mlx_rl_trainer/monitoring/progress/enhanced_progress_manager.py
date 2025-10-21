"""
Enhanced Progress Bar Manager for MLX RL Training

This module provides a comprehensive progress bar management system that coordinates
all progress tracking components using enterprise-grade design patterns. The manager
implements the Singleton pattern to ensure global consistency and provides a unified
interface for progress tracking across the entire training system.

Architecture:
- Singleton Pattern: Global progress manager instance
- Observer Pattern: Real-time progress updates and notifications
- Strategy Pattern: Pluggable progress rendering strategies
- Factory Pattern: Dynamic creation of progress components
- Command Pattern: Encapsulated progress update operations
- Template Method: Standardized progress tracking workflow

Key Features:
- Centralized progress coordination across all training components
- Real-time gradient norm tracking with advanced statistical analysis
- Comprehensive training metrics collection and aggregation
- Thread-safe operations for concurrent training environments
- Multiple display modes (compact, detailed, minimal, dashboard, debug)
- Intelligent caching and performance optimization
- Automatic error recovery and graceful degradation
- Extensible plugin architecture for custom progress indicators

Example:
    ```python
    from mlx_rl_trainer.monitoring.progress import EnhancedProgressManager
    
    # Get singleton instance
    progress_manager = EnhancedProgressManager.get_instance()
    
    # Configure progress tracking
    config = ProgressBarConfig(
        display_mode='detailed',
        show_gradient_norm=True,
        show_memory_usage=True,
        update_frequency=1.0
    )
    progress_manager.configure(config)
    
    # Start training session
    session = progress_manager.start_training_session(
        total_steps=1000,
        session_name="GRPO Training"
    )
    
    # Update progress during training
    for step in range(1000):
        metrics = {
            'loss': 0.5,
            'gradient_norm': 2.3,
            'learning_rate': 0.001,
            'memory_mb': 1024.5
        }
        session.update_progress(step, metrics)
    
    # End training session
    session.complete()
    ```
"""

import time
import threading
import logging
import weakref
from typing import Dict, Any, List, Optional, Union, Callable, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
import json
import uuid

import mlx.core as mx
import numpy as np

from .gradient_tracker import GradientNormTracker, GradientNormResult
from .metrics_collector import TrainingMetricsCollector, MetricDefinition, MetricType
from .progress_renderer import (
    ProgressBarRenderer, ProgressBarRendererFactory, 
    CompactRenderer, DetailedRenderer, MinimalRenderer
)
from .metric_formatters import MetricFormatterFactory
from .exceptions import (
    ProgressTrackingError,
    ConfigurationError,
    ThreadSafetyError,
    create_error_context,
)

logger = logging.getLogger(__name__)


class DisplayMode(Enum):
    """Available progress bar display modes."""
    COMPACT = "compact"
    DETAILED = "detailed"
    MINIMAL = "minimal"
    DASHBOARD = "dashboard"
    DEBUG = "debug"


class ProgressUpdateFrequency(Enum):
    """Progress update frequency settings."""
    REALTIME = 0.1
    FAST = 0.5
    NORMAL = 1.0
    SLOW = 2.0
    BATCH = 5.0


@dataclass
class ProgressBarConfig:
    """
    Configuration for progress bar display and behavior.
    
    This class encapsulates all configuration options for the enhanced
    progress bar system, providing fine-grained control over display
    modes, update frequencies, and feature enablement.
    
    Attributes:
        display_mode: Visual display mode for progress bars
        show_gradient_norm: Whether to display gradient norm information
        show_memory_usage: Whether to display memory usage metrics
        show_performance_stats: Whether to display performance statistics
        show_layer_analysis: Whether to display layer-wise gradient analysis
        update_frequency: How often to update progress display (seconds)
        enable_caching: Whether to enable performance caching
        enable_anomaly_detection: Whether to detect metric anomalies
        max_history_size: Maximum number of historical data points
        terminal_width: Terminal width for display formatting
        color_scheme: Color scheme for progress display
        custom_formatters: Custom metric formatters
        notification_thresholds: Thresholds for progress notifications
    """
    display_mode: DisplayMode = DisplayMode.DETAILED
    show_gradient_norm: bool = True
    show_memory_usage: bool = True
    show_performance_stats: bool = True
    show_layer_analysis: bool = False
    update_frequency: float = 1.0
    enable_caching: bool = True
    enable_anomaly_detection: bool = True
    max_history_size: int = 10000
    terminal_width: Optional[int] = None
    color_scheme: str = "default"
    custom_formatters: Dict[str, str] = field(default_factory=dict)
    notification_thresholds: Dict[str, float] = field(default_factory=dict)
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.update_frequency <= 0:
            raise ConfigurationError("update_frequency must be positive")
        
        if self.max_history_size <= 0:
            raise ConfigurationError("max_history_size must be positive")
        
        if self.terminal_width is not None and self.terminal_width < 40:
            raise ConfigurationError("terminal_width must be at least 40 characters")


@dataclass
class MetricDisplayConfig:
    """
    Configuration for individual metric display properties.
    
    Attributes:
        name: Metric name
        display_name: Human-readable display name
        formatter: Formatter type for the metric
        precision: Number of decimal places to display
        unit: Unit of measurement
        show_trend: Whether to show trend indicators
        show_statistics: Whether to show statistical information
        color_coding: Color coding rules for the metric
        alert_thresholds: Alert thresholds for the metric
    """
    name: str
    display_name: str
    formatter: str = "numeric"
    precision: int = 3
    unit: str = ""
    show_trend: bool = True
    show_statistics: bool = False
    color_coding: Dict[str, Any] = field(default_factory=dict)
    alert_thresholds: Dict[str, float] = field(default_factory=dict)


class TrainingSession:
    """
    Represents an active training session with progress tracking.
    
    This class manages the lifecycle of a training session, coordinating
    progress updates, metric collection, and display rendering. It provides
    a high-level interface for training loops to report progress and metrics.
    
    Features:
    - Session lifecycle management (start, update, pause, resume, complete)
    - Real-time progress tracking with configurable update frequencies
    - Comprehensive metric collection and aggregation
    - Performance monitoring and optimization recommendations
    - Automatic checkpoint and milestone detection
    - Error handling and recovery mechanisms
    """
    
    def __init__(
        self,
        session_id: str,
        total_steps: int,
        session_name: str,
        config: ProgressBarConfig,
        gradient_tracker: GradientNormTracker,
        metrics_collector: TrainingMetricsCollector,
        renderer: ProgressBarRenderer
    ):
        """
        Initialize training session.
        
        Args:
            session_id: Unique session identifier
            total_steps: Total number of training steps
            session_name: Human-readable session name
            config: Progress bar configuration
            gradient_tracker: Gradient norm tracking system
            metrics_collector: Metrics collection system
            renderer: Progress bar renderer
        """
        self.session_id = session_id
        self.total_steps = total_steps
        self.session_name = session_name
        self.config = config
        self.gradient_tracker = gradient_tracker
        self.metrics_collector = metrics_collector
        self.renderer = renderer
        
        # Session state
        self.current_step = 0
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.is_active = True
        self.is_paused = False
        self.completion_time: Optional[float] = None
        
        # Performance tracking
        self.step_times = deque(maxlen=1000)
        self.update_count = 0
        self.last_metrics: Dict[str, Any] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Progress bar instance
        self._progress_bar = None
        
        logger.info(f"Started training session: {session_name} ({session_id})")
    
    def update_progress(
        self,
        step: int,
        metrics: Dict[str, Any],
        gradients: Optional[Dict[str, mx.array]] = None
    ) -> None:
        """
        Update training progress with new metrics.
        
        Args:
            step: Current training step
            metrics: Dictionary of training metrics
            gradients: Gradient dictionary for norm computation (optional)
        """
        with self._lock:
            if not self.is_active or self.is_paused:
                return
            
            current_time = time.time()
            
            # Update step timing
            if self.current_step > 0:
                step_time = current_time - self.last_update_time
                self.step_times.append(step_time)
            
            self.current_step = step
            self.last_update_time = current_time
            self.update_count += 1
            
            # Compute gradient norm if gradients provided
            gradient_norm_result = None
            if gradients:
                try:
                    gradient_norm_result = self.gradient_tracker.compute_norm(gradients)
                    metrics['gradient_norm'] = gradient_norm_result.norm_value
                    metrics['gradient_computation_time'] = gradient_norm_result.computation_time
                except Exception as e:
                    logger.warning(f"Gradient norm computation failed: {e}")
            
            # Add performance metrics
            if self.step_times:
                metrics['step_time_avg'] = np.mean(list(self.step_times))
                metrics['step_time_std'] = np.std(list(self.step_times))
                metrics['steps_per_second'] = 1.0 / max(np.mean(list(self.step_times)), 1e-6)
            
            # Add session metrics
            elapsed_time = current_time - self.start_time
            metrics['elapsed_time'] = elapsed_time
            metrics['progress_percent'] = (step / self.total_steps) * 100
            metrics['eta_seconds'] = self._estimate_eta(step, elapsed_time)
            
            # Collect metrics
            try:
                self.metrics_collector.collect_metrics(metrics, step=step)
            except Exception as e:
                logger.warning(f"Metrics collection failed: {e}")
            
            # Update display
            self._update_display(step, metrics, gradient_norm_result)
            
            # Store last metrics
            self.last_metrics = metrics.copy()
    
    def _estimate_eta(self, current_step: int, elapsed_time: float) -> float:
        """Estimate time to completion."""
        if current_step <= 0:
            return float('inf')
        
        steps_remaining = self.total_steps - current_step
        avg_step_time = elapsed_time / current_step
        
        return steps_remaining * avg_step_time
    
    def _update_display(
        self,
        step: int,
        metrics: Dict[str, Any],
        gradient_norm_result: Optional[GradientNormResult]
    ) -> None:
        """Update progress bar display."""
        try:
            # Get display metrics from collector
            display_metrics = self.metrics_collector.get_display_metrics()
            
            # Add gradient norm details if available
            if gradient_norm_result:
                display_metrics.update({
                    'gradient_norm_strategy': gradient_norm_result.strategy_used,
                    'gradient_computation_time': gradient_norm_result.computation_time,
                    'gradient_cache_hit': gradient_norm_result.cache_hit,
                    'gradient_outliers': gradient_norm_result.outliers_detected
                })
            
            # Render progress bar
            self.renderer.render_progress(
                current_step=step,
                total_steps=self.total_steps,
                metrics=display_metrics,
                session_info={
                    'session_id': self.session_id,
                    'session_name': self.session_name,
                    'elapsed_time': time.time() - self.start_time,
                    'update_count': self.update_count
                }
            )
            
        except Exception as e:
            logger.warning(f"Display update failed: {e}")
    
    def pause(self) -> None:
        """Pause the training session."""
        with self._lock:
            if self.is_active and not self.is_paused:
                self.is_paused = True
                logger.info(f"Paused training session: {self.session_id}")
    
    def resume(self) -> None:
        """Resume the training session."""
        with self._lock:
            if self.is_active and self.is_paused:
                self.is_paused = False
                self.last_update_time = time.time()  # Reset timing
                logger.info(f"Resumed training session: {self.session_id}")
    
    def complete(self) -> Dict[str, Any]:
        """
        Complete the training session and return summary statistics.
        
        Returns:
            Dictionary containing session summary statistics
        """
        with self._lock:
            if not self.is_active:
                return {}
            
            self.is_active = False
            self.completion_time = time.time()
            
            # Calculate session statistics
            total_time = self.completion_time - self.start_time
            
            summary = {
                'session_id': self.session_id,
                'session_name': self.session_name,
                'total_steps': self.total_steps,
                'completed_steps': self.current_step,
                'completion_rate': self.current_step / self.total_steps,
                'total_time': total_time,
                'average_step_time': total_time / max(self.current_step, 1),
                'update_count': self.update_count,
                'final_metrics': self.last_metrics.copy()
            }
            
            # Add performance statistics
            if self.step_times:
                step_times_list = list(self.step_times)
                summary['performance_stats'] = {
                    'min_step_time': min(step_times_list),
                    'max_step_time': max(step_times_list),
                    'avg_step_time': np.mean(step_times_list),
                    'std_step_time': np.std(step_times_list),
                    'p95_step_time': np.percentile(step_times_list, 95),
                    'p99_step_time': np.percentile(step_times_list, 99)
                }
            
            # Get final metrics from collector
            try:
                collector_stats = self.metrics_collector.get_performance_stats()
                summary['metrics_collector_stats'] = collector_stats
            except Exception as e:
                logger.warning(f"Failed to get collector stats: {e}")
            
            # Get gradient tracker stats
            try:
                gradient_stats = self.gradient_tracker.get_performance_stats()
                summary['gradient_tracker_stats'] = gradient_stats
            except Exception as e:
                logger.warning(f"Failed to get gradient tracker stats: {e}")
            
            logger.info(
                f"Completed training session: {self.session_name} "
                f"({self.current_step}/{self.total_steps} steps in {total_time:.2f}s)"
            )
            
            return summary
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current session metrics."""
        with self._lock:
            return self.last_metrics.copy()
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get session information."""
        with self._lock:
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            
            return {
                'session_id': self.session_id,
                'session_name': self.session_name,
                'total_steps': self.total_steps,
                'current_step': self.current_step,
                'progress_percent': (self.current_step / self.total_steps) * 100,
                'elapsed_time': elapsed_time,
                'eta_seconds': self._estimate_eta(self.current_step, elapsed_time),
                'is_active': self.is_active,
                'is_paused': self.is_paused,
                'update_count': self.update_count,
                'start_time': self.start_time,
                'completion_time': self.completion_time
            }


class EnhancedProgressManager:
    """
    Singleton progress manager for coordinating all progress tracking components.
    
    This class implements the Singleton pattern to provide a global point of
    coordination for all progress tracking activities. It manages the lifecycle
    of training sessions, coordinates between different progress components,
    and provides a unified interface for the entire progress tracking system.
    
    Features:
    - Singleton pattern for global coordination
    - Session lifecycle management
    - Component coordination and dependency injection
    - Configuration management and validation
    - Performance monitoring and optimization
    - Error handling and recovery
    - Plugin architecture for extensibility
    """
    
    _instance: Optional['EnhancedProgressManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'EnhancedProgressManager':
        """Implement Singleton pattern with thread safety."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize progress manager (called only once due to Singleton)."""
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        
        # Configuration
        self.config: Optional[ProgressBarConfig] = None
        self.metric_display_configs: Dict[str, MetricDisplayConfig] = {}
        
        # Core components
        self.gradient_tracker: Optional[GradientNormTracker] = None
        self.metrics_collector: Optional[TrainingMetricsCollector] = None
        self.renderer_factory = ProgressBarRendererFactory()
        self.formatter_factory = MetricFormatterFactory()
        
        # Session management
        self.active_sessions: Dict[str, TrainingSession] = {}
        self.session_history: List[Dict[str, Any]] = []
        
        # Thread safety
        self._session_lock = threading.RLock()
        
        # Performance monitoring
        self.manager_stats = {
            'sessions_created': 0,
            'sessions_completed': 0,
            'total_updates': 0,
            'errors_handled': 0,
            'start_time': time.time()
        }
        
        # Plugin system
        self._plugins: Dict[str, Any] = {}
        self._observers: Set[Callable[[str, Dict[str, Any]], None]] = set()
        
        logger.info("Initialized EnhancedProgressManager singleton")
    
    @classmethod
    def get_instance(cls) -> 'EnhancedProgressManager':
        """Get the singleton instance of the progress manager."""
        return cls()
    
    def configure(
        self,
        config: ProgressBarConfig,
        metric_configs: Optional[Dict[str, MetricDisplayConfig]] = None
    ) -> None:
        """
        Configure the progress manager.
        
        Args:
            config: Main progress bar configuration
            metric_configs: Individual metric display configurations
        """
        try:
            # Validate configuration
            config.validate()
            
            self.config = config
            
            if metric_configs:
                self.metric_display_configs.update(metric_configs)
            
            # Initialize core components with configuration
            self._initialize_components()
            
            logger.info(f"Configured progress manager with display mode: {config.display_mode.value}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to configure progress manager: {e}") from e
    
    def _initialize_components(self) -> None:
        """Initialize core progress tracking components."""
        if not self.config:
            raise ConfigurationError("Progress manager not configured")
        
        # Initialize gradient tracker
        self.gradient_tracker = GradientNormTracker(
            enable_caching=self.config.enable_caching,
            cache_size=min(self.config.max_history_size // 10, 1000),
            enable_monitoring=True
        )
        
        # Initialize metrics collector
        self.metrics_collector = TrainingMetricsCollector(
            buffer_size=self.config.max_history_size,
            enable_anomaly_detection=self.config.enable_anomaly_detection,
            update_frequency=self.config.update_frequency
        )
        
        # Register custom metric definitions if provided
        for name, display_config in self.metric_display_configs.items():
            metric_def = MetricDefinition(
                name=name,
                metric_type=MetricType.CUSTOM,
                description=display_config.display_name,
                unit=display_config.unit
            )
            self.metrics_collector.register_metric(metric_def)
        
        logger.debug("Initialized progress manager components")
    
    def start_training_session(
        self,
        total_steps: int,
        session_name: str = "Training Session",
        session_id: Optional[str] = None
    ) -> TrainingSession:
        """
        Start a new training session.
        
        Args:
            total_steps: Total number of training steps
            session_name: Human-readable session name
            session_id: Unique session identifier (auto-generated if None)
            
        Returns:
            TrainingSession instance for progress tracking
            
        Raises:
            ConfigurationError: If progress manager is not configured
            ProgressTrackingError: If session creation fails
        """
        if not self.config:
            raise ConfigurationError("Progress manager must be configured before starting sessions")
        
        if not self.gradient_tracker or not self.metrics_collector:
            raise ConfigurationError("Core components not initialized")
        
        with self._session_lock:
            try:
                # Generate session ID if not provided
                if session_id is None:
                    session_id = f"session_{uuid.uuid4().hex[:8]}"
                
                # Check for duplicate session ID
                if session_id in self.active_sessions:
                    raise ProgressTrackingError(f"Session with ID {session_id} already exists")
                
                # Create renderer for the session
                renderer = self.renderer_factory.create_renderer(
                    self.config.display_mode.value,
                    {
                        'terminal_width': self.config.terminal_width,
                        'color_scheme': self.config.color_scheme,
                        'show_gradient_norm': self.config.show_gradient_norm,
                        'show_memory_usage': self.config.show_memory_usage,
                        'show_performance_stats': self.config.show_performance_stats
                    }
                )
                
                # Create training session
                session = TrainingSession(
                    session_id=session_id,
                    total_steps=total_steps,
                    session_name=session_name,
                    config=self.config,
                    gradient_tracker=self.gradient_tracker,
                    metrics_collector=self.metrics_collector,
                    renderer=renderer
                )
                
                # Register session
                self.active_sessions[session_id] = session
                self.manager_stats['sessions_created'] += 1
                
                # Notify observers
                self._notify_observers('session_started', {
                    'session_id': session_id,
                    'session_name': session_name,
                    'total_steps': total_steps
                })
                
                logger.info(f"Started training session: {session_name} ({session_id})")
                
                return session
                
            except Exception as e:
                self.manager_stats['errors_handled'] += 1
                raise ProgressTrackingError(f"Failed to start training session: {e}") from e
    
    def end_training_session(self, session_id: str) -> Dict[str, Any]:
        """
        End a training session and return summary statistics.
        
        Args:
            session_id: ID of the session to end
            
        Returns:
            Session summary statistics
            
        Raises:
            ProgressTrackingError: If session not found or ending fails
        """
        with self._session_lock:
            try:
                if session_id not in self.active_sessions:
                    raise ProgressTrackingError(f"Session {session_id} not found")
                
                session = self.active_sessions[session_id]
                
                # Complete the session
                summary = session.complete()
                
                # Move to history
                self.session_history.append(summary)
                del self.active_sessions[session_id]
                
                self.manager_stats['sessions_completed'] += 1
                
                # Notify observers
                self._notify_observers('session_ended', {
                    'session_id': session_id,
                    'summary': summary
                })
                
                logger.info(f"Ended training session: {session_id}")
                
                return summary
                
            except Exception as e:
                self.manager_stats['errors_handled'] += 1
                raise ProgressTrackingError(f"Failed to end training session: {e}") from e
    
    def get_active_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all active sessions."""
        with self._session_lock:
            return {
                session_id: session.get_session_info()
                for session_id, session in self.active_sessions.items()
            }
    
    def get_session_history(self) -> List[Dict[str, Any]]:
        """Get history of completed sessions."""
        with self._session_lock:
            return self.session_history.copy()
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get comprehensive manager statistics."""
        current_time = time.time()
        uptime = current_time - self.manager_stats['start_time']
        
        stats = self.manager_stats.copy()
        stats.update({
            'uptime_seconds': uptime,
            'active_sessions_count': len(self.active_sessions),
            'total_sessions_in_history': len(self.session_history),
            'observers_count': len(self._observers),
            'plugins_count': len(self._plugins)
        })
        
        # Add component stats if available
        if self.gradient_tracker:
            stats['gradient_tracker'] = self.gradient_tracker.get_performance_stats()
        
        if self.metrics_collector:
            stats['metrics_collector'] = self.metrics_collector.get_performance_stats()
        
        return stats
    
    def add_observer(self, observer: Callable[[str, Dict[str, Any]], None]) -> None:
        """
        Add an observer for progress manager events.
        
        Args:
            observer: Callback function to receive event notifications
        """
        self._observers.add(observer)
        logger.debug(f"Added progress manager observer: {observer}")
    
    def remove_observer(self, observer: Callable[[str, Dict[str, Any]], None]) -> None:
        """
        Remove an observer.
        
        Args:
            observer: Observer to remove
        """
        self._observers.discard(observer)
        logger.debug(f"Removed progress manager observer: {observer}")
    
    def _notify_observers(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Notify all observers of an event."""
        for observer in self._observers.copy():
            try:
                observer(event_type, event_data)
            except Exception as e:
                logger.warning(f"Observer notification failed: {e}")
    
    def register_plugin(self, name: str, plugin: Any) -> None:
        """
        Register a plugin for extending progress manager functionality.
        
        Args:
            name: Plugin name
            plugin: Plugin instance
        """
        self._plugins[name] = plugin
        logger.info(f"Registered progress manager plugin: {name}")
    
    def get_plugin(self, name: str) -> Optional[Any]:
        """
        Get a registered plugin.
        
        Args:
            name: Plugin name
            
        Returns:
            Plugin instance or None if not found
        """
        return self._plugins.get(name)
    
    def reset(self) -> None:
        """Reset the progress manager to initial state."""
        with self._session_lock:
            # End all active sessions
            for session_id in list(self.active_sessions.keys()):
                try:
                    self.end_training_session(session_id)
                except Exception as e:
                    logger.warning(f"Failed to end session {session_id} during reset: {e}")
            
            # Clear history and stats
            self.session_history.clear()
            self.manager_stats = {
                'sessions_created': 0,
                'sessions_completed': 0,
                'total_updates': 0,
                'errors_handled': 0,
                'start_time': time.time()
            }
            
            # Reset components
            if self.gradient_tracker:
                self.gradient_tracker.reset_stats()
            
            if self.metrics_collector:
                self.metrics_collector.reset_metrics()
            
            logger.info("Reset progress manager to initial state")
    
    def export_session_data(
        self,
        output_path: Path,
        session_ids: Optional[List[str]] = None,
        include_metrics: bool = True,
        format: str = "json"
    ) -> None:
        """
        Export session data to file.
        
        Args:
            output_path: Path for output file
            session_ids: Specific sessions to export (all if None)
            include_metrics: Whether to include detailed metrics
            format: Export format ("json" or "csv")
        """
        with self._session_lock:
            try:
                export_data = {
                    'export_timestamp': time.time(),
                    'manager_stats': self.get_manager_stats(),
                    'active_sessions': {},
                    'session_history': []
                }
                
                # Export active sessions
                for session_id, session in self.active_sessions.items():
                    if session_ids is None or session_id in session_ids:
                        session_data = session.get_session_info()
                        if include_metrics:
                            session_data['current_metrics'] = session.get_current_metrics()
                        export_data['active_sessions'][session_id] = session_data
                
                # Export session history
                for session_summary in self.session_history:
                    if session_ids is None or session_summary['session_id'] in session_ids:
                        export_data['session_history'].append(session_summary)
                
                # Write to file
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                if format.lower() == "json":
                    with open(output_path, 'w') as f:
                        json.dump(export_data, f, indent=2, default=str)
                elif format.lower() == "csv":
                    # Export as CSV (simplified format)
                    import csv
                    with open(output_path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        
                        # Write header
                        writer.writerow([
                            "session_id", "session_name", "status", "progress_percent",
                            "elapsed_time", "total_steps", "current_step"
                        ])
                        
                        # Write active sessions
                        for session_id, session_data in export_data['active_sessions'].items():
                            writer.writerow([
                                session_id,
                                session_data.get('session_name', ''),
                                'active',
                                session_data.get('progress_percent', 0),
                                session_data.get('elapsed_time', 0),
                                session_data.get('total_steps', 0),
                                session_data.get('current_step', 0)
                            ])
                        
                        # Write completed sessions
                        for session_data in export_data['session_history']:
                            writer.writerow([
                                session_data.get('session_id', ''),
                                session_data.get('session_name', ''),
                                'completed',
                                session_data.get('completion_rate', 0) * 100,
                                session_data.get('total_time', 0),
                                session_data.get('total_steps', 0),
                                session_data.get('completed_steps', 0)
                            ])
                
                logger.info(f"Exported session data to {output_path}")
                
            except Exception as e:
                raise ProgressTrackingError(f"Failed to export session data: {e}") from e


# Export all classes and functions
__all__ = [
    'DisplayMode',
    'ProgressUpdateFrequency',
    'ProgressBarConfig',
    'MetricDisplayConfig',
    'TrainingSession',
    'EnhancedProgressManager',
]