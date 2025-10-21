"""
Enhanced Progress Bar System for MLX RL Trainer

This module provides a comprehensive progress tracking system that displays
gradient norms, training metrics, memory usage, and performance indicators
in real-time during training runs.

Architecture:
- Observer Pattern: For metric collection and progress bar updates
- Strategy Pattern: For different progress bar rendering strategies
- Factory Pattern: For creating progress bar configurations
- Singleton Pattern: For global progress bar manager
- Command Pattern: For metric update operations

Key Features:
- Real-time gradient norm computation and display
- Comprehensive training metrics visualization
- Thread-safe metric updates
- Memory usage and performance tracking
- Configurable display formats
- Extensible metric system

Example:
    ```python
    from mlx_rl_trainer.monitoring.progress import EnhancedProgressBarManager
    
    # Initialize progress manager
    progress_manager = EnhancedProgressBarManager.get_instance()
    
    # Configure progress bar
    config = ProgressBarConfig(
        show_gradient_norm=True,
        show_memory_usage=True,
        update_frequency=1.0
    )
    progress_manager.configure(config)
    
    # Start training with enhanced progress tracking
    with progress_manager.create_progress_bar(total_steps=1000) as pbar:
        for step in range(1000):
            # Training step...
            metrics = {
                'loss': 0.5,
                'grad_norm': 2.3,
                'memory_mb': 1024.5
            }
            pbar.update_metrics(metrics)
    ```
"""

from .enhanced_progress_manager import (
    EnhancedProgressManager,
    ProgressBarConfig,
    MetricDisplayConfig,
    TrainingSession,
    DisplayMode,
)
from .gradient_tracker import (
    GradientNormTracker,
    GradientNormStrategy,
    L2NormStrategy,
    LayerWiseNormStrategy,
    AdaptiveNormStrategy,
)
from .metrics_collector import (
    TrainingMetricsCollector,
    MetricBuffer,
    MetricDefinition,
    MetricType,
)
from .progress_renderer import (
    ProgressBarRenderer,
    CompactRenderer,
    DetailedRenderer,
    MinimalRenderer,
)
from .metric_formatters import (
    MetricFormatter,
    NumericFormatter,
    MemoryFormatter,
    TimeFormatter,
    PercentageFormatter,
)
from .exceptions import (
    ProgressTrackingError,
    MetricComputationError,
    RendererError,
    ConfigurationError,
)

__all__ = [
    # Core components
    'EnhancedProgressManager',
    'ProgressBarConfig',
    'MetricDisplayConfig',
    'TrainingSession',
    'DisplayMode',
    
    # Gradient tracking
    'GradientNormTracker',
    'GradientNormStrategy',
    'L2NormStrategy',
    'LayerWiseNormStrategy',
    'AdaptiveNormStrategy',
    
    # Metrics collection
    'TrainingMetricsCollector',
    'MetricBuffer',
    'MetricDefinition',
    'MetricType',
    
    # Rendering
    'ProgressBarRenderer',
    'CompactRenderer',
    'DetailedRenderer',
    'MinimalRenderer',
    
    # Formatting
    'MetricFormatter',
    'NumericFormatter',
    'MemoryFormatter',
    'TimeFormatter',
    'PercentageFormatter',
    
    # Exceptions
    'ProgressTrackingError',
    'MetricComputationError',
    'RendererError',
    'ConfigurationError',
]

# Version information
__version__ = "1.0.0"
__author__ = "MLX RL Trainer Team"
__email__ = "support@mlx-rl-trainer.com"

# Module-level configuration
DEFAULT_UPDATE_FREQUENCY = 1.0  # seconds
DEFAULT_METRIC_BUFFER_SIZE = 100
DEFAULT_GRADIENT_NORM_THRESHOLD = 10.0
DEFAULT_MEMORY_WARNING_THRESHOLD = 8192  # MB

# Logging configuration
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())