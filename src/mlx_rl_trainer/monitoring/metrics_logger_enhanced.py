"""
Enhanced Metrics Logger with Correlation IDs and Structured Logging

This module provides an enhanced metrics logger with comprehensive statistics collection,
correlation ID support for tracking context across operations, and structured logging
for better analysis. It serves as a drop-in replacement for the existing metrics_logger.py
with additional features for enterprise-grade monitoring.

Key Features:
- Correlation ID support for tracking related metrics
- Structured logging with JSON format
- Comprehensive metrics validation
- Thread-safe operations for concurrent environments
- Memory-efficient storage with automatic cleanup
- Multiple output formats (CSV, JSON)
- Real-time statistics and aggregation
"""

import logging
import csv
import json
import threading
import time
import gc
import uuid
import datetime
from typing import Any, Dict, List, Optional, Union, Tuple, Set, Callable
from pathlib import Path
import numpy as np
import traceback
import sys
import os

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

from mlx_rl_trainer.core.config import ExperimentConfig
from mlx_rl_trainer.monitoring.metrics_interface import (
    MetricContext, MetricValidationResult, ValidationStrategy,
    BasicValidationStrategy, StandardValidationStrategy, StrictValidationStrategy,
    MetricValidationLevel, MetricName, MetricValue, MetricDict, CorrelationId
)

# Configure logger
logger = logging.getLogger(__name__)


# Configure structured logging handler if not already configured
class StructuredLogFormatter(logging.Formatter):
    """
    JSON-based structured log formatter.
    
    This formatter outputs logs in a structured JSON format that is
    easier to parse and analyze with log management tools.
    """
    
    def __init__(self, include_timestamp: bool = True):
        """Initialize structured log formatter."""
        super().__init__()
        self.include_timestamp = include_timestamp
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_data = {
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add timestamp if requested
        if self.include_timestamp:
            log_data['timestamp'] = datetime.datetime.fromtimestamp(
                record.created
            ).isoformat()
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add correlation ID if present
        if hasattr(record, 'correlation_id'):
            log_data['correlation_id'] = record.correlation_id
        
        # Add any extra attributes
        for key, value in record.__dict__.items():
            if key not in {
                'args', 'asctime', 'created', 'exc_info', 'exc_text',
                'filename', 'funcName', 'id', 'levelname', 'levelno',
                'lineno', 'module', 'msecs', 'message', 'msg', 'name',
                'pathname', 'process', 'processName', 'relativeCreated',
                'stack_info', 'thread', 'threadName'
            }:
                log_data[key] = value
        
        return json.dumps(log_data)


# Add correlation ID filter for logging
class CorrelationIdFilter(logging.Filter):
    """
    Logging filter that adds correlation IDs to log records.
    
    This filter adds the current correlation ID to log records,
    enabling tracking of related log entries across components.
    """
    
    def __init__(self, correlation_id_provider: Callable[[], Optional[str]]):
        """
        Initialize correlation ID filter.
        
        Args:
            correlation_id_provider: Function that returns the current correlation ID
        """
        super().__init__()
        self.get_correlation_id = correlation_id_provider
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation ID to log record."""
        correlation_id = self.get_correlation_id()
        if correlation_id:
            record.correlation_id = correlation_id
        return True


class EnhancedMetricsLogger:
    """
    Enhanced metrics logger with comprehensive statistics and correlation ID support.

    Features:
    - Correlation ID tracking for related metrics
    - Structured logging with JSON format
    - CSV and JSON export
    - Real-time statistics and aggregation
    - Memory-efficient storage with automatic cleanup
    - Thread-safe operations for concurrent environments
    - Comprehensive validation with configurable strategies
    - Multiple output formats
    """

    def __init__(
        self, 
        config: ExperimentConfig, 
        run_id: str,
        validation_level: MetricValidationLevel = MetricValidationLevel.STANDARD,
        enable_structured_logging: bool = True
    ):
        """
        Initialize enhanced metrics logger.

        Args:
            config: Experiment configuration
            run_id: Unique run identifier
            validation_level: Level of metrics validation to apply
            enable_structured_logging: Whether to enable structured logging
        """
        self.config = config
        self.run_id = run_id
        self.output_dir = config.trainer.output_dir
        self.validation_level = validation_level
        self.enable_structured_logging = enable_structured_logging

        # File paths
        self.csv_path = self.output_dir / "training_metrics.csv"
        self.json_path = self.output_dir / "training_metrics.json"
        self.summary_path = self.output_dir / "metrics_summary.json"
        self.structured_log_path = self.output_dir / "metrics_logs.jsonl"

        # CSV file handle
        self._csv_file = None
        self._csv_writer = None
        self._headers = []

        # JSON storage
        self._json_data: List[Dict[str, Any]] = []

        # Threading
        self._lock = threading.RLock()

        # Statistics
        self._write_count = 0
        self._error_count = 0
        self._last_cleanup_time = time.time()
        self._cleanup_interval = 100

        # Correlation ID tracking
        self._context_stack: List[MetricContext] = []
        self._current_context = MetricContext()

        # Validation strategy based on level
        self._setup_validation_strategy()

        # Configure structured logging if enabled
        if enable_structured_logging:
            self._setup_structured_logging()

        # Initialize
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._open_csv_file()

        logger.info(
            f"Initialized EnhancedMetricsLogger: {self.output_dir}",
            extra={"correlation_id": self._current_context.correlation_id}
        )
    
    def _setup_validation_strategy(self) -> None:
        """Set up the appropriate validation strategy based on level."""
        if self.validation_level == MetricValidationLevel.NONE:
            self.validation_strategy = None
        elif self.validation_level == MetricValidationLevel.BASIC:
            self.validation_strategy = BasicValidationStrategy()
        elif self.validation_level == MetricValidationLevel.STANDARD:
            # Define standard ranges for common metrics
            ranges = {
                "loss": (0.0, None),  # Loss should be non-negative
                "gradient_norm": (0.0, None),  # Gradient norm should be non-negative
                "learning_rate": (0.0, 1.0),  # Learning rate typically between 0 and 1
                "memory_usage_mb": (0.0, None),  # Memory usage should be non-negative
                "step_time_s": (0.0, None),  # Step time should be non-negative
            }
            self.validation_strategy = StandardValidationStrategy(ranges=ranges)
        elif self.validation_level == MetricValidationLevel.STRICT:
            # Define standard ranges
            ranges = {
                "loss": (0.0, None),
                "gradient_norm": (0.0, None),
                "learning_rate": (0.0, 1.0),
                "memory_usage_mb": (0.0, None),
                "step_time_s": (0.0, None),
            }
            
            # Define required metrics
            required_metrics = {
                "loss", "step", "run_id"
            }
            
            # Define custom validators
            def validate_run_id(value: Any) -> Tuple[bool, Optional[str]]:
                if not isinstance(value, str) or not value:
                    return False, "Run ID must be a non-empty string"
                return True, None
            
            custom_validators = {
                "run_id": validate_run_id
            }
            
            self.validation_strategy = StrictValidationStrategy(
                ranges=ranges,
                required_metrics=required_metrics,
                custom_validators=custom_validators
            )
        else:
            # Default to basic validation
            self.validation_strategy = BasicValidationStrategy()
    
    def _setup_structured_logging(self) -> None:
        """Set up structured logging with correlation ID support."""
        # Create a file handler for structured logs
        file_handler = logging.FileHandler(self.structured_log_path)
        file_handler.setFormatter(StructuredLogFormatter())
        
        # Add correlation ID filter
        correlation_id_filter = CorrelationIdFilter(
            lambda: self._current_context.correlation_id if hasattr(self, '_current_context') else None
        )
        file_handler.addFilter(correlation_id_filter)
        
        # Add handler to logger
        metrics_logger = logging.getLogger('mlx_rl_trainer.monitoring')
        metrics_logger.addHandler(file_handler)
        
        # Set level to DEBUG to capture all logs
        metrics_logger.setLevel(logging.DEBUG)

    def _open_csv_file(self) -> None:
        """Open CSV file for writing."""
        try:
            self._csv_file = open(self.csv_path, "a", newline="", encoding="utf-8")
            logger.info(
                f"Opened CSV file: {self.csv_path}",
                extra={"correlation_id": self._current_context.correlation_id}
            )
        except OSError as e:
            logger.error(
                f"Failed to open CSV file: {e}",
                extra={"correlation_id": self._current_context.correlation_id}
            )
            self._csv_file = None
    
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

    def log_metrics(
        self, 
        metrics: Dict[str, Any], 
        step: int,
        context: Optional[MetricContext] = None
    ) -> None:
        """
        Log metrics to all outputs with correlation ID support.

        Args:
            metrics: Dictionary of metrics
            step: Training step
            context: Optional metric context (uses current if None)
        """
        # Use provided context or current context
        ctx = context or self._current_context
        
        if not self._csv_file or self._csv_file.closed:
            logger.warning(
                "CSV file not open, cannot log metrics",
                extra={"correlation_id": ctx.correlation_id}
            )
            return

        # Prepare data
        data = {
            "step": step, 
            "run_id": self.run_id, 
            "timestamp": time.time(),
            "correlation_id": ctx.correlation_id
        }
        
        # Add context tags and metadata
        for key, value in ctx.tags.items():
            data[f"tag_{key}"] = value
        
        # Validate metrics if strategy is available
        if self.validation_strategy:
            validation_results = {}
            for key, value in metrics.items():
                result = self.validation_strategy.validate(key, value)
                validation_results[key] = result
                
                if not result.is_valid:
                    error_msg = "; ".join(result.errors)
                    logger.warning(
                        f"Invalid metric '{key}': {error_msg}",
                        extra={
                            "correlation_id": ctx.correlation_id,
                            "metric_name": key,
                            "metric_value": str(value),
                            "validation_errors": result.errors
                        }
                    )

        # Convert metrics
        for key, value in metrics.items():
            try:
                if (
                    isinstance(value, (mx.array, np.ndarray))
                    if MLX_AVAILABLE
                    else isinstance(value, np.ndarray)
                ):
                    if value.size == 1:
                        data[key] = float(value.item())
                    else:
                        data[key] = float(np.mean(value))
                elif isinstance(value, (int, float, bool, str)) or value is None:
                    data[key] = value
                elif isinstance(value, (list, tuple)):
                    data[f"{key}_count"] = len(value)
                    if value and isinstance(value[0], (int, float)):
                        data[f"{key}_mean"] = np.mean(value)
                else:
                    data[key] = str(value)
            except Exception as e:
                logger.warning(
                    f"Failed to convert metric '{key}': {e}",
                    extra={
                        "correlation_id": ctx.correlation_id,
                        "metric_name": key,
                        "error": str(e)
                    }
                )
                data[key] = "conversion_error"

        # Write to CSV
        with self._lock:
            try:
                # Update CSV writer if needed
                fieldnames = sorted(data.keys())
                if self._csv_writer is None or self._headers != fieldnames:
                    needs_header = (
                        not self.csv_path.exists() or self.csv_path.stat().st_size == 0
                    )
                    self._headers = fieldnames
                    self._csv_writer = csv.DictWriter(
                        self._csv_file, fieldnames=self._headers, extrasaction="ignore"
                    )
                    if needs_header:
                        self._csv_writer.writeheader()

                # Write row
                self._csv_writer.writerow(data)
                self._csv_file.flush()
                self._write_count += 1

                # Store in JSON
                self._json_data.append(data)

                # Periodic cleanup
                current_time = time.time()
                if (
                    self._write_count % self._cleanup_interval == 0
                    or current_time - self._last_cleanup_time > 60
                ):
                    self._cleanup()
                    self._last_cleanup_time = current_time

                logger.debug(
                    f"Logged metrics at step {step}",
                    extra={
                        "correlation_id": ctx.correlation_id,
                        "step": step,
                        "metric_count": len(metrics)
                    }
                )

            except Exception as e:
                self._error_count += 1
                logger.error(
                    f"Error writing metrics: {e}",
                    extra={
                        "correlation_id": ctx.correlation_id,
                        "error": str(e),
                        "error_count": self._error_count
                    }
                )

                if self._error_count < 3:
                    try:
                        self._csv_writer = None
                        logger.info(
                            "Attempting to recreate CSV writer...",
                            extra={"correlation_id": ctx.correlation_id}
                        )
                    except:
                        pass

    def _cleanup(self) -> None:
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

    def export_json(self, filepath: Optional[Path] = None) -> None:
        """
        Export metrics to JSON.
        
        Args:
            filepath: Path to export JSON (uses default if None)
        """
        if filepath is None:
            filepath = self.json_path

        with self._lock:
            with open(filepath, "w") as f:
                json.dump(self._json_data, f, indent=2)

        logger.info(
            f"Exported metrics to JSON: {filepath}",
            extra={
                "correlation_id": self._current_context.correlation_id,
                "record_count": len(self._json_data)
            }
        )

    def generate_summary(self) -> Dict[str, Any]:
        """
        Generate summary statistics.
        
        Returns:
            Dictionary of summary statistics
        """
        if not self._json_data:
            return {}

        with self._lock:
            # Convert to DataFrame for easy analysis
            if PANDAS_AVAILABLE:
                df = pd.DataFrame(self._json_data)

                summary = {
                    "total_steps": len(df), 
                    "metrics": {},
                    "correlation_id": self._current_context.correlation_id,
                    "run_id": self.run_id,
                    "timestamp": time.time()
                }

                # Calculate statistics for numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if col not in ["step", "timestamp"]:
                        summary["metrics"][col] = {
                            "mean": float(df[col].mean()),
                            "std": float(df[col].std()),
                            "min": float(df[col].min()),
                            "max": float(df[col].max()),
                            "final": float(df[col].iloc[-1]),
                        }

                return summary
            else:
                # Basic summary without pandas
                return {
                    "total_steps": len(self._json_data),
                    "write_count": self._write_count,
                    "error_count": self._error_count,
                    "correlation_id": self._current_context.correlation_id,
                    "run_id": self.run_id,
                    "timestamp": time.time()
                }

    def export_summary(self, filepath: Optional[Path] = None) -> None:
        """
        Export summary statistics to JSON.
        
        Args:
            filepath: Path to export summary (uses default if None)
        """
        if filepath is None:
            filepath = self.summary_path

        summary = self.generate_summary()

        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(
            f"Exported summary: {filepath}",
            extra={"correlation_id": self._current_context.correlation_id}
        )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get logger statistics.
        
        Returns:
            Dictionary of logger statistics
        """
        return {
            "write_count": self._write_count,
            "error_count": self._error_count,
            "file_size_mb": self.csv_path.stat().st_size / 1048576
            if self.csv_path.exists()
            else 0,
            "json_entries": len(self._json_data),
            "last_cleanup_time": self._last_cleanup_time,
            "correlation_id": self._current_context.correlation_id,
            "context_stack_depth": len(self._context_stack)
        }

    def close(self) -> None:
        """Close logger and save final data."""
        ctx_id = self._current_context.correlation_id
        
        with self._lock:
            # Close CSV file
            if self._csv_file and not self._csv_file.closed:
                try:
                    self._csv_file.flush()
                    self._csv_file.close()
                    logger.info(
                        f"Closed CSV file. Total writes: {self._write_count}",
                        extra={"correlation_id": ctx_id}
                    )
                except Exception as e:
                    logger.error(
                        f"Error closing CSV file: {e}",
                        extra={"correlation_id": ctx_id, "error": str(e)}
                    )
                finally:
                    self._csv_file = None
                    self._csv_writer = None

            # Export JSON
            if self._json_data:
                self.export_json()
                self.export_summary()

        # Final cleanup
        self._cleanup()
        
        logger.info(
            "Metrics logger closed successfully",
            extra={"correlation_id": ctx_id}
        )


# Dependencies: pandas (optional), numpy, mlx
# Install: pip install pandas numpy mlx
# Usage: logger = EnhancedMetricsLogger(config, run_id="run_001")
#        logger.log_metrics({'loss': 0.5}, step=100)
# Status: Complete and commit-ready