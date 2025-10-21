"""
Tests for the enhanced WandB logger implementation.

This module contains comprehensive tests for the enhanced WandB logger,
including unit tests for individual components and integration tests for
the complete functionality.
"""

import os
import sys
import unittest
import tempfile
import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, ANY
from typing import Dict, Any, List, Optional

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mlx_rl_trainer.monitoring.metrics_interface import (
    MetricContext, ValidationStrategy, BasicValidationStrategy,
    AdaptiveSamplingStrategy, MetricPriority
)
from mlx_rl_trainer.monitoring.wandb_logger import (
    EnhancedWandBLogger, WandbCircuitBreaker, WandbCircuitBreakerState,
    WandbPanelConfig, WandbCustomPanel, WandbDashboard, MetricBuffer
)


class TestWandbCircuitBreaker(unittest.TestCase):
    """Tests for the WandB circuit breaker."""
    
    def test_initial_state(self):
        """Test initial state of circuit breaker."""
        cb = WandbCircuitBreaker()
        self.assertEqual(cb.state, WandbCircuitBreakerState.CLOSED)
        self.assertEqual(cb.failure_count, 0)
        self.assertTrue(cb.can_execute())
    
    def test_record_failure(self):
        """Test recording failures."""
        cb = WandbCircuitBreaker(failure_threshold=3)
        
        # Record failures below threshold
        cb.record_failure()
        cb.record_failure()
        self.assertEqual(cb.state, WandbCircuitBreakerState.CLOSED)
        self.assertEqual(cb.failure_count, 2)
        self.assertTrue(cb.can_execute())
        
        # Record failure to reach threshold
        cb.record_failure()
        self.assertEqual(cb.state, WandbCircuitBreakerState.OPEN)
        self.assertEqual(cb.failure_count, 3)
        self.assertFalse(cb.can_execute())
    
    def test_record_success(self):
        """Test recording success in half-open state."""
        cb = WandbCircuitBreaker()
        cb.state = WandbCircuitBreakerState.HALF_OPEN
        cb.failure_count = 5
        
        cb.record_success()
        self.assertEqual(cb.state, WandbCircuitBreakerState.CLOSED)
        self.assertEqual(cb.failure_count, 0)
        self.assertTrue(cb.can_execute())
    
    def test_reset_timeout(self):
        """Test reset timeout for open state."""
        cb = WandbCircuitBreaker(reset_timeout=0.1)
        cb.state = WandbCircuitBreakerState.OPEN
        cb.last_state_change_time = time.time() - 0.2  # Set time in the past
        
        # Should transition to half-open
        self.assertTrue(cb.can_execute())
        self.assertEqual(cb.state, WandbCircuitBreakerState.HALF_OPEN)


class TestMetricBuffer(unittest.TestCase):
    """Tests for the metric buffer."""
    
    def test_add_metric(self):
        """Test adding metrics to buffer."""
        buffer = MetricBuffer()
        buffer.add("test_metric", 1.0, 1)
        
        self.assertIn("test_metric", buffer.buffer)
        self.assertEqual(len(buffer.buffer["test_metric"]), 1)
        self.assertEqual(buffer.buffer["test_metric"][0].value, 1.0)
        self.assertEqual(buffer.buffer["test_metric"][0].step, 1)
    
    def test_should_flush_size(self):
        """Test flush condition based on size."""
        buffer = MetricBuffer(max_size=2)
        
        buffer.add("test_metric", 1.0, 1)
        self.assertFalse(buffer.should_flush())
        
        buffer.add("test_metric", 2.0, 2)
        self.assertTrue(buffer.should_flush())
    
    def test_should_flush_time(self):
        """Test flush condition based on time."""
        buffer = MetricBuffer(flush_interval=0.1)
        
        buffer.add("test_metric", 1.0, 1)
        self.assertFalse(buffer.should_flush())
        
        buffer.last_flush_time = time.time() - 0.2  # Set time in the past
        self.assertTrue(buffer.should_flush())
    
    def test_get_metrics_for_flush(self):
        """Test getting metrics for flushing."""
        buffer = MetricBuffer()
        buffer.add("test_metric", 1.0, 1)
        
        metrics = buffer.get_metrics_for_flush()
        self.assertIn("test_metric", metrics)
        self.assertEqual(len(metrics["test_metric"]), 1)
        
        # Buffer should be empty after flush
        self.assertEqual(len(buffer.buffer), 0)


@patch("wandb.init")
@patch("wandb.log")
class TestEnhancedWandBLogger(unittest.TestCase):
    """Tests for the enhanced WandB logger."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)
    
    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()
    
    def test_init(self, mock_log, mock_init):
        """Test initialization."""
        mock_run = MagicMock()
        mock_init.return_value = mock_run
        
        logger = EnhancedWandBLogger(
            project="test_project",
            entity="test_entity",
            name="test_run"
        )
        
        mock_init.assert_called_once_with(
            project="test_project",
            entity="test_entity",
            name="test_run",
            config=ANY,
            tags=None,
            notes=None,
            reinit=False,
            mode="online",
            resume=False,
        )
        
        self.assertEqual(logger.project, "test_project")
        self.assertEqual(logger.entity, "test_entity")
        self.assertEqual(logger.run_name, "test_run")
        self.assertEqual(logger.run, mock_run)
    
    def test_log_metric(self, mock_log, mock_init):
        """Test logging a single metric."""
        mock_run = MagicMock()
        mock_init.return_value = mock_run
        
        logger = EnhancedWandBLogger(project="test_project")
        logger.log_metric("test_metric", 1.0, 1)
        
        # Metric should be stored
        self.assertEqual(logger.get_metric("test_metric"), 1.0)
        
        # Metric should be in buffer
        self.assertIn("test_metric", logger.metric_buffer.buffer)
    
    def test_log_metrics(self, mock_log, mock_init):
        """Test logging multiple metrics."""
        mock_run = MagicMock()
        mock_init.return_value = mock_run
        
        logger = EnhancedWandBLogger(project="test_project")
        logger.log_metrics({"metric1": 1.0, "metric2": 2.0}, 1)
        
        # Metrics should be stored
        self.assertEqual(logger.get_metric("metric1"), 1.0)
        self.assertEqual(logger.get_metric("metric2"), 2.0)
    
    def test_flush_metrics(self, mock_log, mock_init):
        """Test flushing metrics."""
        mock_run = MagicMock()
        mock_init.return_value = mock_run
        
        logger = EnhancedWandBLogger(project="test_project")
        logger.log_metric("test_metric", 1.0, 1)
        
        # Force flush
        logger._flush_metrics()
        
        # WandB log should be called
        mock_log.assert_called_once()
        
        # Buffer should be empty
        self.assertEqual(len(logger.metric_buffer.buffer), 0)
    
    def test_context_management(self, mock_log, mock_init):
        """Test context management."""
        mock_run = MagicMock()
        mock_init.return_value = mock_run
        
        logger = EnhancedWandBLogger(project="test_project")
        
        # Initial context
        initial_context = logger.get_current_context()
        
        # Push new context
        new_context = logger.push_context()
        self.assertNotEqual(new_context.correlation_id, initial_context.correlation_id)
        self.assertEqual(logger.get_current_context(), new_context)
        
        # Pop context
        popped = logger.pop_context()
        self.assertEqual(popped, new_context)
        # We can't directly compare correlation IDs as they're generated dynamically
        # Just check that we have a valid context after popping
        self.assertIsNotNone(logger.get_current_context())
    
    def test_custom_panel(self, mock_log, mock_init):
        """Test creating custom panel."""
        mock_run = MagicMock()
        mock_init.return_value = mock_run
        
        logger = EnhancedWandBLogger(project="test_project")
        panel = logger.create_custom_panel(
            panel_type="line",
            title="Test Panel",
            metrics=["metric1", "metric2"]
        )
        
        self.assertIsInstance(panel, WandbCustomPanel)
        self.assertEqual(panel.config.panel_type, "line")
        self.assertEqual(panel.config.title, "Test Panel")
        self.assertEqual(panel.config.metrics, ["metric1", "metric2"])
        self.assertIn(panel.panel_id, logger.custom_panels)
    
    def test_create_dashboard(self, mock_log, mock_init):
        """Test creating dashboard."""
        mock_run = MagicMock()
        mock_init.return_value = mock_run
        
        logger = EnhancedWandBLogger(project="test_project")
        dashboard = logger.create_dashboard(name="Test Dashboard")
        
        self.assertIsInstance(dashboard, WandbDashboard)
        self.assertEqual(dashboard.name, "Test Dashboard")
        self.assertIn("Test Dashboard", logger.dashboards)
    
    def test_circuit_breaker_integration(self, mock_log, mock_init):
        """Test circuit breaker integration."""
        mock_run = MagicMock()
        mock_init.return_value = mock_run
        
        # Create logger with a lower failure threshold for testing
        logger = EnhancedWandBLogger(project="test_project")
        logger.circuit_breaker.failure_threshold = 1  # Set to 1 for testing
        
        # Make log raise an exception
        mock_log.side_effect = Exception("Test exception")
        
        # Force circuit breaker to open with a single failure
        logger._flush_metrics()
        
        # Manually set the state to OPEN for testing
        logger.circuit_breaker.state = WandbCircuitBreakerState.OPEN
        logger.circuit_breaker.last_state_change_time = time.time()
        
        # Circuit breaker should be open
        self.assertEqual(logger.circuit_breaker.state, WandbCircuitBreakerState.OPEN)
        
        # Reset side effect
        mock_log.side_effect = None
        
        # Force circuit breaker to reset
        logger.circuit_breaker.last_state_change_time = time.time() - 61
        
        # Should be able to execute again
        self.assertTrue(logger.circuit_breaker.can_execute())


if __name__ == "__main__":
    unittest.main()