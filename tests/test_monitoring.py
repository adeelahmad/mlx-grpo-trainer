#!/usr/bin/env python
"""
Unit tests for enhanced monitoring system.
"""

import unittest
from pathlib import Path
import tempfile
import shutil
import numpy as np

from mlx_rl_trainer.monitoring.stats_collector import (
    ComprehensiveStatsCollector,
    TrainingStats,
    MetricStats,
)
from mlx_rl_trainer.monitoring.chart_generator import ChartGenerator
from mlx_rl_trainer.utils.memory_profiler import MemoryProfiler


class TestStatsCollector(unittest.TestCase):
    """Test stats collector."""

    def setUp(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.collector = ComprehensiveStatsCollector(output_dir=Path(self.temp_dir))

    def tearDown(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir)

    def test_record_metric(self):
        """Test metric recording."""
        self.collector.record_metric("loss", 0.5, step=0)
        self.assertIn("loss", self.collector.metrics)
        self.assertEqual(len(self.collector.metrics["loss"].values), 1)

    def test_get_statistics(self):
        """Test statistics calculation."""
        for i in range(10):
            self.collector.record_metric("loss", float(i), step=i)

        stats = self.collector.get_metric_stats("loss")
        self.assertIsNotNone(stats)
        self.assertAlmostEqual(stats["mean"], 4.5)

    def test_moving_average(self):
        """Test moving average."""
        for i in range(20):
            self.collector.record_metric("loss", float(i), step=i)

        ma = self.collector.get_moving_averages(window=5)
        self.assertIn("loss", ma)
        self.assertGreater(len(ma["loss"]), 0)


class TestChartGenerator(unittest.TestCase):
    """Test chart generator."""

    def setUp(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.generator = ChartGenerator(output_dir=Path(self.temp_dir))

    def tearDown(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir)

    def test_plot_training_curves(self):
        """Test training curves."""
        data = {"loss": list(np.random.rand(100)), "reward": list(np.random.rand(100))}

        output_path = self.generator.plot_training_curves(data)
        self.assertTrue(output_path.exists())

    def test_plot_reward_distribution(self):
        """Test reward distribution."""
        rewards = list(np.random.rand(100))
        output_path = self.generator.plot_reward_distribution(rewards)
        self.assertTrue(output_path.exists())


class TestMemoryProfiler(unittest.TestCase):
    """Test memory profiler."""

    def setUp(self):
        """Setup test environment."""
        self.profiler = MemoryProfiler()

    def test_take_snapshot(self):
        """Test snapshot."""
        snapshot = self.profiler.take_snapshot()
        self.assertIsNotNone(snapshot)
        self.assertGreaterEqual(snapshot.timestamp, 0)

    def test_memory_health(self):
        """Test health check."""
        is_healthy, message = self.profiler.check_memory_health()
        self.assertIsInstance(is_healthy, bool)
        self.assertIsInstance(message, str)

    def test_cleanup(self):
        """Test cleanup."""
        result = self.profiler.aggressive_cleanup()
        self.assertIn("freed_mb", result)


if __name__ == "__main__":
    unittest.main()
