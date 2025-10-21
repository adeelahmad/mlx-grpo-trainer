# /src/mlx_rl_trainer/monitoring/trainer_integration.py
# Revision: 001
# Goal: Integration helpers for trainer with enhanced monitoring
# Type: New Code
# Description: Helper functions to integrate monitoring into existing trainer

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from mlx_rl_trainer.monitoring.stats_collector import (
    ComprehensiveStatsCollector,
    TrainingStats,
)
from mlx_rl_trainer.monitoring.chart_generator import ChartGenerator
from mlx_rl_trainer.monitoring.wandb_logger import EnhancedWandBLogger
from mlx_rl_trainer.utils.memory_profiler import MemoryProfiler
from mlx_rl_trainer.monitoring.visualization_config import VisualizationConfig
from mlx_rl_trainer.core.config import ExperimentConfig

logger = logging.getLogger(__name__)


class MonitoringManager:
    """
    Centralized monitoring manager.

    Coordinates:
    - Stats collection
    - Chart generation
    - WandB logging
    - Memory profiling
    """

    def __init__(
        self,
        config: ExperimentConfig,
        run_id: str,
        viz_config: Optional[VisualizationConfig] = None,
    ):
        """Initialize monitoring manager."""
        self.config = config
        self.run_id = run_id
        self.viz_config = viz_config or VisualizationConfig()

        # Initialize components
        self.stats_collector = ComprehensiveStatsCollector(
            output_dir=config.trainer.output_dir / "stats"
        )

        self.chart_generator = ChartGenerator(
            output_dir=config.trainer.output_dir / "charts",
            style=self.viz_config.style,
            palette=self.viz_config.palette,
            dpi=self.viz_config.dpi,
        )

        self.memory_profiler = MemoryProfiler()

        # WandB logger (optional)
        self.wandb_logger = None
        if config.monitoring.use_wandb:
            try:
                self.wandb_logger = EnhancedWandBLogger(
                    project=config.monitoring.wandb_project,
                    entity=config.monitoring.wandb_entity,
                    name=config.monitoring.wandb_run_name or run_id,
                    config=config.model_dump(),
                )
            except Exception as e:
                logger.error(f"Failed to initialize WandB: {e}")

        logger.info("Initialized MonitoringManager")

    def log_training_step(
        self,
        step: int,
        loss: float,
        reward: float,
        grad_norm: float,
        learning_rate: float,
        **kwargs,
    ) -> None:
        """Log training step metrics."""
        # Create training stats
        stats = TrainingStats(
            total_loss=loss,
            reward_mean=reward,
            grad_norm=grad_norm,
            learning_rate=learning_rate,
        )
        stats.update_from_dict(kwargs)

        # Record to stats collector
        self.stats_collector.record_step_stats(stats, step)

        # Log to WandB
        if self.wandb_logger:
            self.wandb_logger.log_training_metrics(
                loss=loss,
                reward=reward,
                grad_norm=grad_norm,
                learning_rate=learning_rate,
                step=step,
                **kwargs,
            )

    def generate_charts(self, step: int) -> None:
        """Generate all charts."""
        try:
            # Get data from stats collector
            trends = self.stats_collector.get_recent_trends(n=1000)

            # Generate various charts
            if trends:
                # Training curves
                self.chart_generator.plot_training_curves(
                    data=trends, filename=f"training_curves_step_{step}.png"
                )

                # Dashboard
                self.chart_generator.create_dashboard(
                    stats_data=trends, filename=f"dashboard_step_{step}.png"
                )

                # Memory usage
                memory_data = {k: v for k, v in trends.items() if "memory" in k.lower()}
                if memory_data:
                    self.chart_generator.plot_memory_usage(
                        memory_data=memory_data, filename=f"memory_step_{step}.png"
                    )

                # Reward distribution
                if "step/reward_mean" in trends:
                    self.chart_generator.plot_reward_distribution(
                        rewards=trends["step/reward_mean"],
                        filename=f"reward_dist_step_{step}.png",
                    )

            logger.info(f"Generated charts at step {step}")

        except Exception as e:
            logger.error(f"Failed to generate charts: {e}", exc_info=True)

    def check_memory_health(self) -> bool:
        """Check memory health."""
        is_healthy, message = self.memory_profiler.check_memory_health()

        if not is_healthy:
            logger.warning(f"Memory health check: {message}")

            # Perform cleanup
            self.memory_profiler.aggressive_cleanup()

            # Alert via WandB
            if self.wandb_logger:
                self.wandb_logger.alert(
                    title="Memory Warning", text=message, level="WARN"
                )

        return is_healthy

    def export_all(self) -> None:
        """Export all monitoring data."""
        # Export stats
        self.stats_collector.export_to_json()

        # Export summary
        summary = self.stats_collector.get_summary_report()
        summary_path = self.config.trainer.output_dir / "monitoring_summary.json"

        import json

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("Exported all monitoring data")

    def cleanup(self) -> None:
        """Cleanup monitoring resources."""
        if self.wandb_logger:
            self.wandb_logger.finish()

        self.chart_generator.cleanup()
        self.memory_profiler.aggressive_cleanup()


# Dependencies: All monitoring modules
# Install: See individual module requirements
# Usage: monitor = MonitoringManager(config, run_id)
#        monitor.log_training_step(step=1, loss=0.5, ...)
# Status: Complete and commit-ready
