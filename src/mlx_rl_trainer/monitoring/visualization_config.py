# /src/mlx_rl_trainer/monitoring/visualization_config.py
# Revision: 001
# Goal: Configuration for visualization system
# Type: New Code
# Description: Centralized configuration for charts and plots

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from pathlib import Path


@dataclass
class ChartConfig:
    """Configuration for a single chart type."""

    enabled: bool = True
    filename: str = ""
    title: str = ""
    figsize: Tuple[int, int] = (12, 8)
    dpi: int = 150
    style: str = "darkgrid"
    palette: str = "husl"
    smooth: bool = True
    smoothing_window: int = 10


@dataclass
class VisualizationConfig:
    """Complete visualization configuration."""

    # Output settings
    output_dir: Path = Path("./charts")
    save_format: str = "png"
    dpi: int = 150

    # Style settings
    style: str = "darkgrid"
    palette: str = "husl"
    context: str = "notebook"
    font_scale: float = 1.2

    # Chart generation frequency
    generate_every_n_steps: int = 100
    generate_at_checkpoints: bool = True

    # Individual chart configs
    training_curves: ChartConfig = field(
        default_factory=lambda: ChartConfig(
            filename="training_curves.png", title="Training Curves"
        )
    )

    reward_distribution: ChartConfig = field(
        default_factory=lambda: ChartConfig(
            filename="reward_distribution.png", title="Reward Distribution"
        )
    )

    memory_usage: ChartConfig = field(
        default_factory=lambda: ChartConfig(
            filename="memory_usage.png", title="Memory Usage"
        )
    )

    gradient_flow: ChartConfig = field(
        default_factory=lambda: ChartConfig(
            filename="gradient_flow.png", title="Gradient Flow"
        )
    )

    token_distribution: ChartConfig = field(
        default_factory=lambda: ChartConfig(
            filename="token_distribution.png", title="Token Distribution"
        )
    )

    correlation_matrix: ChartConfig = field(
        default_factory=lambda: ChartConfig(
            filename="correlation_matrix.png",
            title="Metric Correlations",
            figsize=(12, 10),
        )
    )

    dashboard: ChartConfig = field(
        default_factory=lambda: ChartConfig(
            filename="training_dashboard.png",
            title="Training Dashboard",
            figsize=(20, 12),
        )
    )

    # Metrics to track
    primary_metrics: List[str] = field(
        default_factory=lambda: [
            "loss/total",
            "reward/mean",
            "gradient/norm",
            "learning/rate",
        ]
    )

    secondary_metrics: List[str] = field(
        default_factory=lambda: [
            "memory/allocated_mb",
            "tokens/thinking",
            "tokens/answer",
            "kl_divergence",
        ]
    )

    reward_components: List[str] = field(
        default_factory=lambda: [
            "format_reward",
            "content_reward",
            "thinking_reward",
            "mcq_accuracy",
        ]
    )


# Default configuration
DEFAULT_VIZ_CONFIG = VisualizationConfig()


# Dependencies: None (dataclass only)
# Install: N/A (standard library)
# Usage: config = VisualizationConfig(output_dir=Path("./my_charts"))
# Status: Complete and commit-ready
