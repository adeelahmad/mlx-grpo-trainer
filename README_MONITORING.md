# MLX RL Trainer - Enhanced Monitoring System

## Overview

This enhanced monitoring system provides comprehensive statistics collection, advanced visualizations, memory profiling, and WandB integration for the MLX RL Trainer.

## Features

### 1. Comprehensive Stats Collection
- **Real-time metrics tracking**: Loss, reward, gradients, memory, tokens
- **Historical data storage**: Configurable history length with automatic cleanup
- **Statistical analysis**: Mean, std, min, max, percentiles, moving averages
- **Correlation analysis**: Automatic correlation matrix generation
- **Export formats**: CSV, JSON, pandas DataFrame

### 2. Advanced Visualization
- **Training curves**: Multi-metric time series plots with smoothing
- **Distribution plots**: Reward distributions, token distributions with KDE
- **Correlation matrices**: Seaborn heatmaps showing metric relationships
- **Gradient flow**: Layer-wise gradient norm tracking
- **Memory usage**: Real-time memory tracking with threshold alerts
- **Comprehensive dashboards**: Multi-panel overview of training progress

### 3. Memory Profiling
- **MLX memory tracking**: Active, cached, and peak memory
- **System memory monitoring**: RSS, available memory
- **Leak detection**: Automatic detection of memory leaks
- **Health checks**: Threshold-based alerts
- **Automatic cleanup**: Aggressive garbage collection triggers

### 4. Enhanced WandB Integration
- **Custom charts**: Organized metric grouping
- **Sample tables**: Log generations with prompts and references
- **Distribution logging**: Automatic histogram generation
- **Artifact logging**: Checkpoint and model artifact tracking
- **Real-time alerts**: Memory and performance alerts
- **Gradient tracking**: Layer-wise gradient flow

## Installation

`\`\`\bash
# Install required dependencies
pip install numpy pandas matplotlib seaborn wandb psutil mlx

# Or install all at once
pip install -r requirements.txt
`\`\`\

## Configuration

### Monitoring Configuration in YAML

`\`\`\yaml
monitoring:
  # Enable WandB logging
  use_wandb: true
  wandb_project: "mlx-rl-trainer"
  wandb_entity: "your-username"
  wandb_run_name: "experiment-001"

  # Logging frequency
  log_samples_every: 5
  max_logged_samples: 20

  # Chart generation
  generate_charts_every: 100
  generate_at_checkpoints: true

# Trainer configuration
trainer:
  # Memory safety
  memory_safety_threshold_mb: 8000

  # Statistics
  log_memory_usage: true
  reward_smoothing_window: 10
`\`\`\

### Visualization Configuration

`\`\`\python
from mlx_rl_trainer.monitoring.visualization_config import VisualizationConfig

viz_config = VisualizationConfig(
    output_dir=Path("./charts"),
    style="darkgrid",
    palette="husl",
    dpi=150,
    generate_every_n_steps=100
)
`\`\`\

## Usage

### Basic Usage

`\`\`\python
from mlx_rl_trainer.monitoring.trainer_integration import MonitoringManager
from mlx_rl_trainer.core.config import ExperimentConfig

# Load config
config = ExperimentConfig.load_from_yaml("config.yaml")

# Initialize monitoring
monitor = MonitoringManager(
    config=config,
    run_id="run_001"
)

# Training loop
for step in range(num_steps):
    # ... training code ...

    # Log metrics
    monitor.log_training_step(
        step=step,
        loss=loss_value,
        reward=reward_value,
        grad_norm=grad_norm_value,
        learning_rate=lr_value,
        thinking_tokens=thinking_tokens,
        answer_tokens=answer_tokens,
        memory_allocated_mb=memory_mb
    )

    # Check memory health
    if step % 10 == 0:
        monitor.check_memory_health()

    # Generate charts at checkpoints
    if step % 100 == 0:
        monitor.generate_charts(step)

# Final export
monitor.export_all()
monitor.cleanup()
`\`\`\

### Direct Stats Collection

`\`\`\python
from mlx_rl_trainer.monitoring.stats_collector import ComprehensiveStatsCollector

# Initialize
stats = ComprehensiveStatsCollector(
    output_dir=Path("./stats"),
    max_history=10000
)

# Record metrics
stats.record_metric("loss", 0.5, step=100)
stats.record_metric("reward", 0.8, step=100)

# Get statistics
loss_stats = stats.get_metric_stats("loss")
print(f"Loss mean: {loss_stats['mean']:.4f}")

# Get moving averages
averages = stats.get_moving_averages(window=10)

# Export to JSON
stats.export_to_json()
`\`\`\

### Chart Generation

`\`\`\python
from mlx_rl_trainer.monitoring.chart_generator import ChartGenerator

# Initialize
charts = ChartGenerator(
    output_dir=Path("./charts"),
    style="darkgrid",
    dpi=150
)

# Plot training curves
data = {
    'loss': [0.5, 0.4, 0.3, ...],
    'reward': [0.6, 0.7, 0.8, ...],
    'grad_norm': [0.1, 0.15, 0.12, ...]
}
charts.plot_training_curves(data, filename="curves.png")

# Plot reward distribution
rewards = [0.6, 0.7, 0.8, 0.75, 0.82, ...]
charts.plot_reward_distribution(rewards)

# Create dashboard
charts.create_dashboard(data)

# Plot correlation matrix
charts.plot_correlation_matrix(data)
`\`\`\

### Memory Profiling

`\`\`\python
from mlx_rl_trainer.utils.memory_profiler import MemoryProfiler, MemoryMonitor

# Initialize
profiler = MemoryProfiler(
    alert_threshold_mb=8000,
    leak_detection_window=50
)

# Take snapshots
snapshot = profiler.take_snapshot()
print(f"Allocated: {snapshot.mlx_allocated_mb:.1f}MB")

# Check health
is_healthy, message = profiler.check_memory_health()

# Detect leaks
leak_detected, description = profiler.detect_memory_leak()

# Cleanup
profiler.aggressive_cleanup()

# Use context manager
with MemoryMonitor(profiler, "training_step"):
    # ... code to monitor ...
    pass
`\`\`\

### WandB Integration

`\`\`\python
from mlx_rl_trainer.monitoring.wandb_logger import EnhancedWandBLogger

# Initialize
wandb_logger = EnhancedWandBLogger(
    project="my-project",
    name="experiment-001"
)

# Log metrics
wandb_logger.log_training_metrics(
    loss=0.5,
    reward=0.8,
    grad_norm=0.1,
    learning_rate=2e-6,
    step=100
)

# Log distribution
wandb_logger.log_distribution(
    name="rewards",
    values=[0.6, 0.7, 0.8, ...],
    step=100
)

# Log sample table
wandb_logger.log_sample_table(
    prompts=["prompt1", "prompt2"],
    generations=["gen1", "gen2"],
    references=["ref1", "ref2"],
    rewards=[0.8, 0.9],
    step=100
)

# Log chart
wandb_logger.log_chart("training_curves", chart_path)

# Send alert
wandb_logger.alert(
    title="High Memory Usage",
    text="Memory exceeds 8GB",
    level="WARN"
)
`\`\`\

## Output Files

### Directory Structure
`\`\`\
outputs/
├── stats/
│   ├── stats_export_*.json
│   └── metrics_history.csv
├── charts/
│   ├── training_curves_step_*.png
│   ├── dashboard_step_*.png
│   ├── reward_dist_step_*.png
│   ├── memory_step_*.png
│   └── correlation_matrix.png
├── training_metrics.csv
├── training_metrics.json
└── monitoring_summary.json
`\`\`\

### CSV Format
`\`\`\csv
step,run_id,timestamp,loss,reward_mean,grad_norm,learning_rate,...
0,run_001,1234567890.0,0.5,0.8,0.1,2e-06,...
1,run_001,1234567891.0,0.45,0.82,0.12,2e-06,...
`\`\`\

### JSON Summary Format
`\`\`\json
{
  "session": {
    "duration_hours": 2.5,
    "total_steps": 1000,
    "avg_step_time": 0.5,
    "metrics_tracked": 25
  },
  "metrics": {
    "loss": {
      "mean": 0.35,
      "std": 0.15,
      "min": 0.1,
      "max": 0.8,
      "final": 0.2
    },
    "reward_mean": {
      "mean": 0.75,
      "std": 0.1,
      "min": 0.5,
      "max": 0.95,
      "final": 0.9
    }
  }
}
`\`\`\

## Advanced Features

### Custom Metrics
`\`\`\python
# Record custom metrics
stats.record_metric("custom/my_metric", value, step)

# Log to WandB
wandb_logger.log_metrics({
    "custom/metric1": value1,
    "custom/metric2": value2
}, step=step)
`\`\`\

### Memory-Efficient Operation
- Automatic cleanup every 100 steps
- Configurable history length
- Lazy evaluation of statistics
- Efficient storage with deque

### Real-Time Monitoring
- Live WandB dashboard updates
- Automatic alert system
- Memory health checks
- Leak detection

## Troubleshooting

### High Memory Usage
`\`\`\python
# Check memory health
is_healthy, msg = monitor.check_memory_health()

# Force cleanup
monitor.memory_profiler.aggressive_cleanup()

# Reduce history size
stats_collector = ComprehensiveStatsCollector(
    max_history=1000  # Reduce from 10000
)
`\`\`\

### Missing Charts
`\`\`\python
# Verify seaborn is installed
pip install seaborn

# Check output directory permissions
chmod -R 755 outputs/charts/
`\`\`\

### WandB Connection Issues
`\`\`\bash
# Login to WandB
wandb login

# Verify credentials
wandb verify

# Check project settings
wandb project create my-project
`\`\`\

## Performance Tips

1. **Reduce logging frequency**: Increase `\log_samples_every`\
2. **Limit history**: Set `\max_history`\ to reasonable value
3. **Disable expensive operations**: Turn off correlation matrices for large metrics
4. **Use batch logging**: Accumulate metrics before logging
5. **Cleanup regularly**: Call `\cleanup()`\ methods periodically

## API Reference

See individual module documentation:
- `\stats_collector.py`\: Statistics collection
- `\chart_generator.py`\: Visualization
- `\wandb_logger.py`\: WandB integration
- `\memory_profiler.py`\: Memory profiling
- `\trainer_integration.py`\: Training integration

## License

Same as MLX RL Trainer main project.

## Support

For issues and questions:
1. Check the documentation
2. Review example configs
3. Check GitHub issues
4. Contact maintainers

---
**Status**: Production-ready
**Version**: 1.0.0
**Last Updated**: 2024-01-10
