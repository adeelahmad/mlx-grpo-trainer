"""
MLX RL Trainer - Production-ready reinforcement learning framework
"""
__version__ = "0.1.0"

from mlx_rl_trainer.core.trainer import BaseTrainer, TrainingMetrics, EvaluationMetrics
from mlx_rl_trainer.core.exceptions import (
    CustomBaseException,
    ModelLoadError,
    DataLoadError,
    CheckpointError,
    InvalidConfigurationError,
    TrainingRuntimeError,
)
from mlx_rl_trainer.core.config import (
    ExperimentConfig,
    RewardConfig,
    EvaluatorConfig,
    DataConfig,
    ModelConfig,
    TrainerParams,
    GenerationConfig,
    CheckpointConfig,
    MonitoringConfig,
)
from mlx_rl_trainer.core.model_manager import ModelManager
from mlx_rl_trainer.core.checkpoint_manager import CheckpointManager
from mlx_rl_trainer.rewards.registry import RewardRegistry, register_reward
from mlx_rl_trainer.rewards.base_reward import RewardComposer
from mlx_rl_trainer.rewards.context import RewardContext
from mlx_rl_trainer.evaluation.registry import EvaluatorRegistry

__all__ = [
    "BaseTrainer",
    "TrainingMetrics",
    "EvaluationMetrics",
    "CustomBaseException",
    "ModelLoadError",
    "DataLoadError",
    "CheckpointError",
    "InvalidConfigurationError",
    "TrainingRuntimeError",
    "ExperimentConfig",
    "RewardConfig",
    "EvaluatorConfig",
    "DataConfig",
    "ModelConfig",
    "TrainerParams",
    "GenerationConfig",
    "CheckpointConfig",
    "MonitoringConfig",
    "ModelManager",
    "CheckpointManager",
    "RewardRegistry",
    "register_reward",
    "RewardComposer",
    "RewardContext",
    "EvaluatorRegistry",
]
