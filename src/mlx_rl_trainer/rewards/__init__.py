# file_path: mlx_rl_trainer/src/mlx_rl_trainer/rewards/__init__.py
# revision_no: 001
# goals_of_writing_code_block: Top-level __init__.py for the rewards module.
# type_of_code_response: add new code
"""Reward function system"""
from .base_reward import BaseReward, RewardComposer
from .registry import RewardRegistry, register_reward
from .context import RewardContext

# Import concrete reward implementations to ensure they are registered
import mlx_rl_trainer.rewards.format.tag_structure
import mlx_rl_trainer.rewards.content.semantic_similarity
import mlx_rl_trainer.rewards.programming.code_execution
import mlx_rl_trainer.rewards.reasoning.thinking_quality
import mlx_rl_trainer.rewards.content.mcq_accuracy
import mlx_rl_trainer.rewards.content.steps_coverage
import mlx_rl_trainer.rewards.content.answer_quality


__all__ = [
    "BaseReward",
    "RewardComposer",
    "RewardRegistry",
    "register_reward",
    "RewardContext",
]
