# file_path: mlx_rl_trainer/src/mlx_rl_trainer/algorithms/base_algorithm.py
# revision_no: 001
# goals_of_writing_code_block: Abstract base class for all RL algorithms.
# type_of_code_response: add new code
"""
Abstract base class for all Reinforcement Learning algorithms.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import mlx.core as mx
import mlx.nn as nn
import logging

from mlx_rl_trainer.core.config import (
    ExperimentConfig,
)  # Use ExperimentConfig for config

logger = logging.getLogger(__name__)


class BaseAlgorithm(ABC):
    """
    Abstract base class for Reinforcement Learning algorithms like GRPO, PPO.
    Defines the core interface for loss calculation and advantage estimation.
    """

    def __init__(
        self, config: ExperimentConfig, actor_model: nn.Module, ref_model: nn.Module
    ):
        """
        Initializes the base algorithm.

        Args:
            config: The full experiment configuration.
            actor_model: The policy model being trained.
            ref_model: The reference (or old) policy model.
        """
        self.config = config
        self.actor = actor_model
        self.ref = ref_model
        logger.debug(f"BaseAlgorithm initialized for {self.__class__.__name__}.")

    @abstractmethod
    def calculate_loss_and_grads(
        self, rollout_batch: Dict[str, mx.array], full_config: ExperimentConfig
    ) -> Tuple[mx.array, Dict[str, Any], Dict[str, float]]:
        """
        Calculates the algorithm-specific loss and gradients for the actor model.

        Args:
            rollout_batch: A dictionary containing data from generated rollouts.
            full_config: The complete experiment configuration (to access algorithm-specific params).

        Returns:
            A tuple: (scalar_loss_tensor, gradients_dict, additional_metrics_dict).
        """
        raise NotImplementedError

    @abstractmethod
    def compute_advantages(
        self, rewards_flat: mx.array, samples_per_prompt: int
    ) -> mx.array:
        """
        Computes advantage estimates from a flat array of rewards.

        Args:
            rewards_flat: A 1D MLX array of rewards for all generated samples.
            samples_per_prompt: The number of responses generated for each unique prompt.

        Returns:
            A 1D MLX array of advantage estimates.
        """
        raise NotImplementedError
