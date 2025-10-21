# file_path: mlx_rl_trainer/src/mlx_rl_trainer/rewards/content/mcq_accuracy.py
# revision_no: 002
# goals_of_writing_code_block: Update reward function to import `_extract_predicted_letters` from the central text_utils module.
# type_of_code_response: change existing
"""Reward function for Multiple Choice Question (MCQ) accuracy."""

import re
from typing import Dict, Any, List, Optional
import logging

from mlx_rl_trainer.rewards.base_reward import BaseReward
from mlx_rl_trainer.rewards.registry import RewardRegistry
from mlx_rl_trainer.rewards.context import RewardContext
from mlx_rl_trainer.core.config import GenerationConfig
from mlx_rl_trainer.utils.mcq_utils import _extract_predicted_letters
from mlx_rl_trainer.utils.text_utils import _letters_to_canonical

logger = logging.getLogger(__name__)


@RewardRegistry.register("mcq_accuracy")
class MCQAccuracyReward(BaseReward):
    """
    Rewards accuracy on Multiple Choice Questions.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.strict_matching = config.get("strict_matching", True)

    def compute(self, context: RewardContext) -> float:
        """
        Compute MCQ accuracy reward.
        """
        self.validate_inputs(context)

        metadata = context.metadata
        if not metadata or not metadata.get("is_mcq"):
            return 0.0

        correct_letters = _letters_to_canonical(metadata.get("correct_letters", ""))
        if not correct_letters:
            return 0.0

        gold_set = set(correct_letters.split(","))

        # Dynamically create GenerationConfig for extraction utility
        gen_config = GenerationConfig()

        predicted_letters = _extract_predicted_letters(
            context.generated_text,
            metadata.get("mcq_options"),
            gen_config,
        )
        predicted_set = (
            set(predicted_letters.split(",")) if predicted_letters else set()
        )

        if not predicted_set:
            return 0.0

        # Strict matching: sets must be identical
        if self.strict_matching:
            return 1.0 if predicted_set == gold_set else 0.0

        # Partial credit (Jaccard similarity) for non-strict multi-select
        intersection = len(gold_set.intersection(predicted_set))
        union = len(gold_set.union(predicted_set))

        return float(intersection / union) if union > 0 else 0.0
