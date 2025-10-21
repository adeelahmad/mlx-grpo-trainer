# file_path: mlx_rl_trainer/tests/unit/test_rewards/test_base.py
# revision_no: 001
# goals_of_writing_code_block: Unit tests for base reward functionality.
# type_of_code_response: add new code
"""Tests for base reward functionality."""

import pytest
from mlx_rl_trainer.rewards.base_reward import BaseReward
from mlx_rl_trainer.rewards.context import RewardContext


class DummyReward(BaseReward):
    def compute(self, context: RewardContext) -> float:
        return 1.0


def test_base_reward_initialization():
    """Test reward initialization with config."""
    config = {"weight": 0.5}
    reward = DummyReward(config)
    assert reward.weight == 0.5


def test_batch_compute():
    """Test batch computation."""
    reward = DummyReward({"weight": 1.0})
    contexts = [
        RewardContext(
            generated_text="gen1", prompt_text="", reference_completion="ref1"
        ),
        RewardContext(
            generated_text="gen2", prompt_text="", reference_completion="ref2"
        ),
    ]
    results = reward.batch_compute(contexts)
    assert len(results) == 2
    assert all(r == 1.0 for r in results)
