# file_path: mlx_rl_trainer/docs/adding_rewards.md
# revision_no: 001
# goals_of_writing_code_block: Documentation for adding custom reward functions.
# type_of_code_response: add new code
# Adding Custom Reward Functions

This document outlines the process for adding new, custom reward functions to the MLX RL Trainer framework. The system is designed to be extensible, allowing you to plug in new reward logic without modifying core trainer code.

## Quick Start

1.  **Create a New File**: Place your new reward function in an appropriate subdirectory under `src/mlx_rl_trainer/rewards/` (e.g., `rewards/custom/my_reward.py`).
2.  **Inherit from `BaseReward`**: Your reward class must inherit from `mlx_rl_trainer.rewards.base_reward.BaseReward`.
3.  **Implement `compute(self, context: RewardContext)`**: This is the core method where your reward logic resides. It should return a single `float` score.
4.  **Implement `batch_compute(self, contexts: List[RewardContext])` (Optional but Recommended)**: For efficiency, override this method if you can process multiple `RewardContext` objects in a vectorized or batch-optimized manner.
5.  **Register Your Reward**: Decorate your class with `@RewardRegistry.register("your_reward_name")`.
6.  **Import Your Reward**: Ensure your new reward module is imported in `src/mlx_rl_trainer/rewards/__init__.py` so the `RewardRegistry` can discover it.
7.  **Configure in YAML**: Add your reward to the `rewards` section of your experiment's YAML configuration file.

## Example: Custom Reward

Let's say you want to create a reward that penalizes responses for being too long.

```python
# src/mlx_rl_trainer/rewards/custom/response_length_penalty.py

import logging
from typing import Dict, Any
import numpy as np

from mlx_rl_trainer.rewards.base_reward import BaseReward
from mlx_rl_trainer.rewards.registry import RewardRegistry
from mlx_rl_trainer.rewards.context import RewardContext
from mlx_rl_trainer.utils.text_utils import _count_words # Assuming this utility exists

logger = logging.getLogger(__name__)

@RewardRegistry.register("response_length_penalty")
class ResponseLengthPenalty(BaseReward):
    """
    Penalizes responses that exceed a maximum word count.

    Configuration:
        max_words: Maximum allowed words before penalty starts (default: 150)
        penalty_per_word: Penalty applied for each word over `max_words` (default: 0.005)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.max_words = config.get("max_words", 150)
        self.penalty_per_word = config.get("penalty_per_word", 0.005)
        logger.info(f"Initialized ResponseLengthPenalty with max_words: {self.max_words}")

    def compute(self, context: RewardContext) -> float:
        """
        Compute length penalty reward.

        Returns:
            Reward score (1.0 for under max_words, linearly decreasing for over).
        """
        try:
            self.validate_inputs(context)

            word_count = _count_words(context.generated_text)

            if word_count <= self.max_words:
                return 1.0 # No penalty
            else:
                excess_words = word_count - self.max_words
                penalty = excess_words * self.penalty_per_word
                # Reward goes from 1.0 down to 0.0 (or even negative if very long)
                return float(max(0.0, 1.0 - penalty))

        except Exception as e:
            logger.error(f"Response length penalty computation failed: {e}")
            return 0.0
        finally:
            pass

    # Optional: Implement batch_compute for efficiency
    def batch_compute(self, contexts: List[RewardContext]) -> List[float]:
        word_counts = [_count_words(c.generated_text) for c in contexts]
        rewards = []
        for wc in word_counts:
            if wc <= self.max_words:
                rewards.append(1.0)
            else:
                penalty = (wc - self.max_words) * self.penalty_per_word
                rewards.append(float(max(0.0, 1.0 - penalty)))
        return rewards

```

## Configuration

To use this new reward, add it to your experiment's YAML configuration:

```yaml
# configs/experiments/your_experiment.yaml
rewards:
  - name: format_structure
    weight: 0.1
    # ... other configs ...
  - name: content_similarity
    weight: 0.6
    # ... other configs ...
  - name: response_length_penalty # Your new reward!
    weight: 0.3
    config:
      max_words: 200
      penalty_per_word: 0.003
```

## Best Practices for Custom Rewards

-   **Return Values**: Always return a `float` typically between `0.0` and `1.0`. Higher values should indicate a "better" response according to your criteria.
-   **Error Handling**: Wrap your `compute` logic in `try-except` blocks and return `0.0` (or a small negative value for severe issues) on failure. Log errors for debugging.
-   **Efficiency**: If `compute` is slow, consider implementing an optimized `batch_compute` method.
-   **Inputs Validation**: Use `self.validate_inputs(context)` at the start of your `compute` method to ensure `RewardContext` is as expected.
-   **Modularity**: Keep your reward function focused on a single aspect of quality. Combine multiple simple rewards with `RewardComposer`.
-   **Testing**: Write unit tests for your custom reward function to ensure it behaves as expected across various inputs and edge cases.

## Testing

You can add a unit test for your new reward in `tests/unit/test_rewards/test_response_length_penalty.py`:

```python
# tests/unit/test_rewards/test_response_length_penalty.py

import pytest
from datasets import Dataset
from mlx_rl_trainer.rewards.custom.response_length_penalty import ResponseLengthPenalty
from mlx_rl_trainer.rewards.context import RewardContext

class TestResponseLengthPenalty:
    """Test suite for ResponseLengthPenalty."""

    @pytest.fixture
    def reward_fn(self):
        config = {
            "max_words": 10,
            "penalty_per_word": 0.1
        }
        return ResponseLengthPenalty(config)

    def test_within_limit(self, reward_fn):
        """Test response within max_words limit."""
        generated = "This is a short test response." # 6 words
        context = RewardContext(generated_text=generated, prompt_text="", reference_completion="")
        score = reward_fn.compute(context)
        assert score == 1.0

    def test_over_limit(self, reward_fn):
        """Test response exceeding max_words limit."""
        generated = "This is a much longer test response that goes over the limit." # 12 words
        context = RewardContext(generated_text=generated, prompt_text="", reference_completion="")
        score = reward_fn.compute(context)
        # Expected: 1.0 - (2 * 0.1) = 0.8
        assert score == pytest.approx(0.8)

    def test_empty_text(self, reward_fn):
        """Test empty generated text."""
        generated = ""
        context = RewardContext(generated_text=generated, prompt_text="", reference_completion="")
        score = reward_fn.compute(context)
        assert score == 1.0 # No words, within limit (or handle as 0.0 depending on desired behavior)

    def test_just_at_limit(self, reward_fn):
        """Test response exactly at max_words limit."""
        generated = "One two three four five six seven eight nine ten." # 10 words
        context = RewardContext(generated_text=generated, prompt_text="", reference_completion="")
        score = reward_fn.compute(context)
        assert score == 1.0
```
