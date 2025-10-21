from typing import Dict, Any, Set
from mlx_rl_trainer.rewards.base_reward import BaseReward
from mlx_rl_trainer.rewards.registry import RewardRegistry
from mlx_rl_trainer.rewards.context import RewardContext
from mlx_rl_trainer.utils.text_utils import _extract_action_phrases
import logging

logger = logging.getLogger(__name__)


@RewardRegistry.register("steps_coverage")
class StepsCoverageReward(BaseReward):
    """
    Calculates a reward based on how many of the required steps are covered in the generated text.
    Uses Jaccard similarity to measure overlap between required and extracted steps.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Debug logging flag
        self.debug_logging = config.get("debug_logging", True)

        logger.info("StepsCoverageReward initialized with Jaccard similarity metric")

    def compute(self, context: RewardContext) -> float:
        """
        Computes the steps coverage reward.
        The reward is the Jaccard similarity between the set of required steps and the set of extracted steps.

        Formula: intersection(required, extracted) / union(required, extracted)

        Returns:
            float: Score between 0.0 and 1.0
        """
        required_steps = context.metadata.get("required_steps")

        # Handle missing or invalid required steps
        if not required_steps or not isinstance(required_steps, list):
            if self.debug_logging:
                logger.warning(
                    f"StepsCoverage: No required_steps in metadata or invalid type "
                    f"(type={type(required_steps).__name__})"
                )
            return 0.0

        generated_text = context.generated_text
        if not generated_text:
            if self.debug_logging:
                logger.warning("StepsCoverage: Empty generated text")
            return 0.0

        # Extract steps from generated text
        extracted_steps = _extract_action_phrases(generated_text)

        if not extracted_steps:
            if self.debug_logging:
                logger.warning(
                    f"StepsCoverage: No steps extracted from generated text. "
                    f"Required steps: {required_steps}"
                )
            return 0.0

        # Normalize to lowercase for comparison
        required_steps_set = set(step.lower().strip() for step in required_steps)
        extracted_steps_set = set(step.lower().strip() for step in extracted_steps)

        # Calculate Jaccard similarity
        intersection = required_steps_set.intersection(extracted_steps_set)
        union = required_steps_set.union(extracted_steps_set)

        intersection_count = len(intersection)
        union_count = len(union)

        # Edge case: empty union (shouldn't happen, but handle gracefully)
        if union_count == 0:
            score = 1.0 if not required_steps else 0.0
            if self.debug_logging:
                logger.warning(f"StepsCoverage: Empty union, returning {score}")
            return score

        # Calculate Jaccard score
        score = intersection_count / union_count

        # Calculate coverage percentage (how many required steps were found)
        coverage_percent = (
            (intersection_count / len(required_steps_set)) * 100
            if required_steps_set
            else 0.0
        )

        # ALWAYS log detailed breakdown
        if self.debug_logging:
            logger.info(
                f"StepsCoverage | "
                f"required={len(required_steps_set)}, "
                f"extracted={len(extracted_steps_set)}, "
                f"intersection={intersection_count}, "
                f"union={union_count}, "
                f"coverage={coverage_percent:.1f}%, "
                f"jaccard={score:.3f}"
            )

            # Log the actual sets for inspection
            logger.debug(f"Required steps: {sorted(required_steps_set)}")
            logger.debug(f"Extracted steps: {sorted(extracted_steps_set)}")
            logger.debug(f"Matched steps: {sorted(intersection)}")

            # Log missing steps
            missing_steps = required_steps_set - extracted_steps_set
            if missing_steps:
                logger.debug(f"Missing steps: {sorted(missing_steps)}")

            # Log extra steps (not required but generated)
            extra_steps = extracted_steps_set - required_steps_set
            if extra_steps:
                logger.debug(f"Extra steps (not required): {sorted(extra_steps)}")

        return score
