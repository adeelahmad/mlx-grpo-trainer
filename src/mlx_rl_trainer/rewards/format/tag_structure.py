"""
Tag Structure Reward Module

This module provides the TagStructureReward class, which evaluates the structural
correctness of generated text with respect to thinking and answer tags. It rewards
proper tag usage, balanced tags, and appropriate content length in both thinking
and answer sections.

The reward function encourages:
1. Proper use of thinking tags (<think>...</think>)
2. Appropriate thinking section length (not too short, not too verbose)
3. Presence of an answer section after the thinking section
4. Balanced tag structure (each opening tag has a corresponding closing tag)

This module uses the centralized configuration provider and shared tag utilities
to ensure consistent behavior across the reward system.
"""

import logging
import re
from typing import Dict, Any, List, Optional

from mlx_rl_trainer.rewards.base_reward import BaseReward
from mlx_rl_trainer.rewards.registry import RewardRegistry
from mlx_rl_trainer.rewards.context import RewardContext
from mlx_rl_trainer.rewards.config_provider import RewardConfigurable, get_generation_config
from mlx_rl_trainer.utils.tag_utils import (
    extract_think_region,
    extract_answer_region,
    count_tag_occurrences,
    validate_tag_structure,
    calculate_verbosity_penalty
)

logger = logging.getLogger(__name__)


@RewardRegistry.register("format_structure")
class TagStructureReward(BaseReward, RewardConfigurable):
    """
    Rewards the model for adhering to <think>...</think> structure
    followed by direct answer text (no answer tags).

    Encourages concise, compressed thinking by penalizing verbosity.
    
    This reward function evaluates:
    1. Presence of properly balanced thinking tags
    2. Appropriate length of thinking content
    3. Presence of answer content after thinking section
    4. Overall structure correctness
    
    The reward is highest when:
    - Exactly one pair of thinking tags is present
    - Thinking content length is within target range
    - Answer content is present after thinking section
    - No malformed or nested tags
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the TagStructureReward with configuration.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        # Initialize both parent classes
        BaseReward.__init__(self, config)
        RewardConfigurable.__init__(self)
        
        # Get generation config from centralized provider
        gen_config = self.get_generation_config()
        
        # Basic parameters with defaults
        self.min_think_length = config.get("min_think_length", 20)
        self.min_answer_length = config.get("min_answer_length", 15)
        
        # Optimal think length range (from centralized config)
        self.think_target_min = config.get(
            "think_length_target_min",
            getattr(gen_config, "think_length_target_min", 100)
        )
        self.think_target_max = config.get(
            "think_length_target_max",
            getattr(gen_config, "think_length_target_max", 250)
        )
        
        # Penalty strength for length deviation
        self.length_penalty_strength = config.get(
            "length_penalty_strength",
            getattr(gen_config, "think_length_penalty_strength", 0.5)
        )
        
        # Verbosity penalty multiplier (how much to penalize excessive length)
        self.verbosity_penalty_factor = config.get("verbosity_penalty_factor", 2.0)
        
        # Debug logging flag
        self.debug_logging = config.get("debug_logging", True)
        
        # Store tag values for reuse
        self.start_tag = getattr(gen_config, "think_start_tag", "<think>")
        self.end_tag = getattr(gen_config, "think_end_tag", "</think>")
        
        if self.debug_logging:
            logger.debug(f"TagStructureReward initialized with targets: "
                        f"min={self.think_target_min}, max={self.think_target_max}, "
                        f"penalty_strength={self.length_penalty_strength}")

    def _compute_length_score(self, think_length: int) -> float:
        """
        Compute a score based on think length relative to target range.

        This method implements a sophisticated scoring algorithm that rewards thinking
        sections within the target length range and penalizes those that are too short
        or too long, with a stronger penalty for verbosity.
        
        Scoring philosophy:
        - Optimal range (target_min to target_max): 1.0 (perfect score)
        - Too short (below target_min): Gradually decreasing score (linear penalty)
        - Too long (above target_max): More aggressive penalty (exponential decay)
        
        The asymmetric scoring reflects the principle that verbosity is generally
        worse than brevity in thinking sections, as it indicates inefficient reasoning.

        Args:
            think_length: Character count of thinking section

        Returns:
            Score multiplier between 0.0 and 1.0
            
        Examples:
            >>> reward = TagStructureReward({'weight': 1.0})
            >>> reward.think_target_min, reward.think_target_max = 100, 200
            >>> reward._compute_length_score(150)  # Within target range
            1.0
            >>> reward._compute_length_score(50)   # Too short
            0.25
            >>> reward._compute_length_score(300)  # Too long
            0.5
        """
        try:
            # Handle edge case of invalid inputs
            if think_length < 0:
                logger.warning("Negative think length provided, treating as zero")
                return 0.0
                
            # Perfect length - in the sweet spot
            if self.think_target_min <= think_length <= self.think_target_max:
                return 1.0

            # Too short - linear penalty
            if think_length < self.think_target_min:
                if think_length < self.min_think_length:
                    return 0.0  # Way too short

                # Scale from min_think_length to target_min
                range_diff = self.think_target_min - self.min_think_length
                if range_diff <= 0:
                    return 0.5  # Neutral if min/target are equal or inverted

                ratio = (think_length - self.min_think_length) / range_diff
                return 0.5 + (0.5 * max(0.0, min(1.0, ratio)))  # Range: 0.5 to 1.0

            # Too long - exponential penalty (verbosity is bad!)
            excess = think_length - self.think_target_max
            penalty_range = self.think_target_max if self.think_target_max > 0 else 100

            normalized_excess = excess / penalty_range
            penalty = (
                normalized_excess
                * self.length_penalty_strength
                * self.verbosity_penalty_factor
            )

            # Exponential decay for severe verbosity
            score = max(0.0, 1.0 - penalty)
            return score
            
        except Exception as e:
            logger.error(f"Error computing length score: {e}")
            return 0.5  # Return neutral score on error

    def compute(self, context: RewardContext) -> Dict[str, Any]:
        """
        Computes the format structure reward and returns a dictionary.
        
        This method analyzes the structure of the generated text, checking for proper thinking tags
        and content length in both thinking and answer sections. It uses the shared tag utilities
        to ensure consistent behavior across the reward system.
        
        The reward is calculated based on:
        1. Presence and balance of thinking tags
        2. Length of thinking content relative to target range
        3. Presence of answer content after thinking section
        
        Args:
            context: The RewardContext containing the generated text to evaluate
            
        Returns:
            A dictionary with at least a 'reward' key containing a score between 0.0 and 1.0,
            and a 'log' key with detailed information about the evaluation
        """
        try:
            # Validate input context
            if not context or not hasattr(context, 'generated_text'):
                return {"reward": 0.0, "log": {"error": "Invalid context object"}}
                
            generated = context.generated_text or ""
            log_data = {}
            
            # Handle empty or very short text
            if not generated or len(generated.strip()) < 10:
                return {"reward": 0.0, "log": {"error": "Empty or too short generation"}}

            # Get generation config from centralized provider
            gen_config = self.get_generation_config()
            
            # Use shared utilities for tag validation and extraction
            validation_result = validate_tag_structure(generated, gen_config)
            tag_counts = count_tag_occurrences(generated, gen_config)
            
            # Extract text regions with error handling
            try:
                think_text = extract_think_region(generated, gen_config)
                answer_text = extract_answer_region(generated, gen_config)
                think_len = len(think_text.strip())
                answer_len = len(answer_text.strip())
            except Exception as e:
                logger.warning(f"Error extracting text regions: {e}")
                think_text = ""
                answer_text = ""
                think_len = 0
                answer_len = 0

            # Log data for debugging
            log_data = {
                "start_tags": tag_counts["think_start"],
                "end_tags": tag_counts["think_end"],
                "think_len": think_len,
                "answer_len": answer_len,
                "validation": validation_result
            }
            
            if self.debug_logging:
                logger.info(f"TagStructure | {log_data}")

            # Calculate score based on structure
            score = 0.0
            th_s = tag_counts["think_start"]
            th_e = tag_counts["think_end"]
            
            if th_s == 1 and th_e == 1:
                if think_len >= self.min_think_length and answer_len >= self.min_answer_length:
                    try:
                        length_score = self._compute_length_score(think_len)
                        score = 1.0 * length_score
                        log_data["reason"] = "Perfect structure"
                        log_data["length_score"] = length_score
                    except Exception as e:
                        logger.warning(f"Error computing length score: {e}")
                        score = 0.8  # Fallback to a reasonable score
                        log_data["reason"] = "Perfect structure (score calculation error)"
                elif think_len >= self.min_think_length or answer_len >= self.min_answer_length:
                    score = 0.6
                    log_data["reason"] = "Partial content"
                else:
                    score = 0.3
                    log_data["reason"] = "Empty content"
            elif th_s >= 1 and th_e == 0:
                score = 0.3
                log_data["reason"] = "Incomplete think block"
            elif th_s > 1 or th_e > 1:
                score = 0.2
                log_data["reason"] = "Multiple tags"
            elif th_s == 0 and th_e == 0:
                score = 0.1 if len(generated.strip()) > 30 else 0.0
                log_data["reason"] = "No tags"
            else:
                score = 0.2
                log_data["reason"] = "Fallback case"

            # Ensure score is in valid range
            score = max(0.0, min(1.0, score))
            log_data["final_score"] = score
            return {"reward": score, "log": log_data}
            
        except Exception as e:
            # Catch-all exception handler for robustness
            logger.error(f"TagStructureReward compute error: {e}", exc_info=True)
            return {"reward": 0.0, "log": {"error": f"Exception: {str(e)[:100]}"}}
            
    def batch_compute(self, contexts: List[RewardContext]) -> List[Dict[str, Any]]:
        """
        Computes rewards for a batch of contexts by calling compute sequentially.
        
        This method processes each context individually through the compute method,
        applies reward smoothing, and formats the output consistently. This sequential
        approach is more robust than a complex batch implementation as it isolates
        failures to individual contexts rather than failing the entire batch.
        
        Args:
            contexts: List of RewardContext objects to evaluate
            
        Returns:
            List of dictionaries containing reward scores and logs, with each dictionary
            having at least the keys: self.name, "total", and "log"
            
        Note:
            Each result dictionary includes both the reward under the reward name key
            and under the "total" key for compatibility with the reward composition system.
        """
        results = []
        for i, context in enumerate(contexts):
            try:
                # Get the raw result from compute
                result_dict = self.compute(context)
                
                # Extract the reward value
                raw_score = result_dict.get('reward', 0.0)
                
                # Apply smoothing
                smoothed_score = self._smooth_reward(raw_score)
                
                # Create properly formatted output with total key
                output = {
                    self.name: smoothed_score,
                    "total": smoothed_score,  # Add the total key
                    "log": result_dict.get('log', {})
                }
                
                results.append(output)
                
                # Log progress for large batches
                if self.debug_logging and len(contexts) > 10 and i % 10 == 0:
                    logger.debug(f"TagStructureReward: Processed {i+1}/{len(contexts)} contexts")
                    
            except Exception as e:
                logger.error(f"Exception in batch_compute for context {i}: {e}")
                results.append({
                    self.name: 0.0,
                    "total": 0.0,  # Include total key in error case too
                    "log": {"error": f"batch_exception: {str(e)[:100]}"}
                })
        
        return results