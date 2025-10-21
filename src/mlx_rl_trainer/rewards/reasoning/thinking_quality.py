# ============================================================================
# UPDATED thinking_quality.py REWARD FILE
# ============================================================================
# Replace your existing ./reasoning/thinking_quality.py with this:

"""
Enhanced Thinking Quality Reward with Hardware-Constrained Length Penalties

Optimized for M2 MacBook with 96GB RAM and 128 token generation limit.
Prevents endless thinking rambling while maintaining reasoning quality.
"""

import re
from typing import Dict, Any, Optional
from mlx_rl_trainer.rewards.base_reward import BaseReward
from mlx_rl_trainer.rewards.registry import RewardRegistry
from mlx_rl_trainer.rewards.context import RewardContext
from mlx_rl_trainer.utils.text_utils import extract_think_region
from mlx_rl_trainer.core.config import GenerationConfig
import logging

logger = logging.getLogger(__name__)


@RewardRegistry.register("thinking_quality")
class ThinkingQualityReward(BaseReward):
    """
    Evaluates thinking section quality with adaptive length penalties.

    Features:
    - Quality scoring (tags, structure, clarity)
    - Length penalties for excessive thinking (critical for 128 token budget)
    - Conciseness bonuses for optimal length
    - Integration with trainer thinking limits
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Length targets (adjusted for 128 token budget)
        self.target_length_min = config.get("target_length_min", 30)
        self.target_length_max = config.get("target_length_max", 80)
        self.optimal_length_min = config.get("optimal_length_min", 40)
        self.optimal_length_max = config.get("optimal_length_max", 60)

        # Penalties and bonuses
        self.conciseness_bonus = config.get("conciseness_bonus", 0.15)
        self.excessive_length_threshold = config.get("excessive_length_threshold", 90)
        self.excessive_length_penalty = config.get("excessive_length_penalty", 0.5)

        # Use trainer limits if available
        self.use_trainer_limits = config.get("use_trainer_thinking_limits", False)

        # Special tokens to penalize
        self.special_tokens = config.get(
            "special_tokens",
            [
                "<|endoftext|>",
                "<|im_start|>",
                "<think><think>",
                "<|im_end|>",
                "<|end|>",
                "<|begin|>",
                "<|system|>",
                "<|user|>",
                "<|assistant|>",
                "[INST]",
                "[/INST]",
                "<s>",
                "</s>",
                "<pad>",
                "<unk>",
                "<bos>",
                "<eos>",
            ],
        )
        self.special_token_penalty = config.get("special_token_penalty", 0.4)

        # Bad phrases indicating poor reasoning
        self.bad_phrases = config.get(
            "bad_phrases",
            [
                "i think",
                "i believe",
                "maybe",
                "i'm not sure",
                "i will now",
                "i'll start by",
                "let's see",
                "confused",
                "stuck",
                "frustrated",
                "wait, wait",
                "hmm, perhaps",
                "or wait",
                "to be completely honest",
                "basically what happens",
                "long story short",
                "at the end of the day",
                "circular reasoning",
                "insufficient information",
                "too complicated",
                "for some unknown reason",
            ],
        )

        self.tag_misuse_penalty = config.get("tag_misuse_penalty", 0.3)

        # Debug logging flag
        self.debug_logging = config.get("debug_logging", True)

        logger.info(
            f"ThinkingQualityReward initialized: "
            f"target_length=[{self.target_length_min}, {self.target_length_max}], "
            f"optimal=[{self.optimal_length_min}, {self.optimal_length_max}], "
            f"excessive_threshold={self.excessive_length_threshold}"
        )

    def _check_tag_misuse_penalty(
        self, text: str, gen_config: GenerationConfig
    ) -> float:
        """Check for tag misuse (duplicate tags, nested tags)."""
        start_tag = gen_config.think_start_tag
        end_tag = gen_config.think_end_tag

        if not start_tag or not end_tag:
            return 0.0

        start_count = len(re.findall(re.escape(start_tag), text, flags=re.I))
        end_count = len(re.findall(re.escape(end_tag), text, flags=re.I))

        penalty = 0.0

        # Multiple tags or mismatched counts
        if start_count > 1 or end_count > 1 or abs(start_count - end_count) > 1:
            penalty = self.tag_misuse_penalty
            if self.debug_logging:
                logger.warning(
                    f"Tag misuse: start={start_count}, end={end_count}, penalty={penalty}"
                )

        # Nested tags within thinking region
        if start_count == 1 and end_count == 1:
            think_content = extract_think_region(text, gen_config)
            if re.search(r"<think>|<\/think>", think_content, flags=re.I):
                penalty = self.tag_misuse_penalty
                if self.debug_logging:
                    logger.warning(f"Nested tags detected, penalty={penalty}")

        return penalty

    def _check_special_tokens_penalty(self, think_content: str) -> float:
        """Penalize presence of special tokens in thinking."""
        penalty = 0.0
        found_tokens = []
        for token in self.special_tokens:
            if token in think_content:
                penalty += self.special_token_penalty
                found_tokens.append(token)

        if found_tokens and self.debug_logging:
            logger.warning(
                f"Special tokens found: {found_tokens}, total_penalty={penalty:.3f}"
            )

        return penalty

    def _compute_length_score(
        self, think_length: int, trainer_max_tokens: Optional[int] = None
    ) -> float:
        """
        Compute score based on thinking length with hardware constraints.

        Args:
            think_length: Actual thinking token count
            trainer_max_tokens: Optional max from trainer config (overrides)

        Returns:
            Score between 0.0 and 1.0 (can exceed 1.0 with bonus)
        """
        # Use trainer limit if available and enabled
        if self.use_trainer_limits and trainer_max_tokens is not None:
            # Adjust thresholds based on trainer limit
            effective_max = min(self.target_length_max, trainer_max_tokens)
            effective_excessive = min(
                self.excessive_length_threshold, trainer_max_tokens + 10
            )
        else:
            effective_max = self.target_length_max
            effective_excessive = self.excessive_length_threshold

        score = 1.0
        status = "optimal"

        # Too short penalty
        if think_length < self.target_length_min:
            # Linear scale from 0 to 1
            score = max(0.0, think_length / self.target_length_min)
            status = "too_short"

        # Too long penalty
        elif think_length > effective_max:
            # Gradual penalty
            excess_ratio = (think_length - effective_max) / effective_max
            score = max(0.0, 1.0 - (excess_ratio * 0.5))
            status = "too_long"

        # Optimal length bonus
        if self.optimal_length_min <= think_length <= self.optimal_length_max:
            score += self.conciseness_bonus
            status = "optimal_with_bonus"

        # Excessive length harsh penalty (critical for 128 token budget!)
        if think_length > effective_excessive:
            excess_ratio = think_length / effective_excessive
            harsh_penalty = self.excessive_length_penalty * excess_ratio
            score -= harsh_penalty
            status = "excessive"

            # Log warning for monitoring
            if self.debug_logging:
                logger.warning(
                    f"EXCESSIVE thinking length {think_length} > {effective_excessive}, "
                    f"harsh_penalty={harsh_penalty:.3f}, score={score:.3f}"
                )

        if self.debug_logging:
            logger.info(
                f"Length score: length={think_length}, status={status}, "
                f"effective_max={effective_max}, score={score:.3f}"
            )

        return max(0.0, score)

    def compute(self, context: RewardContext) -> Dict[str, Any]:
    	generated = context.generated_text
    	# ⭐ FIX: Ensure all return paths yield a dictionary
    	if not generated or len(generated.strip()) < 10:
    		if self.debug_logging: logger.warning(f"ThinkingQuality: Empty or too short text")
    		return {"reward": 0.0, "log": {"error": "Empty generation"}}

    	gen_config = GenerationConfig()
    	think_content = extract_think_region(generated, gen_config)
    	if not think_content:
    		if self.debug_logging: logger.warning('ThinkingQuality: No thinking content found')
    		return {"reward": 0.0, "log": {"error": "No think content"}}

    	# ... (rest of the function logic remains the same)
    	score = 1.0
    	penalties = {}
    	bonuses = {}

    	tag_misuse_penalty = self._check_tag_misuse_penalty(generated, gen_config)
    	if tag_misuse_penalty > 0:
    		score -= tag_misuse_penalty
    		penalties['tag_misuse'] = tag_misuse_penalty

    	special_token_penalty = self._check_special_tokens_penalty(think_content)
    	if special_token_penalty > 0:
    		score -= special_token_penalty
    		penalties['special_tokens'] = special_token_penalty

    	think_len = len(think_content.strip())
    	max_think = context.metadata.get('max_thinking_tokens')
    	length_score = self._compute_length_score(think_len, max_think)
    	score *= length_score

    	if re.search('(\\n\\s*[-*•]|\\n\\s*\\d+\\.\\s+)', think_content):
    		score += 0.1
    		bonuses['structure'] = 0.1

    	bad_phrase_count = 0
    	text_lower = think_content.lower()
    	for phrase in self.bad_phrases:
    		if phrase in text_lower:
    			score -= 0.15
    			bad_phrase_count += 1

    	if bad_phrase_count > 0:
    		penalties['bad_phrases'] = 0.15 * bad_phrase_count

    	final_score = max(0.0, min(1.0, score))
    	log_data = {'length': think_len, 'length_score': length_score, 'penalties': penalties, 'bonuses': bonuses, 'final_score': final_score}

    	if self.debug_logging:
    		logger.info(f"ThinkingQuality | length={think_len}, length_score={length_score:.3f}, final={final_score:.3f}")

    	return {"reward": final_score, "log": log_data}

