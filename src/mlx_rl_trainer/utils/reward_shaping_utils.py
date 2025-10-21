"""
Reward Shaping Utilities for Hybrid RL+SFT Training

Helpers for constraining model behavior during RL training,
especially for hardware-constrained environments with limited token budgets.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def apply_thinking_length_penalty(
    base_reward: float,
    generated_text: str,
    config: Any,
    tokenizer: Any = None,
) -> Dict[str, float]:
    """
    Apply penalty for excessive thinking tokens.

    Problem: With pure RL on thinking, model may discover that longer
    reasoning correlates with better outcomes, leading to infinite rambling.

    Solution: Penalize thinking that exceeds a reasonable threshold.

    Args:
        base_reward: Original reward from reward function
        generated_text: Full generation including <think>...</think>
        config: Training configuration with thinking constraints
        tokenizer: Optional tokenizer for exact token counting

    Returns:
        Dictionary with:
            - total: Final reward after penalties
            - base: Original reward
            - thinking_penalty: Applied penalty (0 if within limits)
            - thinking_tokens: Number of thinking tokens
    """
    # Get constraints from config
    max_thinking_tokens = getattr(config.trainer, "max_thinking_tokens", 30)
    thinking_penalty_rate = getattr(config.trainer, "thinking_penalty_rate", 0.05)

    # Extract thinking portion
    if "</think>" not in generated_text:
        # No thinking section - warn but don't penalize
        # (This is already problematic in other ways)
        return {
            "total": base_reward,
            "base": base_reward,
            "thinking_penalty": 0.0,
            "thinking_tokens": 0,
        }

    thinking_part = generated_text.split("</think>")[0] + "</think>"

    # Count thinking tokens
    if tokenizer is not None:
        thinking_tokens = len(tokenizer.encode(thinking_part))
    else:
        # Rough estimate if no tokenizer (1 token ≈ 4 chars)
        thinking_tokens = len(thinking_part) // 4

    # Calculate penalty
    if thinking_tokens > max_thinking_tokens:
        excess_tokens = thinking_tokens - max_thinking_tokens
        penalty = thinking_penalty_rate * excess_tokens
        final_reward = base_reward - penalty

        logger.debug(
            f"Thinking penalty: {thinking_tokens} tokens (>{max_thinking_tokens}), penalty={penalty:.3f}"
        )
    else:
        penalty = 0.0
        final_reward = base_reward

    return {
        "total": final_reward,
        "base": base_reward,
        "thinking_penalty": penalty,
        "thinking_tokens": thinking_tokens,
    }


def apply_thinking_efficiency_bonus(
    base_reward: float,
    generated_text: str,
    config: Any,
    tokenizer: Any = None,
) -> Dict[str, float]:
    """
    Reward concise but correct thinking.

    Strategy: Give bonus for achieving high reward with fewer thinking tokens.
    This encourages efficiency rather than just penalizing length.

    Args:
        base_reward: Original reward from reward function
        generated_text: Full generation including <think>...</think>
        config: Training configuration
        tokenizer: Optional tokenizer for exact token counting

    Returns:
        Dictionary with total reward and breakdown
    """
    # Get config
    optimal_thinking_tokens = getattr(config.trainer, "optimal_thinking_tokens", 50)
    efficiency_bonus_weight = getattr(config.trainer, "efficiency_bonus_weight", 0.1)

    # Extract and count thinking tokens
    if "</think>" not in generated_text:
        return {
            "total": base_reward,
            "base": base_reward,
            "efficiency_bonus": 0.0,
            "thinking_tokens": 0,
        }

    thinking_part = generated_text.split("</think>")[0] + "</think>"

    if tokenizer is not None:
        thinking_tokens = len(tokenizer.encode(thinking_part))
    else:
        thinking_tokens = len(thinking_part) // 4

    # Calculate efficiency bonus
    # Formula: bonus = base_reward * weight * (optimal / actual)
    # Capped at 2x to prevent over-rewarding very short thinking
    if thinking_tokens > 0:
        efficiency_ratio = min(optimal_thinking_tokens / thinking_tokens, 2.0)
        bonus = base_reward * efficiency_bonus_weight * (efficiency_ratio - 1.0)
        bonus = max(0.0, bonus)  # Only give bonus, never penalty
    else:
        bonus = 0.0

    final_reward = base_reward + bonus

    if bonus > 0:
        logger.debug(
            f"Efficiency bonus: {thinking_tokens} tokens (optimal={optimal_thinking_tokens}), bonus={bonus:.3f}"
        )

    return {
        "total": final_reward,
        "base": base_reward,
        "efficiency_bonus": bonus,
        "thinking_tokens": thinking_tokens,
    }


def combined_thinking_reward_shaping(
    base_reward: float,
    generated_text: str,
    config: Any,
    tokenizer: Any = None,
) -> Dict[str, float]:
    """
    Apply both penalty and bonus for optimal thinking length control.

    Uses soft penalties: small penalty for excess, small bonus for efficiency.
    This creates a smooth reward landscape that guides toward optimal length.

    Args:
        base_reward: Original reward
        generated_text: Full generation
        config: Training configuration
        tokenizer: Optional tokenizer

    Returns:
        Dictionary with final reward and all components
    """
    # Apply penalty first
    penalty_result = apply_thinking_length_penalty(
        base_reward, generated_text, config, tokenizer
    )

    # Then apply efficiency bonus to penalized reward
    bonus_result = apply_thinking_efficiency_bonus(
        penalty_result["total"], generated_text, config, tokenizer
    )

    return {
        "total": bonus_result["total"],
        "base": base_reward,
        "thinking_penalty": penalty_result.get("thinking_penalty", 0.0),
        "efficiency_bonus": bonus_result.get("efficiency_bonus", 0.0),
        "thinking_tokens": penalty_result.get("thinking_tokens", 0),
        "shaped_reward": bonus_result["total"],
    }


# Example integration into reward composer
class ThinkingConstrainedRewardWrapper:
    """
    Wrapper that adds thinking length constraints to any reward function.

    Usage:
        original_reward_fn = YourRewardFunction()
        constrained_reward_fn = ThinkingConstrainedRewardWrapper(
            original_reward_fn,
            config,
            tokenizer
        )
    """

    def __init__(self, base_reward_fn, config, tokenizer=None):
        self.base_reward_fn = base_reward_fn
        self.config = config
        self.tokenizer = tokenizer
        self.use_penalty = getattr(config.trainer, "use_thinking_penalty", True)
        self.use_bonus = getattr(config.trainer, "use_thinking_bonus", False)

    def compute(self, context):
        """Compute shaped reward with thinking constraints."""
        # Get base reward from original function
        base_reward = self.base_reward_fn.compute(context)

        generated_text = context.generated_text

        if self.use_penalty and self.use_bonus:
            # Use combined shaping
            result = combined_thinking_reward_shaping(
                base_reward, generated_text, self.config, self.tokenizer
            )
            return result["total"]
        elif self.use_penalty:
            # Penalty only
            result = apply_thinking_length_penalty(
                base_reward, generated_text, self.config, self.tokenizer
            )
            return result["total"]
        elif self.use_bonus:
            # Bonus only
            result = apply_thinking_efficiency_bonus(
                base_reward, generated_text, self.config, self.tokenizer
            )
            return result["total"]
        else:
            # No shaping
            return base_reward
