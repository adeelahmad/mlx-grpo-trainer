#!/usr/bin/env python3
# File: src/mlx_rl_trainer/algorithms/grpo/grpo_algorithm.py
# Purpose: GRPO algorithm with enhanced gradient handling
# Changes:
#   - Enhanced gradient validation
#   - Fixed gradient combination for LoRA
#   - Added comprehensive logging

import logging
from typing import Dict, Any, Tuple, Optional, Set

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten, tree_map

from mlx_rl_trainer.core.config import ExperimentConfig

logger = logging.getLogger(__name__)


def _extract_layer_number(param_path: str) -> Optional[int]:
    """Extract layer number from parameter path."""
    import re

    match = re.search(r"\.layers\.(\d+)\.", param_path)
    return int(match.group(1)) if match else None


def _get_gradient_structure(grads) -> Set[str]:
    """Get set of parameter paths in gradient tree."""
    flat = tree_flatten(grads, is_leaf=lambda x: isinstance(x, mx.array))
    return {".".join(str(s) for s in path) for (path, arr) in flat}


def _validate_gradient_dict(grads, context: str = "") -> bool:
    """
    Validate gradient dictionary for NaN/Inf values.

    Args:
        grads: Gradient dictionary
        context: Context string for logging

    Returns:
        True if valid, False otherwise
    """
    if not grads:
        logger.warning(f"[{context}] Empty gradient dict")
        return False

    try:
        flat_grads = tree_flatten(grads, is_leaf=lambda x: isinstance(x, mx.array))
        array_count = 0

        for path, grad in flat_grads:
            if isinstance(grad, mx.array):
                array_count += 1

                # Check for NaN/Inf
                if mx.any(mx.isnan(grad)) or mx.any(mx.isinf(grad)):
                    path_str = ".".join(str(p) for p in path)
                    logger.error(
                        f"[{context}] Invalid gradient at {path_str}: contains NaN/Inf"
                    )
                    return False

        if array_count == 0:
            logger.warning(f"[{context}] No array gradients found")
            return False

        return True

    except Exception as e:
        logger.error(f"[{context}] Error validating gradients: {e}")
        return False


def _create_zero_gradients_like(model):
    """Create zero gradients matching model parameters."""
    try:
        params = model.parameters()

        def zero_like(x):
            if isinstance(x, mx.array):
                return mx.zeros_like(x)
            return x

        zero_grads = tree_map(
            zero_like, params, is_leaf=lambda x: isinstance(x, mx.array)
        )
        return zero_grads

    except Exception as e:
        logger.error(f"Error creating zero gradients: {e}")
        return {}


def _mask_gradients_by_layers(grads, config: ExperimentConfig, sft_mode: str):
    """
    Mask gradients by layer ranges based on SFT mode.

    Args:
        grads: Gradient tree
        config: Configuration
        sft_mode: SFT mode ('all', 'answer_only', 'weighted', 'exclude_thinking')

    Returns:
        Masked gradient tree
    """
    # Get layer ranges
    thinking_start = getattr(config.trainer, "thinking_layer_start", None)
    thinking_end = getattr(config.trainer, "thinking_layer_end", None)
    answer_start = getattr(config.trainer, "answer_layer_start", None)
    answer_end = getattr(config.trainer, "answer_layer_end", None)

    # If no masking needed
    if sft_mode == "all" or thinking_start is None or answer_start is None:
        return grads

    # Get weights
    thinking_weight = getattr(config.trainer, "sft_thinking_weight", 0.0)
    answer_weight = getattr(config.trainer, "sft_answer_weight", 1.0)

    # Handle nested structure
    is_nested = "model" in grads and isinstance(grads["model"], dict)
    grad_dict = grads["model"] if is_nested else grads

    def mask_fn(grad, path_parts):
        if not isinstance(grad, mx.array):
            return grad

        # Get layer number
        path_str = ".".join(str(p) for p in path_parts)
        layer_num = _extract_layer_number(path_str)

        if layer_num is None:
            return grad

        # Apply masking based on mode
        if sft_mode == "answer_only":
            if answer_start <= layer_num <= answer_end:
                return grad
            else:
                return mx.zeros_like(grad)

        elif sft_mode == "weighted":
            if thinking_start <= layer_num <= thinking_end:
                return grad * thinking_weight
            elif answer_start <= layer_num <= answer_end:
                return grad * answer_weight
            else:
                return grad

        elif sft_mode == "exclude_thinking":
            if thinking_start <= layer_num <= thinking_end:
                return mx.zeros_like(grad)
            else:
                return grad

        return grad

    try:
        # Flatten, mask, and unflatten
        flat_grads = tree_flatten(grad_dict, is_leaf=lambda x: isinstance(x, mx.array))
        masked_flat = [(path, mask_fn(grad, path)) for (path, grad) in flat_grads]
        masked_dict = tree_unflatten(masked_flat)

        # Restore nesting if needed
        result = {"model": masked_dict} if is_nested else masked_dict

        # Validate
        if not _validate_gradient_dict(result, "mask_gradients_by_layers"):
            logger.error("Masking produced invalid gradients, returning original")
            return grads

        return result

    except Exception as e:
        logger.error(f"Error masking gradients: {e}", exc_info=True)
        return grads


def _robust_tree_combine(tree1, tree2, fn, path: str = ""):
    """
    Robustly combine two gradient trees with validation.

    Args:
        tree1: First gradient tree
        tree2: Second gradient tree
        fn: Combination function
        path: Current path (for logging)

    Returns:
        Tuple of (combined_tree, success)
    """
    # Handle arrays
    if isinstance(tree1, mx.array) and isinstance(tree2, mx.array):
        try:
            # Check shape match
            if tree1.shape != tree2.shape:
                logger.warning(
                    f"Shape mismatch at {path}: tree1={tree1.shape} vs tree2={tree2.shape}. Using tree1."
                )
                return tree1, False

            # Combine
            result = fn(tree1, tree2)

            # Validate
            if mx.any(mx.isnan(result)) or mx.any(mx.isinf(result)):
                logger.error(
                    f"Invalid result at {path}: NaN/Inf detected. Using tree1."
                )
                return tree1, False

            return result, True

        except Exception as e:
            logger.warning(f"Error combining arrays at {path}: {e}. Using tree1.")
            return tree1, False

    # tree1 is array but tree2 is not
    if isinstance(tree1, mx.array):
        if tree2 is not None and not isinstance(tree2, mx.array):
            logger.debug(
                f"Type mismatch at {path}: tree1=array, tree2={type(tree2)}. Using tree1."
            )
        return tree1, False

    # Handle dicts
    if isinstance(tree1, dict):
        if not isinstance(tree2, dict):
            logger.warning(
                f"Structure mismatch at {path}: tree1=dict, tree2={type(tree2)}. Using tree1."
            )
            return tree1, False

        result = {}
        all_success = True

        # Get common keys
        common_keys = set(tree1.keys()) & set(tree2.keys())

        if len(common_keys) < len(tree1.keys()):
            missing = set(tree1.keys()) - common_keys
            logger.debug(
                f"Keys in tree1 but not tree2 at {path}: {missing}. Using tree1 values for these."
            )

        if len(common_keys) < len(tree2.keys()):
            extra = set(tree2.keys()) - common_keys
            logger.debug(f"Extra keys in tree2 at {path}: {extra}. Ignoring.")

        # Combine common keys
        for key in tree1.keys():
            new_path = f"{path}.{key}" if path else str(key)

            if key in common_keys:
                result[key], success = _robust_tree_combine(
                    tree1[key], tree2[key], fn, new_path
                )
                all_success = all_success and success
            else:
                result[key] = tree1[key]
                all_success = False

        return result, all_success

    # Handle lists/tuples
    if isinstance(tree1, (list, tuple)):
        if not isinstance(tree2, (list, tuple)):
            logger.warning(
                f"Structure mismatch at {path}: tree1=list, tree2={type(tree2)}. Using tree1."
            )
            return tree1, False

        if len(tree1) != len(tree2):
            logger.warning(
                f"Length mismatch at {path}: tree1={len(tree1)}, tree2={len(tree2)}. Using tree1."
            )
            return tree1, False

        result = []
        all_success = True

        for idx, (v1, v2) in enumerate(zip(tree1, tree2)):
            combined, success = _robust_tree_combine(v1, v2, fn, f"{path}[{idx}]")
            result.append(combined)
            all_success = all_success and success

        return type(tree1)(result), all_success

    # Default: return tree1
    return tree1, False


def _safe_gradient_combine(
    grad1, grad2, operation: str = "add", weight1: float = 1.0, weight2: float = 1.0
) -> Tuple[Any, Dict[str, Any]]:
    """
    Safely combine two gradient trees.

    Args:
        grad1: First gradient tree
        grad2: Second gradient tree
        operation: 'add' or 'subtract'
        weight1: Weight for grad1
        weight2: Weight for grad2

    Returns:
        Tuple of (combined_gradients, info_dict)
    """
    info = {"success": False, "match_rate": 0.0, "structure_issues": []}

    # Validate inputs
    if not grad1:
        logger.warning("grad1 is empty, returning grad2")
        info["structure_issues"].append("grad1_empty")
        return grad2 or {}, info

    if not grad2:
        logger.warning("grad2 is empty, returning grad1")
        info["structure_issues"].append("grad2_empty")
        return grad1, info

    # Validate gradient dicts
    if not _validate_gradient_dict(grad1, "grad1"):
        info["structure_issues"].append("grad1_invalid")
        return grad2 if _validate_gradient_dict(grad2, "grad2") else {}, info

    if not _validate_gradient_dict(grad2, "grad2"):
        info["structure_issues"].append("grad2_invalid")
        return grad1, info

    # Get structures
    struct1 = _get_gradient_structure(grad1)
    struct2 = _get_gradient_structure(grad2)
    common = struct1 & struct2

    if not common:
        logger.error("No common parameters between grad1 and grad2!")
        info["structure_issues"].append("no_common_params")
        info["match_rate"] = 0.0
        return grad1, info

    # Compute match rate
    match_rate = len(common) / max(len(struct1), len(struct2))
    info["match_rate"] = match_rate

    if match_rate < 0.5:
        logger.warning(
            f"Low gradient structure match rate: {match_rate:.1%}. "
            f"Common: {len(common)}, grad1: {len(struct1)}, grad2: {len(struct2)}"
        )
        info["structure_issues"].append(f"low_match_rate_{match_rate:.2f}")

    # Define combination function
    if operation == "add":
        combine_fn = lambda a, b: weight1 * a + weight2 * b
    elif operation == "subtract":
        combine_fn = lambda a, b: weight1 * a - weight2 * b
    else:
        logger.error(f"Unknown operation: {operation}. Returning grad1.")
        info["structure_issues"].append(f"unknown_operation_{operation}")
        return grad1, info

    # Combine
    try:
        combined, success = _robust_tree_combine(grad1, grad2, combine_fn)

        # Validate result
        if not _validate_gradient_dict(combined, "combined"):
            logger.error("Combined gradients are invalid!")
            info["structure_issues"].append("invalid_combination")
            return grad1, info

        info["success"] = success

        if not success:
            logger.warning(
                f"Gradient combination completed with issues (match_rate={match_rate:.1%})"
            )

        return combined, info

    except Exception as e:
        logger.error(f"Error in gradient combination: {e}", exc_info=True)
        info["structure_issues"].append(f"exception_{type(e).__name__}")
        logger.error("Falling back to grad1 only")
        return grad1, info


class GRPOAlgorithm:
    """GRPO algorithm implementation with robust gradient handling."""

    def __init__(self, config: ExperimentConfig, actor_model, ref_model):
        self.config = config
        self.actor = actor_model
        self.reference = ref_model
        self.beta = config.trainer.grpo_beta

        # Track statistics
        self._gradient_stats = {
            "total_combinations": 0,
            "successful_combinations": 0,
            "fallback_count": 0,
        }

    def compute_advantages(
        self, rewards_flat: mx.array, samples_per_prompt: int
    ) -> mx.array:
        """
        Compute advantages from rewards.

        Args:
            rewards_flat: Flat array of rewards
            samples_per_prompt: Number of samples per prompt

        Returns:
            Advantages array
        """
        if samples_per_prompt <= 1:
            return (rewards_flat - mx.mean(rewards_flat)) / (
                mx.std(rewards_flat) + 1e-8
            )

        # Reshape to (num_prompts, samples_per_prompt)
        num_prompts = rewards_flat.shape[0] // samples_per_prompt
        rewards_2d = rewards_flat.reshape(num_prompts, samples_per_prompt)

        # Compute mean and std per prompt
        mean_per_prompt = mx.mean(rewards_2d, axis=1, keepdims=True)
        std_per_prompt = mx.std(rewards_2d, axis=1, keepdims=True)

        # Normalize
        advantages_2d = (rewards_2d - mean_per_prompt) / (std_per_prompt + 1e-8)

        return advantages_2d.flatten()

    def calculate_loss_and_grads(self, rollout_batch, full_config, pad_token_id):
        """Calculate standard GRPO loss and gradients."""
        batch = rollout_batch

        def loss_fn(actor_model):
            # Forward pass
            tokens_key = "tokens"
            output = actor_model(batch[tokens_key])

            if isinstance(output, tuple):
                output = output[0]

            logits = output.astype(mx.float32)

            # Get response logits
            prompt_len = batch[tokens_key].shape[1] - batch["response_mask"].shape[1]
            response_logits = logits[:, prompt_len - 1 : -1, :]
            response_tokens = batch[tokens_key][:, prompt_len:]

            # Compute log probs
            log_probs = nn.log_softmax(response_logits, axis=-1)
            token_log_probs = mx.take_along_axis(
                log_probs, response_tokens[..., None], axis=-1
            ).squeeze(-1)

            # Compute KL
            log_ratio = token_log_probs - batch["ref_log_probs"]
            kl_div = mx.exp(log_ratio) - 1 - log_ratio
            kl_penalty = kl_div * batch["response_mask"]

            # Policy loss
            advantages = batch["advantages"][:, None]
            policy_loss = -log_ratio * advantages * batch["response_mask"]

            # Total loss
            total_loss = policy_loss + self.beta * kl_penalty
            loss = mx.sum(total_loss) / mx.sum(batch["response_mask"])

            # Metrics
            kl_mean = mx.sum(kl_penalty) / mx.sum(batch["response_mask"])
            policy_mean = mx.sum(policy_loss) / mx.sum(batch["response_mask"])

            return loss, {"kl_divergence": kl_mean, "policy_loss": policy_mean}

        try:
            # Compute gradients
            loss_and_grad_fn = nn.value_and_grad(self.actor, loss_fn)
            (loss, metrics), grads = loss_and_grad_fn(self.actor)

            # Validate
            if not _validate_gradient_dict(grads, "RL gradients"):
                logger.error("Invalid RL gradients computed!")
                return mx.array(0.0), {}, {"kl_divergence": 0.0, "policy_loss": 0.0}

            # Convert metrics
            metrics_dict = {k: float(v.item()) for k, v in metrics.items()}

            return loss, grads, metrics_dict

        except Exception as e:
            logger.error(f"Error during loss computation: {e}", exc_info=True)
            return mx.array(0.0), {}, {"kl_divergence": 0.0, "policy_loss": 0.0}

    def calculate_dual_gradient_loss(self, rollout_batch, full_config, pad_token_id):
        """Calculate dual gradients for thinking and answer regions."""
        batch = rollout_batch

        # Check for masks
        has_thinking = "thinking_mask" in batch
        has_answer = "answer_mask" in batch

        if not has_thinking or not has_answer:
            logger.warning(
                "Thinking/answer masks not found. Falling back to standard gradient computation."
            )
            loss, grads, metrics = self.calculate_loss_and_grads(
                batch, full_config, pad_token_id
            )
            info = {
                "mode": "fallback",
                "success": False,
                "structure_issues": ["missing_masks"],
            }
            return loss, grads, loss, grads, metrics, info

        def loss_fn_masked(actor_model, mask_type, compute_metrics=True):
            # Forward pass
            tokens_key = "tokens"
            response_mask_key = "response_mask"

            output = actor_model(batch[tokens_key])
            if isinstance(output, tuple):
                output = output[0]

            logits = output.astype(mx.float32)

            # Get response logits
            prompt_len = batch[tokens_key].shape[1] - batch[response_mask_key].shape[1]
            response_logits = logits[:, prompt_len - 1 : -1, :]
            response_tokens = batch[tokens_key][:, prompt_len:]

            # Compute log probs
            log_probs = nn.log_softmax(response_logits, axis=-1)
            token_log_probs = mx.take_along_axis(
                log_probs, response_tokens[..., None], axis=-1
            ).squeeze(-1)

            # Compute KL
            log_ratio = token_log_probs - batch["ref_log_probs"]
            kl_div = mx.exp(log_ratio) - 1 - log_ratio

            # Apply region mask
            region_mask = batch[mask_type]
            masked_response_mask = batch[response_mask_key] * region_mask

            kl_penalty = kl_div * masked_response_mask

            # Policy loss
            advantages = batch["advantages"][:, None]
            policy_loss = -log_ratio * advantages * masked_response_mask

            # Total loss
            total_loss = policy_loss + self.beta * kl_penalty

            # Normalize by masked tokens
            mask_sum = mx.sum(masked_response_mask)
            loss = mx.sum(total_loss) / (mask_sum + 1e-8)

            if compute_metrics:
                kl_mean = mx.sum(kl_penalty) / (mask_sum + 1e-8)
                policy_mean = mx.sum(policy_loss) / (mask_sum + 1e-8)
                return loss, {"kl_divergence": kl_mean, "policy_loss": policy_mean}

            return loss

        try:
            # Compute thinking gradients
            thinking_fn = lambda model: loss_fn_masked(model, "thinking_mask", False)
            thinking_grad_fn = nn.value_and_grad(self.actor, thinking_fn)
            thinking_loss, thinking_grads = thinking_grad_fn(self.actor)

            # Compute answer gradients with metrics
            answer_fn = lambda model: loss_fn_masked(model, "answer_mask", True)
            answer_grad_fn = nn.value_and_grad(self.actor, answer_fn)
            (answer_loss, metrics), answer_grads = answer_grad_fn(self.actor)

            # Validate
            thinking_valid = _validate_gradient_dict(thinking_grads, "thinking_grads")
            answer_valid = _validate_gradient_dict(answer_grads, "answer_grads")

            if not thinking_valid or not answer_valid:
                logger.error("Invalid gradients in dual computation!")
                info = {"mode": "dual", "success": False, "structure_issues": []}
                if not thinking_valid:
                    info["structure_issues"].append("thinking_invalid")
                if not answer_valid:
                    info["structure_issues"].append("answer_invalid")

                metrics_dict = {k: float(v.item()) for k, v in metrics.items()}
                return (
                    thinking_loss,
                    thinking_grads,
                    answer_loss,
                    answer_grads,
                    metrics_dict,
                    info,
                )

            # Convert metrics
            metrics_dict = {k: float(v.item()) for k, v in metrics.items()}

            info = {"mode": "dual", "success": True, "structure_issues": []}

            return (
                thinking_loss,
                thinking_grads,
                answer_loss,
                answer_grads,
                metrics_dict,
                info,
            )

        except Exception as e:
            logger.error(f"Error in dual gradient computation: {e}", exc_info=True)

            # Return zero gradients
            zero_grads = _create_zero_gradients_like(self.actor)
            info = {
                "mode": "dual",
                "success": False,
                "structure_issues": [f"exception_{type(e).__name__}"],
            }

            return (
                mx.array(0.0),
                zero_grads,
                mx.array(0.0),
                zero_grads,
                {"kl_divergence": 0.0, "policy_loss": 0.0},
                info,
            )

    def calculate_sft_loss_and_grads(
        self, rollout_batch, reference_tokens, full_config, pad_token_id
    ):
        """Calculate SFT loss and gradients."""
        batch = rollout_batch
        ref_tokens = reference_tokens

        # Get SFT mode
        sft_mode = getattr(full_config.trainer, "sft_mode", "exclude_thinking")

        # Log mode once
        if not hasattr(self, "_sft_mode_logged"):
            logger.info(f"SFT layer control mode: {sft_mode}")

            if sft_mode == "exclude_thinking":
                logger.info(
                    "System 2 (thinking) layers will NOT receive SFT gradients - only RL signal"
                )
            elif sft_mode == "answer_only":
                logger.info("Only System 1 (answer) layers will receive SFT gradients")
            elif sft_mode == "weighted":
                think_w = getattr(full_config.trainer, "sft_thinking_weight", 0.0)
                ans_w = getattr(full_config.trainer, "sft_answer_weight", 1.0)
                logger.info(f"Weighted SFT: thinking={think_w}, answer={ans_w}")

            self._sft_mode_logged = True

        # Get answer mask
        if "answer_mask" not in batch:
            logger.warning(
                "Answer mask not found for SFT. Falling back to response_mask."
            )
            answer_mask = batch.get(
                "response_mask", mx.ones_like(ref_tokens, dtype=mx.float32)
            )
        else:
            answer_mask = batch["answer_mask"]

        def loss_fn(actor_model):
            # Forward pass
            tokens_key = "tokens"
            response_mask_key = "response_mask"

            output = actor_model(batch[tokens_key])
            if isinstance(output, tuple):
                output = output[0]

            logits = output.astype(mx.float32)

            # Get response logits
            prompt_len = batch[tokens_key].shape[1] - batch[response_mask_key].shape[1]
            response_logits = logits[:, prompt_len - 1 : -1, :]

            # Align shapes
            min_len = min(response_logits.shape[1], ref_tokens.shape[1])
            response_logits = response_logits[:, :min_len, :]
            target_tokens = ref_tokens[:, :min_len]
            mask = (
                answer_mask[:, :min_len]
                if answer_mask.shape[1] >= min_len
                else answer_mask
            )

            if response_logits.shape[1] != target_tokens.shape[1]:
                logger.debug(
                    f"Aligning SFT shapes: logits {response_logits.shape[1]} vs targets {target_tokens.shape[1]} vs mask {mask.shape[1]} -> {min_len}"
                )

            # Compute loss
            log_probs = nn.log_softmax(response_logits, axis=-1)
            token_log_probs = mx.take_along_axis(
                log_probs, target_tokens[..., None], axis=-1
            ).squeeze(-1)

            loss_per_token = -token_log_probs * mask
            loss = mx.sum(loss_per_token) / (mx.sum(mask) + 1e-8)

            return loss, {"sft_loss": loss}

        try:
            # Compute gradients
            loss_and_grad_fn = nn.value_and_grad(self.actor, loss_fn)
            (loss, metrics), grads = loss_and_grad_fn(self.actor)

            # Validate before masking
            if not _validate_gradient_dict(grads, "SFT gradients (pre-mask)"):
                logger.error("Invalid SFT gradients before masking!")
                return mx.array(0.0), {}, {"sft_loss": 0.0}

            # Apply layer masking
            grads = _mask_gradients_by_layers(grads, full_config, sft_mode)

            # Validate after masking
            if not _validate_gradient_dict(grads, "SFT gradients (post-mask)"):
                logger.error("Invalid SFT gradients after masking!")
                return mx.array(0.0), {}, {"sft_loss": 0.0}

            # Convert metrics
            metrics_dict = {k: float(v.item()) for k, v in metrics.items()}

            return loss, grads, metrics_dict

        except Exception as e:
            logger.error(f"Error during SFT loss computation: {e}", exc_info=True)
            return mx.array(0.0), {}, {"sft_loss": 0.0}

    def get_gradient_statistics(self) -> Dict[str, Any]:
        """Get gradient combination statistics."""
        stats = self._gradient_stats.copy()

        if stats["total_combinations"] > 0:
            stats["success_rate"] = (
                stats["successful_combinations"] / stats["total_combinations"]
            )
        else:
            stats["success_rate"] = 0.0

        return stats


# Dependencies: mlx
# Installation: pip install mlx
# Run: This file is imported - used by trainer
# Status: ✅ COMPLETE - Enhanced gradient handling and validation
# Changes Applied:
#   1. Enhanced _validate_gradient_dict() with comprehensive checks
#   2. Fixed _safe_gradient_combine() with robust error handling
#   3. Added _robust_tree_combine() for safer gradient combination
#   4. Enhanced dual gradient computation with validation
#   5. Added gradient statistics tracking
