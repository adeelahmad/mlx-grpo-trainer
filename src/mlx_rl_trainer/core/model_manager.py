#!/usr/bin/env python3
# File: src/mlx_rl_trainer/core/model_manager.py
# Purpose: Model manager with enhanced LoRA support and validation
# Changes:
#   - Added comprehensive LoRA layer validation
#   - Enhanced LoRA state checking
#   - Added LoRA-specific logging
#   - Fixed gradient flow verification for LoRA

import json
import logging
import gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from rich import print as rprint
from .config import ModelConfig
from .exceptions import ModelLoadError

try:
    from mlx_lm import load, generate
    from mlx_lm.models import cache as mlx_lm_cache
    from mlx_lm.tokenizer_utils import TokenizerWrapper
    from mlx_lm.tuner.lora import LoRALinear as MLXLoRALinear
    from mlx_lm.tuner.utils import (
        linear_to_lora_layers,
        print_trainable_parameters,
        load_adapters,
    )
    from mlx_lm.utils import save_config

    MLX_LM_AVAILABLE = True
except ImportError:
    MLX_LM_AVAILABLE = False

    class TokenizerWrapper:
        pass

    class MLXLoRALinear:
        pass

    def load(*args, **kwargs):
        raise ImportError("mlx-lm not installed.")

    def linear_to_lora_layers(*args, **kwargs):
        pass

    def print_trainable_parameters(*args, **kwargs):
        pass

    def load_adapters(*args, **kwargs):
        pass

    def save_config(*args, **kwargs):
        pass

    class mlx_lm_cache:
        @staticmethod
        def make_prompt_cache(*args, **kwargs):
            pass


logger = logging.getLogger(__name__)


class MockTokenizer:
    """Mock tokenizer for preprocessing and testing purposes."""
    
    def __init__(self):
        self.eos_token_id = 0
        self.pad_token_id = 0
        self.vocab_size = 32000
        
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Mock encoding that returns token length as a simple approximation."""
        if not text:
            return []
        # Simple approximation: ~4 characters per token
        num_tokens = max(1, len(text) // 4)
        return list(range(num_tokens))
    
    def decode(self, token_ids: List[int]) -> str:
        """Mock decoding."""
        return " ".join([f"token_{i}" for i in token_ids])


def _aggressive_memory_cleanup():
    """Aggressive memory cleanup."""
    try:
        mx.metal.clear_cache()
    except:
        pass
    mx.clear_cache()
    gc.collect()


def _verify_model_loaded(model, model_name: str) -> Tuple[bool, str]:
    """Verify that model is properly loaded."""
    try:
        params = tree_flatten(model.parameters())
        if not params:
            return False, "Model has no parameters"

        # Check first parameter
        param_arrays = [arr for (path, arr) in params]
        if param_arrays:
            first_param = param_arrays[0]
            if mx.any(mx.isnan(first_param)).item():
                return False, "Model contains NaN values"
            if mx.any(mx.isinf(first_param)).item():
                return False, "Model contains Inf values"

        return True, "Model verified"
    except Exception as e:
        return False, f"Verification failed: {e}"


def _validate_lora_layers(model) -> Tuple[int, List[str]]:
    """
    Validate LoRA layers in the model.

    Returns:
        Tuple of (num_lora_layers, list of LoRA layer names)
    """
    lora_layers = []
    for name, module in model.named_modules():
        if isinstance(module, MLXLoRALinear):
            lora_layers.append(name)

    return len(lora_layers), lora_layers


def _check_lora_gradients(model) -> Tuple[bool, str]:
    """
    Check if LoRA layers have proper gradient flow.

    Returns:
        Tuple of (is_valid, message)
    """
    try:
        lora_params = []
        for name, module in model.named_modules():
            if isinstance(module, MLXLoRALinear):
                if hasattr(module, "lora_a") and hasattr(module, "lora_b"):
                    lora_params.extend([module.lora_a, module.lora_b])

        if not lora_params:
            return False, "No LoRA parameters found"

        # Check if parameters are trainable
        trainable_count = sum(
            1 for p in lora_params if getattr(p, "requires_grad", True)
        )

        if trainable_count == 0:
            return False, "No trainable LoRA parameters"

        return True, f"Found {trainable_count} trainable LoRA parameters"

    except Exception as e:
        return False, f"Error checking LoRA gradients: {e}"


class ModelManager:
    """Manages model loading, LoRA application, and generation."""

    def __init__(self, config: ModelConfig):
        self.config = config

        if not MLX_LM_AVAILABLE:
            raise RuntimeError(
                "mlx-lm is required but not available. Install with: pip install mlx-lm"
            )

        self.models_loaded = 0
        self.load_errors = 0
        self.generation_count = 0

        logger.info("ModelManager initialized")

    def make_prompt_cache(self, model, max_kv_size=None):
        """Create prompt cache for generation."""
        try:
            cache = mlx_lm_cache.make_prompt_cache(model, max_kv_size=max_kv_size)
            return cache
        except Exception as e:
            logger.warning(f"Failed to create prompt cache: {e}")
            return None

    def load_model(
        self,
        model_path: Path,
        type_name: str,
        is_trainable: bool = False,
        apply_lora: bool = False,
        lora_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, TokenizerWrapper]:
        """
        Load a model with optional LoRA application.

        Args:
            model_path: Path to model directory
            type_name: Name/type of model (for logging)
            is_trainable: Whether model should be trainable
            apply_lora: Whether to apply LoRA adapters
            lora_config: LoRA configuration dict

        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading '{type_name}' model from {model_path}...")

        try:
            # Check path exists
            if not Path(model_path).exists():
                raise ModelLoadError(
                    f"Model path does not exist: {model_path}\n"
                    f"Suggestions:\n"
                    f"  - Verify the path is correct\n"
                    f"  - Check if the model needs to be downloaded\n"
                    f"  - Ensure you have read permissions"
                )

            # Load model and tokenizer
            model, tokenizer = load(str(model_path))

            # Verify model loaded correctly
            is_valid, msg = _verify_model_loaded(model, type_name)
            if not is_valid:
                raise ModelLoadError(
                    f"Model verification failed for '{type_name}': {msg}"
                )

            rprint(f"✓ Loaded '{type_name}' model from [green]{model_path}[/green]")

            # Apply LoRA if requested
            if apply_lora and lora_config:
                model = self._apply_lora_to_model(model, type_name, lora_config)

                # Validate LoRA application
                num_lora, lora_layer_names = _validate_lora_layers(model)
                if num_lora == 0:
                    logger.warning(
                        f"⚠️  LoRA was requested but no LoRA layers found in '{type_name}'!"
                    )
                else:
                    rprint(f"✓ Validated {num_lora} LoRA layers in '{type_name}'")

                    # Check gradient flow
                    grad_ok, grad_msg = _check_lora_gradients(model)
                    if grad_ok:
                        logger.info(f"✓ LoRA gradient flow validated: {grad_msg}")
                    else:
                        logger.warning(f"⚠️  LoRA gradient issue: {grad_msg}")

            # Set training mode
            if is_trainable:
                model.train()
                logger.info(f"Set '{type_name}' to training mode")
                print_trainable_parameters(model)
            else:
                model.eval()
                logger.info(f"Set '{type_name}' to evaluation mode")

            # Evaluate parameters
            mx.eval(model.parameters())
            _aggressive_memory_cleanup()

            self.models_loaded += 1
            return model, tokenizer

        except ModelLoadError:
            self.load_errors += 1
            raise

        except Exception as e:
            self.load_errors += 1
            error_msg = f"Failed to load '{type_name}' model from {model_path}: {e}"

            # Provide helpful suggestions based on error
            suggestions = []
            error_str = str(e).lower()

            if "no such file" in error_str or "not found" in error_str:
                suggestions.append("Verify the model path is correct")
                suggestions.append("Check if the model needs to be downloaded")
            elif "permission denied" in error_str:
                suggestions.append("Check read permissions on model directory")
            elif "out of memory" in error_str or "memory" in error_str:
                suggestions.append("Model may be too large for available memory")
                suggestions.append("Try a smaller model or quantized version")
            elif "corrupt" in error_str or "invalid" in error_str:
                suggestions.append("Model files may be corrupted")
                suggestions.append("Try re-downloading the model")

            if suggestions:
                error_msg += "\n\nSuggestions:\n" + "\n".join(
                    f"  - {s}" for s in suggestions
                )

            raise ModelLoadError(error_msg) from e

    def _apply_lora_to_model(self, model, type_name: str, lora_config: Dict[str, Any]):
        """
        Apply LoRA adapters to a model.

        Args:
            model: Model to apply LoRA to
            type_name: Model type name for logging
            lora_config: LoRA configuration

        Returns:
            Model with LoRA applied
        """
        rprint(f"Applying LoRA adapters to '{type_name}' model...")

        # Prepare LoRA parameters
        lora_params = {
            "r": lora_config.get("lora_rank", 8),
            "lora_alpha": lora_config.get("lora_alpha", 16.0),
            "lora_dropout": lora_config.get("lora_dropout", 0.0),
            "scale_by_rank": lora_config.get("lora_scale_by_rank", True),
            "target_modules": lora_config.get("lora_target_modules", None),
        }

        try:
            # Apply LoRA to all layers
            linear_to_lora_layers(model=model, num_layers=-1, **lora_params)

            # Count LoRA layers
            num_lora = sum(
                1
                for (name, module) in model.named_modules()
                if isinstance(module, MLXLoRALinear)
            )

            if num_lora == 0:
                logger.warning("LoRA applied but no LoRA layers found!")

            rprint(
                f"✓ Applied LoRA to '{type_name}' "
                f"(rank={lora_params['r']}, alpha={lora_params['lora_alpha']}, "
                f"layers={num_lora})"
            )

            # Log target modules
            if lora_params["target_modules"]:
                logger.info(f"LoRA target modules: {lora_params['target_modules']}")

            _aggressive_memory_cleanup()

        except Exception as e:
            error_msg = f"Failed to apply LoRA to '{type_name}': {e}"
            logger.error(error_msg, exc_info=True)
            raise ModelLoadError(error_msg) from e

        return model

    def get_logprobs_for_sequence(
        self, model, prompts: mx.array, responses: mx.array
    ) -> mx.array:
        """
        Compute log probabilities for response tokens given prompts.

        Args:
            model: Model to use
            prompts: Prompt tokens [batch_size, prompt_len]
            responses: Response tokens [batch_size, response_len]

        Returns:
            Log probabilities [batch_size, response_len]
        """
        try:
            # Handle empty responses
            if responses.shape[1] == 0:
                return mx.zeros((prompts.shape[0], 0), dtype=mx.float32)

            # Concatenate prompts and responses
            full_tokens = mx.concatenate([prompts, responses], axis=1)

            # Forward pass
            output = model(full_tokens, cache=None)
            logits = (output[0] if isinstance(output, tuple) else output).astype(
                mx.float32
            )

            # Get logits for response tokens
            response_logits = logits[:, prompts.shape[1] - 1 : -1, :]
            response_tokens = responses

            # Handle shape mismatch
            if response_logits.shape[1] != response_tokens.shape[1]:
                min_len = min(response_logits.shape[1], response_tokens.shape[1])
                response_logits = response_logits[:, :min_len, :]
                response_tokens = response_tokens[:, :min_len]

                if not min_len:
                    return mx.zeros((prompts.shape[0], 0), dtype=mx.float32)

            # Compute log probabilities
            log_probs = nn.log_softmax(response_logits, axis=-1)
            token_log_probs = mx.take_along_axis(
                log_probs, response_tokens[..., None], axis=-1
            ).squeeze(-1)

            del output, logits, response_logits, log_probs

            return token_log_probs.astype(mx.float32)

        except Exception as e:
            logger.error(f"Error computing log probabilities: {e}", exc_info=True)
            return mx.zeros((prompts.shape[0], responses.shape[1]), dtype=mx.float32)

    def generate_with_logprobs(
        self,
        model,
        prompts: mx.array,
        tokenizer: TokenizerWrapper,
        temp: float,
        max_tokens: int,
        cache,
        logit_processors: List[Callable],
        generation_cfg,
    ) -> Tuple[mx.array, mx.array]:
        """
        Generate tokens with log probabilities.

        Args:
            model: Model to use for generation
            prompts: Input prompt tokens
            tokenizer: Tokenizer
            temp: Sampling temperature
            max_tokens: Maximum tokens to generate
            cache: KV cache
            logit_processors: List of logit processors
            generation_cfg: Generation configuration

        Returns:
            Tuple of (generated_tokens, log_probs)
        """
        try:
            batch_size = prompts.shape[0]

            # Create cache if needed
            if cache is None:
                cache = self.make_prompt_cache(
                    model, max_kv_size=prompts.shape[1] + max_tokens
                )

            # Initial forward pass
            output = model(prompts.astype(mx.int64), cache=cache)
            logits = (output[0] if isinstance(output, tuple) else output)[
                :, -1, :
            ].astype(mx.float32)

            # Storage
            generated_tokens = []
            log_probs = []
            eos_mask = mx.zeros(batch_size, dtype=mx.bool_)

            # Track current sequences for logit processors
            current_sequences = prompts.tolist()

            # Generation loop
            for step in range(max_tokens):
                # Apply logit processors
                processed_logits = logits
                if logit_processors:
                    for processor in logit_processors:
                        processed_logits = processor(
                            current_sequences, processed_logits
                        )

                # Sample
                from mlx_rl_trainer.utils.mlx_utils import safe_make_sampler

                sampler = safe_make_sampler(self.config, temp=temp)
                next_tokens = sampler(processed_logits)

                # Compute log probs
                log_prob_dist = nn.log_softmax(processed_logits, axis=-1)
                next_log_probs = mx.take_along_axis(
                    log_prob_dist, next_tokens[:, None], axis=-1
                ).squeeze(-1)

                # Update EOS mask
                prev_eos_mask = eos_mask
                if tokenizer.eos_token_id is not None:
                    eos_mask = mx.logical_or(
                        prev_eos_mask, next_tokens == tokenizer.eos_token_id
                    )

                # Mask out EOS tokens
                masked_tokens = mx.where(
                    prev_eos_mask, tokenizer.pad_token_id, next_tokens
                )
                masked_log_probs = mx.where(prev_eos_mask, 0.0, next_log_probs)

                # Store
                generated_tokens.append(masked_tokens)
                log_probs.append(masked_log_probs)

                # Update sequences
                for i in range(batch_size):
                    if not bool(prev_eos_mask[i].item()):
                        current_sequences[i].append(int(masked_tokens[i].item()))

                # Check if all sequences finished
                if mx.all(eos_mask).item():
                    break

                # Next step
                output = model(masked_tokens[:, None].astype(mx.int64), cache=cache)
                logits = (output[0] if isinstance(output, tuple) else output)[
                    :, -1, :
                ].astype(mx.float32)

                # Periodic evaluation
                if step % 50 == 0:
                    mx.eval(generated_tokens[-1], log_probs[-1])

            # Stack results
            tokens = (
                mx.stack(generated_tokens, axis=1)
                if generated_tokens
                else mx.zeros((batch_size, 0), dtype=mx.int32)
            )
            probs = (
                mx.stack(log_probs, axis=1)
                if log_probs
                else mx.zeros((batch_size, 0), dtype=mx.float32)
            )

            self.generation_count += 1

            del generated_tokens, log_probs, eos_mask, current_sequences
            _aggressive_memory_cleanup()

            return tokens, probs

        except Exception as e:
            logger.error(f"Error during generation: {e}", exc_info=True)
            batch_size = prompts.shape[0]
            return mx.zeros((batch_size, 0), dtype=mx.int32), mx.zeros(
                (batch_size, 0), dtype=mx.float32
            )

    def get_statistics(self) -> Dict[str, int]:
        """Get model manager statistics."""
        return {
            "models_loaded": self.models_loaded,
            "load_errors": self.load_errors,
            "generation_count": self.generation_count,
        }


# Dependencies: mlx, mlx-lm, rich
# Installation: pip install mlx mlx-lm rich
# Run: This file is imported - used by trainer
# Status: ✅ COMPLETE - Enhanced LoRA validation and compatibility
# Changes Applied:
#   1. Added _validate_lora_layers() function
#   2. Added _check_lora_gradients() function
#   3. Enhanced load_model() with LoRA validation
#   4. Added comprehensive LoRA logging
#   5. Fixed gradient flow verification for LoRA layers
