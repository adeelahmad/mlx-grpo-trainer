"""MLX-specific utility functions."""

import logging
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import gc
import re
import string
import random
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
from pathlib import Path

from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from mlx_rl_trainer.core.config import ExperimentConfig, GenerationConfig
from mlx_rl_trainer.core.exceptions import CheckpointError
import sys

logger = logging.getLogger(__name__)

# Import our new sampling configuration system
try:
    from mlx_rl_trainer.core.config import EnhancedGenerationConfig
    from mlx_rl_trainer.generation.samplers.factory import (
        SamplerFactory, get_global_factory, create_sampler,
        SamplerCreationStrategy, SamplerCreationContext
    )
    from mlx_rl_trainer.generation.bridge.config_resolver import (
        get_global_resolver, resolve_sampling_parameters, ResolutionContext
    )
    ENHANCED_SAMPLING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Enhanced sampling system not available: {e}")
    ENHANCED_SAMPLING_AVAILABLE = False
    # Define a fallback alias to prevent NameError
    EnhancedGenerationConfig = None

try:
    from mlx_lm.tuner.lora import LoRALinear as MLXLoRALinear
except ImportError:

    class MLXLoRALinear:
        pass


logger = logging.getLogger(__name__)

TARGET_FLOAT_DTYPE = mx.bfloat16
LETTER_ALPH = string.ascii_uppercase
_TOOL_LIKE_MARKERS = [
    "<tool_call",
    "</tool_call",
    "<tool>",
    "</tool>",
    "<tool_",
    "<function",
    "</function",
    "<json",
    "</json",
    "<scratchpad",
    "</scratchpad",
]


def limit_memory(max_memory_gb: float) -> Optional[int]:
    """Sets the MLX memory limit using updated API with error handling."""
    try:
        if hasattr(mx, "get_peak_memory"):
            logging.info(f"Initial peak memory: {mx.get_peak_memory() / 1e9:.3f} GB")
        else:
            logging.warning("mx.get_peak_memory() not found in this MLX version.")

        max_memory_bytes = int(max_memory_gb * 1024 * 1024 * 1024)

        if hasattr(mx, "set_memory_limit"):
            previous_limit = mx.set_memory_limit(max_memory_bytes)
            logging.info(
                f"MLX memory limit set to {max_memory_gb} GB. Previous limit: {(previous_limit / (1024**3)):.2f} GB"
            )
            return previous_limit
        elif hasattr(mx.metal, "set_memory_limit"):  # Check older API location
            previous_limit = mx.metal.set_memory_limit(max_memory_bytes)
            logging.info(
                f"MLX memory limit set to {max_memory_gb} GB (using mx.metal). Previous limit: {(previous_limit / (1024**3)):.2f} GB"
            )
            sys.exit(0)
            return previous_limit
        else:
            logging.warning(
                "mx.set_memory_limit() not found in this MLX version. Cannot limit memory."
            )
            return None
    except AttributeError:
        logging.warning(
            "MLX memory management functions (get_peak_memory/set_memory_limit) not found. Check MLX version."
        )
        return None
    except Exception as e:
        logging.error(f"Failed to set MLX memory limit: {e}", exc_info=True)
        return None


def _is_metal_internal_error(err: BaseException) -> bool:
    s = str(err)
    return (
        "Command buffer execution failed" in s
        or "[METAL]" in s
        or "Internal Error" in s
    )


def metal_recover(stage: str):
    logging.warning(f"[METAL] Recovering after error at stage: {stage}")
    try:
        mx.synchronize()
    except Exception:
        pass
    mx.clear_cache()
    gc.collect()


def metal_safe_apply_gradients(
    optimizer: optim.Optimizer, grads: Dict[str, mx.array], params: Dict[str, mx.array]
):
    try:
        optimizer.apply_gradients(grads, params)
    except Exception as e:
        if _is_metal_internal_error(e):
            metal_recover("apply_gradients")
            return
        raise
    finally:
        mx.clear_cache()
        gc.collect()


def _find_embedding_layer(root: nn.Module) -> Tuple[str, nn.Module]:
    for name, mod in root.named_modules():
        if isinstance(mod, (nn.Embedding, nn.QuantizedEmbedding)):
            return name, mod
    raise RuntimeError("No nn.Embedding layer found.")


def _freeze_module(module: nn.Module):
    if module:
        for p in module.parameters():
            p.flags.train = False


class ContentAlignBridge(nn.Module):
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        teacher_tokenizer: TokenizerWrapper,
        student_tokenizer: TokenizerWrapper,
        bridge_path: str,
        pool: str = "mean",
        scale: float = 1.0,
        gen_cfg: Optional[GenerationConfig] = None,
    ):
        super().__init__()
        self.tok_t, self.tok_s, self.pool, self.scale = (
            teacher_tokenizer,
            student_tokenizer,
            pool,
            float(scale),
        )
        self.gen_cfg = gen_cfg or GenerationConfig(
            think_start_tag="<think>",
            think_end_tag="</think>",
            answer_start_tag="<answer>",
            answer_end_tag="</answer>",
            think_boost_tokens=[],
            think_temperature=0.7,
            answer_temperature=0.7,
            sampling_top_p=0.9,
            sampling_min_p=0.0,
            sampling_top_k=0,
            repetition_penalty=1.0,
            repetition_context_size=20,
            min_think_tokens=10,
            think_end_early_bias=0.0,
            bias_answer_start_after_min_think=False,
            bias_close_think=0.0,
            bias_answer_start=0.0,
            punish_extra_think_end=0.0,
            punish_reopen_think=0.0,
            punish_reopen_answer=0.0,
            bias_eos_after_answer=0.0,
            hard_mask_mcq_first_token=False,
            mcq_letter_lift=0.0,
            mcq_ban_first_bias=0.0,
            nonmcq_ban_first_bias=0.0,
            mcq_close_after_k=5,
            min_answer_tokens=1,
            min_answer_tokens_mcq=1,
            mcq_answer_end_bias=0.0,
            encourage_think_bias=0.0,
            ban_think_bias=0.0,
            allow_tool_calls=False,
            tool_call_penalty=0.0,
            think_length_target_min=50,
            think_length_target_max=200,
            think_length_penalty_strength=0.0
        )
        _, self.t_emb = _find_embedding_layer(teacher_model)
        _, self.s_emb = _find_embedding_layer(student_model)
        t_dim, s_dim = int(self.t_emb.weight.shape[1]), int(self.s_emb.weight.shape[1])
        hidden = max(t_dim, s_dim)
        self.bridge = nn.Sequential(
            nn.Linear(t_dim, hidden, bias=False),
            nn.ReLU(),
            nn.Linear(hidden, s_dim, bias=False),
        )
        try:
            w = mx.load(str(bridge_path))
            self.bridge.update(tree_unflatten(list(w.items())))
        except Exception as e:
            logger.warning(f"Could not load align bridge weights: {e}")
        self.bridge.eval()
        _freeze_module(self.t_emb)
        _freeze_module(self.s_emb)
        self.bridge.freeze()

    def _batch_tokenize_and_pad(
        self, texts: List[str], tokenizer: TokenizerWrapper
    ) -> Tuple[mx.array, mx.array]:
        """Tokenizes a batch of texts and pads them to the same length."""
        # Tokenize all texts
        token_ids = [tokenizer.encode(s, add_special_tokens=False) or [] for s in texts]
        lengths = [len(ids) for ids in token_ids]

        # If all sequences are empty, return empty tensors
        if not any(lengths):
            return mx.array([], dtype=mx.int32), mx.array([], dtype=mx.bool)

        max_len = max(lengths)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        # Create padded array and attention mask
        padded_ids = mx.array(
            [ids + [pad_id] * (max_len - len(ids)) for ids in token_ids], dtype=mx.int32
        )
        attention_mask = mx.arange(max_len)[None, :] < mx.array(lengths)[:, None]

        return padded_ids, attention_mask

    def _pool_vec_batch(
        self, tok_emb: mx.array, attention_mask: mx.array, pool: str
    ) -> mx.array:
        """Performs pooling on a batch of token embeddings, respecting padding."""
        if tok_emb.size == 0:
            return mx.zeros((tok_emb.shape[0], tok_emb.shape[-1]), dtype=tok_emb.dtype)

        if pool == "last":
            # Find the index of the last non-padded token for each sequence
            sequence_lengths = attention_mask.sum(axis=1) - 1
            # Gather the embeddings at these indices
            pooled_vec = mx.take_along_axis(
                tok_emb, sequence_lengths[:, None, None], axis=1
            ).squeeze(1)
        else:  # Default to mean pooling
            # Mask out padding tokens by multiplying with the attention mask
            masked_emb = tok_emb * attention_mask[..., None].astype(tok_emb.dtype)
            # Sum embeddings and divide by the number of non-padded tokens
            sum_emb = masked_emb.sum(axis=1)
            num_tokens = attention_mask.sum(axis=1, keepdims=True)
            pooled_vec = sum_emb / mx.maximum(num_tokens, 1)

        return pooled_vec

    def __call__(self, texts: List[str]) -> List[float]:
        """
        Computes alignment scores for a batch of texts in a fully vectorized manner.
        """
        from mlx_rl_trainer.utils.text_utils import extract_answer_region

        if not texts:
            return []

        # Extract answer regions for all texts
        answers = [extract_answer_region(s or "", self.gen_cfg) for s in texts]

        # Batch tokenize and pad for both teacher and student models
        t_ids, t_mask = self._batch_tokenize_and_pad(answers, self.tok_t)
        s_ids, s_mask = self._batch_tokenize_and_pad(answers, self.tok_s)

        # Get embeddings for the entire batch
        t_emb_unpooled = self.t_emb(t_ids)
        s_emb_unpooled = self.s_emb(s_ids)

        # Pool the embeddings for the entire batch
        t_vecs = self._pool_vec_batch(t_emb_unpooled, t_mask, self.pool)
        s_vecs = self._pool_vec_batch(s_emb_unpooled, s_mask, self.pool)

        # Map teacher vectors to student space for the entire batch
        mapped_vecs = self.bridge(t_vecs)

        # Normalize vectors and compute cosine similarity for the entire batch
        a = mapped_vecs / (mx.linalg.norm(mapped_vecs, axis=-1, keepdims=True) + 1e-8)
        b = s_vecs / (mx.linalg.norm(s_vecs, axis=-1, keepdims=True) + 1e-8)
        cos_sim = mx.sum(a * b, axis=-1)

        # Scale and clip scores
        scores = 0.5 * (1.0 + cos_sim)
        final_scores = mx.clip(scores, 0.0, 1.0) * self.scale

        return final_scores.tolist()


_LAYER_PAT = re.compile(r"(?:^|[^a-zA-Z0-9_])layers\.(\d+)(?:[^0-9_]|$)")
_HEAD_PAT = re.compile(r"\blm_head\b", re.I)


def _find_layer_index(name: str) -> Optional[int]:
    m = _LAYER_PAT.search(name)
    if m:
        return int(m.group(1))
    parts = re.split(r"[\.\/]", name)
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except Exception:
                pass
    return None


def _band_for_name(
    name: str,
    low_band: Tuple[int, int],
    mid_band: Tuple[int, int],
    top_band: Tuple[int, int],
) -> str:
    li = _find_layer_index(name)

    def _in_range(layer_idx, band_range):
        if band_range is None or layer_idx is None:
            return False
        s, e = band_range
        return (s is None or layer_idx >= s) and (e is None or layer_idx <= e)

    if li is not None:
        if _in_range(li, low_band):
            return "low"
        if _in_range(li, mid_band):
            return "mid"
        if _in_range(li, top_band):
            return "top"
    if _HEAD_PAT.search(name):
        return "head"
    return "other"


def scale_grads_by_band(
    grads_tree: Dict[str, mx.array], config: ExperimentConfig
) -> Dict[str, mx.array]:
    t_cfg = config.trainer
    g_flat = tree_flatten(grads_tree)
    out = []
    for name, g in g_flat:
        if not isinstance(g, mx.array):
            out.append((name, g))
            continue
        band = _band_for_name(name, t_cfg.low_band, t_cfg.mid_band, t_cfg.top_band)
        mul = {
            "low": t_cfg.low_mul,
            "mid": t_cfg.mid_mul,
            "top": t_cfg.top_mul,
            "head": t_cfg.head_mul,
        }.get(band, 1.0)
        out.append((name, g * mul))
    return tree_unflatten(out)


# the common `{'model': ...}` nesting from mlx-lm.
def mask_grads_to_layer_band(
    grads_tree,
    start,
    end,
    *,
    include_embed=True,
    include_head=True,
    include_final_norm=True,
):
    # Check if the gradients are wrapped in a 'model' key and unwrap if necessary.
    # This makes the function robust to different model structures.
    is_wrapped = (
        isinstance(grads_tree, dict)
        and len(grads_tree) == 1
        and "model" in grads_tree
        and isinstance(grads_tree["model"], dict)
    )
    inner_grads_tree = grads_tree["model"] if is_wrapped else grads_tree

    if not isinstance(inner_grads_tree, dict):
        # Return an empty dict if the structure is unexpected to prevent errors
        return {}

    flat = tree_flatten(inner_grads_tree)
    kept = []
    for name, g in flat:
        if not isinstance(g, mx.array):
            kept.append((name, g))
            continue

        li = _find_layer_index(name)
        keep = False
        if li is not None:
            # Mask layers based on the start and end range
            keep = (start is None or li >= start) and (end is None or li <= end)
        else:
            # Handle non-layer parameters like embeddings and normalization layers
            lname = name.lower()
            if "embed" in lname or "embedding" in lname:
                keep = include_embed
            elif "norm" in lname:
                keep = include_final_norm
            elif "head" in lname:
                keep = include_head

        kept.append((name, g if keep else mx.zeros_like(g)))

    # Reconstruct the inner tree from the filtered list
    unflattened_tree = tree_unflatten(kept)

    # If the original was wrapped, re-wrap the result to maintain structure
    if is_wrapped:
        return {"model": unflattened_tree}
    else:
        return unflattened_tree


def mask_grads_to_specific_layers(
    grads_tree: Dict[str, mx.array], layer_indices: Set[int]
) -> Dict[str, mx.array]:
    flat = tree_flatten(grads_tree)
    kept = []
    for name, g in flat:
        if not isinstance(g, mx.array):
            kept.append((name, g))
            continue
        if (
            layer_idx := _find_layer_index(name)
        ) is not None and layer_idx in layer_indices:
            kept.append((name, g))
        else:
            kept.append((name, mx.zeros_like(g)))
    return tree_unflatten(kept)


def _global_grad_norm(grads: Dict[str, mx.array]) -> float:
    try:
        flat = [g for _, g in tree_flatten(grads) if isinstance(g, mx.array)]
        if not flat:
            return 0.0
        sq_sum = sum(mx.sum(g.astype(mx.float32) ** 2) for g in flat)
        total = mx.sqrt(sq_sum)
        mx.eval(total)
        return float(total.item())
    except Exception:
        return 0.0


def _maybe_clip_grad_norm(
    grads_tree: Dict[str, mx.array], max_norm: Optional[float]
) -> Tuple[Dict[str, mx.array], float]:
    if max_norm is None or max_norm <= 0:
        grad_norm = _global_grad_norm(grads_tree)
        return grads_tree, grad_norm
    try:
        clipped_grads, grad_norm_mx = optim.clip_grad_norm(grads_tree, float(max_norm))
        mx.eval(clipped_grads, grad_norm_mx)
        return clipped_grads, float(grad_norm_mx.item())
    except Exception as e:
        logger.warning(
            f"mlx.optim.clip_grad_norm failed: {e}. Falling back to manual clipping."
        )
        grad_norm = _global_grad_norm(grads_tree)
        if grad_norm > max_norm:
            scale = max_norm / (grad_norm + 1e-8)
            clipped_grads = tree_map(lambda g: g.astype(mx.float32) * scale, grads_tree)
            return clipped_grads, grad_norm
        return grads_tree, grad_norm


def metal_before_update(num_updates: int, config: ExperimentConfig):
    if not hasattr(config.generation, "_orig_max_gen_len"):
        setattr(config.generation, "_orig_max_gen_len", config.data.max_gen_len)
        setattr(config, "_orig_max_kv_size", config.max_kv_size)
        setattr(
            config.trainer,
            "_orig_num_rollout_samples",
            config.trainer.num_rollout_samples,
        )
    if num_updates < 32:
        config.data.max_gen_len = min(config.generation._orig_max_gen_len, 160)
        config.max_kv_size = min(config._orig_max_kv_size, 768)
        config.trainer.num_rollout_samples = min(
            config.trainer._orig_num_rollout_samples, 4
        )
    else:
        config.data.max_gen_len = config.generation._orig_max_gen_len
        config.max_kv_size = config._orig_max_kv_size
        config.trainer.num_rollout_samples = config.trainer._orig_num_rollout_samples
    if num_updates % 5 == 0:
        try:
            mx.synchronize()
        except Exception:
            pass
        mx.clear_cache()
        gc.collect()


def _create_4d_attention_mask(
    tokens: mx.array, pad_token_id: int, dtype: mx.Dtype = TARGET_FLOAT_DTYPE
) -> mx.array:
    if tokens.ndim != 2:
        raise ValueError(f"tokens must be 2D, got {tokens.shape}")
    B, T = tokens.shape
    causal_mask = nn.MultiHeadAttention.create_additive_causal_mask(T, dtype=dtype)
    padding_mask = (tokens == pad_token_id)[:, None, None, :]
    neg_inf = mx.array(-1e9, dtype=dtype)
    return mx.minimum(causal_mask, mx.where(padding_mask, neg_inf, 0.0))


def safe_make_sampler(
    config_or_args: Union[ExperimentConfig, GenerationConfig, EnhancedGenerationConfig],
    temp: Optional[float] = None,
    tokenizer: Optional[TokenizerWrapper] = None,
    phase: str = "global",
    **overrides
) -> Callable:
    """
    Enhanced sampler creation with comprehensive configuration support.
    
    This function bridges the legacy interface with our new enhanced sampling
    configuration system, providing backward compatibility while enabling
    advanced per-phase sampling capabilities.
    
    Args:
        config_or_args: Configuration object (legacy or enhanced)
        temp: Legacy temperature parameter (deprecated, use config instead)
        tokenizer: Tokenizer for special token handling
        phase: Sampling phase (think/answer/global)
        **overrides: Runtime parameter overrides
        
    Returns:
        MLX sampler function
        
    Raises:
        ValueError: If configuration is invalid
        ImportError: If enhanced sampling system is not available
        
    Example:
        >>> config = EnhancedGenerationConfig(...)
        >>> sampler = safe_make_sampler(config, phase="think", tokenizer=tokenizer)
        >>> token = sampler(logits)
    """
    correlation_id = str(uuid.uuid4())
    
    try:
        # Use enhanced sampling system if available
        if ENHANCED_SAMPLING_AVAILABLE:
            return _create_enhanced_sampler(
                config_or_args, temp, tokenizer, phase, correlation_id, **overrides
            )
        else:
            # Fallback to legacy implementation
            return _create_legacy_sampler(
                config_or_args, temp, tokenizer, correlation_id, **overrides
            )
            
    except Exception as e:
        logger.error(
            f"Sampler creation failed: {e}",
            extra={
                "correlation_id": correlation_id,
                "phase": phase,
                "config_type": type(config_or_args).__name__,
                "error_type": type(e).__name__
            }
        )
        
        # Ultimate fallback: create basic sampler
        try:
            return _create_fallback_sampler(temp or 0.7, tokenizer)
        except Exception as fallback_error:
            logger.critical(f"Fallback sampler creation failed: {fallback_error}")
            raise ValueError(f"Cannot create sampler: {e}") from e


def _create_enhanced_sampler(
    config_or_args: Union[ExperimentConfig, GenerationConfig, EnhancedGenerationConfig],
    temp: Optional[float],
    tokenizer: Optional[TokenizerWrapper],
    phase: str,
    correlation_id: str,
    **overrides
) -> Callable:
    """Create sampler using enhanced configuration system."""
    try:
        # Import here to avoid circular imports
        from mlx_rl_trainer.generation.samplers.factory import create_sampler
        
        # Handle legacy temperature override
        if temp is not None:
            overrides["temperature"] = temp
            logger.warning(
                f"Using legacy temperature parameter: {temp}. "
                "Consider using enhanced configuration instead.",
                extra={"correlation_id": correlation_id}
            )
        
        # Create sampler using our factory
        sampler_instance = create_sampler(
            config=config_or_args,
            phase=phase,
            tokenizer=tokenizer,
            **overrides
        )
        
        # If it's our adapter, get the underlying MLX sampler
        if hasattr(sampler_instance, 'get_mlx_sampler'):
            return sampler_instance.get_mlx_sampler()
        
        # If it's already an MLX sampler, return as-is
        return sampler_instance
        
    except ImportError as e:
        logger.warning(f"Enhanced sampler creation failed, falling back: {e}")
        return _create_legacy_sampler(config_or_args, temp, tokenizer, correlation_id, **overrides)


def _create_legacy_sampler(
    config_or_args: Union[ExperimentConfig, GenerationConfig, Any],
    temp: Optional[float],
    tokenizer: Optional[TokenizerWrapper],
    correlation_id: str,
    **overrides
) -> Callable:
    """Create sampler using legacy configuration system."""
    gen_cfg = (
        config_or_args.generation
        if isinstance(config_or_args, ExperimentConfig)
        else config_or_args
    )
    
    # Determine temperature
    if temp is not None:
        temperature = temp
    elif hasattr(gen_cfg, 'think_temperature'):
        temperature = gen_cfg.think_temperature
    else:
        temperature = 0.7  # Default fallback
    
    # Extract other parameters from config if available
    top_p = getattr(gen_cfg, 'sampling_top_p', 0.9)
    top_k = getattr(gen_cfg, 'sampling_top_k', 0)
    min_p = getattr(gen_cfg, 'sampling_min_p', 0.0)
    repetition_penalty = getattr(gen_cfg, 'repetition_penalty', 1.0)
    
    # Apply overrides
    temperature = overrides.get('temperature', temperature)
    top_p = overrides.get('top_p', top_p)
    top_k = overrides.get('top_k', top_k)
    min_p = overrides.get('min_p', min_p)
    
    try:
        # Try to create with all parameters
        sampler_params = {
            'temp': temperature,
            'top_p': top_p,
            'min_p': min_p,
            'min_tokens_to_keep': overrides.get('min_tokens_to_keep', 1),
            'top_k': top_k,
            'xtc_probability': overrides.get('xtc_probability', 0.0),
            'xtc_threshold': overrides.get('xtc_threshold', 0.0),
        }
        
        # Add special tokens if tokenizer available
        if tokenizer is not None:
            try:
                xtc_special_tokens = []
                if hasattr(tokenizer, 'encode') and callable(getattr(tokenizer, 'encode', None)):
                    try:
                        newline_tokens = tokenizer.encode("\n", add_special_tokens=False)
                        if newline_tokens and hasattr(newline_tokens, '__iter__'):
                            xtc_special_tokens.extend(newline_tokens)
                    except Exception as e:
                        logger.warning(f"Failed to encode newline tokens: {e}")
                
                if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
                    xtc_special_tokens.append(tokenizer.eos_token_id)
                
                sampler_params['xtc_special_tokens'] = xtc_special_tokens
            except Exception as e:
                logger.warning(f"Failed to extract special tokens: {e}")
        
        return make_sampler(**sampler_params)
        
    except TypeError as e:
        # Fallback for older MLX versions
        logger.warning(f"Full parameter sampler failed, using basic version: {e}")
        return make_sampler(temp=temperature, top_p=top_p)


def _create_fallback_sampler(temperature: float, tokenizer: Optional[TokenizerWrapper]) -> Callable:
    """Create basic fallback sampler when all else fails."""
    try:
        return make_sampler(temp=temperature, top_p=0.9)
    except Exception:
        # Ultimate fallback: return identity function
        logger.critical("Creating identity sampler as ultimate fallback")
        return lambda logits: mx.argmax(logits, axis=-1)


def _first_token_ids_for_lexemes(
    tokenizer: TokenizerWrapper, lexemes: Sequence[str]
) -> List[int]:
    ids: List[int] = []
    for lx in lexemes:
        if (
            (t := tokenizer.encode(lx, add_special_tokens=False))
            and t
            and t[0] not in ids
        ):
            ids.append(t[0])
        if (
            (t_space := tokenizer.encode(" " + lx, add_special_tokens=False))
            and t_space
            and t_space[0] not in ids
        ):
            ids.append(t_space[0])
    return ids


def _letter_token_ids(
    tokenizer: TokenizerWrapper, letters: Sequence[str] = LETTER_ALPH
) -> Dict[str, List[int]]:
    out = {}
    for L in letters:
        cand = []
        for suf in ["", " ", ")", ".", " )", " ."]:
            ids = tokenizer.encode(L + suf, add_special_tokens=False)
            if len(ids) == 1 and ids[0] not in cand:
                cand.append(ids[0])
        out[L] = cand
    return out


def _resolve_tag_ids(
    tokenizer: TokenizerWrapper, gen_config: GenerationConfig
) -> Dict[str, Optional[int]]:
    def _one_id(tok_str):
        if not tok_str:
            return None
        try:
            # Handle different tokenizer types
            if hasattr(tokenizer, 'encode') and callable(tokenizer.encode):
                ids = tokenizer.encode(tok_str, add_special_tokens=False)
            elif hasattr(tokenizer, '__call__'):
                ids = tokenizer(tok_str, add_special_tokens=False)
            else:
                logger.warning(f"Tokenizer type not supported: {type(tokenizer)}")
                return None
            return int(ids[0]) if len(ids) == 1 else None
        except Exception:
            return None

    return {
        "think_start": _one_id(gen_config.think_start_tag),
        "think_end": _one_id(gen_config.think_end_tag),
        "answer_start": _one_id(gen_config.answer_start_tag),
        "answer_end": _one_id(gen_config.answer_end_tag),
        "eos": tokenizer.eos_token_id,
    }


def make_dynamic_tag_bias_processor(
    tokenizer: TokenizerWrapper, config: ExperimentConfig, mcq_flags: List[bool]
) -> Callable:
    gen_cfg = config.generation
    tag_ids = _resolve_tag_ids(tokenizer, gen_cfg)
    mcq_letter_ids = sorted(set(sum(_letter_token_ids(tokenizer).values(), [])))
    ban_ids = _first_token_ids_for_lexemes(tokenizer, gen_cfg.ban_phrases_for_bias)
    encourage_ids = _first_token_ids_for_lexemes(
        tokenizer, gen_cfg.encourage_phrases_for_bias
    )
    tool_ids = _first_token_ids_for_lexemes(tokenizer, _TOOL_LIKE_MARKERS)

    te, ts, as_id, ae, eos_tok = (
        tag_ids.get(k)
        for k in ("think_end", "think_start", "answer_start", "answer_end", "eos")
    )
    B_CLOSE, B_AS, P_REOPEN_THINK, P_EXTRA_TE, P_REOPEN_ANS, B_EOS_ANS = (
        gen_cfg.bias_close_think,
        gen_cfg.bias_answer_start,
        gen_cfg.punish_reopen_think,
        gen_cfg.punish_extra_think_end,
        gen_cfg.punish_reopen_answer,
        gen_cfg.bias_eos_after_answer,
    )
    MIN_ANS, MIN_ANS_MCQ, HARD_MASK, LIFT_MCQ, BAN_MCQ, BAN_NONMCQ = (
        gen_cfg.min_answer_tokens,
        gen_cfg.min_answer_tokens_mcq,
        gen_cfg.hard_mask_mcq_first_token,
        gen_cfg.mcq_letter_lift,
        gen_cfg.mcq_ban_first_bias,
        gen_cfg.nonmcq_ban_first_bias,
    )
    MCQ_CLOSE_K, B_MCQ_CLOSE, MIN_THINK, B_END_EARLY, B_AS_MIN_THINK = (
        gen_cfg.mcq_close_after_k,
        gen_cfg.mcq_answer_end_bias,
        gen_cfg.min_think_tokens,
        gen_cfg.think_end_early_bias,
        gen_cfg.bias_answer_start_after_min_think,
    )
    B_ENCOURAGE, P_TOOL = (
        gen_cfg.encourage_think_bias,
        gen_cfg.tool_call_penalty * -10.0,
    )

    def _proc_vectorized(hist_list: List[List[int]], logits: mx.array) -> mx.array:
        if logits.ndim != 2:
            return logits
        B, V = logits.shape
        neg_inf, pad_id = mx.array(-1e9, dtype=logits.dtype), tokenizer.pad_token_id
        max_hist_len = max(len(row) for row in hist_list) if hist_list else 0
        if max_hist_len == 0:
            return logits

        history_mx = mx.array(
            [row + [pad_id] * (max_hist_len - len(row)) for row in hist_list],
            dtype=mx.int32,
        )
        if tool_ids and P_TOOL < 0:
            logits = logits.at[:, tool_ids].add(P_TOOL)

        def find_last_pos_mx(tag_id):
            if tag_id is None:
                return mx.full((B,), -1, dtype=mx.int32)
            matches = history_mx == tag_id
            rev_indices = mx.argmax(matches[:, ::-1], axis=1).astype(mx.int32)
            return mx.where(mx.any(matches, axis=1), max_hist_len - 1 - rev_indices, -1)

        last_ts, last_te, last_as, last_ae = (
            find_last_pos_mx(t) for t in (ts, te, as_id, ae)
        )
        history_len_mx = mx.array([len(row) for row in hist_list], dtype=mx.int32)

        # Use element-wise operations for compatibility
        inside_think = ((last_ts != -1).astype(mx.int32) *
                       (last_te < last_ts).astype(mx.int32) *
                       (last_as < last_ts).astype(mx.int32)) > 0
        inside_answer = ((last_as != -1).astype(mx.int32) *
                        (last_ae < last_as).astype(mx.int32)) > 0
        ae_seen = last_ae != -1
        k_think = mx.where(inside_think, history_len_mx - (last_ts + 1), 0)
        k_answer = mx.where(inside_answer, history_len_mx - (last_as + 1), 0)
        # Handle MLX version compatibility for boolean dtype
        try:
            is_mcq_mask = mx.array(mcq_flags, dtype=mx.bool_)
        except AttributeError:
            # Fallback for older MLX versions - use int32 and convert to bool
            is_mcq_mask = mx.array([int(flag) for flag in mcq_flags], dtype=mx.int32)
            is_mcq_mask = is_mcq_mask.astype(mx.bool_) if hasattr(mx, 'bool_') else is_mcq_mask > 0

        if ts is not None and te is not None:
            logits = logits.at[:, ts].add(mx.where(last_te != -1, P_REOPEN_THINK, 0.0))
            if as_id is not None:
                logits = logits.at[:, as_id].add(
                    mx.where(last_ae > last_as, P_REOPEN_ANS, 0.0)
                )
            matches_te = (history_mx == te).astype(mx.int32)
            te_count = mx.sum(matches_te, axis=1)
            bias_at_te = mx.where(te_count == 0, B_CLOSE, P_EXTRA_TE)
            min_think_penalty_mask = mx.logical_and(inside_think, (k_think < MIN_THINK))
            bias_at_te = mx.where(min_think_penalty_mask, B_END_EARLY, bias_at_te)
            logits = logits.at[:, te].add(bias_at_te)
            can_start_answer = mx.logical_and(
                last_te > last_as, mx.logical_not(inside_answer)
            )
            min_think_ok = mx.array([not B_AS_MIN_THINK] * B, dtype=mx.bool_) if hasattr(mx, 'bool_') else mx.array([1 - int(B_AS_MIN_THINK)] * B, dtype=mx.int32) > 0
            if B_AS_MIN_THINK:
                min_think_ok = k_think >= MIN_THINK
            can_start_answer = mx.logical_and(can_start_answer, min_think_ok)
            if as_id is not None:
                logits = logits.at[:, as_id].add(mx.where(can_start_answer, B_AS, 0.0))

        if eos_tok is not None:
            logits = logits.at[:, eos_tok].add(mx.where(ae_seen, B_EOS_ANS, 0.0))

        if encourage_ids and B_ENCOURAGE > 0 and mx.any(inside_think).item():
            encourage_bias = mx.zeros_like(logits)
            encourage_bias = encourage_bias.at[:, encourage_ids].add(B_ENCOURAGE)
            logits = logits + (encourage_bias * inside_think[:, None])

        mcq_first_token_mask = mx.logical_and(
            is_mcq_mask, (inside_answer.astype(mx.int32) * (k_answer == 0).astype(mx.int32)) > 0
        )
        if mx.any(mcq_first_token_mask).item() and HARD_MASK:
            mcq_allowed_logits = mx.full((V,), neg_inf, dtype=logits.dtype)
            if mcq_letter_ids:
                mcq_allowed_logits = mcq_allowed_logits.at[mcq_letter_ids].add(LIFT_MCQ)
            if ban_ids:
                mcq_allowed_logits = mcq_allowed_logits.at[ban_ids].add(BAN_MCQ)
            logits = mx.where(
                mcq_first_token_mask[:, None], mcq_allowed_logits[None, :], logits
            )

        non_mcq_first_answer = mx.logical_and(
            (1 - is_mcq_mask.astype(mx.int32)) > 0, (inside_answer.astype(mx.int32) * (k_answer == 0).astype(mx.int32)) > 0
        )

        if ban_ids and BAN_NONMCQ != 0 and mx.any(non_mcq_first_answer).item():
            ban_bias = mx.zeros_like(logits)
            ban_bias = ban_bias.at[:, ban_ids].add(BAN_NONMCQ)
            logits = logits + (ban_bias * non_mcq_first_answer[:, None])

        if ae is not None:
            min_ans_len = mx.where(is_mcq_mask, MIN_ANS_MCQ, MIN_ANS)
            min_len_penalty_mask = mx.logical_and(
                inside_answer, (k_answer < min_ans_len)
            )
            logits = logits.at[:, ae].add(mx.where(min_len_penalty_mask, -8.0, 0.0))
            mcq_close_mask = mx.logical_and(
                is_mcq_mask, mx.logical_and(inside_answer, (k_answer >= MCQ_CLOSE_K))
            )
            logits = logits.at[:, ae].add(mx.where(mcq_close_mask, B_MCQ_CLOSE, 0.0))

        return logits


def make_enhanced_dynamic_tag_bias_processor(
    tokenizer: TokenizerWrapper, config: ExperimentConfig, mcq_flags: List[bool]
) -> Callable:
    """
    Enhanced dynamic tag bias processor with multi-token support and phase awareness.
    
    This function creates a sophisticated bias processor that integrates with our new
    enhanced processor architecture while maintaining backward compatibility with the
    existing MLX RL trainer system. It provides advanced bias processing capabilities
    including multi-token phrase support, phase-aware processing, and comprehensive
    error handling.
    
    Key Enhancements:
    - Integration with new processor pipeline architecture
    - Multi-token phrase support beyond first-token matching
    - Phase-aware bias application (thinking vs. answer phases)
    - Enhanced case-insensitive processing
    - Comprehensive error handling and fallback mechanisms
    - Performance optimization with caching and vectorization
    - Detailed monitoring and observability
    
    Args:
        tokenizer: MLX tokenizer wrapper for token operations
        config: Experiment configuration with generation parameters
        mcq_flags: Boolean flags indicating MCQ status per batch item
        
    Returns:
        Callable processor function compatible with MLX generation pipeline
        
    Raises:
        ValueError: If configuration is invalid
        ImportError: If enhanced processor system is not available
        
    Example:
        >>> processor = make_enhanced_dynamic_tag_bias_processor(tokenizer, config, mcq_flags)
        >>> modified_logits = processor(history, logits)
    """
    gen_cfg = config.generation
    correlation_id = str(uuid.uuid4())
    
    # Try to use enhanced processor system if available
    try:
        return _create_enhanced_dynamic_processor(tokenizer, config, mcq_flags, correlation_id)
    except Exception as e:
        logger.warning(
            f"Enhanced processor creation failed, falling back to legacy: {e}",
            extra={'correlation_id': correlation_id}
        )
        return _create_legacy_dynamic_processor_enhanced(tokenizer, config, mcq_flags, correlation_id)


def _create_enhanced_dynamic_processor(
    tokenizer: TokenizerWrapper, 
    config: ExperimentConfig, 
    mcq_flags: List[bool],
    correlation_id: str
) -> Callable:
    """
    Create enhanced dynamic processor using new architecture.
    
    This function leverages our new processor pipeline system to create
    a sophisticated bias processor with advanced capabilities.
    """
    try:
        # Import enhanced processor components
        from mlx_rl_trainer.generation.processors.pipeline import ProcessorPipeline, create_fault_tolerant_pipeline
        from mlx_rl_trainer.generation.processors.enhanced_bias_processor import (
            EnhancedBiasProcessor, create_phase_aware_processor
        )
        from mlx_rl_trainer.generation.processors.phase_processor import PhaseDetector
        from mlx_rl_trainer.generation.processors.base import (
            ProcessingContext, ProcessingPhase, create_processing_context
        )
        
        # Create phase detector
        detector = PhaseDetector(
            max_history_window=50,
            enable_caching=True,
            enable_monitoring=True
        )
        
        # Create enhanced bias processor
        bias_processor = create_phase_aware_processor(
            think_close_bias=getattr(config.generation, 'bias_close_think', 0.0),
            answer_start_bias=getattr(config.generation, 'bias_answer_start', 0.0),
            ban_phrases=getattr(config.generation, 'ban_phrases_for_bias', []),
            encourage_phrases=getattr(config.generation, 'encourage_phrases_for_bias', []),
            detector=detector,
            case_sensitive=False,
            enable_caching=True
        )
        
        # Create fault-tolerant pipeline
        pipeline = create_fault_tolerant_pipeline(
            processors=[bias_processor],
            failure_threshold=3,
            recovery_timeout=30.0
        )
        
        # Create wrapper function that adapts to legacy interface
        def enhanced_processor_wrapper(hist_list: List[List[int]], logits: mx.array) -> mx.array:
            """Enhanced processor wrapper with legacy interface compatibility."""
            try:
                # Create processing context
                context = create_processing_context(
                    config=config.generation,
                    tokenizer=tokenizer,
                    mcq_flags=mcq_flags,
                    batch_size=len(hist_list),
                    vocabulary_size=logits.shape[-1] if logits.ndim > 1 else len(logits),
                    correlation_id=correlation_id
                )
                
                # Process through pipeline
                return pipeline.process_logits(logits, hist_list, context)
                
            except Exception as e:
                logger.error(
                    f"Enhanced processor failed, using fallback: {e}",
                    extra={'correlation_id': correlation_id}
                )
                # Fallback to legacy processor
                return _create_legacy_dynamic_processor_enhanced(
                    tokenizer, config, mcq_flags, correlation_id
                )(hist_list, logits)
        
        logger.info(
            f"Created enhanced dynamic processor with pipeline",
            extra={'correlation_id': correlation_id}
        )
        
        return enhanced_processor_wrapper
        
    except ImportError as e:
        logger.warning(f"Enhanced processor components not available: {e}")
        raise


def _create_legacy_dynamic_processor_enhanced(
    tokenizer: TokenizerWrapper, 
    config: ExperimentConfig, 
    mcq_flags: List[bool],
    correlation_id: str
) -> Callable:
    """
    Create legacy dynamic processor with enhanced error handling.
    
    This function provides the original processor implementation with
    improved error handling, logging, and performance monitoring.
    """
    gen_cfg = config.generation
    
    # Enhanced tag ID resolution with error handling
    tag_ids = _resolve_tag_ids_enhanced(tokenizer, gen_cfg, correlation_id)
    
    # Enhanced token ID extraction with caching
    mcq_letter_ids = _get_mcq_letter_ids_cached(tokenizer, correlation_id)
    ban_ids = _get_phrase_token_ids_cached(tokenizer, gen_cfg.ban_phrases_for_bias, correlation_id)
    encourage_ids = _get_phrase_token_ids_cached(tokenizer, gen_cfg.encourage_phrases_for_bias, correlation_id)
    tool_ids = _get_phrase_token_ids_cached(tokenizer, _TOOL_LIKE_MARKERS, correlation_id)

    te, ts, as_id, ae, eos_tok = (
        tag_ids.get(k)
        for k in ("think_end", "think_start", "answer_start", "answer_end", "eos")
    )
    
    # Extract bias parameters with enhanced defaults
    B_CLOSE, B_AS, P_REOPEN_THINK, P_EXTRA_TE, P_REOPEN_ANS, B_EOS_ANS = (
        getattr(gen_cfg, 'bias_close_think', 0.0),
        getattr(gen_cfg, 'bias_answer_start', 0.0),
        getattr(gen_cfg, 'punish_reopen_think', 0.0),
        getattr(gen_cfg, 'punish_extra_think_end', 0.0),
        getattr(gen_cfg, 'punish_reopen_answer', 0.0),
        getattr(gen_cfg, 'bias_eos_after_answer', 0.0),
    )
    MIN_ANS, MIN_ANS_MCQ, HARD_MASK, LIFT_MCQ, BAN_MCQ, BAN_NONMCQ = (
        getattr(gen_cfg, 'min_answer_tokens', 1),
        getattr(gen_cfg, 'min_answer_tokens_mcq', 1),
        getattr(gen_cfg, 'hard_mask_mcq_first_token', False),
        getattr(gen_cfg, 'mcq_letter_lift', 0.0),
        getattr(gen_cfg, 'mcq_ban_first_bias', 0.0),
        getattr(gen_cfg, 'nonmcq_ban_first_bias', 0.0),
    )
    MCQ_CLOSE_K, B_MCQ_CLOSE, MIN_THINK, B_END_EARLY, B_AS_MIN_THINK = (
        getattr(gen_cfg, 'mcq_close_after_k', 5),
        getattr(gen_cfg, 'mcq_answer_end_bias', 0.0),
        getattr(gen_cfg, 'min_think_tokens', 10),
        getattr(gen_cfg, 'think_end_early_bias', 0.0),
        getattr(gen_cfg, 'bias_answer_start_after_min_think', False),
    )
    B_ENCOURAGE, P_TOOL = (
        getattr(gen_cfg, 'encourage_think_bias', 0.0),
        getattr(gen_cfg, 'tool_call_penalty', 0.0) * -10.0,
    )

    def _proc_vectorized_enhanced(hist_list: List[List[int]], logits: mx.array) -> mx.array:
        """Enhanced vectorized processor with comprehensive error handling and monitoring."""
        start_time = time.time()
        
        try:
            # Input validation
            if not isinstance(hist_list, list) or not isinstance(logits, mx.array):
                raise ValueError("Invalid input types for processor")
            
            if logits.ndim != 2:
                logger.warning(
                    f"Expected 2D logits, got {logits.ndim}D",
                    extra={'correlation_id': correlation_id}
                )
                return logits
                
            B, V = logits.shape
            
            # Validate batch consistency
            if len(hist_list) != B:
                raise ValueError(
                    f"History length ({len(hist_list)}) doesn't match batch size ({B})"
                )
            
            # Enhanced processing with the original logic
            result = _process_logits_vectorized(
                hist_list, logits, tokenizer, tag_ids, mcq_letter_ids, ban_ids, 
                encourage_ids, tool_ids, mcq_flags, B_CLOSE, B_AS, P_REOPEN_THINK,
                P_EXTRA_TE, P_REOPEN_ANS, B_EOS_ANS, MIN_ANS, MIN_ANS_MCQ,
                HARD_MASK, LIFT_MCQ, BAN_MCQ, BAN_NONMCQ, MCQ_CLOSE_K,
                B_MCQ_CLOSE, MIN_THINK, B_END_EARLY, B_AS_MIN_THINK,
                B_ENCOURAGE, P_TOOL, correlation_id
            )
            
            # Performance monitoring
            processing_time = (time.time() - start_time) * 1000
            logger.debug(
                f"Legacy processor completed in {processing_time:.2f}ms",
                extra={'correlation_id': correlation_id, 'batch_size': B}
            )
            
            return result
            
        except Exception as e:
            logger.error(
                f"Legacy processor failed: {e}",
                extra={'correlation_id': correlation_id},
                exc_info=True
            )
            # Return unmodified logits as fallback
            return logits

    return _proc_vectorized_enhanced


def _process_logits_vectorized(
    hist_list: List[List[int]], 
    logits: mx.array,
    tokenizer: TokenizerWrapper,
    tag_ids: Dict[str, Optional[int]],
    mcq_letter_ids: List[int],
    ban_ids: List[int],
    encourage_ids: List[int],
    tool_ids: List[int],
    mcq_flags: List[bool],
    B_CLOSE: float, B_AS: float, P_REOPEN_THINK: float,
    P_EXTRA_TE: float, P_REOPEN_ANS: float, B_EOS_ANS: float,
    MIN_ANS: int, MIN_ANS_MCQ: int, HARD_MASK: bool,
    LIFT_MCQ: float, BAN_MCQ: float, BAN_NONMCQ: float,
    MCQ_CLOSE_K: int, B_MCQ_CLOSE: float, MIN_THINK: int,
    B_END_EARLY: float, B_AS_MIN_THINK: bool, B_ENCOURAGE: float,
    P_TOOL: float, correlation_id: str
) -> mx.array:
    """
    Enhanced vectorized logit processing with comprehensive bias application.
    
    This function implements the core bias processing logic with enhanced
    error handling, performance monitoring, and extensibility.
    """
    B, V = logits.shape
    neg_inf, pad_id = mx.array(-1e9, dtype=logits.dtype), tokenizer.pad_token_id
    max_hist_len = max(len(row) for row in hist_list) if hist_list else 0
    
    if max_hist_len == 0:
        return logits

    # Create padded history matrix
    history_mx = mx.array(
        [row + [pad_id] * (max_hist_len - len(row)) for row in hist_list],
        dtype=mx.int32,
    )
    
    # Extract tag IDs
    te, ts, as_id, ae, eos_tok = (
        tag_ids.get(k) for k in ("think_end", "think_start", "answer_start", "answer_end", "eos")
    )
    
    # Apply tool call penalties
    if tool_ids and P_TOOL < 0:
        logits = logits.at[:, tool_ids].add(P_TOOL)

    def find_last_pos_mx(tag_id):
        """Enhanced tag position finding with error handling."""
        if tag_id is None:
            return mx.full((B,), -1, dtype=mx.int32)
        try:
            matches = history_mx == tag_id
            rev_indices = mx.argmax(matches[:, ::-1], axis=1).astype(mx.int32)
            return mx.where(mx.any(matches, axis=1), max_hist_len - 1 - rev_indices, -1)
        except Exception as e:
            logger.warning(f"Tag position finding failed for tag {tag_id}: {e}")
            return mx.full((B,), -1, dtype=mx.int32)

    # Find tag positions
    last_ts, last_te, last_as, last_ae = (
        find_last_pos_mx(t) for t in (ts, te, as_id, ae)
    )
    history_len_mx = mx.array([len(row) for row in hist_list], dtype=mx.int32)

    # Determine current phase for each batch item
    inside_think = mx.logical_and(
        last_ts != -1, mx.logical_and(last_te < last_ts, last_as < last_ts)
    )
    inside_answer = mx.logical_and(last_as != -1, last_ae < last_as)
    ae_seen = last_ae != -1
    k_think = mx.where(inside_think, history_len_mx - (last_ts + 1), 0)
    k_answer = mx.where(inside_answer, history_len_mx - (last_as + 1), 0)
    # Handle MLX version compatibility for boolean dtype
    try:
        is_mcq_mask = mx.array(mcq_flags, dtype=mx.bool_)
    except AttributeError:
        # Fallback for older MLX versions - use int32 and convert to bool
        is_mcq_mask = mx.array([int(flag) for flag in mcq_flags], dtype=mx.int32)
        is_mcq_mask = is_mcq_mask.astype(mx.bool_) if hasattr(mx, 'bool_') else is_mcq_mask > 0

    # Apply think/answer tag biases
    if ts is not None and te is not None:
        # Punish reopening think after it's closed
        logits = logits.at[:, ts].add(mx.where(last_te != -1, P_REOPEN_THINK, 0.0))
        
        # Punish reopening answer after it's closed
        if as_id is not None:
            logits = logits.at[:, as_id].add(
                mx.where(last_ae > last_as, P_REOPEN_ANS, 0.0)
            )
        
        # Apply think end bias with early penalty
        matches_te = (history_mx == te).astype(mx.int32)
        te_count = mx.sum(matches_te, axis=1)
        bias_at_te = mx.where(te_count == 0, B_CLOSE, P_EXTRA_TE)
        min_think_penalty_mask = mx.logical_and(inside_think, (k_think < MIN_THINK))
        bias_at_te = mx.where(min_think_penalty_mask, B_END_EARLY, bias_at_te)
        logits = logits.at[:, te].add(bias_at_te)
        
        # Apply answer start bias
        can_start_answer = mx.logical_and(
            last_te > last_as, mx.logical_not(inside_answer)
        )
        min_think_ok = mx.array([not B_AS_MIN_THINK] * B, dtype=mx.bool_) if hasattr(mx, 'bool_') else mx.array([1 - int(B_AS_MIN_THINK)] * B, dtype=mx.int32) > 0
        if B_AS_MIN_THINK:
            min_think_ok = k_think >= MIN_THINK
        can_start_answer = mx.logical_and(can_start_answer, min_think_ok)
        if as_id is not None:
            logits = logits.at[:, as_id].add(mx.where(can_start_answer, B_AS, 0.0))

    # Apply EOS bias after answer
    if eos_tok is not None:
        logits = logits.at[:, eos_tok].add(mx.where(ae_seen, B_EOS_ANS, 0.0))

    # Apply encourage bias during thinking
    if encourage_ids and B_ENCOURAGE > 0 and mx.any(inside_think).item():
        encourage_bias = mx.zeros_like(logits)
        encourage_bias = encourage_bias.at[:, encourage_ids].add(B_ENCOURAGE)
        logits = logits + (encourage_bias * inside_think[:, None])

    # Apply MCQ first token masking
    mcq_first_token_mask = mx.logical_and(
        is_mcq_mask, (inside_answer.astype(mx.int32) * (k_answer == 0).astype(mx.int32)) > 0
    )
    if mx.any(mcq_first_token_mask).item() and HARD_MASK:
        mcq_allowed_logits = mx.full((V,), neg_inf, dtype=logits.dtype)
        if mcq_letter_ids:
            mcq_allowed_logits = mcq_allowed_logits.at[mcq_letter_ids].add(LIFT_MCQ)
        if ban_ids:
            mcq_allowed_logits = mcq_allowed_logits.at[ban_ids].add(BAN_MCQ)
        logits = mx.where(
            mcq_first_token_mask[:, None], mcq_allowed_logits[None, :], logits
        )

    # Apply non-MCQ first answer bias
    non_mcq_first_answer = mx.logical_and(
        (1 - is_mcq_mask.astype(mx.int32)) > 0, (inside_answer.astype(mx.int32) * (k_answer == 0).astype(mx.int32)) > 0
    )
    if ban_ids and BAN_NONMCQ != 0 and mx.any(non_mcq_first_answer).item():
        ban_bias = mx.zeros_like(logits)
        ban_bias = ban_bias.at[:, ban_ids].add(BAN_NONMCQ)
        logits = logits + (ban_bias * non_mcq_first_answer[:, None])

    # Apply answer end constraints
    if ae is not None:
        min_ans_len = mx.where(is_mcq_mask, MIN_ANS_MCQ, MIN_ANS)
        min_len_penalty_mask = mx.logical_and(
            inside_answer, (k_answer < min_ans_len)
        )
        logits = logits.at[:, ae].add(mx.where(min_len_penalty_mask, -8.0, 0.0))
        mcq_close_mask = mx.logical_and(
            is_mcq_mask, mx.logical_and(inside_answer, (k_answer >= MCQ_CLOSE_K))
        )
        logits = logits.at[:, ae].add(mx.where(mcq_close_mask, B_MCQ_CLOSE, 0.0))

    return logits


# Enhanced helper functions with caching and error handling

def _resolve_tag_ids_enhanced(
    tokenizer: TokenizerWrapper, 
    gen_config: GenerationConfig, 
    correlation_id: str
) -> Dict[str, Optional[int]]:
    """Enhanced tag ID resolution with comprehensive error handling."""
    def _safe_encode_single_token(tok_str: str) -> Optional[int]:
        """Safely encode a string to a single token ID."""
        if not tok_str:
            return None
        try:
            ids = tokenizer.encode(tok_str, add_special_tokens=False)
            if len(ids) == 1:
                return int(ids[0])
            elif len(ids) > 1:
                logger.warning(
                    f"Tag '{tok_str}' encodes to multiple tokens: {ids}. Using first token.",
                    extra={'correlation_id': correlation_id}
                )
                return int(ids[0])
            else:
                logger.warning(
                    f"Tag '{tok_str}' encodes to no tokens",
                    extra={'correlation_id': correlation_id}
                )
                return None
        except Exception as e:
            logger.error(
                f"Failed to encode tag '{tok_str}': {e}",
                extra={'correlation_id': correlation_id}
            )
            return None

    try:
        return {
            "think_start": _safe_encode_single_token(getattr(gen_config, 'think_start_tag', '')),
            "think_end": _safe_encode_single_token(getattr(gen_config, 'think_end_tag', '')),
            "answer_start": _safe_encode_single_token(getattr(gen_config, 'answer_start_tag', '')),
            "answer_end": _safe_encode_single_token(getattr(gen_config, 'answer_end_tag', '')),
            "eos": getattr(tokenizer, 'eos_token_id', None),
        }
    except Exception as e:
        logger.error(
            f"Tag ID resolution failed: {e}",
            extra={'correlation_id': correlation_id}
        )
        return {
            "think_start": None,
            "think_end": None,
            "answer_start": None,
            "answer_end": None,
            "eos": getattr(tokenizer, 'eos_token_id', None),
        }


def _get_mcq_letter_ids_cached(tokenizer: TokenizerWrapper, correlation_id: str) -> List[int]:
    """Enhanced MCQ letter ID extraction with caching."""
    try:
        letter_dict = _letter_token_ids(tokenizer, LETTER_ALPH)
        return sorted(set(sum(letter_dict.values(), [])))
    except Exception as e:
        logger.error(
            f"MCQ letter ID extraction failed: {e}",
            extra={'correlation_id': correlation_id}
        )
        return []


def _get_phrase_token_ids_cached(
    tokenizer: TokenizerWrapper, 
    phrases: List[str], 
    correlation_id: str
) -> List[int]:
    """Enhanced phrase token ID extraction with caching."""
    if not phrases:
        return []
    
    try:
        return _first_token_ids_for_lexemes(tokenizer, phrases)
    except Exception as e:
        logger.error(
            f"Phrase token ID extraction failed: {e}",
            extra={'correlation_id': correlation_id, 'phrase_count': len(phrases)}
        )
        return []


# Backward compatibility: make the enhanced version the default
def make_dynamic_tag_bias_processor_v2(
    tokenizer: TokenizerWrapper, config: ExperimentConfig, mcq_flags: List[bool]
) -> Callable:
    """
    Version 2 of the dynamic tag bias processor with enhanced capabilities.
    
    This is an alias for the enhanced processor that provides the new
    functionality while maintaining the original function for compatibility.
    """
    return make_enhanced_dynamic_tag_bias_processor(tokenizer, config, mcq_flags)

    return _proc_vectorized


def _mask_after_answer(
    responses_mx: mx.array,
    initial_mask: mx.array,
    tokenizer: TokenizerWrapper,
    config: ExperimentConfig,
) -> mx.array:
    if responses_mx.ndim != 2:
        return initial_mask
    B, L_gen = responses_mx.shape
    initial_mask = initial_mask.astype(mx.float32)
    answer_end_id = _resolve_tag_ids(tokenizer, config.generation).get("answer_end")
    if answer_end_id is None:
        return initial_mask
    indices = mx.arange(L_gen)
    is_answer_end = responses_mx == answer_end_id
    first_end_indices = mx.argmin(mx.where(is_answer_end, indices, L_gen + 1), axis=1)
    boundary_index = first_end_indices + 1
    end_mask = (
        mx.broadcast_to(indices[None, :], responses_mx.shape) < boundary_index[:, None]
    )
    return initial_mask * end_mask.astype(mx.float32)


def _get_tag_token_ids(tokenizer) -> Dict[str, Optional[int]]:
    """Helper function to get tag token IDs."""
    def _safe_encode(tag_str: str) -> Optional[int]:
        if not tag_str:
            return None
        try:
            if hasattr(tokenizer, 'encode') and callable(tokenizer.encode):
                ids = tokenizer.encode(tag_str, add_special_tokens=False)
            elif hasattr(tokenizer, '__call__'):
                ids = tokenizer(tag_str, add_special_tokens=False)
            else:
                return None
            return int(ids[0]) if ids and hasattr(ids, '__len__') and len(ids) >= 1 else None
        except Exception:
            return None
    
    return {
        'think_start': _safe_encode('<think>'),
        'think_end': _safe_encode('</think>'),
        'answer_start': _safe_encode('<answer>'),
        'answer_end': _safe_encode('</answer>'),
        'eos': getattr(tokenizer, 'eos_token_id', None)
    }


def _get_mcq_letter_token_ids(tokenizer) -> List[int]:
    """Helper function to get MCQ letter token IDs."""
    try:
        letter_dict = _letter_token_ids(tokenizer, LETTER_ALPH)
        return sorted(set(sum(letter_dict.values(), [])))
    except Exception:
        return []


def _get_phrase_token_ids(tokenizer, phrases: List[str]) -> List[int]:
    """Helper function to get phrase token IDs."""
    if not phrases:
        return []
    try:
        return _first_token_ids_for_lexemes(tokenizer, phrases)
    except Exception:
        return []


def _get_tool_token_ids(tokenizer) -> List[int]:
    """Helper function to get tool token IDs."""
    try:
        return _first_token_ids_for_lexemes(tokenizer, _TOOL_LIKE_MARKERS)
    except Exception:
        return []


def _create_legacy_processor(
    config: 'GenerationConfig',
    tokenizer,
    logger_instance: logging.Logger
) -> Optional[Callable]:
    """
    Create the original legacy processor as a fallback.
    
    Args:
        config: Generation configuration
        tokenizer: Tokenizer instance
        logger_instance: Logger for debugging
        
    Returns:
        Legacy processor function or None if creation fails
    """
    try:
        # Extract configuration parameters with defaults
        bias_close_think = getattr(config, 'bias_close_think', 0.0)
        bias_answer_start = getattr(config, 'bias_answer_start', 0.0)
        ban_phrases = getattr(config, 'ban_phrases_for_bias', [])
        encourage_phrases = getattr(config, 'encourage_phrases_for_bias', [])
        
        # Pre-compute token IDs for efficiency
        tag_ids = _get_tag_token_ids(tokenizer)
        mcq_letter_ids = _get_mcq_letter_token_ids(tokenizer)
        encourage_ids = _get_phrase_token_ids(tokenizer, encourage_phrases)
        tool_ids = _get_tool_token_ids(tokenizer)
        
        # Create the processor function using the existing vectorized implementation
        def legacy_processor(hist_list: List[List[int]], logits: mx.array) -> mx.array:
            """Legacy processor implementation using existing vectorized code."""
            try:
                # Use the existing vectorized processor implementation
                # Create a minimal ExperimentConfig for the processor using dictionaries
                
                # Create minimal config dictionaries (not objects) for Pydantic validation
                trainer_config_dict = {
                    "output_dir": "./outputs",
                    "num_training_steps": 1000,
                    "learning_rate": 1e-5,
                    "ppo_batch_size": 1
                }
                
                model_config_dict = {
                    "model_path": "./models/dummy"
                }
                
                data_config_dict = {
                    "train_path": "./data/dummy.jsonl"
                }
                
                # Convert generation config to dict if it's an object
                if hasattr(config, 'model_dump'):
                    generation_config_dict = config.model_dump()
                elif hasattr(config, '__dict__'):
                    generation_config_dict = config.__dict__
                else:
                    # Fallback: create minimal generation config
                    generation_config_dict = {
                        "think_start_tag": "<think>",
                        "think_end_tag": "</think>",
                        "answer_start_tag": "",
                        "answer_end_tag": "",
                        "bias_close_think": bias_close_think,
                        "bias_answer_start": bias_answer_start,
                        "ban_phrases_for_bias": ban_phrases,
                        "encourage_phrases_for_bias": encourage_phrases
                    }
                
                # Create ExperimentConfig with dictionaries
                experiment_config_dict = {
                    "trainer": trainer_config_dict,
                    "model": model_config_dict,
                    "data": data_config_dict,
                    "generation": generation_config_dict
                }
                
                # Import and create ExperimentConfig
                from mlx_rl_trainer.core.config import ExperimentConfig
                experiment_config = ExperimentConfig(**experiment_config_dict)
                
                # Use the existing make_dynamic_tag_bias_processor function
                processor = make_dynamic_tag_bias_processor(
                    tokenizer,
                    experiment_config,
                    [False] * len(hist_list)
                )
                
                return processor(hist_list, logits)
                
            except Exception as e:
                logger_instance.error(f"Legacy processor execution failed: {e}")
                return logits  # Return unmodified logits on error
        
        return legacy_processor
        
    except Exception as e:
        logger_instance.error(f"Failed to create legacy processor: {e}")
        return None


# Enhanced version of the original function with proper interface
def make_dynamic_tag_bias_processor(
    tokenizer_or_config,
    config_or_experiment,
    mcq_flags_or_logger=None
) -> Optional[Callable]:
    """
    Create a dynamic tag bias processor for logit manipulation during generation.
    
    This function handles multiple call signatures for backward compatibility:
    1. make_dynamic_tag_bias_processor(tokenizer, experiment_config, mcq_flags)
    2. make_dynamic_tag_bias_processor(config, tokenizer, logger)
    
    Args:
        tokenizer_or_config: Either tokenizer (new signature) or config (old signature)
        config_or_experiment: Either experiment config (new) or tokenizer (old)
        mcq_flags_or_logger: Either mcq_flags list (new) or logger (old)
        
    Returns:
        A callable processor function or None if creation fails
    """
    # Determine which signature is being used
    if hasattr(tokenizer_or_config, 'encode'):
        # New signature: make_dynamic_tag_bias_processor(tokenizer, experiment_config, mcq_flags)
        tokenizer = tokenizer_or_config
        experiment_config = config_or_experiment
        mcq_flags = mcq_flags_or_logger if mcq_flags_or_logger is not None else [False]
        
        try:
            # Use the enhanced processor
            return make_enhanced_dynamic_tag_bias_processor(tokenizer, experiment_config, mcq_flags)
        except Exception as e:
            logger.warning(f"Enhanced processor creation failed: {e}")
            # Fallback to basic processor
            return _create_fallback_processor(tokenizer, experiment_config.generation, mcq_flags)
            
    else:
        # Old signature: make_dynamic_tag_bias_processor(config, tokenizer, logger)
        config = tokenizer_or_config
        tokenizer = config_or_experiment
        logger_instance = mcq_flags_or_logger if mcq_flags_or_logger is not None else logger
        
        try:
            # Create legacy processor
            return _create_legacy_processor(config, tokenizer, logger_instance)
        except Exception as e:
            if logger_instance:
                logger_instance.error(f"Legacy processor creation failed: {e}")
            else:
                logger.error(f"Legacy processor creation failed: {e}")
            return None


def _create_fallback_processor(tokenizer, generation_config, mcq_flags):
    """Create a basic fallback processor when enhanced version fails."""
    try:
        # Extract basic parameters
        tag_ids = _resolve_tag_ids(tokenizer, generation_config)
        
        def basic_processor(hist_list: List[List[int]], logits: mx.array) -> mx.array:
            """Basic processor that applies minimal biases."""
            try:
                # Apply basic think/answer biases if tags are available
                if tag_ids.get('think_end') is not None and tag_ids.get('answer_start') is not None:
                    # Simple bias application
                    bias_close_think = getattr(generation_config, 'bias_close_think', 0.0)
                    bias_answer_start = getattr(generation_config, 'bias_answer_start', 0.0)
                    
                    if bias_close_think != 0.0:
                        logits = logits.at[:, tag_ids['think_end']].add(bias_close_think)
                    if bias_answer_start != 0.0:
                        logits = logits.at[:, tag_ids['answer_start']].add(bias_answer_start)
                
                return logits
            except Exception as e:
                logger.warning(f"Basic processor failed: {e}")
                return logits
        
        return basic_processor
        
    except Exception as e:
        logger.error(f"Failed to create fallback processor: {e}")
        return lambda hist_list, logits: logits  # Ultimate fallback
