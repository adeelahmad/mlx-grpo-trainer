"""
Utilities related to sampling and logit processing during text generation.
Most core logic has been moved to utils/mlx_utils.py for centralization.
This file re-exports them for clarity within the generation module.
"""
import logging
import mlx.core as mx
import mlx.nn as nn
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.sample_utils import make_sampler, make_logits_processors

from mlx_rl_trainer.core.config import ExperimentConfig

# Import the actual implementations from their new central location in utils
from mlx_rl_trainer.utils.mlx_utils import (
    _first_token_ids_for_lexemes,
    _letter_token_ids,
    _resolve_tag_ids,
    make_dynamic_tag_bias_processor,
    safe_make_sampler,
)

logger = logging.getLogger(__name__)


def selective_softmax(logits: mx.array, tokens: mx.array) -> mx.array:
    """
    Computes `log_softmax` on `logits` and then gathers the probabilities
    corresponding to the `tokens` (targets). Used for extracting log-probabilities
    for specific generated tokens.
    """
    if logits.ndim != 3:
        raise ValueError(f"Logits must be 3D (B, T, V), got {logits.shape}")
    if tokens.ndim != 2:
        raise ValueError(f"Tokens must be 2D (B, T), got {tokens.shape}")

    log_probs_all = nn.log_softmax(logits.astype(mx.float32), axis=-1)

    # Expand tokens to (B, T, 1) to use with mx.take_along_axis
    tokens_expanded = tokens[..., None]

    log_probs = mx.take_along_axis(
        log_probs_all, tokens_expanded.astype(mx.int64), axis=-1
    ).squeeze(-1)
    return log_probs.astype(mx.float32)
