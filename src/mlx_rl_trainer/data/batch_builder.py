# Path: src/mlx_rl_trainer/data/batch_builder.py
import logging
from typing import Dict, Any, List, Tuple, Union
from datasets import Dataset
import mlx.core as mx
import numpy as np
import gc

from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_rl_trainer.core.config import DataConfig
from mlx_rl_trainer.utils.text_utils import (
    apply_chat_template_wrapper,
    clean_completion_string,
)
from mlx_rl_trainer.rewards.format.tag_structure import (
    extract_think_region,
    extract_answer_region,
)

logger = logging.getLogger(__name__)


def _compose_prompt_from_jsonl_sample(sample: Dict[str, Any]) -> Tuple[str, str, str]:
    """Processes a single sample from a JSONL-loaded dataset."""
    prompt_text = sample.get("prompt", "")
    completion = clean_completion_string(sample.get("completion", ""))
    ref_think = extract_think_region(completion, None)
    ref_ans = extract_answer_region(completion, None) or completion.strip()
    return prompt_text, ref_ans, ref_think


# ⭐ MODIFIED: Main function now handles both data types
def build_rollout_batch(
    tokenizer: TokenizerWrapper,
    dataset: Dataset,
    indices: List[int],
    config: DataConfig,
) -> Tuple[List[Dict[str, Any]], mx.array, int]:
    """
    Builds a batch of rollout prompts from either a JSONL-based dataset
    or a pre-tokenized .npy-based dataset.
    """
    if not indices:
        return [], mx.array([], dtype=mx.int32), 0

    # --- Check dataset type ---
    is_pretokenized = "prompt_tokens" in dataset.column_names

    if is_pretokenized:
        return _build_from_pretokenized(tokenizer, dataset, indices, config)
    else:
        return _build_from_jsonl(tokenizer, dataset, indices, config)


# ⭐ NEW: Logic for building a batch from pre-tokenized data
def _build_from_pretokenized(
    tokenizer: TokenizerWrapper,
    dataset: Dataset,
    indices: List[int],
    config: DataConfig,
) -> Tuple[List[Dict[str, Any]], mx.array, int]:
    """Builds a batch from a pre-tokenized .npy dataset."""
    prompts_data = []
    token_lists = []
    max_len_in_batch = 0

    batch_samples = dataset.select(indices)

    for i, raw in enumerate(batch_samples):
        p_tokens = raw["prompt_tokens"]
        c_tokens = raw["completion_tokens"]

        # We need to decode to get the string representations for reward calculation
        prompt_text = tokenizer.decode(p_tokens)
        completion_text = tokenizer.decode(c_tokens)

        ref_think = extract_think_region(completion_text, None)
        ref_ans = (
            extract_answer_region(completion_text, None) or completion_text.strip()
        )

        entry = {
            "original_index": indices[i],
            "text": prompt_text,
            "tokens": p_tokens,
            "ref_answer_str": ref_ans,
            "ref_think_str": ref_think,
        }
        prompts_data.append(entry)
        token_lists.append(p_tokens)
        max_len_in_batch = max(max_len_in_batch, len(p_tokens))

    if not prompts_data:
        return [], mx.array([], dtype=mx.int32), 0

    # Padding logic remains the same
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    padded_array = np.full((len(token_lists), max_len_in_batch), pad_id, dtype=np.int32)
    for idx, tok in enumerate(token_lists):
        pad_len = max_len_in_batch - len(tok)
        padded_array[idx, pad_len:] = tok

    prompts_mx = mx.array(padded_array)
    gc.collect()
    return prompts_data, prompts_mx, max_len_in_batch


# ⭐ RENAMED: Logic for building a batch from the original JSONL dataset
def _build_from_jsonl(
    tokenizer: TokenizerWrapper,
    dataset: Dataset,
    indices: List[int],
    config: DataConfig,
) -> Tuple[List[Dict[str, Any]], mx.array, int]:
    """Builds a batch from a raw JSONL dataset (original logic)."""
    prompts_data = []
    token_lists = []
    max_len_in_batch = 0

    batch_samples = dataset.select(indices)

    for i, raw in enumerate(batch_samples):
        prompt_text, ref_ans, ref_think = _compose_prompt_from_jsonl_sample(raw)

        formatted_prompt = apply_chat_template_wrapper(tokenizer, prompt_text, "")
        p_tokens = tokenizer.encode(formatted_prompt, add_special_tokens=False)

        if len(p_tokens) > config.max_prompt_len:
            p_tokens = p_tokens[-config.max_prompt_len :]

        if not p_tokens:
            continue

        entry = {
            "original_index": indices[i],
            "text": formatted_prompt,
            "tokens": p_tokens,
            "ref_answer_str": ref_ans,
            "ref_think_str": ref_think,
        }
        prompts_data.append(entry)
        token_lists.append(p_tokens)
        max_len_in_batch = max(max_len_in_batch, len(p_tokens))

    if not prompts_data:
        return [], mx.array([], dtype=mx.int32), 0

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    padded_array = np.full((len(token_lists), max_len_in_batch), pad_id, dtype=np.int32)
    for idx, tok in enumerate(token_lists):
        pad_len = max_len_in_batch - len(tok)
        padded_array[idx, pad_len:] = tok

    prompts_mx = mx.array(padded_array)
    gc.collect()
    return prompts_data, prompts_mx, max_len_in_batch
