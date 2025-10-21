# file_path: mlx_rl_trainer/src/mlx_rl_trainer/evaluation/general/perplexity.py
# revision_no: 001
# goals_of_writing_code_block: Perplexity evaluation evaluator implementation.
# type_of_code_response: add new code
"""Perplexity evaluation evaluator implementation."""

import logging
import random
import mlx.core as mx
import mlx.nn as nn  # For cross_entropy
import numpy as np
from typing import Dict, Any, List
from datasets import Dataset
from mlx_lm.tokenizer_utils import TokenizerWrapper

from mlx_rl_trainer.evaluation.base_evaluator import BaseEvaluator
from mlx_rl_trainer.evaluation.registry import EvaluatorRegistry
from mlx_rl_trainer.core.trainer import EvaluationMetrics
from mlx_rl_trainer.utils.mlx_utils import (
    _create_4d_attention_mask,
)  # For masking utility
from mlx_rl_trainer.utils.text_utils import (
    apply_chat_template_wrapper,
)  # For consistent prompting

logger = logging.getLogger(__name__)


@EvaluatorRegistry.register("perplexity")
class PerplexityEvaluator(BaseEvaluator):
    """
    Evaluates model performance based on perplexity (exponential of average negative log-likelihood)
    on a given dataset.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.max_seq_len = config.get("max_seq_len", 512)
        self.batch_size = config.get("batch_size", 4)
        logger.info(f"PerplexityEvaluator initialized with config: {config}")

    def evaluate(
        self, model: Any, tokenizer: TokenizerWrapper, dataset: Dataset
    ) -> EvaluationMetrics:
        logger.info(f"Running Perplexity evaluation on {len(dataset)} samples.")

        model.eval()  # Set model to evaluation mode
        total_nll = 0.0  # Total negative log-likelihood
        total_tokens = 0

        for i in range(0, len(dataset), self.batch_size):
            batch_samples = dataset.select(
                range(i, min(i + self.batch_size, len(dataset)))
            )

            # Tokenize and prepare batch for perplexity calculation
            input_texts = [
                s.get("prompt", "") + " " + s.get("completion", "")
                for s in batch_samples
            ]

            # Apply chat template if a system prompt is part of the overall config
            from mlx_rl_trainer.core.config import (
                ExperimentConfig,
            )  # Temporarily import to access system prompt

            mock_exp_config = ExperimentConfig(
                trainer={}, model={}, data={}, rewards=[], system_prompt=""
            )  # Minimal instance
            if hasattr(self.config, "system_prompt") and self.config.get(
                "system_prompt"
            ):
                mock_exp_config.system_prompt = self.config[
                    "system_prompt"
                ]  # Use configured system prompt

            formatted_texts = [
                apply_chat_template_wrapper(
                    tokenizer, text, mock_exp_config.system_prompt
                )
                for text in input_texts
            ]
            encoded_batch = [
                tokenizer.encode(t, add_special_tokens=True) for t in formatted_texts
            ]

            # Pad to max_seq_len, truncate if longer
            padded_ids = []
            for ids in encoded_batch:
                if len(ids) > self.max_seq_len:
                    padded_ids.append(ids[: self.max_seq_len])
                else:
                    padded_ids.append(
                        ids + [tokenizer.pad_token_id] * (self.max_seq_len - len(ids))
                    )

            input_ids_mx = mx.array(padded_ids, dtype=mx.int32)

            # Forward pass to get logits
            # For NLL, we need attention mask for padding
            attn_mask = _create_4d_attention_mask(
                input_ids_mx, tokenizer.pad_token_id, dtype=mx.float32
            )
            logits = model(input_ids_mx, mask=attn_mask)

            # Calculate cross-entropy loss (negative log-likelihood)
            # Targets are shifted: predict token i from context up to i-1
            labels = input_ids_mx[:, 1:]
            logits_shifted = logits[:, :-1, :]

            # Calculate loss only for non-padding tokens
            mask = (labels != tokenizer.pad_token_id).astype(mx.float32)

            loss_per_token = nn.losses.cross_entropy(
                logits_shifted, labels, reduction="none"
            )
            masked_loss = loss_per_token * mask

            total_nll += mx.sum(masked_loss).item()
            total_tokens += mx.sum(mask).item()

        if total_tokens == 0:
            return EvaluationMetrics(task_name=self.name, perplexity=float("inf"))

        avg_nll = total_nll / total_tokens
        perplexity = float(
            np.exp(avg_nll)
        )  # Perplexity is exp(average negative log-likelihood)

        logger.info(f"Perplexity Evaluation: Perplexity = {perplexity:.2f}")

        return EvaluationMetrics(
            task_name=self.name,
            perplexity=perplexity,
            additional_info={"avg_nll": avg_nll, "num_tokens": total_tokens},
        )
