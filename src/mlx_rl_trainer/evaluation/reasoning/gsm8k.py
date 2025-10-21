"""GSM8K reasoning benchmark evaluator implementation."""
import logging, random
import mlx.core as mx
from typing import Dict, Any, List, Optional
from datasets import Dataset
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.generate import generate

from mlx_rl_trainer.evaluation.base_evaluator import BaseEvaluator
from mlx_rl_trainer.evaluation.registry import EvaluatorRegistry
from mlx_rl_trainer.core.trainer import EvaluationMetrics
from mlx_rl_trainer.core.exceptions import DataLoadError  # CORRECTED IMPORT
from mlx_rl_trainer.core.config import GenerationConfig
from mlx_rl_trainer.utils.text_utils import (
    _extract_final_numeric,
    apply_chat_template_wrapper,
)
from mlx_rl_trainer.utils.mlx_utils import safe_make_sampler

logger = logging.getLogger(__name__)


@EvaluatorRegistry.register("gsm8k")
class GSM8KEvaluator(BaseEvaluator):
    """Evaluates model performance on the GSM8K mathematical reasoning benchmark."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_samples_to_evaluate = config.get("num_samples", 10)
        self.max_gen_len = config.get("max_gen_len", 256)
        self.system_prompt = config.get("system_prompt")
        self.gen_temp = config.get("temperature", 0.0)
        self.top_p = config.get("top_p", 1.0)
        self.top_k = config.get("top_k", 0)

    def evaluate(
        self, model: Any, tokenizer: TokenizerWrapper, dataset: Dataset
    ) -> EvaluationMetrics:
        sampled_problems = dataset.shuffle(seed=self.config.get("seed", 42)).select(
            range(min(len(dataset), self.num_samples_to_evaluate))
        )
        total_problems = len(sampled_problems)
        if total_problems == 0:
            return EvaluationMetrics(task_name=self.name, pass_rate=0.0)
        correct_count = 0
        for problem in sampled_problems:
            prompt, reference_answer = problem.get("question", ""), problem.get(
                "answer", ""
            )
            if not prompt or not reference_answer:
                continue

            generated_text = self._generate_solution(model, tokenizer, prompt)
            pred_num = _extract_final_numeric(generated_text)
            gold_num = _extract_final_numeric(reference_answer)
            if pred_num is not None and gold_num is not None and pred_num == gold_num:
                correct_count += 1

        accuracy = float(correct_count / total_problems) if total_problems > 0 else 0.0
        logger.info(
            f"GSM8K Evaluation: Accuracy = {accuracy:.4f} ({correct_count}/{total_problems})"
        )
        return EvaluationMetrics(
            task_name=self.name,
            pass_rate=accuracy,
            additional_info={"num_samples": total_problems},
        )

    def _generate_solution(
        self, model: Any, tokenizer: TokenizerWrapper, prompt: str
    ) -> str:
        model.eval()
        formatted_prompt = apply_chat_template_wrapper(
            tokenizer, prompt, self.system_prompt
        )
        gen_cfg = GenerationConfig(
            temperature=self.gen_temp,
            sampling_top_p=self.top_p,
            sampling_top_k=self.top_k,
        )
        sampler = safe_make_sampler(gen_cfg, temp=self.gen_temp)
        return generate(
            model,
            tokenizer,
            prompt=formatted_prompt,
            max_tokens=self.max_gen_len,
            temp=self.gen_temp,
            sampler=sampler,
        )
