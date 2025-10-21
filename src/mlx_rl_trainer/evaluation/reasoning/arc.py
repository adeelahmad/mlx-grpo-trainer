"""ARC reasoning benchmark evaluator implementation."""
import logging, random, re
from typing import Dict, Any, List, Optional
from datasets import Dataset
from mlx_lm.tokenizer_utils import TokenizerWrapper
import mlx.core as mx
from mlx_lm.generate import generate

from mlx_rl_trainer.evaluation.base_evaluator import BaseEvaluator
from mlx_rl_trainer.evaluation.registry import EvaluatorRegistry
from mlx_rl_trainer.core.trainer import EvaluationMetrics
from mlx_rl_trainer.core.exceptions import DataLoadError  # CORRECTED IMPORT
from mlx_rl_trainer.core.config import GenerationConfig
from mlx_rl_trainer.utils.text_utils import apply_chat_template_wrapper
from mlx_rl_trainer.utils.mlx_utils import safe_make_sampler

logger = logging.getLogger(__name__)


@EvaluatorRegistry.register("arc")
class ARCEvaluator(BaseEvaluator):
    """Evaluates model performance on the ARC (AI2 Reasoning Challenge) benchmark."""

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
            prompt, choices = problem.get("prompt", ""), problem.get(
                "choices", {"text": [], "label": []}
            )
            correct_label = problem.get("answerKey", "")
            if not prompt or not choices["text"] or not correct_label:
                continue

            generated_text = self._generate_solution(model, tokenizer, prompt, choices)
            predicted_label = self._extract_arc_answer(generated_text)
            if predicted_label == correct_label:
                correct_count += 1

        accuracy = float(correct_count / total_problems) if total_problems > 0 else 0.0
        logger.info(
            f"ARC Evaluation: Accuracy = {accuracy:.4f} ({correct_count}/{total_problems})"
        )
        return EvaluationMetrics(
            task_name=self.name,
            pass_rate=accuracy,
            additional_info={"num_samples": total_problems},
        )

    def _generate_solution(
        self,
        model: Any,
        tokenizer: TokenizerWrapper,
        prompt: str,
        choices: Dict[str, List[str]],
    ) -> str:
        full_prompt = (
            f"{prompt}\nChoices:\n"
            + "\n".join(
                [
                    f"{label}) {text}"
                    for label, text in zip(choices["label"], choices["text"])
                ]
            )
            + "\nAnswer:"
        )
        model.eval()
        formatted_prompt = apply_chat_template_wrapper(
            tokenizer, full_prompt, self.system_prompt
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

    def _extract_arc_answer(self, generated_text: str) -> Optional[str]:
        match = re.search(r"\b([A-Z])\s*$", generated_text.strip())
        return match.group(1) if match else None
