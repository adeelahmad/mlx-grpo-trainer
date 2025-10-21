"""HumanEval benchmark evaluator implementation."""
import logging, random, json, concurrent.futures
from multiprocessing import get_context
from typing import Dict, Any, List, Optional
from datasets import Dataset
from mlx_lm.tokenizer_utils import TokenizerWrapper
import mlx.core as mx
from mlx_lm.models import cache
from mlx_lm.sample_utils import make_logits_processors
from mlx_lm.generate import generate  # Use the high-level generate function

from mlx_rl_trainer.evaluation.base_evaluator import BaseEvaluator
from mlx_rl_trainer.evaluation.registry import EvaluatorRegistry
from mlx_rl_trainer.core.trainer import EvaluationMetrics
from mlx_rl_trainer.core.exceptions import DataLoadError  # CORRECTED IMPORT
from mlx_rl_trainer.rewards.programming.code_execution import (
    _execute_code_in_isolated_process_wrapper,
)
from mlx_rl_trainer.core.config import GenerationConfig
from mlx_rl_trainer.utils.text_utils import apply_chat_template_wrapper
from mlx_rl_trainer.utils.mlx_utils import safe_make_sampler

logger = logging.getLogger(__name__)


@EvaluatorRegistry.register("human_eval")
class HumanEvalEvaluator(BaseEvaluator):
    """Evaluates model performance on the HumanEval code generation benchmark."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.k_values = config.get("k_values", [1])
        self.num_problems_to_evaluate = config.get("num_samples", 10)
        self.timeout = config.get("timeout", 5)
        self.max_gen_len = config.get("max_gen_len", 512)
        self.gen_temp = config.get("temperature", 0.0)
        self.top_p = config.get("top_p", 1.0)
        self.top_k = config.get("top_k", 0)
        self.system_prompt = config.get("system_prompt", None)

        self.code_execution_config = {
            "timeout": self.timeout,
            "memory_limit": config.get("memory_limit", 512),
            "allow_imports": config.get("allow_imports", ["math", "re", "json"]),
        }
        self.code_execution_config_json = json.dumps(self.code_execution_config)

    def evaluate(
        self, model: Any, tokenizer: TokenizerWrapper, dataset: Dataset
    ) -> EvaluationMetrics:
        if not dataset or len(dataset) == 0:
            raise DataLoadError("HumanEval dataset is empty.")
        sampled_problems = dataset.shuffle(seed=self.config.get("seed", 42)).select(
            range(min(len(dataset), self.num_problems_to_evaluate))
        )
        total_problems = len(sampled_problems)
        if total_problems == 0:
            return EvaluationMetrics(task_name=self.name, pass_rate=0.0)

        max_k_required = max(self.k_values)
        passed_at_k_counts: Dict[int, int] = {k: 0 for k in self.k_values}
        logger.info(
            f"Evaluating {total_problems} problems for {self.name} (max_k={max_k_required})..."
        )
        mp_context = get_context("spawn")

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max(1, self.config.get("num_workers", 1)), mp_context=mp_context
        ) as executor:
            for problem in sampled_problems:
                problem_prompt = problem.get("prompt", "")
                problem_test_cases = [
                    json.loads(tc) for tc in problem.get("test_cases", []) if tc
                ]
                if not problem_prompt or not problem_test_cases:
                    continue

                generated_solutions = self._generate_solutions(
                    model, tokenizer, problem_prompt, max_k_required
                )

                futures = [
                    executor.submit(
                        _execute_code_in_isolated_process_wrapper,
                        solution_text,
                        json.dumps(problem_test_cases),
                        self.code_execution_config_json,
                    )
                    for solution_text in generated_solutions
                ]

                solution_pass_scores = []
                for future in concurrent.futures.as_completed(futures):
                    try:
                        solution_pass_scores.append(
                            future.result(timeout=self.timeout + 2)
                        )
                    except Exception:
                        solution_pass_scores.append(0.0)

                solution_pass_statuses = [
                    score > 0.99 for score in solution_pass_scores
                ]
                for k_val in sorted(self.k_values):
                    if any(solution_pass_statuses[:k_val]):
                        passed_at_k_counts[k_val] += 1

        final_pass_rates = {
            f"pass@{k}": count / total_problems
            for k, count in passed_at_k_counts.items()
        }
        primary_pass_rate = final_pass_rates.get("pass@1", 0.0)

        logger.info(f"{self.name} Evaluation Summary: {final_pass_rates}")
        return EvaluationMetrics(
            task_name=self.name,
            pass_rate=primary_pass_rate,
            additional_info=final_pass_rates,
        )

    def _generate_solutions(
        self, model: Any, tokenizer: TokenizerWrapper, prompt: str, num_solutions: int
    ) -> List[str]:
        model.eval()
        formatted_prompt = apply_chat_template_wrapper(
            tokenizer, prompt, self.system_prompt
        )

        gen_cfg_for_sampler = GenerationConfig(
            temperature=self.gen_temp,
            sampling_top_p=self.top_p,
            sampling_top_k=self.top_k,
        )
        sampler = safe_make_sampler(gen_cfg_for_sampler, temp=self.gen_temp)

        generated_texts = []
        for _ in range(num_solutions):
            # Using mlx_lm.generate for simplicity in evaluation scripts
            response = generate(
                model,
                tokenizer,
                prompt=formatted_prompt,
                max_tokens=self.max_gen_len,
                temp=self.gen_temp,
                sampler=sampler,
            )
            generated_texts.append(response)
        return generated_texts
