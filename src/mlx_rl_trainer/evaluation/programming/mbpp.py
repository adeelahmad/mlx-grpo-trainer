"""MBPP programming benchmark evaluator implementation."""
import logging, random, json, concurrent.futures
from multiprocessing import get_context, Queue
from typing import Dict, Any, List, Optional
from datasets import Dataset
from mlx_lm.tokenizer_utils import TokenizerWrapper
import mlx.core as mx
from mlx_lm.generate import generate  # Use the high-level generate function

from mlx_rl_trainer.evaluation.base_evaluator import BaseEvaluator
from mlx_rl_trainer.evaluation.registry import EvaluatorRegistry
from mlx_rl_trainer.core.trainer import EvaluationMetrics
from mlx_rl_trainer.core.exceptions import DataLoadError  # CORRECTED IMPORT
from mlx_rl_trainer.core.config import GenerationConfig
from mlx_rl_trainer.rewards.programming.code_execution import (
    _execute_code_in_isolated_process_wrapper,
)
from mlx_rl_trainer.utils.text_utils import apply_chat_template_wrapper
from mlx_rl_trainer.utils.mlx_utils import safe_make_sampler

logger = logging.getLogger(__name__)


@EvaluatorRegistry.register("mbpp")
class MBPPEvaluator(BaseEvaluator):
    """Evaluates model performance on the MBPP (Mostly Basic Python Problems) benchmark."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_problems_to_evaluate = config.get("num_samples", 10)
        self.max_gen_len = config.get("max_gen_len", 512)
        self.timeout = config.get("timeout", 10)
        self.gen_temp = config.get("temperature", 0.0)
        self.top_p = config.get("top_p", 1.0)
        self.top_k = config.get("top_k", 0)
        self.system_prompt = config.get("system_prompt")

        self.code_execution_config = {
            "timeout": self.timeout,
            "memory_limit": config.get("memory_limit", 512),
            "allow_imports": config.get("allow_imports", ["math", "re", "json"]),
        }
        self.code_execution_config_json = json.dumps(self.code_execution_config)
        logger.info(f"MBPPEvaluator initialized with config: {config}")

    def evaluate(
        self, model: Any, tokenizer: TokenizerWrapper, dataset: Dataset
    ) -> EvaluationMetrics:
        sampled_problems = dataset.shuffle(seed=self.config.get("seed", 42)).select(
            range(min(len(dataset), self.num_problems_to_evaluate))
        )
        total_problems = len(sampled_problems)
        if total_problems == 0:
            return EvaluationMetrics(task_name=self.name, pass_rate=0.0)

        correct_count = 0
        mp_context = get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max(1, self.config.get("num_workers", 1)), mp_context=mp_context
        ) as executor:
            for problem in sampled_problems:
                prompt = problem.get("prompt", "")
                test_cases_code = "\n".join(problem.get("test_list", []))
                if not prompt or not test_cases_code:
                    continue

                generated_code = self._generate_solution(model, tokenizer, prompt)
                full_executable_code = f"{generated_code}\n{test_cases_code}"

                result_queue = mp_context.Queue(1)
                future = executor.submit(
                    _execute_code_in_isolated_process_wrapper,
                    full_executable_code,
                    json.dumps([]),
                    self.code_execution_config_json,
                    result_queue,
                )

                try:
                    pass_score = result_queue.get(timeout=self.timeout + 2)
                    if pass_score > 0.99:
                        correct_count += 1
                except Exception as e:
                    logger.warning(f"MBPP execution failed for a sample: {e}")

        pass_rate = float(correct_count / total_problems)
        logger.info(
            f"MBPP Evaluation: Pass Rate = {pass_rate:.4f} ({correct_count}/{total_problems})"
        )
        return EvaluationMetrics(
            task_name=self.name,
            pass_rate=pass_rate,
            additional_info={"num_samples": total_problems},
        )

    def _generate_solution(
        self, model: Any, tokenizer: TokenizerWrapper, prompt: str
    ) -> str:
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

        return generate(
            model,
            tokenizer,
            prompt=formatted_prompt,
            max_tokens=self.max_gen_len,
            temp=self.gen_temp,
            sampler=sampler,
        )
