# file_path: mlx_rl_trainer/src/mlx_rl_trainer/evaluation/__init__.py
# revision_no: 001
# goals_of_writing_code_block: Top-level __init__.py for the evaluation module.
# type_of_code_response: add new code
"""Evaluation benchmarks and metrics."""

from .base_evaluator import BaseEvaluator
from .registry import EvaluatorRegistry, register_evaluator

# Import concrete evaluator implementations to ensure they are registered
import mlx_rl_trainer.evaluation.programming.human_eval
import mlx_rl_trainer.evaluation.reasoning.gsm8k
import mlx_rl_trainer.evaluation.reasoning.arc
import mlx_rl_trainer.evaluation.general.perplexity


__all__ = ["BaseEvaluator", "EvaluatorRegistry", "register_evaluator"]
