# file_path: mlx_rl_trainer/docs/adding_evaluators.md
# revision_no: 001
# goals_of_writing_code_block: Documentation for adding custom evaluation benchmarks.
# type_of_code_response: add new code
# Adding Custom Evaluation Benchmarks

This document outlines the process for adding new, custom evaluation benchmarks to the MLX RL Trainer framework. The system is designed for extensibility, allowing you to plug in new evaluation logic without modifying core trainer code.

## Quick Start

1.  **Create a New File**: Place your new evaluator in an appropriate subdirectory under `src/mlx_rl_trainer/evaluation/` (e.g., `evaluation/custom/my_benchmark.py`).
2.  **Inherit from `BaseEvaluator`**: Your evaluator class must inherit from `mlx_rl_trainer.evaluation.base_evaluator.BaseEvaluator`.
3.  **Implement `evaluate(self, model, tokenizer, dataset)`**: This is the core method where your evaluation logic resides. It should return an `EvaluationMetrics` object.
4.  **Register Your Evaluator**: Decorate your class with `@EvaluatorRegistry.register("your_benchmark_name")`.
5.  **Import Your Evaluator**: Ensure your new evaluator module is imported in `src/mlx_rl_trainer/evaluation/__init__.py` so the `EvaluatorRegistry` can discover it.
6.  **Configure in YAML**: Add your evaluator to the `evaluation.benchmarks` section of your experiment's YAML configuration file.

## Example: Custom Text Summarization Benchmark

Let's say you want to evaluate the model's ability to summarize text.

```python
# src/mlx_rl_trainer/evaluation/custom/summarization_benchmark.py

import logging
from typing import Dict, Any
from datasets import Dataset
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.sample_utils import make_logits_processors
import mlx.core as mx
import mlx.nn as nn
from mlx_rl_trainer.utils.mlx_utils import safe_make_sampler
from mlx_rl_trainer.core.trainer import EvaluationMetrics, DataLoadError
from mlx_rl_trainer.evaluation.base_evaluator import BaseEvaluator
from mlx_rl_trainer.evaluation.registry import EvaluatorRegistry
# Assuming you have a metric for summarization, e.g., ROUGE or simply length match
# from mlx_rl_trainer.utils.text_metrics import calculate_rouge_score # Placeholder

logger = logging.getLogger(__name__)

@EvaluatorRegistry.register("summarization_benchmark")
class SummarizationEvaluator(BaseEvaluator):
    """
    Evaluates model performance on a custom text summarization task.

    Configuration:
        num_samples: Number of samples to evaluate (default: 10)
        max_gen_len: Maximum tokens for generated summary (default: 100)
        temperature: Generation temperature (default: 0.0 for greedy)
        top_p: Top-p sampling (default: 1.0)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_samples_to_evaluate = config.get("num_samples", 10)
        self.max_gen_len = config.get("max_gen_len", 100)
        self.temperature = config.get("temperature", 0.0)
        self.top_p = config.get("top_p", 1.0)
        logger.info(f"Initialized SummarizationEvaluator with config: {config}")

    def evaluate(self, model: Any, tokenizer: TokenizerWrapper, dataset: Dataset) -> EvaluationMetrics:
        """
        Run summarization evaluation on the model.

        Args:
            model: The MLX model to evaluate.
            tokenizer: The tokenizer.
            dataset: The dataset (expected to contain 'document' and 'summary' fields).

        Returns:
            An `EvaluationMetrics` object.
        """
        if not dataset or len(dataset) == 0:
            raise DataLoadError("Summarization dataset is empty.")

        sampled_problems = dataset.shuffle(seed=self.config.get("seed", 42)).select(
            range(min(len(dataset), self.num_samples_to_evaluate))
        )

        total_problems = len(sampled_problems)
        if total_problems == 0: return EvaluationMetrics(task_name=self.name, pass_rate=0.0)

        total_score = 0.0

        model.eval() # Set model to eval mode
        sampler = safe_make_sampler(self.config, temp=self.temperature)
        logits_processors = make_logits_processors(
            repetition_penalty=1.1, repetition_context_size=20, logit_bias=None
        )

        for problem in sampled_problems:
            document = problem.get('document', '')
            reference_summary = problem.get('summary', '')

            if not document or not reference_summary: continue

            # 1. Generate summary
            prompt = f"Summarize the following document:\n{document}\nSummary:"
            encoded_prompt = tokenizer.encode(prompt, add_special_tokens=True)
            prompt_mx = mx.array([encoded_prompt], dtype=mx.int32)

            # Simple greedy generation loop (replace with actual mlx_lm generation)
            generated_ids = []
            current_ids = encoded_prompt

            for _ in range(self.max_gen_len):
                # Mock: model forward pass
                # logits = model(mx.array([current_ids]))[:, -1, :].astype(mx.float32)
                # For this stub, we'll just mock token generation
                logits = mx.random.normal((1, tokenizer.vocab_size)) # Mock logits

                for proc_fn in logits_processors:
                    logits = proc_fn([current_ids], logits)

                next_token = sampler(logits)

                if next_token.item() == tokenizer.eos_token_id: break

                generated_ids.append(next_token.item())
                current_ids.append(next_token.item())

            generated_summary = tokenizer.decode(generated_ids, skip_special_tokens=True)

            # 2. Score summary (mock or use real metric)
            # score = calculate_rouge_score(generated_summary, reference_summary) # Placeholder for real metric
            score = 1.0 - (abs(len(generated_summary.split()) - len(reference_summary.split())) /
                           max(1, len(reference_summary.split()))) # Mock score based on length difference

            total_score += max(0.0, score)

        avg_score = total_score / total_problems
        logger.info(f"Summarization Evaluation: Average Score = {avg_score:.4f}")

        return EvaluationMetrics(task_name=self.name, pass_rate=avg_score, additional_info={"num_samples": total_problems})

```

## Configuration

To use this new evaluator, add it to your experiment's YAML configuration:

```yaml
# configs/experiments/your_experiment.yaml
evaluation:
  benchmarks:
    - name: summarization_benchmark # Your new evaluator!
      config:
        num_samples: 20
        max_gen_len: 120
        temperature: 0.5
```

## Best Practices for Custom Evaluators

-   **Return Values**: Always return a `float` typically between `0.0` and `1.0`. Higher values should indicate a "better" response according to your criteria.
-   **Error Handling**: Wrap your `evaluate` logic in `try-except` blocks and return `EvaluationMetrics` with appropriate defaults on failure. Log errors for debugging.
-   **Efficiency**: Optimize generation and metric calculation for large datasets.
-   **Inputs Validation**: Use `self.validate_inputs(context)` if you have a custom validation method (BaseEvaluator does not currently have one for `evaluate` directly, but `DataLoadError` can be used).
-   **Modularity**: Keep your evaluator focused on a specific benchmark.
-   **Testing**: Write unit tests for your custom evaluator to ensure it behaves as expected across various inputs and edge cases.

## Testing

You can add a unit test for your new evaluator in `tests/unit/test_evaluation/test_summarization_benchmark.py`:

```python
# tests/unit/test_evaluation/test_summarization_benchmark.py

import pytest
from datasets import Dataset
from mlx_rl_trainer.evaluation.custom.summarization_benchmark import SummarizationEvaluator
from mlx_rl_trainer.core.model_manager import MockModel, MockTokenizer # For mock models

class TestSummarizationEvaluator:
    """Test suite for SummarizationEvaluator."""

    @pytest.fixture
    def mock_tokenizer(self):
        return MockTokenizer() # Uses default vocab

    @pytest.fixture
    def mock_model(self):
        return MockModel() # Basic mock model

    @pytest.fixture
    def mock_dataset(self):
        # Create a dummy dataset for summarization
        data = {
            "document": [
                "This is a long document about AI. It discusses neural networks, machine learning, and deep learning. Summarization is an important task in NLP.",
                "Another document talks about climate change, renewable energy, and sustainable practices. The impact on the environment is critical."
            ],
            "summary": [
                "AI, neural networks, deep learning, NLP summarization.",
                "Climate change, renewable energy, sustainable environment."
            ]
        }
        return Dataset.from_dict(data)

    @pytest.fixture
    def evaluator_config(self):
        return {
            "num_samples": 2,
            "max_gen_len": 50,
            "temperature": 0.0 # Greedy for predictable tests
        }

    def test_evaluate_returns_metrics(self, mock_model, mock_tokenizer, mock_dataset, evaluator_config):
        """Test if evaluate method returns correct metrics format."""
        evaluator = SummarizationEvaluator(evaluator_config)
        metrics = evaluator.evaluate(mock_model, mock_tokenizer, mock_dataset)

        assert isinstance(metrics.task_name, str)
        assert isinstance(metrics.pass_rate, float)
        assert isinstance(metrics.additional_metrics, dict)
        assert metrics.task_name == "summarization_benchmark"
        assert metrics.pass_rate >= 0.0 and metrics.pass_rate <= 1.0
        assert "num_samples" in metrics.additional_metrics
        assert metrics.additional_metrics["num_samples"] == 2

    def test_empty_dataset_returns_zero(self, mock_model, mock_tokenizer, evaluator_config):
        """Test handling of empty dataset."""
        empty_dataset = Dataset.from_dict({"document": [], "summary": []})
        evaluator = SummarizationEvaluator(evaluator_config)
        metrics = evaluator.evaluate(mock_model, mock_tokenizer, empty_dataset)
        assert metrics.pass_rate == 0.0

    # You would add more specific tests here, for example:
    # - test_score_calculation_for_perfect_summary
    # - test_score_calculation_for_bad_summary
    # - test_generation_parameters_are_used
    # - test_error_handling_in_generation
```
