"""Code execution reward function."""

import subprocess
import tempfile
import ast
import os
import json
import logging
import concurrent.futures
from multiprocessing import get_context
from typing import Dict, Any, List, Optional
from pathlib import Path

from mlx_rl_trainer.rewards.base_reward import BaseReward
from mlx_rl_trainer.rewards.registry import RewardRegistry
from mlx_rl_trainer.rewards.context import RewardContext
from mlx_rl_trainer.utils.text_utils import _extract_python_code

logger = logging.getLogger(__name__)


def _validate_syntax_isolated(code: str) -> bool:
    """Checks if code is valid Python syntax (for isolated process)."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False
    except Exception as e:
        logger.debug(f"Error during isolated AST parsing: {e}")
        return False


def _execute_code_in_isolated_process_wrapper(
    code: str, test_cases_json: str, config_json: str
) -> float:
    """
    Wrapper function to execute code in an isolated process.
    It deserializes config and test_cases, then calls the actual execution logic and returns a score.
    """
    filepath: Optional[Path] = None
    try:
        test_cases = json.loads(test_cases_json)
        config = json.loads(config_json)
        timeout = config.get("timeout", 5)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            filepath = Path(f.name)

        passed_count = 0
        if not test_cases:  # For benchmarks like MBPP
            if _validate_syntax_isolated(code):
                process_result = subprocess.run(
                    ["python", str(filepath)],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
                if process_result.returncode == 0:
                    return 1.0  # Success if it runs without error
                else:
                    logger.warning(
                        f"MBPP-style execution failed with stderr: {process_result.stderr}"
                    )
                    return 0.0
            else:
                return 0.0

        for test_input_dict in test_cases:
            test_input = str(test_input_dict.get("input", "")).encode("utf-8")
            expected_output = str(test_input_dict.get("expected", "")).strip()

            process_result = subprocess.run(
                ["python", str(filepath)],
                input=test_input,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )

            stdout_output = process_result.stdout.strip()
            if stdout_output == expected_output:
                passed_count += 1

        return float(passed_count / max(1, len(test_cases)))

    except subprocess.TimeoutExpired:
        logger.warning(f"Isolated code execution timed out after {timeout}s.")
        return 0.0
    except Exception as e:
        logger.error(
            f"Exception in isolated code execution process: {e}", exc_info=True
        )
        return 0.0
    finally:
        if filepath and filepath.exists():
            try:
                filepath.unlink()
            except OSError as e:
                logger.error(f"Failed to delete temp code file {filepath}: {e}")


@RewardRegistry.register("code_execution")
class CodeExecutionReward(BaseReward):
    """Rewards generated code based on its execution success against provided test cases."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.timeout = config.get("timeout", 5)
        self.num_workers = config.get("num_workers", 1)
        self.code_execution_config = {
            "timeout": self.timeout,
            "memory_limit": config.get("memory_limit", 512),
            "allow_imports": config.get("allow_imports", ["math", "re", "json"]),
        }
        self.code_execution_config_json = json.dumps(self.code_execution_config)
        logger.info(f"Initialized CodeExecutionReward with timeout: {self.timeout}s.")

    def _validate_syntax(self, code: str) -> bool:
        """Checks if the extracted code has valid Python syntax using ast (in main process)."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            logger.debug("Generated code has syntax errors.")
            return False
        except Exception as e:
            logger.debug(f"Unexpected error during AST parsing: {e}")
            return False

    def compute(self, context: RewardContext) -> float:
        # For a single item, it's inefficient to spin up a pool, but we can reuse the wrapper.
        # We'll use a single-shot ProcessPoolExecutor to handle it cleanly.
        return self.batch_compute([context])[0]["total"]

    def batch_compute(self, contexts: List[RewardContext]) -> List[Dict[str, float]]:
        rewards_for_batch: List[Dict[str, float]] = [
            {self.name: 0.0, "total": 0.0} for _ in contexts
        ]

        mp_context = get_context("spawn")
        futures_with_indices = {}
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.num_workers, mp_context=mp_context
        ) as executor:
            for idx, context in enumerate(contexts):
                code = _extract_python_code(context.generated_text)
                if not code.strip() or not self._validate_syntax(code):
                    continue

                future = executor.submit(
                    _execute_code_in_isolated_process_wrapper,
                    code,
                    json.dumps(context.test_cases),
                    self.code_execution_config_json,
                )
                futures_with_indices[future] = idx

            for future in concurrent.futures.as_completed(futures_with_indices):
                original_idx = futures_with_indices[future]
                try:
                    pass_rate = future.result(timeout=self.timeout + 3)
                    final_score = 0.3 + (0.7 * float(pass_rate))
                    score = float(max(0.0, min(1.0, final_score)))
                    rewards_for_batch[original_idx] = {
                        self.name: score,
                        "total": score,
                    }
                except Exception as e:
                    logger.error(
                        f"Error in batch execution for context {original_idx}: {e}",
                        exc_info=True,
                    )
        return rewards_for_batch
