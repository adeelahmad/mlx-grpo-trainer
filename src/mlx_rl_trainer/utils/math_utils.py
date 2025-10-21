# file_path: mlx_rl_trainer/src/mlx_rl_trainer/utils/math_utils.py
# revision_no: 001
# goals_of_writing_code_block: Math utility functions.
# type_of_code_response: add new code
"""Math utility functions."""

import numpy as np
import mlx.core as mx
from typing import List


def safe_divide(
    numerator: float, denominator: float, default_if_zero: float = 0.0
) -> float:
    """Performs division safely, returning a default value if denominator is zero."""
    if denominator == 0:
        return default_if_zero
    return numerator / denominator


def safe_mean(data: List[float]) -> float:
    """Calculates the mean of a list of floats, handling empty lists."""
    if not data:
        return 0.0
    return float(np.mean(data))


def safe_std(data: List[float]) -> float:
    """Calculates the standard deviation of a list of floats, handling empty lists."""
    if not data or len(data) < 2:
        return 0.0
    return float(np.std(data))


def softmax(x: mx.array, axis: int = -1) -> mx.array:
    """Computes softmax for an MLX array."""
    max_val = mx.max(x, axis=axis, keepdims=True)
    exp_x = mx.exp(x - max_val)
    sum_exp_x = mx.sum(exp_x, axis=axis, keepdims=True)
    return exp_x / sum_exp_x


def log_softmax(x: mx.array, axis: int = -1) -> mx.array:
    """Computes log-softmax for an MLX array."""
    max_val = mx.max(x, axis=axis, keepdims=True)
    exp_x = mx.exp(x - max_val)
    sum_exp_x = mx.sum(exp_x, axis=axis, keepdims=True)
    return x - max_val - mx.log(sum_exp_x)
