# file_path: mlx_rl_trainer/src/mlx_rl_trainer/generation/__init__.py
# revision_no: 001
# goals_of_writing_code_block: Initialize the generation module.
# type_of_code_response: add new code
"""Text generation utilities."""
from .generator import (
    generate_rollouts_for_batch,
)  # FIX: Change to direct import for rollout generator

from .sampler_utils import (
    make_dynamic_tag_bias_processor,
    safe_make_sampler,
    selective_softmax,
)  # Import sampler specific utilities

# Add common generation types/helpers here for easier import
__all__ = [
    "generate_rollouts_for_batch",
    "make_dynamic_tag_bias_processor",
    "safe_make_sampler",
    "selective_softmax",
]
