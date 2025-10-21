#!/usr/bin/env python3
# File: src/mlx_rl_trainer/algorithms/grpo/grpo_trainer.py
# Purpose: GRPO Trainer with fixed WandB logging, memory management, and LoRA support
# FIXED: TypeError - MLX arrays not JSON serializable in WandB logging

import logging
import time
import gc
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map

from mlx_rl_trainer.core.trainer import BaseTrainer, TrainingMetrics, EvaluationMetrics
from mlx_rl_trainer.monitoring.progress import (
    EnhancedProgressManager,
    ProgressBarConfig,
    DisplayMode
)
from mlx_rl_trainer.utils.mlx_utils import _maybe_clip_grad_norm
from mlx_rl_trainer.generation.generator import generate_rollouts_for_batch
from .grpo_algorithm import GRPOAlgorithm
from mlx.utils import tree_flatten, tree_unflatten

logger = logging.getLogger(__name__)

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning(
        "psutil not installed. Memory monitoring limited. Run 'pip install psutil'."
    )


def _convert_to_serializable(obj: Any) -> Any:
    """
    Recursively convert MLX arrays and numpy arrays to Python scalars.

    Args:
        obj: Object to convert

    Returns:
        JSON-serializable object
    """
    if isinstance(obj, mx.array):
        if obj.size == 1:
            return float(obj.item())
        else:
            return float(mx.mean(obj).item())
    elif isinstance(obj, np.ndarray):
        if obj.size == 1:
            return float(obj.item())
        else:
            return float(np.mean(obj))
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    else:
        # Try to convert to string as fallback
        try:
            return str(obj)
        except:
            return None


def safe_tree_add(tree1, tree2):
    """Safely add two gradient trees."""
    if not tree1:
        return tree2
    if not tree2:
        return tree1

    flat1 = tree_flatten(tree1)
    dict2 = dict(tree_flatten(tree2))
    result = []

    for path, grad1 in flat1:
        if path in dict2:
            result.append((path, grad1 + dict2.pop(path)))
        else:
            result.append((path, grad1))

    result.extend(dict2.items())
    del flat1, dict2
    return tree_unflatten(result)


@dataclass
class TokenTracker:
    """Track token statistics during training."""

    total_tokens: int = 0
    thinking_tokens: int = 0
    answer_tokens: int = 0
    dual_gradient_tokens: int = 0
    standard_gradient_tokens: int = 0
    layer_wise_tokens: Dict[int, int] = field(default_factory=dict)

    def update(self, thinking: int, answer: int, is_dual: bool):
        """Update token counts."""
        self.thinking_tokens += thinking
        self.answer_tokens += answer
        total = thinking + answer
        self.total_tokens += total

        if is_dual:
            self.dual_gradient_tokens += total
        else:
            self.standard_gradient_tokens += total

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        total = max(self.total_tokens, 1)
        return {
            "tokens/total": self.total_tokens,
            "tokens/thinking": self.thinking_tokens,
            "tokens/answer": self.answer_tokens,
            "tokens/dual_gradient": self.dual_gradient_tokens,
            "tokens/standard_gradient": self.standard_gradient_tokens,
            "tokens/thinking_pct": self.thinking_tokens / total * 100,
            "tokens/answer_pct": self.answer_tokens / total * 100,
        }


class MemoryMonitor:
    """Monitor memory usage and detect leaks."""

    def __init__(self, safety_threshold_mb: float = 2048.0):
        self.safety_threshold_mb = safety_threshold_mb
        self.history = []
        self.max_length = 50

    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        stats = {}

        try:
            stats["mlx_cache_mb"] = mx.get_cache_memory() / 1024**2
            stats["mlx_active_mb"] = mx.get_active_memory() / 1024**2
            stats["mlx_peak_mb"] = mx.get_peak_memory() / 1024**2
        except Exception as e:
            logger.debug(f"Could not get MLX memory stats: {e}")

        if PSUTIL_AVAILABLE:
            mem = psutil.virtual_memory()
            stats["system_available_mb"] = mem.available / 1024**2
            stats["system_total_mb"] = mem.total / 1024**2
            stats["system_used_pct"] = mem.percent
        else:
            stats["system_available_mb"] = float("inf")

        stats["active_mb"] = stats.get("mlx_active_mb", 0.0)
        return stats

    def check_safety(self, ref_completion_length: int = 0) -> Tuple[bool, str]:
        """Check if memory usage is safe."""
        stats = self.get_memory_stats()
        self.record(stats)

        if not stats:
            return True, "Could not get memory stats"

        available = stats.get("system_available_mb", float("inf"))
        if available < self.safety_threshold_mb:
            return (
                False,
                f"Low system memory: {available:.1f}MB available, threshold is {self.safety_threshold_mb:.1f}MB",
            )

        if ref_completion_length > 2000:
            return False, f"Long reference completion: {ref_completion_length} tokens"

        trend = self.get_trend()
        cache_mb = stats.get("mlx_cache_mb", 0)
        if trend and trend.startswith("INCREASING") and cache_mb > 1024:
            return False, f"Possible MLX memory leak: Cache memory is {trend}"

        return True, "OK"

    def record(self, stats: Dict[str, float]):
        """Record memory statistics."""
        if stats:
            self.history.append({**stats, "timestamp": time.time()})
            if len(self.history) > self.max_length:
                del self.history[0]

    def get_trend(self) -> str:
        """Get memory usage trend."""
        if len(self.history) < 20:
            return "INSUFFICIENT_DATA"

        key = "mlx_cache_mb"
        if not all(key in s for s in self.history[-20:]):
            return "INSUFFICIENT_DATA"

        recent = [s.get(key, 0) for s in self.history[-10:]]
        older = [s.get(key, 0) for s in self.history[-20:-10]]

        recent_avg = np.mean(recent)
        older_avg = np.mean(older)
        base = max(older_avg, 1.0)
        change = (recent_avg - older_avg) / base * 100

        if change > 20:
            return f"INCREASING ({change:+.1f}%)"
        elif change < -20:
            return f"DECREASING ({change:+.1f}%)"
        else:
            return f"STABLE ({change:+.1f}%)"


def terminal_alert(message: str, level: str = "INFO"):
    """Display terminal alert with color."""
    try:
        sys.stdout.write("\a")
        sys.stdout.flush()

        colors = {
            "INFO": "\033[94m",
            "WARNING": "\033[93m",
            "ERROR": "\033[91m",
            "RESET": "\033[0m",
        }

        color = colors.get(level, colors["INFO"])
        reset = colors["RESET"]

        width = min(len(message) + 4, 80)
        print(f"\n{color}{'='*width}{reset}")
        print(f"{color}  {message}{reset}")
        print(f"{color}{'='*width}{reset}\n")
    except:
        pass


class GRPOTrainer(BaseTrainer):
    """GRPO Trainer with comprehensive WandB logging and memory management."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # WandB initialization
        self.wandb = None
        if hasattr(self.config, "monitoring") and self.config.monitoring.use_wandb:
            try:
                import wandb

                self.wandb = wandb
                self._init_wandb()
            except ImportError:
                logger.warning("WandB not installed. Install with: pip install wandb")

        # Tracking
        self.token_tracker = TokenTracker()
        self.memory_monitor = MemoryMonitor(
            safety_threshold_mb=getattr(
                self.config.trainer, "memory_safety_threshold_mb", 1000
            )
        )

        # Chart data
        self.chart_data = {
            "loss_history": [],
            "reward_history": [],
            "memory_history": [],
            "token_history": [],
            "gradient_norms": {},
        }
        self._max_history = 100

        # Error tracking
        self.metal_error_count = 0
        self.max_metal_errors = 3
        
        # Enhanced progress bar system (inherits from BaseTrainer)
        self._grpo_training_session = None
        self._setup_grpo_progress_bar()

    def _setup_grpo_progress_bar(self):
        """Setup GRPO-specific progress bar configuration."""
        try:
            # Initialize progress manager if available
            if hasattr(self, 'progress_manager') and self.progress_manager:
                # Configure progress bar for GRPO training
                from mlx_rl_trainer.monitoring.progress import ProgressBarConfig, DisplayMode
                
                grpo_config = ProgressBarConfig(
                    display_mode=DisplayMode.DETAILED,
                    show_gradient_norm=True,
                    show_memory_usage=True,
                    show_token_stats=True,
                    update_frequency=1.0,
                    enable_dual_gradient_tracking=getattr(
                        self.config.trainer, "use_dual_gradients", False
                    ),
                    enable_layer_wise_tracking=True
                )
                
                # Configure the progress manager
                self.progress_manager.configure(grpo_config)
                logger.info("GRPO progress bar system configured successfully")
                
            else:
                logger.debug("Progress manager not available - using basic progress tracking")
                
        except Exception as e:
            logger.warning(f"Failed to setup GRPO progress bar: {e}")
            # Don't fail initialization if progress bar setup fails
            self._grpo_training_session = None

    def _init_wandb(self):
        """Initialize WandB with comprehensive configuration."""
        if not self.wandb:
            return

        try:
            # Prepare config
            config = {
                "model": str(self.config.model.model_path),
                "learning_rate": self.config.trainer.learning_rate,
                "batch_size": self.config.trainer.ppo_batch_size,
                "grad_accum_steps": self.config.trainer.grad_accum_steps,
                "use_dual_gradients": getattr(
                    self.config.trainer, "use_dual_gradients", False
                ),
                "use_sft_on_answer": getattr(
                    self.config.trainer, "use_sft_on_answer", False
                ),
                "thinking_layer_start": getattr(
                    self.config.trainer, "thinking_layer_start", 22
                ),
                "thinking_layer_end": getattr(
                    self.config.trainer, "thinking_layer_end", 30
                ),
                "answer_layer_end": getattr(
                    self.config.trainer, "answer_layer_end", 36
                ),
                "use_lora": self.config.model.use_lora,
                "lora_rank": self.config.model.lora_rank
                if self.config.model.use_lora
                else None,
            }

            # Initialize WandB
            self.wandb.init(
                project=self.config.monitoring.wandb_project,
                name=self.config.monitoring.wandb_run_name or self._run_id,
                config=config,
                tags=getattr(self.config.monitoring, "tags", []),
            )

            # Define custom charts
            self._define_wandb_charts()

            logger.info(f"WandB initialized: {self.wandb.run.url}")
        except Exception as e:
            logger.error(f"Failed to initialize WandB: {e}")
            self.wandb = None

    def _define_wandb_charts(self):
        """Define custom WandB charts for better visualization."""
        if not self.wandb or not self.wandb.run:
            return

        try:
            # Define step metric
            self.wandb.define_metric("step")

            # Define all metrics with step
            self.wandb.define_metric("memory/*", step_metric="step")
            self.wandb.define_metric("tokens/*", step_metric="step")
            self.wandb.define_metric("loss/*", step_metric="step")
            self.wandb.define_metric("gradients/layer_*", step_metric="step")
            self.wandb.define_metric("training/*", step_metric="step")
            self.wandb.define_metric("generation/*", step_metric="step")
            self.wandb.define_metric("rewards/*", step_metric="step")

            logger.info("WandB custom charts defined")
        except Exception as e:
            logger.warning(f"Could not define WandB charts: {e}")

    def _setup(self):
        """Setup models and optimizer."""
        # Load models
        self.actor_model, self.tokenizer = self.model_manager.load_model(
            self.config.model.model_path,
            "actor",
            is_trainable=True,
            apply_lora=self.config.model.use_lora,
            lora_config=self.config.model.model_dump(),
        )

        self.ref_model, _ = self.model_manager.load_model(
            self.config.model.ref_model_path, "reference", is_trainable=False
        )

        # Initialize algorithm
        self.grpo_algorithm = GRPOAlgorithm(
            self.config, self.actor_model, self.ref_model
        )

        # Log compilation info
        if getattr(self.config.trainer, "use_compile", True):
            logger.info(
                "MLX graph compilation enabled - first iteration will compile functions"
            )
            logger.info(
                "Compilation provides ~2-5x speedup and better memory efficiency"
            )
            logger.info("Set config.trainer.use_compile=False to disable for debugging")

        # Setup optimizer
        self.optimizer = optim.AdamW(
            learning_rate=self.config.trainer.learning_rate,
            betas=(
                self.config.trainer.optimizer_beta1,
                self.config.trainer.optimizer_beta2,
            ),
            weight_decay=self.config.trainer.optimizer_weight_decay,
        )

        from mlx_lm.tuner.utils import build_schedule

        self.lr_scheduler = build_schedule(self.config.trainer.lr_schedule_config)

        # Load checkpoint
        resume_step, metadata = self.checkpoint_manager.load_latest_state(
            self.actor_model, self.optimizer
        )

        # Apply gradient checkpointing if configured
        if self.config.use_grad_checkpointing:
            logger.info("Applying gradient checkpointing to transformer layers...")
            try:
                from mlx_lm.tuner.trainer import grad_checkpoint

                model = getattr(self.actor_model, "model", self.actor_model)
                if hasattr(model, "layers") and isinstance(model.layers, list):
                    count = 0
                    for layer in model.layers:
                        if (
                            self.config.grad_checkpoint_layers
                            and count < self.config.grad_checkpoint_layers
                        ):
                            grad_checkpoint(layer)
                            count += 1
                    logger.info(f"Gradient checkpointing applied to {count} layers")
            except Exception as e:
                logger.error(
                    f"Failed to apply gradient checkpointing: {e}", exc_info=True
                )

        return metadata.get("num_updates", 0), metadata.get("epoch", 0)

    def _track_layer_gradients(self, grads, step: int):
        """Track gradient norms by layer for visualization."""
        layer_norms = {}

        for path, grad in tree_flatten(grads):
            if "layers." in path:
                import re

                match = re.search(r"layers\.(\d+)\.", path)
                if match:
                    layer_idx = int(match.group(1))
                    if isinstance(grad, mx.array):
                        norm = float(mx.sqrt(mx.sum(grad**2)).item())
                        key = f"layer_{layer_idx}"
                        if key not in layer_norms:
                            layer_norms[key] = []
                        layer_norms[key].append(norm)

        # Compute average norm per layer
        metrics = {}
        for key, norms in layer_norms.items():
            avg_norm = np.mean(norms)
            metrics[f"gradients/{key}_norm"] = avg_norm

            # Track history for charts
            if key not in self.chart_data["gradient_norms"]:
                self.chart_data["gradient_norms"][key] = []
            history = self.chart_data["gradient_norms"][key]
            history.append({"step": step, "norm": avg_norm})
            if len(history) > self._max_history:
                del history[0]

        del layer_norms
        return metrics

    def _check_pre_iteration_safety(
        self, batch_data: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check memory safety before iteration."""
        # Get memory stats
        mem_stats = self.memory_monitor.get_memory_stats()
        self.memory_monitor.record(mem_stats)

        # Check reference completion length
        max_ref_len = 0
        if "prompts_data" in batch_data:
            for sample in batch_data["prompts_data"]:
                ref_ans = sample.get("ref_answer_str", "")
                if ref_ans:
                    ref_len = len(self.tokenizer.encode(ref_ans))
                    max_ref_len = max(max_ref_len, ref_len) * 6

        # Check safety
        is_safe, reason = self.memory_monitor.check_safety(max_ref_len)

        # Get trend
        trend = self.memory_monitor.get_trend()
        if trend:
            logger.debug(f"Memory trend: {trend}")

        return is_safe, reason

    def _save_checkpoint_with_retry(
        self,
        step: int,
        reason: str = "regular",
        is_final: bool = False,
        max_retries: int = 3,
    ):
        """Save checkpoint with retry logic."""
        for attempt in range(max_retries):
            try:
                # Get memory stats
                mem_stats = self.memory_monitor.get_memory_stats()

                # Prepare metadata
                metadata = {
                    "num_updates": step,
                    "epoch": self.current_epoch,
                    "reason": reason,
                    "log_id": self._run_id,
                    "save_optimizer_state": self.config.checkpointing.save_optimizer_state,
                    "memory_peak_mb": self.memory_peak,
                    "memory_current_mb": mem_stats.get("mlx_active_mb", 0),
                    "token_stats": self.token_tracker.to_dict(),
                }

                # Save checkpoint
                self.checkpoint_manager.save_checkpoint(
                    step=step,
                    model=self.actor_model,
                    optimizer=self.optimizer,
                    metadata=metadata,
                    current_metric=self.checkpoint_manager.best_metric,
                )

                logger.info(f"✓ Checkpoint saved successfully (reason: {reason})")
                return True

            except Exception as e:
                delay = 2**attempt
                logger.error(
                    f"Checkpoint save failed (attempt {attempt+1}/{max_retries}): {e}"
                )

                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {delay}s...")
                    terminal_alert(
                        f"Checkpoint save failed! Retrying in {delay}s... (attempt {attempt+1}/{max_retries})",
                        level="WARNING",
                    )
                    time.sleep(delay)
                    gc.collect()
                    try:
                        mx.metal.clear_cache()
                    except:
                        pass
                else:
                    msg = f"CRITICAL: Checkpoint save failed after {max_retries} attempts!\nError: {str(e)}\nCheck: disk space, permissions, path validity"
                    terminal_alert(msg, level="ERROR")

                    # Log error
                    error_log = (
                        Path(self.config.checkpointing.checkpoint_dir)
                        / "checkpoint_errors.log"
                    )
                    try:
                        with open(error_log, "a") as f:
                            f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
                    except:
                        pass

                    return False

        return False

    def _generate_charts(self, step: int):
        """Generate training progress charts with memory efficiency."""
        try:
            import matplotlib as mpl

            mpl.use("Agg")
            import matplotlib.pyplot as plt

            output_dir = self.config.trainer.output_dir / "charts"
            output_dir.mkdir(exist_ok=True, parents=True)

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f"Training Progress - Step {step}", fontsize=16)

            # Loss plot
            if self.chart_data["loss_history"]:
                ax = axes[0, 0]
                steps = [entry["step"] for entry in self.chart_data["loss_history"]]
                losses = [entry["loss"] for entry in self.chart_data["loss_history"]]
                ax.plot(steps, losses, "b-", linewidth=2)
                ax.set_xlabel("Step")
                ax.set_ylabel("Loss")
                ax.set_title("Training Loss")
                ax.grid(True, alpha=0.3)

            # Reward plot
            if self.chart_data["reward_history"]:
                ax = axes[0, 1]
                steps = [entry["step"] for entry in self.chart_data["reward_history"]]
                rewards = [
                    entry["reward"] for entry in self.chart_data["reward_history"]
                ]
                ax.plot(steps, rewards, "g-", linewidth=2)
                ax.set_xlabel("Step")
                ax.set_ylabel("Reward")
                ax.set_title("Average Reward")
                ax.grid(True, alpha=0.3)

            # Memory plot
            if self.chart_data["memory_history"]:
                ax = axes[1, 0]
                steps = [entry["step"] for entry in self.chart_data["memory_history"]]
                active = [
                    entry.get("mlx_active_mb", 0)
                    for entry in self.chart_data["memory_history"]
                ]
                peak = [
                    entry.get("mlx_peak_mb", 0)
                    for entry in self.chart_data["memory_history"]
                ]
                ax.plot(steps, active, "r-", label="Active MLX", linewidth=2)
                ax.plot(steps, peak, "r--", label="Peak MLX", linewidth=1, alpha=0.5)
                ax.set_xlabel("Step")
                ax.set_ylabel("Memory (MB)")
                ax.set_title("Memory Usage")
                ax.legend()
                ax.grid(True, alpha=0.3)

            # Token distribution plot
            if self.chart_data["token_history"]:
                ax = axes[1, 1]
                steps = [entry["step"] for entry in self.chart_data["token_history"]]
                thinking = [
                    entry["thinking"] for entry in self.chart_data["token_history"]
                ]
                answer = [entry["answer"] for entry in self.chart_data["token_history"]]
                ax.plot(steps, thinking, "b-", label="Thinking", linewidth=2)
                ax.plot(steps, answer, "orange", label="Answer", linewidth=2)
                ax.set_xlabel("Step")
                ax.set_ylabel("Token Count")
                ax.set_title("Token Distribution")
                ax.legend()
                ax.grid(True, alpha=0.3)

            plt.tight_layout()

            chart_path = output_dir / f"training_chart_step_{step}.png"
            plt.savefig(chart_path, dpi=150, bbox_inches="tight")
            plt.close()

            logger.info(f"Chart saved: {chart_path}")

            if self.wandb and self.wandb.run:
                self.wandb.log(
                    {"training_chart": self.wandb.Image(str(chart_path))}, step=step
                )

            del fig, axes

        except Exception as e:
            logger.warning(f"Could not generate charts: {e}")

    def _aggressive_memory_cleanup(self):
        """Aggressive memory cleanup."""
        try:
            mx.metal.clear_cache()
        except:
            pass
        mx.clear_cache()
        gc.collect()
        for _ in range(3):
            gc.collect()

    def _compute_global_grad_norm(self, grads) -> float:
        """Compute global gradient norm."""
        if not grads:
            return 0.0

        total = mx.array(0.0)
        for path, grad in tree_flatten(grads):
            if isinstance(grad, mx.array):
                total += mx.sum(grad**2)

        norm = float(mx.sqrt(total).item())
        del total
        return norm

    def train_step(self, rollout_batch: Dict[str, Any], update_step: int):
        """
        Enhanced training step with comprehensive logging and dual gradient support.

        FIXED: Convert all MLX arrays to Python scalars before WandB logging.
        """
        step_start = time.time()
        batch = rollout_batch
        metrics = {}

        # Check if we have thinking/answer masks for dual gradients
        has_dual_masks = "thinking_mask" in batch and "answer_mask" in batch
        use_sft = (
            has_dual_masks
            and hasattr(self.config.trainer, "use_sft_on_answer")
            and self.config.trainer.use_sft_on_answer
            and "reference_tokens" in batch
        )

        # Track tokens
        if has_dual_masks:
            thinking_tokens = int(mx.sum(batch["thinking_mask"]).item())
            answer_tokens = int(mx.sum(batch["answer_mask"]).item())
            self.token_tracker.update(thinking_tokens, answer_tokens, True)
        else:
            total_tokens = int(mx.sum(batch.get("response_mask", mx.array([0]))).item())
            self.token_tracker.update(total_tokens // 2, total_tokens // 2, False)

        # Import gradient utilities
        from mlx_rl_trainer.algorithms.grpo.grpo_algorithm import (
            _safe_gradient_combine,
            _validate_gradient_dict,
        )

        # Initialize gradient accumulation
        accumulated_grads = None
        total_loss = 0.0

        # Compute gradients with dual gradient support
        if (
            has_dual_masks
            and hasattr(self.config.trainer, "use_dual_gradients")
            and self.config.trainer.use_dual_gradients
        ):
            # Dual gradient computation
            (
                thinking_loss,
                thinking_grads,
                answer_loss,
                answer_grads,
                loss_info,
                grad_info,
            ) = self.grpo_algorithm.calculate_dual_gradient_loss(
                batch, self.config, self.tokenizer.pad_token_id
            )

            if not grad_info["success"]:
                logger.warning(
                    f"Dual gradient computation issues: {grad_info['structure_issues']}"
                )

            # ⭐ FIX: Use .get() for safe access to 'match_rate'
            if grad_info.get("match_rate", 1.0) < 0.8:
                logger.warning(
                    f"Low gradient structure match rate: {grad_info.get('match_rate', 0.0):.1%}"
                )

            # Get layer ranges and weights
            thinking_start = getattr(self.config.trainer, "thinking_layer_start", 0)
            thinking_end = getattr(self.config.trainer, "thinking_layer_end", 15)
            answer_start = getattr(self.config.trainer, "answer_layer_start", 16)
            answer_end = getattr(self.config.trainer, "answer_layer_end", 31)

            thinking_weight = getattr(
                self.config.trainer, "thinking_gradient_weight", 0.5
            )
            answer_weight = getattr(self.config.trainer, "answer_gradient_weight", 1.0)

            logger.debug(
                f"Dual gradients - Thinking: layers {thinking_start}-{thinking_end} (weight={thinking_weight}), "
                f"Answer: layers {answer_start}-{answer_end} (weight={answer_weight})"
            )

            # Combine gradients
            accumulated_grads, combine_info = _safe_gradient_combine(
                thinking_grads,
                answer_grads,
                operation="add",
                weight1=thinking_weight,
                weight2=answer_weight,
            )

            if not combine_info["success"]:
                logger.error(
                    f"Gradient combination failed: {combine_info['structure_issues']}"
                )
                logger.warning("Falling back to thinking gradients only")
                accumulated_grads = thinking_grads
            elif combine_info.get("match_rate", 0.0) < 0.8:
                logger.warning(
                    f"Gradient combination partial match: {combine_info['match_rate']:.1%}"
                )

            # Compute total loss
            total_loss = (
                thinking_weight * thinking_loss + answer_weight * answer_loss
            ) / (thinking_weight + answer_weight)

            # Track token usage
            thinking_tok = mx.sum(batch["thinking_mask"]).item()
            answer_tok = mx.sum(batch["answer_mask"]).item()
            total_tok = thinking_tok + answer_tok

            if total_tok > 0:
                thinking_ratio = thinking_tok / total_tok
                answer_ratio = answer_tok / total_tok
            else:
                thinking_ratio = 0.5
                answer_ratio = 0.5

            metrics.update(
                {
                    "training/thinking_tokens": thinking_tok,
                    "training/answer_tokens": answer_tok,
                    "training/thinking_ratio": thinking_ratio,
                    "training/answer_ratio": answer_ratio,
                    "loss/thinking_loss": thinking_loss.item(),
                    "loss/answer_rl_loss": answer_loss.item(),
                }
            )

        else:
            # Standard gradient computation
            (
                loss_val,
                accumulated_grads,
                loss_info,
            ) = self.grpo_algorithm.calculate_loss_and_grads(
                batch, self.config, self.tokenizer.pad_token_id
            )
            total_loss = float(loss_val.item())

        # Add SFT gradients if enabled
        if (
            hasattr(self.config.trainer, "sft_weight")
            and self.config.trainer.sft_weight > 0
        ):
            if "reference_tokens" in batch:
                (
                    sft_loss,
                    sft_grads,
                    sft_info,
                ) = self.grpo_algorithm.calculate_sft_loss_and_grads(
                    batch,
                    batch["reference_tokens"],
                    self.config,
                    self.tokenizer.pad_token_id,
                )
                loss_info.update(sft_info)

                sft_weight = self.config.trainer.sft_weight
                accumulated_grads, sft_combine_info = _safe_gradient_combine(
                    accumulated_grads,
                    sft_grads,
                    operation="add",
                    weight1=1.0,
                    weight2=sft_weight,
                )

                if not sft_combine_info["success"]:
                    logger.error(
                        f"SFT gradient combination failed: {sft_combine_info['structure_issues']}"
                    )
                    logger.warning("Using RL gradients only")
                else:
                    if sft_combine_info.get("match_rate", 0.0) < 0.8:
                        logger.warning(
                            f"SFT combination partial match: {sft_combine_info['match_rate']:.1%}"
                        )

                total_loss = total_loss + sft_weight * float(sft_loss.item())
                metrics["loss/answer_sft_loss"] = float(sft_loss.item())
            else:
                logger.warning("SFT enabled but no reference_tokens in batch")

        # Validate final gradients
        if not _validate_gradient_dict(accumulated_grads, "final gradients"):
            logger.error("Final gradients are invalid! Skipping optimizer update.")
            return (
                TrainingMetrics(
                    loss=float(total_loss),
                    learning_rate=self.optimizer.learning_rate.item()
                    if hasattr(self.optimizer.learning_rate, "item")
                    else self.optimizer.learning_rate,
                    grad_norm=0.0,
                ),
                {},
                loss_info,
            )

        # Apply gradient accumulation scaling
        accum_steps = self.config.trainer.grad_accum_steps
        if accum_steps > 1:
            accumulated_grads = tree_map(lambda g: g / accum_steps, accumulated_grads)

        # Update optimizer
        self.optimizer.update(self.actor_model, accumulated_grads)
        mx.eval(self.actor_model.parameters(), self.optimizer.state)

        # Compute gradient norm
        grad_norm = self._compute_global_grad_norm(accumulated_grads)

        # Track layer-wise gradients
        layer_grad_metrics = self._track_layer_gradients(accumulated_grads, update_step)

        # Prepare comprehensive metrics
        metrics.update(
            {
                "loss/total": total_loss,
                "training/reward_mean": batch["advantages"].mean().item(),
                "training/reward_std": batch["advantages"].std().item(),
                "training/learning_rate": self.lr_scheduler(update_step),
                "training/kl_divergence": loss_info.get("kl_divergence", 0.0),
                "training/grad_norm": grad_norm,
                "training/step_time_s": time.time() - step_start
            }
        )
        metrics.update(layer_grad_metrics)

        # Log to WandB with proper serialization
        if self.wandb and self.wandb.run:
            wandb_metrics = {**metrics, "step": update_step}
            wandb_metrics.update(self.token_tracker.to_dict())

            # Add memory stats
            mem_stats = self.memory_monitor.get_memory_stats()
            for key, val in mem_stats.items():
                wandb_metrics[f"memory/{key}"] = val

            # CRITICAL FIX: Convert all arrays to serializable types
            wandb_metrics = _convert_to_serializable(wandb_metrics)

            try:
                self.wandb.log(_convert_to_serializable(wandb_metrics))
            except Exception as e:
                logger.error(f"Failed to log to WandB: {e}")
                # Try logging just the basics
                try:
                    basic_metrics = {
                        "step": update_step,
                        "loss/total": float(total_loss),
                        "training/reward_mean": float(metrics["training/reward_mean"]),
                    }
                    self.wandb.log(_convert_to_serializable(basic_metrics))
                except:
                    logger.error("Even basic WandB logging failed")

        # Update chart data
        self.chart_data["loss_history"].append(
            {"step": update_step, "loss": total_loss}
        )
        if len(self.chart_data["loss_history"]) > self._max_history:
            del self.chart_data["loss_history"][0]

        self.chart_data["reward_history"].append(
            {"step": update_step, "reward": metrics["training/reward_mean"]}
        )
        if len(self.chart_data["reward_history"]) > self._max_history:
            del self.chart_data["reward_history"][0]

        if has_dual_masks:
            self.chart_data["token_history"].append(
                {
                    "step": update_step,
                    "thinking": metrics.get("training/thinking_tokens", 0),
                    "answer": metrics.get("training/answer_tokens", 0),
                }
            )
            if len(self.chart_data["token_history"]) > self._max_history:
                del self.chart_data["token_history"][0]

        # Create training metrics object
        training_metrics = TrainingMetrics(
            loss=total_loss,
            reward_mean=metrics["training/reward_mean"],
            reward_std=metrics["training/reward_std"],
            grad_norm=grad_norm,
            learning_rate=metrics["training/learning_rate"],
            step_time_s=metrics["training/step_time_s"],
            kl_divergence=metrics["training/kl_divergence"],
            epoch=self.current_epoch,
            step=update_step,
        )

        return training_metrics, accumulated_grads, metrics

    def generate_rollouts(self, batch_data: Dict[str, Any], update_step: int):
        """Generate rollouts with error handling."""
        try:
            prompts_data = batch_data.get("prompts_data", [])
            is_invalid = any(p.get("is_invalid_sample", False) for p in prompts_data)

            (
                rollout_batch,
                avg_reward,
                reward_breakdown,
                gen_metrics,
            ) = generate_rollouts_for_batch(
                model=self.actor_model,
                ref_model=self.ref_model,
                tokenizer=self.tokenizer,
                prompts_data=prompts_data,
                dataset=self.data_manager._train_dataset,
                config=self.config,
                reward_composer=self.reward_composer,
                run_id=self._run_id,
                current_update=update_step,
                is_invalid_batch=is_invalid,
            )

            return rollout_batch, avg_reward, reward_breakdown, gen_metrics

        except RuntimeError as e:
            if "METAL" in str(e) or "Command buffer" in str(e):
                self.metal_error_count += 1
                logger.error(
                    f"Metal error in generation ({self.metal_error_count}/{self.max_metal_errors}): {e}"
                )

                gc.collect()
                try:
                    mx.metal.clear_cache()
                except:
                    pass

                if self.metal_error_count >= self.max_metal_errors:
                    terminal_alert(
                        "CRITICAL: Multiple Metal errors. Saving checkpoint and exiting.",
                        level="ERROR",
                    )
                    self._save_checkpoint_with_retry(update_step, reason="metal_error")
                    raise

                return {}, 0.0, {}, {}
            else:
                raise

    def log_comprehensive_metrics(
        self,
        step: int,
        step_metrics: Dict[str, Any],
        generation_metrics: Optional[Dict[str, Any]] = None,
    ):
        """Log comprehensive metrics to all monitoring systems."""
        # Get memory stats
        mem_stats = self.memory_monitor.get_memory_stats()
        if mem_stats:
            mem_metrics = {f"memory/{k}": v for k, v in mem_stats.items()}
            step_metrics.update(mem_metrics)

            # Update chart data
            self.chart_data["memory_history"].append({"step": step, **mem_stats})
            if len(self.chart_data["memory_history"]) > self._max_history:
                del self.chart_data["memory_history"][0]

        # Add token stats
        token_stats = self.token_tracker.to_dict()
        step_metrics.update(token_stats)

        # Add generation metrics
        if generation_metrics:
            step_metrics.update(generation_metrics)

        # Log to WandB with serialization
        if self.wandb and self.wandb.run:
            wandb_metrics = {**step_metrics, "step": step}
            wandb_metrics = _convert_to_serializable(wandb_metrics)

            try:
                self.wandb.log(wandb_metrics)
            except Exception as e:
                logger.error(f"Failed to log to WandB: {e}")

        # Log to metrics logger
        if self.metrics_logger:
            self.metrics_logger.log_metrics(step_metrics, step=step)

    async def run(self, should_shutdown):
        """Main training loop with aggressive memory management."""
        start_step, self.current_epoch = self._setup()

        if start_step > 0:
            self.global_step = start_step + 1
            logger.info(f"Resumed from checkpoint at step {start_step}")
        else:
            self.global_step = 0
            logger.info("Starting training from scratch")

        if self.tokenizer:
            self.data_manager.set_tokenizer(self.tokenizer)

        await self.data_manager.load_datasets()

        # Initialize enhanced progress bar for GRPO
        if self.progress_manager:
            try:
                self._grpo_training_session = self.progress_manager.start_training_session(
                    total_steps=self.config.trainer.num_training_steps,
                    session_name=f"GRPO Training - {self._run_id}"
                )
                logger.info("Enhanced GRPO progress tracking started")
            except Exception as e:
                logger.warning(f"Failed to start enhanced GRPO progress session: {e}")
                self._grpo_training_session = None

        from tqdm import trange

        progress_bar = trange(
            self.global_step,
            self.config.trainer.num_training_steps,
            initial=self.global_step,
            desc="Training",
            unit="step",
        )

        dataloader_iter = iter([])
        grad_accum_steps = self.config.trainer.grad_accum_steps

        with progress_bar:
            while self.global_step < self.config.trainer.num_training_steps:
                if should_shutdown():
                    logger.info("Shutdown requested")
                    break

                try:
                    # Get batch
                    try:
                        batch = next(dataloader_iter)
                    except StopIteration:
                        self.current_epoch += 1
                        logger.info(f"Epoch {self.current_epoch}")

                        dataloader_iter = iter(
                            self.data_manager.get_dataloader(
                                "train", self.config.trainer.ppo_batch_size
                            )
                        )
                        batch = next(dataloader_iter)

                    # Safety check
                    is_safe, safety_msg = self._check_pre_iteration_safety(batch)
                    if not is_safe:
                        logger.warning(f"Safety check failed: {safety_msg}")
                        terminal_alert(
                            f"Safety checkpoint triggered: {safety_msg}",
                            level="WARNING",
                        )

                        self._save_checkpoint_with_retry(
                            self.global_step, reason="safety"
                        )

                        gc.collect()
                        try:
                            mx.metal.clear_cache()
                        except:
                            pass

                        time.sleep(2)

                    # Gradient accumulation
                    accumulated_grads = None
                    total_loss = 0.0
                    total_reward = 0.0
                    num_valid_steps = 0
                    combined_gen_metrics = {}

                    for accum_idx in range(grad_accum_steps):
                        # Generate rollouts
                        (
                            rollout_batch,
                            avg_reward,
                            gen_metrics,
                            reward_details,
                        ) = self.generate_rollouts(batch, self.global_step)

                        if not rollout_batch or "tokens" not in rollout_batch:
                            logger.warning(f"Empty rollout at step {self.global_step}")
                            continue

                        # Train step
                        train_metrics, step_grads, detailed_metrics = self.train_step(
                            rollout_batch, self.global_step
                        )

                        total_loss += train_metrics.loss
                        total_reward += avg_reward
                        num_valid_steps += 1

                        # Accumulate generation metrics
                        if gen_metrics:
                            for key, value in gen_metrics.items():
                                combined_gen_metrics[key] = (
                                    combined_gen_metrics.get(key, 0.0) + value
                                )

                        # Accumulate gradients
                        if step_grads:
                            mx.eval(step_grads)
                            if accumulated_grads is None:
                                accumulated_grads = step_grads
                            else:
                                accumulated_grads = tree_map(
                                    mx.add, accumulated_grads, step_grads
                                )

                            mx.eval(accumulated_grads)

                        # Cleanup
                        del rollout_batch, train_metrics, step_grads
                        gc.collect()

                    # Apply accumulated gradients
                    if accumulated_grads and num_valid_steps > 0:
                        self.optimizer.learning_rate = self.lr_scheduler(
                            self.global_step
                        )
                        self.optimizer.apply_gradients(
                            accumulated_grads, self.actor_model.trainable_parameters()
                        )
                        mx.eval(self.actor_model.parameters())

                        avg_loss = total_loss / num_valid_steps
                        avg_reward = total_reward / num_valid_steps

                        # Log metrics
                        self.log_comprehensive_metrics(
                            self.global_step, detailed_metrics, combined_gen_metrics
                        )

                        # Update enhanced progress bar
                        if self._grpo_training_session:
                            try:
                                grpo_metrics = {
                                    'loss': avg_loss,
                                    'reward_mean': avg_reward,
                                    'gradient_norm': self._compute_global_grad_norm(accumulated_grads) if accumulated_grads else 0.0,
                                    'learning_rate': float(self.lr_scheduler(self.global_step)),
                                    'total_tokens': self.token_tracker.total_tokens,
                                    'thinking_tokens': self.token_tracker.thinking_tokens,
                                    'answer_tokens': self.token_tracker.answer_tokens,
                                    'memory_usage_mb': self.memory_monitor.get_memory_stats().get('mlx_active_mb', 0),
                                    'step_time': time.time() - step_start_time if 'step_start_time' in locals() else 0.0
                                }
                                
                                # Add GRPO-specific metrics
                                if hasattr(self, 'chart_data') and self.chart_data.get('loss_history'):
                                    grpo_metrics['loss_trend'] = 'improving' if len(self.chart_data['loss_history']) > 1 and \
                                        self.chart_data['loss_history'][-1]['loss'] < self.chart_data['loss_history'][-2]['loss'] else 'stable'
                                
                                self._grpo_training_session.update_progress(
                                    step=self.global_step,
                                    metrics=grpo_metrics,
                                    gradients=accumulated_grads
                                )
                            except Exception as e:
                                logger.warning(f"Enhanced GRPO progress update failed: {e}")

                        # Update basic progress bar
                        progress_bar.set_postfix(
                            {
                                "Loss": f"{avg_loss:.4f}",
                                "Reward": f"{avg_reward:.3f}",
                                "Tokens": f"{self.token_tracker.total_tokens}",
                            }
                        )
                        progress_bar.update(1)

                        # Periodic checkpoint
                        should_save = (
                            self.config.checkpointing.save_every > 0
                            and (self.global_step + 1)
                            % self.config.checkpointing.save_every
                            == 0
                        )

                        if should_save:
                            self._generate_charts(self.global_step)
                            self._save_checkpoint_with_retry(
                                self.global_step, reason="regular"
                            )

                        del accumulated_grads

                    self.global_step += 1

                    # Periodic memory cleanup
                    if self.global_step % 10 == 0:
                        gc.collect()
                        try:
                            mx.metal.clear_cache()
                        except:
                            pass

                except RuntimeError as e:
                    if "METAL" in str(e) or "Command buffer" in str(e):
                        logger.error(
                            f"Metal command buffer error at step {self.global_step}: {e}"
                        )
                        terminal_alert(
                            "Metal error encountered! Saving checkpoint...",
                            level="ERROR",
                        )

                        self._save_checkpoint_with_retry(
                            self.global_step, reason="metal_error"
                        )

                        gc.collect()
                        try:
                            mx.metal.clear_cache()
                        except:
                            pass

                        # Reset error count periodically
                        if self.global_step % 10 == 0:
                            self.metal_error_count = 0

                        if self.metal_error_count < self.max_metal_errors:
                            logger.info("Attempting to continue training...")
                            time.sleep(5)
                            continue
                        else:
                            logger.error("Too many Metal errors. Exiting.")
                            break
                    else:
                        raise

        # Final checkpoint and cleanup
        self._generate_charts(self.global_step)
        self._save_checkpoint_with_retry(self.global_step, reason="final")

        if self.wandb and self.wandb.run:
            self.wandb.finish()

    def evaluate(self, update_step: int) -> List[EvaluationMetrics]:
        """Evaluate the model."""
        logger.info(f"Evaluation at step {update_step}")
        return []
