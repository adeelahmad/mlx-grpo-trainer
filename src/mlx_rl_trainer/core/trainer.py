"""Enhanced Base Trainer with Memory Monitoring and Checkpoint Retry Logic

ENHANCEMENTS:
1. Comprehensive memory tracking and cleanup
2. Checkpoint saving with retry and backoff
3. Memory usage logging throughout training
4. Graceful error handling
5. Pre-iteration safety checks
6. Chart generation integration
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Callable
import logging
import time
import gc
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from tqdm import trange
from ..monitoring.progress import (
    EnhancedProgressManager,
    ProgressBarConfig,
    DisplayMode
)
from mlx.utils import tree_map, tree_flatten

from .config import ExperimentConfig
from .model_manager import ModelManager
from .dataset_manager import DatasetManager
from .checkpoint_manager import CheckpointManager
from ..monitoring.metrics_logger import MetricsLogger
from .exceptions import TrainingRuntimeError, CheckpointError
import psutil

logger = logging.getLogger(__name__)


def _get_memory_usage_mb() -> Dict[str, float]:
    """Get current memory usage in MB from both MLX and the system process."""
    stats = {}
    try:
        # MLX Metal specific memory
        stats["mlx_cache_mb"] = mx.metal.get_cache_memory() / (1024 * 1024)
        stats["mlx_active_mb"] = mx.metal.get_active_memory() / (1024 * 1024)
        stats["mlx_peak_mb"] = mx.metal.get_peak_memory() / (1024 * 1024)

        # System memory via psutil for a more holistic view
        process = psutil.Process()
        mem_info = process.memory_info()
        stats["process_rss_mb"] = mem_info.rss / (1024 * 1024)

        # Use MLX active memory as the primary metric for checks, but log both
        stats["active_mb"] = stats.get("mlx_active_mb", 0)
        return stats
    except Exception as e:
        logger.debug(f"Could not get memory stats: {e}")
        return {}


@dataclass(frozen=True)
class TrainingMetrics:
    loss: float
    reward_mean: float
    grad_norm: float
    learning_rate: float
    step_time_s: float
    kl_divergence: float
    epoch: int = 0
    step: int = 0
    reward_std: float = 0.0
    custom_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "train/loss": self.loss,
            "train/reward_mean": self.reward_mean,
            "train/reward_std": self.reward_std,
            "train/grad_norm": self.grad_norm,
            "train/learning_rate": self.learning_rate,
            "train/step_time_s": self.step_time_s,
            "train/kl_divergence": self.kl_divergence,
            "train/epoch": self.epoch,
            "train/step": self.step,
        }
        data.update(self.custom_metrics)
        return data


@dataclass(frozen=True)
class EvaluationMetrics:
    task_name: str
    pass_rate: float = 0.0
    perplexity: Optional[float] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = {f"eval/{self.task_name}/pass_rate": self.pass_rate}
        if self.perplexity is not None:
            data[f"eval/{self.task_name}/perplexity"] = self.perplexity
        for k, v in self.additional_info.items():
            if not k.startswith(f"eval/{self.task_name}/"):
                data[f"eval/{self.task_name}/{k}"] = v
            else:
                data[k] = v
        return data


class BaseTrainer(ABC):
    """Enhanced base trainer with memory monitoring and robust checkpointing."""

    def __init__(
        self,
        config: ExperimentConfig,
        model_manager: ModelManager,
        data_manager: DatasetManager,
        checkpoint_manager: CheckpointManager,
        reward_composer: Any,
        paged_kv_cache: Optional[Any],
        metrics_logger: Optional[MetricsLogger] = None,
    ):
        (
            self.config,
            self.model_manager,
            self.data_manager,
            self.checkpoint_manager,
            self.reward_composer,
            self.paged_kv_cache,
            self.metrics_logger,
        ) = (
            config,
            model_manager,
            data_manager,
            checkpoint_manager,
            reward_composer,
            paged_kv_cache,
            metrics_logger,
        )

        (
            self.actor_model,
            self.ref_model,
            self.tokenizer,
            self.optimizer,
            self.lr_scheduler,
        ) = (None, None, None, None, None)

        self.global_step, self.current_epoch = 0, 0
        self._run_id = (
            self.metrics_logger.run_id
            if self.metrics_logger
            else f"run_{time.strftime('%Y%m%d-%H%M%S')}"
        )

        # Memory optimization flags
        self.use_mixed_precision = getattr(config.trainer, "use_mixed_precision", False)
        self.alternate_dual_gradients = getattr(
            config.trainer, "alternate_dual_gradients", False
        )
        self.log_memory_usage = getattr(
            config.trainer, "log_memory_usage", True
        )  # Default to True now

        # Memory monitoring
        self.memory_history = []
        self.memory_peak = 0.0

        # Enhanced progress bar system
        self.progress_manager = EnhancedProgressManager.get_instance()
        self._training_session = None
        self._setup_progress_bar()

        # Checkpoint retry config
        self.checkpoint_max_retries = getattr(config.checkpointing, "max_retries", 3)
        self.checkpoint_retry_delay = getattr(
            config.checkpointing, "retry_delay_seconds", 2
        )

        # Log optimization status
        if self.use_mixed_precision:
            logger.info("✓ Mixed precision training ENABLED")
        if self.alternate_dual_gradients:
            logger.info("✓ Alternating dual gradients ENABLED")
        if self.log_memory_usage:
            logger.info("✓ Memory usage logging ENABLED")

        logger.info("BaseTrainer initialized with enhanced progress tracking")

    @abstractmethod
    def _setup(self) -> Tuple[int, int]:
        raise NotImplementedError

    @abstractmethod
    def train_step(
        self, rollout_batch: Dict[str, mx.array], update_step: int
    ) -> Tuple[TrainingMetrics, Dict[str, mx.array], Any]:
        raise NotImplementedError

    @abstractmethod
    def generate_rollouts(
        self, batch_data: Dict[str, Any], update_step: int
    ) -> Tuple[Dict[str, mx.array], float, Dict[str, float], Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, update_step: int) -> List[EvaluationMetrics]:
        raise NotImplementedError

    def _setup_progress_bar(self) -> None:
        """Setup enhanced progress bar configuration."""
        try:
            # Configure progress bar based on trainer settings
            config = ProgressBarConfig(
                display_mode=DisplayMode.DETAILED,
                show_gradient_norm=True,
                show_memory_usage=self.log_memory_usage,
                show_performance_stats=True,
                show_layer_analysis=False,
                update_frequency=1.0,
                enable_caching=True,
                enable_anomaly_detection=True,
                max_history_size=1000
            )
            
            # Configure the progress manager
            self.progress_manager.configure(config)
            
            logger.debug("Enhanced progress bar configured successfully")
            
        except Exception as e:
            logger.warning(f"Failed to setup enhanced progress bar: {e}")
            # Fallback to basic progress tracking
            self.progress_manager = None

    def _terminal_alert(self, message: str, level: str = "INFO"):
        """Create terminal alert with color and bell."""
        try:
            # Terminal bell
            sys.stdout.write("\a")
            sys.stdout.flush()

            colors = {
                "INFO": "\033[94m",
                "WARNING": "\033[93m",
                "ERROR": "\033[91m",
                "SUCCESS": "\033[92m",
                "RESET": "\033[0m",
            }

            color = colors.get(level, colors["INFO"])
            reset = colors["RESET"]

            box_width = min(len(message) + 4, 80)
            print(f"\n{color}{'=' * box_width}{reset}")
            print(f"{color}  {message}{reset}")
            print(f"{color}{'=' * box_width}{reset}\n")
        except:
            pass

    def save_final_checkpoint(self, reason: str = "final"):
        """Save final checkpoint with retry logic."""
        if self.actor_model:
            self._save_checkpoint_with_retry(
                step=self.global_step, reason=reason, is_final=True
            )

    def _save_checkpoint_with_retry(
        self,
        step: int,
        reason: str = "regular",
        is_final: bool = False,
        max_retries: Optional[int] = None,
    ) -> bool:
        """
        Save checkpoint with retry logic and backoff.

        Args:
            step: Current training step
            reason: Reason for checkpoint
            is_final: Whether this is the final checkpoint
            max_retries: Override default max retries

        Returns:
            bool: Success status
        """
        max_retries = max_retries or self.checkpoint_max_retries

        for attempt in range(max_retries):
            try:
                # Get current memory stats for metadata
                mem_stats = _get_memory_usage_mb()

                metadata = {
                    "num_updates": step,
                    "epoch": self.current_epoch,
                    "reason": reason,
                    "log_id": self._run_id,
                    "save_optimizer_state": self.config.checkpointing.save_optimizer_state,
                    "memory_peak_mb": self.memory_peak,
                    "memory_current_mb": mem_stats.get("allocated_mb", 0),
                }

                self.checkpoint_manager.save_checkpoint(
                    step=step,
                    model=self.actor_model,
                    optimizer=self.optimizer,
                    metadata=metadata,
                    current_metric=self.checkpoint_manager.best_metric,
                )

                logger.info(
                    f"✓ Checkpoint saved (reason: {reason}, attempt: {attempt + 1})"
                )

                if is_final:
                    self._terminal_alert(
                        f"Final checkpoint saved successfully!", level="SUCCESS"
                    )

                return True

            except Exception as e:
                wait_time = self.checkpoint_retry_delay * (
                    2**attempt
                )  # Exponential backoff
                logger.error(
                    f"Checkpoint save failed (attempt {attempt + 1}/{max_retries}): {e}"
                )

                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {wait_time}s...")
                    self._terminal_alert(
                        f"Checkpoint save failed! Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})",
                        level="WARNING",
                    )

                    # Try to free up space
                    time.sleep(wait_time)
                    self._aggressive_memory_cleanup()
                else:
                    error_msg = (
                        f"CRITICAL: Checkpoint save failed after {max_retries} attempts!\n"
                        f"Error: {str(e)}\n"
                        f"Check: disk space, permissions, path validity"
                    )
                    self._terminal_alert(error_msg, level="ERROR")

                    # Log to file
                    checkpoint_dir = Path(self.config.checkpointing.checkpoint_dir)
                    error_log = checkpoint_dir / "checkpoint_errors.log"
                    try:
                        with open(error_log, "a") as f:
                            f.write(
                                f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] {error_msg}\n"
                            )
                    except:
                        pass

                    return False

        return False

    def _aggressive_memory_cleanup(self):
        """Aggressively free memory with enhanced cleanup."""
        try:
            mx.metal.clear_cache()
        except:
            pass

        mx.clear_cache()
        gc.collect()

        # Force Python garbage collection
        for _ in range(3):
            gc.collect()

    def _scale_gradients_inplace(self, grads: Dict, scale: float):
        """Scale gradients in-place."""
        return tree_map(lambda g: g * scale if isinstance(g, mx.array) else g, grads)

    def _log_memory_if_enabled(self, stage: str, step: int):
        """Enhanced memory logging with history tracking."""
        if not self.log_memory_usage:
            return

        mem_stats = _get_memory_usage_mb()
        if not mem_stats:
            return

        # Track peak
        current_allocated = mem_stats.get("allocated_mb", 0)
        if current_allocated > self.memory_peak:
            self.memory_peak = current_allocated

        # Update history
        self.memory_history.append(
            {"step": step, "stage": stage, "timestamp": time.time(), **mem_stats}
        )

        # Keep only recent history
        if len(self.memory_history) > 1000:
            self.memory_history = self.memory_history[-1000:]

        # Log to metrics logger
        if self.metrics_logger:
            log_dict = {
                f"memory/{stage}/cache_mb": mem_stats.get("cache_mb", 0),
                f"memory/{stage}/allocated_mb": mem_stats.get("allocated_mb", 0),
                f"memory/{stage}/peak_mb": mem_stats.get("peak_mb", 0),
                f"memory/peak_session_mb": self.memory_peak,
            }
            self.metrics_logger.log_metrics(log_dict, step=step)

        # Periodic console logging
        if step % 10 == 0:
            logger.debug(
                f"Memory [{stage}]: "
                f"Active={mem_stats.get('allocated_mb', 0):.1f}MB, "
                f"Peak={mem_stats.get('peak_mb', 0):.1f}MB, "
                f"Session Peak={self.memory_peak:.1f}MB"
            )

        # Alert on high memory usage
        if current_allocated > 8000:  # 8GB threshold
            logger.warning(f"⚠️ High memory usage: {current_allocated:.1f}MB")

    def _detect_memory_leak(self) -> Optional[str]:
        """Detect potential memory leaks from history."""
        if len(self.memory_history) < 50:
            return None

        recent = [h["allocated_mb"] for h in self.memory_history[-50:]]
        older = (
            [h["allocated_mb"] for h in self.memory_history[-100:-50]]
            if len(self.memory_history) >= 100
            else None
        )

        if older:
            recent_avg = np.mean(recent)
            older_avg = np.mean(older)
            increase_pct = ((recent_avg - older_avg) / older_avg) * 100

            if increase_pct > 30:  # 30% increase
                return f"Possible memory leak detected: {increase_pct:.1f}% increase over last 50 steps"

        return None

    def _check_memory_safety(self, step: int) -> Tuple[bool, str]:
        """
        Check if it's safe to continue based on memory usage.

        Returns:
            (is_safe, reason)
        """
        mem_stats = _get_memory_usage_mb()
        if not mem_stats:
            return True, "Could not check memory"

        active = mem_stats.get("active_mb", 0)
        # The trainer parameter is a ceiling for active memory, not total system memory
        threshold = getattr(self.config.trainer, "memory_safety_threshold_mb", 9000.0)

        # Check absolute threshold
        if active > threshold:
            return (
                False,
                f"Memory usage ({active:.1f}MB) exceeds threshold ({threshold}MB)",
            )

        # Check for memory leak
        leak_warning = self._detect_memory_leak()
        if leak_warning:
            logger.warning(f"⚠️ {leak_warning}")
            if (
                active > threshold * 0.8
            ):  # Trigger safety if leak detected and we are at 80% of threshold
                return False, leak_warning

        return True, "OK"

    async def run(self, should_shutdown: Callable[[], bool]):
        """Enhanced training loop with memory monitoring and error handling."""
        resumed_step, self.current_epoch = self._setup()

        if resumed_step > 0:
            self.global_step = resumed_step + 1
            logger.info(f"Resumed from checkpoint at step {resumed_step}")
        else:
            self.global_step = 0
            logger.info("Starting training from scratch")

        if self.tokenizer:
            self.data_manager.set_tokenizer(self.tokenizer)

        await self.data_manager.load_datasets()

        # Initialize enhanced progress bar
        if self.progress_manager:
            try:
                self._training_session = self.progress_manager.start_training_session(
                    total_steps=self.config.trainer.num_training_steps,
                    session_name=f"Training - {self._run_id}"
                )
                logger.info("Enhanced progress tracking started")
            except Exception as e:
                logger.warning(f"Failed to start enhanced progress session: {e}")
                self._training_session = None
        
        # Fallback to basic tqdm progress bar
        pbar = trange(
            self.global_step,
            self.config.trainer.num_training_steps,
            initial=self.global_step,
            desc="Training",
            unit="step",
        )

        train_data_iterator = iter([])
        grad_accum_steps = self.config.trainer.grad_accum_steps
        grad_scale = 1.0 / grad_accum_steps

        training_completed = False

        # Initial memory log
        self._log_memory_if_enabled("startup", self.global_step)

        with pbar:
            while self.global_step < self.config.trainer.num_training_steps:
                if should_shutdown():
                    logger.info("Shutdown requested")
                    break

                try:
                    # PRE-ITERATION SAFETY CHECK
                    is_safe, reason = self._check_memory_safety(self.global_step)
                    if not is_safe:
                        logger.warning(f"⚠️ Memory safety check failed: {reason}")
                        self._terminal_alert(
                            f"Safety checkpoint triggered: {reason}", level="WARNING"
                        )

                        # Save safety checkpoint
                        self._save_checkpoint_with_retry(
                            self.global_step, reason="memory_safety"
                        )

                        # Aggressive cleanup
                        self._aggressive_memory_cleanup()
                        time.sleep(2)

                    # Memory tracking: before accumulation
                    self._log_memory_if_enabled("before_accum", self.global_step)

                    # Streaming aggregation
                    accum_grads = None
                    sum_loss = 0.0
                    sum_reward = 0.0
                    sum_kl = 0.0
                    count_microbatches = 0
                    aggregated_raw_rewards = {}
                    training_performed = False

                    for accum_idx in range(grad_accum_steps):
                        try:
                            batch_data = next(train_data_iterator)
                        except StopIteration:
                            self.current_epoch += 1
                            logger.info(f"Epoch {self.current_epoch}")
                            train_data_iterator = iter(
                                self.data_manager.get_dataloader(
                                    "train", self.config.trainer.ppo_batch_size
                                )
                            )
                            try:
                                batch_data = next(train_data_iterator)
                            except StopIteration:
                                raise TrainingRuntimeError("Dataset is empty")

                        # Memory tracking: before rollout
                        if accum_idx == 0:
                            self._log_memory_if_enabled(
                                "before_rollout", self.global_step
                            )

                        # Generate rollouts
                        (
                            rollout_batch,
                            avg_reward_mb,
                            raw_reward_components_mb,
                            generation_metrics,
                        ) = self.generate_rollouts(batch_data, self.global_step)

                        # Memory tracking: after rollout
                        if accum_idx == 0:
                            self._log_memory_if_enabled(
                                "after_rollout", self.global_step
                            )

                        if (
                            not rollout_batch
                            or "tokens" not in rollout_batch
                            or not isinstance(rollout_batch["tokens"], mx.array)
                            or rollout_batch["tokens"].size == 0
                        ):
                            logger.warning(
                                f"Invalid rollout at step {self.global_step}"
                            )
                            del batch_data, rollout_batch
                            self._aggressive_memory_cleanup()
                            continue

                        # Train step
                        metrics_mb, grads_mb, step_metrics = self.train_step(
                            rollout_batch, self.global_step
                        )

                        # Memory tracking: after train step
                        if accum_idx == 0:
                            self._log_memory_if_enabled(
                                "after_train_step", self.global_step
                            )

                        # Aggregate metrics
                        sum_loss += metrics_mb.loss
                        sum_kl += metrics_mb.kl_divergence
                        sum_reward += avg_reward_mb
                        count_microbatches += 1

                        if raw_reward_components_mb:
                            for k, v in raw_reward_components_mb.items():
                                aggregated_raw_rewards[k] = (
                                    aggregated_raw_rewards.get(k, 0.0) + v
                                )

                        # Accumulate gradients
                        if grads_mb:
                            grads_mb_scaled = self._scale_gradients_inplace(
                                grads_mb, grad_scale
                            )

                            if accum_grads is None:
                                accum_grads = grads_mb_scaled
                            else:
                                accum_grads = tree_map(
                                    mx.add, accum_grads, grads_mb_scaled
                                )
                                mx.eval(tree_flatten(accum_grads))

                            del grads_mb, grads_mb_scaled

                        # Cleanup
                        del batch_data, rollout_batch, metrics_mb, step_metrics
                        self._aggressive_memory_cleanup()

                    # Memory tracking: after accumulation
                    self._log_memory_if_enabled("after_accum", self.global_step)

                    # Apply gradients
                    if accum_grads and self.optimizer and count_microbatches > 0:
                        # Compute grad norm
                        flat_grads = [
                            v
                            for _, v in tree_flatten(accum_grads)
                            if isinstance(v, mx.array)
                        ]
                        mx.eval(flat_grads)

                        grad_norm = np.linalg.norm(
                            [
                                np.linalg.norm(np.array(v.flatten().astype(mx.float32)))
                                for v in flat_grads
                            ]
                        )
                        del flat_grads

                        # Update
                        self.optimizer.learning_rate = self.lr_scheduler(
                            self.global_step
                        )
                        self.optimizer.apply_gradients(
                            accum_grads, self.actor_model.trainable_parameters()
                        )
                        mx.eval(self.actor_model.parameters(), self.optimizer.state)

                        training_performed = True

                        # Cleanup
                        del accum_grads
                        self._aggressive_memory_cleanup()

                        # Memory tracking: after optimizer
                        self._log_memory_if_enabled("after_optimizer", self.global_step)

                        # Compute averages
                        avg_loss = sum_loss / count_microbatches
                        avg_reward_mean = sum_reward / count_microbatches
                        avg_kl = sum_kl / count_microbatches

                        # Log metrics
                        if self.metrics_logger:
                            log_dict = {
                                "train/loss": avg_loss,
                                "train/reward_mean": avg_reward_mean,
                                "train/grad_norm": grad_norm,
                                "train/learning_rate": float(
                                    self.optimizer.learning_rate
                                ),
                                "train/kl_divergence": avg_kl,
                                "train/epoch": self.current_epoch,
                                "train/step": self.global_step,
                            }

                            if aggregated_raw_rewards:
                                for k, v in aggregated_raw_rewards.items():
                                    log_dict[f"train/rewards/raw_{k}"] = (
                                        v / count_microbatches
                                    )

                            self.metrics_logger.log_metrics(
                                log_dict, step=self.global_step
                            )

                        # Update progress
                        pbar.set_postfix(
                            {
                                "Loss": f"{avg_loss:.4f}",
                                "Reward": f"{avg_reward_mean:.3f}",
                                "Memory": f"{self.memory_peak:.0f}MB",
                            }
                        )
                        pbar.update(1)

                        # Checkpointing
                        is_eval = (
                            self.config.trainer.eval_every > 0
                            and (self.global_step + 1) % self.config.trainer.eval_every
                            == 0
                        )
                        is_save = (
                            self.config.checkpointing.save_every > 0
                            and (self.global_step + 1)
                            % self.config.checkpointing.save_every
                            == 0
                        )
                        is_final = (
                            self.global_step
                            == self.config.trainer.num_training_steps - 1
                        )

                        if is_final:
                            training_completed = True

                        if is_eval or is_final:
                            self._log_memory_if_enabled("before_eval", self.global_step)
                            self._aggressive_memory_cleanup()

                            eval_results = self.evaluate(self.global_step)

                            if self.metrics_logger:
                                for metric in eval_results:
                                    self.metrics_logger.log_metrics(
                                        metric.to_dict(), step=self.global_step
                                    )

                            del eval_results
                            self._aggressive_memory_cleanup()
                            self._log_memory_if_enabled("after_eval", self.global_step)

                        if (is_save or is_final) and training_performed:
                            self._log_memory_if_enabled(
                                "before_checkpoint", self.global_step
                            )
                            self._aggressive_memory_cleanup()

                            self._save_checkpoint_with_retry(
                                self.global_step,
                                reason="final" if is_final else "regular",
                            )

                            self._aggressive_memory_cleanup()
                            self._log_memory_if_enabled(
                                "after_checkpoint", self.global_step
                            )

                    self.global_step += 1

                    # Periodic cleanup
                    if self.global_step % 10 == 0:
                        self._aggressive_memory_cleanup()
                        self._log_memory_if_enabled(
                            "periodic_cleanup", self.global_step
                        )

                except RuntimeError as e:
                    error_msg = str(e)
                    if "METAL" in error_msg or "Command buffer" in error_msg:
                        logger.error(
                            f"Metal command buffer error at step {self.global_step}: {e}"
                        )
                        self._terminal_alert(
                            "Metal error! Saving checkpoint...", level="ERROR"
                        )

                        # Save emergency checkpoint
                        self._save_checkpoint_with_retry(
                            self.global_step, reason="metal_error"
                        )

                        # Aggressive recovery
                        self._aggressive_memory_cleanup()
                        time.sleep(5)

                        logger.info("Attempting to continue...")
                        continue
                    else:
                        raise

        # Final cleanup and checkpoint
        self._aggressive_memory_cleanup()

        if should_shutdown() or (
            not training_completed and self.global_step > resumed_step
        ):
            reason = "interrupted" if should_shutdown() else "completed"
            self.save_final_checkpoint(reason=reason)

        # Final memory log
        self._log_memory_if_enabled("final", self.global_step)

        # Memory summary
        if self.memory_history:
            logger.info(f"Memory Summary:")
            logger.info(f"  Peak Session: {self.memory_peak:.1f}MB")
            logger.info(f"  Final: {self.memory_history[-1]['allocated_mb']:.1f}MB")
