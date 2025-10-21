"""
Enhanced Checkpoint Management with Better Error Handling and Alerts

ENHANCEMENTS:
1. Better error messages with actionable suggestions
2. Terminal alerts for critical issues
3. Disk space checking before save
4. Corrupted checkpoint detection and recovery
5. Improved memory cleanup
6. Better logging of checkpoint operations
7. Atomic save with verification

BACKWARD COMPATIBLE: All existing functionality preserved
"""
import json
import logging
import os
import re
import shutil
import time
import gc
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Set

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
from rich import print as rprint

from .exceptions import CheckpointError

try:
    from mlx_lm.tuner.lora import LoRALinear as MLXLoRALinear
except ImportError:

    class MLXLoRALinear:
        pass


logger = logging.getLogger(__name__)


def _terminal_alert(message: str, level: str = "INFO"):
    """Create terminal alert with color and bell."""
    try:
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


def _check_disk_space(path: Path, required_mb: float = 1000.0) -> Tuple[bool, str]:
    """
    Check if sufficient disk space is available.

    Args:
        path: Path to check
        required_mb: Required space in MB

    Returns:
        (has_space, message)
    """
    try:
        import shutil as sh

        stat = sh.disk_usage(path)
        available_mb = stat.free / (1024 * 1024)

        if available_mb < required_mb:
            return (
                False,
                f"Low disk space: {available_mb:.1f}MB available, {required_mb:.1f}MB required",
            )

        return True, f"Sufficient disk space: {available_mb:.1f}MB available"
    except Exception as e:
        logger.warning(f"Could not check disk space: {e}")
        return True, "Could not verify disk space"


class CheckpointManager:
    """
    Enhanced checkpoint manager with better error handling and alerts.

    FEATURES:
    - Atomic checkpoint saving with verification
    - Disk space checking before save
    - Corrupted checkpoint detection
    - Terminal alerts for critical issues
    - Better error messages with suggestions
    - Improved memory management
    """

    def __init__(
        self,
        output_dir: Path,
        keep_last_n: int = 3,
        save_best: bool = True,
        base_model_path: Optional[Path] = None,
        min_disk_space_mb: float = 1000.0,  # NEW: Configurable minimum disk space
    ):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.save_best = save_best
        self.base_model_path = base_model_path
        self.min_disk_space_mb = min_disk_space_mb
        self._warned_about_missing_path = False

        self.best_metric: float = -float("inf")
        self._checkpoints: List[Path] = []
        self.resume_from_path: Optional[Path] = None

        # Statistics
        self.save_count = 0
        self.failed_save_count = 0
        self.last_save_time = 0.0

        self._load_existing_checkpoints()

        # Log initialization
        logger.info(f"CheckpointManager initialized:")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Keep last N: {self.keep_last_n}")
        logger.info(f"  Save best: {self.save_best}")
        logger.info(f"  Min disk space: {self.min_disk_space_mb}MB")

    def _aggressive_memory_cleanup(self):
        """Aggressively free memory."""
        try:
            mx.metal.clear_cache()
        except:
            pass
        mx.clear_cache()
        gc.collect()

    def _get_step_from_path(self, path: Path) -> Optional[int]:
        """Extract the training step number from a checkpoint path name."""
        match = re.search(r"update_(\d+)$", path.name)
        if match:
            return int(match.group(1))
        return None

    def _verify_checkpoint_integrity(self, checkpoint_path: Path) -> Tuple[bool, str]:
        """
        Verify checkpoint integrity.

        Returns:
            (is_valid, message)
        """
        try:
            # Check metadata exists
            metadata_file = checkpoint_path / "metadata.json"
            if not metadata_file.exists():
                return False, "Missing metadata.json"

            # Check metadata is valid JSON
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
            except json.JSONDecodeError:
                return False, "Corrupted metadata.json"

            # Check for model weights
            has_adapters = (checkpoint_path / "adapters.safetensors").exists()
            has_model = (checkpoint_path / "model.safetensors").exists()

            if not (has_adapters or has_model):
                return (
                    False,
                    "Missing model weights (adapters.safetensors or model.safetensors)",
                )

            return True, "Checkpoint is valid"

        except Exception as e:
            return False, f"Verification error: {e}"

    def _load_existing_checkpoints(self):
        """Load and sort existing checkpoints, removing corrupted ones."""
        found_dirs_with_steps: List[Tuple[int, Path]] = []
        corrupted_checkpoints = []

        for p in self.output_dir.iterdir():
            if p.is_dir() and (p / "metadata.json").is_file():
                # Verify checkpoint integrity
                is_valid, message = self._verify_checkpoint_integrity(p)

                if is_valid:
                    step = self._get_step_from_path(p)
                    if step is not None:
                        found_dirs_with_steps.append((step, p))
                else:
                    logger.warning(
                        f"Corrupted checkpoint detected: {p.name} - {message}"
                    )
                    corrupted_checkpoints.append(p)

        # Remove corrupted checkpoints
        if corrupted_checkpoints:
            _terminal_alert(
                f"Found {len(corrupted_checkpoints)} corrupted checkpoints. "
                "These will be removed to prevent loading issues.",
                level="WARNING",
            )

            for corrupted in corrupted_checkpoints:
                try:
                    shutil.rmtree(corrupted, ignore_errors=True)
                    logger.info(f"Removed corrupted checkpoint: {corrupted.name}")
                except Exception as e:
                    logger.error(
                        f"Failed to remove corrupted checkpoint {corrupted.name}: {e}"
                    )

        found_dirs_with_steps.sort(key=lambda x: x[0])
        self._checkpoints = [p for _, p in found_dirs_with_steps]
        del found_dirs_with_steps

        # Load best metric
        best_symlink = self.output_dir / "best"
        if best_symlink.is_symlink():
            try:
                resolved_path = best_symlink.resolve(strict=True)
                if resolved_path.is_dir():
                    metadata_file = resolved_path / "metadata.json"
                    if metadata_file.is_file():
                        with open(metadata_file, "r") as f:
                            metadata = json.load(f)
                            self.best_metric = metadata.get(
                                "current_metric", -float("inf")
                            )
                            del metadata
            except FileNotFoundError:
                logger.warning("Symlink 'best' is dangling. Removing it.")
                best_symlink.unlink()
            except Exception as e:
                logger.warning(f"Could not load best_metric from 'best' symlink: {e}")

        logger.info(f"Found {len(self._checkpoints)} valid checkpoints")

    def _copy_base_model_files(self, dest_path: Path):
        """Copy essential config and tokenizer files from base model."""
        if not self.base_model_path or not self.base_model_path.exists():
            return

        files_to_copy: Set[str] = {
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "added_tokens.json",
            "chat_template.jinja",
        }

        # Add glob patterns
        for file_pattern in ["*.model", "*.txt", "*.py"]:
            for f_path in self.base_model_path.glob(file_pattern):
                files_to_copy.add(f_path.name)

        # Copy files
        copied_count = 0
        for file_name in files_to_copy:
            source_file = self.base_model_path / file_name
            if source_file.is_file():
                try:
                    shutil.copy2(source_file, dest_path / file_name)
                    copied_count += 1
                except Exception as e:
                    logger.warning(f"Failed to copy {file_name}: {e}")

        del files_to_copy

        if copied_count > 0:
            logger.debug(f"Copied {copied_count} base model files")

    def save_checkpoint(
        self,
        step: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        metadata: Dict[str, Any],
        current_metric: Optional[float] = None,
    ):
        """
        Save a complete, portable checkpoint atomically with verification.

        ENHANCED:
        - Disk space checking
        - Atomic save with verification
        - Better error messages
        - Terminal alerts on issues
        """
        start_time = time.time()

        # Handle None metric
        if current_metric is None:
            current_metric = self.best_metric

        # Check disk space BEFORE attempting save
        has_space, space_message = _check_disk_space(
            self.output_dir, self.min_disk_space_mb
        )
        if not has_space:
            error_msg = f"Cannot save checkpoint: {space_message}"
            logger.error(error_msg)
            _terminal_alert(
                f"{error_msg}\nPlease free up disk space or adjust checkpoint settings.",
                level="ERROR",
            )
            self.failed_save_count += 1
            raise CheckpointError(error_msg)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_{timestamp}_update_{step}"
        temp_path = self.output_dir / f".{checkpoint_name}.tmp"
        final_path = self.output_dir / checkpoint_name

        # Clean up any existing temp directory
        if temp_path.exists():
            logger.warning(f"Removing stale temp directory: {temp_path.name}")
            shutil.rmtree(temp_path, ignore_errors=True)

        try:
            temp_path.mkdir(parents=True)

            # Copy base model files
            if self.base_model_path:
                self._copy_base_model_files(temp_path)
            elif not self._warned_about_missing_path:
                rprint(
                    "[yellow]Warning: `base_model_path` not provided. "
                    "Checkpoints will not be self-contained.[/yellow]"
                )
                self._warned_about_missing_path = True

            # Determine if this is a LoRA model
            is_lora = any(
                isinstance(m, MLXLoRALinear) for _, m in model.named_modules()
            )

            # Save model weights
            if is_lora:
                adapter_params = dict(tree_flatten(model.trainable_parameters()))
                if adapter_params:
                    mx.save_safetensors(
                        str(temp_path / "adapters.safetensors"), adapter_params
                    )
                    del adapter_params
                    self._aggressive_memory_cleanup()
                    logger.debug("Saved LoRA adapters")
            else:
                full_params = dict(tree_flatten(model.parameters()))
                mx.save_safetensors(str(temp_path / "model.safetensors"), full_params)
                del full_params
                self._aggressive_memory_cleanup()
                logger.debug("Saved full model weights")

            # Save optimizer state if requested
            if metadata.get("save_optimizer_state", False) and optimizer:
                try:
                    optimizer_state = dict(tree_flatten(optimizer.state))
                    mx.save_safetensors(
                        str(temp_path / "optimizer.safetensors"),
                        optimizer_state,
                    )
                    del optimizer_state
                    self._aggressive_memory_cleanup()
                    logger.debug("Saved optimizer state")
                except Exception as e:
                    logger.warning(f"Failed to save optimizer state: {e}")
                    # Continue anyway - optimizer state is optional

            # Save metadata
            metadata["step"] = step
            metadata["current_metric"] = current_metric
            metadata["timestamp"] = timestamp
            metadata["save_duration_s"] = 0.0  # Will update after save

            with open(temp_path / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2, default=str)

            # Verify checkpoint before finalizing
            is_valid, verify_message = self._verify_checkpoint_integrity(temp_path)
            if not is_valid:
                raise CheckpointError(
                    f"Checkpoint verification failed: {verify_message}"
                )

            # Atomic rename
            os.rename(temp_path, final_path)

            # Update duration in metadata
            save_duration = time.time() - start_time
            try:
                metadata_file = final_path / "metadata.json"
                with open(metadata_file, "r") as f:
                    md = json.load(f)
                md["save_duration_s"] = save_duration
                with open(metadata_file, "w") as f:
                    json.dump(md, f, indent=2, default=str)
            except:
                pass  # Non-critical

            self._checkpoints.append(final_path)
            self.save_count += 1
            self.last_save_time = time.time()

            rprint(
                f"✓ Checkpoint saved: [cyan]{final_path.name}[/cyan] "
                f"(Metric: {current_metric:.4f}, Duration: {save_duration:.1f}s)"
            )

            # Update symlinks
            self._update_symlink(final_path, "latest")

            # Update best checkpoint
            if self.is_best_metric(current_metric):
                self.best_metric = current_metric
                self._update_symlink(final_path, "best")
                rprint(f"🏆 New best checkpoint! Metric: {current_metric:.4f}")

            # Rotate old checkpoints
            self._rotate_checkpoints()

            # Final cleanup
            self._aggressive_memory_cleanup()

        except CheckpointError:
            # Re-raise checkpoint errors (already have good messages)
            if temp_path.exists():
                shutil.rmtree(temp_path, ignore_errors=True)
            raise

        except Exception as e:
            # Wrap other exceptions with helpful context
            if temp_path.exists():
                shutil.rmtree(temp_path, ignore_errors=True)

            self.failed_save_count += 1

            # Provide actionable error message
            error_msg = f"Checkpoint save failed at step {step}: {e}"
            suggestions = []

            if "No space left" in str(e) or "Disk quota" in str(e):
                suggestions.append("Free up disk space")
                suggestions.append("Reduce checkpoint frequency (increase save_every)")
                suggestions.append("Reduce keep_last_n")
            elif "Permission denied" in str(e):
                suggestions.append("Check write permissions on checkpoint directory")
                suggestions.append(f"Directory: {self.output_dir}")
            elif "Read-only" in str(e):
                suggestions.append("Checkpoint directory is read-only")
                suggestions.append("Choose a different output directory")

            if suggestions:
                error_msg += f"\n\nSuggestions:\n" + "\n".join(
                    f"  - {s}" for s in suggestions
                )

            logger.error(error_msg)
            raise CheckpointError(error_msg) from e

    def load_latest_state(
        self, model: nn.Module, optimizer: Optional[optim.Optimizer] = None
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Load the latest checkpoint to resume training.

        ENHANCED:
        - Better error messages
        - Checkpoint verification
        - Corrupted checkpoint recovery
        """
        chosen_path = self.resume_from_path

        if not chosen_path:
            latest_symlink = self.output_dir / "latest"
            if latest_symlink.is_symlink():
                try:
                    chosen_path = latest_symlink.resolve(strict=True)
                except FileNotFoundError:
                    logger.warning(
                        "Symlink 'latest' is dangling. Searching for last checkpoint."
                    )
                    latest_symlink.unlink()
                    if self._checkpoints:
                        chosen_path = self._checkpoints[-1]
            elif self._checkpoints:
                chosen_path = self._checkpoints[-1]

        if not chosen_path or not chosen_path.exists():
            rprint("[yellow]No checkpoint found. Starting from scratch.[/yellow]")
            return 0, {}

        # Verify checkpoint before loading
        is_valid, verify_message = self._verify_checkpoint_integrity(chosen_path)
        if not is_valid:
            error_msg = f"Cannot load checkpoint {chosen_path.name}: {verify_message}"
            logger.error(error_msg)
            _terminal_alert(
                f"{error_msg}\nThis checkpoint is corrupted. "
                "Training will start from scratch or use an earlier checkpoint.",
                level="ERROR",
            )

            # Try to find an earlier valid checkpoint
            for checkpoint in reversed(self._checkpoints[:-1]):
                is_valid, _ = self._verify_checkpoint_integrity(checkpoint)
                if is_valid:
                    chosen_path = checkpoint
                    logger.info(f"Using earlier checkpoint: {checkpoint.name}")
                    break
            else:
                # No valid checkpoint found
                rprint(
                    "[yellow]No valid checkpoints found. Starting from scratch.[/yellow]"
                )
                return 0, {}

        rprint(f"Resuming from: [green]{chosen_path.name}[/green]")

        try:
            # Load metadata
            metadata_file = chosen_path / "metadata.json"
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            # Determine checkpoint type
            is_lora = any(
                isinstance(m, MLXLoRALinear) for _, m in model.named_modules()
            )
            adapters_file = chosen_path / "adapters.safetensors"
            model_file = chosen_path / "model.safetensors"

            # Load model weights
            if is_lora and adapters_file.is_file():
                from mlx_lm.tuner.utils import load_adapters

                load_adapters(model, str(chosen_path))
                rprint("✓ Loaded LoRA adapters")
            elif model_file.is_file():
                weights = list(mx.load(str(model_file)).items())
                model.load_weights(weights)
                del weights
                self._aggressive_memory_cleanup()
                rprint("✓ Loaded full model weights")
            else:
                raise CheckpointError(
                    f"No model weights found. Expected "
                    f"{'adapters.safetensors' if is_lora else 'model.safetensors'}"
                )

            # Load optimizer state
            optimizer_loaded = False
            if optimizer is not None:
                optimizer_file = chosen_path / "optimizer.safetensors"
                if metadata.get("save_optimizer_state") and optimizer_file.is_file():
                    try:
                        optimizer_state_items = list(
                            mx.load(str(optimizer_file)).items()
                        )
                        optimizer.state = tree_unflatten(optimizer_state_items)
                        del optimizer_state_items
                        optimizer_loaded = True
                        self._aggressive_memory_cleanup()
                        rprint("✓ Loaded optimizer state")
                    except Exception as e:
                        logger.warning(f"Failed to load optimizer state: {e}")

            # Evaluate parameters
            params_to_eval = list(model.parameters())
            if optimizer_loaded and optimizer is not None:
                params_to_eval.extend(list(optimizer.state.values()))
            mx.eval(params_to_eval)
            del params_to_eval

            # Update best metric
            self.best_metric = metadata.get("current_metric", -float("inf"))
            resumed_step = metadata.get("step", metadata.get("num_updates", 0))

            # Log statistics
            logger.info(f"Checkpoint statistics:")
            logger.info(f"  Step: {resumed_step}")
            logger.info(f"  Metric: {self.best_metric:.4f}")
            logger.info(f"  Timestamp: {metadata.get('timestamp', 'unknown')}")

            if "save_duration_s" in metadata:
                logger.info(f"  Save duration: {metadata['save_duration_s']:.1f}s")

            self._aggressive_memory_cleanup()

            return resumed_step, metadata

        except CheckpointError:
            raise
        except Exception as e:
            error_msg = f"Failed to load checkpoint {chosen_path.name}: {e}"
            logger.error(error_msg)
            _terminal_alert(error_msg, level="ERROR")
            raise CheckpointError(error_msg) from e

    def _update_symlink(self, target_path: Path, link_name: str):
        """Atomically update a symlink."""
        link_path = self.output_dir / link_name

        if link_path.is_symlink() or link_path.exists():
            link_path.unlink()

        try:
            os.symlink(
                os.path.relpath(target_path, self.output_dir),
                link_path,
                target_is_directory=True,
            )
        except Exception as e:
            logger.warning(f"Failed to create symlink '{link_name}': {e}")

    def _rotate_checkpoints(self):
        """Delete old checkpoints, keeping the last N and the best one."""
        if len(self._checkpoints) <= self.keep_last_n:
            return

        # Find best checkpoint
        best_path = None
        best_symlink = self.output_dir / "best"
        if best_symlink.is_symlink():
            try:
                best_path = best_symlink.resolve(strict=True)
            except FileNotFoundError:
                best_symlink.unlink()

        # Keep last N and best
        checkpoints_to_keep: Set[Path] = set(self._checkpoints[-self.keep_last_n :])
        if best_path:
            checkpoints_to_keep.add(best_path)

        # Delete old checkpoints
        checkpoints_to_delete = [
            chk for chk in self._checkpoints if chk not in checkpoints_to_keep
        ]

        for chk in checkpoints_to_delete:
            if chk.exists():
                try:
                    rprint(f"Removing old checkpoint: [red]{chk.name}[/red]")
                    shutil.rmtree(chk, ignore_errors=True)
                except Exception as e:
                    logger.warning(f"Failed to delete checkpoint {chk.name}: {e}")

        # Update internal list
        self._checkpoints = sorted(
            list(checkpoints_to_keep),
            key=lambda p: self._get_step_from_path(p) or -1,
        )

        del checkpoints_to_delete, checkpoints_to_keep

    def is_best_metric(self, current_metric: Optional[float]) -> bool:
        """Check if current metric is the best."""
        if current_metric is None:
            return False
        return self.save_best and current_metric > self.best_metric

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get checkpoint manager statistics.

        NEW: Useful for monitoring and debugging
        """
        return {
            "total_checkpoints": len(self._checkpoints),
            "save_count": self.save_count,
            "failed_save_count": self.failed_save_count,
            "best_metric": self.best_metric,
            "last_save_time": self.last_save_time,
            "keep_last_n": self.keep_last_n,
            "save_best": self.save_best,
        }
