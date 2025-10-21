#!/usr/bin/env python
# Path: src/mlx_rl_trainer/scripts/train.py
import sys, logging, asyncio, uuid, random, signal, time, json
from pathlib import Path
import argparse
import mlx.core as mx
from rich.console import Console
from rich.logging import RichHandler
from rich import print as rprint, traceback as rich_traceback
import numpy as np
import psutil
from mlx.utils import tree_flatten, tree_unflatten


# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from mlx_rl_trainer.core.config import ExperimentConfig
from mlx_rl_trainer.core.model_manager import ModelManager
from mlx_rl_trainer.data.dataset_manager import DatasetManager
from mlx_rl_trainer.core.checkpoint_manager import CheckpointManager
from mlx_rl_trainer.core.exceptions import CustomBaseException, TrainingRuntimeError
from mlx_rl_trainer.algorithms.grpo.grpo_trainer import GRPOTrainer
from mlx_rl_trainer.rewards.registry import RewardRegistry
from mlx_rl_trainer.rewards.base_reward import RewardComposer
from mlx_rl_trainer.rewards.context import RewardContext
from mlx_rl_trainer.monitoring.metrics_logger import MetricsLogger, _emit_plots_from_csv
from mlx_rl_trainer.utils import limit_memory

# Import rewards and evaluators to register them
import mlx_rl_trainer.rewards
import mlx_rl_trainer.evaluation

rich_traceback.install(show_locals=False)
console = Console(stderr=True, force_terminal=True)
logger = logging.getLogger(__name__)

shutdown_requested = False
wandb_run = None


def handle_signal(signum, frame):
    global shutdown_requested
    if not shutdown_requested:
        rprint(
            "\n[bold yellow]Shutdown requested. Finishing current step and saving checkpoint...[/bold yellow]"
        )
        shutdown_requested = True


def path_to_str(d):
    if isinstance(d, dict):
        return {k: path_to_str(v) for k, v in d.items()}
    if isinstance(d, list):
        return [path_to_str(v) for v in d]
    if isinstance(d, Path):
        return str(d)
    return d


async def _async_main():
    global shutdown_requested, wandb_run
    parser = argparse.ArgumentParser(description="MLX RL Trainer")
    
    # Configuration file (now optional)
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help="Path to configuration YAML file (optional if model-path and data-path are provided)"
    )
    
    # Core training parameters
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to the model directory"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to training data (JSONL file or directory)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Output directory for checkpoints and logs"
    )
    
    # Training parameters
    parser.add_argument(
        "--num-training-steps",
        type=int,
        default=1000,
        help="Number of training steps"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-6,
        help="Learning rate"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for training"
    )
    parser.add_argument(
        "--num-rollout-samples",
        type=int,
        default=2,
        help="Number of rollout samples"
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps"
    )
    
    # Model parameters
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Enable LoRA fine-tuning"
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=8,
        help="LoRA adapter rank"
    )
    
    # Generation parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Temperature for text generation"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.6,
        help="Top-p sampling parameter"
    )
    
    # Monitoring
    parser.add_argument(
        "--wandb-project",
        type=str,
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging"
    )
    
    # Other parameters
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (overrides auto-resume)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    
    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    handlers = [
        RichHandler(markup=True, rich_tracebacks=True, console=console, level=log_level)
    ]
    logging.basicConfig(level=log_level, handlers=handlers, force=True)

    # Load configuration from file or create from CLI arguments
    try:
        if args.config:
            # Load from configuration file
            config = ExperimentConfig.load_from_yaml(Path(args.config))
            logger.info(f"Loaded configuration from: {args.config}")
            
            # Override config with CLI arguments if provided
            if args.model_path:
                config.model.model_path = Path(args.model_path)
            if args.data_path:
                config.data.train_path = Path(args.data_path)
            if args.output_dir != "./outputs":
                config.trainer.output_dir = Path(args.output_dir)
            if args.num_training_steps != 1000:
                config.trainer.num_training_steps = args.num_training_steps
            if args.learning_rate != 2e-6:
                config.trainer.learning_rate = args.learning_rate
            if args.batch_size != 1:
                config.trainer.ppo_batch_size = args.batch_size
            if args.num_rollout_samples != 2:
                config.trainer.num_rollout_samples = args.num_rollout_samples
            if args.grad_accum_steps != 1:
                config.trainer.grad_accum_steps = args.grad_accum_steps
            if args.use_lora:
                config.model.use_lora = True
            if args.lora_rank != 8:
                config.model.lora_rank = args.lora_rank
            if args.temperature != 0.3:
                config.generation.answer_temperature = args.temperature
            if args.top_p != 0.6:
                config.generation.sampling_top_p = args.top_p
            if args.wandb_project:
                config.monitoring.wandb_project = args.wandb_project
            if args.no_wandb:
                config.monitoring.use_wandb = False
                
        else:
            # Create configuration from CLI arguments
            if not args.model_path or not args.data_path:
                logger.critical("ERROR: Either --config must be provided, or both --model-path and --data-path must be specified")
                parser.print_help()
                sys.exit(1)
            
            logger.info("Creating configuration from CLI arguments")
            
            # Expand paths to handle ~ and relative paths
            import os
            model_path = os.path.expanduser(args.model_path)
            data_path = os.path.expanduser(args.data_path)
            output_dir = os.path.expanduser(args.output_dir)
            
            # Create minimal configuration from CLI arguments
            config_dict = {
                "trainer": {
                    "output_dir": output_dir,
                    "num_training_steps": args.num_training_steps,
                    "learning_rate": args.learning_rate,
                    "ppo_batch_size": args.batch_size,
                    "num_rollout_samples": args.num_rollout_samples,
                    "grad_accum_steps": args.grad_accum_steps,
                },
                "model": {
                    "model_path": model_path,
                    "use_lora": args.use_lora,
                    "lora_rank": args.lora_rank,
                },
                "data": {
                    "train_path": data_path,
                },
                "generation": {
                    "answer_temperature": args.temperature,
                    "sampling_top_p": args.top_p,
                },
                "monitoring": {
                    "use_wandb": not args.no_wandb,
                    "wandb_project": args.wandb_project or "mlx-rl-trainer",
                },
                "rewards": [
                    {"name": "semantic_similarity", "weight": 0.7},
                    {"name": "format_structure", "weight": 0.3}
                ],
                "evaluation": []
            }
            
            config = ExperimentConfig(**config_dict)
            logger.info("Configuration created from CLI arguments")
            
        logging.debug("***** Final Config ******")
        logging.debug(config)
        
    except Exception as e:
        logger.critical(f"FATAL CONFIGURATION ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    # config.trainer.output_dir = config.trainer.output_dir / run_id
    config.trainer.output_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(
        config.trainer.output_dir / f"training_debug_{run_id}.log",
        mode="a",
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logging.getLogger().addHandler(file_handler)

    logger.info(
        f"Starting run with ID: [bold magenta]{run_id}[/bold magenta], output to {config.trainer.output_dir}"
    )

    random.seed(config.trainer.seed)
    np.random.seed(config.trainer.seed)
    mx.random.seed(config.trainer.seed)
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    if config.monitoring.use_wandb and WANDB_AVAILABLE:
        try:
            wandb_cfg = path_to_str(config.model_dump())
            wandb_cfg["run_id"] = run_id
            wandb_run = wandb.init(
                project=config.monitoring.wandb_project,
                # entity=config.monitoring.wandb_entity,
                name=config.monitoring.wandb_run_name or run_id,
                config=wandb_cfg,
                resume="allow",
                id=run_id,
            )
            logger.info(f"W&B logging initialized: {wandb_run.url}")
            from mlx_rl_trainer.monitoring import metrics_logger

            metrics_logger.wandb_run = wandb_run
        except Exception as e:
            logger.error(
                f"W&B initialization failed: {e}. Disabling W&B.", exc_info=True
            )
            config.monitoring.use_wandb = False
            wandb_run = None

    rewards = [
        (RewardRegistry.create(rc.name, rc.config), rc.weight) for rc in config.rewards
    ]
    reward_composer = RewardComposer(rewards, context_cls=RewardContext)

    model_manager = ModelManager(config.model)
    data_manager = DatasetManager(config.data, tokenizer=None)
    checkpoint_manager = CheckpointManager(
        config.trainer.output_dir / config.checkpointing.save_dir,
        keep_last_n=config.checkpointing.keep_last_n,  # CORRECTED from keep_best_n
        save_best=True,
        base_model_path=config.model.model_path,
    )
    if args.resume:
        checkpoint_manager.resume_from_path = Path(args.resume)

    metrics_logger = MetricsLogger(config, run_id)

    paged_kv_cache = None

    if config.trainer.algorithm == "grpo":
        trainer = GRPOTrainer(
            config,
            model_manager,
            data_manager,
            checkpoint_manager,
            reward_composer,
            paged_kv_cache,
            metrics_logger,
        )
    else:
        raise ValueError(f"Unknown algorithm: {config.trainer.algorithm}")

    try:
        await trainer.run(lambda: shutdown_requested)
        logger.info("[bold green]Training completed successfully![/bold green]")
    except CustomBaseException as e:
        logger.critical(f"A predictable error halted training: {e}", exc_info=True)
        if trainer and trainer.global_step > 0:
            trainer.save_final_checkpoint(reason="error_halt")
        sys.exit(1)
    except Exception as e:
        logger.critical(
            f"An unexpected error occurred during training: {e}", exc_info=True
        )
        if trainer and trainer.global_step > 0:
            trainer.save_final_checkpoint(reason="unexpected_crash")
        sys.exit(1)
    finally:
        logger.info("Application shutdown sequence initiated.")
        if metrics_logger:
            metrics_logger.close()
            _emit_plots_from_csv(
                metrics_logger.file_path, config.trainer.output_dir, config, run_id
            )
        if wandb_run:
            wandb_run.finish()
        logger.info("[bold blue]All resources released. Shutdown complete.[/bold blue]")


def main():
    rprint(f"MLX using device: [bold cyan]{mx.default_device()}[/bold cyan]")

    # Log memory usage (optional)
    current_peak_mem = mx.get_peak_memory()
    initial_peak_mem = mx.get_peak_memory()
    if current_peak_mem > initial_peak_mem:
        logging.debug(
            f"Peak Mem: {current_peak_mem / 1e9:.3f} GB (Delta: {(current_peak_mem - initial_peak_mem) / 1e9:.3f} GB)"
        )
    else:
        process = psutil.Process()
        mem_info = process.memory_info()
        logging.debug(f"RSS Mem: {mem_info.rss / (1024 * 1024):.2f} MB")

    limit_memory(80)

    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
