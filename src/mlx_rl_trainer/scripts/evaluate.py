# file_path: mlx_rl_trainer/scripts/evaluate.py
# revision_no: 001
# goals_of_writing_code_block: Script for standalone model evaluation.
# type_of_code_response: add new code
"""Script for standalone model evaluation."""

import sys
import logging
from pathlib import Path
import json
import argparse
import uuid
import mlx.core as mx

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mlx_rl_trainer.core.config import ExperimentConfig
from mlx_rl_trainer.core.model_manager import ModelManager
from mlx_rl_trainer.data.dataset_manager import DatasetManager
from mlx_rl_trainer.evaluation.registry import EvaluatorRegistry
from mlx_rl_trainer.core.trainer import EvaluationMetrics
from mlx_rl_trainer.core.exceptions import ModelLoadError, DataLoadError

# Import rich for nice console output
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install as rich_install

rich_install(show_locals=False)
console = Console(stderr=True, force_terminal=True)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for standalone evaluation."""
    parser = argparse.ArgumentParser(description="MLX RL Trainer - Evaluation Script")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file (used for model and data paths, eval configs)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint directory (e.g., 'outputs/run_001/checkpoint_step_1000')",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        nargs="+",
        help="Specific benchmark(s) to run (e.g., 'human_eval', 'gsm8k'). Overrides config.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./eval_outputs",
        help="Directory to save evaluation results.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging verbosity level.",
    )

    args = parser.parse_args()

    # 1. Setup Logging
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    handlers = [
        RichHandler(markup=True, rich_tracebacks=True, level=log_level, console=console)
    ]
    logging.basicConfig(level=log_level, handlers=handlers, force=True)

    eval_run_id = str(uuid.uuid4())
    output_dir = Path(args.output_dir) / eval_run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(
        output_dir / f"evaluation_log_{eval_run_id}.log", mode="a", encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    logging.getLogger().addHandler(file_handler)

    logger.info(
        f"Starting evaluation run with ID: [bold magenta]{eval_run_id}[/bold magenta]. Outputting to: {output_dir}"
    )

    # 2. Load Configuration
    config_path = Path(args.config)
    try:
        exp_config = ExperimentConfig.load_from_yaml(config_path)
        logger.info(f"Loaded experiment configuration from {config_path}")
    except Exception as e:
        logger.critical(f"FATAL CONFIGURATION ERROR: {e}")
        sys.exit(1)

    # Override checkpoint path in model config
    exp_config.model.model_path = Path(args.checkpoint)

    # 3. Load Model and Tokenizer
    model_manager = ModelManager(exp_config.model)
    try:
        model, tokenizer = model_manager.load_model(
            Path(args.checkpoint), "eval_model", is_trainable=False
        )
        logger.info(f"Loaded model from checkpoint: {args.checkpoint}")
    except ModelLoadError as e:
        logger.critical(f"Failed to load model for evaluation: {e}")
        sys.exit(1)

    # 4. Load Dataset
    data_manager = DatasetManager(exp_config.data, tokenizer)
    try:
        data_manager.load_datasets()
        val_dataset = data_manager._val_dataset  # Use validation dataset for evaluation
        if val_dataset is None:
            raise DataLoadError("No validation dataset available for evaluation.")
        logger.info(f"Loaded validation dataset with {len(val_dataset)} samples.")
    except DataLoadError as e:
        logger.critical(f"Failed to load dataset for evaluation: {e}")
        sys.exit(1)

    # 5. Determine Evaluators to Run
    evaluator_configs_to_run: List[Dict[str, Any]] = []
    if args.benchmark:  # If specific benchmarks are requested via CLI
        for bench_name in args.benchmark:
            found = False
            for cfg in exp_config.evaluation:
                if cfg.name == bench_name:
                    evaluator_configs_to_run.append(
                        cfg.model_dump()
                    )  # Use model_dump to convert Pydantic to dict
                    found = True
                    break
            if not found:
                logger.warning(
                    f"Requested benchmark '{bench_name}' not found in config. Skipping."
                )
        if not evaluator_configs_to_run:
            logger.critical("No valid benchmarks to run based on CLI input.")
            sys.exit(1)
    else:  # Use all evaluators from config
        evaluator_configs_to_run = [cfg.model_dump() for cfg in exp_config.evaluation]
        if not evaluator_configs_to_run:
            logger.critical("No evaluators configured in the YAML file.")
            sys.exit(1)

    # 6. Run Evaluation
    all_results: List[EvaluationMetrics] = []
    for eval_cfg_dict in evaluator_configs_to_run:
        try:
            # Merge global generation/system prompt config into evaluator's config
            merged_eval_config = eval_cfg_dict.copy()
            merged_eval_config.update(
                {
                    "system_prompt": exp_config.system_prompt,
                    "max_kv_size": exp_config.max_kv_size,
                    "max_gen_len": exp_config.data.max_gen_len,  # Max generated tokens for eval
                    "temperature": exp_config.generation.temperature,
                    "top_p": exp_config.generation.top_p,
                    "top_k": exp_config.generation.top_k,
                }
            )

            evaluator = EvaluatorRegistry.create(
                merged_eval_config["name"], merged_eval_config
            )
            results = evaluator.evaluate(model, tokenizer, val_dataset)
            all_results.append(results)
            logger.info(f"Evaluation for '{evaluator.name}' completed.")

            # Save individual benchmark results
            with open(
                output_dir / f"{evaluator.name}_results.json", "w", encoding="utf-8"
            ) as f:
                json.dump(
                    results.to_dict(), f, indent=4
                )  # Assuming EvaluationMetrics has to_dict()
                logger.info(
                    f"Saved results for {evaluator.name} to {output_dir / f'{evaluator.name}_results.json'}"
                )

        except Exception as e:
            logger.error(
                f"Error running evaluator '{eval_cfg_dict['name']}': {e}", exc_info=True
            )

    logger.info("All evaluations completed.")

    # Optional: Aggregate all results into a single file
    if all_results:
        summary_path = output_dir / "evaluation_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in all_results], f, indent=4)
        logger.info(f"Evaluation summary saved to {summary_path}")


if __name__ == "__main__":
    # Set default device - try GPU first, fallback to CPU
    try:
        mx.set_default_device(mx.gpu)
    except:
        mx.set_default_device(mx.cpu)
    main()
