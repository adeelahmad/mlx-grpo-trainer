# file_path: mlx_rl_trainer/scripts/data_preprocessing.py
# revision_no: 001
# goals_of_writing_code_block: Script for data preprocessing and dataset creation.
# type_of_code_response: add new code
"""Script for data preprocessing and dataset creation."""

import sys
import logging
from pathlib import Path
import argparse
import json
import asyncio
import aiofiles
from typing import List, Dict, Any, Optional
from datasets import Dataset, Features, Value, load_dataset as hf_load_dataset
from rich.logging import RichHandler

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mlx_rl_trainer.core.config import DataConfig, ExperimentConfig
from mlx_rl_trainer.core.exceptions import DataLoadError
from mlx_rl_trainer.core.model_manager import (
    MockTokenizer,
)  # For tokenizing during preprocessing
from mlx_rl_trainer.data.dataset_manager import (
    _normalize_record,
    _contains_keywords,
    _looks_garbage,
)  # Reuse helpers
from mlx_rl_trainer.utils.text_utils import (
    apply_chat_template_wrapper,
)  # For consistent chat formatting

logger = logging.getLogger(__name__)


async def _async_load_and_process_jsonl(
    path: Path,
    prompt_key: str,
    answer_key: str,
    filter_keywords: List[str],
    system_prompt: str,
    tokenizer: MockTokenizer,
) -> List[Dict[str, Any]]:
    """Asynchronously loads, normalizes, filters, and tokenizes a JSONL file."""
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}")

    processed_data: List[Dict[str, Any]] = []
    async with aiofiles.open(path, mode="r", encoding="utf-8") as f:
        async for line in f:
            if line.strip():
                try:
                    obj = json.loads(line)
                    normalized_rec = _normalize_record(obj, prompt_key, answer_key, system_prompt)

                    if normalized_rec:
                        p = normalized_rec["prompt"]
                        c = normalized_rec["completion"]
                        s = (
                            normalized_rec["system"]
                            if normalized_rec["system"]
                            else system_prompt
                        )
                        tc = normalized_rec[
                            "test_cases"
                        ]  # test_cases are already list of dicts
                        meta = normalized_rec["meta"]

                        # Apply filters
                        if filter_keywords and (
                            _contains_keywords(p, filter_keywords)
                            or _contains_keywords(c, filter_keywords)
                        ):
                            continue
                        if _looks_garbage(p) or len(p.strip()) < 5:
                            continue

                        # Apply chat template and tokenize (for length checks, not actual token IDs for storage)
                        formatted_prompt = apply_chat_template_wrapper(tokenizer, p, s)
                        encoded_prompt_len = len(
                            tokenizer.encode(formatted_prompt, add_special_tokens=False)
                        )

                        processed_data.append(
                            {
                                "prompt": p,
                                "completion": c,
                                "system": s,
                                "test_cases": tc,  # Ensure test_cases remain a list of dicts
                                "is_invalid_sample": normalized_rec[
                                    "is_invalid_sample"
                                ],
                                "meta": meta,
                                "formatted_prompt": formatted_prompt,  # Store for debug
                                "encoded_prompt_len": encoded_prompt_len,
                            }
                        )
                except json.JSONDecodeError:
                    logger.warning(
                        f"Skipping malformed JSONL line in {path}: {line[:80]}..."
                    )
                except Exception as e:
                    logger.warning(
                        f"Error processing line from {path}: {e}. Skipping.",
                        exc_info=True,
                    )
    return processed_data


async def _async_load_and_process_hf(
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    prompt_key: str,
    answer_key: str,
    filter_keywords: List[str],
    system_prompt: str,
    tokenizer: MockTokenizer,
) -> List[Dict[str, Any]]:
    """Asynchronously loads, normalizes, filters, and tokenizes a HuggingFace dataset."""
    try:
        ds = await asyncio.to_thread(
            hf_load_dataset, dataset_name, dataset_config, split=split
        )

        processed_data: List[Dict[str, Any]] = []
        for obj in ds:
            normalized_rec = _normalize_record(obj, prompt_key, answer_key, system_prompt)
            if normalized_rec:
                p = normalized_rec["prompt"]
                c = normalized_rec["completion"]
                s = (
                    normalized_rec["system"]
                    if normalized_rec["system"]
                    else system_prompt
                )
                tc = normalized_rec["test_cases"]
                meta = normalized_rec["meta"]

                # Apply filters
                if filter_keywords and (
                    _contains_keywords(p, filter_keywords)
                    or _contains_keywords(c, filter_keywords)
                ):
                    continue
                if _looks_garbage(p) or len(p.strip()) < 5:
                    continue

                # Apply chat template and tokenize (for length checks)
                formatted_prompt = apply_chat_template_wrapper(tokenizer, p, s)
                encoded_prompt_len = len(
                    tokenizer.encode(formatted_prompt, add_special_tokens=False)
                )

                processed_data.append(
                    {
                        "prompt": p,
                        "completion": c,
                        "system": s,
                        "test_cases": tc,
                        "is_invalid_sample": normalized_rec["is_invalid_sample"],
                        "meta": meta,
                        "formatted_prompt": formatted_prompt,
                        "encoded_prompt_len": encoded_prompt_len,
                    }
                )
        return processed_data
    except Exception as e:
        raise DataLoadError(
            f"Async HF dataset load and process failed for {dataset_name}: {e}"
        ) from e


async def preprocess_and_save_dataset(
    config: ExperimentConfig,
    output_train_path: Path,
    output_val_path: Optional[Path] = None,
):
    """
    Loads, preprocesses, and saves the training and validation datasets.
    """
    logger.info("Starting dataset preprocessing...")

    tokenizer = (
        MockTokenizer()
    )  # Use mock tokenizer for preprocessing token length estimates
    # Configure special tokens if any (already handled in main training for actual tokenizer)

    tasks = []
    # Train data
    train_path = config.data.train_path
    if train_path.suffix.lower() in (".jsonl", ".ndjson"):
        tasks.append(
            _async_load_and_process_jsonl(
                train_path,
                config.data.dataset_prompt_key,
                config.data.dataset_answer_key,
                config.data.dataset_filter_keywords,
                config.system_prompt,
                tokenizer,
            )
        )
    else:  # Assume HF dataset
        tasks.append(
            _async_load_and_process_hf(
                train_path.as_posix(),
                config.data.get("dataset_config"),
                config.data.get("dataset_train_split", "train"),
                config.data.dataset_prompt_key,
                config.data.dataset_answer_key,
                config.data.dataset_filter_keywords,
                config.system_prompt,
                tokenizer,
            )
        )

    # Val data
    if config.data.val_path:
        val_path = config.data.val_path
        if val_path.suffix.lower() in (".jsonl", ".ndjson"):
            tasks.append(
                _async_load_and_process_jsonl(
                    val_path,
                    config.data.dataset_prompt_key,
                    config.data.dataset_answer_key,
                    config.data.dataset_filter_keywords,
                    config.system_prompt,
                    tokenizer,
                )
            )
        else:  # Assume HF dataset
            tasks.append(
                _async_load_and_process_hf(
                    val_path.as_posix(),
                    config.data.get("dataset_config"),
                    config.data.get("dataset_val_split", "test"),
                    config.data.dataset_prompt_key,
                    config.data.dataset_answer_key,
                    config.data.dataset_filter_keywords,
                    config.system_prompt,
                    tokenizer,
                )
            )

    results = await asyncio.gather(*tasks)
    train_processed_data = results[0]
    val_processed_data = results[1] if len(results) > 1 else []

    # Create HuggingFace Datasets
    common_features = Features(
        {
            "prompt": Value("string"),
            "completion": Value("string"),
            "system": Value("string"),
            "test_cases": [Value("string")],  # Store as stringified list
            "is_invalid_sample": Value("bool"),
            "meta": {
                "is_mcq": Value("bool"),
                "options": [Value("string")],
                "correct_letters": Value("string"),
                "multi_select": Value("bool"),
                "correct_indices": [Value("int32")],
            },  # Nested meta
            "formatted_prompt": Value("string"),
            "encoded_prompt_len": Value("int32"),
        }
    )

    train_dataset = Dataset.from_list(train_processed_data, features=common_features)
    val_dataset = (
        Dataset.from_list(val_processed_data, features=common_features)
        if val_processed_data
        else None
    )

    # Save processed datasets
    output_train_path.parent.mkdir(parents=True, exist_ok=True)
    train_dataset.to_json(output_train_path, indent=2)
    logger.info(
        f"Processed training dataset saved to: {output_train_path} ({len(train_dataset)} samples)."
    )

    if val_dataset:
        output_val_path.parent.mkdir(parents=True, exist_ok=True)
        val_dataset.to_json(output_val_path, indent=2)
        logger.info(
            f"Processed validation dataset saved to: {output_val_path} ({len(val_dataset)} samples)."
        )

    logger.info("Dataset preprocessing completed.")


def main():
    parser = argparse.ArgumentParser(
        description="MLX RL Trainer - Data Preprocessing Script"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment configuration YAML file.",
    )
    parser.add_argument(
        "--output-train-path",
        type=str,
        default="./data/processed_train.json",
        help="Path to save the processed training dataset.",
    )
    parser.add_argument(
        "--output-val-path",
        type=str,
        default="./data/processed_val.json",
        help="Path to save the processed validation dataset (optional).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging verbosity level.",
    )

    args = parser.parse_args()

    # Setup Logging
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, handlers=[RichHandler()], force=True)
    logger = logging.getLogger(__name__)

    logger.info("Starting data preprocessing script.")

    # Load ExperimentConfig
    config_path = Path(args.config)
    try:
        exp_config = ExperimentConfig.load_from_yaml(config_path)
        logger.info(f"Loaded experiment configuration from {config_path}.")
    except Exception as e:
        logger.critical(f"FATAL CONFIGURATION ERROR: {e}")
        sys.exit(1)

    # Run preprocessing
    try:
        asyncio.run(
            preprocess_and_save_dataset(
                exp_config,
                Path(args.output_train_path),
                Path(args.output_val_path) if args.output_val_path else None,
            )
        )
    except Exception as e:
        logger.critical(f"Data preprocessing failed: {e}", exc_info=True)
        sys.exit(1)

    logger.info("Data preprocessing script finished successfully.")


if __name__ == "__main__":
    main()

# Dependencies: datasets, aiofiles, mlx_lm, pydantic, rich
# Actions: Install: pip install datasets aiofiles
#          Run: python scripts/data_preprocessing.py --config configs/experiments/code_gen_base.yaml --output-train-path data/train.json --output-val-path data/val.json
# Status: Complete, production-ready.
