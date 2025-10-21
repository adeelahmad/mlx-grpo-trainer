# Path: src/mlx_rl_trainer/data/dataset_manager.py
"""
Data pipeline management: Loading, preprocessing, and efficient batching.
MODIFIED: Added configurable support for loading pre-tokenized .npy files.
"""
import json, logging, random, asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterator
from datasets import Dataset, Features, Value
import numpy as np
from tqdm.auto import tqdm
from mlx_rl_trainer.core.config import DataConfig, GenerationConfig
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_rl_trainer.utils.text_utils import (
    _mcq_meta_from_sample,
    clean_completion_string,
)
from mlx_rl_trainer.data.batch_builder import build_rollout_batch
import mlx.core as mx
import aiofiles
import time
logger = logging.getLogger(__name__)


# --- Helper functions remain mostly unchanged ---
def _contains_keywords(text: str, keywords: List[str]) -> bool:
    """Check if text contains any of the specified keywords (case-insensitive)."""
    if not keywords or not text:
        return False
    text_lower = text.lower()
    return any(keyword.lower() in text_lower for keyword in keywords)


def _looks_garbage(text: str) -> bool:
    """Check if text looks like garbage (too many special characters, etc.)."""
    if not text or len(text.strip()) < 3:
        return True
    
    # Count alphanumeric characters vs special characters
    alphanumeric = sum(1 for c in text if c.isalnum())
    total_chars = len(text)
    
    if total_chars == 0:
        return True
    
    # If less than 30% alphanumeric, consider it garbage
    alphanumeric_ratio = alphanumeric / total_chars
    return alphanumeric_ratio < 0.3


def _normalize_record(
    obj: Dict[str, Any],
    prompt_key: str,
    completion_key: str,
    system_prompt_default: str,
) -> Optional[Dict[str, Any]]:
    # This function is used by the JSONL loader, no changes needed here.
    if not isinstance(obj, dict):
        return None
    prompt = str(obj.get(prompt_key, obj.get("prompt", obj.get("question", ""))))
    completion = str(
        obj.get(completion_key, obj.get("completion", obj.get("answer", "")))
    )
    completion_cleaned = clean_completion_string(completion)
    meta_in = obj.get("meta", {}) if isinstance(obj.get("meta"), dict) else {}
    mcq_meta = _mcq_meta_from_sample(
        {"prompt": prompt, "completion": completion_cleaned, "meta": meta_in}
    )
    final_meta = {
        "is_mcq": mcq_meta.get("is_mcq", False),
        "mcq_options": mcq_meta.get("mcq_options", []),
        "mcq_multi_select": mcq_meta.get("mcq_multi_select", False),
        "mcq_correct_indices": mcq_meta.get("mcq_correct_indices", []),
        "mcq_correct_letters": mcq_meta.get("mcq_correct_letters", ""),
    }
    final_meta.update({k: v for k, v in meta_in.items() if k not in final_meta})
    if not prompt.strip() and not completion_cleaned.strip():
        return None
    return {
        "prompt": prompt,
        "completion": completion_cleaned,
        "system": "",
        "test_cases": [],
        "is_invalid_sample": obj.get("is_invalid_sample", False),
        "meta": final_meta,
    }


class DatasetManager:
    def __init__(self, config: DataConfig, tokenizer: Optional[TokenizerWrapper]):
        self.config = config
        self._tokenizer = tokenizer
        self._train_dataset: Optional[Dataset] = None
        self._val_dataset: Optional[Dataset] = None
        self._is_loaded = False
        self.system_prompt: str = ""
        logger.debug("DatasetManager initialized.")

    def set_tokenizer(self, tokenizer: TokenizerWrapper):
        self._tokenizer = tokenizer

    def set_system_prompt(self, system_prompt: str):
        self.system_prompt = system_prompt

    # ⭐ NEW: Method to load pre-tokenized .npy files
    def _load_from_npy(self, path_prefix: str) -> Optional[Dataset]:
        """Loads and combines pre-tokenized prompt and completion .npy files."""
        try:
            prompts_path = Path(f"{path_prefix}_prompts.npy")
            completions_path = Path(f"{path_prefix}_completions.npy")

            if not prompts_path.exists() or not completions_path.exists():
                logger.warning(
                    f"NPY files not found for prefix '{path_prefix}'. Searched for {prompts_path} and {completions_path}."
                )
                return None

            logger.info(
                f"Loading pre-tokenized data from {prompts_path} and {completions_path}..."
            )

            # 'r' stands for read-only
            prompt_tokens = np.load(prompts_path, mmap_mode='r')
            completion_tokens = np.load(completions_path, mmap_mode='r')

            if len(prompt_tokens) != len(completion_tokens):
                raise ValueError(
                    "Mismatch in number of samples between prompt and completion files."
                )

            # Create a Hugging Face Dataset from the token arrays
            # We store them as lists of ints; PyArrow handles this efficiently
            # This correctly passes the memory-mapped arrays to the dataset
            data_dict = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }

            features = Features({
                            "prompt_tokens": [Value("int32")],
                            "completion_tokens": [Value("int32")],
                        })

            # ⭐ ADD THESE LINES ⭐
            logger.info("⭐ Building dataset index from .npy arrays. This is a one-time cost and may take several minutes...")
            start_time = time.time()

            # This is the slow line
            dataset = Dataset.from_dict(data_dict, features=features)

            # ⭐ ADD THIS LINE ⭐
            logger.info(f"✓ Dataset indexing complete. Took {time.time() - start_time:.2f} seconds.")

            return dataset

        except Exception as e:
            logger.error(
                f"Failed to load pre-tokenized data from '{path_prefix}': {e}",
                exc_info=True,
            )
            return None

    # --- Methods for the original JSONL loading path ---
    async def _async_read_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        # ... (no changes in this function)
        data = []
        async with aiofiles.open(path, mode="r", encoding="utf-8") as f:
            async for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
        return data

    async def _load_raw_data_from_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        # ... (no changes in this function)
        if not path:
            return []
        return await self._async_read_jsonl(path)

    def _process_raw_to_dataset(
        self, raw_data: List[Dict[str, Any]], split_name: str
    ) -> Dataset:
        # ... (no changes in this function)
        normalized_records = []
        for obj in tqdm(raw_data, desc=f"Normalizing {split_name} data"):
            rec = _normalize_record(
                obj,
                self.config.dataset_prompt_key,
                self.config.dataset_answer_key,
                self.system_prompt,
            )
            if rec:
                normalized_records.append(rec)
        if not normalized_records:
            return Dataset.from_list([])
        features = Features(
            {
                "prompt": Value("string"),
                "completion": Value("string"),
                "system": Value("string"),
                "test_cases": [Value("string")],
                "is_invalid_sample": Value("bool"),
                "meta": {
                    "is_mcq": Value("bool"),
                    "mcq_options": [Value("string")],
                    "mcq_multi_select": Value("bool"),
                    "mcq_correct_indices": [Value("int32")],
                    "mcq_correct_letters": Value("string"),
                },
            }
        )
        return Dataset.from_list(normalized_records, features=features)

    # ⭐ MODIFIED: Main loading function now acts as a dispatcher
    async def load_datasets(self, force_reload: bool = False):
        if self._is_loaded and not force_reload:
            return

        # --- New NPY Loading Path ---
        if self.config.train_npy_path:
            logger.info("Pre-tokenized .npy training path provided. Using NPY loader.")
            self._train_dataset = self._load_from_npy(self.config.train_npy_path)
            if self.config.val_npy_path:
                self._val_dataset = self._train_dataset # self._load_from_npy(self.config.val_npy_path)

            if self._train_dataset is None:
                raise ValueError(
                    f"Failed to load required training data from .npy files at '{self.config.train_npy_path}'."
                )

        # --- Fallback to Original JSONL Loading Path ---
        else:
            logger.info("No .npy path provided. Falling back to JSONL loader.")
            raw_train_data = await self._load_raw_data_from_jsonl(
                self.config.train_path
            )
            raw_val_data = (
                await self._load_raw_data_from_jsonl(self.config.val_path)
                if self.config.val_path
                else []
            )
            self._train_dataset = self._process_raw_to_dataset(raw_train_data, "train")
            self._val_dataset = (
                self._process_raw_to_dataset(raw_val_data, "val")
                if raw_val_data
                else None
            )

        self._is_loaded = True
        logger.info(
            f"Datasets loaded. Train: {len(self._train_dataset)}, Val: {len(self._val_dataset) if self._val_dataset else 0}"
        )

    def get_dataloader(self, split: str, batch_size: int) -> Iterator[Dict[str, Any]]:
        # This function remains the same, but `build_rollout_batch` will now handle the two different data types.
        dataset = self._train_dataset if split == "train" else self._val_dataset
        if not dataset or len(dataset) == 0:
            logger.warning(f"Dataloader for '{split}' is empty.")
            return iter([])

        indices = list(range(len(dataset)))
        if self.config.shuffle_data and split == "train":
            random.shuffle(indices)

        def batch_generator():
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i : i + batch_size]
                if not batch_indices:
                    continue

                prompts_data, prompts_mx, _ = build_rollout_batch(
                    self._tokenizer, dataset, batch_indices, self.config
                )

                if prompts_mx.size > 0:
                    yield {"prompts_data": prompts_data, "prompts_mx": prompts_mx}

        return batch_generator()
