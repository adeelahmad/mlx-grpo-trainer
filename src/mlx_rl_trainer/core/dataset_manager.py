"""
Data pipeline management: Loading, preprocessing, and efficient batching.
"""
import json, logging, random, re, asyncio, gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterator
from datasets import Dataset, Features, Value, load_dataset
from tqdm.auto import tqdm
from mlx_rl_trainer.core.config import (
    DataConfig,
    THINK_STYLE_PROMPT_LITERAL,
    GenerationConfig,
    ExperimentConfig,
)
from mlx_rl_trainer.core.exceptions import DataLoadError
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_rl_trainer.utils.text_utils import (
    _contains_keywords,
    _mcq_meta_from_sample,
    apply_chat_template_wrapper,
    extract_think_region,
    _looks_garbage,
    clean_completion_string,
)
from mlx_rl_trainer.data.batch_builder import build_rollout_batch
import mlx.core as mx

logger = logging.getLogger(__name__)


def _normalize_record(
    obj: Dict[str, Any],
    prompt_key: str,
    completion_key: str,
    system_prompt_default: str,
) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return None

    def _s(x: Any) -> str:
        return str(x) if x is not None else ""

    prompt = _s(obj.get(prompt_key, obj.get("prompt", obj.get("question", ""))))
    completion = _s(
        obj.get(completion_key, obj.get("completion", obj.get("answer", "")))
    )
    system = _s(obj.get("system", system_prompt_default))

    gen_config_default = GenerationConfig()
    completion_cleaned = clean_completion_string(completion)

    meta_in = obj.get("meta", {}) if isinstance(obj.get("meta"), dict) else {}
    mcq_meta = _mcq_meta_from_sample(
        {"prompt": prompt, "completion": completion_cleaned, "meta": meta_in}
    )

    # Create a final, standardized meta dictionary for EVERY record
    final_meta = {
        "is_mcq": mcq_meta.get("is_mcq", False),
        "mcq_options": mcq_meta.get("mcq_options", []),
        "mcq_multi_select": mcq_meta.get("mcq_multi_select", False),
        "mcq_correct_indices": mcq_meta.get("mcq_correct_indices", []),
        "mcq_correct_letters": mcq_meta.get("mcq_correct_letters", ""),
    }
    # Merge any other original meta keys that don't conflict
    final_meta.update({k: v for k, v in meta_in.items() if k not in final_meta})

    test_cases = obj.get("test_cases", [])
    if not isinstance(test_cases, list):
        test_cases = [test_cases] if test_cases is not None else []
    # Convert dict test cases to JSON strings for PyArrow compatibility
    test_cases_str = [
        json.dumps(tc) if isinstance(tc, dict) else str(tc) for tc in test_cases
    ]

    if not prompt.strip() and not completion_cleaned.strip() and not system.strip():
        return None

    return {
        "prompt": prompt,
        "completion": completion_cleaned,
        "system": system,
        "test_cases": test_cases_str,
        "is_invalid_sample": obj.get("is_invalid_sample", False),
        "meta": final_meta,
    }


class DatasetManager:
    def __init__(self, config: ExperimentConfig):
        self.exp_config = config
        self.config = config.data
        self._tokenizer: Optional[TokenizerWrapper] = None
        self._train_dataset: Optional[Dataset] = None
        self._val_dataset: Optional[Dataset] = None
        self._is_loaded = False
        # Chunk size for processing large datasets
        self._processing_chunk_size = 10000
        logger.debug("DatasetManager initialized.")

    def set_tokenizer(self, tokenizer: TokenizerWrapper):
        self._tokenizer = tokenizer

    def _aggressive_memory_cleanup(self):
        """Aggressively free memory."""
        try:
            mx.metal.clear_cache()
        except:
            pass
        mx.clear_cache()
        gc.collect()

    async def _async_read_jsonl_streaming(self, path: Path) -> Iterator[Dict[str, Any]]:
        """Stream JSONL file line by line to avoid loading entire file into memory."""
        if not path.is_file():
            raise FileNotFoundError(f"Data file not found: {path}")

        import aiofiles

        async with aiofiles.open(path, mode="r", encoding="utf-8") as f:
            async for line in f:
                if line.strip():
                    try:
                        yield json.loads(line.strip())
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Malformed JSONL line in {path.name}, skipping."
                        )

    async def _async_read_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        """Read JSONL file - kept for compatibility but processes in chunks."""
        if not path.is_file():
            raise FileNotFoundError(f"Data file not found: {path}")

        data = []
        chunk = []

        import aiofiles

        async with aiofiles.open(path, mode="r", encoding="utf-8") as f:
            async for line in f:
                if line.strip():
                    try:
                        chunk.append(json.loads(line.strip()))

                        # Process in chunks to avoid memory spikes
                        if len(chunk) >= self._processing_chunk_size:
                            data.extend(chunk)
                            chunk = []
                            # Periodic cleanup
                            if len(data) % (self._processing_chunk_size * 5) == 0:
                                gc.collect()

                    except json.JSONDecodeError:
                        logger.warning(
                            f"Malformed JSONL line in {path.name}, skipping."
                        )

            # Add remaining chunk
            if chunk:
                data.extend(chunk)
                del chunk

        return data

    async def load_datasets(self, force_reload: bool = False):
        if self._is_loaded and not force_reload:
            return

        async def load_raw_data_for_split(
            path: Path, split_name: str
        ) -> List[Dict[str, Any]]:
            if not path:
                return []

            if path.suffix.lower() in [".jsonl", ".ndjson"]:
                raw_data = await self._async_read_jsonl(path)
                return raw_data
            elif path.suffix.lower() == ".json":
                import aiofiles

                raw_content = await (
                    await aiofiles.open(path, mode="r", encoding="utf-8")
                ).read()
                data = json.loads(raw_content)
                del raw_content
                gc.collect()
                return data
            else:
                hf_split_name = "train" if split_name == "train" else "test"
                dataset_obj = await asyncio.to_thread(
                    load_dataset, path.as_posix(), split=hf_split_name
                )
                data = (
                    dataset_obj.to_list()
                    if hasattr(dataset_obj, "to_list")
                    else list(dataset_obj)
                )
                del dataset_obj
                gc.collect()
                return data

        # Load train data
        raw_train_data = await load_raw_data_for_split(self.config.train_path, "train")
        self._train_dataset = self._process_raw_to_dataset(raw_train_data, "train")

        # Immediately free raw train data
        del raw_train_data
        self._aggressive_memory_cleanup()

        # Load validation data
        if self.config.val_path:
            raw_val_data = await load_raw_data_for_split(self.config.val_path, "val")
            self._val_dataset = self._process_raw_to_dataset(raw_val_data, "val")

            # Immediately free raw val data
            del raw_val_data
            self._aggressive_memory_cleanup()
        else:
            self._val_dataset = None

        self._is_loaded = True
        logger.info(
            f"Datasets loaded. Train: {len(self._train_dataset)}, Val: {len(self._val_dataset) if self._val_dataset else 0}"
        )

    def _process_raw_to_dataset(
        self, raw_data: List[Dict[str, Any]], split_name: str
    ) -> Dataset:
        """Process raw data in chunks to minimize memory footprint."""
        if not raw_data:
            logger.warning(f"No raw data provided for {split_name}.")
            return Dataset.from_list([])

        normalized_records = []
        chunk_size = self._processing_chunk_size

        # Process in chunks
        for chunk_start in range(0, len(raw_data), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(raw_data))
            chunk = raw_data[chunk_start:chunk_end]

            for obj in chunk:
                rec = _normalize_record(
                    obj,
                    self.config.dataset_prompt_key,
                    self.config.dataset_answer_key,
                    self.exp_config.system_prompt,
                )

                if (
                    rec
                    and not _looks_garbage(rec["prompt"])
                    and not _looks_garbage(rec["completion"])
                ):
                    if not self.config.dataset_filter_keywords or not (
                        _contains_keywords(
                            rec["prompt"], self.config.dataset_filter_keywords
                        )
                        or _contains_keywords(
                            rec["completion"], self.config.dataset_filter_keywords
                        )
                    ):
                        normalized_records.append(rec)

            # Cleanup after each chunk
            del chunk
            if chunk_start % (chunk_size * 5) == 0:
                gc.collect()

        if not normalized_records:
            logger.warning(f"No valid records found for {split_name}.")
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

        # Create dataset and immediately free normalized_records
        dataset = Dataset.from_list(normalized_records, features=features)
        del normalized_records
        gc.collect()

        return dataset

    def get_dataloader(self, split: str, batch_size: int) -> Iterator[Dict[str, Any]]:
        dataset = self._train_dataset if split == "train" else self._val_dataset
        if not dataset or len(dataset) == 0:
            logger.warning(f"Dataloader for '{split}' is empty.")
            return iter([])

        indices = list(range(len(dataset)))
        # if self.config.shuffle_data and split == "train":
        random.shuffle(indices)

        def batch_generator():
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i : i + batch_size]
                if not batch_indices:
                    continue

                # Pass the full ExperimentConfig to the batch builder
                prompts_data, prompts_mx, _ = build_rollout_batch(
                    self._tokenizer, dataset, batch_indices, self.exp_config
                )

                if prompts_mx.size > 0:
                    batch_dict = {
                        "prompts_data": prompts_data,
                        "prompts_mx": prompts_mx,
                    }
                    yield batch_dict

                    # Cleanup after yielding batch
                    del batch_dict
                else:
                    # Clean up even if we don't yield
                    del prompts_data, prompts_mx

                # Periodic aggressive cleanup every 10 batches
                if i % (batch_size * 10) == 0:
                    self._aggressive_memory_cleanup()

        return batch_generator()
