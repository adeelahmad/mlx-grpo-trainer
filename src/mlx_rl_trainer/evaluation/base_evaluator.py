"""Abstract interface for all evaluation benchmarks."""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
import json
from pathlib import Path
from datasets import Dataset, load_dataset
from mlx_lm.tokenizer_utils import TokenizerWrapper

from mlx_rl_trainer.core.trainer import EvaluationMetrics
from mlx_rl_trainer.core.exceptions import DataLoadError  # CORRECTED IMPORT

logger = logging.getLogger(__name__)


class BaseEvaluator(ABC):
    """Abstract base class for all evaluation benchmarks."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__
        self.dataset: Optional[Dataset] = None
        logger.debug(f"Initialized {self.name} evaluator with config: {config}")

    def load_dataset(
        self,
        dataset_path: str,
        dataset_subset: Optional[str] = None,
        split: str = "test",
    ) -> Dataset:
        """Loads an evaluation dataset from HuggingFace Hub or a local JSONL file."""
        try:
            path = Path(dataset_path)
            if path.exists() and path.suffix == ".jsonl":
                with open(path, "r", encoding="utf-8") as f:
                    data = [json.loads(line) for line in f if line.strip()]
                self.dataset = Dataset.from_list(data)
            else:
                self.dataset = load_dataset(dataset_path, dataset_subset, split=split)
            logger.info(
                f"Loaded dataset for {self.name}: {dataset_path} (split: {split}) with {len(self.dataset)} samples."
            )
            return self.dataset
        except Exception as e:
            raise DataLoadError(
                f"Failed to load dataset for {self.name} from {dataset_path}: {e}"
            ) from e

    @abstractmethod
    def evaluate(
        self, model: Any, tokenizer: TokenizerWrapper, dataset: Dataset
    ) -> Any:
        """Abstract method to run evaluation on the model."""
        pass

    def __repr__(self) -> str:
        return f"{self.name}(config={self.config})"
