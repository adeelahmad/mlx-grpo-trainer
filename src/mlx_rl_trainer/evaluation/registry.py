# file_path: mlx_rl_trainer/src/mlx_rl_trainer/evaluation/registry.py
# revision_no: 001
# goals_of_writing_code_block: Evaluator registry for plugin architecture.
# type_of_code_response: add new code
"""Evaluator registry for plugin architecture."""

from typing import Dict, Type, Any
import logging

from mlx_rl_trainer.evaluation.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


class EvaluatorRegistry:
    """
    Central registry for evaluation benchmark plugins.

    Allows dynamic registration and instantiation of `BaseEvaluator` subclasses
    by their unique string names. This promotes a modular and extensible
    evaluation framework.
    """

    _evaluators: Dict[str, Type[BaseEvaluator]] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register an evaluator class.

        Args:
            name: A unique string identifier for this evaluator.
        """

        def decorator(evaluator_class: Type[BaseEvaluator]):
            if name in cls._evaluators:
                logger.warning(
                    f"Overwriting existing evaluator: '{name}' with {evaluator_class.__name__}."
                )
            cls._evaluators[name] = evaluator_class
            logger.info(
                f"Registered evaluator: '{name}' -> {evaluator_class.__name__}."
            )
            return evaluator_class

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseEvaluator]:
        """
        Retrieves a registered evaluator class by its name.

        Args:
            name: The string identifier of the evaluator to retrieve.

        Returns:
            The `BaseEvaluator` subclass associated with the given name.

        Raises:
            KeyError: If no evaluator is registered with the given name.
        """
        if name not in cls._evaluators:
            available = ", ".join(cls._evaluators.keys())
            raise KeyError(
                f"Evaluator '{name}' not found. Available evaluators: [{available}]."
            )
        return cls._evaluators[name]

    @classmethod
    def create(cls, name: str, config: Dict[str, Any]) -> BaseEvaluator:
        """
        Instantiates a registered evaluator using its configuration.

        Args:
            name: The string identifier of the evaluator to create.
            config: A dictionary of configuration parameters to pass to the evaluator's constructor.

        Returns:
            An initialized instance of the `BaseEvaluator` subclass.

        Raises:
            Exception: If instantiation fails (e.g., due to invalid config).
        """
        try:
            evaluator_class = cls.get(name)
            instance = evaluator_class(config)
            logger.debug(f"Created evaluator instance: '{name}'.")
            return instance
        except Exception as e:
            logger.error(f"Failed to create evaluator '{name}': {e}", exc_info=True)
            raise

    @classmethod
    def list_available(cls) -> list[str]:
        """
        Returns a list of all currently registered evaluator names.
        """
        return list(cls._evaluators.keys())

    @classmethod
    def clear(cls) -> None:
        """
        Clears all registered evaluator functions from the registry.
        Primarily used for testing or dynamic reloading scenarios.
        """
        cls._evaluators.clear()
        logger.info("Cleared evaluator registry.")


register_evaluator = EvaluatorRegistry.register
