"""Abstract base class for reward functions."""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import logging
import numpy as np

from .context import RewardContext

logger = logging.getLogger(__name__)


class BaseReward(ABC):
    """
    Abstract base class for all reward functions.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__
        self.weight = config.get("weight", 1.0)
        self.smoothing_window_size = config.get("smoothing_window_size", 5)
        self._reward_history: List[float] = []
        logger.debug(f"Initialized {self.name} with config: {config}")

    def _validate_result_dict(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates and normalizes the result dictionary from compute methods.
        
        Ensures the result dictionary has the required format and types.
        
        Args:
            result: The result dictionary to validate
            
        Returns:
            A normalized result dictionary with at least a 'reward' key
            
        Raises:
            ValueError: If the result is missing required keys or has invalid values
        """
        if not isinstance(result, dict):
            logger.warning(f"{self.name}: compute returned non-dict result, creating fallback")
            return {"reward": 0.0, "log": {"error": "Invalid return type"}}
            
        if "reward" not in result:
            logger.warning(f"{self.name}: compute result missing 'reward' key, using 0.0")
            result["reward"] = 0.0
            
        reward_value = result["reward"]
        if not isinstance(reward_value, (int, float)):
            logger.warning(f"{self.name}: reward value is not a number, converting to float")
            try:
                result["reward"] = float(reward_value) if reward_value is not None else 0.0
            except (ValueError, TypeError):
                logger.error(f"{self.name}: could not convert reward value to float")
                result["reward"] = 0.0
                
        # Ensure reward is in valid range [0.0, 1.0]
        result["reward"] = max(0.0, min(1.0, result["reward"]))
        
        # Ensure log is a dictionary if present
        if "log" in result and not isinstance(result["log"], dict):
            logger.warning(f"{self.name}: log is not a dictionary, converting")
            result["log"] = {"value": str(result["log"])}
            
        return result

    @abstractmethod
    def compute(self, context: RewardContext) -> Dict[str, Any]:
        """
        Compute reward for a single response.
        
        This method must be implemented by all reward subclasses.
        
        ⭐ MUST return a dictionary with at least a 'reward' key.
        e.g., {"reward": 0.8, "log": {"details": ...}}
        
        Args:
            context: The RewardContext object containing all necessary data
            
        Returns:
            A dictionary with at least a 'reward' key containing a float value
            between 0.0 and 1.0, and optionally a 'log' key with additional info
            
        Raises:
            NotImplementedError: This is an abstract method that must be implemented
        """
        raise NotImplementedError

    def _smooth_reward(self, current_reward: float) -> float:
        """Applies simple moving average smoothing to the reward."""
        self._reward_history.append(current_reward)
        if len(self._reward_history) > self.smoothing_window_size:
            self._reward_history.pop(0)
        return float(np.mean(self._reward_history))

    def batch_compute(self, contexts: List[RewardContext]) -> List[Dict[str, Any]]:
        """
        Default implementation calls `compute()` for each item.
        
        This is the generic fallback for all reward functions. It handles validation,
        error handling, and result normalization for each context in the batch.
        
        Args:
            contexts: A list of RewardContext objects to process
            
        Returns:
            A list of dictionaries containing reward scores and logs
        """
        if not isinstance(contexts, list):
            logger.error(f"{self.name}: batch_compute received non-list contexts")
            return [{"total": 0.0, self.name: 0.0, "log": {"error": "Invalid contexts type"}}]
            
        rewards_list = []
        for i, context in enumerate(contexts):
            try:
                # Validate the context
                try:
                    self.validate_inputs(context)
                except (ValueError, TypeError) as e:
                    logger.warning(f"{self.name}: Context validation failed: {e}")
                    rewards_list.append({
                        self.name: 0.0,
                        "total": 0.0,
                        "log": {"error": f"Invalid context: {str(e)}"}
                    })
                    continue
                    
                # Compute the reward
                result_dict = self.compute(context)
                logger.debug(f"{self.name}: Raw result_dict {result_dict}")
                
                # Validate and normalize the result
                result_dict = self._validate_result_dict(result_dict)

                # Extract the float score from the dictionary
                raw_score = result_dict.get('reward', 0.0)

                # Smooth the float score
                smoothed_score = self._smooth_reward(raw_score)

                # Structure the final output for the composer
                output = {
                    self.name: smoothed_score,
                    "total": smoothed_score,
                    "log": result_dict.get('log', {})
                }
                rewards_list.append(output)
                
            except Exception as e:
                logger.error(
                    f"Batch computation failed in {self.name} for context {i}: {e}",
                    exc_info=True,
                )
                rewards_list.append({
                    self.name: 0.0,
                    "total": 0.0,
                    "log": {"error": f"Exception: {str(e)}"}
                })
                
        return rewards_list

    def validate_inputs(self, context: RewardContext) -> None:
        """
        Validates the input context for reward computation.
        
        Performs comprehensive validation of the RewardContext object to ensure
        it contains all required fields with appropriate values before computation.
        
        Args:
            context: The RewardContext object to validate
            
        Raises:
            ValueError: If the context is invalid or missing required fields
            TypeError: If the context is not a RewardContext instance
        """
        # Type validation
        if not isinstance(context, RewardContext):
            raise TypeError(f"Context must be RewardContext, got {type(context)}")
            
        # Required field validation
        if context.generated_text is None:
            raise ValueError("Context missing required field: generated_text")
            
        if context.prompt_text is None:
            raise ValueError("Context missing required field: prompt_text")
            
        if context.reference_completion is None:
            raise ValueError("Context missing required field: reference_completion")
            
        # Type validation for fields
        if not isinstance(context.generated_text, str):
            raise TypeError(f"generated_text must be a string, got {type(context.generated_text)}")
            
        if not isinstance(context.prompt_text, str):
            raise TypeError(f"prompt_text must be a string, got {type(context.prompt_text)}")
            
        if not isinstance(context.reference_completion, str):
            raise TypeError(f"reference_completion must be a string, got {type(context.reference_completion)}")
            
        if not isinstance(context.test_cases, list):
            raise TypeError(f"test_cases must be a list, got {type(context.test_cases)}")
            
        if not isinstance(context.metadata, dict):
            raise TypeError(f"metadata must be a dictionary, got {type(context.metadata)}")
            
        # Validate test cases if present
        for i, test_case in enumerate(context.test_cases):
            if not isinstance(test_case, dict):
                raise TypeError(f"Test case at index {i} must be a dictionary, got {type(test_case)}")
                
        # Log validation success at debug level
        logger.debug(f"Context validation passed for {self.name}")

    def __repr__(self) -> str:
        return f"{self.name}(config={self.config})"


class InvalidWeightError(ValueError):
    """Exception raised for invalid reward weights."""
    pass


class RewardComposer:
    """
    Composes multiple `BaseReward` functions with specified weights.
    
    This class combines multiple reward functions into a single composite reward,
    applying weights to each component and ensuring proper normalization.
    
    Features:
    - Strong validation of reward weights
    - Automatic weight normalization
    - Comprehensive error handling
    - Detailed logging of weight distribution
    
    Example:
        ```python
        reward1 = SemanticSimilarityReward({"weight": 0.7})
        reward2 = TagStructureReward({"weight": 0.3})
        composer = RewardComposer([(reward1, 0.7), (reward2, 0.3)])
        results = composer.batch_compute(contexts)
        ```
    """

    def __init__(
        self,
        rewards: List[Tuple[BaseReward, float]],
        context_cls: type = RewardContext,
        auto_normalize: bool = True,
        validate_weights: bool = True
    ):
        """
        Initialize the RewardComposer with a list of reward functions and weights.
        
        Args:
            rewards: List of (reward_function, weight) tuples
            context_cls: The context class to use (default: RewardContext)
            auto_normalize: Whether to automatically normalize weights to sum to 1.0
            validate_weights: Whether to perform strict validation of weights
            
        Raises:
            InvalidWeightError: If weights are invalid and validate_weights is True
        """
        if not rewards:
            raise ValueError("RewardComposer requires at least one reward function")
            
        # Validate individual weights
        if validate_weights:
            self._validate_weights(rewards)
            
        # Store original weights for reference
        self.original_rewards = list(rewards)
        self.original_weight_sum = sum(weight for _, weight in rewards)
        
        # Normalize weights if requested
        if auto_normalize and not (0.99 <= self.original_weight_sum <= 1.01):
            logger.info(f"Normalizing reward weights (original sum: {self.original_weight_sum:.4f})")
            self.rewards = self._normalize_weights(rewards)
            self.weights_normalized = True
        else:
            self.rewards = rewards
            self.weights_normalized = False
            
        # Calculate the actual weight sum used
        self.total_weight_sum = sum(weight for _, weight in self.rewards)
        self.context_cls = context_cls
        
        # Log weight information
        if not (0.99 <= self.total_weight_sum <= 1.01):
            logger.warning(
                f"Reward weights do not sum to 1.0 (got {self.total_weight_sum:.4f})."
            )
            
        # Log detailed weight distribution
        weight_info = ", ".join([f"{r.name}: {w:.3f}" for r, w in self.rewards])
        logger.info(f"Initialized RewardComposer with {len(rewards)} rewards: {weight_info}")
        
    def _validate_weights(self, rewards: List[Tuple[BaseReward, float]]) -> None:
        """
        Validate individual reward weights.
        
        Args:
            rewards: List of (reward_function, weight) tuples
            
        Raises:
            InvalidWeightError: If any weight is invalid
        """
        for i, (reward, weight) in enumerate(rewards):
            # Check if weight is a number
            if not isinstance(weight, (int, float)):
                raise InvalidWeightError(
                    f"Weight for reward '{reward.name}' must be a number, got {type(weight)}"
                )
                
            # Check if weight is negative
            if weight < 0:
                raise InvalidWeightError(
                    f"Weight for reward '{reward.name}' cannot be negative (got {weight})"
                )
                
            # Check if weight is too large
            if weight > 10.0:
                logger.warning(
                    f"Weight for reward '{reward.name}' is unusually large ({weight}). "
                    f"Consider normalizing weights."
                )
                
        # Check if any weights are provided
        total = sum(weight for _, weight in rewards)
        if total <= 0:
            raise InvalidWeightError(
                f"Sum of weights must be positive (got {total})"
            )
            
    def _normalize_weights(self, rewards: List[Tuple[BaseReward, float]]) -> List[Tuple[BaseReward, float]]:
        """
        Normalize weights to sum to 1.0.
        
        Args:
            rewards: List of (reward_function, weight) tuples
            
        Returns:
            List of (reward_function, normalized_weight) tuples
        """
        total = sum(weight for _, weight in rewards)
        if total <= 0:
            # Fallback to equal weights if total is zero or negative
            equal_weight = 1.0 / len(rewards)
            logger.warning(
                f"Cannot normalize non-positive weights (sum={total}). "
                f"Using equal weights ({equal_weight:.4f}) instead."
            )
            return [(reward, equal_weight) for reward, _ in rewards]
            
        # Normalize weights
        return [(reward, weight / total) for reward, weight in rewards]

    def batch_compute(self, contexts: List[RewardContext]) -> List[Dict[str, float]]:
        """
        Computes rewards for a batch, leveraging individual batch_compute methods.
        
        This method orchestrates the computation of rewards across multiple reward functions
        for a batch of contexts. It handles error cases gracefully and ensures consistent
        output format regardless of individual reward function behavior.
        
        Args:
            contexts: List of RewardContext objects to evaluate
            
        Returns:
            List of dictionaries containing reward scores and logs, with each dictionary
            having reward function names as keys and their scores as values, plus a "total" key
            with the weighted sum of all rewards.
        """
        if not contexts:
            logger.warning("Empty contexts list provided to RewardComposer.batch_compute")
            return []
            
        # Dictionary to store results from each reward function
        all_individual_batch_results: Dict[str, List[Dict[str, Any]]] = {}

        # Process each reward function
        for reward_fn, _ in self.rewards:
            try:
                # Get batch results from the reward function
                batch_result = reward_fn.batch_compute(contexts)
                
                # Validate batch_result is a list
                if not isinstance(batch_result, list):
                    logger.error(f"Reward {reward_fn.name} batch_compute() returned {type(batch_result)}, expected List[Dict[str, Any]]")
                    # Create fallback result
                    batch_result = [{"total": 0.0, reward_fn.name: 0.0, "log": {"error": "Invalid return type"}} for _ in contexts]
                
                # Validate each item in the batch result
                for i, item in enumerate(batch_result):
                    if not isinstance(item, dict):
                        logger.error(f"Reward {reward_fn.name} batch_compute()[{i}] returned {type(item)}, expected Dict[str, Any]")
                        batch_result[i] = {"total": 0.0, reward_fn.name: 0.0, "log": {"error": "Invalid item type"}}
                    elif "total" not in item:
                        logger.warning(f"Reward {reward_fn.name} batch_compute()[{i}] missing 'total' key, using 0.0")
                        item["total"] = 0.0
                
                # Ensure batch_result has the same length as contexts
                if len(batch_result) != len(contexts):
                    logger.error(f"Reward {reward_fn.name} batch_compute() returned {len(batch_result)} results, expected {len(contexts)}")
                    # Pad or truncate to match contexts length
                    if len(batch_result) < len(contexts):
                        # Pad with default values
                        batch_result.extend([
                            {"total": 0.0, reward_fn.name: 0.0, "log": {"error": "Missing result"}}
                            for _ in range(len(contexts) - len(batch_result))
                        ])
                    else:
                        # Truncate to match contexts length
                        batch_result = batch_result[:len(contexts)]
                
                all_individual_batch_results[reward_fn.name] = batch_result
                
            except Exception as e:
                logger.error(f"Exception in {reward_fn.name}.batch_compute(): {e}", exc_info=True)
                # Create fallback result for this reward function
                all_individual_batch_results[reward_fn.name] = [
                    {"total": 0.0, reward_fn.name: 0.0, "log": {"error": f"Exception: {str(e)}"}}
                    for _ in contexts
                ]

        # Compose the final results
        composed_batch_results: List[Dict[str, float]] = []
        for i in range(len(contexts)):
            try:
                # Dictionary to store individual reward scores for this sample
                individual_results_for_sample = {}
                weighted_sum_for_sample = 0.0
                logs_for_sample = {}

                # Process each reward function's result for this sample
                for reward_fn, weight in self.rewards:
                    try:
                        # Get the result for this reward function and sample
                        result_item = all_individual_batch_results[reward_fn.name][i]
                        
                        # Extract and validate the score
                        if not isinstance(result_item, dict):
                            logger.error(f"Reward {reward_fn.name} sample {i}: Expected dict, got {type(result_item)}")
                            raw_score_for_sample = 0.0
                        else:
                            raw_score_for_sample = result_item.get("total", 0.0)
                            if not isinstance(raw_score_for_sample, (int, float)):
                                logger.warning(f"Reward {reward_fn.name} sample {i}: 'total' is {type(raw_score_for_sample)}, converting to float")
                                raw_score_for_sample = float(raw_score_for_sample) if raw_score_for_sample is not None else 0.0
                            
                            # Collect logs if available
                            if "log" in result_item and isinstance(result_item["log"], dict):
                                logs_for_sample[reward_fn.name] = result_item["log"]
                        
                        # Store the score and add to weighted sum
                        individual_results_for_sample[reward_fn.name] = raw_score_for_sample
                        weighted_sum_for_sample += raw_score_for_sample * weight
                        
                    except Exception as e:
                        logger.error(f"Batch compose failed for reward '{reward_fn.name}' sample idx {i}: {e}", exc_info=True)
                        individual_results_for_sample[reward_fn.name] = 0.0

                # Calculate the final weighted score
                if self.total_weight_sum > 0:
                    final_total_for_sample = weighted_sum_for_sample / self.total_weight_sum
                else:
                    # Fallback if weights sum to zero
                    final_total_for_sample = sum(individual_results_for_sample.values()) / len(self.rewards) if self.rewards else 0.0
                    logger.warning(f"Using unweighted average for sample {i} due to zero weight sum")
                
                # Ensure the final score is in the valid range [0.0, 1.0]
                individual_results_for_sample["total"] = float(np.clip(final_total_for_sample, 0.0, 1.0))
                
                # Add logs to the result
                if logs_for_sample:
                    individual_results_for_sample["logs"] = logs_for_sample
                
                composed_batch_results.append(individual_results_for_sample)
                
            except Exception as e:
                logger.error(f"Failed to compose results for sample {i}: {e}", exc_info=True)
                # Fallback result for this sample
                composed_batch_results.append({"total": 0.0, "error": str(e)})

        return composed_batch_results
