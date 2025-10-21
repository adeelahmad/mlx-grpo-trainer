# file_path: mlx_rl_trainer/src/mlx_rl_trainer/rewards/registry.py
# revision_no: 002
# goals_of_writing_code_block: Plugin registry system for reward functions with robust validation.
# type_of_code_response: change existing
"""
Reward function registry for plugin architecture.

This module implements a registry pattern for reward functions, allowing dynamic
registration and instantiation of reward components with comprehensive validation.
"""

from typing import Dict, Type, Any, List, Optional, Set
import logging
import inspect
import re

from mlx_rl_trainer.rewards.base_reward import BaseReward

logger = logging.getLogger(__name__)


class RewardValidationError(Exception):
    """Exception raised for reward validation errors."""
    pass


class RewardConfigValidationError(RewardValidationError):
    """Exception raised when a reward configuration fails validation."""
    pass


class RewardRegistry:
    """
    Central registry for reward function plugins.

    Allows dynamic registration and instantiation of `BaseReward` subclasses
    by their unique string names. This promotes a modular and extensible
    reward system with robust validation.
    
    Features:
        - Dynamic registration of reward functions
        - Type-safe instantiation with configuration validation
        - Comprehensive error handling with detailed feedback
        - Support for introspection of required configuration parameters
    """

    _rewards: Dict[str, Type[BaseReward]] = {}
    _required_config_params: Dict[str, Set[str]] = {}

    @classmethod
    def _extract_required_params(cls, reward_class: Type[BaseReward]) -> Set[str]:
        """
        Extract required configuration parameters from a reward class's __init__ method.
        
        Uses introspection to analyze the constructor signature and docstring to determine
        which configuration parameters are required.
        
        Args:
            reward_class: The BaseReward subclass to analyze
            
        Returns:
            A set of required parameter names
        """
        required_params = set()
        
        try:
            # Inspect the __init__ method signature
            init_signature = inspect.signature(reward_class.__init__)
            
            # Skip 'self' and 'config' parameters
            # The first parameter is 'self', the second is 'config'
            # We're looking for parameters that are accessed from the config dict
            
            # Analyze the __init__ method source code to find config.get() calls
            try:
                init_source = inspect.getsource(reward_class.__init__)
                
                # Look for patterns like: self.param = config["param"] or self.param = config.get("param")
                # This is a simple heuristic and might not catch all cases
                required_matches = re.findall(r'self\.[\w_]+ = config\["([\w_]+)"\]', init_source)
                required_params.update(required_matches)
                
                # Also check for config.get() calls without default values
                # This is a more complex pattern that might need refinement
                get_matches = re.findall(r'config\.get\("([\w_]+)"\)', init_source)
                required_params.update(get_matches)
                
            except (IOError, TypeError):
                logger.debug(f"Could not analyze source code for {reward_class.__name__}")
            
            # If we couldn't determine required params, assume 'weight' is required
            # as it's used by the base class
            if not required_params:
                required_params.add("weight")
                
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to extract parameters from {reward_class.__name__}: {e}")
            # Default to requiring 'weight' if we can't determine requirements
            required_params.add("weight")
            
        return required_params

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a reward function class.

        Registers the reward class and analyzes its constructor to determine
        required configuration parameters.

        Args:
            name: A unique string identifier for this reward function.
        """

        def decorator(reward_class: Type[BaseReward]):
            if name in cls._rewards:
                logger.warning(
                    f"Overwriting existing reward: '{name}' with {reward_class.__name__}."
                )
            
            # Store the reward class
            cls._rewards[name] = reward_class
            
            # Extract and store required configuration parameters
            required_params = cls._extract_required_params(reward_class)
            cls._required_config_params[name] = required_params
            
            logger.info(f"Registered reward: '{name}' -> {reward_class.__name__} "
                       f"(required params: {', '.join(required_params) or 'none'}).")
            return reward_class

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseReward]:
        """
        Retrieves a registered reward class by its name.

        Args:
            name: The string identifier of the reward to retrieve.

        Returns:
            The `BaseReward` subclass associated with the given name.

        Raises:
            KeyError: If no reward is registered with the given name.
        """
        if name not in cls._rewards:
            available = ", ".join(cls._rewards.keys())
            raise KeyError(
                f"Reward '{name}' not found. Available rewards: [{available}]."
            )
        return cls._rewards[name]

    @classmethod
    def validate_config(cls, name: str, config: Dict[str, Any]) -> None:
        """
        Validates a reward configuration against the required parameters.
        
        Ensures that all required parameters are present in the config and
        that their types are appropriate.
        
        Args:
            name: The string identifier of the reward to validate.
            config: A dictionary of configuration parameters to validate.
            
        Raises:
            RewardConfigValidationError: If validation fails.
        """
        if not isinstance(config, dict):
            raise RewardConfigValidationError(
                f"Config must be a dictionary, got {type(config).__name__}"
            )
            
        # Check for required parameters
        if name in cls._required_config_params:
            required_params = cls._required_config_params[name]
            missing_params = [param for param in required_params if param not in config]
            
            if missing_params:
                raise RewardConfigValidationError(
                    f"Missing required config parameters for reward '{name}': {', '.join(missing_params)}"
                )
        
        # Validate weight parameter if present
        if "weight" in config:
            weight = config["weight"]
            if not isinstance(weight, (int, float)):
                raise RewardConfigValidationError(
                    f"Weight must be a number, got {type(weight).__name__}"
                )
            if weight < 0:
                raise RewardConfigValidationError(
                    f"Weight must be non-negative, got {weight}"
                )
                
        # Validate common parameters that have specific requirements
        if "min_length" in config and not isinstance(config["min_length"], int):
            raise RewardConfigValidationError(
                f"min_length must be an integer, got {type(config['min_length']).__name__}"
            )
            
        if "smoothing_window_size" in config and not isinstance(config["smoothing_window_size"], int):
            raise RewardConfigValidationError(
                f"smoothing_window_size must be an integer, got {type(config['smoothing_window_size']).__name__}"
            )
            
        logger.debug(f"Config validation passed for reward '{name}'")

    @classmethod
    def create(cls, name: str, config: Dict[str, Any]) -> BaseReward:
        """
        Instantiates a registered reward function using its configuration.

        Validates the configuration before instantiation to ensure all required
        parameters are present and have appropriate types.

        Args:
            name: The string identifier of the reward to create.
            config: A dictionary of configuration parameters to pass to the reward's constructor.

        Returns:
            An initialized instance of the `BaseReward` subclass.

        Raises:
            KeyError: If no reward is registered with the given name.
            RewardConfigValidationError: If the configuration is invalid.
            Exception: If instantiation fails for any other reason.
        """
        try:
            # Get the reward class
            reward_class = cls.get(name)
            
            # Create a copy of the config to avoid modifying the original
            config_copy = config.copy()
            
            # Ensure weight parameter exists with a default value if not provided
            if "weight" not in config_copy:
                config_copy["weight"] = 1.0
                logger.debug(f"Using default weight=1.0 for reward '{name}'")
                
            # Validate the configuration
            cls.validate_config(name, config_copy)
            
            # Instantiate the reward
            instance = reward_class(config_copy)
            
            logger.debug(f"Created reward instance: '{name}'.")
            return instance
        except RewardValidationError as e:
            logger.error(f"Validation failed for reward '{name}': {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to create reward '{name}': {e}", exc_info=True)
            raise

    @classmethod
    def get_required_params(cls, name: str) -> Set[str]:
        """
        Returns the set of required parameters for a reward.
        
        Args:
            name: The string identifier of the reward.
            
        Returns:
            A set of parameter names that are required for the reward.
            
        Raises:
            KeyError: If no reward is registered with the given name.
        """
        if name not in cls._rewards:
            raise KeyError(f"Reward '{name}' not found.")
        
        return cls._required_config_params.get(name, set())
    
    @classmethod
    def get_config_schema(cls, name: str) -> Dict[str, Dict[str, Any]]:
        """
        Returns a schema describing the configuration parameters for a reward.
        
        This provides information about parameter types, requirements, and defaults
        that can be used for validation or UI generation.
        
        Args:
            name: The string identifier of the reward.
            
        Returns:
            A dictionary mapping parameter names to their schema information.
            
        Raises:
            KeyError: If no reward is registered with the given name.
        """
        if name not in cls._rewards:
            raise KeyError(f"Reward '{name}' not found.")
            
        reward_class = cls._rewards[name]
        required_params = cls._required_config_params.get(name, set())
        
        # Start with common parameters
        schema = {
            "weight": {
                "type": "number",
                "required": "weight" in required_params,
                "default": 1.0,
                "description": "Weight factor for this reward in composition"
            },
            "smoothing_window_size": {
                "type": "integer",
                "required": "smoothing_window_size" in required_params,
                "default": 5,
                "description": "Window size for reward smoothing"
            }
        }
        
        # Try to extract additional parameters from docstring
        try:
            docstring = reward_class.__doc__ or ""
            config_section = re.search(r"Configuration:\s*(.*?)(?:\n\n|\Z)", docstring, re.DOTALL)
            
            if config_section:
                config_text = config_section.group(1)
                param_matches = re.findall(r"(\w+)\s*\(([\w\s,]+)\):\s*([^\n]+)", config_text)
                
                for param_name, param_type, description in param_matches:
                    param_type = param_type.strip().lower()
                    schema[param_name] = {
                        "type": "string" if "str" in param_type else
                               "boolean" if "bool" in param_type else
                               "integer" if "int" in param_type else
                               "number" if any(t in param_type for t in ["float", "num"]) else
                               "object",
                        "required": param_name in required_params,
                        "description": description.strip()
                    }
        except Exception as e:
            logger.debug(f"Could not extract config schema from docstring for {name}: {e}")
            
        return schema

    @classmethod
    def list_available(cls) -> List[str]:
        """
        Returns a list of all currently registered reward names.
        
        Returns:
            A list of string identifiers for all registered rewards.
        """
        return list(cls._rewards.keys())
    
    @classmethod
    def get_reward_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        Returns detailed information about all registered rewards.
        
        This is useful for generating documentation or UI components.
        
        Returns:
            A dictionary mapping reward names to their metadata.
        """
        info = {}
        for name in cls._rewards:
            reward_class = cls._rewards[name]
            required_params = cls._required_config_params.get(name, set())
            
            # Extract description from docstring
            docstring = reward_class.__doc__ or ""
            description = docstring.split("\n\n")[0].strip() if docstring else ""
            
            info[name] = {
                "class_name": reward_class.__name__,
                "description": description,
                "required_params": list(required_params),
                "config_schema": cls.get_config_schema(name)
            }
            
        return info

    @classmethod
    def clear(cls) -> None:
        """
        Clears all registered reward functions from the registry.
        Primarily used for testing or dynamic reloading scenarios.
        """
        cls._rewards.clear()
        cls._required_config_params.clear()
        logger.info("Cleared reward registry.")


# Convenience alias for the register decorator
register_reward = RewardRegistry.register
