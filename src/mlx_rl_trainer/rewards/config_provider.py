"""
Centralized Configuration Provider for Reward Functions

This module provides a centralized configuration management system for reward functions,
implementing the Dependency Injection and Singleton patterns to ensure consistent
configuration across all reward components.

Key features:
- Singleton pattern for global configuration access
- Lazy loading of configuration files
- Configuration validation
- Default configuration fallbacks
- Thread-safe implementation
"""

import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Union, TypeVar, Type, cast

from mlx_rl_trainer.core.config import (
    GenerationConfig,
    ExperimentConfig,
    load_config
)

# Type variable for generic configuration types
T = TypeVar('T')

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Base exception for configuration-related errors."""
    pass


class ConfigurationNotFoundError(ConfigurationError):
    """Raised when a configuration file cannot be found."""
    pass


class ConfigurationValidationError(ConfigurationError):
    """Raised when configuration validation fails."""
    pass


class RewardConfigProvider:
    """
    Centralized configuration provider for reward functions.
    
    This class implements the Singleton pattern to ensure that all reward functions
    use the same configuration instance. It also provides methods for accessing
    specific configuration sections and values.
    
    Attributes:
        _instance: Singleton instance of the configuration provider
        _lock: Thread lock for thread-safe singleton access
        _config: Loaded configuration object
        _config_path: Path to the configuration file
        _default_config_path: Default path to look for configuration
    """
    
    _instance: Optional['RewardConfigProvider'] = None
    _lock = threading.RLock()
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration provider.
        
        Args:
            config_path: Path to the configuration file (optional)
        """
        self._config: Optional[ExperimentConfig] = None
        self._config_path = config_path
        # List of default config paths to try in order
        self._default_config_paths = [
            "config.yaml",
            "test_config.yaml",
            "test_config_minimal.yaml"
        ]
        self._initialized = False
        self._default_gen_config = None  # Will be lazily initialized if needed
    
    @classmethod
    def get_instance(cls, config_path: Optional[Union[str, Path]] = None) -> 'RewardConfigProvider':
        """
        Get or create the singleton instance of the configuration provider.
        
        Args:
            config_path: Path to the configuration file (optional)
            
        Returns:
            The singleton instance of the configuration provider
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(config_path)
        elif config_path is not None and cls._instance._config_path != config_path:
            # If a different config path is provided, update the instance
            with cls._lock:
                cls._instance._config_path = config_path
                cls._instance._config = None  # Force reload
                cls._instance._initialized = False
        
        return cls._instance
    
    def initialize(self) -> None:
        """
        Initialize the configuration provider by loading the configuration.
        
        This method is called automatically when configuration is accessed,
        but can also be called explicitly to pre-load configuration.
        
        Raises:
            ConfigurationNotFoundError: If the configuration file cannot be found
            ConfigurationValidationError: If the configuration is invalid
        """
        if self._initialized:
            return
            
        with self._lock:
            if self._initialized:  # Double-check after acquiring lock
                return
                
            try:
                # Try to load from specified path first
                if self._config_path:
                    config_path = Path(self._config_path)
                    if config_path.exists():
                        self._config = load_config(config_path)
                        logger.info(f"Loaded configuration from {config_path}")
                        self._initialized = True
                        return
                    else:
                        logger.warning(f"Configuration file not found at {config_path}")
                
                # Try default paths in order
                for default_path_str in self._default_config_paths:
                    default_path = Path(default_path_str)
                    if default_path.exists():
                        self._config = load_config(default_path)
                        logger.info(f"Loaded configuration from default path {default_path}")
                        self._initialized = True
                        return
                
                # Try environment variable
                env_config_path = os.environ.get("MLX_RL_CONFIG_PATH")
                if env_config_path:
                    env_path = Path(env_config_path)
                    if env_path.exists():
                        self._config = load_config(env_path)
                        logger.info(f"Loaded configuration from environment variable path {env_path}")
                        self._initialized = True
                        return
                
                # If we get here, no configuration was found
                # Instead of raising an error, create a default configuration
                logger.warning(f"No configuration file found. Using default configuration.")
                self._config = self._create_default_config()
                self._initialized = True
                return
                
            except Exception as e:
                if isinstance(e, ConfigurationError):
                    raise
                logger.error(f"Error initializing configuration: {e}")
                # Create a minimal default configuration as a last resort
                logger.warning("Creating minimal default configuration due to error")
                self._config = self._create_default_config()
                self._initialized = True
                
    def _create_default_config(self) -> Any:
        """
        Create a default configuration object when no configuration file is found.
        
        This method creates a minimal ExperimentConfig with default values for
        the most important configuration parameters, particularly those needed
        by reward functions.
        
        Returns:
            A default ExperimentConfig object
        """
        try:
            from mlx_rl_trainer.core.config import ExperimentConfig, GenerationConfig
            
            # Create a default GenerationConfig
            gen_config = GenerationConfig(
                think_start_tag="<think>",
                think_end_tag="</think>",
                answer_start_tag="",
                answer_end_tag="",
                think_boost_tokens=8,
                think_temperature=0.2,
                answer_temperature=0.3,
                sampling_top_p=0.6,
                sampling_min_p=0.0,
                sampling_top_k=80,
                repetition_penalty=1.1,
                repetition_context_size=20,
                min_tokens_to_keep=1,
                min_think_tokens=32,
                think_end_early_bias=-20.0,
                bias_answer_start_after_min_think=True,
                bias_close_think=12.0,
                bias_answer_start=10.0,
                punish_extra_think_end=-22.0,
                punish_reopen_think=-10.0,
                punish_reopen_answer=-9.0,
                bias_eos_after_answer=4.0,
                hard_mask_mcq_first_token=True,
                mcq_letter_lift=8.0,
                mcq_ban_first_bias=-14.0,
                nonmcq_ban_first_bias=-12.0,
                mcq_close_after_k=1,
                min_answer_tokens=8,
                min_answer_tokens_mcq=1,
                mcq_answer_end_bias=9.0,
                encourage_think_bias=4.5,
                ban_think_bias=-5.0,
                allow_tool_calls=True,
                tool_call_penalty=0.0,
                think_length_target_min=100,
                think_length_target_max=250,
                think_length_penalty_strength=0.5
            )
            
            # Create a minimal ExperimentConfig
            config = ExperimentConfig(
                experiment_name="default_config",
                description="Default configuration created by RewardConfigProvider",
                generation=gen_config,
                rewards=[]  # Empty rewards list
            )
            
            logger.info("Created default configuration")
            return config
            
        except Exception as e:
            logger.error(f"Failed to create default configuration: {e}")
            
            # Create a minimal object with the required attributes
            class MinimalConfig:
                def __init__(self):
                    self.experiment_name = "minimal_default_config"
                    self.description = "Minimal default configuration"
                    self.generation = self._create_minimal_gen_config()
                    self.rewards = []
                    
                def _create_minimal_gen_config(self):
                    class MinimalGenConfig:
                        def __init__(self):
                            self.think_start_tag = "<think>"
                            self.think_end_tag = "</think>"
                            self.answer_start_tag = ""
                            self.answer_end_tag = ""
                            self.think_length_target_min = 100
                            self.think_length_target_max = 250
                            self.think_length_penalty_strength = 0.5
                    return MinimalGenConfig()
            
            logger.info("Created minimal default configuration")
            return MinimalConfig()
    
    def get_config(self) -> ExperimentConfig:
        """
        Get the loaded configuration object.
        
        Returns:
            The loaded configuration object
            
        Raises:
            ConfigurationNotFoundError: If the configuration file cannot be found
            ConfigurationValidationError: If the configuration is invalid
        """
        if not self._initialized:
            self.initialize()
        
        if self._config is None:
            raise ConfigurationError("Configuration not loaded")
            
        return self._config
    
    def get_generation_config(self) -> GenerationConfig:
        """
        Get the generation configuration section.
        
        Returns:
            The generation configuration section
            
        Raises:
            ConfigurationError: If the configuration is not loaded or invalid
        """
        config = self.get_config()
        return config.generation
    
    def get_reward_config(self, reward_name: str) -> Dict[str, Any]:
        """
        Get the configuration for a specific reward function.
        
        Args:
            reward_name: Name of the reward function
            
        Returns:
            Configuration dictionary for the specified reward function
            
        Raises:
            ConfigurationError: If the configuration is not loaded or invalid
        """
        config = self.get_config()
        
        # Find the reward configuration by name
        for reward_config in config.rewards:
            if reward_config.name == reward_name:
                return reward_config.config
        
        # Return empty dict if not found
        logger.warning(f"No configuration found for reward '{reward_name}', using empty config")
        return {}
    
    def get_config_section(self, section_name: str, default: Optional[T] = None) -> Union[T, Any]:
        """
        Get a specific section of the configuration.
        
        Args:
            section_name: Name of the configuration section
            default: Default value to return if section not found
            
        Returns:
            The requested configuration section, or default if not found
            
        Raises:
            ConfigurationError: If the configuration is not loaded
        """
        config = self.get_config()
        
        if hasattr(config, section_name):
            return getattr(config, section_name)
        
        logger.warning(f"Configuration section '{section_name}' not found, using default")
        return default
    
    def get_config_value(self, path: str, default: Any = None) -> Any:
        """
        Get a specific configuration value using dot notation.
        
        Args:
            path: Path to the configuration value (e.g., "generation.think_start_tag")
            default: Default value to return if not found
            
        Returns:
            The requested configuration value, or default if not found
            
        Raises:
            ConfigurationError: If the configuration is not loaded
        """
        config = self.get_config()
        
        # Split the path into parts
        parts = path.split(".")
        
        # Navigate through the configuration
        current = config
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            elif isinstance(current, dict) and part in current:
                current = current[part]
            else:
                logger.warning(f"Configuration value '{path}' not found, using default")
                return default
        
        return current


class RewardConfigurable:
    """
    Mixin class for reward functions that need configuration.
    
    This class provides methods for accessing configuration values and
    initializing configuration from a centralized provider.
    """
    
    def __init__(self, config_provider: Optional[RewardConfigProvider] = None):
        """
        Initialize the configurable reward function.
        
        Args:
            config_provider: Configuration provider instance (optional)
        """
        self._config_provider = config_provider or RewardConfigProvider.get_instance()
    
    def get_generation_config(self) -> GenerationConfig:
        """
        Get the generation configuration.
        
        Returns:
            The generation configuration
        """
        return self._config_provider.get_generation_config()
    
    def get_reward_config(self, reward_name: str) -> Dict[str, Any]:
        """
        Get the configuration for a specific reward function.
        
        Args:
            reward_name: Name of the reward function
            
        Returns:
            Configuration dictionary for the specified reward function
        """
        return self._config_provider.get_reward_config(reward_name)
    
    def get_config_value(self, path: str, default: Any = None) -> Any:
        """
        Get a specific configuration value using dot notation.
        
        Args:
            path: Path to the configuration value (e.g., "generation.think_start_tag")
            default: Default value to return if not found
            
        Returns:
            The requested configuration value, or default if not found
        """
        return self._config_provider.get_config_value(path, default)


# Convenience function for getting the configuration provider
def get_config_provider(config_path: Optional[Union[str, Path]] = None) -> RewardConfigProvider:
    """
    Get the singleton instance of the configuration provider.
    
    Args:
        config_path: Path to the configuration file (optional)
        
    Returns:
        The singleton instance of the configuration provider
    """
    return RewardConfigProvider.get_instance(config_path)


# Convenience function for getting the generation configuration
def get_generation_config() -> GenerationConfig:
    """
    Get the generation configuration.
    
    Returns:
        The generation configuration
    """
    return get_config_provider().get_generation_config()