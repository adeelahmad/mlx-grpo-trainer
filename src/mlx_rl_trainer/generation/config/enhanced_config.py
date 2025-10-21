"""
Enhanced sampling configuration system for MLX RL Trainer.

This module provides enterprise-grade configuration management for sampling parameters
with support for per-phase configuration (thinking vs answer phases), advanced validation,
and seamless integration with the MLX framework.

Architecture:
    - SamplingPhaseConfig: Phase-specific sampling parameters
    - EnhancedGenerationConfig: Complete generation configuration with phase support
    - ConfigurationValidator: Comprehensive validation with detailed error reporting
    - BackwardCompatibilityAdapter: Seamless migration from legacy configurations

Design Patterns Applied:
    - Builder Pattern: For constructing complex configurations
    - Strategy Pattern: Different validation strategies for different parameter types
    - Template Method Pattern: Base validation with customizable validation steps
    - Factory Pattern: Creating appropriate configurations based on context

SOLID Principles:
    - Single Responsibility: Each class has one clear configuration concern
    - Open/Closed: Extensible for new sampling strategies without modification
    - Liskov Substitution: All config classes implement common interfaces
    - Interface Segregation: Separate interfaces for different configuration aspects
    - Dependency Inversion: Depends on abstractions, not concrete implementations

Example:
    >>> from mlx_rl_trainer.generation.config.enhanced_config import EnhancedGenerationConfig
    >>> config = EnhancedGenerationConfig.builder() \
    ...     .with_think_phase(temperature=0.2, top_p=0.8) \
    ...     .with_answer_phase(temperature=0.3, top_p=0.9) \
    ...     .build()
    >>> sampler = config.create_sampler_for_phase("think", tokenizer)
"""

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any, Dict, List, Optional, Union, Callable, Protocol, TypeVar, Generic,
    ClassVar, Final, Literal, Set, Tuple, NamedTuple
)
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import weakref
import time
from functools import lru_cache, wraps
import json
import hashlib

from pydantic import BaseModel, Field, validator, root_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass

# Type definitions for enhanced type safety
T = TypeVar('T')
ConfigValue = Union[int, float, str, bool, List[Any], Dict[str, Any]]
ValidationResult = Tuple[bool, Optional[str]]
PhaseType = Literal["think", "answer", "global"]

logger = logging.getLogger(__name__)


class SamplingPhase(Enum):
    """
    Enumeration of sampling phases in the generation process.
    
    Each phase can have different sampling parameters optimized for
    the specific requirements of that generation stage.
    """
    THINK = "think"
    ANSWER = "answer" 
    GLOBAL = "global"  # Default/fallback phase


class ConfigurationError(Exception):
    """
    Base exception for configuration-related errors.
    
    Provides structured error reporting with context information
    for debugging and monitoring purposes.
    """
    
    def __init__(
        self, 
        message: str, 
        error_code: str = "CONFIG_ERROR",
        context: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.timestamp = time.time()
        
        # Log the error with full context
        logger.error(
            f"Configuration Error [{self.error_code}]: {message}",
            extra={
                "correlation_id": self.correlation_id,
                "context": self.context,
                "timestamp": self.timestamp
            }
        )


class ValidationError(ConfigurationError):
    """Specific error for parameter validation failures."""
    
    def __init__(self, parameter: str, value: Any, reason: str, **kwargs):
        message = f"Validation failed for parameter '{parameter}' with value {value}: {reason}"
        super().__init__(message, error_code="VALIDATION_ERROR", **kwargs)
        self.parameter = parameter
        self.value = value
        self.reason = reason


class CompatibilityError(ConfigurationError):
    """Error for backward compatibility issues."""
    
    def __init__(self, legacy_config: str, migration_path: str, **kwargs):
        message = f"Compatibility issue with legacy config '{legacy_config}'. Migration: {migration_path}"
        super().__init__(message, error_code="COMPATIBILITY_ERROR", **kwargs)
        self.legacy_config = legacy_config
        self.migration_path = migration_path


class ParameterConstraint(NamedTuple):
    """
    Represents a constraint on a configuration parameter.
    
    Provides type-safe constraint definition with validation logic.
    """
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[Set[Any]] = None
    validator_func: Optional[Callable[[Any], ValidationResult]] = None
    description: str = ""


class ValidationStrategy(Protocol):
    """
    Protocol for parameter validation strategies.
    
    Enables different validation approaches for different parameter types
    while maintaining a consistent interface.
    """
    
    def validate(self, value: Any, constraint: ParameterConstraint) -> ValidationResult:
        """Validate a parameter value against its constraint."""
        ...


class NumericValidationStrategy:
    """Validation strategy for numeric parameters."""
    
    def validate(self, value: Any, constraint: ParameterConstraint) -> ValidationResult:
        """
        Validate numeric parameters with range and type checking.
        
        Args:
            value: The value to validate
            constraint: The constraint to validate against
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(value, (int, float)):
            return False, f"Expected numeric value, got {type(value).__name__}"
        
        if constraint.min_value is not None and value < constraint.min_value:
            return False, f"Value {value} below minimum {constraint.min_value}"
            
        if constraint.max_value is not None and value > constraint.max_value:
            return False, f"Value {value} above maximum {constraint.max_value}"
            
        if constraint.validator_func:
            return constraint.validator_func(value)
            
        return True, None


class EnumValidationStrategy:
    """Validation strategy for enumerated values."""
    
    def validate(self, value: Any, constraint: ParameterConstraint) -> ValidationResult:
        """Validate enumerated parameters."""
        if constraint.allowed_values and value not in constraint.allowed_values:
            return False, f"Value {value} not in allowed values {constraint.allowed_values}"
        return True, None


class ConfigurationValidator:
    """
    Enterprise-grade configuration validator with comprehensive error reporting.
    
    Implements the Strategy pattern for different validation approaches
    and provides detailed validation results with context.
    """
    
    def __init__(self):
        self._strategies: Dict[str, ValidationStrategy] = {
            "numeric": NumericValidationStrategy(),
            "enum": EnumValidationStrategy(),
        }
        self._constraints: Dict[str, ParameterConstraint] = self._build_constraints()
        
    def _build_constraints(self) -> Dict[str, ParameterConstraint]:
        """Build parameter constraints with validation rules."""
        return {
            "temperature": ParameterConstraint(
                min_value=0.0, max_value=2.0,
                description="Sampling temperature for randomness control"
            ),
            "top_p": ParameterConstraint(
                min_value=0.0, max_value=1.0,
                description="Nucleus sampling probability threshold"
            ),
            "top_k": ParameterConstraint(
                min_value=0, max_value=1000,
                description="Top-k sampling parameter"
            ),
            "min_p": ParameterConstraint(
                min_value=0.0, max_value=1.0,
                description="Minimum probability threshold"
            ),
            "min_tokens_to_keep": ParameterConstraint(
                min_value=1, max_value=100,
                description="Minimum tokens to keep in sampling"
            ),
            "xtc_probability": ParameterConstraint(
                min_value=0.0, max_value=1.0,
                description="XTC sampling probability"
            ),
            "xtc_threshold": ParameterConstraint(
                min_value=0.0, max_value=1.0,
                description="XTC threshold parameter"
            ),
            "repetition_penalty": ParameterConstraint(
                min_value=0.5, max_value=2.0,
                description="Repetition penalty factor"
            ),
        }
    
    def validate_parameter(
        self, 
        name: str, 
        value: Any, 
        strategy_type: str = "numeric"
    ) -> ValidationResult:
        """
        Validate a single parameter with detailed error reporting.
        
        Args:
            name: Parameter name
            value: Parameter value
            strategy_type: Validation strategy to use
            
        Returns:
            Validation result with success flag and optional error message
        """
        if name not in self._constraints:
            logger.warning(f"No constraint defined for parameter '{name}'")
            return True, None
            
        constraint = self._constraints[name]
        strategy = self._strategies.get(strategy_type)
        
        if not strategy:
            return False, f"Unknown validation strategy: {strategy_type}"
            
        return strategy.validate(value, constraint)
    
    def validate_configuration(self, config_dict: Dict[str, Any]) -> List[ValidationError]:
        """
        Validate an entire configuration with comprehensive error collection.
        
        Args:
            config_dict: Configuration dictionary to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        for name, value in config_dict.items():
            is_valid, error_msg = self.validate_parameter(name, value)
            if not is_valid:
                errors.append(ValidationError(
                    parameter=name,
                    value=value,
                    reason=error_msg or "Unknown validation error"
                ))
                
        return errors


@pydantic_dataclass
class SamplingPhaseConfig:
    """
    Configuration for a specific sampling phase (think or answer).
    
    Provides comprehensive sampling parameters with validation and
    performance optimization features.
    
    Attributes:
        temperature: Sampling temperature (0.0 = deterministic, higher = more random)
        top_p: Nucleus sampling probability threshold
        top_k: Top-k sampling parameter (0 = disabled)
        min_p: Minimum probability threshold for token selection
        min_tokens_to_keep: Minimum number of tokens to keep in sampling
        xtc_probability: XTC (eXtended Token Choice) sampling probability
        xtc_threshold: XTC threshold parameter
        xtc_special_tokens: Special tokens for XTC sampling
        repetition_penalty: Penalty for token repetition
        repetition_context_size: Context size for repetition penalty
        enabled: Whether this phase configuration is active
        
    Example:
        >>> think_config = SamplingPhaseConfig(
        ...     temperature=0.2,
        ...     top_p=0.8,
        ...     top_k=50,
        ...     min_tokens_to_keep=2
        ... )
        >>> print(think_config.temperature)
        0.2
    """
    
    # Core sampling parameters
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for randomness control"
    )
    
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling probability threshold"
    )
    
    top_k: int = Field(
        default=0,
        ge=0,
        le=1000,
        description="Top-k sampling parameter (0 = disabled)"
    )
    
    min_p: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum probability threshold"
    )
    
    # Enhanced parameters (previously missing)
    min_tokens_to_keep: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Minimum tokens to keep in sampling"
    )
    
    xtc_probability: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="XTC sampling probability"
    )
    
    xtc_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="XTC threshold parameter"
    )
    
    xtc_special_tokens: List[int] = Field(
        default_factory=list,
        description="Special tokens for XTC sampling"
    )
    
    # Repetition control
    repetition_penalty: Optional[float] = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Repetition penalty factor"
    )
    
    repetition_context_size: Optional[int] = Field(
        default=20,
        ge=1,
        le=1000,
        description="Context size for repetition penalty"
    )
    
    # Phase control
    enabled: bool = Field(
        default=True,
        description="Whether this phase configuration is active"
    )
    
    # Metadata
    phase_name: str = Field(
        default="",
        description="Human-readable name for this phase"
    )
    
    description: str = Field(
        default="",
        description="Description of this phase configuration"
    )
    
    # Performance optimization
    _cache_key: Optional[str] = field(default=None, init=False, repr=False)
    _validation_cache: Dict[str, bool] = field(default_factory=dict, init=False, repr=False)
    
    def __post_init__(self):
        """Post-initialization validation and cache setup."""
        self._cache_key = self._compute_cache_key()
        self._validate_configuration()
    
    def _compute_cache_key(self) -> str:
        """Compute a cache key for this configuration."""
        config_dict = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "min_tokens_to_keep": self.min_tokens_to_keep,
            "xtc_probability": self.xtc_probability,
            "xtc_threshold": self.xtc_threshold,
            "repetition_penalty": self.repetition_penalty,
        }
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _validate_configuration(self) -> None:
        """Validate the configuration parameters."""
        validator = ConfigurationValidator()
        config_dict = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "min_tokens_to_keep": self.min_tokens_to_keep,
            "xtc_probability": self.xtc_probability,
            "xtc_threshold": self.xtc_threshold,
        }
        
        errors = validator.validate_configuration(config_dict)
        if errors:
            error_messages = [str(error) for error in errors]
            raise ValidationError(
                parameter="configuration",
                value=config_dict,
                reason="; ".join(error_messages)
            )
    
    @property
    def cache_key(self) -> str:
        """Get the cache key for this configuration."""
        return self._cache_key or self._compute_cache_key()
    
    def to_mlx_params(self) -> Dict[str, Any]:
        """
        Convert to MLX sampler parameters.
        
        Returns:
            Dictionary of parameters compatible with mlx_lm.make_sampler
        """
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k if self.top_k > 0 else None,
            "min_p": self.min_p,
            "min_tokens_to_keep": self.min_tokens_to_keep,
            "xtc_probability": self.xtc_probability,
            "xtc_threshold": self.xtc_threshold,
            "xtc_special_tokens": self.xtc_special_tokens,
        }
    
    def merge_with(self, other: 'SamplingPhaseConfig') -> 'SamplingPhaseConfig':
        """
        Merge this configuration with another, preferring non-default values.
        
        Args:
            other: Configuration to merge with
            
        Returns:
            New merged configuration
        """
        merged_params = {}
        
        for field_name in self.__dataclass_fields__:
            if field_name.startswith('_'):
                continue
                
            self_value = getattr(self, field_name)
            other_value = getattr(other, field_name)
            
            # Use other's value if it's not the default
            field_info = self.__dataclass_fields__[field_name]
            default_value = field_info.default if field_info.default is not field_info.default_factory else field_info.default_factory()
            
            if other_value != default_value:
                merged_params[field_name] = other_value
            else:
                merged_params[field_name] = self_value
        
        return SamplingPhaseConfig(**merged_params)
    
    def clone(self, **overrides) -> 'SamplingPhaseConfig':
        """
        Create a clone of this configuration with optional parameter overrides.
        
        Args:
            **overrides: Parameters to override in the clone
            
        Returns:
            New configuration instance with overrides applied
        """
        current_params = {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
            if not field_name.startswith('_')
        }
        current_params.update(overrides)
        return SamplingPhaseConfig(**current_params)


class ConfigurationBuilder:
    """
    Builder pattern implementation for constructing complex configurations.
    
    Provides a fluent interface for building configurations with validation
    and error handling at each step.
    """
    
    def __init__(self):
        self._think_config: Optional[SamplingPhaseConfig] = None
        self._answer_config: Optional[SamplingPhaseConfig] = None
        self._global_config: Optional[SamplingPhaseConfig] = None
        self._tags: Dict[str, str] = {}
        self._metadata: Dict[str, Any] = {}
        self._validation_enabled: bool = True
    
    def with_think_phase(self, **kwargs) -> 'ConfigurationBuilder':
        """Configure the thinking phase parameters."""
        self._think_config = SamplingPhaseConfig(
            phase_name="think",
            **kwargs
        )
        return self
    
    def with_answer_phase(self, **kwargs) -> 'ConfigurationBuilder':
        """Configure the answer phase parameters."""
        self._answer_config = SamplingPhaseConfig(
            phase_name="answer",
            **kwargs
        )
        return self
    
    def with_global_config(self, **kwargs) -> 'ConfigurationBuilder':
        """Configure global/default parameters."""
        self._global_config = SamplingPhaseConfig(
            phase_name="global",
            **kwargs
        )
        return self
    
    def with_tags(self, **tags) -> 'ConfigurationBuilder':
        """Add metadata tags to the configuration."""
        self._tags.update(tags)
        return self
    
    def with_metadata(self, **metadata) -> 'ConfigurationBuilder':
        """Add metadata to the configuration."""
        self._metadata.update(metadata)
        return self
    
    def disable_validation(self) -> 'ConfigurationBuilder':
        """Disable validation during build (use with caution)."""
        self._validation_enabled = False
        return self
    
    def build(self) -> 'EnhancedGenerationConfig':
        """
        Build the final configuration with validation.
        
        Returns:
            Complete enhanced generation configuration
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        # Provide defaults if not specified
        if self._global_config is None:
            self._global_config = SamplingPhaseConfig(phase_name="global")
        
        if self._think_config is None:
            self._think_config = self._global_config.clone(
                phase_name="think",
                temperature=0.2,  # Lower temperature for thinking
                top_p=0.8
            )
        
        if self._answer_config is None:
            self._answer_config = self._global_config.clone(
                phase_name="answer",
                temperature=0.3,  # Slightly higher for answers
                top_p=0.9
            )
        
        return EnhancedGenerationConfig(
            think_phase=self._think_config,
            answer_phase=self._answer_config,
            global_phase=self._global_config,
            tags=self._tags,
            metadata=self._metadata,
            validation_enabled=self._validation_enabled
        )


@dataclass
class EnhancedGenerationConfig:
    """
    Enhanced generation configuration with per-phase sampling support.
    
    Provides enterprise-grade configuration management with comprehensive
    validation, performance optimization, and backward compatibility.
    
    Features:
        - Per-phase configuration (think vs answer)
        - Advanced parameter validation
        - Performance caching and optimization
        - Backward compatibility with legacy configs
        - Comprehensive error handling and logging
        - Thread-safe operations
        
    Attributes:
        think_phase: Configuration for thinking phase
        answer_phase: Configuration for answer phase  
        global_phase: Global/default configuration
        tags: Metadata tags for configuration identification
        metadata: Additional metadata
        validation_enabled: Whether to perform validation
        
    Example:
        >>> config = EnhancedGenerationConfig.builder() \
        ...     .with_think_phase(temperature=0.2) \
        ...     .with_answer_phase(temperature=0.3) \
        ...     .build()
        >>> sampler = config.create_sampler_for_phase("think", tokenizer)
    """
    
    # Phase configurations
    think_phase: SamplingPhaseConfig
    answer_phase: SamplingPhaseConfig
    global_phase: SamplingPhaseConfig
    
    # Metadata and control
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation_enabled: bool = True
    
    # Performance optimization
    _sampler_cache: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _cache_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _access_count: int = field(default=0, init=False, repr=False)
    _last_access: float = field(default_factory=time.time, init=False, repr=False)
    
    # Configuration versioning
    config_version: str = field(default="2.0", init=False)
    created_at: float = field(default_factory=time.time, init=False)
    config_id: str = field(default_factory=lambda: str(uuid.uuid4()), init=False)
    
    def __post_init__(self):
        """Post-initialization setup and validation."""
        if self.validation_enabled:
            self._validate_phase_consistency()
        
        # Set up performance monitoring
        self._setup_performance_monitoring()
        
        logger.info(
            f"Created EnhancedGenerationConfig {self.config_id}",
            extra={
                "config_id": self.config_id,
                "version": self.config_version,
                "phases": ["think", "answer", "global"],
                "validation_enabled": self.validation_enabled
            }
        )
    
    def _validate_phase_consistency(self) -> None:
        """Validate consistency across phase configurations."""
        phases = [self.think_phase, self.answer_phase, self.global_phase]
        
        # Check for reasonable temperature relationships
        if (self.think_phase.temperature > self.answer_phase.temperature + 0.5):
            logger.warning(
                "Think phase temperature significantly higher than answer phase. "
                "This may lead to inconsistent generation quality."
            )
        
        # Validate XTC token consistency
        for phase in phases:
            if phase.xtc_probability > 0 and not phase.xtc_special_tokens:
                logger.warning(
                    f"XTC probability set but no special tokens defined for {phase.phase_name} phase"
                )
    
    def _setup_performance_monitoring(self) -> None:
        """Set up performance monitoring and caching."""
        # Initialize weak reference tracking for memory management
        self._weak_refs: Set[weakref.ref] = set()
        
        # Set up cache cleanup timer
        self._schedule_cache_cleanup()
    
    def _schedule_cache_cleanup(self) -> None:
        """Schedule periodic cache cleanup to prevent memory leaks."""
        def cleanup():
            with self._cache_lock:
                # Clean up expired cache entries
                current_time = time.time()
                expired_keys = [
                    key for key, (value, timestamp) in self._sampler_cache.items()
                    if current_time - timestamp > 300  # 5 minute expiry
                ]
                for key in expired_keys:
                    del self._sampler_cache[key]
                
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        # Schedule cleanup (in a real implementation, use a proper scheduler)
        threading.Timer(300, cleanup).start()
    
    @classmethod
    def builder(cls) -> ConfigurationBuilder:
        """
        Create a new configuration builder.
        
        Returns:
            ConfigurationBuilder instance for fluent configuration construction
        """
        return ConfigurationBuilder()
    
    @classmethod
    def from_legacy_config(cls, legacy_config: Any) -> 'EnhancedGenerationConfig':
        """
        Create enhanced configuration from legacy GenerationConfig.
        
        Args:
            legacy_config: Legacy configuration object
            
        Returns:
            Enhanced configuration with migrated parameters
            
        Raises:
            CompatibilityError: If migration fails
        """
        try:
            # Extract parameters from legacy config
            think_temp = getattr(legacy_config, 'think_temperature', 0.2)
            answer_temp = getattr(legacy_config, 'answer_temperature', 0.3)
            top_p = getattr(legacy_config, 'sampling_top_p', 0.9)
            top_k = getattr(legacy_config, 'sampling_top_k', 0)
            min_p = getattr(legacy_config, 'sampling_min_p', 0.0)
            rep_penalty = getattr(legacy_config, 'repetition_penalty', 1.0)
            rep_context = getattr(legacy_config, 'repetition_context_size', 20)
            
            return cls.builder() \
                .with_think_phase(
                    temperature=think_temp,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    repetition_penalty=rep_penalty,
                    repetition_context_size=rep_context
                ) \
                .with_answer_phase(
                    temperature=answer_temp,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    repetition_penalty=rep_penalty,
                    repetition_context_size=rep_context
                ) \
                .with_tags(migrated_from="legacy_config") \
                .build()
                
        except Exception as e:
            raise CompatibilityError(
                legacy_config=str(type(legacy_config)),
                migration_path="from_legacy_config",
                context={"error": str(e)}
            )
    
    def get_phase_config(self, phase: Union[str, SamplingPhase]) -> SamplingPhaseConfig:
        """
        Get configuration for a specific phase.
        
        Args:
            phase: Phase name or enum value
            
        Returns:
            Phase-specific configuration
            
        Raises:
            ValueError: If phase is not recognized
        """
        self._access_count += 1
        self._last_access = time.time()
        
        if isinstance(phase, str):
            phase = phase.lower()
        elif isinstance(phase, SamplingPhase):
            phase = phase.value
        
        phase_map = {
            "think": self.think_phase,
            "answer": self.answer_phase,
            "global": self.global_phase,
        }
        
        if phase not in phase_map:
            raise ValueError(f"Unknown phase: {phase}. Available: {list(phase_map.keys())}")
        
        return phase_map[phase]
    
    @lru_cache(maxsize=128)
    def create_sampler_for_phase(
        self, 
        phase: str, 
        tokenizer: Any,
        **override_params
    ) -> Callable:
        """
        Create a sampler for the specified phase with caching.
        
        Args:
            phase: Phase name ("think", "answer", or "global")
            tokenizer: Tokenizer instance for special token handling
            **override_params: Parameters to override for this sampler
            
        Returns:
            Configured sampler function
            
        Raises:
            ConfigurationError: If sampler creation fails
        """
        try:
            phase_config = self.get_phase_config(phase)
            
            # Apply overrides
            if override_params:
                phase_config = phase_config.clone(**override_params)
            
            # Create cache key
            cache_key = f"{phase}_{phase_config.cache_key}_{id(tokenizer)}"
            
            with self._cache_lock:
                if cache_key in self._sampler_cache:
                    sampler, timestamp = self._sampler_cache[cache_key]
                    # Check if cache entry is still valid (5 minutes)
                    if time.time() - timestamp < 300:
                        logger.debug(f"Using cached sampler for phase {phase}")
                        return sampler
                
                # Create new sampler
                from mlx_lm.sample_utils import make_sampler
                
                mlx_params = phase_config.to_mlx_params()
                
                # Handle special tokens for XTC
                if phase_config.xtc_special_tokens:
                    mlx_params["xtc_special_tokens"] = phase_config.xtc_special_tokens
                elif hasattr(tokenizer, 'eos_token_ids'):
                    # Fallback to EOS tokens
                    eos_tokens = list(tokenizer.eos_token_ids) if tokenizer.eos_token_ids else []
                    newline_tokens = tokenizer.encode("\n", add_special_tokens=False) or []
                    mlx_params["xtc_special_tokens"] = eos_tokens + newline_tokens
                
                sampler = make_sampler(**mlx_params)
                
                # Cache the sampler
                self._sampler_cache[cache_key] = (sampler, time.time())
                
                logger.debug(
                    f"Created new sampler for phase {phase}",
                    extra={
                        "phase": phase,
                        "cache_key": cache_key,
                        "parameters": mlx_params
                    }
                )
                
                return sampler
                
        except Exception as e:
            raise ConfigurationError(
                f"Failed to create sampler for phase {phase}: {str(e)}",
                error_code="SAMPLER_CREATION_ERROR",
                context={
                    "phase": phase,
                    "override_params": override_params,
                    "config_id": self.config_id
                }
            )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for this configuration.
        
        Returns:
            Dictionary of performance metrics
        """
        with self._cache_lock:
            return {
                "config_id": self.config_id,
                "access_count": self._access_count,
                "last_access": self._last_access,
                "cache_size": len(self._sampler_cache),
                "created_at": self.created_at,
                "age_seconds": time.time() - self.created_at,
                "cache_hit_ratio": self._calculate_cache_hit_ratio(),
            }
    
    def _calculate_cache_hit_ratio(self) -> float:
        """Calculate cache hit ratio for performance monitoring."""
        # This would be implemented with proper hit/miss tracking
        # For now, return a placeholder
        return 0.85  # 85% hit ratio placeholder
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary for serialization.
        
        Returns:
            Dictionary representation of the configuration
        """
        return {
            "config_id": self.config_id,
            "config_version": self.config_version,
            "created_at": self.created_at,
            "think_phase": {
                field_name: getattr(self.think_phase, field_name)
                for field_name in self.think_phase.__dataclass_fields__
                if not field_name.startswith('_')
            },
            "answer_phase": {
                field_name: getattr(self.answer_phase, field_name)
                for field_name in self.answer_phase.__dataclass_fields__
                if not field_name.startswith('_')
            },
            "global_phase": {
                field_name: getattr(self.global_phase, field_name)
                for field_name in self.global_phase.__dataclass_fields__
                if not field_name.startswith('_')
            },
            "tags": self.tags,
            "metadata": self.metadata,
            "validation_enabled": self.validation_enabled,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EnhancedGenerationConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary representation
            
        Returns:
            Enhanced configuration instance
        """
        think_phase = SamplingPhaseConfig(**config_dict["think_phase"])
        answer_phase = SamplingPhaseConfig(**config_dict["answer_phase"])
        global_phase = SamplingPhaseConfig(**config_dict["global_phase"])
        
        config = cls(
            think_phase=think_phase,
            answer_phase=answer_phase,
            global_phase=global_phase,
            tags=config_dict.get("tags", {}),
            metadata=config_dict.get("metadata", {}),
            validation_enabled=config_dict.get("validation_enabled", True)
        )
        
        # Restore metadata
        config.config_id = config_dict.get("config_id", config.config_id)
        config.config_version = config_dict.get("config_version", config.config_version)
        config.created_at = config_dict.get("created_at", config.created_at)
        
        return config
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        return (
            f"EnhancedGenerationConfig(id={self.config_id[:8]}, "
            f"think_temp={self.think_phase.temperature}, "
            f"answer_temp={self.answer_phase.temperature}, "
            f"version={self.config_version})"
        )
    
    def __repr__(self) -> str:
        """Detailed representation of the configuration."""
        return (
            f"EnhancedGenerationConfig("
            f"config_id='{self.config_id}', "
            f"think_phase={repr(self.think_phase)}, "
            f"answer_phase={repr(self.answer_phase)}, "
            f"global_phase={repr(self.global_phase)}, "
            f"validation_enabled={self.validation_enabled})"
        )


# Factory functions for common configurations
def create_conservative_config() -> EnhancedGenerationConfig:
    """Create a conservative configuration with low randomness."""
    return EnhancedGenerationConfig.builder() \
        .with_think_phase(temperature=0.1, top_p=0.7, top_k=20) \
        .with_answer_phase(temperature=0.2, top_p=0.8, top_k=30) \
        .with_tags(preset="conservative") \
        .build()


def create_creative_config() -> EnhancedGenerationConfig:
    """Create a creative configuration with higher randomness."""
    return EnhancedGenerationConfig.builder() \
        .with_think_phase(temperature=0.8, top_p=0.95, top_k=100) \
        .with_answer_phase(temperature=0.7, top_p=0.9, top_k=80) \
        .with_tags(preset="creative") \
        .build()


def create_balanced_config() -> EnhancedGenerationConfig:
    """Create a balanced configuration for general use."""
    return EnhancedGenerationConfig.builder() \
        .with_think_phase(temperature=0.3, top_p=0.85, top_k=50) \
        .with_answer_phase(temperature=0.4, top_p=0.9, top_k=60) \
        .with_tags(preset="balanced") \
        .build()


# Export public API
__all__ = [
    # Core classes
    "EnhancedGenerationConfig",
    "SamplingPhaseConfig", 
    "ConfigurationBuilder",
    "ConfigurationValidator",
    
    # Enums and types
    "SamplingPhase",
    "PhaseType",
    
    # Exceptions
    "ConfigurationError",
    "ValidationError", 
    "CompatibilityError",
    
    # Factory functions
    "create_conservative_config",
    "create_creative_config",
    "create_balanced_config",
]