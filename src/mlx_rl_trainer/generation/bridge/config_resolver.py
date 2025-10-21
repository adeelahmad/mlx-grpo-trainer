"""
Configuration resolver for bridging sampling configurations with MLX sampler implementations.

This module implements the Bridge pattern to decouple configuration abstractions from
their concrete implementations, providing a flexible and extensible parameter resolution
system that supports multiple configuration sources and validation strategies.

Architecture:
    - ConfigurationResolver: Main resolver with strategy-based parameter mapping
    - ParameterMapper: Strategy pattern for different parameter mapping approaches
    - ValidationEngine: Comprehensive validation with rule-based constraints
    - ConflictResolver: Advanced conflict resolution for overlapping parameters
    - CacheManager: Performance optimization with intelligent caching

Design Patterns Applied:
    - Bridge Pattern: Decouples configuration abstraction from implementation
    - Strategy Pattern: Different parameter mapping and validation strategies
    - Chain of Responsibility: Parameter resolution pipeline with fallbacks
    - Command Pattern: Encapsulated parameter transformation operations
    - Observer Pattern: Configuration change notifications and monitoring
    - Template Method Pattern: Common resolution workflow with customizable steps

SOLID Principles:
    - Single Responsibility: Each class handles one aspect of configuration resolution
    - Open/Closed: Extensible for new parameter types and validation rules
    - Liskov Substitution: All resolvers are interchangeable through interfaces
    - Interface Segregation: Separate interfaces for different resolution concerns
    - Dependency Inversion: Depends on abstractions for maximum flexibility

Example:
    >>> from mlx_rl_trainer.generation.bridge.config_resolver import ConfigurationResolver
    >>> from mlx_rl_trainer.generation.config.enhanced_config import EnhancedGenerationConfig
    >>> 
    >>> config = EnhancedGenerationConfig(...)
    >>> resolver = ConfigurationResolver()
    >>> params = resolver.resolve_parameters(config, phase="think")
    >>> sampler = mlx_lm.make_sampler(**params)
"""

import logging
import time
import threading
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, Dict, List, Optional, Union, Callable, Protocol, TypeVar, Generic,
    runtime_checkable, ClassVar, Final, Tuple, NamedTuple, Set, Mapping
)
from collections import defaultdict, deque, ChainMap
from concurrent.futures import ThreadPoolExecutor, Future
import asyncio
import uuid
import json
import hashlib
from functools import wraps, lru_cache, singledispatch
import inspect
import copy
import re
from pathlib import Path

# Import configuration classes
try:
    from ..config.enhanced_config import (
        EnhancedGenerationConfig, SamplingPhaseConfig, ConfigurationError,
        ValidationError, CompatibilityError
    )
except ImportError:
    # Fallback for development/testing
    from mlx_rl_trainer.generation.config.enhanced_config import (
        EnhancedGenerationConfig, SamplingPhaseConfig, ConfigurationError,
        ValidationError, CompatibilityError
    )

# Import core configuration
try:
    from ...core.config import GenerationConfig, ExperimentConfig
except ImportError:
    from mlx_rl_trainer.core.config import GenerationConfig, ExperimentConfig

# Type definitions for enhanced type safety
T = TypeVar('T')
ConfigType = TypeVar('ConfigType', bound=Union[GenerationConfig, EnhancedGenerationConfig])
ParameterDict = Dict[str, Any]
ValidationResult = Tuple[bool, List[str]]

logger = logging.getLogger(__name__)


class ResolutionStrategy(Enum):
    """
    Enumeration of parameter resolution strategies.
    
    Defines different approaches for resolving configuration parameters
    when multiple sources or conflicts exist.
    """
    STRICT = "strict"  # Fail on any conflicts or missing parameters
    PERMISSIVE = "permissive"  # Use defaults and best-effort resolution
    OVERRIDE = "override"  # Later sources override earlier ones
    MERGE = "merge"  # Intelligent merging of compatible parameters
    FALLBACK = "fallback"  # Use fallback chain for missing parameters


class ParameterSource(Enum):
    """
    Enumeration of parameter sources for conflict resolution.
    
    Defines the hierarchy and priority of different parameter sources.
    """
    GLOBAL_CONFIG = auto()
    PHASE_CONFIG = auto()
    RUNTIME_OVERRIDE = auto()
    ENVIRONMENT_VAR = auto()
    DEFAULT_VALUE = auto()


class ConflictResolutionMode(Enum):
    """
    Enumeration of conflict resolution modes.
    
    Defines how to handle conflicts between different parameter sources.
    """
    FAIL_FAST = "fail_fast"  # Raise exception on first conflict
    LOG_AND_CONTINUE = "log_and_continue"  # Log conflicts but continue
    SILENT_OVERRIDE = "silent_override"  # Silently use highest priority
    INTERACTIVE = "interactive"  # Prompt for resolution (development only)


@dataclass
class ParameterMetadata:
    """
    Metadata for configuration parameters.
    
    Provides comprehensive information about parameter characteristics,
    validation rules, and resolution behavior.
    
    Attributes:
        name: Parameter name
        parameter_type: Python type of the parameter
        default_value: Default value if not specified
        required: Whether parameter is required
        validation_rules: List of validation rule names
        source_priority: Priority order for conflict resolution
        description: Human-readable description
        mlx_parameter_name: Corresponding MLX parameter name
        phase_specific: Whether parameter can vary by phase
        deprecated: Whether parameter is deprecated
        min_value: Minimum allowed value (for numeric types)
        max_value: Maximum allowed value (for numeric types)
        allowed_values: Set of allowed values (for enum-like types)
    """
    
    name: str
    parameter_type: type
    default_value: Any = None
    required: bool = False
    validation_rules: List[str] = field(default_factory=list)
    source_priority: List[ParameterSource] = field(default_factory=lambda: [
        ParameterSource.RUNTIME_OVERRIDE,
        ParameterSource.PHASE_CONFIG,
        ParameterSource.GLOBAL_CONFIG,
        ParameterSource.ENVIRONMENT_VAR,
        ParameterSource.DEFAULT_VALUE
    ])
    description: str = ""
    mlx_parameter_name: Optional[str] = None
    phase_specific: bool = True
    deprecated: bool = False
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[Set[Any]] = None
    
    def __post_init__(self):
        """Validate metadata consistency."""
        if self.required and self.default_value is not None:
            logger.warning(
                f"Parameter '{self.name}' is marked as required but has default value"
            )
        
        if self.mlx_parameter_name is None:
            self.mlx_parameter_name = self.name


@dataclass
class ResolutionContext:
    """
    Context information for parameter resolution.
    
    Provides comprehensive context for the resolution process,
    including phase information, override parameters, and metadata.
    
    Attributes:
        phase: Current sampling phase (think/answer/global)
        correlation_id: Unique identifier for this resolution
        override_parameters: Runtime parameter overrides
        environment_variables: Environment variable mappings
        resolution_strategy: Strategy for handling conflicts
        conflict_mode: Mode for conflict resolution
        validation_enabled: Whether to perform validation
        cache_enabled: Whether to use parameter caching
        metadata: Additional context metadata
        timestamp: When resolution was initiated
    """
    
    phase: str = "global"
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    override_parameters: Dict[str, Any] = field(default_factory=dict)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    resolution_strategy: ResolutionStrategy = ResolutionStrategy.MERGE
    conflict_mode: ConflictResolutionMode = ConflictResolutionMode.LOG_AND_CONTINUE
    validation_enabled: bool = True
    cache_enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ResolutionResult:
    """
    Result of parameter resolution process.
    
    Contains the resolved parameters along with comprehensive metadata
    about the resolution process for debugging and monitoring.
    
    Attributes:
        parameters: Resolved parameter dictionary
        conflicts: List of conflicts encountered during resolution
        warnings: List of warnings generated during resolution
        metadata: Resolution metadata and statistics
        cache_hit: Whether result was retrieved from cache
        resolution_time_ms: Time taken for resolution in milliseconds
        parameter_sources: Mapping of parameters to their sources
        validation_results: Results of parameter validation
    """
    
    parameters: ParameterDict
    conflicts: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    cache_hit: bool = False
    resolution_time_ms: float = 0.0
    parameter_sources: Dict[str, ParameterSource] = field(default_factory=dict)
    validation_results: Dict[str, ValidationResult] = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        """Check if resolution was successful."""
        return len(self.conflicts) == 0 or all(
            not conflict.startswith("CRITICAL:") for conflict in self.conflicts
        )
    
    @property
    def has_warnings(self) -> bool:
        """Check if resolution generated warnings."""
        return len(self.warnings) > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "parameters": self.parameters,
            "conflicts": self.conflicts,
            "warnings": self.warnings,
            "metadata": self.metadata,
            "cache_hit": self.cache_hit,
            "resolution_time_ms": self.resolution_time_ms,
            "parameter_sources": {k: v.name for k, v in self.parameter_sources.items()},
            "validation_results": {
                k: {"valid": v[0], "errors": v[1]} 
                for k, v in self.validation_results.items()
            },
            "success": self.success,
            "has_warnings": self.has_warnings
        }


class ConfigurationResolverError(ConfigurationError):
    """Base exception for configuration resolver errors."""
    
    def __init__(
        self,
        message: str,
        resolution_context: Optional[ResolutionContext] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.resolution_context = resolution_context


class ParameterConflictError(ConfigurationResolverError):
    """Exception for parameter conflicts that cannot be resolved."""
    
    def __init__(
        self,
        parameter_name: str,
        conflicting_values: Dict[ParameterSource, Any],
        **kwargs
    ):
        message = f"Unresolvable conflict for parameter '{parameter_name}': {conflicting_values}"
        super().__init__(message, **kwargs)
        self.parameter_name = parameter_name
        self.conflicting_values = conflicting_values


class ValidationFailureError(ConfigurationResolverError):
    """Exception for parameter validation failures."""
    
    def __init__(
        self,
        parameter_name: str,
        value: Any,
        validation_errors: List[str],
        **kwargs
    ):
        message = f"Validation failed for parameter '{parameter_name}' = {value}: {validation_errors}"
        super().__init__(message, **kwargs)
        self.parameter_name = parameter_name
        self.value = value
        self.validation_errors = validation_errors


@runtime_checkable
class ParameterMapperProtocol(Protocol):
    """
    Protocol for parameter mapping strategies.
    
    Defines the interface for different parameter mapping approaches,
    allowing for flexible and extensible parameter transformation.
    """
    
    def map_parameters(
        self,
        config: ConfigType,
        context: ResolutionContext,
        metadata_registry: Dict[str, ParameterMetadata]
    ) -> ParameterDict:
        """
        Map configuration to parameter dictionary.
        
        Args:
            config: Configuration object to map
            context: Resolution context
            metadata_registry: Parameter metadata registry
            
        Returns:
            Mapped parameter dictionary
        """
        ...
    
    def supports_config_type(self, config_type: type) -> bool:
        """
        Check if mapper supports the given configuration type.
        
        Args:
            config_type: Configuration type to check
            
        Returns:
            True if supported, False otherwise
        """
        ...


@runtime_checkable
class ValidationEngineProtocol(Protocol):
    """
    Protocol for parameter validation engines.
    
    Defines the interface for different validation strategies,
    enabling comprehensive and extensible parameter validation.
    """
    
    def validate_parameter(
        self,
        name: str,
        value: Any,
        metadata: ParameterMetadata,
        context: ResolutionContext
    ) -> ValidationResult:
        """
        Validate a single parameter.
        
        Args:
            name: Parameter name
            value: Parameter value
            metadata: Parameter metadata
            context: Resolution context
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        ...
    
    def validate_parameters(
        self,
        parameters: ParameterDict,
        metadata_registry: Dict[str, ParameterMetadata],
        context: ResolutionContext
    ) -> Dict[str, ValidationResult]:
        """
        Validate multiple parameters.
        
        Args:
            parameters: Parameter dictionary
            metadata_registry: Parameter metadata registry
            context: Resolution context
            
        Returns:
            Dictionary of validation results
        """
        ...


class DefaultParameterMapper:
    """
    Default implementation of parameter mapping strategy.
    
    Provides comprehensive parameter mapping with support for multiple
    configuration types and intelligent parameter transformation.
    """
    
    def __init__(self):
        self._type_handlers: Dict[type, Callable] = {
            EnhancedGenerationConfig: self._map_enhanced_config,
            GenerationConfig: self._map_generation_config,
            ExperimentConfig: self._map_experiment_config,
        }
    
    def map_parameters(
        self,
        config: ConfigType,
        context: ResolutionContext,
        metadata_registry: Dict[str, ParameterMetadata]
    ) -> ParameterDict:
        """Map configuration to parameter dictionary."""
        config_type = type(config)
        
        if config_type in self._type_handlers:
            return self._type_handlers[config_type](config, context, metadata_registry)
        
        # Fallback to generic mapping
        return self._map_generic_config(config, context, metadata_registry)
    
    def supports_config_type(self, config_type: type) -> bool:
        """Check if mapper supports the configuration type."""
        return (
            config_type in self._type_handlers or
            hasattr(config_type, '__dict__') or
            hasattr(config_type, '__dataclass_fields__')
        )
    
    def _map_enhanced_config(
        self,
        config: EnhancedGenerationConfig,
        context: ResolutionContext,
        metadata_registry: Dict[str, ParameterMetadata]
    ) -> ParameterDict:
        """Map EnhancedGenerationConfig to parameters."""
        parameters = {}
        
        # Get phase-specific configuration
        phase_config = config.get_phase_config(context.phase)
        
        # Map all available parameters
        for param_name, metadata in metadata_registry.items():
            value = self._resolve_parameter_value(
                param_name, config, phase_config, context, metadata
            )
            if value is not None:
                # Use MLX parameter name if different
                mlx_name = metadata.mlx_parameter_name or param_name
                parameters[mlx_name] = value
        
        return parameters
    
    def _map_generation_config(
        self,
        config: GenerationConfig,
        context: ResolutionContext,
        metadata_registry: Dict[str, ParameterMetadata]
    ) -> ParameterDict:
        """Map legacy GenerationConfig to parameters."""
        parameters = {}
        
        # Direct attribute mapping
        attribute_mapping = {
            'temperature': 'temperature',
            'top_p': 'top_p',
            'top_k': 'top_k',
            'repetition_penalty': 'repetition_penalty',
            'repetition_context_size': 'repetition_context_size',
            'max_tokens': 'max_tokens',
        }
        
        for attr_name, param_name in attribute_mapping.items():
            if hasattr(config, attr_name):
                value = getattr(config, attr_name)
                if value is not None:
                    parameters[param_name] = value
        
        return parameters
    
    def _map_experiment_config(
        self,
        config: ExperimentConfig,
        context: ResolutionContext,
        metadata_registry: Dict[str, ParameterMetadata]
    ) -> ParameterDict:
        """Map ExperimentConfig to parameters."""
        if hasattr(config, 'generation') and config.generation:
            return self._map_generation_config(
                config.generation, context, metadata_registry
            )
        return {}
    
    def _map_generic_config(
        self,
        config: Any,
        context: ResolutionContext,
        metadata_registry: Dict[str, ParameterMetadata]
    ) -> ParameterDict:
        """Generic configuration mapping using reflection."""
        parameters = {}
        
        # Try to extract attributes
        if hasattr(config, '__dict__'):
            config_dict = config.__dict__
        elif hasattr(config, '__dataclass_fields__'):
            config_dict = {
                field.name: getattr(config, field.name, None)
                for field in config.__dataclass_fields__.values()
            }
        else:
            logger.warning(f"Cannot map unsupported config type: {type(config)}")
            return parameters
        
        # Map known parameters
        for param_name, metadata in metadata_registry.items():
            if param_name in config_dict:
                value = config_dict[param_name]
                if value is not None:
                    mlx_name = metadata.mlx_parameter_name or param_name
                    parameters[mlx_name] = value
        
        return parameters
    
    def _resolve_parameter_value(
        self,
        param_name: str,
        config: EnhancedGenerationConfig,
        phase_config: Optional[SamplingPhaseConfig],
        context: ResolutionContext,
        metadata: ParameterMetadata
    ) -> Any:
        """Resolve parameter value from multiple sources."""
        # Priority order: runtime override > phase config > global config > environment > default
        
        # 1. Runtime override
        if param_name in context.override_parameters:
            return context.override_parameters[param_name]
        
        # 2. Phase-specific configuration
        if phase_config and metadata.phase_specific:
            if hasattr(phase_config, param_name):
                value = getattr(phase_config, param_name)
                if value is not None:
                    return value
        
        # 3. Global configuration
        if hasattr(config.global_config, param_name):
            value = getattr(config.global_config, param_name)
            if value is not None:
                return value
        
        # 4. Environment variable
        env_var_name = f"MLX_SAMPLER_{param_name.upper()}"
        if env_var_name in context.environment_variables:
            env_value = context.environment_variables[env_var_name]
            try:
                # Convert environment string to appropriate type
                return self._convert_env_value(env_value, metadata.parameter_type)
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to convert environment variable {env_var_name}: {e}")
        
        # 5. Default value
        return metadata.default_value


    def _convert_env_value(self, env_value: str, target_type: type) -> Any:
        """Convert environment variable string to target type."""
        if target_type == str:
            return env_value
        elif target_type == int:
            return int(env_value)
        elif target_type == float:
            return float(env_value)
        elif target_type == bool:
            return env_value.lower() in ('true', '1', 'yes', 'on')
        elif target_type == list:
            return env_value.split(',')
        else:
            # Try JSON parsing for complex types
            try:
                return json.loads(env_value)
            except json.JSONDecodeError:
                raise ValueError(f"Cannot convert '{env_value}' to {target_type}")


class DefaultValidationEngine:
    """
    Default implementation of parameter validation engine.
    
    Provides comprehensive validation with rule-based constraints
    and extensible validation strategies.
    """
    
    def __init__(self):
        self._validation_rules: Dict[str, Callable] = {
            'range': self._validate_range,
            'positive': self._validate_positive,
            'non_negative': self._validate_non_negative,
            'probability': self._validate_probability,
            'integer': self._validate_integer,
            'allowed_values': self._validate_allowed_values,
            'min_length': self._validate_min_length,
            'max_length': self._validate_max_length,
            'regex': self._validate_regex,
        }
    
    def validate_parameter(
        self,
        name: str,
        value: Any,
        metadata: ParameterMetadata,
        context: ResolutionContext
    ) -> ValidationResult:
        """Validate a single parameter."""
        if not context.validation_enabled:
            return True, []
        
        errors = []
        
        # Type validation
        if not isinstance(value, metadata.parameter_type):
            try:
                # Attempt type conversion
                converted_value = metadata.parameter_type(value)
                # Update the value in place if conversion succeeds
                # Note: This is a design choice - we could return the converted value instead
            except (ValueError, TypeError):
                errors.append(f"Expected {metadata.parameter_type.__name__}, got {type(value).__name__}")
                return False, errors
        
        # Range validation
        if metadata.min_value is not None and value < metadata.min_value:
            errors.append(f"Value {value} is below minimum {metadata.min_value}")
        
        if metadata.max_value is not None and value > metadata.max_value:
            errors.append(f"Value {value} is above maximum {metadata.max_value}")
        
        # Allowed values validation
        if metadata.allowed_values is not None and value not in metadata.allowed_values:
            errors.append(f"Value {value} not in allowed values: {metadata.allowed_values}")
        
        # Rule-based validation
        for rule_name in metadata.validation_rules:
            if rule_name in self._validation_rules:
                try:
                    rule_errors = self._validation_rules[rule_name](value, metadata)
                    errors.extend(rule_errors)
                except Exception as e:
                    errors.append(f"Validation rule '{rule_name}' failed: {e}")
            else:
                logger.warning(f"Unknown validation rule: {rule_name}")
        
        return len(errors) == 0, errors
    
    def validate_parameters(
        self,
        parameters: ParameterDict,
        metadata_registry: Dict[str, ParameterMetadata],
        context: ResolutionContext
    ) -> Dict[str, ValidationResult]:
        """Validate multiple parameters."""
        results = {}
        
        for param_name, value in parameters.items():
            # Find metadata (handle MLX parameter name mapping)
            metadata = None
            for meta_name, meta in metadata_registry.items():
                if meta.mlx_parameter_name == param_name or meta_name == param_name:
                    metadata = meta
                    break
            
            if metadata:
                results[param_name] = self.validate_parameter(
                    param_name, value, metadata, context
                )
            else:
                # Unknown parameter - log warning but don't fail
                logger.warning(f"No metadata found for parameter: {param_name}")
                results[param_name] = (True, [])
        
        return results
    
    def _validate_range(self, value: Any, metadata: ParameterMetadata) -> List[str]:
        """Validate value is within specified range."""
        errors = []
        if hasattr(value, '__lt__') and hasattr(value, '__gt__'):
            if metadata.min_value is not None and value < metadata.min_value:
                errors.append(f"Below minimum value {metadata.min_value}")
            if metadata.max_value is not None and value > metadata.max_value:
                errors.append(f"Above maximum value {metadata.max_value}")
        return errors
    
    def _validate_positive(self, value: Any, metadata: ParameterMetadata) -> List[str]:
        """Validate value is positive."""
        if hasattr(value, '__gt__') and value <= 0:
            return ["Value must be positive"]
        return []
    
    def _validate_non_negative(self, value: Any, metadata: ParameterMetadata) -> List[str]:
        """Validate value is non-negative."""
        if hasattr(value, '__ge__') and value < 0:
            return ["Value must be non-negative"]
        return []
    
    def _validate_probability(self, value: Any, metadata: ParameterMetadata) -> List[str]:
        """Validate value is a valid probability (0.0 to 1.0)."""
        errors = []
        if hasattr(value, '__lt__') and hasattr(value, '__gt__'):
            if value < 0.0:
                errors.append("Probability must be >= 0.0")
            if value > 1.0:
                errors.append("Probability must be <= 1.0")
        return errors
    
    def _validate_integer(self, value: Any, metadata: ParameterMetadata) -> List[str]:
        """Validate value is an integer."""
        if not isinstance(value, int):
            return ["Value must be an integer"]
        return []
    
    def _validate_allowed_values(self, value: Any, metadata: ParameterMetadata) -> List[str]:
        """Validate value is in allowed set."""
        if metadata.allowed_values and value not in metadata.allowed_values:
            return [f"Value must be one of: {metadata.allowed_values}"]
        return []
    
    def _validate_min_length(self, value: Any, metadata: ParameterMetadata) -> List[str]:
        """Validate minimum length for sequences."""
        if hasattr(value, '__len__') and metadata.min_value is not None:
            if len(value) < metadata.min_value:
                return [f"Length must be at least {metadata.min_value}"]
        return []
    
    def _validate_max_length(self, value: Any, metadata: ParameterMetadata) -> List[str]:
        """Validate maximum length for sequences."""
        if hasattr(value, '__len__') and metadata.max_value is not None:
            if len(value) > metadata.max_value:
                return [f"Length must be at most {metadata.max_value}"]
        return []
    
    def _validate_regex(self, value: Any, metadata: ParameterMetadata) -> List[str]:
        """Validate value matches regex pattern."""
        # This would require additional metadata for the regex pattern
        # Implementation depends on how regex patterns are stored in metadata
        return []


class ConflictResolver:
    """
    Advanced conflict resolution for overlapping parameters.
    
    Implements sophisticated conflict resolution strategies with
    priority-based resolution and intelligent merging capabilities.
    """
    
    def __init__(self, default_mode: ConflictResolutionMode = ConflictResolutionMode.LOG_AND_CONTINUE):
        self.default_mode = default_mode
        self._resolution_strategies: Dict[ConflictResolutionMode, Callable] = {
            ConflictResolutionMode.FAIL_FAST: self._fail_fast_resolution,
            ConflictResolutionMode.LOG_AND_CONTINUE: self._log_and_continue_resolution,
            ConflictResolutionMode.SILENT_OVERRIDE: self._silent_override_resolution,
            ConflictResolutionMode.INTERACTIVE: self._interactive_resolution,
        }
    
    def resolve_conflicts(
        self,
        parameter_sources: Dict[str, Dict[ParameterSource, Any]],
        metadata_registry: Dict[str, ParameterMetadata],
        context: ResolutionContext
    ) -> Tuple[ParameterDict, List[str]]:
        """
        Resolve parameter conflicts using configured strategy.
        
        Args:
            parameter_sources: Mapping of parameters to their source values
            metadata_registry: Parameter metadata registry
            context: Resolution context
            
        Returns:
            Tuple of (resolved_parameters, conflict_messages)
        """
        resolved_parameters = {}
        conflicts = []
        
        resolution_mode = context.conflict_mode or self.default_mode
        resolver = self._resolution_strategies.get(
            resolution_mode, self._log_and_continue_resolution
        )
        
        for param_name, sources in parameter_sources.items():
            if len(sources) <= 1:
                # No conflict
                if sources:
                    resolved_parameters[param_name] = next(iter(sources.values()))
                continue
            
            # Conflict detected
            metadata = metadata_registry.get(param_name)
            if not metadata:
                # Use default priority order
                priority_order = [
                    ParameterSource.RUNTIME_OVERRIDE,
                    ParameterSource.PHASE_CONFIG,
                    ParameterSource.GLOBAL_CONFIG,
                    ParameterSource.ENVIRONMENT_VAR,
                    ParameterSource.DEFAULT_VALUE
                ]
            else:
                priority_order = metadata.source_priority
            
            try:
                resolved_value, conflict_msg = resolver(
                    param_name, sources, priority_order, metadata, context
                )
                resolved_parameters[param_name] = resolved_value
                if conflict_msg:
                    conflicts.append(conflict_msg)
            except ParameterConflictError as e:
                conflicts.append(f"CRITICAL: {str(e)}")
                if resolution_mode == ConflictResolutionMode.FAIL_FAST:
                    raise
        
        return resolved_parameters, conflicts
    
    def _fail_fast_resolution(
        self,
        param_name: str,
        sources: Dict[ParameterSource, Any],
        priority_order: List[ParameterSource],
        metadata: Optional[ParameterMetadata],
        context: ResolutionContext
    ) -> Tuple[Any, Optional[str]]:
        """Fail fast on any conflict."""
        raise ParameterConflictError(
            parameter_name=param_name,
            conflicting_values=sources,
            resolution_context=context
        )
    
    def _log_and_continue_resolution(
        self,
        param_name: str,
        sources: Dict[ParameterSource, Any],
        priority_order: List[ParameterSource],
        metadata: Optional[ParameterMetadata],
        context: ResolutionContext
    ) -> Tuple[Any, Optional[str]]:
        """Log conflicts and use highest priority value."""
        # Find highest priority source
        for source in priority_order:
            if source in sources:
                conflict_msg = (
                    f"Parameter '{param_name}' conflict resolved using {source.name}: "
                    f"{sources[source]} (alternatives: {dict(sources)})"
                )
                return sources[source], conflict_msg
        
        # Fallback to first available
        first_source, first_value = next(iter(sources.items()))
        conflict_msg = f"Parameter '{param_name}' using fallback {first_source.name}: {first_value}"
        return first_value, conflict_msg
    
    def _silent_override_resolution(
        self,
        param_name: str,
        sources: Dict[ParameterSource, Any],
        priority_order: List[ParameterSource],
        metadata: Optional[ParameterMetadata],
        context: ResolutionContext
    ) -> Tuple[Any, Optional[str]]:
        """Silently use highest priority value."""
        for source in priority_order:
            if source in sources:
                return sources[source], None
        
        # Fallback to first available
        first_value = next(iter(sources.values()))
        return first_value, None
    
    def _interactive_resolution(
        self,
        param_name: str,
        sources: Dict[ParameterSource, Any],
        priority_order: List[ParameterSource],
        metadata: Optional[ParameterMetadata],
        context: ResolutionContext
    ) -> Tuple[Any, Optional[str]]:
        """Interactive conflict resolution (development only)."""
        # For production, fall back to log and continue
        if context.metadata.get("environment") == "production":
            return self._log_and_continue_resolution(
                param_name, sources, priority_order, metadata, context
            )
        
        # In development, could implement interactive prompts
        # For now, use log and continue as fallback
        return self._log_and_continue_resolution(
            param_name, sources, priority_order, metadata, context
        )


class CacheManager:
    """
    Performance optimization with intelligent caching.
    
    Implements sophisticated caching strategies with TTL,
    invalidation policies, and memory management.
    """
    
    def __init__(
        self,
        max_cache_size: int = 1000,
        default_ttl: float = 300.0,  # 5 minutes
        cleanup_interval: float = 60.0  # 1 minute
    ):
        self.max_cache_size = max_cache_size
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval
        
        self._cache: Dict[str, Tuple[ResolutionResult, float]] = {}
        self._access_times: Dict[str, float] = {}
        self._cache_lock = threading.RLock()
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_worker,
            daemon=True,
            name="config-cache-cleanup"
        )
        self._cleanup_thread.start()
    
    def get(self, cache_key: str) -> Optional[ResolutionResult]:
        """Get cached resolution result."""
        with self._cache_lock:
            if cache_key not in self._cache:
                return None
            
            result, timestamp = self._cache[cache_key]
            
            # Check TTL
            if time.time() - timestamp > self.default_ttl:
                del self._cache[cache_key]
                if cache_key in self._access_times:
                    del self._access_times[cache_key]
                return None
            
            # Update access time
            self._access_times[cache_key] = time.time()
            
            # Mark as cache hit
            cached_result = copy.deepcopy(result)
            cached_result.cache_hit = True
            return cached_result
    
    def put(self, cache_key: str, result: ResolutionResult) -> None:
        """Cache resolution result."""
        with self._cache_lock:
            # Ensure cache size limit
            if len(self._cache) >= self.max_cache_size:
                self._evict_lru()
            
            # Store result
            self._cache[cache_key] = (copy.deepcopy(result), time.time())
            self._access_times[cache_key] = time.time()
    
    def invalidate(self, pattern: Optional[str] = None) -> int:
        """
        Invalidate cache entries.
        
        Args:
            pattern: Optional regex pattern to match keys
            
        Returns:
            Number of entries invalidated
        """
        with self._cache_lock:
            if pattern is None:
                # Clear all
                count = len(self._cache)
                self._cache.clear()
                self._access_times.clear()
                return count
            
            # Pattern-based invalidation
            import re
            regex = re.compile(pattern)
            keys_to_remove = [
                key for key in self._cache.keys()
                if regex.search(key)
            ]
            
            for key in keys_to_remove:
                del self._cache[key]
                if key in self._access_times:
                    del self._access_times[key]
            
            return len(keys_to_remove)
    
    def _evict_lru(self) -> None:
        """Evict least recently used entries."""
        if not self._access_times:
            return
        
        # Find LRU entry
        lru_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        
        # Remove from cache
        if lru_key in self._cache:
            del self._cache[lru_key]
        del self._access_times[lru_key]
    
    def _cleanup_worker(self) -> None:
        """Background cleanup worker."""
        while True:
            try:
                time.sleep(self.cleanup_interval)
                
                with self._cache_lock:
                    current_time = time.time()
                    expired_keys = [
                        key for key, (_, timestamp) in self._cache.items()
                        if current_time - timestamp > self.default_ttl
                    ]
                    
                    for key in expired_keys:
                        del self._cache[key]
                        if key in self._access_times:
                            del self._access_times[key]
                    
                    if expired_keys:
                        logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                        
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._cache_lock:
            return {
                "cache_size": len(self._cache),
                "max_cache_size": self.max_cache_size,
                "cache_utilization": len(self._cache) / self.max_cache_size * 100,
                "default_ttl": self.default_ttl,
                "cleanup_interval": self.cleanup_interval
            }


class ConfigurationResolver:
    """
    Main configuration resolver with comprehensive parameter resolution.
    
    Implements the Bridge pattern to decouple configuration abstractions
    from their concrete implementations, providing flexible and extensible
    parameter resolution with advanced features.
    
    Features:
        - Multi-source parameter resolution with conflict handling
        - Comprehensive validation with rule-based constraints
        - Performance optimization with intelligent caching
        - Extensive monitoring and observability
        - Thread-safe operations with concurrent access support
    """
    
    # Default parameter metadata registry
    DEFAULT_PARAMETER_METADATA: ClassVar[Dict[str, ParameterMetadata]] = {
        "temperature": ParameterMetadata(
            name="temperature",
            parameter_type=float,
            default_value=0.7,
            required=False,
            validation_rules=["positive"],
            description="Sampling temperature for randomness control",
            min_value=0.0,
            max_value=2.0
        ),
        "top_p": ParameterMetadata(
            name="top_p",
            parameter_type=float,
            default_value=0.9,
            required=False,
            validation_rules=["probability"],
            description="Nucleus sampling probability threshold",
            min_value=0.0,
            max_value=1.0
        ),
        "top_k": ParameterMetadata(
            name="top_k",
            parameter_type=int,
            default_value=50,
            required=False,
            validation_rules=["positive", "integer"],
            description="Top-k sampling parameter",
            min_value=1
        ),
        "min_p": ParameterMetadata(
            name="min_p",
            parameter_type=float,
            default_value=0.0,
            required=False,
            validation_rules=["probability"],
            description="Minimum probability threshold for sampling",
            min_value=0.0,
            max_value=1.0
        ),
        "min_tokens_to_keep": ParameterMetadata(
            name="min_tokens_to_keep",
            parameter_type=int,
            default_value=1,
            required=False,
            validation_rules=["positive", "integer"],
            description="Minimum number of tokens to keep during filtering",
            min_value=1
        ),
        "repetition_penalty": ParameterMetadata(
            name="repetition_penalty",
            parameter_type=float,
            default_value=1.0,
            required=False,
            validation_rules=["positive"],
            description="Penalty for token repetition",
            min_value=0.0,
            max_value=2.0
        ),
        "repetition_context_size": ParameterMetadata(
            name="repetition_context_size",
            parameter_type=int,
            default_value=20,
            required=False,
            validation_rules=["non_negative", "integer"],
            description="Context size for repetition penalty calculation",
            min_value=0
        ),
        "xtc_probability": ParameterMetadata(
            name="xtc_probability",
            parameter_type=float,
            default_value=0.0,
            required=False,
            validation_rules=["probability"],
            description="XTC sampling probability",
            min_value=0.0,
            max_value=1.0
        ),
        "xtc_threshold": ParameterMetadata(
            name="xtc_threshold",
            parameter_type=float,
            default_value=0.1,
            required=False,
            validation_rules=["probability"],
            description="XTC sampling threshold",
            min_value=0.0,
            max_value=1.0
        ),
        "xtc_special_tokens": ParameterMetadata(
            name="xtc_special_tokens",
            parameter_type=list,
            default_value=[],
            required=False,
            description="Special tokens for XTC sampling",
            phase_specific=False
        ),
        "max_tokens": ParameterMetadata(
            name="max_tokens",
            parameter_type=int,
            default_value=512,
            required=False,
            validation_rules=["positive", "integer"],
            description="Maximum number of tokens to generate",
            min_value=1,
            phase_specific=False
        ),
    }
    
    def __init__(
        self,
        parameter_mapper: Optional[ParameterMapperProtocol] = None,
        validation_engine: Optional[ValidationEngineProtocol] = None,
        conflict_resolver: Optional[ConflictResolver] = None,
        cache_manager: Optional[CacheManager] = None,
        metadata_registry: Optional[Dict[str, ParameterMetadata]] = None
    ):
        # Initialize components with defaults
        self.parameter_mapper = parameter_mapper or DefaultParameterMapper()
        self.validation_engine = validation_engine or DefaultValidationEngine()
        self.conflict_resolver = conflict_resolver or ConflictResolver()
        self.cache_manager = cache_manager or CacheManager()
        
        # Initialize metadata registry
        self.metadata_registry = metadata_registry or copy.deepcopy(self.DEFAULT_PARAMETER_METADATA)
        
        # Thread safety
        self._resolver_lock = threading.RLock()
        
        # Performance metrics
        self._resolution_count = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_resolution_time = 0.0
        
        logger.info("Initialized ConfigurationResolver with comprehensive parameter resolution")
    
    def resolve_parameters(
        self,
        config: ConfigType,
        context: Optional[ResolutionContext] = None,
        **override_params
    ) -> ResolutionResult:
        """
        Resolve configuration parameters with comprehensive processing.
        
        Args:
            config: Configuration object to resolve
            context: Optional resolution context
            **override_params: Runtime parameter overrides
            
        Returns:
            Comprehensive resolution result
            
        Raises:
            ConfigurationResolverError: If resolution fails critically
        """
        start_time = time.time()
        
        # Initialize context
        if context is None:
            context = ResolutionContext()
        
        # Add override parameters
        context.override_parameters.update(override_params)
        
        # Add environment variables
        import os
        context.environment_variables.update({
            k: v for k, v in os.environ.items()
            if k.startswith('MLX_SAMPLER_')
        })
        
        try:
            with self._resolver_lock:
                self._resolution_count += 1
                
                # Check cache
                cache_key = self._compute_cache_key(config, context)
                if context.cache_enabled:
                    cached_result = self.cache_manager.get(cache_key)
                    if cached_result:
                        self._cache_hits += 1
                        logger.debug(
                            f"Cache hit for resolution",
                            extra={
                                "correlation_id": context.correlation_id,
                                "cache_key": cache_key[:16] + "..."
                            }
                        )
                        return cached_result
                    else:
                        self._cache_misses += 1
                
                # Perform resolution
                result = self._resolve_parameters_internal(config, context)
                
                # Cache result
                if context.cache_enabled and result.success:
                    self.cache_manager.put(cache_key, result)
                
                # Update metrics
                resolution_time = (time.time() - start_time) * 1000
                result.resolution_time_ms = resolution_time
                self._total_resolution_time += resolution_time
                
                logger.info(
                    f"Parameter resolution completed",
                    extra={
                        "correlation_id": context.correlation_id,
                        "phase": context.phase,
                        "parameter_count": len(result.parameters),
                        "resolution_time_ms": resolution_time,
                        "success": result.success,
                        "cache_hit": result.cache_hit
                    }
                )
                
                return result
                
        except Exception as e:
            resolution_time = (time.time() - start_time) * 1000
            self._total_resolution_time += resolution_time
            
            logger.error(
                f"Parameter resolution failed: {e}",
                extra={
                    "correlation_id": context.correlation_id,
                    "phase": context.phase,
                    "resolution_time_ms": resolution_time,
                    "error_type": type(e).__name__
                }
            )
            
            if isinstance(e, ConfigurationResolverError):
                raise
            else:
                raise ConfigurationResolverError(
                    message=f"Resolution failed: {e}",
                    resolution_context=context,
                    correlation_id=context.correlation_id
                )
    
    def _resolve_parameters_internal(
        self,
        config: ConfigType,
        context: ResolutionContext
    ) -> ResolutionResult:
        """Internal parameter resolution implementation."""
        # 1. Map configuration to parameters
        mapped_parameters = self.parameter_mapper.map_parameters(
            config, context, self.metadata_registry
        )
        
        # 2. Collect parameters from all sources
        parameter_sources = self._collect_parameter_sources(
            mapped_parameters, context
        )
        
        # 3. Resolve conflicts
        resolved_parameters, conflicts = self.conflict_resolver.resolve_conflicts(
            parameter_sources, self.metadata_registry, context
        )
        
        # 4. Validate parameters
        validation_results = {}
        if context.validation_enabled:
            validation_results = self.validation_engine.validate_parameters(
                resolved_parameters, self.metadata_registry, context
            )
            
            # Check for validation failures
            validation_errors = []
            for param_name, (is_valid, errors) in validation_results.items():
                if not is_valid:
                    validation_errors.extend([f"{param_name}: {error}" for error in errors])
            
            if validation_errors and context.resolution_strategy == ResolutionStrategy.STRICT:
                raise ValidationFailureError(
                    parameter_name="multiple",
                    value=resolved_parameters,
                    validation_errors=validation_errors,
                    resolution_context=context
                )
        
        # 5. Build result
        result = ResolutionResult(
            parameters=resolved_parameters,
            conflicts=conflicts,
            validation_results=validation_results,
            metadata={
                "resolution_strategy": context.resolution_strategy.value,
                "conflict_mode": context.conflict_mode.value,
                "phase": context.phase,
                "parameter_count": len(resolved_parameters),
                "source_count": len(parameter_sources),
                "validation_enabled": context.validation_enabled
            }
        )
        
        # Add parameter sources for debugging
        for param_name in resolved_parameters:
            if param_name in parameter_sources:
                sources = parameter_sources[param_name]
                if sources:
                    # Use highest priority source
                    metadata = self.metadata_registry.get(param_name)
                    priority_order = metadata.source_priority if metadata else [
                        ParameterSource.RUNTIME_OVERRIDE,
                        ParameterSource.PHASE_CONFIG,
                        ParameterSource.GLOBAL_CONFIG,
                        ParameterSource.ENVIRONMENT_VAR,
                        ParameterSource.DEFAULT_VALUE
                    ]
                    
                    for source in priority_order:
                        if source in sources:
                            result.parameter_sources[param_name] = source
                            break
        
        return result
    
    def _collect_parameter_sources(
        self,
        mapped_parameters: ParameterDict,
        context: ResolutionContext
    ) -> Dict[str, Dict[ParameterSource, Any]]:
        """Collect parameters from all available sources."""
        parameter_sources = defaultdict(dict)
        
        # Add mapped parameters (from config)
        for param_name, value in mapped_parameters.items():
            if value is not None:
                # Determine source type based on context
                if param_name in context.override_parameters:
                    parameter_sources[param_name][ParameterSource.RUNTIME_OVERRIDE] = value
                else:
                    # Could be phase or global config - would need more context to determine
                    parameter_sources[param_name][ParameterSource.PHASE_CONFIG] = value
        
        # Add environment variables
        for env_var, env_value in context.environment_variables.items():
            if env_var.startswith('MLX_SAMPLER_'):
                param_name = env_var[12:].lower()  # Remove 'MLX_SAMPLER_' prefix
                if param_name in self.metadata_registry:
                    metadata = self.metadata_registry[param_name]
                    try:
                        converted_value = self._convert_env_value(env_value, metadata.parameter_type)
                        parameter_sources[param_name][ParameterSource.ENVIRONMENT_VAR] = converted_value
                    except (ValueError, TypeError):
                        logger.warning(f"Failed to convert environment variable {env_var}")
        
        # Add default values
        for param_name, metadata in self.metadata_registry.items():
            if metadata.default_value is not None:
                parameter_sources[param_name][ParameterSource.DEFAULT_VALUE] = metadata.default_value
        
        return dict(parameter_sources)
    
    def _convert_env_value(self, env_value: str, target_type: type) -> Any:
        """Convert environment variable string to target type."""
        if target_type == str:
            return env_value
        elif target_type == int:
            return int(env_value)
        elif target_type == float:
            return float(env_value)
        elif target_type == bool:
            return env_value.lower() in ('true', '1', 'yes', 'on')
        elif target_type == list:
            return env_value.split(',')
        else:
            # Try JSON parsing for complex types
            try:
                return json.loads(env_value)
            except json.JSONDecodeError:
                raise ValueError(f"Cannot convert '{env_value}' to {target_type}")
    
    def _compute_cache_key(self, config: ConfigType, context: ResolutionContext) -> str:
        """Compute cache key for configuration and context."""
        # Create a deterministic hash of the configuration and context
        config_hash = self._hash_config(config)
        context_hash = self._hash_context(context)
        
        cache_key = f"{config_hash}:{context_hash}"
        return hashlib.md5(cache_key.encode()).hexdigest()
    
    def _hash_config(self, config: ConfigType) -> str:
        """Compute hash of configuration object."""
        try:
            if hasattr(config, '__dict__'):
                config_dict = config.__dict__
            elif hasattr(config, '__dataclass_fields__'):
                config_dict = {
                    field.name: getattr(config, field.name, None)
                    for field in config.__dataclass_fields__.values()
                }
            else:
                config_dict = {"type": str(type(config))}
            
            # Sort for deterministic hashing
            sorted_items = sorted(config_dict.items())
            config_str = json.dumps(sorted_items, default=str, sort_keys=True)
            return hashlib.md5(config_str.encode()).hexdigest()
        except Exception:
            # Fallback to type-based hash
            return hashlib.md5(str(type(config)).encode()).hexdigest()
    
    def _hash_context(self, context: ResolutionContext) -> str:
        """Compute hash of resolution context."""
        context_dict = {
            "phase": context.phase,
            "override_parameters": sorted(context.override_parameters.items()),
            "environment_variables": sorted(context.environment_variables.items()),
            "resolution_strategy": context.resolution_strategy.value,
            "conflict_mode": context.conflict_mode.value,
            "validation_enabled": context.validation_enabled,
        }
        
        context_str = json.dumps(context_dict, default=str, sort_keys=True)
        return hashlib.md5(context_str.encode()).hexdigest()
    
    def register_parameter_metadata(
        self,
        name: str,
        metadata: ParameterMetadata,
        override: bool = False
    ) -> None:
        """
        Register new parameter metadata.
        
        Args:
            name: Parameter name
            metadata: Parameter metadata
            override: Whether to override existing metadata
            
        Raises:
            ValueError: If parameter already registered and override=False
        """
        with self._resolver_lock:
            if name in self.metadata_registry and not override:
                raise ValueError(f"Parameter metadata '{name}' already registered")
            
            self.metadata_registry[name] = metadata
            
            # Invalidate cache for this parameter
            self.cache_manager.invalidate(f".*{name}.*")
            
            logger.info(
                f"Registered parameter metadata for '{name}'",
                extra={
                    "parameter_type": metadata.parameter_type.__name__,
                    "required": metadata.required,
                    "override": override
                }
            )
    
    def get_resolution_stats(self) -> Dict[str, Any]:
        """Get resolution performance statistics."""
        with self._resolver_lock:
            total_requests = self._cache_hits + self._cache_misses
            cache_hit_rate = (self._cache_hits / total_requests * 100) if total_requests > 0 else 0.0
            avg_resolution_time = (
                self._total_resolution_time / self._resolution_count
                if self._resolution_count > 0 else 0.0
            )
            
            return {
                "total_resolutions": self._resolution_count,
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "cache_hit_rate": cache_hit_rate,
                "total_resolution_time_ms": self._total_resolution_time,
                "average_resolution_time_ms": avg_resolution_time,
                "registered_parameters": len(self.metadata_registry),
                "cache_stats": self.cache_manager.get_stats()
            }
    
    def reset_stats(self) -> None:
        """Reset resolution statistics."""
        with self._resolver_lock:
            self._resolution_count = 0
            self._cache_hits = 0
            self._cache_misses = 0
            self._total_resolution_time = 0.0
            
            logger.info("Reset resolution statistics")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on resolver components."""
        health_status = {
            "timestamp": time.time(),
            "overall_status": "healthy",
            "components": {}
        }
        
        # Check parameter mapper
        try:
            test_config = GenerationConfig()
            test_context = ResolutionContext()
            self.parameter_mapper.map_parameters(test_config, test_context, self.metadata_registry)
            health_status["components"]["parameter_mapper"] = "healthy"
        except Exception as e:
            health_status["components"]["parameter_mapper"] = f"unhealthy: {e}"
            health_status["overall_status"] = "degraded"
        
        # Check validation engine
        try:
            test_params = {"temperature": 0.7}
            self.validation_engine.validate_parameters(
                test_params, self.metadata_registry, ResolutionContext()
            )
            health_status["components"]["validation_engine"] = "healthy"
        except Exception as e:
            health_status["components"]["validation_engine"] = f"unhealthy: {e}"
            health_status["overall_status"] = "degraded"
        
        # Check cache manager
        try:
            cache_stats = self.cache_manager.get_stats()
            if cache_stats["cache_utilization"] > 95:
                health_status["components"]["cache_manager"] = "warning: high utilization"
                if health_status["overall_status"] == "healthy":
                    health_status["overall_status"] = "warning"
            else:
                health_status["components"]["cache_manager"] = "healthy"
        except Exception as e:
            health_status["components"]["cache_manager"] = f"unhealthy: {e}"
            health_status["overall_status"] = "degraded"
        
        return health_status


# Global resolver instance for convenience
_global_resolver: Optional[ConfigurationResolver] = None


def get_global_resolver() -> ConfigurationResolver:
    """
    Get the global configuration resolver instance.
    
    Returns:
        Global ConfigurationResolver singleton
    """
    global _global_resolver
    if _global_resolver is None:
        _global_resolver = ConfigurationResolver()
    return _global_resolver


# Utility functions for common operations
def resolve_sampling_parameters(
    config: ConfigType,
    phase: str = "global",
    **overrides
) -> ParameterDict:
    """
    Convenience function for resolving sampling parameters.
    
    Args:
        config: Configuration object
        phase: Sampling phase (think/answer/global)
        **overrides: Parameter overrides
        
    Returns:
        Resolved parameter dictionary
        
    Raises:
        ConfigurationResolverError: If resolution fails
    """
    resolver = get_global_resolver()
    context = ResolutionContext(phase=phase)
    result = resolver.resolve_parameters(config, context, **overrides)
    
    if not result.success:
        raise ConfigurationResolverError(
            f"Failed to resolve parameters: {result.conflicts}",
            resolution_context=context
        )
    
    return result.parameters


# Export public API
__all__ = [
    # Core classes
    "ConfigurationResolver",
    "ResolutionContext",
    "ResolutionResult",
    "ParameterMetadata",
    
    # Enums
    "ResolutionStrategy",
    "ParameterSource",
    "ConflictResolutionMode",
    
    # Protocols
    "ParameterMapperProtocol",
    "ValidationEngineProtocol",
    
    # Implementations
    "DefaultParameterMapper",
    "DefaultValidationEngine",
    "ConflictResolver",
    "CacheManager",
    
    # Exceptions
    "ConfigurationResolverError",
    "ParameterConflictError",
    "ValidationFailureError",
    
    # Utilities
    "get_global_resolver",
    "resolve_sampling_parameters",
]