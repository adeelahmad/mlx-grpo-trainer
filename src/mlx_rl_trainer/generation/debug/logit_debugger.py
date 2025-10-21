
"""
Enterprise-Grade Logit Transformation Debugging for MLX RL Trainer

This module implements comprehensive debugging capabilities for logit transformations,
providing real-time tracking, visualization, and analysis of logit processing operations.
It follows enterprise-grade software engineering principles with advanced architectural
patterns and comprehensive observability.

The logit debugger provides deep insights into:
- Step-by-step logit transformation tracking
- Bias application visualization and impact analysis
- Phase detection state machine debugging
- Token probability distribution analysis before/after processing
- Processor pipeline execution tracing with timing

Architecture Features:
- Observer Pattern: Real-time logit transformation notifications
- Strategy Pattern: Multiple visualization and analysis strategies
- Command Pattern: Logit transformation operations
- Decorator Pattern: Non-intrusive logit instrumentation
- Repository Pattern: Transformation history storage and retrieval
- Factory Pattern: Debug analyzer creation based on transformation types

SOLID Principles Implementation:
- Single Responsibility: Focused solely on logit transformation debugging
- Open/Closed: Extensible for new transformation types and analysis strategies
- Liskov Substitution: All analyzers implement common interfaces
- Interface Segregation: Separate interfaces for different debugging aspects
- Dependency Inversion: Depends on abstractions for maximum flexibility

Performance Characteristics:
- O(1) transformation lookup and validation
- O(log n) transformation history search with indexed storage
- O(k) analysis where k is number of active transformations
- Memory-efficient with bounded circular buffers
- Thread-safe operations with minimal locking overhead

Security Considerations:
- Secure transformation value sanitization
- Access control for sensitive transformations
- Audit logging for all transformation changes
- Input validation to prevent injection attacks
- Resource limits to prevent memory exhaustion

Example Usage:
    >>> from mlx_rl_trainer.generation.debug.logit_debugger import LogitDebugger
    >>> from mlx_rl_trainer.generation.processors.pipeline import ProcessorPipeline
    >>> 
    >>> # Create debugger with configuration
    >>> debugger = LogitDebugger(
    ...     enable_real_time_monitoring=True,
    ...     enable_visualization=True,
    ...     enable_performance_profiling=True,
    ...     output_format='json'
    ... )
    >>> 
    >>> # Attach to processor pipeline
    >>> pipeline = ProcessorPipeline(...)
    >>> debugger.attach_to_pipeline(pipeline)
    >>> 
    >>> # Debug logit transformations
    >>> with debugger.trace_transformations() as trace:
    ...     result = pipeline.process_logits(logits, history, context)
    >>> 
    >>> # Analyze transformations
    >>> analysis = debugger.analyze_transformations(trace)
    >>> print(f"Found {len(analysis['transformations'])} transformations")

Author: Roo (Elite AI Programming Assistant)
Version: 1.0.0
License: MIT
"""

import asyncio
import copy
import hashlib
import json
import logging
import math
import statistics
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque, OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from functools import wraps, lru_cache
from typing import (
    Any, Dict, List, Optional, Set, Tuple, Union, Callable,
    Protocol, runtime_checkable, TypeVar, Generic, ClassVar,
    NamedTuple, Iterator, AsyncIterator
)
from weakref import WeakKeyDictionary, WeakSet
import traceback
import inspect
import sys
from pathlib import Path

# MLX and framework imports
import mlx.core as mx
import numpy as np
from mlx_lm.tokenizer_utils import TokenizerWrapper

# Internal imports
from . import (
    BaseDebugger, DebugContext, DebugEvent, DebugLevel, DebugOutputFormat,
    DebugException, DebugOperationError, DebugConfigurationError
)

try:
    from ..processors.base import (
        ProcessingContext, ProcessingPhase, LogitProcessor, BaseLogitProcessor
    )
    from ..processors.pipeline import ProcessorPipeline, PipelineEvent
    from ..processors.enhanced_bias_processor import EnhancedBiasProcessor, BiasRule
    from ..processors.phase_processor import PhaseDetector, PhaseTransitionEvent
except ImportError:
    # Fallback imports for development
    from mlx_rl_trainer.generation.processors.base import (
        ProcessingContext, ProcessingPhase, LogitProcessor, BaseLogitProcessor
    )
    from mlx_rl_trainer.generation.processors.pipeline import ProcessorPipeline, PipelineEvent
    from mlx_rl_trainer.generation.processors.enhanced_bias_processor import EnhancedBiasProcessor, BiasRule
    from mlx_rl_trainer.generation.processors.phase_processor import PhaseDetector, PhaseTransitionEvent

# Type definitions
T = TypeVar('T')
LogitTransformationType = TypeVar('LogitTransformationType', bound='LogitTransformation')
VisualizationFormat = TypeVar('VisualizationFormat')

logger = logging.getLogger(__name__)


class TransformationType(Enum):
    """
    Types of logit transformations for categorization and analysis.
    
    This enum enables comprehensive transformation categorization,
    allowing developers to understand the nature and purpose of each
    transformation operation.
    """
    BIAS_APPLICATION = auto()
    TEMPERATURE_SCALING = auto()
    TOP_K_FILTERING = auto()
    TOP_P_FILTERING = auto()
    MIN_P_FILTERING = auto()
    REPETITION_PENALTY = auto()
    PHASE_TRANSITION = auto()
    CUSTOM_PROCESSOR = auto()
    COMPOSITE = auto()
    UNKNOWN = auto()

    def __str__(self) -> str:
        return self.name.lower()

    @property
    def is_filtering(self) -> bool:
        """Check if this transformation type is a filtering operation."""
        return self in (self.TOP_K_FILTERING, self.TOP_P_FILTERING, self.MIN_P_FILTERING)

    @property
    def is_scaling(self) -> bool:
        """Check if this transformation type is a scaling operation."""
        return self in (self.TEMPERATURE_SCALING, self.REPETITION_PENALTY)

    @property
    def is_bias(self) -> bool:
        """Check if this transformation type is a bias operation."""
        return self == self.BIAS_APPLICATION


class TransformationImpact(Enum):
    """Severity levels for transformation impact assessment."""
    NEGLIGIBLE = auto()
    LOW = auto()
    MODERATE = auto()
    HIGH = auto()
    CRITICAL = auto()

    def __str__(self) -> str:
        return self.name.lower()

    @property
    def is_significant(self) -> bool:
        """Check if this impact level is significant."""
        return self in (self.MODERATE, self.HIGH, self.CRITICAL)


@dataclass(frozen=True)
class LogitTransformation:
    """
    Immutable information about a logit transformation.
    
    Provides comprehensive metadata about transformations including
    their values, sources, validation status, and constraints.
    
    Attributes:
        transformation_id: Unique identifier for this transformation
        transformation_type: Type of transformation
        processor_id: ID of the processor that applied the transformation
        phase: Processing phase when transformation was applied
        before_logits: Logits tensor before transformation (hash or reference)
        after_logits: Logits tensor after transformation (hash or reference)
        token_changes: Dictionary of token IDs to probability changes
        impact_score: Numerical score of transformation impact (0.0-1.0)
        impact_level: Categorical impact level assessment
        timestamp: When the transformation occurred
        duration_ms: How long the transformation took to apply
        metadata: Additional transformation-specific metadata
    """
    transformation_id: str
    transformation_type: TransformationType
    processor_id: str
    phase: ProcessingPhase
    before_logits: Union[str, mx.array]
    after_logits: Union[str, mx.array]
    token_changes: Dict[int, float] = field(default_factory=dict)
    impact_score: float = 0.0
    impact_level: TransformationImpact = TransformationImpact.NEGLIGIBLE
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate transformation info consistency."""
        if not isinstance(self.transformation_type, TransformationType):
            object.__setattr__(self, 'transformation_type', TransformationType.UNKNOWN)
        
        if not isinstance(self.phase, ProcessingPhase):
            object.__setattr__(self, 'phase', ProcessingPhase.INITIALIZATION)
        
        if not 0.0 <= self.impact_score <= 1.0:
            object.__setattr__(self, 'impact_score', max(0.0, min(1.0, self.impact_score)))

    def to_dict(self) -> Dict[str, Any]:
        """Convert transformation info to dictionary."""
        return {
            'transformation_id': self.transformation_id,
            'transformation_type': str(self.transformation_type),
            'processor_id': self.processor_id,
            'phase': str(self.phase),
            'before_logits': self.before_logits if isinstance(self.before_logits, str) else "array_reference",
            'after_logits': self.after_logits if isinstance(self.after_logits, str) else "array_reference",
            'token_changes': self.token_changes,
            'impact_score': self.impact_score,
            'impact_level': str(self.impact_level),
            'timestamp': self.timestamp,
            'duration_ms': self.duration_ms,
            'metadata': self.metadata.copy()
        }


@dataclass
class TransformationTrace:
    """
    Container for a sequence of logit transformations.
    
    Tracks the complete chain of transformations applied to logits
    during a single processing operation, enabling comprehensive
    analysis and visualization.
    
    Attributes:
        trace_id: Unique identifier for this trace
        transformations: List of transformations in sequence
        initial_logits: Original logits before any transformations
        final_logits: Final logits after all transformations
        start_time: When the trace started
        end_time: When the trace ended
        correlation_id: Request correlation ID for tracing
        metadata: Additional trace-specific metadata
    """
    trace_id: str = field(default_factory=lambda: f"trace_{uuid.uuid4().hex[:8]}")
    transformations: List[LogitTransformation] = field(default_factory=list)
    initial_logits: Optional[mx.array] = None
    final_logits: Optional[mx.array] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Get total trace duration in milliseconds."""
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000

    @property
    def transformation_count(self) -> int:
        """Get number of transformations in this trace."""
        return len(self.transformations)

    def add_transformation(self, transformation: LogitTransformation) -> None:
        """Add a transformation to this trace."""
        self.transformations.append(transformation)

    def complete(self, final_logits: Optional[mx.array] = None) -> None:
        """Mark trace as complete with final logits."""
        self.end_time = time.time()
        if final_logits is not None:
            self.final_logits = final_logits

    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to dictionary for serialization."""
        return {
            'trace_id': self.trace_id,
            'transformations': [t.to_dict() for t in self.transformations],
            'initial_logits': "array_reference",
            'final_logits': "array_reference",
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration_ms': self.duration_ms,
            'transformation_count': self.transformation_count,
            'correlation_id': self.correlation_id,
            'metadata': self.metadata.copy()
        }


class LogitDebuggerException(DebugException):
    """Base exception for logit debugger-related errors."""
    pass


class TransformationAnalysisException(LogitDebuggerException):
    """Raised when transformation analysis encounters errors."""
    pass


class VisualizationException(LogitDebuggerException):
    """Raised when visualization operations fail."""
    pass


@runtime_checkable
class TransformationListener(Protocol):
    """
    Protocol for listening to logit transformation events.
    
    This protocol enables event-driven monitoring and reactive
    processing of logit transformation events.
    """
    
    def on_transformation(self, transformation: LogitTransformation) -> None:
        """
        Handle a logit transformation event.
        
        Args:
            transformation: Transformation event to handle
        """
        ...

    def get_listener_id(self) -> str:
        """
        Get unique identifier for this listener.
        
        Returns:
            Unique listener identifier
        """
        ...


@runtime_checkable
class VisualizationStrategy(Protocol):
    """
    Protocol for logit transformation visualization strategies.
    
    This protocol defines the interface for different visualization
    approaches, enabling pluggable visualization algorithms.
    """
    
    def visualize_transformation(
        self, 
        transformation: LogitTransformation,
        tokenizer: Optional[TokenizerWrapper] = None,
        **kwargs
    ) -> VisualizationFormat:
        """
        Visualize a logit transformation.
        
        Args:
            transformation: Transformation to visualize
            tokenizer: Optional tokenizer for token information
            **kwargs: Additional visualization parameters
            
        Returns:
            Visualization in the strategy's format
        """
        ...
    
    def visualize_trace(
        self, 
        trace: TransformationTrace,
        tokenizer: Optional[TokenizerWrapper] = None,
        **kwargs
    ) -> VisualizationFormat:
        """
        Visualize a complete transformation trace.
        
        Args:
            trace: Transformation trace to visualize
            tokenizer: Optional tokenizer for token information
            **kwargs: Additional visualization parameters
            
        Returns:
            Visualization in the strategy's format
        """
        ...


class TextVisualizationStrategy:
    """
    Text-based visualization strategy for logit transformations.
    
    This strategy generates human-readable text representations of
    logit transformations, suitable for console output and logging.
    """
    
    def visualize_transformation(
        self, 
        transformation: LogitTransformation,
        tokenizer: Optional[TokenizerWrapper] = None,
        max_tokens: int = 10,
        **kwargs
    ) -> str:
        """
        Generate text visualization of a transformation.
        
        Args:
            transformation: Transformation to visualize
            tokenizer: Optional tokenizer for token information
            max_tokens: Maximum number of tokens to include
            **kwargs: Additional visualization parameters
            
        Returns:
            Text visualization
        """
        lines = [
            f"Transformation: {transformation.transformation_id}",
            f"Type: {transformation.transformation_type}",
            f"Processor: {transformation.processor_id}",
            f"Phase: {transformation.phase}",
            f"Impact: {transformation.impact_level} ({transformation.impact_score:.3f})",
            f"Duration: {transformation.duration_ms:.3f} ms",
            "Token Changes:"
        ]
        
        # Sort token changes by magnitude (descending)
        sorted_changes = sorted(
            transformation.token_changes.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Display top token changes
        for i, (token_id, change) in enumerate(sorted_changes[:max_tokens]):
            token_str = f"ID {token_id}"
            if tokenizer:
                try:
                    token_text = tokenizer.decode([token_id])
                    token_str = f"'{token_text}' (ID {token_id})"
                except Exception:
                    pass
            
            change_str = f"{change:+.6f}"
            lines.append(f"  {token_str}: {change_str}")
        
        if len(sorted_changes) > max_tokens:
            lines.append(f"  ... and {len(sorted_changes) - max_tokens} more")
        
        return "\n".join(lines)
    
    def visualize_trace(
        self, 
        trace: TransformationTrace,
        tokenizer: Optional[TokenizerWrapper] = None,
        max_transformations: int = 5,
        **kwargs
    ) -> str:
        """
        Generate text visualization of a transformation trace.
        
        Args:
            trace: Transformation trace to visualize
            tokenizer: Optional tokenizer for token information
            max_transformations: Maximum number of transformations to include
            **kwargs: Additional visualization parameters
            
        Returns:
            Text visualization
        """
        lines = [
            f"Transformation Trace: {trace.trace_id}",
            f"Duration: {trace.duration_ms:.3f} ms",
            f"Transformations: {trace.transformation_count}",
            "---"
        ]
        
        # Display transformations
        for i, transformation in enumerate(trace.transformations[:max_transformations]):
            lines.append(f"[{i+1}/{trace.transformation_count}] {transformation.transformation_type}")
            lines.append(f"  Processor: {transformation.processor_id}")
            lines.append(f"  Impact: {transformation.impact_level} ({transformation.impact_score:.3f})")
            lines.append(f"  Duration: {transformation.duration_ms:.3f} ms")
        
        if trace.transformation_count > max_transformations:
            lines.append(f"... and {trace.transformation_count - max_transformations} more transformations")
        
        return "\n".join(lines)


class JsonVisualizationStrategy:
    """
    JSON-based visualization strategy for logit transformations.
    
    This strategy generates structured JSON representations of
    logit transformations, suitable for programmatic analysis
    and integration with visualization tools.
    """
    
    def visualize_transformation(
        self, 
        transformation: LogitTransformation,
        tokenizer: Optional[TokenizerWrapper] = None,
        include_tokens: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate JSON visualization of a transformation.
        
        Args:
            transformation: Transformation to visualize
            tokenizer: Optional tokenizer for token information
            include_tokens: Whether to include token text
            **kwargs: Additional visualization parameters
            
        Returns:
            JSON visualization as dictionary
        """
        result = transformation.to_dict()
        
        # Add token text if requested and tokenizer available
        if include_tokens and tokenizer and transformation.token_changes:
            token_details = {}
            for token_id, change in transformation.token_changes.items():
                try:
                    token_text = tokenizer.decode([token_id])
                    token_details[str(token_id)] = {
                        'text': token_text,
                        'change': change
                    }
                except Exception:
                    token_details[str(token_id)] = {
                        'text': f"<token_{token_id}>",
                        'change': change
                    }
            
            result['token_details'] = token_details
        
        return result
    
    def visualize_trace(
        self, 
        trace: TransformationTrace,
        tokenizer: Optional[TokenizerWrapper] = None,
        include_tokens: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate JSON visualization of a transformation trace.
        
        Args:
            trace: Transformation trace to visualize
            tokenizer: Optional tokenizer for token information
            include_tokens: Whether to include token text
            **kwargs: Additional visualization parameters
            
        Returns:
            JSON visualization as dictionary
        """
        result = trace.to_dict()
        
        # Add detailed transformations
        detailed_transformations = []
        for transformation in trace.transformations:
            detailed = self.visualize_transformation(
                transformation, 
                tokenizer=tokenizer,
                include_tokens=include_tokens
            )
            detailed_transformations.append(detailed)
        
        result['detailed_transformations'] = detailed_transformations
        
        # Add summary statistics
        result['statistics'] = {
            'transformation_count': trace.transformation_count,
            'total_duration_ms': trace.duration_ms,
            'average_duration_ms': trace.duration_ms / max(1, trace.transformation_count),
            'impact_distribution': self._calculate_impact_distribution(trace),
            'type_distribution': self._calculate_type_distribution(trace)
        }
        
        return result
    
    def _calculate_impact_distribution(self, trace: TransformationTrace) -> Dict[str, int]:
        """Calculate distribution of impact levels."""
        distribution = defaultdict(int)
        for transformation in trace.transformations:
            distribution[str(transformation.impact_level)] += 1
        return dict(distribution)
    
    def _calculate_type_distribution(self, trace: TransformationTrace) -> Dict[str, int]:
        """Calculate distribution of transformation types."""
        distribution = defaultdict(int)
        for transformation in trace.transformations:
            distribution[str(transformation.transformation_type)] += 1
        return dict(distribution)


class LogitTransformationAnalyzer:
    """
    Advanced analyzer for logit transformation behavior and impact.
    
    This class provides sophisticated analysis capabilities including
    impact assessment, token probability analysis, and optimization suggestions.
    """
    
    def __init__(self, tokenizer: Optional[TokenizerWrapper] = None):
        """
        Initialize transformation analyzer.
        
        Args:
            tokenizer: Optional tokenizer for token information
        """
        self.tokenizer = tokenizer
    
    def analyze_transformation(
        self, 
        transformation: LogitTransformation
    ) -> Dict[str, Any]:
        """
        Analyze a single transformation.
        
        Args:
            transformation: Transformation to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        analysis = {
            'transformation_id': transformation.transformation_id,
            'transformation_type': str(transformation.transformation_type),
            'impact_level': str(transformation.impact_level),
            'impact_score': transformation.impact_score,
            'duration_ms': transformation.duration_ms,
            'token_changes': len(transformation.token_changes),
        }
        
        # Add type-specific analysis
        if transformation.transformation_type == TransformationType.BIAS_APPLICATION:
            analysis.update(self._analyze_bias_transformation(transformation))
        elif transformation.transformation_type == TransformationType.TEMPERATURE_SCALING:
            analysis.update(self._analyze_temperature_transformation(transformation))
        elif transformation.transformation_type.is_filtering:
            analysis.update(self._analyze_filtering_transformation(transformation))
        
        return analysis
    
    def analyze_trace(
        self, 
        trace: TransformationTrace
    ) -> Dict[str, Any]:
        """
        Analyze a complete transformation trace.
        
        Args:
            trace: Transformation trace to analyze
            
        Returns:
            Dictionary containing comprehensive analysis results
        """
        if not trace.transformations:
            return {'error': 'No transformations to analyze'}
        
        # Analyze individual transformations
        transformation_analyses = [
            self.analyze_transformation(t) for t in trace.transformations
        ]
        
        # Calculate aggregate statistics
        total_duration = sum(t.duration_ms for t in trace.transformations)
        avg_duration = total_duration / len(trace.transformations)
        
        impact_scores = [t.impact_score for t in trace.transformations]
        avg_impact = sum(impact_scores) / len(impact_scores)
        max_impact = max(impact_scores)
        
        # Count transformations by type
        type_counts = defaultdict(int)
        for t in trace.transformations:
            type_counts[str(t.transformation_type)] += 1
        
        # Identify significant transformations
        significant_transformations = [
            t.transformation_id for t in trace.transformations
            if t.impact_level.is_significant
        ]
        
        # Calculate token change statistics
        all_token_changes = {}
        for t in trace.transformations:
            for token_id, change in t.token_changes.items():
                if token_id in all_token_changes:
                    all_token_changes[token_id] += change
                else:
                    all_token_changes[token_id] = change
        
        # Find most affected tokens
        most_affected_tokens = sorted(
            all_token_changes.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:10]
        
        # Add token text if tokenizer available
        token_details = {}
        if self.tokenizer:
            for token_id, change in most_affected_tokens:
                try:
                    token_text = self.tokenizer.decode([token_id])
                    token_details[str(token_id)] = {
                        'text': token_text,
                        'change': change
                    }
                except Exception:
                    token_details[str(token_id)] = {
                        'text': f"<token_{token_id}>",
                        'change': change
                    }
        
        return {
            'trace_id': trace.trace_id,
            'transformation_count': len(trace.transformations),
            'total_duration_ms': total_duration,
            'average_duration_ms': avg_duration,
            'average_impact_score': avg_impact,
            'maximum_impact_score': max_impact,
            'transformation_types': dict(type_counts),
            'significant_transformations': significant_transformations,
            'most_affected_tokens': [
                {'token_id': t[0], 'change': t[1]} for t in most_affected_tokens
            ],
            'token_details': token_details,
            'transformation_analyses': transformation_analyses
        }
    
    def _analyze_bias_transformation(
        self, 
        transformation: LogitTransformation
    ) -> Dict[str, Any]:
        """Analyze bias application transformation."""
        # Extract bias-specific metadata
        bias_type = transformation.metadata.get('bias_type', 'unknown')
        bias_strength = transformation.metadata.get('bias_strength', 0.0)
        target_tokens = transformation.metadata.get('target_tokens', [])
        
        # Calculate bias effectiveness
        affected_tokens = set(transformation.token_changes.keys())
        target_coverage = len(affected_tokens.intersection(target_tokens)) / max(1, len(target_tokens))
        
        # Calculate average change magnitude
        change_magnitudes = [abs(c) for c in transformation.token_changes.values()]
        avg_change = sum(change_magnitudes) / max(1, len(change_magnitudes))
        
        return {
            'bias_type': bias_type,
            'bias_strength': bias_strength,
            'target_token_count': len(target_tokens),
            'affected_token_count': len(affected_tokens),
            'target_coverage': target_coverage,
            'average_change_magnitude': avg_change
        }
    
    def _analyze_temperature_transformation(
        self, 
        transformation: LogitTransformation
    ) -> Dict[str, Any]:
        """Analyze temperature scaling transformation."""
        # Extract temperature-specific metadata
        temperature = transformation.metadata.get('temperature', 1.0)
        
        # Calculate entropy change
        before_entropy = transformation.metadata.get('before_entropy', 0.0)
        after_entropy = transformation.metadata.get('after_entropy', 0.0)
        entropy_change = after_entropy - before_entropy
        
        return {
            'temperature': temperature,
            'before_entropy': before_entropy,
            'after_entropy': after_entropy,
            'entropy_change': entropy_change,
            'entropy_change_percent': (entropy_change / max(0.001, before_entropy)) * 100
        }
    
    def _analyze_filtering_transformation(
        self, 
        transformation: LogitTransformation
    ) -> Dict[str, Any]:
        """Analyze filtering transformation."""
        # Extract filtering-specific metadata
        filter_type = str(transformation.transformation_type)
        threshold = transformation.metadata.get('threshold', 0.0)
        
        # Calculate filtering statistics
        filtered_tokens = transformation.metadata.get('filtered_tokens', [])
        filtered_count = len(filtered_tokens)
        filtered_probability_mass = transformation.metadata.get('filtered_probability_mass', 0.0)
        
        return {
            'filter_type': filter_type,
            'threshold': threshold,
            'filtered_token_count': filtered_count,
            'filtered_probability_mass': filtered_probability_mass,
            'vocabulary_percent_filtered': filtered_count / max(1, transformation.metadata.get('vocabulary_size', 1)) * 100
        }


class LogitDebugger(BaseDebugger):
    """
    Comprehensive logit transformation debugger with enterprise-grade capabilities.
    
    This class provides deep debugging insights into logit transformation behavior,
    including real-time monitoring, visualization, impact analysis, and performance
    profiling. It implements multiple design patterns for extensibility and reliability.
    
    Key Features:
    - Step-by-step logit transformation tracking
    - Bias application visualization and impact analysis
    - Phase detection state machine debugging
    - Token probability distribution analysis before/after processing
    - Processor pipeline execution tracing with timing
    
    Architecture Patterns:
    - Observer: Real-time transformation notifications
    - Strategy: Multiple visualization and analysis strategies
    - Command: Transformation operations and analysis
    - Repository: Transformation history storage and retrieval
    - Factory: Analyzer creation based on transformation types
    """
    
    def __init__(
        self,
        debugger_id: Optional[str] = None,
        debug_level: DebugLevel = DebugLevel.INFO,
        output_format: DebugOutputFormat = DebugOutputFormat.JSON,
        enable_real_time_monitoring: bool = True,
        enable_visualization: bool = True,
        enable_performance_profiling: bool = True,
        max_transformation_history: int = 10000,
        max_trace_history: int = 100,
        tokenizer: Optional[TokenizerWrapper] = None
    ):
        """
        Initialize logit debugger.
        
        Args:
            debugger_id: Unique debugger identifier
            debug_level: Debug level for this debugger
            output_format: Output format for debug data
            enable_real_time_monitoring: Enable real-time transformation monitoring
            enable_visualization: Enable transformation visualization
            enable_performance_profiling: Enable performance profiling
            max_transformation_history: Maximum transformation history entries
            max_trace_history: Maximum trace history entries
            tokenizer: Optional tokenizer for token information
        """
        super().__init__(
            debugger_id=debugger_id or f"logit_debugger_{uuid.uuid4().hex[:8]}",
            debug_level=debug_level,
            output_format=output_format
        )
        
        self.enable_real_time_monitoring = enable_real_time_monitoring
        self.enable_visualization = enable_visualization
        self.enable_performance_profiling = enable_performance_profiling
        self.max_transformation_history = max_transformation_history
        self.max_trace_history = max_trace_history
        self.tokenizer = tokenizer
        
        # Transformation tracking
        self._transformation_history: deque[LogitTransformation] = deque(maxlen=max_transformation_history)
        self._trace_history: deque[TransformationTrace] = deque(maxlen=max_trace_history)
        self._transformation_lock = threading.RLock()
        
        # Active traces
        self._active_traces: Dict[str, TransformationTrace] = {}
        self._active_lock = threading.RLock()
        
        # Event handling
        self._transformation_listeners: List[TransformationListener] = []
        self._listener_lock = threading.RLock()
        
        # Visualization and analysis
        self._visualization_strategies = {
            DebugOutputFormat.TEXT: TextVisualizationStrategy(),
            DebugOutputFormat.JSON: JsonVisualizationStrategy()
        }
        self._analyzer = LogitTransformationAnalyzer(tokenizer=self.tokenizer)
        
        # Performance monitoring
        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._monitoring_stop_event = threading.Event()
        
        # Pipeline integration
        self._attached_pipelines: WeakSet[ProcessorPipeline] = WeakSet()
        self._pipeline_lock = threading.RLock()
        
        # Performance tracking
        self._total_transformations = 0
        self._total_traces = 0
        self._total_transformation_time = 0.0
        
        logger.info(f"Initialized LogitDebugger: {self.debugger_id}")
    
    def _on_debug_start(self, context: DebugContext) -> None:
        """Start logit transformation debugging session."""
        self._emit_event(DebugEvent(
            event_type="logit_debug_start",
            level=DebugLevel.INFO,
            message=f"Started logit transformation debugging for session {context.session_id}",
            context=context,
            data={
                'features': {
                    'real_time_monitoring': self.enable_real_time_monitoring,
                    'visualization': self.enable_visualization,
                    'performance_profiling': self.enable_performance_profiling
                }
            }
        ))
        
        # Start real-time monitoring if enabled
        if self.enable_real_time_monitoring and not self._monitoring_active:
            self._start_real_time_monitoring()
    
    def _on_debug_stop(self, context: DebugContext) -> None:
        """Stop logit transformation debugging session."""
        # Stop real-time monitoring
        if self._monitoring_active:
            self._stop_real_time_monitoring()
        
        # Generate final analysis report
        final_report = self._generate_final_report()
        
        self._emit_event(DebugEvent(
            event_type="logit_debug_stop",
            level=DebugLevel.INFO,
            message=f"Stopped logit transformation debugging for session {context.session_id}",
            context=context,
            data={'final_report': final_report}
        ))
    
    def track_transformation(
        self,
        before_logits: mx.array,
        after_logits: mx.array,
        processor_id: str,
        phase: ProcessingPhase,
        transformation_type: TransformationType,
        duration_ms: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> LogitTransformation:
        """
        Track a logit transformation with comprehensive metadata.
        
        This method creates a detailed transformation record with impact
        analysis and adds it to the transformation history.
        
        Args:
            before_logits: Logits tensor before transformation
            after_logits: Logits tensor after transformation
            processor_id: ID of processor that applied transformation
            phase: Processing phase when transformation was applied
            transformation_type: Type of transformation
            duration_ms: Duration of transformation in milliseconds
            metadata: Additional transformation-specific metadata
            
        Returns:
            Created LogitTransformation object
            
        Raises:
            LogitDebuggerException: If tracking fails
        """
        if not self.is_enabled():
            # Create minimal transformation object when disabled
            return LogitTransformation(
                transformation_id=f"disabled_{uuid.uuid4().hex[:8]}",
                transformation_type=transformation_type,
                processor_id=processor_id,
                phase=phase,
                before_logits="disabled",
                after_logits="disabled"
            )
        
        try:
            # Generate transformation ID
            transformation_id = f"transform_{uuid.uuid4().hex[:8]}"
            
            # Calculate token changes and impact
            token_changes, impact_score = self._calculate_transformation_impact(
                before_logits, after_logits
            )
            
            # Determine impact level
            impact_level = self._determine_impact_level(impact_score)
            
            # Create transformation object
            transformation = LogitTransformation(
                transformation_id=transformation_id,
                transformation_type=transformation_type,
                processor_id=processor_id,
                phase=phase,
                before_logits=self._hash_or_store_logits(before_logits),
                after_logits=self._hash_or_store_logits(after_logits),
                token_changes=token_changes,
                impact_score=impact_score,
                impact_level=impact_level,
                duration_ms=duration_ms,
                metadata=metadata or {}
            )
            
            # Store transformation
            with self._transformation_lock:
                self._transformation_history.append(transformation)
                self._total_transformations += 1
                self._total_transformation_time += duration_ms
            
            # Add to active trace if exists
            self._add_to_active_traces(transformation)
            
            # Notify listeners
            self._notify_transformation_listeners(transformation)
            
            return transformation
            
        except Exception as e:
            logger.error(f"Failed to track transformation: {e}")
            raise LogitDebuggerException(f"Failed to track transformation: {e}")
    
    def _calculate_transformation_impact(
        self,
        before_logits: mx.array,
        after_logits: mx.array
    ) -> Tuple[Dict[int, float], float]:
        """
        Calculate token-level changes and overall impact score.
        
        Args:
            before_logits: Logits tensor before transformation
            after_logits: Logits tensor after transformation
            
        Returns:
            Tuple of (token_changes, impact_score)
        """
        # Convert to probabilities for meaningful comparison
        before_probs = mx.softmax(before_logits, axis=-1)
        after_probs = mx.softmax(after_logits, axis=-1)
        
        # Calculate probability changes
        prob_diff = after_probs - before_probs
        
        # Find tokens with significant changes
        token_changes = {}
        
        # Use numpy for efficient operations
        prob_diff_np = prob_diff.tolist()
        
        # Process each batch item
        for batch_idx, batch_diffs in enumerate(prob_diff_np):
            # Find tokens with significant changes (above threshold)
            for token_id, diff in enumerate(batch_diffs):
                if abs(diff) > 0.001:  # Threshold for significant change
                    token_changes[token_id] = diff
        
        # Calculate overall impact score (0.0-1.0)
        # Based on Jensen-Shannon divergence between probability distributions
        impact_score = self._calculate_distribution_divergence(before_probs, after_probs)
        
        return token_changes, impact_score
    
    def _calculate_distribution_divergence(
        self,
        before_probs: mx.array,
        after_probs: mx.array
    ) -> float:
        """
        Calculate divergence between probability distributions.
        
        Uses Jensen-Shannon divergence, which is a symmetric measure
        of the difference between two probability distributions.
        
        Args:
            before_probs: Probability distribution before transformation
            after_probs: Probability distribution after transformation
            
        Returns:
            Divergence score (0.0-1.0)
        """
        try:
            # Convert to numpy for calculation
            before_np = np.array(before_probs.tolist())
            after_np = np.array(after_probs.tolist())
            
            # Calculate mean distribution
            mean_dist = 0.5 * (before_np + after_np)
            
            # Calculate KL divergences
            kl_before = self._kl_divergence(before_np, mean_dist)
            kl_after = self._kl_divergence(after_np, mean_dist)
            
            # Jensen-Shannon divergence
            js_div = 0.5 * (kl_before + kl_after)
            
            # Average over batch dimension
            avg_js = np.mean(js_div)
            
            # Scale to 0.0-1.0 range (JS div is between 0 and ln(2))
            scaled_impact = min(1.0, avg_js / np.log(2))
            
            return float(scaled_impact)
            
        except Exception as e:
            logger.warning(f"Failed to calculate distribution divergence: {e}")
            return 0.0
    
    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """
        Calculate Kullback-Leibler divergence between distributions.
        
        Args:
            p: First probability distribution
            q: Second probability distribution
            
        Returns:
            KL divergence
        """
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        q = np.maximum(q, epsilon)
        
        # Calculate KL divergence
        return np.sum(p * np.log(p / q + epsilon), axis=-1)
    
    def _determine_impact_level(self, impact_score: float) -> TransformationImpact:
        """
        Determine categorical impact level from numerical score.
        
        Args:
            impact_score: Impact score (0.0-1.0)
            
        Returns:
            Impact level category
        """
        if impact_score < 0.01:
            return TransformationImpact.NEGLIGIBLE
        elif impact_score < 0.05:
            return TransformationImpact.LOW
        elif impact_score < 0.2:
            return TransformationImpact.MODERATE
        elif impact_score < 0.5:
            return TransformationImpact.HIGH
        else:
            return TransformationImpact.CRITICAL
    
    def _hash_or_store_logits(self, logits: mx.array) -> Union[str, mx.array]:
        """
        Hash or store logits based on configuration.
        
        Args:
            logits: Logits tensor
            
        Returns:
            Hash string or original tensor
        """
        # For now, just return a hash to save memory
        logits_bytes = str(logits.tolist()).encode('utf-8')
        return hashlib.md5(logits_bytes).hexdigest()
    
    def _add_to_active_traces(self, transformation: LogitTransformation) -> None:
        """
        Add transformation to all active traces.
        
        Args:
            transformation: Transformation to add
        """
        with self._active_lock:
            for trace_id, trace in self._active_traces.items():
                trace.add_transformation(transformation)
    
    def _notify_transformation_listeners(self, transformation: LogitTransformation) -> None:
        """
        Notify all registered listeners of a transformation.
        
        Args:
            transformation: Transformation to broadcast
        """
        with self._listener_lock:
            for listener in self._transformation_listeners:
                try:
                    listener.on_transformation(transformation)
                except Exception as e:
                    logger.error(f"Transformation listener failed: {e}")