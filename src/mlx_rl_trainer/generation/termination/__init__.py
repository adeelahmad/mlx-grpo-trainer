"""
Enhanced Generation Termination System

This module provides enterprise-grade text generation termination capabilities
with comprehensive EOS token handling, multiple stopping criteria, and robust
fallback mechanisms for the MLX RL Trainer framework.

Architecture:
    - TerminationStrategy: Strategy pattern for different termination approaches
    - EOSTokenDetector: Sophisticated EOS token detection with multi-format support
    - GenerationTerminator: Main orchestrator with circuit breaker patterns
    - TerminationMetrics: Comprehensive monitoring and observability
    - TerminationConfig: Configuration management with validation

Design Patterns Applied:
    - Strategy Pattern: Multiple termination strategies with common interface
    - Observer Pattern: Event-driven termination monitoring and logging
    - Chain of Responsibility: Multiple stopping criteria evaluation
    - Circuit Breaker Pattern: Fault tolerance for termination failures
    - Factory Pattern: Dynamic termination strategy creation
    - Template Method Pattern: Common termination workflow with customizable steps

SOLID Principles:
    - Single Responsibility: Each class handles one aspect of termination
    - Open/Closed: Extensible for new termination strategies without modification
    - Liskov Substitution: All strategies are interchangeable through protocols
    - Interface Segregation: Separate interfaces for different termination concerns
    - Dependency Inversion: Depends on abstractions for maximum flexibility

Example:
    >>> from mlx_rl_trainer.generation.termination import GenerationTerminator
    >>> from mlx_rl_trainer.generation.termination.config import TerminationConfig
    >>> 
    >>> config = TerminationConfig(
    ...     max_tokens=512,
    ...     eos_tokens=['<|endoftext|>', '</s>'],
    ...     enable_smart_stopping=True
    ... )
    >>> terminator = GenerationTerminator(config)
    >>> should_stop = terminator.should_terminate(tokens, logits, step)
"""

from .core import (
    GenerationTerminator,
    TerminationResult,
    TerminationReason,
    TerminationState,
    TerminationError,
    TerminationTimeoutError,
    InvalidTerminationConfigError
)

from .strategies import (
    TerminationStrategy,
    EOSTokenStrategy,
    MaxLengthStrategy,
    RepetitionStrategy,
    QualityStrategy,
    HybridStrategy,
    TerminationStrategyFactory
)

from .detectors import (
    EOSTokenDetector,
    TokenDetectionResult,
    TokenDetectionStrategy,
    MultiFormatEOSDetector,
    SmartEOSDetector,
    FallbackEOSDetector
)

from .config import (
    TerminationConfig,
    EOSTokenConfig,
    RepetitionConfig,
    QualityConfig,
    PerformanceConfig,
    validate_termination_config
)

from .metrics import (
    TerminationMetrics,
    TerminationEvent,
    TerminationAnalyzer,
    PerformanceTracker
)

from .utils import (
    create_terminator,
    get_default_config,
    analyze_generation_quality,
    detect_repetition_patterns,
    compute_termination_confidence
)

# Version information
__version__ = "1.0.0"
__author__ = "MLX RL Trainer Team"
__email__ = "support@mlx-rl-trainer.ai"

# Export public API
__all__ = [
    # Core components
    "GenerationTerminator",
    "TerminationResult",
    "TerminationReason",
    "TerminationState",
    
    # Exceptions
    "TerminationError",
    "TerminationTimeoutError", 
    "InvalidTerminationConfigError",
    
    # Strategies
    "TerminationStrategy",
    "EOSTokenStrategy",
    "MaxLengthStrategy",
    "RepetitionStrategy",
    "QualityStrategy",
    "HybridStrategy",
    "TerminationStrategyFactory",
    
    # Detectors
    "EOSTokenDetector",
    "TokenDetectionResult",
    "TokenDetectionStrategy",
    "MultiFormatEOSDetector",
    "SmartEOSDetector",
    "FallbackEOSDetector",
    
    # Configuration
    "TerminationConfig",
    "EOSTokenConfig",
    "RepetitionConfig",
    "QualityConfig",
    "PerformanceConfig",
    "validate_termination_config",
    
    # Metrics and monitoring
    "TerminationMetrics",
    "TerminationEvent",
    "TerminationAnalyzer",
    "PerformanceTracker",
    
    # Utilities
    "create_terminator",
    "get_default_config",
    "analyze_generation_quality",
    "detect_repetition_patterns",
    "compute_termination_confidence"
]

# Module metadata
__module_info__ = {
    "name": "mlx_rl_trainer.generation.termination",
    "description": "Enterprise-grade text generation termination system",
    "version": __version__,
    "dependencies": [
        "mlx-core>=0.15.0",
        "numpy>=1.24.0",
        "typing-extensions>=4.5.0"
    ],
    "optional_dependencies": {
        "monitoring": ["prometheus-client>=0.16.0"],
        "visualization": ["matplotlib>=3.7.0", "seaborn>=0.12.0"],
        "advanced_metrics": ["scipy>=1.10.0", "scikit-learn>=1.3.0"]
    }
}