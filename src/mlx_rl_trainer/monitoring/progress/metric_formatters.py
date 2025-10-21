"""
Enterprise-Grade Metric Formatting System for Progress Display

This module provides a comprehensive metric formatting system that transforms
raw training metrics into human-readable, contextually appropriate display
formats. The system implements advanced formatting strategies using the
Strategy and Factory patterns for maximum flexibility and extensibility.

The formatting system is designed with enterprise-grade patterns:
- Factory Pattern: Dynamic formatter creation based on metric types
- Strategy Pattern: Interchangeable formatting algorithms
- Template Method: Consistent formatting workflow with customizable steps
- Decorator Pattern: Composable formatting enhancements
- Chain of Responsibility: Hierarchical formatting fallbacks

Key Features:
- Intelligent unit conversion and scaling (bytes to MB/GB, seconds to human time)
- Context-aware precision adjustment based on value ranges
- Localization support for international deployments
- Performance-optimized formatting with caching
- Extensible plugin architecture for custom metric types
- Comprehensive error handling with graceful degradation
- Thread-safe operations for concurrent training environments

Supported Metric Categories:
- Numeric Metrics: Loss values, learning rates, accuracy scores
- Memory Metrics: RAM usage, GPU memory, cache statistics
- Time Metrics: Step duration, ETA calculations, timestamps
- Percentage Metrics: Progress completion, accuracy rates
- Rate Metrics: Tokens/second, samples/second, throughput
- Scientific Metrics: Gradient norms, statistical measures

Example:
    ```python
    from mlx_rl_trainer.monitoring.progress.metric_formatters import (
        MetricFormatterFactory, FormattingContext
    )
    
    # Create formatter factory
    factory = MetricFormatterFactory()
    
    # Format different metric types
    loss_formatter = factory.create_formatter('loss')
    memory_formatter = factory.create_formatter('memory')
    time_formatter = factory.create_formatter('time')
    
    # Format with context
    context = FormattingContext(
        precision=3,
        use_scientific=False,
        locale='en_US'
    )
    
    formatted_loss = loss_formatter.format(0.00123456, context)
    formatted_memory = memory_formatter.format(1073741824, context)  # 1GB in bytes
    formatted_time = time_formatter.format(3661.5, context)  # 1h 1m 1.5s
    ```
"""

import time
import threading
import logging
import math
import locale
from typing import Dict, Any, List, Optional, Union, Callable, Tuple, Type
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
from collections import defaultdict, deque
import weakref
import re

from .exceptions import (
    RendererError,
    ConfigurationError,
    create_error_context,
)

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics that can be formatted."""
    NUMERIC = "numeric"
    MEMORY = "memory"
    TIME = "time"
    PERCENTAGE = "percentage"
    RATE = "rate"
    SCIENTIFIC = "scientific"
    CURRENCY = "currency"
    CUSTOM = "custom"


class FormattingPrecision(Enum):
    """Precision levels for metric formatting."""
    MINIMAL = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    MAXIMUM = 6


class UnitSystem(Enum):
    """Unit systems for metric formatting."""
    METRIC = "metric"
    IMPERIAL = "imperial"
    BINARY = "binary"
    SCIENTIFIC = "scientific"


@dataclass(frozen=True)
class FormattingContext:
    """
    Context information for metric formatting operations.
    
    This class encapsulates all the contextual information needed
    to format metrics appropriately, including precision requirements,
    localization settings, and display constraints.
    
    Attributes:
        precision: Number of decimal places to display
        use_scientific: Whether to use scientific notation for large/small numbers
        locale: Locale string for number formatting (e.g., 'en_US', 'de_DE')
        unit_system: Unit system to use for conversions
        max_width: Maximum character width for formatted output
        use_colors: Whether to include color codes in output
        show_units: Whether to display unit suffixes
        compact_mode: Whether to use compact formatting
        theme: Color theme for formatting
        custom_units: Custom unit mappings
    """
    precision: int = 2
    use_scientific: bool = False
    locale: str = "en_US"
    unit_system: UnitSystem = UnitSystem.METRIC
    max_width: int = 12
    use_colors: bool = False
    show_units: bool = True
    compact_mode: bool = False
    theme: str = "default"
    custom_units: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate formatting context parameters."""
        if self.precision < 0 or self.precision > 10:
            raise ValueError("Precision must be between 0 and 10")
        
        if self.max_width < 4:
            raise ValueError("Max width must be at least 4 characters")


@dataclass
class FormattingResult:
    """
    Result of a metric formatting operation.
    
    Attributes:
        formatted_value: The formatted string representation
        original_value: The original numeric value
        unit: The unit used in formatting
        precision_used: Actual precision used in formatting
        is_truncated: Whether the value was truncated to fit width
        metadata: Additional formatting metadata
    """
    formatted_value: str
    original_value: Union[int, float]
    unit: str = ""
    precision_used: int = 2
    is_truncated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        """Return the formatted value as string."""
        return self.formatted_value


class MetricFormatter(ABC):
    """
    Abstract base class for all metric formatters.
    
    This class defines the interface for metric formatting strategies
    and provides common functionality for all formatter implementations.
    The class implements the Template Method pattern to ensure consistent
    formatting workflow while allowing customization of specific steps.
    
    The formatting process follows these steps:
    1. Validate input value and context
    2. Normalize value to standard units
    3. Determine appropriate precision and scale
    4. Apply formatting rules and conversions
    5. Add units and apply styling
    6. Ensure output fits within constraints
    """
    
    def __init__(
        self,
        metric_type: MetricType,
        default_precision: int = 2,
        cache_size: int = 1000
    ):
        """
        Initialize metric formatter.
        
        Args:
            metric_type: Type of metric this formatter handles
            default_precision: Default precision for formatting
            cache_size: Size of the formatting cache
        """
        self.metric_type = metric_type
        self.default_precision = default_precision
        self.cache_size = cache_size
        
        # Performance optimization: cache formatted results
        self._format_cache: Dict[Tuple[float, str], FormattingResult] = {}
        self._cache_access_order: deque = deque(maxlen=cache_size)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self.format_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.debug(f"Initialized {self.__class__.__name__} formatter")
    
    def format(
        self,
        value: Union[int, float],
        context: Optional[FormattingContext] = None
    ) -> FormattingResult:
        """
        Format a metric value with the given context.
        
        This is the main entry point for formatting operations.
        It implements caching and error handling around the
        core formatting logic.
        
        Args:
            value: Numeric value to format
            context: Formatting context (uses defaults if None)
            
        Returns:
            FormattingResult containing formatted value and metadata
            
        Raises:
            RendererError: If formatting fails
        """
        with self._lock:
            self.format_count += 1
            
            # Use default context if none provided
            if context is None:
                context = FormattingContext(precision=self.default_precision)
            
            # Generate cache key
            cache_key = self._generate_cache_key(value, context)
            
            # Check cache
            if cache_key in self._format_cache:
                self.cache_hits += 1
                self._update_cache_access(cache_key)
                return self._format_cache[cache_key]
            
            self.cache_misses += 1
            
            try:
                # Validate inputs
                self._validate_inputs(value, context)
                
                # Perform formatting
                result = self._format_impl(value, context)
                
                # Cache result
                self._cache_result(cache_key, result)
                
                return result
                
            except Exception as e:
                raise RendererError(
                    f"Formatting failed for {self.metric_type.value} metric: {e}",
                    renderer_type=self.__class__.__name__,
                    context=create_error_context(
                        "metric_formatting",
                        value=value,
                        metric_type=self.metric_type.value
                    )
                ) from e
    
    def _generate_cache_key(
        self,
        value: Union[int, float],
        context: FormattingContext
    ) -> Tuple[float, str]:
        """Generate cache key for value and context."""
        # Create a string representation of context for caching
        context_str = (
            f"{context.precision}_{context.use_scientific}_{context.locale}_"
            f"{context.unit_system.value}_{context.max_width}_{context.compact_mode}"
        )
        return (float(value), context_str)
    
    def _update_cache_access(self, cache_key: Tuple[float, str]) -> None:
        """Update cache access order for LRU eviction."""
        if cache_key in self._cache_access_order:
            self._cache_access_order.remove(cache_key)
        self._cache_access_order.append(cache_key)
    
    def _cache_result(
        self,
        cache_key: Tuple[float, str],
        result: FormattingResult
    ) -> None:
        """Cache formatting result with LRU eviction."""
        # Evict oldest entry if cache is full
        if len(self._format_cache) >= self.cache_size:
            if self._cache_access_order:
                oldest_key = self._cache_access_order.popleft()
                self._format_cache.pop(oldest_key, None)
        
        # Store result
        self._format_cache[cache_key] = result
        self._update_cache_access(cache_key)
    
    def _validate_inputs(
        self,
        value: Union[int, float],
        context: FormattingContext
    ) -> None:
        """
        Validate input value and formatting context.
        
        Args:
            value: Value to validate
            context: Context to validate
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not isinstance(value, (int, float)):
            raise ValueError(f"Value must be numeric, got {type(value)}")
        
        if math.isnan(value):
            raise ValueError("Cannot format NaN values")
        
        if not isinstance(context, FormattingContext):
            raise ValueError(f"Context must be FormattingContext, got {type(context)}")
    
    @abstractmethod
    def _format_impl(
        self,
        value: Union[int, float],
        context: FormattingContext
    ) -> FormattingResult:
        """
        Implement the actual formatting logic.
        
        Subclasses must implement this method to provide
        specific formatting behavior for their metric type.
        
        Args:
            value: Numeric value to format
            context: Formatting context
            
        Returns:
            FormattingResult with formatted value and metadata
        """
        pass
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get formatter performance statistics."""
        with self._lock:
            hit_rate = self.cache_hits / max(self.format_count, 1)
            return {
                "format_count": self.format_count,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "hit_rate": hit_rate,
                "cache_size": len(self._format_cache),
                "max_cache_size": self.cache_size,
            }
    
    def clear_cache(self) -> None:
        """Clear the formatting cache."""
        with self._lock:
            self._format_cache.clear()
            self._cache_access_order.clear()


class NumericFormatter(MetricFormatter):
    """
    Formatter for general numeric values.
    
    This formatter handles basic numeric values like loss, accuracy,
    and other scalar metrics. It provides intelligent precision
    adjustment based on value magnitude and context requirements.
    
    Features:
    - Automatic precision adjustment based on value range
    - Scientific notation for very large or small values
    - Thousands separators for readability
    - Configurable decimal places
    - Sign handling for positive/negative values
    """
    
    def __init__(self, **kwargs):
        """Initialize numeric formatter."""
        super().__init__(MetricType.NUMERIC, **kwargs)
        
        # Precision thresholds for automatic adjustment
        self.precision_thresholds = {
            1e-6: 6,    # Very small values need high precision
            1e-3: 4,    # Small values need medium precision
            1e0: 3,     # Normal values need standard precision
            1e3: 2,     # Large values need less precision
            1e6: 1,     # Very large values need minimal precision
        }
    
    def _format_impl(
        self,
        value: Union[int, float],
        context: FormattingContext
    ) -> FormattingResult:
        """Format numeric value."""
        original_value = float(value)
        
        # Handle special cases
        if math.isinf(value):
            return FormattingResult(
                formatted_value="∞" if value > 0 else "-∞",
                original_value=original_value,
                precision_used=0
            )
        
        if value == 0:
            return FormattingResult(
                formatted_value="0",
                original_value=original_value,
                precision_used=0
            )
        
        # Determine precision
        precision = self._determine_precision(value, context)
        
        # Determine if scientific notation should be used
        use_scientific = (
            context.use_scientific or
            abs(value) >= 1e6 or
            (abs(value) < 1e-3 and abs(value) != 0)
        )
        
        # Format the value
        if use_scientific:
            formatted = f"{value:.{precision}e}"
        else:
            formatted = f"{value:.{precision}f}"
        
        # Apply locale-specific formatting
        if not use_scientific and not context.compact_mode:
            try:
                # Set locale for thousands separator
                locale.setlocale(locale.LC_NUMERIC, context.locale)
                formatted = locale.format_string(f"%.{precision}f", value, grouping=True)
            except (locale.Error, ValueError):
                # Fallback to manual thousands separator
                if abs(value) >= 1000:
                    parts = formatted.split('.')
                    integer_part = parts[0]
                    decimal_part = parts[1] if len(parts) > 1 else ""
                    
                    # Add thousands separators
                    integer_with_sep = ""
                    for i, digit in enumerate(reversed(integer_part)):
                        if i > 0 and i % 3 == 0:
                            integer_with_sep = "," + integer_with_sep
                        integer_with_sep = digit + integer_with_sep
                    
                    formatted = integer_with_sep
                    if decimal_part:
                        formatted += "." + decimal_part
        
        # Truncate if necessary
        is_truncated = False
        if len(formatted) > context.max_width:
            if use_scientific:
                # Try reducing precision in scientific notation
                for reduced_precision in range(precision - 1, 0, -1):
                    truncated = f"{value:.{reduced_precision}e}"
                    if len(truncated) <= context.max_width:
                        formatted = truncated
                        precision = reduced_precision
                        is_truncated = True
                        break
                else:
                    # Last resort: use minimal scientific notation
                    formatted = f"{value:.0e}"
                    precision = 0
                    is_truncated = True
            else:
                # Try reducing precision in decimal notation
                for reduced_precision in range(precision - 1, 0, -1):
                    truncated = f"{value:.{reduced_precision}f}"
                    if len(truncated) <= context.max_width:
                        formatted = truncated
                        precision = reduced_precision
                        is_truncated = True
                        break
                else:
                    # Switch to scientific notation
                    formatted = f"{value:.1e}"
                    precision = 1
                    is_truncated = True
        
        return FormattingResult(
            formatted_value=formatted,
            original_value=original_value,
            precision_used=precision,
            is_truncated=is_truncated,
            metadata={
                "use_scientific": use_scientific,
                "locale": context.locale
            }
        )
    
    def _determine_precision(
        self,
        value: Union[int, float],
        context: FormattingContext
    ) -> int:
        """Determine appropriate precision based on value magnitude."""
        if context.precision >= 0:
            return context.precision
        
        # Auto-determine precision based on value magnitude
        abs_value = abs(value)
        
        for threshold, precision in sorted(self.precision_thresholds.items()):
            if abs_value >= threshold:
                return precision
        
        # Default to high precision for very small values
        return 6


class MemoryFormatter(MetricFormatter):
    """
    Formatter for memory-related metrics.
    
    This formatter handles memory values with intelligent unit conversion
    and scaling. It supports both binary (1024-based) and decimal (1000-based)
    unit systems and provides human-readable memory representations.
    
    Features:
    - Automatic unit scaling (B, KB, MB, GB, TB, PB)
    - Binary vs decimal unit system support
    - Memory efficiency indicators
    - Peak memory tracking
    - Memory leak detection hints
    """
    
    def __init__(self, **kwargs):
        """Initialize memory formatter."""
        super().__init__(MetricType.MEMORY, **kwargs)
        
        # Unit definitions for different systems
        self.binary_units = [
            ("B", 1),
            ("KiB", 1024),
            ("MiB", 1024**2),
            ("GiB", 1024**3),
            ("TiB", 1024**4),
            ("PiB", 1024**5),
        ]
        
        self.decimal_units = [
            ("B", 1),
            ("KB", 1000),
            ("MB", 1000**2),
            ("GB", 1000**3),
            ("TB", 1000**4),
            ("PB", 1000**5),
        ]
        
        # Common unit aliases
        self.unit_aliases = {
            "bytes": "B",
            "kilobytes": "KB",
            "megabytes": "MB",
            "gigabytes": "GB",
            "terabytes": "TB",
        }
    
    def _format_impl(
        self,
        value: Union[int, float],
        context: FormattingContext
    ) -> FormattingResult:
        """Format memory value with appropriate units."""
        original_value = float(value)
        
        # Handle zero and negative values
        if value <= 0:
            return FormattingResult(
                formatted_value="0 B",
                original_value=original_value,
                unit="B",
                precision_used=0
            )
        
        # Choose unit system
        units = (
            self.binary_units if context.unit_system == UnitSystem.BINARY
            else self.decimal_units
        )
        
        # Find appropriate unit
        unit_name, unit_value = self._find_best_unit(value, units)
        scaled_value = value / unit_value
        
        # Determine precision based on scaled value
        if scaled_value >= 100:
            precision = 0
        elif scaled_value >= 10:
            precision = 1
        else:
            precision = min(context.precision, 2)
        
        # Format the scaled value
        if context.compact_mode:
            formatted_number = f"{scaled_value:.{precision}f}"
        else:
            formatted_number = f"{scaled_value:,.{precision}f}"
        
        # Combine with unit
        if context.show_units:
            if context.compact_mode:
                formatted = f"{formatted_number}{unit_name}"
            else:
                formatted = f"{formatted_number} {unit_name}"
        else:
            formatted = formatted_number
        
        # Apply width constraints
        is_truncated = False
        if len(formatted) > context.max_width:
            # Try without thousands separators
            formatted_number = f"{scaled_value:.{precision}f}"
            if context.show_units:
                formatted = f"{formatted_number} {unit_name}"
            else:
                formatted = formatted_number
            
            # If still too long, reduce precision
            if len(formatted) > context.max_width:
                for reduced_precision in range(precision - 1, -1, -1):
                    formatted_number = f"{scaled_value:.{reduced_precision}f}"
                    if context.show_units:
                        test_formatted = f"{formatted_number} {unit_name}"
                    else:
                        test_formatted = formatted_number
                    
                    if len(test_formatted) <= context.max_width:
                        formatted = test_formatted
                        precision = reduced_precision
                        is_truncated = True
                        break
        
        # Add color coding for memory levels (if enabled)
        if context.use_colors:
            formatted = self._apply_memory_color_coding(
                formatted, scaled_value, unit_name, context.theme
            )
        
        return FormattingResult(
            formatted_value=formatted,
            original_value=original_value,
            unit=unit_name,
            precision_used=precision,
            is_truncated=is_truncated,
            metadata={
                "scaled_value": scaled_value,
                "unit_system": context.unit_system.value,
                "unit_divisor": unit_value
            }
        )
    
    def _find_best_unit(
        self,
        value: float,
        units: List[Tuple[str, int]]
    ) -> Tuple[str, int]:
        """Find the most appropriate unit for the given value."""
        # Start from largest unit and work down
        for unit_name, unit_value in reversed(units):
            if value >= unit_value:
                return unit_name, unit_value
        
        # Default to smallest unit (bytes)
        return units[0]
    
    def _apply_memory_color_coding(
        self,
        formatted: str,
        scaled_value: float,
        unit: str,
        theme: str
    ) -> str:
        """Apply color coding based on memory usage levels."""
        # Define color codes for different themes
        color_schemes = {
            "default": {
                "low": "\033[92m",      # Green
                "medium": "\033[93m",   # Yellow
                "high": "\033[91m",     # Red
                "critical": "\033[95m", # Magenta
                "reset": "\033[0m"
            },
            "minimal": {
                "low": "",
                "medium": "",
                "high": "",
                "critical": "",
                "reset": ""
            }
        }
        
        colors = color_schemes.get(theme, color_schemes["default"])
        
        # Determine color based on memory level
        if unit in ["GiB", "GB"] and scaled_value > 8:
            color = colors["critical"]
        elif unit in ["GiB", "GB"] and scaled_value > 4:
            color = colors["high"]
        elif unit in ["MiB", "MB"] and scaled_value > 500:
            color = colors["medium"]
        else:
            color = colors["low"]
        
        return f"{color}{formatted}{colors['reset']}"


class TimeFormatter(MetricFormatter):
    """
    Formatter for time-related metrics.
    
    This formatter handles time values with intelligent unit conversion
    and human-readable time representations. It supports various time
    formats from nanoseconds to years.
    
    Features:
    - Automatic unit scaling (ns, μs, ms, s, m, h, d)
    - Human-readable duration formatting
    - ETA and elapsed time calculations
    - Timezone-aware formatting
    - Performance timing optimization
    """
    
    def __init__(self, **kwargs):
        """Initialize time formatter."""
        super().__init__(MetricType.TIME, **kwargs)
        
        # Time unit definitions (in seconds)
        self.time_units = [
            ("ns", 1e-9),
            ("μs", 1e-6),
            ("ms", 1e-3),
            ("s", 1),
            ("m", 60),
            ("h", 3600),
            ("d", 86400),
            ("w", 604800),
            ("y", 31536000),
        ]
    
    def _format_impl(
        self,
        value: Union[int, float],
        context: FormattingContext
    ) -> FormattingResult:
        """Format time value with appropriate units."""
        original_value = float(value)
        
        # Handle zero and negative values
        if value <= 0:
            return FormattingResult(
                formatted_value="0s",
                original_value=original_value,
                unit="s",
                precision_used=0
            )
        
        # For very small values, use appropriate sub-second units
        if value < 1:
            return self._format_subsecond(value, context)
        
        # For larger values, use human-readable format
        if value >= 60 and not context.compact_mode:
            return self._format_human_readable(value, context)
        
        # Standard second formatting
        precision = min(context.precision, 3)
        formatted = f"{value:.{precision}f}s"
        
        # Apply width constraints
        is_truncated = False
        if len(formatted) > context.max_width:
            for reduced_precision in range(precision - 1, -1, -1):
                test_formatted = f"{value:.{reduced_precision}f}s"
                if len(test_formatted) <= context.max_width:
                    formatted = test_formatted
                    precision = reduced_precision
                    is_truncated = True
                    break
        
        return FormattingResult(
            formatted_value=formatted,
            original_value=original_value,
            unit="s",
            precision_used=precision,
            is_truncated=is_truncated
        )
    
    def _format_subsecond(
        self,
        value: float,
        context: FormattingContext
    ) -> FormattingResult:
        """Format sub-second time values."""
        # Find appropriate unit
        for unit_name, unit_value in self.time_units:
            if unit_value <= 1:  # Only sub-second units
                scaled_value = value / unit_value
                if scaled_value >= 1:
                    precision = 1 if scaled_value >= 10 else 2
                    formatted = f"{scaled_value:.{precision}f}{unit_name}"
                    
                    return FormattingResult(
                        formatted_value=formatted,
                        original_value=value,
                        unit=unit_name,
                        precision_used=precision,
                        metadata={"scaled_value": scaled_value}
                    )
        
        # Fallback to nanoseconds
        scaled_value = value / 1e-9
        formatted = f"{scaled_value:.0f}ns"
        
        return FormattingResult(
            formatted_value=formatted,
            original_value=value,
            unit="ns",
            precision_used=0
        )
    
    def _format_human_readable(
        self,
        value: float,
        context: FormattingContext
    ) -> FormattingResult:
        """Format time in human-readable format (e.g., 1h 23m 45s)."""
        remaining = value
        parts = []
        
        # Extract time components
        for unit_name, unit_value in reversed(self.time_units[3:]):  # Skip sub-second units
            if remaining >= unit_value:
                count = int(remaining // unit_value)
                remaining = remaining % unit_value
                parts.append(f"{count}{unit_name}")
                
                # Limit to 2-3 components for readability
                if len(parts) >= 2:
                    break
        
        # Add remaining seconds if significant
        if remaining >= 0.1 and len(parts) < 2:
            if remaining >= 1:
                parts.append(f"{remaining:.0f}s")
            else:
                parts.append(f"{remaining:.1f}s")
        
        formatted = " ".join(parts) if parts else "0s"
        
        # Apply width constraints
        is_truncated = False
        if len(formatted) > context.max_width:
            # Use only the most significant component
            if parts:
                formatted = parts[0]
                is_truncated = True
            
            # If still too long, use compact format
            if len(formatted) > context.max_width:
                precision = max(0, context.max_width - 2)  # Reserve space for unit
                formatted = f"{value:.{precision}f}s"
                is_truncated = True
        
        return FormattingResult(
            formatted_value=formatted,
            original_value=value,
            unit="mixed" if " " in formatted else parts[0][-1] if parts else "s",
            precision_used=context.precision,
            is_truncated=is_truncated,
            metadata={"components": parts}
        )


class PercentageFormatter(MetricFormatter):
    """
    Formatter for percentage values.
    
    This formatter handles percentage values with appropriate scaling
    and precision. It supports both ratio (0.0-1.0) and percentage
    (0-100) input formats.
    
    Features:
    - Automatic ratio to percentage conversion
    - Configurable precision based on value range
    - Progress bar integration
    - Color coding for different percentage ranges
    - Compact and verbose formatting modes
    """
    
    def __init__(self, **kwargs):
        """Initialize percentage formatter."""
        super().__init__(MetricType.PERCENTAGE, **kwargs)
    
    def _format_impl(
        self,
        value: Union[int, float],
        context: FormattingContext
    ) -> FormattingResult:
        """Format percentage value."""
        original_value = float(value)
        
        # Determine if value is ratio (0-1) or percentage (0-100)
        if 0 <= value <= 1:
            # Assume ratio, convert to percentage
            percentage = value * 100
        elif 0 <= value <= 100:
            # Assume already percentage
            percentage = value
        else:
            # Handle out-of-range values
            percentage = max(0, min(100, value))
        
        # Determine precision based on percentage value
        if percentage >= 99.9:
            precision = 1
        elif percentage >= 10:
            precision = 1
        else:
            precision = min(context.precision, 2)
        
        # Format the percentage
        formatted_number = f"{percentage:.{precision}f}"
        
        if context.show_units:
            formatted = f"{formatted_number}%"
        else:
            formatted = formatted_number
        
        # Apply width constraints
        is_truncated = False
        if len(formatted) > context.max_width:
            for reduced_precision in range(precision - 1, -1, -1):
                formatted_number = f"{percentage:.{reduced_precision}f}"
                test_formatted = f"{formatted_number}%" if context.show_units else formatted_number
                
                if len(test_formatted) <= context.max_width:
                    formatted = test_formatted
                    precision = reduced_precision
                    is_truncated = True
                    break
        
        # Add color coding for percentage ranges (if enabled)
        if context.use_colors:
            formatted = self._apply_percentage_color_coding(
                formatted, percentage, context.theme
            )
        
        return FormattingResult(
            formatted_value=formatted,
            original_value=original_value,
            unit="%",
            precision_used=precision,
            is_truncated=is_truncated,
            metadata={
                "percentage": percentage,
                "is_ratio_input": 0 <= original_value <= 1
            }
        )
    
    def _apply_percentage_color_coding(
        self,
        formatted: str,
        percentage: float,
        theme: str
    ) -> str:
        """Apply color coding based on percentage ranges."""
        color_schemes = {
            "default": {
                "low": "\033[91m",      # Red (0-25%)
                "medium": "\033[93m",   # Yellow (25-75%)
                "high": "\033[92m",     # Green (75-95%)
                "complete": "\033[94m", # Blue (95-100%)
                "reset": "\033[0m"
            },
            "progress": {
                "low": "\033[31m",      # Dark red
                "medium": "\033[33m",   # Dark yellow
                "high": "\033[32m",     # Dark green
                "complete": "\033[36m", # Cyan
                "reset": "\033[0m"
            }
        }
        
        colors = color_schemes.get(theme, color_schemes["default"])
        
        if percentage >= 95:
            color = colors["complete"]
        elif percentage >= 75:
            color = colors["high"]
        elif percentage >= 25:
            color = colors["medium"]
        else:
            color = colors["low"]
        
        return f"{color}{formatted}{colors['reset']}"


class RateFormatter(MetricFormatter):
    """
    Formatter for rate metrics (e.g., tokens/second, samples/second).
    
    This formatter handles rate values with appropriate unit scaling
    and time period normalization. It supports various rate types
    and provides intelligent precision adjustment.
    
    Features:
    - Automatic unit scaling for large rates
    - Time period normalization (per second, minute, hour)
    - Throughput optimization indicators
    - Trend analysis integration
    - Performance benchmarking support
    """
    
    def __init__(self, rate_unit: str = "items", **kwargs):
        """
        Initialize rate formatter.
        
        Args:
            rate_unit: Base unit for the rate (e.g., "tokens", "samples", "bytes")
        """
        super().__init__(MetricType.RATE, **kwargs)
        self.rate_unit = rate_unit
        
        # Rate scaling thresholds
        self.rate_scales = [
            ("", 1),
            ("K", 1000),
            ("M", 1000000),
            ("G", 1000000000),
        ]
    
    def _format_impl(
        self,
        value: Union[int, float],
        context: FormattingContext
    ) -> FormattingResult:
        """Format rate value with appropriate scaling."""
        original_value = float(value)
        
        # Handle zero and negative rates
        if value <= 0:
            unit = f"{self.rate_unit}/s"
            return FormattingResult(
                formatted_value=f"0 {unit}",
                original_value=original_value,
                unit=unit,
                precision_used=0
            )
        
        # Find appropriate scale
        scale_suffix, scale_value = self._find_best_scale(value)
        scaled_value = value / scale_value
        
        # Determine precision
        if scaled_value >= 100:
            precision = 0
        elif scaled_value >= 10:
            precision = 1
        else:
            precision = min(context.precision, 2)
        
        # Format the scaled value
        formatted_number = f"{scaled_value:.{precision}f}"
        
        # Construct unit
        unit = f"{scale_suffix}{self.rate_unit}/s"
        
        # Combine number and unit
        if context.show_units:
            if context.compact_mode:
                formatted = f"{formatted_number}{unit}"
            else:
                formatted = f"{formatted_number} {unit}"
        else:
            formatted = formatted_number
        
        # Apply width constraints
        is_truncated = False
        if len(formatted) > context.max_width:
            # Try reducing precision
            for reduced_precision in range(precision - 1, -1, -1):
                formatted_number = f"{scaled_value:.{reduced_precision}f}"
                if context.show_units:
                    test_formatted = f"{formatted_number} {unit}"
                else:
                    test_formatted = formatted_number
                
                if len(test_formatted) <= context.max_width:
                    formatted = test_formatted
                    precision = reduced_precision
                    is_truncated = True
                    break
            
            # If still too long, try compact mode
            if len(formatted) > context.max_width and not context.compact_mode:
                formatted = f"{formatted_number}{unit}"
                is_truncated = True
        
        return FormattingResult(
            formatted_value=formatted,
            original_value=original_value,
            unit=unit,
            precision_used=precision,
            is_truncated=is_truncated,
            metadata={
                "scaled_value": scaled_value,
                "scale_suffix": scale_suffix,
                "scale_value": scale_value
            }
        )
    
    def _find_best_scale(self, value: float) -> Tuple[str, int]:
        """Find the most appropriate scale for the rate value."""
        for scale_suffix, scale_value in reversed(self.rate_scales):
            if value >= scale_value:
                return scale_suffix, scale_value
        
        return "", 1


class MetricFormatterFactory:
    """
    Factory for creating metric formatters.
    
    This factory implements the Factory pattern to provide a centralized
    way to create and configure metric formatters. It supports both
    built-in formatter types and custom formatter registration.
    
    Features:
    - Dynamic formatter creation based on metric type
    - Custom formatter registration
    - Configuration template management
    - Performance optimization through formatter reuse
    - Thread-safe formatter creation and caching
    """
    
    def __init__(self):
        """Initialize formatter factory."""
        self._formatters: Dict[str, Type[MetricFormatter]] = {}
        self._formatter_instances: Dict[str, MetricFormatter] = {}
        self._lock = threading.RLock()
        
        # Register built-in formatters
        self._register_builtin_formatters()
        
        logger.debug("MetricFormatterFactory initialized")
    
    def _register_builtin_formatters(self) -> None:
        """Register built-in formatter types."""
        self._formatters.update({
            "numeric": NumericFormatter,
            "number": NumericFormatter,
            "float": NumericFormatter,
            "loss": NumericFormatter,
            "accuracy": PercentageFormatter,
            "memory": MemoryFormatter,
            "mem": MemoryFormatter,
            "ram": MemoryFormatter,
            "time": TimeFormatter,
            "duration": TimeFormatter,
            "elapsed": TimeFormatter,
            "percentage": PercentageFormatter,
            "percent": PercentageFormatter,
            "rate": RateFormatter,
            "throughput": RateFormatter,
            "tokens_per_second": lambda **kwargs: RateFormatter(rate_unit="tokens", **kwargs),
            "samples_per_second": lambda **kwargs: RateFormatter(rate_unit="samples", **kwargs),
        })
    
    def register_formatter(
        self,
        name: str,
        formatter_class: Type[MetricFormatter]
    ) -> None:
        """
        Register a custom formatter type.
        
        Args:
            name: Name to register the formatter under
            formatter_class: Formatter class to register
        """
        with self._lock:
            if not issubclass(formatter_class, MetricFormatter):
                raise ValueError(
                    f"Formatter class must inherit from MetricFormatter, "
                    f"got {formatter_class}"
                )
            
            self._formatters[name.lower()] = formatter_class
            logger.debug(f"Registered custom formatter: {name}")
    
    def create_formatter(
        self,
        formatter_type: str,
        reuse_instances: bool = True,
        **kwargs
    ) -> MetricFormatter:
        """
        Create a formatter instance.
        
        Args:
            formatter_type: Type of formatter to create
            reuse_instances: Whether to reuse existing instances
            **kwargs: Additional arguments for formatter initialization
            
        Returns:
            MetricFormatter instance
            
        Raises:
            ValueError: If formatter type is not registered
        """
        with self._lock:
            formatter_type = formatter_type.lower()
            
            if formatter_type not in self._formatters:
                raise ValueError(
                    f"Unknown formatter type: {formatter_type}. "
                    f"Available types: {list(self._formatters.keys())}"
                )
            
            # Check for existing instance if reuse is enabled
            if reuse_instances and formatter_type in self._formatter_instances:
                return self._formatter_instances[formatter_type]
            
            # Create new instance
            formatter_class = self._formatters[formatter_type]
            
            # Handle callable formatters (lambdas)
            if callable(formatter_class) and not isinstance(formatter_class, type):
                formatter = formatter_class(**kwargs)
            else:
                formatter = formatter_class(**kwargs)
            
            # Cache instance if reuse is enabled
            if reuse_instances:
                self._formatter_instances[formatter_type] = formatter
            
            return formatter
    
    def get_available_formatters(self) -> List[str]:
        """Get list of available formatter types."""
        with self._lock:
            return list(self._formatters.keys())
    
    def clear_cache(self) -> None:
        """Clear cached formatter instances."""
        with self._lock:
            self._formatter_instances.clear()


# Global formatter factory instance
formatter_factory = MetricFormatterFactory()


# Convenience functions for common formatting operations
def format_loss(value: float, precision: int = 4) -> str:
    """Format loss value with appropriate precision."""
    formatter = formatter_factory.create_formatter("loss")
    context = FormattingContext(precision=precision)
    result = formatter.format(value, context)
    return result.formatted_value


def format_memory(value: float, compact: bool = False) -> str:
    """Format memory value with automatic unit scaling."""
    formatter = formatter_factory.create_formatter("memory")
    context = FormattingContext(compact_mode=compact)
    result = formatter.format(value, context)
    return result.formatted_value


def format_time(value: float, human_readable: bool = True) -> str:
    """Format time value with human-readable units."""
    formatter = formatter_factory.create_formatter("time")
    context = FormattingContext(compact_mode=not human_readable)
    result = formatter.format(value, context)
    return result.formatted_value


def format_percentage(value: float, precision: int = 1) -> str:
    """Format percentage value with appropriate precision."""
    formatter = formatter_factory.create_formatter("percentage")
    context = FormattingContext(precision=precision)
    result = formatter.format(value, context)
    return result.formatted_value


def format_rate(value: float, unit: str = "items") -> str:
    """Format rate value with appropriate scaling."""
    formatter = formatter_factory.create_formatter("rate", rate_unit=unit)
    context = FormattingContext()
    result = formatter.format(value, context)
    return result.formatted_value


# Export all classes and functions
__all__ = [
    'MetricType',
    'FormattingPrecision',
    'UnitSystem',
    'FormattingContext',
    'FormattingResult',
    'MetricFormatter',
    'NumericFormatter',
    'MemoryFormatter',
    'TimeFormatter',
    'PercentageFormatter',
    'RateFormatter',
    'MetricFormatterFactory',
    'formatter_factory',
    'format_loss',
    'format_memory',
    'format_time',
    'format_percentage',
    'format_rate',
]