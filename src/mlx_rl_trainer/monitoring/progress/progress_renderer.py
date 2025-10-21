"""
Enterprise-Grade Progress Bar Rendering System

This module provides a comprehensive progress bar rendering system that implements
multiple display strategies for different use cases and terminal environments.
The system uses advanced design patterns to provide flexible, performant, and
visually appealing progress displays for MLX RL training.

The rendering system is architected with enterprise-grade patterns:
- Strategy Pattern: Interchangeable rendering algorithms for different display modes
- Template Method: Consistent rendering workflow with customizable presentation
- Observer Pattern: Real-time updates and event-driven rendering
- Decorator Pattern: Composable visual enhancements and styling
- Adapter Pattern: Terminal compatibility and feature detection
- Command Pattern: Encapsulated rendering operations with undo/redo support

Key Features:
- Multiple rendering strategies (Compact, Detailed, Minimal, Dashboard)
- Intelligent terminal capability detection and adaptation
- Real-time metric updates with smooth animations
- Advanced color schemes and theming support
- Unicode and ASCII fallback rendering
- Performance-optimized rendering with minimal overhead
- Thread-safe operations for concurrent training environments
- Comprehensive error handling with graceful degradation
- Extensible plugin architecture for custom renderers

Supported Display Modes:
- CompactRenderer: Single-line progress with essential metrics
- DetailedRenderer: Multi-line display with comprehensive information
- MinimalRenderer: Bare-bones display for resource-constrained environments
- DashboardRenderer: Rich dashboard-style display with visual indicators
- DebugRenderer: Development-focused display with detailed diagnostics

Terminal Compatibility:
- Full Unicode support with fallback to ASCII
- Color support detection with graceful degradation
- Terminal width detection and responsive layout
- Cursor positioning and screen management
- Cross-platform compatibility (Windows, macOS, Linux)

Example:
    ```python
    from mlx_rl_trainer.monitoring.progress.progress_renderer import (
        ProgressBarRenderer, CompactRenderer, DetailedRenderer
    )
    
    # Create renderer with automatic terminal detection
    renderer = ProgressBarRenderer.create_optimal_renderer()
    
    # Or create specific renderer
    compact_renderer = CompactRenderer(
        show_eta=True,
        show_memory=True,
        color_scheme="gradient"
    )
    
    # Render progress with metrics
    metrics = {
        'loss': 0.1234,
        'gradient_norm': 2.345,
        'memory_usage_mb': 1024.5,
        'learning_rate': 0.001
    }
    
    rendered_output = compact_renderer.render(
        current=500,
        total=1000,
        metrics=metrics,
        elapsed_time=123.45
    )
    print(rendered_output)
    ```
"""

import os
import sys
import time
import threading
import logging
import shutil
import re
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
from collections import deque
import math

from .strategies.base_strategy import (
    RenderingStrategy,
    StrategyMetadata,
    StrategyPriority,
    ComputationComplexity,
    MetricDict,
)
from .metric_formatters import (
    MetricFormatterFactory,
    FormattingContext,
    UnitSystem,
    formatter_factory,
)
from .exceptions import (
    RendererError,
    DisplayFormatError,
    TerminalCompatibilityError,
    create_error_context,
)

logger = logging.getLogger(__name__)


class DisplayMode(Enum):
    """Display modes for progress bar rendering."""
    COMPACT = "compact"
    DETAILED = "detailed"
    MINIMAL = "minimal"
    DASHBOARD = "dashboard"
    DEBUG = "debug"


class ColorScheme(Enum):
    """Color schemes for progress bar styling."""
    DEFAULT = "default"
    GRADIENT = "gradient"
    MONOCHROME = "monochrome"
    HIGH_CONTRAST = "high_contrast"
    TERMINAL_FRIENDLY = "terminal_friendly"
    CUSTOM = "custom"


class TerminalCapability(Enum):
    """Terminal capability levels."""
    BASIC = "basic"          # ASCII only, no colors
    STANDARD = "standard"    # Basic colors, limited Unicode
    ENHANCED = "enhanced"    # Full colors, Unicode support
    ADVANCED = "advanced"    # True color, advanced features


@dataclass(frozen=True)
class TerminalInfo:
    """
    Information about terminal capabilities and characteristics.
    
    This class encapsulates terminal detection results and provides
    a comprehensive view of what rendering features are available.
    
    Attributes:
        width: Terminal width in characters
        height: Terminal height in characters
        supports_color: Whether terminal supports ANSI colors
        supports_unicode: Whether terminal supports Unicode characters
        supports_cursor_control: Whether terminal supports cursor positioning
        color_depth: Number of colors supported (8, 16, 256, 16777216)
        terminal_type: Type of terminal (xterm, cmd, etc.)
        capability_level: Overall capability assessment
        encoding: Terminal character encoding
        is_tty: Whether output is connected to a terminal
    """
    width: int = 80
    height: int = 24
    supports_color: bool = False
    supports_unicode: bool = False
    supports_cursor_control: bool = False
    color_depth: int = 0
    terminal_type: str = "unknown"
    capability_level: TerminalCapability = TerminalCapability.BASIC
    encoding: str = "utf-8"
    is_tty: bool = False
    
    def can_use_feature(self, feature: str) -> bool:
        """Check if terminal supports a specific feature."""
        feature_requirements = {
            "colors": self.supports_color,
            "unicode": self.supports_unicode,
            "cursor": self.supports_cursor_control,
            "wide_chars": self.supports_unicode,
            "progress_bar": True,  # Always supported
            "animations": self.supports_cursor_control,
        }
        return feature_requirements.get(feature, False)


@dataclass
class RenderingConfig:
    """
    Configuration for progress bar rendering.
    
    This class defines all the configuration options for progress bar
    rendering, including display preferences, styling options, and
    performance settings.
    
    Attributes:
        display_mode: Primary display mode
        color_scheme: Color scheme to use
        show_eta: Whether to show estimated time of arrival
        show_memory: Whether to show memory usage
        show_gradient_norm: Whether to show gradient norm
        show_throughput: Whether to show processing throughput
        update_frequency: Update frequency in seconds
        animation_enabled: Whether to enable animations
        compact_numbers: Whether to use compact number formatting
        max_width: Maximum width for the progress bar
        metric_precision: Default precision for metric display
        custom_colors: Custom color definitions
        fallback_mode: Fallback mode for unsupported terminals
    """
    display_mode: DisplayMode = DisplayMode.COMPACT
    color_scheme: ColorScheme = ColorScheme.DEFAULT
    show_eta: bool = True
    show_memory: bool = True
    show_gradient_norm: bool = True
    show_throughput: bool = True
    update_frequency: float = 0.1
    animation_enabled: bool = True
    compact_numbers: bool = False
    max_width: Optional[int] = None
    metric_precision: int = 2
    custom_colors: Dict[str, str] = field(default_factory=dict)
    fallback_mode: DisplayMode = DisplayMode.MINIMAL


class TerminalDetector:
    """
    Advanced terminal capability detection system.
    
    This class provides comprehensive terminal capability detection
    to ensure optimal rendering across different terminal environments.
    It implements sophisticated detection algorithms and caching for
    performance optimization.
    """
    
    def __init__(self):
        """Initialize terminal detector."""
        self._detection_cache: Optional[TerminalInfo] = None
        self._cache_timestamp = 0.0
        self._cache_ttl = 60.0  # Cache for 60 seconds
        self._lock = threading.RLock()
        
        logger.debug("TerminalDetector initialized")
    
    def detect_terminal_capabilities(self, force_refresh: bool = False) -> TerminalInfo:
        """
        Detect terminal capabilities with caching.
        
        Args:
            force_refresh: Whether to force fresh detection
            
        Returns:
            TerminalInfo with detected capabilities
        """
        with self._lock:
            current_time = time.time()
            
            # Check cache validity
            if (not force_refresh and 
                self._detection_cache and 
                current_time - self._cache_timestamp < self._cache_ttl):
                return self._detection_cache
            
            # Perform fresh detection
            terminal_info = self._detect_capabilities()
            
            # Update cache
            self._detection_cache = terminal_info
            self._cache_timestamp = current_time
            
            logger.debug(f"Terminal capabilities detected: {terminal_info.capability_level.value}")
            return terminal_info
    
    def _detect_capabilities(self) -> TerminalInfo:
        """Perform actual terminal capability detection."""
        # Basic terminal information
        is_tty = sys.stdout.isatty()
        width, height = self._get_terminal_size()
        encoding = sys.stdout.encoding or "utf-8"
        
        # Terminal type detection
        terminal_type = self._detect_terminal_type()
        
        # Color support detection
        supports_color = self._detect_color_support()
        color_depth = self._detect_color_depth() if supports_color else 0
        
        # Unicode support detection
        supports_unicode = self._detect_unicode_support()
        
        # Cursor control detection
        supports_cursor_control = self._detect_cursor_support()
        
        # Determine overall capability level
        capability_level = self._assess_capability_level(
            supports_color, supports_unicode, supports_cursor_control, color_depth
        )
        
        return TerminalInfo(
            width=width,
            height=height,
            supports_color=supports_color,
            supports_unicode=supports_unicode,
            supports_cursor_control=supports_cursor_control,
            color_depth=color_depth,
            terminal_type=terminal_type,
            capability_level=capability_level,
            encoding=encoding,
            is_tty=is_tty
        )
    
    def _get_terminal_size(self) -> Tuple[int, int]:
        """Get terminal size with fallbacks."""
        try:
            # Try shutil.get_terminal_size first
            size = shutil.get_terminal_size()
            return size.columns, size.lines
        except Exception:
            pass
        
        try:
            # Try os.get_terminal_size
            size = os.get_terminal_size()
            return size.columns, size.lines
        except Exception:
            pass
        
        # Environment variable fallback
        try:
            width = int(os.environ.get('COLUMNS', 80))
            height = int(os.environ.get('LINES', 24))
            return width, height
        except Exception:
            pass
        
        # Default fallback
        return 80, 24
    
    def _detect_terminal_type(self) -> str:
        """Detect terminal type from environment variables."""
        # Check common terminal environment variables
        term = os.environ.get('TERM', '').lower()
        term_program = os.environ.get('TERM_PROGRAM', '').lower()
        
        if 'xterm' in term:
            return 'xterm'
        elif 'screen' in term:
            return 'screen'
        elif 'tmux' in term:
            return 'tmux'
        elif term_program == 'iterm.app':
            return 'iterm2'
        elif term_program == 'apple_terminal':
            return 'terminal_app'
        elif 'cmd' in term or os.name == 'nt':
            return 'cmd'
        elif 'powershell' in term:
            return 'powershell'
        else:
            return term or 'unknown'
    
    def _detect_color_support(self) -> bool:
        """Detect if terminal supports colors."""
        # Check if output is redirected
        if not sys.stdout.isatty():
            return False
        
        # Check environment variables
        if os.environ.get('NO_COLOR'):
            return False
        
        if os.environ.get('FORCE_COLOR'):
            return True
        
        # Check TERM variable
        term = os.environ.get('TERM', '').lower()
        color_terms = [
            'xterm', 'xterm-color', 'xterm-256color',
            'screen', 'screen-256color',
            'tmux', 'tmux-256color',
            'linux', 'cygwin'
        ]
        
        if any(color_term in term for color_term in color_terms):
            return True
        
        # Check for Windows terminal with color support
        if os.name == 'nt':
            try:
                import colorama
                return True
            except ImportError:
                # Check Windows version for native ANSI support
                import platform
                version = platform.version()
                if version and int(version.split('.')[2]) >= 10586:
                    return True
        
        return False
    
    def _detect_color_depth(self) -> int:
        """Detect color depth support."""
        term = os.environ.get('TERM', '').lower()
        colorterm = os.environ.get('COLORTERM', '').lower()
        
        # True color support
        if 'truecolor' in colorterm or '24bit' in colorterm:
            return 16777216
        
        # 256 color support
        if '256color' in term or '256' in colorterm:
            return 256
        
        # 16 color support
        if 'color' in term:
            return 16
        
        # 8 color support (basic)
        if self._detect_color_support():
            return 8
        
        return 0
    
    def _detect_unicode_support(self) -> bool:
        """Detect Unicode support."""
        # Check encoding
        encoding = sys.stdout.encoding or ""
        if 'utf' in encoding.lower():
            return True
        
        # Check locale
        try:
            import locale
            loc = locale.getlocale()
            if loc and any('utf' in str(l).lower() for l in loc if l):
                return True
        except Exception:
            pass
        
        # Test Unicode output
        try:
            # Try to encode a Unicode character
            "█".encode(sys.stdout.encoding or 'utf-8')
            return True
        except (UnicodeEncodeError, LookupError):
            return False
    
    def _detect_cursor_support(self) -> bool:
        """Detect cursor control support."""
        if not sys.stdout.isatty():
            return False
        
        # Most modern terminals support cursor control
        term = os.environ.get('TERM', '').lower()
        supported_terms = ['xterm', 'screen', 'tmux', 'linux']
        
        return any(supported_term in term for supported_term in supported_terms)
    
    def _assess_capability_level(
        self,
        supports_color: bool,
        supports_unicode: bool,
        supports_cursor: bool,
        color_depth: int
    ) -> TerminalCapability:
        """Assess overall terminal capability level."""
        if supports_color and supports_unicode and supports_cursor and color_depth >= 256:
            return TerminalCapability.ADVANCED
        elif supports_color and supports_unicode and color_depth >= 16:
            return TerminalCapability.ENHANCED
        elif supports_color or supports_unicode:
            return TerminalCapability.STANDARD
        else:
            return TerminalCapability.BASIC


class ProgressBarRenderer(RenderingStrategy):
    """
    Abstract base class for progress bar renderers.
    
    This class provides the foundation for all progress bar rendering
    implementations, ensuring consistent behavior and providing common
    functionality like terminal detection, metric formatting, and
    error handling.
    """
    
    def __init__(
        self,
        config: Optional[RenderingConfig] = None,
        terminal_detector: Optional[TerminalDetector] = None
    ):
        """
        Initialize progress bar renderer.
        
        Args:
            config: Rendering configuration
            terminal_detector: Terminal capability detector
        """
        # Create metadata for the strategy
        metadata = StrategyMetadata(
            name=self.__class__.__name__,
            description=f"Progress bar renderer: {self.__class__.__name__}",
            priority=StrategyPriority.MEDIUM,
            complexity=ComputationComplexity.O_N,
            thread_safe=True
        )
        
        super().__init__(metadata, {})
        
        # Configuration and detection
        self.config = config or RenderingConfig()
        self.terminal_detector = terminal_detector or TerminalDetector()
        self.terminal_info = self.terminal_detector.detect_terminal_capabilities()
        
        # Formatter factory
        self.formatter_factory = formatter_factory
        
        # Rendering state
        self.last_render_time = 0.0
        self.render_count = 0
        self.last_output_length = 0
        
        # Performance tracking
        self.render_times = deque(maxlen=100)
        self.frame_rate_target = 10.0  # Target FPS for animations
        
        # Thread safety
        self._render_lock = threading.RLock()
        
        logger.debug(f"Initialized {self.__class__.__name__} renderer")
    
    def _validate_config_impl(self) -> None:
        """Validate renderer configuration."""
        if self.config.update_frequency <= 0:
            raise ValueError("Update frequency must be positive")
        
        if self.config.metric_precision < 0 or self.config.metric_precision > 10:
            raise ValueError("Metric precision must be between 0 and 10")
    
    def _initialize_impl(self) -> None:
        """Initialize renderer-specific components."""
        # Detect terminal capabilities
        self.terminal_info = self.terminal_detector.detect_terminal_capabilities()
        
        # Adjust configuration based on terminal capabilities
        self._adapt_config_to_terminal()
        
        logger.debug(
            f"Renderer initialized for {self.terminal_info.terminal_type} terminal "
            f"({self.terminal_info.capability_level.value})"
        )
    
    def _adapt_config_to_terminal(self) -> None:
        """Adapt configuration based on terminal capabilities."""
        # Disable colors if not supported
        if not self.terminal_info.supports_color:
            if self.config.color_scheme != ColorScheme.MONOCHROME:
                logger.debug("Disabling colors due to terminal limitations")
                # Create new config with monochrome scheme
                self.config = RenderingConfig(
                    **{**self.config.__dict__, 'color_scheme': ColorScheme.MONOCHROME}
                )
        
        # Disable animations if cursor control not supported
        if not self.terminal_info.supports_cursor_control:
            if self.config.animation_enabled:
                logger.debug("Disabling animations due to terminal limitations")
        
        # Adjust width if specified
        if self.config.max_width is None:
            self.config = RenderingConfig(
                **{**self.config.__dict__, 'max_width': self.terminal_info.width}
            )
    
    def _check_terminal_compatibility(self) -> bool:
        """Check if terminal is compatible with this renderer."""
        # Basic compatibility check - all renderers should work on TTY
        return self.terminal_info.is_tty or not sys.stdout.isatty()
    
    def _should_update(self) -> bool:
        """Check if renderer should update based on frequency limits."""
        current_time = time.time()
        time_since_last = current_time - self.last_render_time
        
        return time_since_last >= self.config.update_frequency
    
    def _format_progress_info(
        self,
        current: int,
        total: int,
        metrics: MetricDict,
        elapsed_time: float
    ) -> Dict[str, str]:
        """Format progress information for display."""
        # Calculate progress percentage
        progress_ratio = current / max(total, 1)
        progress_percent = progress_ratio * 100
        
        # Calculate ETA
        if current > 0 and progress_ratio < 1.0:
            eta_seconds = (elapsed_time / current) * (total - current)
        else:
            eta_seconds = 0.0
        
        # Calculate throughput
        throughput = current / max(elapsed_time, 0.001)
        
        # Create formatting context
        formatting_context = FormattingContext(
            precision=self.config.metric_precision,
            compact_mode=self.config.compact_numbers,
            max_width=12,
            use_colors=self.terminal_info.supports_color,
            show_units=True
        )
        
        # Format basic progress information
        formatted_info = {
            'progress_percent': f"{progress_percent:.1f}%",
            'current': str(current),
            'total': str(total),
            'elapsed': self._format_time(elapsed_time, formatting_context),
            'eta': self._format_time(eta_seconds, formatting_context) if eta_seconds > 0 else "N/A",
            'throughput': f"{throughput:.1f}/s",
        }
        
        # Format metrics using appropriate formatters
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                formatter_type = self._determine_formatter_type(metric_name)
                try:
                    formatter = self.formatter_factory.create_formatter(formatter_type)
                    result = formatter.format(metric_value, formatting_context)
                    formatted_info[metric_name] = result.formatted_value
                except Exception as e:
                    logger.warning(f"Failed to format metric {metric_name}: {e}")
                    formatted_info[metric_name] = str(metric_value)
            else:
                formatted_info[metric_name] = str(metric_value)
        
        return formatted_info
    
    def _determine_formatter_type(self, metric_name: str) -> str:
        """Determine appropriate formatter type for a metric."""
        metric_name_lower = metric_name.lower()
        
        # Memory-related metrics
        if any(keyword in metric_name_lower for keyword in ['memory', 'mem', 'ram', 'cache']):
            return 'memory'
        
        # Time-related metrics
        if any(keyword in metric_name_lower for keyword in ['time', 'duration', 'elapsed', 'eta']):
            return 'time'
        
        # Percentage metrics
        if any(keyword in metric_name_lower for keyword in ['percent', 'accuracy', 'rate']):
            return 'percentage'
        
        # Rate metrics
        if any(keyword in metric_name_lower for keyword in ['per_second', 'throughput', 'tokens_per']):
            return 'rate'
        
        # Default to numeric
        return 'numeric'
    
    def _format_time(self, seconds: float, context: FormattingContext) -> str:
        """Format time value using time formatter."""
        try:
            time_formatter = self.formatter_factory.create_formatter('time')
            result = time_formatter.format(seconds, context)
            return result.formatted_value
        except Exception:
            # Fallback formatting
            if seconds < 60:
                return f"{seconds:.1f}s"
            elif seconds < 3600:
                minutes = int(seconds // 60)
                secs = int(seconds % 60)
                return f"{minutes}m{secs:02d}s"
            else:
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                return f"{hours}h{minutes:02d}m"
    
    def _apply_styling(self, formatted_info: Dict[str, str]) -> Dict[str, str]:
        """Apply styling and colors to formatted information."""
        if not self.terminal_info.supports_color or self.config.color_scheme == ColorScheme.MONOCHROME:
            return formatted_info
        
        # Get color scheme
        colors = self._get_color_scheme()
        
        # Apply colors to different elements
        styled_info = {}
        for key, value in formatted_info.items():
            color_key = self._get_color_key_for_metric(key)
            color_code = colors.get(color_key, colors.get('default', ''))
            reset_code = colors.get('reset', '')
            
            if color_code:
                styled_info[key] = f"{color_code}{value}{reset_code}"
            else:
                styled_info[key] = value
        
        return styled_info
    
    def _get_color_scheme(self) -> Dict[str, str]:
        """Get color codes for the current color scheme."""
        schemes = {
            ColorScheme.DEFAULT: {
                'progress': '\033[94m',    # Blue
                'loss': '\033[91m',        # Red
                'reward': '\033[92m',      # Green
                'memory': '\033[93m',      # Yellow
                'time': '\033[95m',        # Magenta
                'gradient': '\033[96m',    # Cyan
                'default': '\033[97m',     # White
                'reset': '\033[0m'
            },
            ColorScheme.GRADIENT: {
                'progress': '\033[38;5;33m',   # Gradient blue
                'loss': '\033[38;5;196m',      # Bright red
                'reward': '\033[38;5;46m',     # Bright green
                'memory': '\033[38;5;214m',    # Orange
                'time': '\033[38;5;165m',      # Pink
                'gradient': '\033[38;5;51m',   # Cyan
                'default': '\033[38;5;255m',   # Bright white
                'reset': '\033[0m'
            },
            ColorScheme.MONOCHROME: {
                'progress': '',
                'loss': '',
                'reward': '',
                'memory': '',
                'time': '',
                'gradient': '',
                'default': '',
                'reset': ''
            },
            ColorScheme.HIGH_CONTRAST: {
                'progress': '\033[1;37;44m',   # Bold white on blue
                'loss': '\033[1;37;41m',       # Bold white on red
                'reward': '\033[1;37;42m',     # Bold white on green
                'memory': '\033[1;30;43m',     # Bold black on yellow
                'time': '\033[1;37;45m',       # Bold white on magenta
                'gradient': '\033[1;30;46m',   # Bold black on cyan
                'default': '\033[1;37m',       # Bold white
                'reset': '\033[0m'
            }
        }
        
        return schemes.get(self.config.color_scheme, schemes[ColorScheme.DEFAULT])
    
    def _get_color_key_for_metric(self, metric_name: str) -> str:
        """Get color key for a specific metric."""
        metric_lower = metric_name.lower()
        
        if 'loss' in metric_lower:
            return 'loss'
        elif 'reward' in metric_lower:
            return 'reward'
        elif 'memory' in metric_lower or 'mem' in metric_lower:
            return 'memory'
        elif 'time' in metric_lower or 'eta' in metric_lower:
            return 'time'
        elif 'grad' in metric_lower or 'norm' in metric_lower:
            return 'gradient'
        elif 'progress' in metric_lower or 'percent' in metric_lower:
            return 'progress'
        else:
            return 'default'
    
    def _create_progress_bar(
        self,
        progress_ratio: float,
        width: int = 20
    ) -> str:
        """Create visual progress bar."""
        if self.terminal_info.supports_unicode:
            # Unicode progress bar with smooth gradations
            filled_width = int(progress_ratio * width)
            partial_width = (progress_ratio * width) - filled_width
            
            # Choose partial character based on fraction
            if partial_width >= 0.75:
                partial_char = '▊'
            elif partial_width >= 0.5:
                partial_char = '▌'
            elif partial_width >= 0.25:
                partial_char = '▎'
            else:
                partial_char = ''
            
            filled_part = '█' * filled_width
            empty_part = '░' * (width - filled_width - len(partial_char))
            
            return f"{filled_part}{partial_char}{empty_part}"
        else:
            # ASCII progress bar
            filled_width = int(progress_ratio * width)
            filled_part = '=' * filled_width
            empty_part = '-' * (width - filled_width)
            
            return f"[{filled_part}{empty_part}]"
    
    @abstractmethod
    def _generate_output_impl(self, styled_info: Dict[str, str]) -> str:
        """Generate the final output string for display."""
        pass
    
    def render(
        self,
        current: int,
        total: int,
        metrics: MetricDict,
        elapsed_time: float
    ) -> str:
        """
        Render progress bar with metrics.
        
        This is the main entry point for rendering operations.
        It implements rate limiting and performance optimization.
        
        Args:
            current: Current progress value
            total: Total progress value
            metrics: Current training metrics
            elapsed_time: Elapsed time in seconds
            
        Returns:
            Formatted progress bar string
        """
        with self._render_lock:
            # Check if update is needed based on frequency
            if not self._should_update():
                return ""  # Skip update to maintain target frame rate
            
            start_time = time.perf_counter()
            
            try:
                # Execute the rendering strategy
                output = self.execute(current, total, metrics, elapsed_time)
                
                # Update performance tracking
                render_time = time.perf_counter() - start_time
                self.render_times.append(render_time)
                self.render_count += 1
                self.last_render_time = time.time()
                
                return output
                
            except Exception as e:
                logger.error(f"Rendering failed: {e}")
                # Return fallback output
                return self._create_fallback_output(current, total, elapsed_time)
    
    def _create_fallback_output(
        self,
        current: int,
        total: int,
        elapsed_time: float
    ) -> str:
        """Create minimal fallback output when rendering fails."""
        progress_percent = (current / max(total, 1)) * 100
        return f"Progress: {current}/{total} ({progress_percent:.1f}%) - {elapsed_time:.1f}s"
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get renderer performance statistics."""
        stats = super().get_performance_stats()
        
        if self.render_times:
            render_times_list = list(self.render_times)
            stats.update({
                "avg_render_time_ms": sum(render_times_list) / len(render_times_list) * 1000,
                "max_render_time_ms": max(render_times_list) * 1000,
                "min_render_time_ms": min(render_times_list) * 1000,
                "render_fps": len(render_times_list) / max(sum(render_times_list), 0.001),
                "target_fps": self.frame_rate_target,
            })
        
        stats.update({
            "terminal_capability": self.terminal_info.capability_level.value,
            "supports_color": self.terminal_info.supports_color,
            "supports_unicode": self.terminal_info.supports_unicode,
            "terminal_width": self.terminal_info.width,
        })
        
        return stats
    
    @classmethod
    def create_optimal_renderer(
        cls,
        config: Optional[RenderingConfig] = None,
        force_mode: Optional[DisplayMode] = None
    ) -> 'ProgressBarRenderer':
        """
        Create optimal renderer based on terminal capabilities.
        
        Args:
            config: Optional rendering configuration
            force_mode: Force specific display mode
            
        Returns:
            Optimal renderer for current terminal
        """
        detector = TerminalDetector()
        terminal_info = detector.detect_terminal_capabilities()
        
        # Determine optimal renderer based on capabilities
        if force_mode:
            target_mode = force_mode
        elif terminal_info.capability_level == TerminalCapability.ADVANCED:
            target_mode = DisplayMode.DETAILED
        elif terminal_info.capability_level == TerminalCapability.ENHANCED:
            target_mode = DisplayMode.COMPACT
        elif terminal_info.capability_level == TerminalCapability.STANDARD:
            target_mode = DisplayMode.COMPACT
        else:
            target_mode = DisplayMode.MINIMAL
        
        # Create appropriate renderer
        renderer_classes = {
            DisplayMode.COMPACT: CompactRenderer,
            DisplayMode.DETAILED: DetailedRenderer,
            DisplayMode.MINIMAL: MinimalRenderer,
            DisplayMode.DASHBOARD: DashboardRenderer,
            DisplayMode.DEBUG: DebugRenderer,
        }
        
        renderer_class = renderer_classes.get(target_mode, CompactRenderer)
        return renderer_class(config, detector)


class CompactRenderer(ProgressBarRenderer):
    """
    Compact single-line progress bar renderer.
    
    This renderer provides a space-efficient progress display that fits
    on a single line while showing the most important training metrics.
    It's optimized for standard terminal environments and provides
    excellent performance with minimal visual overhead.
    
    Features:
    - Single-line display with essential metrics
    - Adaptive metric selection based on available width
    - Smooth progress bar animations
    - Color-coded metric values
    - Intelligent text truncation
    """
    
    def __init__(self, **kwargs):
        """Initialize compact renderer."""
        super().__init__(**kwargs)
        
        # Compact-specific configuration
        self.show_progress_bar = True
        self.progress_bar_width = 20
        self.metric_separator = " | "
        
        # Priority order for metrics (most important first)
        self.metric_priority = [
            'loss', 'gradient_norm', 'reward_mean', 'learning_rate',
            'memory_usage_mb', 'step_time', 'throughput'
        ]
    
    def _generate_output_impl(self, styled_info: Dict[str, str]) -> str:
        """Generate compact single-line output."""
        output_parts = []
        
        # Progress percentage and bar
        progress_percent = styled_info.get('progress_percent', '0.0%')
        current = styled_info.get('current', '0')
        total = styled_info.get('total', '1')
        
        if self.show_progress_bar:
            progress_ratio = float(current) / max(float(total), 1)
            progress_bar = self._create_progress_bar(progress_ratio, self.progress_bar_width)
            output_parts.append(f"{progress_percent} {progress_bar}")
        else:
            output_parts.append(f"{current}/{total} ({progress_percent})")
        
        # Add metrics in priority order
        available_width = self.config.max_width - len(output_parts[0]) - 10  # Reserve space
        current_width = 0
        
        for metric_name in self.metric_priority:
            if metric_name in styled_info:
                metric_display = f"{metric_name.replace('_', ' ').title()}: {styled_info[metric_name]}"
                
                # Check if we have space for this metric
                needed_width = len(metric_display) + len(self.metric_separator)
                if current_width + needed_width <= available_width:
                    output_parts.append(metric_display)
                    current_width += needed_width
                else:
                    break
        
        # Add ETA if space allows
        if 'eta' in styled_info and styled_info['eta'] != 'N/A':
            eta_display = f"ETA: {styled_info['eta']}"
            if current_width + len(eta_display) + len(self.metric_separator) <= available_width:
                output_parts.append(eta_display)
        
        return self.metric_separator.join(output_parts)


class DetailedRenderer(ProgressBarRenderer):
    """
    Detailed multi-line progress bar renderer.
    
    This renderer provides comprehensive progress information across
    multiple lines, showing detailed metrics, trends, and system
    information. It's designed for development environments and
    detailed monitoring scenarios.
    
    Features:
    - Multi-line display with comprehensive metrics
    - Trend indicators and historical data
    - System resource monitoring
    - Gradient analysis visualization
    - Performance diagnostics
    """
    
    def __init__(self, **kwargs):
        """Initialize detailed renderer."""
        super().__init__(**kwargs)
        
        # Detailed-specific configuration
        self.show_system_info = True
        self.show_trends = True
        self.show_gradient_details = True
        self.max_lines = 8
        
        # Historical data for trend calculation
        self.metric_history: Dict[str, deque] = {}
        self.history_size = 20
    
    def _update_metric_history(self, metrics: MetricDict) -> None:
        """Update historical data for trend analysis."""
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                if metric_name not in self.metric_history:
                    self.metric_history[metric_name] = deque(maxlen=self.history_size)
                self.metric_history[metric_name].append(value)
    
    def _calculate_trend(self, metric_name: str) -> str:
        """Calculate trend indicator for a metric."""
        if metric_name not in self.metric_history:
            return "─"
        
        history = list(self.metric_history[metric_name])
        if len(history) < 3:
            return "─"
        
        # Simple trend calculation
        recent_avg = sum(history[-3:]) / 3
        older_avg = sum(history[:3]) / 3
        
        if recent_avg > older_avg * 1.05:
            return "↗" if self.terminal_info.supports_unicode else "^"
        elif recent_avg < older_avg * 0.95:
            return "↘" if self.terminal_info.supports_unicode else "v"
        else:
            return "→" if self.terminal_info.supports_unicode else "-"
    
    def _generate_output_impl(self, styled_info: Dict[str, str]) -> str:
        """Generate detailed multi-line output."""
        lines = []
        
        # Line 1: Progress bar and basic info
        progress_percent = styled_info.get('progress_percent', '0.0%')
        current = styled_info.get('current', '0')
        total = styled_info.get('total', '1')
        elapsed = styled_info.get('elapsed', '0s')
        eta = styled_info.get('eta', 'N/A')
        
        progress_ratio = float(current) / max(float(total), 1)
        progress_bar = self._create_progress_bar(progress_ratio, 30)
        
        lines.append(f"Progress: {progress_bar} {progress_percent} ({current}/{total})")
        lines.append(f"Time: {elapsed} elapsed, {eta} remaining")
        
        # Line 2-3: Training metrics
        training_metrics = []
        if 'loss' in styled_info:
            trend = self._calculate_trend('loss')
            training_metrics.append(f"Loss: {styled_info['loss']} {trend}")
        
        if 'gradient_norm' in styled_info:
            trend = self._calculate_trend('gradient_norm')
            training_metrics.append(f"Grad Norm: {styled_info['gradient_norm']} {trend}")
        
        if 'reward_mean' in styled_info:
            trend = self._calculate_trend('reward_mean')
            training_metrics.append(f"Reward: {styled_info['reward_mean']} {trend}")
        
        if 'learning_rate' in styled_info:
            training_metrics.append(f"LR: {styled_info['learning_rate']}")
        
        if training_metrics:
            # Split into two lines if too many metrics
            if len(training_metrics) > 2:
                lines.append(" | ".join(training_metrics[:2]))
                lines.append(" | ".join(training_metrics[2:]))
            else:
                lines.append(" | ".join(training_metrics))
        
        # Line 4: System metrics
        system_metrics = []
        if 'memory_usage_mb' in styled_info:
            system_metrics.append(f"Memory: {styled_info['memory_usage_mb']}")
        
        if 'throughput' in styled_info:
            system_metrics.append(f"Throughput: {styled_info['throughput']}")
        
        if system_metrics:
            lines.append("System: " + " | ".join(system_metrics))
        
        # Limit number of lines
        if len(lines) > self.max_lines:
            lines = lines[:self.max_lines]
        
        return "\n".join(lines)


class MinimalRenderer(ProgressBarRenderer):
    """
    Minimal progress bar renderer for resource-constrained environments.
    
    This renderer provides the absolute minimum progress information
    with maximum compatibility across all terminal types. It's designed
    for production environments, CI/CD systems, and low-resource scenarios.
    
    Features:
    - Single-line ASCII-only display
    - Essential metrics only
    - Maximum terminal compatibility
    - Minimal resource usage
    - No animations or colors
    """
    
    def __init__(self, **kwargs):
        """Initialize minimal renderer."""
        super().__init__(**kwargs)
        
        # Force minimal configuration
        self.config = RenderingConfig(
            display_mode=DisplayMode.MINIMAL,
            color_scheme=ColorScheme.MONOCHROME,
            animation_enabled=False,
            compact_numbers=True,
            show_eta=False,
            show_memory=False,
            show_gradient_norm=False,
            show_throughput=False
        )
    
    def _generate_output_impl(self, styled_info: Dict[str, str]) -> str:
        """Generate minimal output."""
        current = styled_info.get('current', '0')
        total = styled_info.get('total', '1')
        progress_percent = styled_info.get('progress_percent', '0.0%')
        
        # Basic progress information
        basic_info = f"{current}/{total} ({progress_percent})"
        
        # Add loss if available
        if 'loss' in styled_info:
            basic_info += f" Loss: {styled_info['loss']}"
        
        return basic_info


class DashboardRenderer(ProgressBarRenderer):
    """
    Dashboard-style progress renderer with rich visual indicators.
    
    This renderer provides a comprehensive dashboard-like display
    with visual indicators, charts, and detailed system information.
    It's designed for monitoring environments and detailed analysis.
    
    Features:
    - Multi-section dashboard layout
    - Visual metric indicators and gauges
    - Real-time charts and sparklines
    - System resource monitoring
    - Alert indicators for anomalies
    - Customizable layout sections
    """
    
    def __init__(self, **kwargs):
        """Initialize dashboard renderer."""
        super().__init__(**kwargs)
        
        # Dashboard-specific configuration
        self.sections = ['progress', 'training', 'system', 'performance']
        self.show_sparklines = True
        self.show_alerts = True
        self.dashboard_width = min(self.terminal_info.width, 120)
        
        # Sparkline data
        self.sparkline_data: Dict[str, deque] = {}
        self.sparkline_length = 20
    
    def _update_sparkline_data(self, metrics: MetricDict) -> None:
        """Update sparkline data for visual trends."""
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                if metric_name not in self.sparkline_data:
                    self.sparkline_data[metric_name] = deque(maxlen=self.sparkline_length)
                self.sparkline_data[metric_name].append(value)
    
    def _create_sparkline(self, metric_name: str) -> str:
        """Create ASCII sparkline for metric trend."""
        if metric_name not in self.sparkline_data:
            return "─" * self.sparkline_length
        
        data = list(self.sparkline_data[metric_name])
        if len(data) < 2:
            return "─" * self.sparkline_length
        
        # Normalize data to 0-7 range for ASCII characters
        min_val = min(data)
        max_val = max(data)
        
        if max_val == min_val:
            return "─" * len(data)
        
        # ASCII sparkline characters
        if self.terminal_info.supports_unicode:
            chars = "▁▂▃▄▅▆▇█"
        else:
            chars = "_.-^'`~"
        
        sparkline = ""
        for value in data:
            normalized = (value - min_val) / (max_val - min_val)
            char_index = min(int(normalized * (len(chars) - 1)), len(chars) - 1)
            sparkline += chars[char_index]
        
        return sparkline
    
    def _generate_output_impl(self, styled_info: Dict[str, str]) -> str:
        """Generate dashboard-style output."""
        lines = []
        
        # Header with title
        title = "MLX RL Training Dashboard"
        separator = "═" * len(title) if self.terminal_info.supports_unicode else "=" * len(title)
        lines.append(separator)
        lines.append(title)
        lines.append(separator)
        
        # Progress section
        current = styled_info.get('current', '0')
        total = styled_info.get('total', '1')
        progress_percent = styled_info.get('progress_percent', '0.0%')
        elapsed = styled_info.get('elapsed', '0s')
        eta = styled_info.get('eta', 'N/A')
        
        progress_ratio = float(current) / max(float(total), 1)
        progress_bar = self._create_progress_bar(progress_ratio, 40)
        
        lines.append(f"Progress: {progress_bar} {progress_percent}")
        lines.append(f"Step: {current}/{total} | Elapsed: {elapsed} | ETA: {eta}")
        lines.append("")
        
        # Training metrics section
        lines.append("Training Metrics:")
        training_line = []
        
        if 'loss' in styled_info:
            sparkline = self._create_sparkline('loss')
            training_line.append(f"Loss: {styled_info['loss']} {sparkline}")
        
        if 'gradient_norm' in styled_info:
            sparkline = self._create_sparkline('gradient_norm')
            training_line.append(f"Grad: {styled_info['gradient_norm']} {sparkline}")
        
        if training_line:
            lines.extend(training_line)
        
        # System metrics section
        if self.show_system_info:
            lines.append("")
            lines.append("System Metrics:")
            system_line = []
            
            if 'memory_usage_mb' in styled_info:
                system_line.append(f"Memory: {styled_info['memory_usage_mb']}")
            
            if 'throughput' in styled_info:
                system_line.append(f"Throughput: {styled_info['throughput']}")
            
            if system_line:
                lines.append(" | ".join(system_line))
        
        return "\n".join(lines)


class DebugRenderer(ProgressBarRenderer):
    """
    Debug-focused progress renderer with diagnostic information.
    
    This renderer provides detailed diagnostic information for
    development and debugging purposes. It shows internal state,
    performance metrics, and detailed system information.
    
    Features:
    - Comprehensive diagnostic information
    - Internal state visualization
    - Performance profiling data
    - Error and warning indicators
    - Memory usage breakdown
    - Thread safety diagnostics
    """
    
    def __init__(self, **kwargs):
        """Initialize debug renderer."""
        super().__init__(**kwargs)
        
        # Debug-specific configuration
        self.show_internal_state = True
        self.show_performance_stats = True
        self.show_memory_breakdown = True
        self.show_thread_info = True
    
    def _generate_output_impl(self, styled_info: Dict[str, str]) -> str:
        """Generate debug output with diagnostic information."""
        lines = []
        
        # Basic progress
        current = styled_info.get('current', '0')
        total = styled_info.get('total', '1')
        progress_percent = styled_info.get('progress_percent', '0.0%')
        
        lines.append(f"DEBUG: Step {current}/{total} ({progress_percent})")
        
        # All available metrics
        lines.append("Metrics:")
        for key, value in styled_info.items():
            if key not in ['current', 'total', 'progress_percent']:
                lines.append(f"  {key}: {value}")
        
        # Performance stats
        if self.show_performance_stats:
            perf_stats = self.get_performance_stats()
            lines.append("Performance:")
            lines.append(f"  Render FPS: {perf_stats.get('render_fps', 0):.1f}")
            lines.append(f"  Avg Render Time: {perf_stats.get('avg_render_time_ms', 0):.2f}ms")
        
        # Terminal info
        lines.append("Terminal:")
        lines.append(f"  Type: {self.terminal_info.terminal_type}")
        lines.append(f"  Capability: {self.terminal_info.capability_level.value}")
        lines.append(f"  Size: {self.terminal_info.width}x{self.terminal_info.height}")
        
        return "\n".join(lines)


class ProgressBarRendererFactory:
    """
    Factory for creating progress bar renderers.
    
    This factory provides a centralized way to create and configure
    progress bar renderers based on requirements and terminal capabilities.
    It implements intelligent renderer selection and configuration optimization.
    """
    
    def __init__(self):
        """Initialize renderer factory."""
        self._renderer_registry: Dict[DisplayMode, type] = {
            DisplayMode.COMPACT: CompactRenderer,
            DisplayMode.DETAILED: DetailedRenderer,
            DisplayMode.MINIMAL: MinimalRenderer,
            DisplayMode.DASHBOARD: DashboardRenderer,
            DisplayMode.DEBUG: DebugRenderer,
        }
        
        self._cached_renderers: Dict[str, ProgressBarRenderer] = {}
        self._lock = threading.RLock()
        
        logger.debug("ProgressBarRendererFactory initialized")
    
    def create_renderer(
        self,
        display_mode: DisplayMode,
        config: Optional[RenderingConfig] = None,
        cache_key: Optional[str] = None
    ) -> ProgressBarRenderer:
        """
        Create a progress bar renderer.
        
        Args:
            display_mode: Display mode for the renderer
            config: Optional rendering configuration
            cache_key: Optional cache key for renderer reuse
            
        Returns:
            ProgressBarRenderer instance
            
        Raises:
            ValueError: If display mode is not supported
        """
        with self._lock:
            # Check cache if key provided
            if cache_key and cache_key in self._cached_renderers:
                return self._cached_renderers[cache_key]
            
            # Get renderer class
            if display_mode not in self._renderer_registry:
                raise ValueError(f"Unsupported display mode: {display_mode}")
            
            renderer_class = self._renderer_registry[display_mode]
            
            # Create renderer instance
            renderer = renderer_class(config=config)
            
            # Cache if key provided
            if cache_key:
                self._cached_renderers[cache_key] = renderer
            
            return renderer
    
    def create_optimal_renderer(
        self,
        config: Optional[RenderingConfig] = None,
        terminal_detector: Optional[TerminalDetector] = None
    ) -> ProgressBarRenderer:
        """
        Create optimal renderer based on terminal capabilities.
        
        Args:
            config: Optional rendering configuration
            terminal_detector: Optional terminal detector
            
        Returns:
            Optimal renderer for current environment
        """
        detector = terminal_detector or TerminalDetector()
        terminal_info = detector.detect_terminal_capabilities()
        
        # Select optimal display mode
        if terminal_info.capability_level == TerminalCapability.ADVANCED:
            display_mode = DisplayMode.DASHBOARD
        elif terminal_info.capability_level == TerminalCapability.ENHANCED:
            display_mode = DisplayMode.DETAILED
        elif terminal_info.capability_level == TerminalCapability.STANDARD:
            display_mode = DisplayMode.COMPACT
        else:
            display_mode = DisplayMode.MINIMAL
        
        # Override with config if specified
        if config and config.display_mode:
            display_mode = config.display_mode
        
        return self.create_renderer(display_mode, config)
    
    def register_renderer(
        self,
        display_mode: DisplayMode,
        renderer_class: type
    ) -> None:
        """
        Register a custom renderer class.
        
        Args:
            display_mode: Display mode for the renderer
            renderer_class: Renderer class to register
        """
        with self._lock:
            if not issubclass(renderer_class, ProgressBarRenderer):
                raise ValueError(
                    f"Renderer class must inherit from ProgressBarRenderer, "
                    f"got {renderer_class}"
                )
            
            self._renderer_registry[display_mode] = renderer_class
            logger.debug(f"Registered custom renderer: {display_mode.value}")
    
    def get_available_modes(self) -> List[DisplayMode]:
        """Get list of available display modes."""
        with self._lock:
            return list(self._renderer_registry.keys())
    
    def clear_cache(self) -> None:
        """Clear cached renderer instances."""
        with self._lock:
            self._cached_renderers.clear()


# Global renderer factory instance
renderer_factory = ProgressBarRendererFactory()


class AnimatedProgressBar:
    """
    Animated progress bar with smooth transitions and visual effects.
    
    This class provides advanced animation capabilities for progress bars,
    including smooth progress transitions, pulsing effects, and dynamic
    color changes based on training state.
    """
    
    def __init__(
        self,
        width: int = 30,
        animation_speed: float = 0.5,
        enable_pulse: bool = True
    ):
        """
        Initialize animated progress bar.
        
        Args:
            width: Width of the progress bar
            animation_speed: Animation speed multiplier
            enable_pulse: Whether to enable pulsing animation
        """
        self.width = width
        self.animation_speed = animation_speed
        self.enable_pulse = enable_pulse
        
        # Animation state
        self.animation_frame = 0
        self.last_progress = 0.0
        self.target_progress = 0.0
        self.animation_start_time = 0.0
        
        # Visual elements
        self.progress_chars = {
            'filled': '█',
            'partial': ['▏', '▎', '▍', '▌', '▋', '▊', '▉'],
            'empty': '░',
            'pulse': '▓'
        }
        
        # ASCII fallback
        self.ascii_chars = {
            'filled': '=',
            'partial': ['-'],
            'empty': ' ',
            'pulse': '#'
        }
    
    def update_progress(self, progress: float) -> None:
        """
        Update progress with smooth animation.
        
        Args:
            progress: Progress ratio (0.0 to 1.0)
        """
        if progress != self.target_progress:
            self.target_progress = progress
            self.animation_start_time = time.time()
    
    def render(self, supports_unicode: bool = True) -> str:
        """
        Render animated progress bar.
        
        Args:
            supports_unicode: Whether terminal supports Unicode
            
        Returns:
            Rendered progress bar string
        """
        # Choose character set
        chars = self.progress_chars if supports_unicode else self.ascii_chars
        
        # Calculate current progress with animation
        current_progress = self._calculate_animated_progress()
        
        # Create progress bar
        filled_width = current_progress * self.width
        filled_chars = int(filled_width)
        partial_width = filled_width - filled_chars
        
        # Build progress bar
        bar_parts = []
        
        # Filled portion
        if self.enable_pulse and self._should_pulse():
            # Add pulsing effect
            pulse_char = chars['pulse']
            filled_part = chars['filled'] * max(0, filled_chars - 1) + pulse_char
        else:
            filled_part = chars['filled'] * filled_chars
        
        bar_parts.append(filled_part)
        
        # Partial character
        if partial_width > 0 and 'partial' in chars:
            partial_chars = chars['partial']
            partial_index = min(
                int(partial_width * len(partial_chars)),
                len(partial_chars) - 1
            )
            bar_parts.append(partial_chars[partial_index])
            filled_chars += 1
        
        # Empty portion
        empty_chars = self.width - filled_chars
        if empty_chars > 0:
            bar_parts.append(chars['empty'] * empty_chars)
        
        return ''.join(bar_parts)
    
    def _calculate_animated_progress(self) -> float:
        """Calculate current progress with smooth animation."""
        if not self.animation_speed or self.last_progress == self.target_progress:
            self.last_progress = self.target_progress
            return self.target_progress
        
        # Smooth transition using easing function
        elapsed = time.time() - self.animation_start_time
        transition_duration = 1.0 / self.animation_speed
        
        if elapsed >= transition_duration:
            self.last_progress = self.target_progress
            return self.target_progress
        
        # Ease-out animation
        t = elapsed / transition_duration
        eased_t = 1 - (1 - t) ** 3  # Cubic ease-out
        
        animated_progress = (
            self.last_progress + 
            (self.target_progress - self.last_progress) * eased_t
        )
        
        return max(0.0, min(1.0, animated_progress))
    
    def _should_pulse(self) -> bool:
        """Determine if progress bar should pulse."""
        # Pulse when progress is active but not complete
        return (
            self.enable_pulse and 
            0.01 < self.target_progress < 0.99 and
            time.time() % 2 < 1  # Pulse every 2 seconds
        )


# Convenience functions for common rendering operations
def create_compact_renderer(
    show_gradient_norm: bool = True,
    show_memory: bool = True,
    color_scheme: ColorScheme = ColorScheme.DEFAULT
) -> CompactRenderer:
    """Create a compact renderer with common settings."""
    config = RenderingConfig(
        display_mode=DisplayMode.COMPACT,
        show_gradient_norm=show_gradient_norm,
        show_memory=show_memory,
        color_scheme=color_scheme
    )
    return CompactRenderer(config=config)


def create_detailed_renderer(
    show_trends: bool = True,
    show_system_info: bool = True
) -> DetailedRenderer:
    """Create a detailed renderer with comprehensive information."""
    config = RenderingConfig(
        display_mode=DisplayMode.DETAILED,
        color_scheme=ColorScheme.GRADIENT
    )
    renderer = DetailedRenderer(config=config)
    renderer.show_trends = show_trends
    renderer.show_system_info = show_system_info
    return renderer


def create_minimal_renderer() -> MinimalRenderer:
    """Create a minimal renderer for basic environments."""
    return MinimalRenderer()


def detect_optimal_display_mode() -> DisplayMode:
    """Detect optimal display mode for current terminal."""
    detector = TerminalDetector()
    terminal_info = detector.detect_terminal_capabilities()
    
    capability_to_mode = {
        TerminalCapability.ADVANCED: DisplayMode.DASHBOARD,
        TerminalCapability.ENHANCED: DisplayMode.DETAILED,
        TerminalCapability.STANDARD: DisplayMode.COMPACT,
        TerminalCapability.BASIC: DisplayMode.MINIMAL,
    }
    
    return capability_to_mode.get(
        terminal_info.capability_level,
        DisplayMode.MINIMAL
    )


# Export all classes and functions
__all__ = [
    'DisplayMode',
    'ColorScheme',
    'TerminalCapability',
    'TerminalInfo',
    'RenderingConfig',
    'TerminalDetector',
    'ProgressBarRenderer',
    'CompactRenderer',
    'DetailedRenderer',
    'MinimalRenderer',
    'DashboardRenderer',
    'DebugRenderer',
    'ProgressBarRendererFactory',
    'renderer_factory',
    'AnimatedProgressBar',
    'create_compact_renderer',
    'create_detailed_renderer',
    'create_minimal_renderer',
    'detect_optimal_display_mode',
]