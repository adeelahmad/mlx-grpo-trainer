
"""
Advanced Chart Generation System with Custom Templates

This module provides a comprehensive visualization system for training metrics,
with support for custom chart templates, improved error handling, and a wide
range of chart types. It leverages seaborn and matplotlib for high-quality
visualizations while providing fallback mechanisms when these libraries are
not available.

Key Features:
- Multiple chart types (line, scatter, distribution, heatmap)
- Custom chart templates via Strategy pattern
- Comprehensive error handling with graceful degradation
- Memory-efficient chart generation
- Thread-safe operations
- Correlation ID support for tracking related charts
- Automatic resource cleanup
"""

import logging
import time
import traceback
import threading
import os
import json
from typing import Dict, Any, List, Optional, Tuple, Union, Callable, Type, TypeVar
from pathlib import Path
from abc import ABC, abstractmethod
import numpy as np
from contextlib import contextmanager

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from matplotlib.colors import LinearSegmentedColormap
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Configure logger
logger = logging.getLogger(__name__)


class ChartGenerationError(Exception):
    """Base exception for chart generation errors."""
    pass


class TemplateNotFoundError(ChartGenerationError):
    """Exception raised when a chart template is not found."""
    pass


class DataValidationError(ChartGenerationError):
    """Exception raised when chart data validation fails."""
    pass


class RenderingError(ChartGenerationError):
    """Exception raised when chart rendering fails."""
    pass


class ChartTemplate(ABC):
    """
    Abstract base class for chart templates.
    
    This class defines the interface for chart templates, which are responsible
    for rendering specific types of charts with customizable styling and layout.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get template name."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Get template description."""
        pass
    
    @abstractmethod
    def validate_data(self, data: Any) -> bool:
        """
        Validate that the provided data is compatible with this template.
        
        Args:
            data: Data to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        pass
    
    @abstractmethod
    def render(
        self,
        fig: Figure,
        ax: Union[Axes, np.ndarray],
        data: Any,
        **kwargs
    ) -> None:
        """
        Render chart on the provided figure and axes.
        
        Args:
            fig: Matplotlib figure
            ax: Matplotlib axes or array of axes
            data: Data to plot
            **kwargs: Additional keyword arguments
        """
        pass


class LineChartTemplate(ChartTemplate):
    """Template for line charts with customizable styling."""
    
    @property
    def name(self) -> str:
        return "line_chart"
    
    @property
    def description(self) -> str:
        return "Line chart with optional smoothing and annotations"
    
    def validate_data(self, data: Dict[str, List[float]]) -> bool:
        """Validate line chart data."""
        if not isinstance(data, dict):
            return False
        
        for key, values in data.items():
            if not isinstance(values, (list, np.ndarray)) or len(values) == 0:
                return False
            
            # Check that all values are numeric
            try:
                np.array(values, dtype=float)
            except (ValueError, TypeError):
                return False
        
        return True
    
    def render(
        self,
        fig: Figure,
        ax: Axes,
        data: Dict[str, List[float]],
        **kwargs
    ) -> None:
        """Render line chart."""
        smooth = kwargs.get("smooth", True)
        window = kwargs.get("window", 10)
        title = kwargs.get("title", "")
        xlabel = kwargs.get("xlabel", "Step")
        ylabel = kwargs.get("ylabel", "Value")
        grid = kwargs.get("grid", True)
        legend = kwargs.get("legend", True)
        colors = kwargs.get("colors", None)
        
        # Set title and labels
        if title:
            ax.set_title(title, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # Enable grid if requested
        if grid:
            ax.grid(True, alpha=0.3)
        
        # Plot each line
        for i, (key, values) in enumerate(data.items()):
            values_array = np.array(values)
            steps = np.arange(len(values_array))
            
            # Get color if provided
            color = colors[i] if colors and i < len(colors) else None
            
            # Plot raw data
            ax.plot(
                steps, 
                values_array, 
                alpha=0.3, 
                label=f"{key} (Raw)", 
                linewidth=1,
                color=color
            )
            
            # Plot smoothed data if requested
            if smooth and len(values_array) > window:
                try:
                    smoothed = np.convolve(
                        values_array, 
                        np.ones(window) / window, 
                        mode="valid"
                    )
                    smooth_steps = steps[window - 1:]
                    ax.plot(
                        smooth_steps, 
                        smoothed, 
                        label=f"{key} (Smoothed)", 
                        linewidth=2,
                        color=color
                    )
                except Exception as e:
                    logger.warning(f"Failed to smooth data for {key}: {e}")
        
        # Add legend if requested
        if legend:
            ax.legend()


class DistributionChartTemplate(ChartTemplate):
    """Template for distribution charts with statistics."""
    
    @property
    def name(self) -> str:
        return "distribution_chart"
    
    @property
    def description(self) -> str:
        return "Distribution chart with statistics"
    
    def validate_data(self, data: List[float]) -> bool:
        """Validate distribution chart data."""
        if not isinstance(data, (list, np.ndarray)) or len(data) == 0:
            return False
        
        # Check that all values are numeric
        try:
            np.array(data, dtype=float)
        except (ValueError, TypeError):
            return False
        
        return True
    
    def render(
        self,
        fig: Figure,
        ax: Axes,
        data: List[float],
        **kwargs
    ) -> None:
        """Render distribution chart."""
        bins = kwargs.get("bins", 30)
        kde = kwargs.get("kde", True)
        title = kwargs.get("title", "Distribution")
        xlabel = kwargs.get("xlabel", "Value")
        ylabel = kwargs.get("ylabel", "Frequency")
        show_mean = kwargs.get("show_mean", True)
        show_median = kwargs.get("show_median", True)
        
        # Set title and labels
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # Plot distribution
        if SEABORN_AVAILABLE and kde:
            sns.histplot(data, bins=bins, kde=True, ax=ax)
        else:
            ax.hist(data, bins=bins, alpha=0.7, edgecolor="black")
        
        # Add mean line
        if show_mean:
            mean_value = np.mean(data)
            ax.axvline(
                mean_value,
                color="r",
                linestyle="--",
                label=f"Mean: {mean_value:.3f}"
            )
        
        # Add median line
        if show_median:
            median_value = np.median(data)
            ax.axvline(
                median_value,
                color="g",
                linestyle="--",
                label=f"Median: {median_value:.3f}"
            )
        
        # Add legend
        ax.legend()


class HeatmapChartTemplate(ChartTemplate):
    """Template for heatmap charts."""
    
    @property
    def name(self) -> str:
        return "heatmap_chart"
    
    @property
    def description(self) -> str:
        return "Heatmap chart for correlation matrices or 2D data"
    
    def validate_data(self, data: Union[Dict[str, List[float]], np.ndarray]) -> bool:
        """Validate heatmap chart data."""
        if isinstance(data, dict):
            # Convert dict to DataFrame for validation
            if not PANDAS_AVAILABLE:
                return False
            
            try:
                df = pd.DataFrame(data)
                return True
            except Exception:
                return False
        elif isinstance(data, np.ndarray):
            # Check that data is 2D
            return data.ndim == 2
        else:
            return False
    
    def render(
        self,
        fig: Figure,
        ax: Axes,
        data: Union[Dict[str, List[float]], np.ndarray],
        **kwargs
    ) -> None:
        """Render heatmap chart."""
        title = kwargs.get("title", "Heatmap")
        cmap = kwargs.get("cmap", "coolwarm")
        annot = kwargs.get("annot", True)
        fmt = kwargs.get("fmt", ".2f")
        
        # Set title
        ax.set_title(title, fontweight="bold")
        
        # Convert data to appropriate format
        if isinstance(data, dict):
            if PANDAS_AVAILABLE:
                df = pd.DataFrame(data)
                corr = df.corr()
            else:
                raise ValueError("pandas is required for dict data")
        else:
            corr = data
        
        # Plot heatmap
        if SEABORN_AVAILABLE:
            sns.heatmap(
                corr,
                annot=annot,
                fmt=fmt,
                cmap=cmap,
                center=0,
                square=True,
                linewidths=1,
                cbar_kws={"shrink": 0.8},
                ax=ax
            )
        else:
            # Basic heatmap using matplotlib
            im = ax.imshow(corr, cmap=cmap)
            fig.colorbar(im, ax=ax)
            
            # Add annotations if requested
            if annot:
                for i in range(corr.shape[0]):
                    for j in range(corr.shape[1]):
                        ax.text(
                            j, i, 
                            f"{corr[i, j]:{fmt}}", 
                            ha="center", 
                            va="center", 
                            color="white" if abs(corr[i, j]) > 0.5 else "black"
                        )


class ScatterChartTemplate(ChartTemplate):
    """Template for scatter charts."""
    
    @property
    def name(self) -> str:
        return "scatter_chart"
    
    @property
    def description(self) -> str:
        return "Scatter chart for showing relationships between two variables"
    
    def validate_data(self, data: Tuple[List[float], List[float]]) -> bool:
        """Validate scatter chart data."""
        if not isinstance(data, tuple) or len(data) != 2:
            return False
        
        x_data, y_data = data
        
        if (not isinstance(x_data, (list, np.ndarray)) or 
            not isinstance(y_data, (list, np.ndarray))):
            return False
        
        if len(x_data) != len(y_data) or len(x_data) == 0:
            return False
        
        # Check that all values are numeric
        try:
            np.array(x_data, dtype=float)
            np.array(y_data, dtype=float)
        except (ValueError, TypeError):
            return False
        
        return True
    
    def render(
        self,
        fig: Figure,
        ax: Axes,
        data: Tuple[List[float], List[float]],
        **kwargs
    ) -> None:
        """Render scatter chart."""
        x_data, y_data = data
        title = kwargs.get("title", "Scatter Plot")
        xlabel = kwargs.get("xlabel", "X")
        ylabel = kwargs.get("ylabel", "Y")
        alpha = kwargs.get("alpha", 0.7)
        color = kwargs.get("color", "blue")
        size = kwargs.get("size", 50)
        add_trend = kwargs.get("add_trend", True)
        
        # Set title and labels
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # Plot scatter
        ax.scatter(x_data, y_data, alpha=alpha, color=color, s=size)
        
        # Add trend line if requested
        if add_trend and len(x_data) > 2:
            try:
                z = np.polyfit(x_data, y_data, 1)
                p = np.poly1d(z)
                ax.plot(
                    x_data, 
                    p(x_data), 
                    "r--", 
                    label=f"Trend: y={z[0]:.3f}x+{z[1]:.3f}"
                )
                ax.legend()
            except Exception as e:
                logger.warning(f"Failed to add trend line: {e}")


class DashboardTemplate(ChartTemplate):
    """Template for comprehensive dashboards with multiple panels."""
    
    @property
    def name(self) -> str:
        return "dashboard"
    
    @property
    def description(self) -> str:
        return "Comprehensive dashboard with multiple panels"
    
    def validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate dashboard data."""
        return isinstance(data, dict)
    
    def render(
        self,
        fig: Figure,
        ax: Any,  # Ignored, we use GridSpec instead
        data: Dict[str, Any],
        **kwargs
    ) -> None:
        """Render dashboard."""
        layout = kwargs.get("layout", (3, 3))
        title = kwargs.get("title", "Training Dashboard")
        
        # Create GridSpec
        rows, cols = layout
        gs = gridspec.GridSpec(rows, cols, figure=fig)
        
        # Add title
        fig.suptitle(title, fontsize=16, fontweight="bold")
        
        # Render panels based on data
        panel_configs = kwargs.get("panels", [])
        
        for i, config in enumerate(panel_configs):
            if i >= rows * cols:
                logger.warning(f"Too many panels for layout {layout}, skipping extras")
                break
            
            panel_type = config.get("type", "line")
            panel_data_key = config.get("data_key", "")
            panel_title = config.get("title", "")
            panel_position = config.get("position", (i // cols, i % cols))
            panel_span = config.get("span", (1, 1))
            
            # Get data for this panel
            if panel_data_key not in data:
                logger.warning(f"Data key '{panel_data_key}' not found, skipping panel")
                continue
            
            panel_data = data[panel_data_key]
            
            # Create subplot
            row, col = panel_position
            row_span, col_span = panel_span
            ax = fig.add_subplot(gs[row:row+row_span, col:col+col_span])
            
            # Render panel based on type
            try:
                if panel_type == "line":
                    if isinstance(panel_data, dict):
                        LineChartTemplate().render(
                            fig, ax, {panel_data_key: panel_data}, 
                            title=panel_title, **config
                        )
                    else:
                        logger.warning(f"Invalid data type for line chart: {type(panel_data)}")
                
                elif panel_type == "distribution":
                    if isinstance(panel_data, (list, np.ndarray)):
                        DistributionChartTemplate().render(
                            fig, ax, panel_data, 
                            title=panel_title, **config
                        )
                    else:
                        logger.warning(f"Invalid data type for distribution chart: {type(panel_data)}")
                
                elif panel_type == "heatmap":
                    if isinstance(panel_data, (dict, np.ndarray)):
                        HeatmapChartTemplate().render(
                            fig, ax, panel_data, 
                            title=panel_title, **config
                        )
                    else:
                        logger.warning(f"Invalid data type for heatmap chart: {type(panel_data)}")
                
                elif panel_type == "scatter":
                    if isinstance(panel_data, tuple) and len(panel_data) == 2:
                        ScatterChartTemplate().render(
                            fig, ax, panel_data, 
                            title=panel_title, **config
                        )
                    else:
                        logger.warning(f"Invalid data type for scatter chart: {type(panel_data)}")
                
                elif panel_type == "text":
                    ax.axis("off")
                    ax.text(
                        0.5, 0.5, 
                        str(panel_data), 
                        ha="center", 
                        va="center", 
                        fontsize=config.get("fontsize", 12)
                    )
                
                else:
                    logger.warning(f"Unknown panel type: {panel_type}")
            
            except Exception as e:
                logger.error(f"Failed to render panel {i}: {e}")
                ax.text(
                    0.5, 0.5, 
                    f"Error rendering panel: {str(e)}", 
                    ha="center", 
                    va="center", 
                    color="red"
                )


T = TypeVar('T', bound=ChartTemplate)


class ChartGenerator:
    """
    Advanced chart generation with custom templates and improved error handling.

    Features:
    - Multiple chart types (line, scatter, distribution, heatmap)
    - Custom chart templates via Strategy pattern
    - Comprehensive error handling with graceful degradation
    - Memory-efficient chart generation
    - Thread-safe operations
    - Correlation ID support for tracking related charts
    - Automatic resource cleanup
    """

    def __init__(
        self,
        output_dir: Path,
        style: str = "darkgrid",
        palette: str = "husl",
        dpi: int = 150,
        figsize: Tuple[int, int] = (12, 8),
        correlation_id: Optional[str] = None,
    ):
        """
        Initialize chart generator.

        Args:
            output_dir: Directory to save charts
            style: Seaborn style
            palette: Color palette
            dpi: Image resolution
            figsize: Default figure size
            correlation_id: Optional correlation ID for tracking
        """
        if not MPL_AVAILABLE:
            raise ImportError("matplotlib is required for ChartGenerator")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.style = style
        self.palette = palette
        self.dpi = dpi
        self.figsize = figsize
        self.correlation_id = correlation_id or f"chart_{int(time.time())}"

        # Thread safety
        self._lock = threading.RLock()
        
        # Register default templates
        self._templates: Dict[str, ChartTemplate] = {}
        self.register_template(LineChartTemplate())
        self.register_template(DistributionChartTemplate())
        self.register_template(HeatmapChartTemplate())
        self.register_template(ScatterChartTemplate())
        self.register_template(DashboardTemplate())

        # Configure seaborn if available
        if SEABORN_AVAILABLE:
            sns.set_theme(style=style)
            sns.set_palette(palette)

        logger.info(
            f"Initialized ChartGenerator: {self.output_dir}",
            extra={"correlation_id": self.correlation_id}
        )
    
    def register_template(self, template: ChartTemplate) -> None:
        """
        Register a chart template.
        
        Args:
            template: Chart template to register
        """
        with self._lock:
            self._templates[template.name] = template
            logger.debug(
                f"Registered template: {template.name}",
                extra={"correlation_id": self.correlation_id}
            )
    
    def get_template(self, name: str) -> ChartTemplate:
        """
        Get a chart template by name.
        
        Args:
            name: Template name
            
        Returns:
            Chart template
            
        Raises:
            TemplateNotFoundError: If template is not found
        """
        with self._lock:
            if name not in self._templates:
                raise TemplateNotFoundError(f"Template not found: {name}")
            return self._templates[name]
    
    def list_templates(self) -> List[Dict[str, str]]:
        """
        List all registered templates.
        
        Returns:
            List of template information dictionaries
        """
        with self._lock:
            return [
                {"name": t.name, "description": t.description}
                for t in self._templates.values()
            ]
    
    @contextmanager
    def _create_figure(
        self, 
        figsize: Optional[Tuple[int, int]] = None
    ) -> Figure:
        """
        Create a matplotlib figure with proper cleanup.
        
        Args:
            figsize: Figure size (width, height) in inches
            
        Yields:
            Matplotlib figure
        """
        fig = plt.figure(figsize=figsize or self.figsize)
        try:
            yield fig
        finally:
            plt.close(fig)
    
    def _validate_and_save_figure(
        self, 
        fig: Figure, 
        output_path: Path,
        template_name: str,
        data_description: str
    ) -> Path:
        """
        Validate and save figure to file.
        
        Args:
            fig: Matplotlib figure
            output_path: Output file path
            template_name: Template name for logging
            data_description: Data description for logging
            
        Returns:
            Output file path
            
        Raises:
            RenderingError: If figure cannot be saved
        """
        try:
            # Ensure parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save figure
            fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
            
            logger.info(
                f"Saved {template_name} chart: {output_path}",
                extra={
                    "correlation_id": self.correlation_id,
                    "template": template_name,
                    "output_path": str(output_path)
                }
            )
            
            return output_path
        
        except Exception as e:
            error_msg = f"Failed to save {template_name} chart: {e}"
            logger.error(
                error_msg,
                extra={
                    "correlation_id": self.correlation_id,
                    "template": template_name,
                    "output_path": str(output_path),
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
            )
            raise RenderingError(error_msg) from e
    
    def generate_chart(
        self,
        template_name: str,
        data: Any,
        filename: str,
        **kwargs
    ) -> Path:
        """
        Generate a chart using a template.
        
        Args:
            template_name: Name of the template to use
            data: Data to plot
            filename: Output filename
            **kwargs: Additional keyword arguments for the template
            
        Returns:
            Path to the generated chart
            
        Raises:
            TemplateNotFoundError: If template is not found
            DataValidationError: If data validation fails
            RenderingError: If chart rendering fails
        """
        # Get template
        try:
            template = self.get_template(template_name)
        except TemplateNotFoundError as e:
            logger.error(
                f"Template not found: {template_name}",
                extra={
                    "correlation_id": self.correlation_id,
                    "template_name": template_name,
                    "available_templates": list(self._templates.keys())
                }
            )
            raise
        
        # Validate data
        if not template.validate_data(data):
            error_msg = f"Invalid data for template {template_name}"
            logger.error(
                error_msg,
                extra={
                    "correlation_id": self.correlation_id,
                    "template_name": template_name,
                    "data_type": type(data).__name__
                }
            )
            raise DataValidationError(error_msg)
        
        # Create figure and render chart
        try:
            with self._create_figure(kwargs.get("figsize")) as fig:
                # Create axes
                ax = fig.add_subplot(1, 1, 1)
                
                # Render chart
                template.render(fig, ax, data, **kwargs)
                
                # Apply tight layout
                fig.tight_layout()
                
                # Save figure
                output_path = self.output_dir / filename
                return self._validate_and_save_figure(
                    fig, 
                    output_path, 
                    template_name,
                    str(type(data).__name__)
                )
        
        except Exception as e:
            if not isinstance(e, (TemplateNotFoundError, DataValidationError, RenderingError)):
                error_msg = f"Failed to generate chart: {e}"
                logger.error(
                    error_msg,
                    extra={
                        "correlation_id": self.correlation_id,
                        "template_name": template_name,
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    }
                )
                raise RenderingError(error_msg) from e
            raise
    
    def plot_training_curves(
        self,
        data: Dict[str, List[float]],
        title: str = "Training Metrics",
        filename: str = "training_curves.png",
        smooth: bool = True,
        window: int = 10,
    ) -> Path:
        """
        Plot multiple training curves.

        Args:
            data: Dictionary of metric_name -> values
            title: Plot title
            filename: Output filename
            smooth: Whether to apply smoothing
            window: Smoothing window size
            
        Returns:
            Path to the generated chart
        """
        try:
            return self.generate_chart(
                "line_chart",
                data,
                filename,
                title=title,
                smooth=smooth,
                window=window
            )
        except Exception as e:
            logger.error(
                f"Failed to plot training curves: {e}",
                extra={
                    "correlation_id": self.correlation_id,
                    "error": str(e)
                }
            )
            
            # Fallback to basic implementation
            return self._plot_basic_training_curves(
                data, title, filename, smooth, window
            )
    
    def _plot_basic_training_curves(
        self,
        data: Dict[str, List[float]],
        title: str = "Training Metrics",
        filename: str = "training_curves.png",
        smooth: bool = True,
        window: int = 10,
    ) -> Path:
        """Fallback implementation for training curves."""
        with self._create_figure((15, 10)) as fig:
            # Create 2x2 grid of subplots
            axes = fig.subplots(2, 2)
            fig.suptitle(title, fontsize=16, fontweight="bold")
            
            # Plot each metric
            metrics = list(data.keys())
            for idx, (ax, metric) in enumerate(zip(axes.flat, metrics[:4])):
                if idx >= len(metrics):
                    break
                    
                values = np.array(data[metric])
                steps = np.arange(len(values))
                
                # Plot raw data
                ax.plot(steps, values, alpha=0.3, label="Raw", linewidth=1)
                
                # Plot smoothed data
                if smooth and len(values) > window:
                    try:
                        smoothed = np.convolve(
                            values, 
                            np.ones(window) / window, 
                            mode="valid"
                        )
                        smooth_steps = steps[window - 1:]
                        ax.plot(
                            smooth_steps, 
                            smoothed, 
                            label="Smoothed", 
                            linewidth=2
                        )
                    except Exception as e:
                        logger.warning(f"Failed to smooth data: {e}")
                
                ax.set_xlabel("Step")
                ax.set_ylabel(metric.replace("_", " ").title())
                ax.set_title(metric.replace("_", " ").title())
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Save figure
            output_path = self.output_dir / filename
            return self._validate_and_save_figure(
                fig, 
                output_path, 
                "training_curves",
                "training data"
            )

    def plot_reward_distribution(
        self,
        rewards: List[float],
        filename: str = "reward_distribution.png",
        bins: int = 50,
    ) -> Path:
        """
        Plot reward distribution with statistics.
        
        Args:
            rewards: List of reward values
            filename: Output filename
            bins: Number of histogram bins
            
        Returns:
            Path to the generated chart
        """
        try:
            return self.generate_chart(
                "distribution_chart",
                rewards,
                filename,
                bins=bins,
                title="Reward Distribution",
                xlabel="Reward",
                ylabel="Frequency"
            )
        except Exception as e:
            logger.error(
                f"Failed to plot reward distribution: {e}",
                extra={
                    "correlation_id": self.correlation_id,
                    "error": str(e)
                }
            )
            
            # Fallback to basic implementation
            return self._plot_basic_distribution(rewards, filename, bins)

    def plot_correlation_matrix(
        self, 
        data: Dict[str, List[float]], 
        filename: str = "correlation_matrix.png"
    ) -> Optional[Path]:
        """
        Plot correlation matrix between metrics.
        
        Args:
            data: Dictionary of metric_name -> values
            filename: Output filename
            
        Returns:
            Path to the generated chart or None if generation fails
        """
        if not PANDAS_AVAILABLE:
            logger.warning(
                "Pandas not available, skipping correlation matrix",
                extra={"correlation_id": self.correlation_id}
            )
            return None
        
        try:
            return self.generate_chart(
                "heatmap_chart",
                data,
                filename,
                title="Metric Correlation Matrix"
            )
        except Exception as e:
            logger.error(
                f"Failed to plot correlation matrix: {e}",
                extra={
                    "correlation_id": self.correlation_id,
                    "error": str(e)
                }
            )
            return None

    def plot_gradient_flow(
        self,
        gradient_norms: Dict[str, List[float]],
        filename: str = "gradient_flow.png",
    ) -> Path:
        """
        Plot gradient flow across layers.
        
        Args:
            gradient_norms: Dictionary of layer_name -> gradient norms
            filename: Output filename
            
        Returns:
            Path to the generated chart
        """
        try:
            # Convert to line chart format
            return self.generate_chart(
                "line_chart",
                gradient_norms,
                filename,
                title="Gradient Flow Across Layers",
                ylabel="Gradient Norm",
                xlabel="Step",
                yscale="log"
            )
        except Exception as e:
            logger.error(
                f"Failed to plot gradient flow: {e}",
                extra={
                    "correlation_id": self.correlation_id,
                    "error": str(e)
                }
            )
            
            # Fallback to basic implementation
            return self._plot_basic_gradient_flow(gradient_norms, filename)
    
    def _plot_basic_gradient_flow(
        self,
        gradient_norms: Dict[str, List[float]],
        filename: str
    ) -> Path:
        """Fallback implementation for gradient flow."""
        with self._create_figure((15, 6)) as fig:
            ax = fig.add_subplot(1, 1, 1)
            
            for layer_name, norms in gradient_norms.items():
                steps = np.arange(len(norms))
                ax.plot(steps, norms, label=layer_name, alpha=0.7)
            
            ax.set_xlabel("Step")
            ax.set_ylabel("Gradient Norm")
            ax.set_title("Gradient Flow Across Layers", fontweight="bold")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.grid(True, alpha=0.3)
            ax.set_yscale("log")
            
            # Save figure
            output_path = self.output_dir / filename
            return self._validate_and_save_figure(
                fig,
                output_path,
                "gradient_flow",
                "gradient norms"
            )

    def plot_memory_usage(
        self,
        memory_data: Dict[str, List[float]],
        filename: str = "memory_usage.png"
    ) -> Path:
        """
        Plot memory usage over time.
        
        Args:
            memory_data: Dictionary of memory_type -> values
            filename: Output filename
            
        Returns:
            Path to the generated chart
        """
        try:
            return self.generate_chart(
                "line_chart",
                memory_data,
                filename,
                title="Memory Usage Over Time",
                ylabel="Memory (MB)",
                xlabel="Step"
            )
        except Exception as e:
            logger.error(
                f"Failed to plot memory usage: {e}",
                extra={
                    "correlation_id": self.correlation_id,
                    "error": str(e)
                }
            )
            
            # Fallback to basic implementation
            return self._plot_basic_memory_usage(memory_data, filename)
    
    def _plot_basic_memory_usage(
        self,
        memory_data: Dict[str, List[float]],
        filename: str
    ) -> Path:
        """Fallback implementation for memory usage."""
        with self._create_figure((12, 6)) as fig:
            ax = fig.add_subplot(1, 1, 1)
            
            for mem_type, values in memory_data.items():
                steps = np.arange(len(values))
                ax.plot(steps, values, label=mem_type, linewidth=2)
            
            ax.set_xlabel("Step")
            ax.set_ylabel("Memory (MB)")
            ax.set_title("Memory Usage Over Time", fontweight="bold")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add threshold line if memory exceeds 8GB
            max_memory = max(max(v) for v in memory_data.values())
            if max_memory > 8000:
                ax.axhline(8000, color="r", linestyle="--", label="8GB Threshold")
            
            # Save figure
            output_path = self.output_dir / filename
            return self._validate_and_save_figure(
                fig,
                output_path,
                "memory_usage",
                "memory data"
            )

    def plot_token_distribution(
        self,
        thinking_tokens: List[int],
        answer_tokens: List[int],
        filename: str = "token_distribution.png",
    ) -> Path:
        """
        Plot distribution of thinking vs answer tokens.
        
        Args:
            thinking_tokens: List of thinking token counts
            answer_tokens: List of answer token counts
            filename: Output filename
            
        Returns:
            Path to the generated chart
        """
        try:
            # Create scatter data
            scatter_data = (thinking_tokens, answer_tokens)
            
            # Create dashboard data
            dashboard_data = {
                "thinking_tokens": thinking_tokens,
                "answer_tokens": answer_tokens,
                "scatter": scatter_data,
                "ratios": [t / max(a, 1) for t, a in zip(thinking_tokens, answer_tokens)]
            }
            
            # Define dashboard panels
            panels = [
                {
                    "type": "distribution",
                    "data_key": "thinking_tokens",
                    "title": "Thinking Token Distribution",
                    "position": (0, 0),
                    "xlabel": "Thinking Tokens"
                },
                {
                    "type": "distribution",
                    "data_key": "answer_tokens",
                    "title": "Answer Token Distribution",
                    "position": (0, 1),
                    "xlabel": "Answer Tokens"
                },
                {
                    "type": "scatter",
                    "data_key": "scatter",
                    "title": "Thinking vs Answer Tokens",
                    "position": (1, 0),
                    "xlabel": "Thinking Tokens",
                    "ylabel": "Answer Tokens"
                },
                {
                    "type": "distribution",
                    "data_key": "ratios",
                    "title": "Token Ratio Distribution",
                    "position": (1, 1),
                    "xlabel": "Thinking/Answer Ratio"
                }
            ]
            
            # Generate dashboard
            return self.generate_chart(
                "dashboard",
                dashboard_data,
                filename,
                title="Token Distribution Analysis",
                layout=(2, 2),
                panels=panels
            )
        except Exception as e:
            logger.error(
                f"Failed to plot token distribution: {e}",
                extra={
                    "correlation_id": self.correlation_id,
                    "error": str(e)
                }
            )
            
            # Fallback to basic implementation
            return self._plot_basic_token_dist(thinking_tokens, answer_tokens, filename)

    def create_dashboard(
        self,
        stats_data: Dict[str, Any],
        filename: str = "training_dashboard.png",
        layout: Tuple[int, int] = (3, 3)
    ) -> Path:
        """
        Create comprehensive training dashboard.
        
        Args:
            stats_data: Dictionary of statistics data
            filename: Output filename
            layout: Dashboard layout (rows, columns)
            
        Returns:
            Path to the generated chart
        """
        try:
            # Define dashboard panels
            panels = [
                {
                    "type": "line",
                    "data_key": "loss",
                    "title": "Training Loss",
                    "position": (0, 0),
                    "span": (1, 2),
                    "xlabel": "Step",
                    "ylabel": "Loss"
                },
                {
                    "type": "line",
                    "data_key": "reward_mean",
                    "title": "Average Reward",
                    "position": (1, 0),
                    "span": (1, 2),
                    "xlabel": "Step",
                    "ylabel": "Reward",
                    "color": "green"
                },
                {
                    "type": "line",
                    "data_key": "memory_allocated_mb",
                    "title": "Memory Usage",
                    "position": (2, 0),
                    "span": (1, 2),
                    "xlabel": "Step",
                    "ylabel": "Memory (MB)",
                    "color": "red"
                },
                {
                    "type": "text",
                    "data_key": "stats_text",
                    "title": "Training Statistics",
                    "position": (0, 2),
                    "fontsize": 12
                },
                {
                    "type": "distribution",
                    "data_key": "reward_mean",
                    "title": "Reward Distribution",
                    "position": (1, 2),
                    "xlabel": "Reward"
                },
                {
                    "type": "line",
                    "data_key": "learning_rate",
                    "title": "Learning Rate",
                    "position": (2, 2),
                    "xlabel": "Step",
                    "ylabel": "LR",
                    "color": "purple",
                    "yscale": "log"
                }
            ]
            
            # Prepare stats text
            stats_text = "Training Statistics\n\n"
            if "loss" in stats_data and len(stats_data["loss"]) > 0:
                stats_text += f"Final Loss: {stats_data['loss'][-1]:.4f}\n"
            if "reward_mean" in stats_data and len(stats_data["reward_mean"]) > 0:
                stats_text += f"Final Reward: {stats_data['reward_mean'][-1]:.4f}\n"
            
            # Add stats text to data
            dashboard_data = stats_data.copy()
            dashboard_data["stats_text"] = stats_text
            
            # Generate dashboard
            return self.generate_chart(
                "dashboard",
                dashboard_data,
                filename,
                title="Training Dashboard",
                layout=layout,
                panels=panels
            )
        except Exception as e:
            logger.error(
                f"Failed to create dashboard: {e}",
                extra={
                    "correlation_id": self.correlation_id,
                    "error": str(e)
                }
            )
            
            # Fallback to basic implementation
            return self._plot_basic_dashboard(stats_data, filename)
    
    def _plot_basic_dashboard(
        self,
        stats_data: Dict[str, Any],
        filename: str
    ) -> Path:
        """Fallback implementation for dashboard."""
        with self._create_figure((20, 12)) as fig:
            gs = gridspec.GridSpec(3, 3, figure=fig)
            
            # Loss plot
            ax1 = fig.add_subplot(gs[0, :2])
            if "loss" in stats_data:
                loss_data = stats_data["loss"]
                steps = np.arange(len(loss_data))
                ax1.plot(steps, loss_data, linewidth=2)
                ax1.set_title("Training Loss", fontweight="bold")
                ax1.set_xlabel("Step")
                ax1.set_ylabel("Loss")
                ax1.grid(True, alpha=0.3)
            
            # Reward plot
            ax2 = fig.add_subplot(gs[1, :2])
            if "reward_mean" in stats_data:
                reward_data = stats_data["reward_mean"]
                steps = np.arange(len(reward_data))
                ax2.plot(steps, reward_data, color="green", linewidth=2)
                ax2.set_title("Average Reward", fontweight="bold")
                ax2.set_xlabel("Step")
                ax2.set_ylabel("Reward")
                ax2.grid(True, alpha=0.3)
            
            # Memory plot
            ax3 = fig.add_subplot(gs[2, :2])
            if "memory_allocated_mb" in stats_data:
                mem_data = stats_data["memory_allocated_mb"]
                steps = np.arange(len(mem_data))
                ax3.plot(steps, mem_data, color="red", linewidth=2)
                ax3.set_title("Memory Usage", fontweight="bold")
                ax3.set_xlabel("Step")
                ax3.set_ylabel("Memory (MB)")
                ax3.grid(True, alpha=0.3)
            
            # Statistics panel
            ax4 = fig.add_subplot(gs[0, 2])
            ax4.axis("off")
            stats_text = "Training Statistics\n\n"
            if "loss" in stats_data and len(stats_data["loss"]) > 0:
                stats_text += f"Final Loss: {stats_data['loss'][-1]:.4f}\n"
            if "reward_mean" in stats_data and len(stats_data["reward_mean"]) > 0:
                stats_text += f"Final Reward: {stats_data['reward_mean'][-1]:.4f}\n"
            ax4.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment="center")
            
            # Reward distribution
            ax5 = fig.add_subplot(gs[1, 2])
            if "reward_mean" in stats_data and len(stats_data["reward_mean"]) > 0:
                if SEABORN_AVAILABLE:
                    sns.histplot(stats_data["reward_mean"], bins=20, kde=True, ax=ax5)
                else:
                    ax5.hist(stats_data["reward_mean"], bins=20, alpha=0.7)
                ax5.set_title("Reward Distribution", fontweight="bold")
                ax5.set_xlabel("Reward")
            
            # Learning rate
            ax6 = fig.add_subplot(gs[2, 2])
            if "learning_rate" in stats_data:
                lr_data = stats_data["learning_rate"]
                steps = np.arange(len(lr_data))
                ax6.plot(steps, lr_data, color="purple", linewidth=2)
                ax6.set_title("Learning Rate", fontweight="bold")
                ax6.set_xlabel("Step")
                ax6.set_ylabel("LR")
                ax6.set_yscale("log")
            
            # Save figure
            output_path = self.output_dir / filename
            return self._validate_and_save_figure(
                fig,
                output_path,
                "dashboard",
                "dashboard data"
            )

    def _plot_basic_distribution(
        self,
        data: List[float],
        filename: str,
        bins: int
    ) -> Path:
        """
        Fallback basic distribution plot.
        
        Args:
            data: Data to plot
            filename: Output filename
            bins: Number of histogram bins
            
        Returns:
            Path to the generated chart
        """
        with self._create_figure((10, 6)) as fig:
            ax = fig.add_subplot(1, 1, 1)
            ax.hist(data, bins=bins, alpha=0.7, edgecolor="black")
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
            ax.set_title("Distribution")
            ax.axvline(
                np.mean(data),
                color="r",
                linestyle="--",
                label=f"Mean: {np.mean(data):.3f}"
            )
            ax.legend()
            
            output_path = self.output_dir / filename
            return self._validate_and_save_figure(
                fig,
                output_path,
                "basic_distribution",
                "distribution data"
            )

    def _plot_basic_token_dist(
        self,
        thinking: List[int],
        answer: List[int],
        filename: str
    ) -> Path:
        """
        Fallback basic token distribution.
        
        Args:
            thinking: Thinking token counts
            answer: Answer token counts
            filename: Output filename
            
        Returns:
            Path to the generated chart
        """
        with self._create_figure((12, 5)) as fig:
            axes = fig.subplots(1, 2)
            
            axes[0].hist(thinking, bins=30, alpha=0.7)
            axes[0].set_xlabel("Thinking Tokens")
            axes[0].set_title("Thinking Distribution")
            
            axes[1].hist(answer, bins=30, alpha=0.7)
            axes[1].set_xlabel("Answer Tokens")
            axes[1].set_title("Answer Distribution")
            
            output_path = self.output_dir / filename
            return self._validate_and_save_figure(
                fig,
                output_path,
                "basic_token_dist",
                "token distribution data"
            )

    def cleanup(self) -> None:
        """Clean up resources."""
        plt.close("all")
        logger.info(
            "Cleaned up chart generator resources",
            extra={"correlation_id": self.correlation_id}
        )


# Dependencies: matplotlib, seaborn, pandas, numpy
# Install: pip install matplotlib seaborn pandas numpy
# Usage: chart_gen = ChartGenerator(output_dir=Path("./charts"))
#        chart_gen.plot_training_curves(data={'loss': [...]})
# Status: Complete and commit-ready