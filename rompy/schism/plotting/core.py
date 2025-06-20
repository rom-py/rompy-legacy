"""
Core plotting base classes and utilities for SCHISM visualization.

This module provides the foundation for all SCHISM plotting functionality,
including base classes, configuration models, and common utilities.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic import BaseModel, Field, field_validator

from rompy.core.types import RompyBaseModel

logger = logging.getLogger(__name__)


class PlotConfig(RompyBaseModel):
    """
    Configuration model for SCHISM plotting parameters.

    This model defines common plotting parameters with validation
    and default values for consistent plotting behavior.
    """

    # Figure parameters
    figsize: Tuple[float, float] = Field(
        default=(12, 8),
        description="Figure size in inches (width, height)"
    )
    dpi: int = Field(
        default=100,
        description="Figure resolution in dots per inch"
    )

    # Colormap parameters
    cmap: str = Field(
        default="viridis",
        description="Colormap for data visualization"
    )
    vmin: Optional[float] = Field(
        default=None,
        description="Minimum value for colormap normalization"
    )
    vmax: Optional[float] = Field(
        default=None,
        description="Maximum value for colormap normalization"
    )

    # Geographic parameters
    add_coastlines: bool = Field(
        default=True,
        description="Whether to add coastlines to the plot"
    )
    add_gridlines: bool = Field(
        default=True,
        description="Whether to add coordinate gridlines"
    )
    projection: str = Field(
        default="PlateCarree",
        description="Cartopy projection for geographic plots"
    )

    # Grid parameters
    show_grid: bool = Field(
        default=True,
        description="Whether to show the computational grid"
    )
    grid_alpha: float = Field(
        default=0.3,
        description="Transparency of grid lines"
    )
    grid_color: str = Field(
        default="gray",
        description="Color of grid lines"
    )

    # Boundary parameters
    show_boundaries: bool = Field(
        default=True,
        description="Whether to show model boundaries"
    )
    boundary_colors: Dict[str, str] = Field(
        default={"ocean": "red", "land": "green", "tidal": "blue"},
        description="Colors for different boundary types"
    )
    boundary_linewidth: float = Field(
        default=2.0,
        description="Line width for boundary plotting"
    )

    # Colorbar parameters
    add_colorbar: bool = Field(
        default=True,
        description="Whether to add a colorbar"
    )
    colorbar_label: Optional[str] = Field(
        default=None,
        description="Label for the colorbar"
    )
    colorbar_orientation: str = Field(
        default="vertical",
        description="Orientation of the colorbar"
    )

    # Title and labels
    title: Optional[str] = Field(
        default=None,
        description="Plot title"
    )
    xlabel: Optional[str] = Field(
        default="Longitude",
        description="X-axis label"
    )
    ylabel: Optional[str] = Field(
        default="Latitude",
        description="Y-axis label"
    )

    @field_validator('figsize')
    @classmethod
    def validate_figsize(cls, v):
        """Validate figure size."""
        if len(v) != 2:
            raise ValueError("figsize must be a tuple of two values")
        if any(x <= 0 for x in v):
            raise ValueError("figsize values must be positive")
        return v

    @field_validator('dpi')
    @classmethod
    def validate_dpi(cls, v):
        """Validate DPI."""
        if v <= 0:
            raise ValueError("dpi must be positive")
        return v

    @field_validator('grid_alpha')
    @classmethod
    def validate_grid_alpha(cls, v):
        """Validate grid alpha."""
        if not 0 <= v <= 1:
            raise ValueError("grid_alpha must be between 0 and 1")
        return v

    @field_validator('boundary_linewidth')
    @classmethod
    def validate_boundary_linewidth(cls, v):
        """Validate boundary line width."""
        if v <= 0:
            raise ValueError("boundary_linewidth must be positive")
        return v

    @field_validator('colorbar_orientation')
    @classmethod
    def validate_colorbar_orientation(cls, v):
        """Validate colorbar orientation."""
        if v not in ['vertical', 'horizontal']:
            raise ValueError("colorbar_orientation must be 'vertical' or 'horizontal'")
        return v


class BasePlotter(ABC):
    """
    Abstract base class for all SCHISM plotters.

    This class provides common functionality and interface for all
    specialized plotting classes.

    Parameters
    ----------
    config : Optional[Any]
        SCHISM configuration object
    grid_file : Optional[Union[str, Path]]
        Path to grid file if config is not provided
    plot_config : Optional[PlotConfig]
        Plotting configuration parameters
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        grid_file: Optional[Union[str, Path]] = None,
        plot_config: Optional[PlotConfig] = None
    ):
        """Initialize base plotter."""
        self.config = config
        self.grid_file = Path(grid_file) if grid_file else None
        self.plot_config = plot_config or PlotConfig()

        # Initialize grid reference
        self._grid = None
        self._grid_loaded = False

        # Validate initialization
        self._validate_initialization()

    def _validate_initialization(self):
        """Validate plotter initialization."""
        if not self.config and not self.grid_file:
            raise ValueError("Either config or grid_file must be provided")

        if self.grid_file and not self.grid_file.exists():
            raise FileNotFoundError(f"Grid file not found: {self.grid_file}")

    @property
    def grid(self):
        """
        Get grid object, loading if necessary.

        Returns
        -------
        grid : Any
            Grid object from config or loaded from file
        """
        if not self._grid_loaded:
            self._load_grid()
        return self._grid

    def _load_grid(self):
        """Load grid from config or file."""
        if self.config and hasattr(self.config, 'grid'):
            self._grid = self.config.grid
            logger.info("Grid loaded from configuration")
        elif self.grid_file:
            # Import here to avoid circular imports
            from ..grid import SCHISMGrid
            self._grid = SCHISMGrid.from_file(self.grid_file)
            logger.info(f"Grid loaded from file: {self.grid_file}")
        else:
            raise ValueError("No grid available in config or file")

        self._grid_loaded = True

    def create_figure(
        self,
        ax: Optional[Axes] = None,
        use_cartopy: bool = True,
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """
        Create figure and axes for plotting.

        Parameters
        ----------
        ax : Optional[Axes]
            Existing axes to use. If None, new figure is created.
        use_cartopy : bool
            Whether to use cartopy projection
        **kwargs : dict
            Additional arguments for figure creation

        Returns
        -------
        fig : Figure
            Figure object
        ax : Axes
            Axes object
        """
        if ax is not None:
            return ax.get_figure(), ax

        # Create new figure
        figsize = kwargs.get('figsize', self.plot_config.figsize)
        dpi = kwargs.get('dpi', self.plot_config.dpi)

        if use_cartopy and self.plot_config.add_coastlines:
            try:
                import cartopy.crs as ccrs
                import cartopy.feature as cfeature

                # Set up projection
                if self.plot_config.projection == "PlateCarree":
                    projection = ccrs.PlateCarree()
                else:
                    projection = ccrs.PlateCarree()  # Default fallback

                fig = plt.figure(figsize=figsize, dpi=dpi)
                ax = fig.add_subplot(111, projection=projection)

                # Add geographic features
                if self.plot_config.add_coastlines:
                    ax.coastlines(resolution='50m')

                if self.plot_config.add_gridlines:
                    ax.gridlines(draw_labels=True, alpha=0.5)

            except (ImportError, Exception) as e:
                logger.warning(f"Cartopy not available or failed: {e}, using regular matplotlib axes")
                fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        else:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        return fig, ax

    def add_colorbar(
        self,
        fig: Figure,
        ax: Axes,
        mappable: Any,
        label: Optional[str] = None
    ) -> Any:
        """
        Add colorbar to plot.

        Parameters
        ----------
        fig : Figure
            Figure object
        ax : Axes
            Axes object
        mappable : Any
            Mappable object (e.g., from pcolormesh, contourf)
        label : Optional[str]
            Colorbar label

        Returns
        -------
        cbar : Any
            Colorbar object
        """
        if not self.plot_config.add_colorbar:
            return None

        label = label or self.plot_config.colorbar_label
        orientation = self.plot_config.colorbar_orientation

        cbar = fig.colorbar(mappable, ax=ax, orientation=orientation)

        if label:
            cbar.set_label(label)

        return cbar

    def set_plot_labels(
        self,
        ax: Axes,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None
    ):
        """
        Set plot labels and title.

        Parameters
        ----------
        ax : Axes
            Axes object
        title : Optional[str]
            Plot title
        xlabel : Optional[str]
            X-axis label
        ylabel : Optional[str]
            Y-axis label
        """
        title = title or self.plot_config.title

        # Only use default labels if no labels are explicitly provided AND no labels are already set
        if xlabel is None and not ax.get_xlabel():
            xlabel = self.plot_config.xlabel
        if ylabel is None and not ax.get_ylabel():
            ylabel = self.plot_config.ylabel

        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

    def finalize_plot(
        self,
        fig: Figure,
        ax: Axes,
        title: Optional[str] = None,
        tight_layout: bool = True
    ) -> Tuple[Figure, Axes]:
        """
        Finalize plot with labels and formatting.

        Parameters
        ----------
        fig : Figure
            Figure object
        ax : Axes
            Axes object
        title : Optional[str]
            Plot title
        tight_layout : bool
            Whether to apply tight layout

        Returns
        -------
        fig : Figure
            Figure object
        ax : Axes
            Axes object
        """
        self.set_plot_labels(ax, title=title)

        if tight_layout:
            fig.tight_layout()

        return fig, ax

    @abstractmethod
    def plot(self, **kwargs) -> Tuple[Figure, Axes]:
        """
        Abstract method for plotting functionality.

        Must be implemented by subclasses.
        """
        pass


class PlotValidator:
    """
    Validator for plotting data and parameters.

    This class provides validation methods for common plotting scenarios
    and data types used in SCHISM visualization.
    """

    @staticmethod
    def validate_dataset(ds: xr.Dataset, required_vars: Optional[List[str]] = None) -> bool:
        """
        Validate xarray dataset for plotting.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset to validate
        required_vars : Optional[List[str]]
            List of required variables

        Returns
        -------
        bool
            True if dataset is valid

        Raises
        ------
        ValueError
            If dataset is invalid
        """
        if not isinstance(ds, xr.Dataset):
            raise ValueError("Input must be an xarray Dataset")

        if required_vars:
            missing_vars = [var for var in required_vars if var not in ds.data_vars]
            if missing_vars:
                raise ValueError(f"Missing required variables: {missing_vars}")

        return True

    @staticmethod
    def validate_coordinates(ds: xr.Dataset, required_coords: Optional[List[str]] = None) -> bool:
        """
        Validate dataset coordinates.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset to validate
        required_coords : Optional[List[str]]
            List of required coordinates

        Returns
        -------
        bool
            True if coordinates are valid

        Raises
        ------
        ValueError
            If coordinates are invalid
        """
        if required_coords:
            missing_coords = [coord for coord in required_coords if coord not in ds.coords]
            if missing_coords:
                raise ValueError(f"Missing required coordinates: {missing_coords}")

        return True

    @staticmethod
    def validate_time_dimension(ds: xr.Dataset, var_name: str) -> bool:
        """
        Validate time dimension in dataset variable.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset to validate
        var_name : str
            Variable name to check

        Returns
        -------
        bool
            True if time dimension is valid

        Raises
        ------
        ValueError
            If time dimension is invalid
        """
        if var_name not in ds.data_vars:
            raise ValueError(f"Variable {var_name} not found in dataset")

        if 'time' not in ds[var_name].dims:
            raise ValueError(f"Variable {var_name} does not have time dimension")

        return True

    @staticmethod
    def validate_spatial_dimensions(ds: xr.Dataset, var_name: str) -> bool:
        """
        Validate spatial dimensions in dataset variable.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset to validate
        var_name : str
            Variable name to check

        Returns
        -------
        bool
            True if spatial dimensions are valid

        Raises
        ------
        ValueError
            If spatial dimensions are invalid
        """
        if var_name not in ds.data_vars:
            raise ValueError(f"Variable {var_name} not found in dataset")

        spatial_dims = ['lon', 'lat', 'longitude', 'latitude', 'x', 'y', 'node']
        var_dims = ds[var_name].dims

        has_spatial = any(dim in var_dims for dim in spatial_dims)
        if not has_spatial:
            raise ValueError(f"Variable {var_name} does not have spatial dimensions")

        return True
