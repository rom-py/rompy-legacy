"""
SCHISM Plotting Module

This module provides a unified interface for visualizing SCHISM model input data,
including grid visualization, boundary conditions, atmospheric forcing, and tidal data.

Main Classes:
    SchismPlotter: Primary interface for all SCHISM plotting functionality

Main Functions:
    plot_schism_overview: Create comprehensive overview plots
    plot_grid: Plot grid and bathymetry
    plot_boundary_data: Plot boundary condition data
    plot_atmospheric_data: Plot atmospheric forcing data
    plot_tidal_data: Plot tidal data

Example Usage:
    >>> from rompy.schism.plotting import SchismPlotter
    >>> plotter = SchismPlotter(config)
    >>> fig, ax = plotter.plot_overview()
    >>> fig.show()
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from .core import BasePlotter, PlotConfig
from .grid import GridPlotter
from .data import DataPlotter
from .overview import OverviewPlotter
from .validation import ValidationPlotter, ModelValidator, ValidationResult
from .utils import setup_cartopy_axis, validate_file_exists

logger = logging.getLogger(__name__)


class SchismPlotter:
    """
    Unified interface for SCHISM model visualization.
    
    This class provides a comprehensive plotting interface for all SCHISM input data types,
    including grids, boundary conditions, atmospheric forcing, and tidal data.
    
    Parameters
    ----------
    config : Optional[Any]
        SCHISM configuration object. If provided, plotting methods will use
        configuration data automatically.
    grid_file : Optional[Union[str, Path]]
        Path to grid file (hgrid.gr3). Required if config is not provided.
    
    Attributes
    ----------
    config : Any
        SCHISM configuration object
    grid_plotter : GridPlotter
        Grid-specific plotting functionality
    data_plotter : DataPlotter
        Data-specific plotting functionality
    """
    
    def __init__(
        self,
        config: Optional[Any] = None,
        grid_file: Optional[Union[str, Path]] = None,
    ):
        """Initialize SchismPlotter with configuration or grid file."""
        self.config = config
        self.grid_file = Path(grid_file) if grid_file else None
        
        # Initialize sub-plotters
        self.grid_plotter = GridPlotter(config=config, grid_file=grid_file)
        self.data_plotter = DataPlotter(config=config, grid_file=grid_file)
        self.overview_plotter = OverviewPlotter(config=config, grid_file=grid_file)
        self.validation_plotter = ValidationPlotter(config=config, grid_file=grid_file)
        
        # Validate initialization
        if not config and not grid_file:
            raise ValueError("Either config or grid_file must be provided")
            
        if grid_file and not validate_file_exists(grid_file):
            raise FileNotFoundError(f"Grid file not found: {grid_file}")

    def plot_overview(
        self,
        figsize: Tuple[float, float] = (16, 12),
        include_boundaries: bool = True,
        include_bathymetry: bool = True,
        include_atmospheric: bool = True,
        include_tidal: bool = True,
        **kwargs
    ) -> Tuple[Figure, np.ndarray]:
        """
        Create comprehensive overview plot of SCHISM model setup.
        
        Parameters
        ----------
        figsize : Tuple[float, float], optional
            Figure size in inches. Default is (16, 12).
        include_boundaries : bool, optional
            Whether to include boundary condition visualization. Default is True.
        include_bathymetry : bool, optional
            Whether to include bathymetry visualization. Default is True.
        include_atmospheric : bool, optional
            Whether to include atmospheric forcing visualization. Default is True.
        include_tidal : bool, optional
            Whether to include tidal data visualization. Default is True.
        **kwargs : dict
            Additional keyword arguments passed to plotting functions.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        axes : numpy.ndarray
            Array of axes objects.
            
        Examples
        --------
        >>> plotter = SchismPlotter(config)
        >>> fig, axes = plotter.plot_overview()
        >>> fig.savefig('schism_overview.png', dpi=300, bbox_inches='tight')
        """
        # Create subplot layout
        fig, axes = plt.subplots(2, 2, figsize=figsize, 
                                subplot_kw={'projection': setup_cartopy_axis()})
        axes = axes.flatten()
        
        # Plot grid and bathymetry
        if include_bathymetry:
            self.grid_plotter.plot_bathymetry(ax=axes[0], **kwargs)
            axes[0].set_title("Grid and Bathymetry")
        else:
            self.grid_plotter.plot_grid(ax=axes[0], **kwargs)
            axes[0].set_title("Grid")
            
        # Plot boundaries
        if include_boundaries:
            self.grid_plotter.plot_boundaries(ax=axes[1], **kwargs)
            axes[1].set_title("Boundary Conditions")
            
        # Plot atmospheric forcing if available
        if include_atmospheric and self._has_atmospheric_data():
            self.data_plotter.plot_atmospheric_spatial(ax=axes[2], **kwargs)
            axes[2].set_title("Atmospheric Forcing")
        else:
            axes[2].text(0.5, 0.5, "No Atmospheric Data", 
                        ha='center', va='center', transform=axes[2].transAxes)
            axes[2].set_title("Atmospheric Forcing")
            
        # Plot tidal data if available
        if include_tidal and self._has_tidal_data():
            self.data_plotter.plot_tidal_boundaries(ax=axes[3], **kwargs)
            axes[3].set_title("Tidal Boundaries")
        else:
            axes[3].text(0.5, 0.5, "No Tidal Data",
                        ha='center', va='center', transform=axes[3].transAxes)
            axes[3].set_title("Tidal Data")
            
        plt.tight_layout()
        return fig, axes

    def plot_comprehensive_overview(
        self,
        figsize: Tuple[float, float] = (20, 16),
        include_validation: bool = True,
        include_quality_metrics: bool = True,
        include_data_summary: bool = True,
        save_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> Tuple[Figure, Dict[str, Axes]]:
        """
        Create comprehensive multi-panel overview plot.
        
        This creates a detailed overview with the following panels:
        - Grid and bathymetry visualization
        - Boundary conditions and forcing locations
        - Data quality assessment plots
        - Model setup validation summary
        - Time series overview of forcing data
        
        Parameters
        ----------
        figsize : Tuple[float, float], optional
            Figure size in inches. Default is (20, 16).
        include_validation : bool, optional
            Include model validation panel. Default is True.
        include_quality_metrics : bool, optional
            Include grid quality metrics panel. Default is True.
        include_data_summary : bool, optional
            Include data summary panel. Default is True.
        save_path : Optional[Union[str, Path]], optional
            Path to save the plot. If None, plot is not saved.
        **kwargs : dict
            Additional keyword arguments passed to plotting functions.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        axes : Dict[str, matplotlib.axes.Axes]
            Dictionary of axes objects keyed by panel name.
            
        Examples
        --------
        >>> plotter = SchismPlotter(config)
        >>> fig, axes = plotter.plot_comprehensive_overview()
        >>> fig.show()
        """
        return self.overview_plotter.plot_comprehensive_overview(
            figsize=figsize,
            include_validation=include_validation,
            include_quality_metrics=include_quality_metrics,
            include_data_summary=include_data_summary,
            save_path=save_path,
            **kwargs
        )

    def plot_grid_analysis_overview(
        self,
        figsize: Tuple[float, float] = (16, 12),
        save_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> Tuple[Figure, Dict[str, Axes]]:
        """
        Create grid-focused analysis overview.
        
        This creates a detailed grid analysis with:
        - Grid structure and bathymetry
        - Element quality metrics
        - Boundary condition locations
        - Grid statistics and histograms
        
        Parameters
        ----------
        figsize : Tuple[float, float], optional
            Figure size in inches. Default is (16, 12).
        save_path : Optional[Union[str, Path]], optional
            Path to save the plot. If None, plot is not saved.
        **kwargs : dict
            Additional keyword arguments.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        axes : Dict[str, matplotlib.axes.Axes]
            Dictionary of axes objects keyed by panel name.
        """
        return self.overview_plotter.plot_grid_analysis_overview(
            figsize=figsize,
            save_path=save_path,
            **kwargs
        )

    def plot_data_analysis_overview(
        self,
        figsize: Tuple[float, float] = (16, 12),
        save_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> Tuple[Figure, Dict[str, Axes]]:
        """
        Create data-focused analysis overview.
        
        This creates a comprehensive data analysis with:
        - Atmospheric forcing overview
        - Boundary data time series
        - Data quality metrics
        - Temporal coverage analysis
        
        Parameters
        ----------
        figsize : Tuple[float, float], optional
            Figure size in inches. Default is (16, 12).
        save_path : Optional[Union[str, Path]], optional
            Path to save the plot. If None, plot is not saved.
        **kwargs : dict
            Additional keyword arguments.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        axes : Dict[str, matplotlib.axes.Axes]
            Dictionary of axes objects keyed by panel name.
        """
        return self.overview_plotter.plot_data_analysis_overview(
            figsize=figsize,
            save_path=save_path,
            **kwargs
        )

    def plot_validation_summary(
        self,
        figsize: Tuple[float, float] = (14, 10),
        save_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> Tuple[Figure, Dict[str, Axes]]:
        """
        Create comprehensive validation summary plot.
        
        This creates a detailed validation report with:
        - Overall validation status overview
        - Validation results by category
        - Detailed validation results table
        - Validation process timeline
        
        Parameters
        ----------
        figsize : Tuple[float, float], optional
            Figure size in inches. Default is (14, 10).
        save_path : Optional[Union[str, Path]], optional
            Path to save the plot. If None, plot is not saved.
        **kwargs : dict
            Additional keyword arguments.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        axes : Dict[str, matplotlib.axes.Axes]
            Dictionary of axes objects keyed by panel name.
        """
        return self.validation_plotter.plot_validation_summary(
            figsize=figsize,
            save_path=save_path,
            **kwargs
        )

    def plot_quality_assessment(
        self,
        figsize: Tuple[float, float] = (12, 8),
        save_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """
        Create model quality assessment radar chart.
        
        This creates a radar chart showing quality scores across different
        categories of model setup validation.
        
        Parameters
        ----------
        figsize : Tuple[float, float], optional
            Figure size in inches. Default is (12, 8).
        save_path : Optional[Union[str, Path]], optional
            Path to save the plot. If None, plot is not saved.
        **kwargs : dict
            Additional keyword arguments.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes.Axes
            The axes object.
        """
        return self.validation_plotter.plot_quality_assessment(
            figsize=figsize,
            save_path=save_path,
            **kwargs
        )

    def run_model_validation(self) -> List[ValidationResult]:
        """
        Run comprehensive model validation checks.
        
        This runs all available validation checks on the model setup
        and returns detailed results.
        
        Returns
        -------
        List[ValidationResult]
            List of validation results with status and details.
            
        Examples
        --------
        >>> plotter = SchismPlotter(config)
        >>> results = plotter.run_model_validation()
        >>> for result in results:
        ...     print(f"{result.check_name}: {result.status}")
        """
        return self.validation_plotter.validator.run_all_validations()

    def plot_grid(self, **kwargs) -> Tuple[Figure, Axes]:
        """
        Plot SCHISM grid.
        
        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed to GridPlotter.plot_grid.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes.Axes
            The axes object.
        """
        return self.grid_plotter.plot_grid(**kwargs)

    def plot_bathymetry(self, **kwargs) -> Tuple[Figure, Axes]:
        """
        Plot SCHISM bathymetry.
        
        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed to GridPlotter.plot_bathymetry.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes.Axes
            The axes object.
        """
        return self.grid_plotter.plot_bathymetry(**kwargs)

    def plot_boundaries(self, **kwargs) -> Tuple[Figure, Axes]:
        """
        Plot SCHISM boundaries.
        
        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed to GridPlotter.plot_boundaries.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes.Axes
            The axes object.
        """
        return self.grid_plotter.plot_boundaries(**kwargs)

    def plot_boundary_data(
        self,
        file_path: Union[str, Path],
        variable: Optional[str] = None,
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """
        Plot boundary condition data from SCHISM input files.
        
        Parameters
        ----------
        file_path : Union[str, Path]
            Path to boundary data file (e.g., SAL_3D.th.nc, TEM_3D.th.nc, uv3D.th.nc).
        variable : Optional[str]
            Variable to plot. If None, the first available variable is used.
        **kwargs : dict
            Additional keyword arguments passed to DataPlotter.plot_boundary_data.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes.Axes
            The axes object.
        """
        return self.data_plotter.plot_boundary_data(file_path, variable, **kwargs)

    def plot_atmospheric_data(
        self,
        variable: str = "air",
        parameter: Optional[str] = None,
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """
        Plot atmospheric forcing data.
        
        Parameters
        ----------
        variable : str, optional
            Type of atmospheric data ('air', 'rad', 'prc'). Default is 'air'.
        parameter : Optional[str]
            Specific parameter to plot. If None, a suitable parameter is chosen.
        **kwargs : dict
            Additional keyword arguments passed to DataPlotter.plot_atmospheric_spatial.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes.Axes
            The axes object.
        """
        return self.data_plotter.plot_atmospheric_spatial(variable, parameter, **kwargs)

    def plot_tidal_data(self, **kwargs) -> Tuple[Figure, Axes]:
        """
        Plot tidal data.
        
        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed to DataPlotter.plot_tidal_boundaries.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes.Axes
            The axes object.
        """
        return self.data_plotter.plot_tidal_boundaries(**kwargs)

    def plot_gr3_file(
        self,
        file_path: Union[str, Path],
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """
        Plot .gr3 property files with appropriate colormaps.
        
        Parameters
        ----------
        file_path : Union[str, Path]
            Path to .gr3 file.
        **kwargs : dict
            Additional keyword arguments passed to DataPlotter.plot_gr3_file.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes.Axes
            The axes object.
        """
        return self.data_plotter.plot_gr3_file(file_path, **kwargs)

    def plot_bctides_file(
        self,
        file_path: Union[str, Path],
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """
        Plot bctides.in configuration file.
        
        Parameters
        ----------
        file_path : Union[str, Path]
            Path to bctides.in file.
        **kwargs : dict
            Additional keyword arguments passed to DataPlotter.plot_bctides_file.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes.Axes
            The axes object.
        """
        return self.data_plotter.plot_bctides_file(file_path, **kwargs)

    def _has_atmospheric_data(self) -> bool:
        """Check if atmospheric data is available in configuration."""
        if not self.config:
            return False
        return (hasattr(self.config, 'data') and 
                hasattr(self.config.data, 'sflux') and 
                self.config.data.sflux is not None)

    def _has_tidal_data(self) -> bool:
        """Check if tidal data is available in configuration."""
        if not self.config:
            return False
        return (hasattr(self.config, 'data') and 
                hasattr(self.config.data, 'tides') and 
                self.config.data.tides is not None)


# Convenience functions for direct usage
def plot_schism_overview(
    config: Optional[Any] = None,
    grid_file: Optional[Union[str, Path]] = None,
    **kwargs
) -> Tuple[Figure, np.ndarray]:
    """
    Create comprehensive overview plot of SCHISM model setup.
    
    Parameters
    ----------
    config : Optional[Any]
        SCHISM configuration object.
    grid_file : Optional[Union[str, Path]]
        Path to grid file if config is not provided.
    **kwargs : dict
        Additional keyword arguments passed to SchismPlotter.plot_overview.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : numpy.ndarray
        Array of axes objects.
    """
    plotter = SchismPlotter(config=config, grid_file=grid_file)
    return plotter.plot_overview(**kwargs)


def plot_grid(
    config: Optional[Any] = None,
    grid_file: Optional[Union[str, Path]] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot SCHISM grid.
    
    Parameters
    ----------
    config : Optional[Any]
        SCHISM configuration object.
    grid_file : Optional[Union[str, Path]]
        Path to grid file if config is not provided.
    **kwargs : dict
        Additional keyword arguments passed to GridPlotter.plot_grid.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.
    """
    plotter = SchismPlotter(config=config, grid_file=grid_file)
    return plotter.plot_grid(**kwargs)


def plot_boundary_data(
    file_path: Union[str, Path],
    config: Optional[Any] = None,
    grid_file: Optional[Union[str, Path]] = None,
    variable: Optional[str] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot boundary condition data from SCHISM input files.
    
    Parameters
    ----------
    file_path : Union[str, Path]
        Path to boundary data file.
    config : Optional[Any]
        SCHISM configuration object.
    grid_file : Optional[Union[str, Path]]
        Path to grid file if config is not provided.
    variable : Optional[str]
        Variable to plot.
    **kwargs : dict
        Additional keyword arguments.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.
    """
    plotter = SchismPlotter(config=config, grid_file=grid_file)
    return plotter.plot_boundary_data(file_path, variable, **kwargs)


# Export main interface
__all__ = [
    'SchismPlotter',
    'OverviewPlotter',
    'ValidationPlotter',
    'ModelValidator',
    'ValidationResult',
    'plot_schism_overview',
    'plot_grid', 
    'plot_boundary_data',
]