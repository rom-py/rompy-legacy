"""
TROUBLESHOOTING & USAGE NOTES
-----------------------------
If you encounter errors about missing grid/data attributes, logical keys, or file compatibility:

1. Ensure you are using logical keys (see below) in all plotting calls, not raw file paths.
2. Your grid file (hgrid.gr3) must be fully preprocessed and compatible with SCHISM plotting.
   - The grid object must expose boundary information and coordinate arrays (x, y).
   - For advanced plotting (quality metrics, boundary overlays), the grid must have attributes like 'pylibs_hgrid', 'skewness', and 'aspect_ratios'.
3. All required data files must exist in your SCHISM output directory and be referenced in your config.
4. If you see errors about missing attributes, check your preprocessing steps and config references.
5. For a full list of logical keys and grid requirements, see below.

For further help, see the developer documentation or contact the maintainers.

SCHISM Plotting Module

This module provides a unified interface for visualizing SCHISM model input data,
including grid visualization, boundary conditions, atmospheric forcing, tidal data,
and time series animations.

Main Classes:
    SchismPlotter: Primary interface for all SCHISM plotting functionality
    AnimationPlotter: Specialized class for time series animations
    AnimationConfig: Configuration for animation parameters

Main Functions:
    plot_schism_overview: Create comprehensive overview plots
    plot_grid: Plot grid and bathymetry
    plot_boundary_data: Plot boundary condition data
    plot_atmospheric_data: Plot atmospheric forcing data
    plot_tidal_data: Plot tidal data

Animation Functions:
    animate_boundary_data: Create boundary data time series animations
    animate_atmospheric_data: Create atmospheric forcing animations
    animate_grid_data: Create grid-based data animations
    create_multi_variable_animation: Create multi-panel animations

Example Usage:
    >>> from rompy.schism.plotting import SchismPlotter, AnimationConfig
    >>> plotter = SchismPlotter(config)
    >>> fig, ax = plotter.plot_overview()
    >>> fig.show()

    >>> # Create time series animation
    >>> anim_config = AnimationConfig(frame_rate=15, show_time_label=True)
    >>> plotter = SchismPlotter(config, animation_config=anim_config)
    >>> anim = plotter.animate_boundary_data('boundary.th.nc', 'temperature', 'temp_animation.mp4')
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .animation import AnimationConfig, AnimationPlotter
from .core import BasePlotter, PlotConfig
from .data import DataPlotter
from .grid import GridPlotter
from .overview import OverviewPlotter
from .utils import setup_cartopy_axis, validate_file_exists
from .validation import ModelValidator, ValidationPlotter, ValidationResult

logger = logging.getLogger(__name__)


class SchismPlotter:
    """
    Unified interface for SCHISM model visualization.

    Supported Logical Keys (for file_map):
    --------------------------------------------------
    These keys are auto-resolved from the SCHISM config and output directory.
    Use these keys in all plotting calls instead of file paths.

    Boundary Data:
        - "salinity_3d"      : SAL_3D.th.nc
        - "temperature_3d"   : TEM_3D.th.nc
        - "velocity_3d"      : uv3D.th.nc
        - "elevation_2d"     : elev2D.th.nc
    Atmospheric Data:
        - "sflux_air_1"      : sflux/air_1.0001.nc
        - "sflux_air_2"      : sflux/air_2.0001.nc
        - "sflux_rad_1"      : sflux/rad_1.0001.nc
        - "sflux_rad_2"      : sflux/rad_2.0001.nc
        - "sflux_prc_1"      : sflux/prc_1.0001.nc
        - "sflux_prc_2"      : sflux/prc_2.0001.nc
    Tidal Data:
        - "tidal_elevations" : config.data.boundary_conditions.tidal_data.elevations
        - "tidal_velocities" : config.data.boundary_conditions.tidal_data.velocities
        - "tidal_constituents": config.data.boundary_conditions.tidal_data.constituents
    Other Files:
        - "bctides"           : bctides.in
        - "hotstart"          : hotstart.nc
        - "grid"              : hgrid.gr3
        - "property_*"        : *.gr3 property files (e.g., "property_depth", "property_bathymetry")
        - "wave_*"            : wave output files (e.g., "wave_ww3")

    If a logical key is missing, check your SCHISM config and output directory.
    For more details, see the _build_file_map_from_config() method.

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
    animation_plotter : AnimationPlotter
        Animation-specific plotting functionality
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        animation_config: Optional[AnimationConfig] = None,
    ):
        """Initialize SchismPlotter with configuration or grid file.

        If grid_file is not provided, attempts to locate hgrid.gr3 in the prepared model space
        using the config object's output_dir or staging_dir.
        """
        self.config = config
        self.animation_config = animation_config

        #
        # Build file_map from config and output directory
        self.file_map = self._build_file_map_from_config()

        # Initialize sub-plotters with resolved grid file and file_map
        self.grid_plotter = GridPlotter(config=config)
        self.data_plotter = DataPlotter(config=config, file_map=self.file_map)
        self.overview_plotter = OverviewPlotter(config=config)
        self.validation_plotter = ValidationPlotter(config=config)
        self.animation_plotter = AnimationPlotter(
            config=config,
            animation_config=animation_config,
        )

    def _build_file_map_from_config(self) -> dict:
        """
        Auto-resolve all canonical SCHISM output files from config and output directory.
        Returns a mapping of logical keys to file paths for use by DataPlotter and others.
        """
        file_map = {}
        config = self.config
        # Resolve output directory
        output_dir = None
        if config is not None:
            if hasattr(config, "output_dir") and config.output_dir:
                output_dir = Path(config.output_dir)
            elif hasattr(config, "staging_dir") and config.staging_dir:
                output_dir = Path(config.staging_dir)
        if output_dir is None:
            return file_map

        # Grid file
        grid_file = output_dir / "hgrid.gr3"
        if grid_file.exists():
            file_map["grid"] = str(grid_file)

        # Boundary NetCDFs (by convention)
        for fname, key in [
            ("elev2D.th.nc", "elevation_2d"),
            ("uv3D.th.nc", "velocity_3d"),
            ("TEM_3D.th.nc", "temperature_3d"),
            ("SAL_3D.th.nc", "salinity_3d"),
        ]:
            fpath = output_dir / fname
            if fpath.exists():
                file_map[key] = str(fpath)

        # Sflux files (air, rad, prc)
        sflux_dir = output_dir / "sflux"
        if sflux_dir.exists():
            for key, fname in [
                ("sflux_air_1", "air_1.0001.nc"),
                ("sflux_air_2", "air_2.0001.nc"),
                ("sflux_rad_1", "rad_1.0001.nc"),
                ("sflux_rad_2", "rad_2.0001.nc"),
                ("sflux_prc_1", "prc_1.0001.nc"),
                ("sflux_prc_2", "prc_2.0001.nc"),
            ]:
                fpath = sflux_dir / fname
                if fpath.exists():
                    file_map[key] = str(fpath)

        # bctides file
        bctides_file = output_dir / "bctides.in"
        if bctides_file.exists():
            file_map["bctides"] = str(bctides_file)

        # Hotstart file
        hotstart_file = output_dir / "hotstart.nc"
        if hotstart_file.exists():
            file_map["hotstart"] = str(hotstart_file)

        # GR3 property files
        for gr3file in output_dir.glob("*.gr3"):
            file_map[f"property_{gr3file.stem}"] = str(gr3file)

        # Tidal NetCDFs (from config references)
        if config is not None and hasattr(config, "data") and config.data is not None:
            data = config.data
            # Tidal data
            if (
                hasattr(data, "boundary_conditions")
                and data.boundary_conditions is not None
            ):
                bc = data.boundary_conditions
                if hasattr(bc, "tidal_data") and bc.tidal_data is not None:
                    tidal_data = bc.tidal_data
                    if hasattr(tidal_data, "elevations") and tidal_data.elevations:
                        file_map["tidal_elevations"] = str(tidal_data.elevations)
                    if hasattr(tidal_data, "velocities") and tidal_data.velocities:
                        file_map["tidal_velocities"] = str(tidal_data.velocities)
                    if hasattr(tidal_data, "constituents") and tidal_data.constituents:
                        file_map["tidal_constituents"] = tidal_data.constituents
        # Wave files
        if config is not None and hasattr(config, "data") and config.data is not None:
            data = config.data
            if hasattr(data, "wave") and data.wave is not None:
                wave = data.wave
                if hasattr(wave, "id") and hasattr(wave, "outfile"):
                    wave_file = output_dir / f"{wave.id}.nc"
                    if wave_file.exists():
                        file_map[f"wave_{wave.id}"] = str(wave_file)
        return file_map

    def plot_overview(
        self,
        figsize: Tuple[float, float] = (16, 12),
        include_boundaries: bool = True,
        include_bathymetry: bool = True,
        include_atmospheric: bool = True,
        include_tidal: bool = True,
        **kwargs,
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
        fig, axes = plt.subplots(
            2, 2, figsize=figsize, subplot_kw={"projection": setup_cartopy_axis()}
        )
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
            logger.warning("No atmospheric data available for overview plot")
            axes[2].set_title("Atmospheric Forcing")

        # Plot tidal data if available
        if include_tidal and self._has_tidal_data():
            self.data_plotter.plot_tidal_boundaries(ax=axes[3], **kwargs)
            axes[3].set_title("Tidal Boundaries")
        else:
            axes[3].text(
                0.5,
                0.5,
                "No Tidal Data",
                ha="center",
                va="center",
                transform=axes[3].transAxes,
            )
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
        **kwargs,
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
            **kwargs,
        )

    def plot_grid_analysis_overview(
        self,
        figsize: Tuple[float, float] = (16, 12),
        save_path: Optional[Union[str, Path]] = None,
        **kwargs,
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
            figsize=figsize, save_path=save_path, **kwargs
        )

    def plot_data_analysis_overview(
        self,
        figsize: Tuple[float, float] = (16, 12),
        save_path: Optional[Union[str, Path]] = None,
        **kwargs,
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
            figsize=figsize, save_path=save_path, **kwargs
        )

    def plot_validation_summary(
        self,
        figsize: Tuple[float, float] = (14, 10),
        save_path: Optional[Union[str, Path]] = None,
        **kwargs,
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
            figsize=figsize, save_path=save_path, **kwargs
        )

    def plot_quality_assessment(
        self,
        figsize: Tuple[float, float] = (12, 8),
        save_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> Figure:
        """
        Plot quality assessment of model setup.

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
        """
        return self.validation_plotter.plot_quality_assessment(
            figsize=figsize, save_path=save_path, **kwargs
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
        data_type: str,  # logical key, e.g., "salinity_3d", "temperature_3d", "elevation_2d"
        variable: Optional[str] = None,
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        """
        Plot boundary condition data from SCHISM input files using logical key.

        Parameters
        ----------
        data_type : str
            Logical key for boundary data (e.g., "salinity_3d", "temperature_3d", "elevation_2d").
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
        if data_type not in self.file_map:
            raise ValueError(f"File for '{data_type}' not found in file_map.")
        return self.data_plotter.plot_boundary_data(
            self.file_map[data_type], variable, **kwargs
        )

    def plot_atmospheric_data(
        self, variable: str = "air", parameter: Optional[str] = None, **kwargs
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

    def plot_tidal_inputs_at_points(
        self,
        sample_points: Optional[List[Tuple[float, float]]] = None,
        n_points: int = 4,
        time_hours: float = 24.0,
        plot_type: str = "elevation",
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        """
        Plot tidal inputs (elevation or velocity) time series at sample grid points.

        Parameters
        ----------
        sample_points : Optional[List[Tuple[float, float]]]
            List of (lon, lat) coordinates for sample points. If None, representative
            points are automatically selected from boundary nodes.
        n_points : int, optional
            Number of sample points to use if sample_points is None. Default is 4.
        time_hours : float, optional
            Duration in hours for the time series. Default is 24.0 hours.
        plot_type : str, optional
            Type of plot: 'elevation', 'velocity_u', 'velocity_v', 'velocity_magnitude'.
            Default is 'elevation'.
        **kwargs : dict
            Additional keyword arguments passed to DataPlotter.plot_tidal_inputs_at_points.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes.Axes
            The axes object.
        """
        return self.data_plotter.plot_tidal_inputs_at_points(
            sample_points=sample_points,
            n_points=n_points,
            time_hours=time_hours,
            plot_type=plot_type,
            **kwargs,
        )

    def plot_tidal_amplitude_phase_maps(
        self, constituent: str = "M2", variable: str = "elevation", **kwargs
    ) -> Tuple[Figure, Axes]:
        """
        Plot spatial distribution of tidal amplitude and phase for a constituent.

        Parameters
        ----------
        constituent : str, optional
            Tidal constituent to plot. Default is "M2".
        variable : str, optional
            Variable to plot: 'elevation', 'velocity_u', 'velocity_v'. Default is 'elevation'.
        **kwargs : dict
            Additional keyword arguments passed to DataPlotter.plot_tidal_amplitude_phase_maps.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes.Axes or tuple of matplotlib.axes.Axes
            The axes object(s).
        """
        return self.data_plotter.plot_tidal_amplitude_phase_maps(
            constituent=constituent, variable=variable, **kwargs
        )

    def plot_tidal_analysis_overview(
        self,
        figsize: Tuple[float, float] = (20, 12),
        constituents: Optional[List[str]] = None,
        time_hours: float = 24.0,
        n_sample_points: int = 4,
        save_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> Tuple[Figure, Dict[str, Axes]]:
        """
        Create comprehensive tidal analysis overview with multiple panels.

        This creates a detailed tidal analysis with:
        - Tidal amplitude and phase spatial maps
        - Time series at sample boundary points
        - Constituent comparison plots
        - Tidal boundary visualization

        Parameters
        ----------
        figsize : Tuple[float, float], optional
            Figure size in inches. Default is (20, 12).
        constituents : Optional[List[str]], optional
            List of constituents to analyze. If None, uses all available constituents.
        time_hours : float, optional
            Duration in hours for time series plots. Default is 24.0 hours.
        n_sample_points : int, optional
            Number of sample points for time series. Default is 4.
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

        Examples
        --------
        >>> plotter = SchismPlotter(config)
        >>> fig, axes = plotter.plot_tidal_analysis_overview()
        >>> fig.show()
        """
        try:
            # Check if we have tidal data
            if not self._has_tidal_data():
                fig, ax = plt.subplots(figsize=figsize)
                ax.text(
                    0.5,
                    0.5,
                    "No tidal data available for analysis",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title("Tidal Analysis Overview - No Data")
                return fig, {"error": ax}

            # Get available constituents
            if constituents is None:
                if (
                    self.config
                    and hasattr(self.config, "data")
                    and hasattr(self.config.data, "boundary_conditions")
                ):
                    bc = self.config.data.boundary_conditions
                    constituents = bc.constituents[:3]  # Limit to first 3 for display
                else:
                    constituents = ["M2", "S2", "N2"]  # Default constituents

            # Create subplot layout
            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(
                3, 4, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1, 1]
            )

            axes = {}

            # Top row: Amplitude map and boundary points map
            primary_const = constituents[0] if constituents else "M2"

            # Elevation amplitude and boundary points
            ax1 = fig.add_subplot(gs[0, :2], projection=setup_cartopy_axis())
            ax2 = fig.add_subplot(gs[0, 2:], projection=setup_cartopy_axis())

            try:
                # Plot amplitude map
                self.data_plotter.plot_tidal_amplitude_phase_maps(
                    constituent=primary_const, variable="elevation", ax=ax1, **kwargs
                )
                axes["elevation_amp"] = ax1

                # Plot boundary points map showing sample locations
                self._plot_boundary_points_map(ax2, n_sample_points, **kwargs)
                axes["boundary_points"] = ax2

            except Exception as e:
                logger.error(f"Error plotting amplitude map: {e}")
                ax1.set_title("Amplitude Map - Error")
                logger.error(f"Error plotting boundary points: {e}")
                ax2.set_title("Boundary Points - Error")

            # Middle row: Processed data time series at sample points (from bctides.in)
            ax3 = fig.add_subplot(gs[1, :2])
            ax4 = fig.add_subplot(gs[1, 2:])

            try:
                # Processed elevation time series (from bctides.in)
                self.data_plotter.plot_tidal_inputs_at_points(
                    n_points=n_sample_points,
                    time_hours=time_hours,
                    plot_type="elevation",
                    ax=ax3,
                    **kwargs,
                )
                ax3.set_title(
                    "Processed Elevation (from bctides.in)", fontweight="bold"
                )
                axes["processed_elevation"] = ax3

                # Processed velocity magnitude time series (from bctides.in)
                self.data_plotter.plot_tidal_inputs_at_points(
                    n_points=n_sample_points,
                    time_hours=time_hours,
                    plot_type="velocity_magnitude",
                    ax=ax4,
                    **kwargs,
                )
                ax4.set_title("Processed Velocity (from bctides.in)", fontweight="bold")
                axes["processed_velocity"] = ax4

            except Exception as e:
                ax3.text(
                    0.5,
                    0.5,
                    f"Error plotting processed elevation:\n{str(e)[:50]}...",
                    ha="center",
                    va="center",
                    transform=ax3.transAxes,
                    fontsize=8,
                )
                ax4.text(
                    0.5,
                    0.5,
                    f"Error plotting processed velocity:\n{str(e)[:50]}...",
                    ha="center",
                    va="center",
                    transform=ax4.transAxes,
                    fontsize=8,
                )

            # Bottom row: Input data time series for comparison (from original netCDF files)
            ax5 = fig.add_subplot(gs[2, :2])
            ax6 = fig.add_subplot(gs[2, 2:])

            try:
                # Input elevation time series (from original TPXO/netCDF files)
                self._plot_input_data_comparison(
                    ax=ax5,
                    n_points=n_sample_points,
                    time_hours=time_hours,
                    plot_type="elevation",
                    **kwargs,
                )
                ax5.set_title("Input Elevation (from TPXO netCDF)", fontweight="bold")
                axes["input_elevation"] = ax5

                # Input velocity time series (from original TPXO/netCDF files)
                self._plot_input_data_comparison(
                    ax=ax6,
                    n_points=n_sample_points,
                    time_hours=time_hours,
                    plot_type="velocity_magnitude",
                    **kwargs,
                )
                ax6.set_title("Input Velocity (from TPXO netCDF)", fontweight="bold")
                axes["input_velocity"] = ax6

            except Exception as e:
                ax5.text(
                    0.5,
                    0.5,
                    f"Error plotting input elevation:\n{str(e)[:50]}...",
                    ha="center",
                    va="center",
                    transform=ax5.transAxes,
                    fontsize=8,
                )
                ax6.text(
                    0.5,
                    0.5,
                    f"Error plotting input velocity:\n{str(e)[:50]}...",
                    ha="center",
                    va="center",
                    transform=ax6.transAxes,
                    fontsize=8,
                )

            plt.tight_layout()
            fig.suptitle(
                "SCHISM Tidal Analysis: Input vs Processed Data Comparison",
                fontsize=16,
                fontweight="bold",
                y=0.98,
            )

            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches="tight")
                logger.info(f"Tidal analysis overview saved to {save_path}")

            return fig, axes

        except Exception as e:
            logger.error(f"Error creating tidal analysis overview: {e}")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(
                0.5,
                0.5,
                f"Error creating tidal analysis overview:\n{str(e)}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7),
            )
            ax.set_title("Tidal Analysis Overview - Error")
            return fig, {"error": ax}

    def _plot_input_data_comparison(
        self,
        ax,
        n_points: int = 4,
        time_hours: float = 24.0,
        plot_type: str = "elevation",
        **kwargs,
    ):
        """
        Plot input data time series from original netCDF files for comparison.

        This method reads data directly from the original TPXO netCDF files
        to compare with the processed boundary data.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot on
        n_points : int
            Number of sample points to plot
        time_hours : float
            Duration in hours for time series
        plot_type : str
            Type of plot: 'elevation' or 'velocity_magnitude'
        **kwargs : dict
            Additional plotting parameters
        """
        try:
            # Check if we have tidal boundary conditions configuration
            if not (
                self.config
                and hasattr(self.config, "data")
                and hasattr(self.config.data, "boundary_conditions")
            ):
                ax.text(
                    0.5,
                    0.5,
                    "No boundary conditions configuration available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                return

            bc = self.config.data.boundary_conditions
            if not (hasattr(bc, "tidal_data") and bc.tidal_data):
                ax.text(
                    0.5,
                    0.5,
                    "No tidal data files specified in configuration",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                return

            # Get same sample points as processed data for fair comparison
            sample_points = self.data_plotter._get_representative_boundary_points(
                n_points
            )
            if sample_points is None or len(sample_points) == 0:
                logger.warning(
                    "No boundary points available for sampling in input data comparison plot"
                )
                ax.set_title("No Boundary Points")
                return

            # Get data file path
            if plot_type == "elevation":
                data_file = bc.tidal_data.elevations
                ylabel = "Elevation (m)"
            elif plot_type == "velocity_magnitude":
                data_file = bc.tidal_data.velocities
                ylabel = "Velocity Magnitude (m/s)"
            else:
                ax.text(
                    0.5,
                    0.5,
                    f"Unsupported plot type: {plot_type}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                return

            if not Path(data_file).exists():
                logger.error(
                    f"Input data file not found: {Path(data_file).name} for input data comparison plot"
                )
                ax.set_title("Input Data File Not Found")
                return

            # Use the same time series computation methods as the processed data plotter
            import numpy as np

            dt = 0.5  # 30-minute intervals to match processed data
            times = self.data_plotter._compute_standardized_time_axis(time_hours, dt)
            constituents = bc.constituents if bc.constituents else ["M2", "S2", "N2"]

            # Compute time series using existing methods
            if plot_type == "elevation":
                time_series_data = (
                    self.data_plotter._compute_tidal_elevation_timeseries(
                        data_file, sample_points, constituents, times
                    )
                )
            else:  # velocity_magnitude
                time_series_data = self.data_plotter._compute_tidal_velocity_timeseries(
                    data_file, sample_points, constituents, times, plot_type
                )

            # Align data to standardized time axis if needed
            if len(time_series_data) > 0 and len(time_series_data[0]) != len(times):
                original_times = np.arange(0, len(time_series_data[0]) * dt, dt)
                aligned_data = []
                for data_series in time_series_data:
                    aligned_series = self.data_plotter._align_data_to_time_axis(
                        data_series, original_times, times
                    )
                    aligned_data.append(aligned_series)
                time_series_data = aligned_data

            # Plot time series for each point
            from matplotlib import cm

            colors = cm.get_cmap("tab10")(np.linspace(0, 1, len(sample_points)))

            for i, ((lon, lat), color) in enumerate(zip(sample_points, colors)):
                if i < len(time_series_data):
                    ax.plot(
                        times,
                        time_series_data[i],
                        color=color,
                        linewidth=2,
                        label=f"Point {i+1} ({lon:.2f}°, {lat:.2f}°)",
                        alpha=0.8,
                    )

            # Format plot
            ax.set_xlabel("Time (hours)")
            ax.set_ylabel(ylabel)
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
            ax.grid(True, alpha=0.3)

            # Add data source information
            source_text = f"Source: {Path(data_file).name}\nConstituents: {', '.join(constituents)}"
            ax.text(
                0.02,
                0.98,
                source_text,
                transform=ax.transAxes,
                verticalalignment="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
            )

        except Exception as e:
            logger.error(f"Error plotting input data comparison: {e}")
            ax.text(
                0.5,
                0.5,
                f"Error plotting input data:\n{str(e)}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7),
            )

    def _plot_points_map_with_overlays(
        self,
        ax,
        sample_points,
        label_prefix,
        grid=None,
        boundary_overlay=True,
        **kwargs,
    ):
        """
        Plot SCHISM grid, boundary overlays, and sample points with enhanced styling.
        Used by both boundary and atmospheric points maps for code reuse.
        """
        try:
            # Import cartopy for coordinate transforms
            try:
                import cartopy.crs as ccrs

                transform = ccrs.PlateCarree()
            except ImportError:
                transform = None

            # Plot SCHISM grid structure if available
            if grid is not None:
                from .utils import add_grid_overlay

                add_grid_overlay(ax, grid, alpha=0.2, color="lightgray", linewidth=0.3)
                logger.info("Added SCHISM grid mesh overlay")
                if boundary_overlay:
                    from .utils import add_boundary_overlay

                    boundary_colors = {
                        "ocean": "red",
                        "land": "darkgreen",
                        "tidal": "blue",
                    }
                    add_boundary_overlay(
                        ax, grid, boundary_colors=boundary_colors, linewidth=2.5
                    )
                    logger.info("Added SCHISM boundary overlay")

            # Plot sample points with enhanced styling
            lons = [pt[0] for pt in sample_points]
            lats = [pt[1] for pt in sample_points]
            ax.scatter(
                lons,
                lats,
                c="magenta",
                s=250,
                edgecolors="black",
                linewidth=3,
                marker="o",
                alpha=1.0,
                zorder=20,
                transform=transform,
                label="Sample Points",
            )
            for i, (lon, lat) in enumerate(sample_points):
                ax.annotate(
                    f"{label_prefix}{i+1}",
                    (lon, lat),
                    xytext=(8, 8),
                    textcoords="offset points",
                    fontsize=13,
                    fontweight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="yellow",
                        alpha=1.0,
                        edgecolor="black",
                        linewidth=2,
                    ),
                    transform=transform,
                    zorder=25,
                )
            ax.legend(loc="upper right", fontsize=10)
            if len(lons) > 0 and len(lats) > 0:
                lon_range = max(lons) - min(lons)
                lat_range = max(lats) - min(lats)
                lon_margin = max(lon_range * 0.15, 0.5)
                lat_margin = max(lat_range * 0.15, 0.5)
                ax.set_xlim(min(lons) - lon_margin, max(lons) + lon_margin)
                ax.set_ylim(min(lats) - lat_margin, max(lats) + lat_margin)
            ax.set_title(
                f"SCHISM Grid & {label_prefix} Points (n={len(sample_points)})",
                fontsize=12,
                fontweight="bold",
            )
            ax.set_xlabel("Longitude", fontsize=10)
            ax.set_ylabel("Latitude", fontsize=10)
            legend_text = "\n".join(
                [
                    f"{label_prefix}{i+1}: ({lon:.3f}°, {lat:.3f}°)"
                    for i, (lon, lat) in enumerate(sample_points)
                ]
            )
            ax.text(
                0.02,
                0.98,
                legend_text,
                transform=ax.transAxes,
                verticalalignment="top",
                fontsize=8,
                fontfamily="monospace",
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor="lightblue",
                    alpha=0.9,
                    edgecolor="navy",
                    linewidth=1,
                ),
            )
            if boundary_overlay:
                try:
                    from matplotlib.lines import Line2D

                    legend_elements = [
                        Line2D([0], [0], color="red", lw=3, label="Ocean Boundary"),
                        Line2D(
                            [0], [0], color="darkgreen", lw=3, label="Land Boundary"
                        ),
                        Line2D([0], [0], color="lightgray", lw=1, label="Grid Mesh"),
                    ]
                    ax.legend(
                        handles=legend_elements,
                        loc="lower right",
                        fontsize=8,
                        frameon=True,
                        fancybox=True,
                        shadow=True,
                    )
                except Exception as e:
                    logger.debug(f"Could not add boundary legend: {e}")
        except Exception as e:
            logger.error(f"Error plotting SCHISM grid and points: {e}")
            ax.text(
                0.5,
                0.5,
                f"Error plotting SCHISM grid:\n{str(e)}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
            )
            ax.set_title("SCHISM Grid - Error")

    def _plot_boundary_points_map(self, ax, n_points: int = 4, **kwargs):
        """
        Plot SCHISM grid and boundary points map showing sample locations used for time series.
        """
        try:
            sample_points = self.data_plotter._get_representative_boundary_points(
                n_points
            )
            if sample_points is None or len(sample_points) == 0:
                logger.warning(
                    "No boundary points available for boundary points map plot"
                )
                ax.set_title("SCHISM Grid - No Data")
                return
            grid = self.grid_plotter.grid
            self._plot_points_map_with_overlays(
                ax,
                sample_points,
                label_prefix="P",
                grid=grid,
                boundary_overlay=True,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"Error plotting SCHISM grid and boundary points: {e}")
            ax.set_title("SCHISM Grid - Error")

    def plot_atmospheric_analysis_overview(
        self,
        figsize: Tuple[float, float] = (20, 12),
        time_hours: float = 24.0,
        n_sample_points: int = 4,
        plot_type: str = "wind_speed",
        variable: str = "air",
        save_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> Tuple[Figure, Dict[str, Axes]]:
        """
        Create comprehensive atmospheric analysis overview with input vs processed data comparison.

        This creates a detailed atmospheric analysis with:
        - Atmospheric spatial distribution maps
        - Time series at sample points from input data
        - Time series at sample points from processed sflux data
        - Atmospheric data visualization

        Parameters
        ----------
        figsize : Tuple[float, float], optional
            Figure size in inches. Default is (20, 12).
        time_hours : float, optional
            Duration in hours for time series plots. Default is 24.0 hours.
        n_sample_points : int, optional
            Number of sample points for time series. Default is 4.
        plot_type : str, optional
            Type of atmospheric plot: 'wind_speed', 'wind_u', 'wind_v', 'pressure', 'temperature'.
            Default is 'wind_speed'.
        variable : str, optional
            Atmospheric variable type: 'air', 'rad', 'prc'. Default is 'air'.
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

        Examples
        --------
        >>> plotter = SchismPlotter(config)
        >>> fig, axes = plotter.plot_atmospheric_analysis_overview()
        >>> fig.show()
        """
        try:
            # Check if we have atmospheric data
            if not self._has_atmospheric_data():
                fig, ax = plt.subplots(figsize=figsize)
                ax.text(
                    0.5,
                    0.5,
                    "No atmospheric data available for analysis",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title("Atmospheric Analysis Overview - No Data")
                return fig, {"error": ax}

            # Create subplot layout
            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1])

            axes = {}

            # Top row: Atmospheric spatial maps
            ax1 = fig.add_subplot(gs[0, :2], projection=setup_cartopy_axis())
            ax2 = fig.add_subplot(gs[0, 2:], projection=setup_cartopy_axis())

            try:
                # Plot atmospheric spatial distribution
                self.data_plotter.plot_atmospheric_spatial(
                    variable=variable, ax=ax1, **kwargs
                )
                axes["atmospheric_spatial"] = ax1

                # Plot atmospheric sample points map
                self._plot_atmospheric_points_map(ax2, n_sample_points, **kwargs)
                axes["sample_points"] = ax2

            except Exception as e:
                logger.error(f"Error plotting atmospheric spatial: {e}")
                ax1.set_title("Atmospheric Spatial - Error")
                logger.error(f"Error plotting sample points: {e}")
                ax2.set_title("Sample Points - Error")

            # Bottom row: Input vs processed data comparison
            ax3 = fig.add_subplot(gs[1, :2])
            ax4 = fig.add_subplot(gs[1, 2:])

            try:
                # Input atmospheric data time series
                self.data_plotter.plot_atmospheric_inputs_at_points(
                    n_points=n_sample_points,
                    time_hours=time_hours,
                    plot_type=plot_type,
                    variable=variable,
                    ax=ax3,
                    **kwargs,
                )
                ax3.set_title(
                    f"Input {plot_type.replace('_', ' ').title()} (from {variable.upper()} data)",
                    fontweight="bold",
                )
                axes["input_data"] = ax3

                # Processed atmospheric data time series
                self.data_plotter.plot_processed_atmospheric_data(
                    n_points=n_sample_points,
                    time_hours=time_hours,
                    plot_type=plot_type,
                    variable=variable,
                    ax=ax4,
                    **kwargs,
                )
                ax4.set_title(
                    f"Processed {plot_type.replace('_', ' ').title()} (from sflux files)",
                    fontweight="bold",
                )
                axes["processed_data"] = ax4

            except Exception as e:
                ax3.text(
                    0.5,
                    0.5,
                    f"Error plotting input data:\n{str(e)[:50]}...",
                    ha="center",
                    va="center",
                    transform=ax3.transAxes,
                    fontsize=8,
                )
                ax4.text(
                    0.5,
                    0.5,
                    f"Error plotting processed data:\n{str(e)[:50]}...",
                    ha="center",
                    va="center",
                    transform=ax4.transAxes,
                    fontsize=8,
                )

            plt.tight_layout()
            fig.suptitle(
                f"SCHISM Atmospheric Analysis: Input vs Processed Data Comparison ({variable.upper()})",
                fontsize=16,
                fontweight="bold",
                y=0.98,
            )

            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches="tight")
                logger.info(f"Atmospheric analysis overview saved to {save_path}")

            return fig, axes

        except Exception as e:
            logger.error(f"Error creating atmospheric analysis overview: {e}")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(
                0.5,
                0.5,
                f"Error creating atmospheric analysis overview:\n{str(e)}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7),
            )
            ax.set_title("Atmospheric Analysis Overview - Error")
            return fig, {"error": ax}

    def plot_ocean_boundary_analysis_overview(
        self,
        figsize: Tuple[float, float] = (20, 12),
        time_hours: float = 24.0,
        n_sample_points: int = 4,
        plot_type: str = "elevation",
        boundary_type: str = "2d",
        save_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> Tuple[Figure, Dict[str, Axes]]:
        """
        Create comprehensive ocean boundary analysis overview with input vs processed data comparison.

        This creates a detailed ocean boundary analysis with:
        - Ocean boundary spatial visualization
        - Time series at sample boundary points from input data
        - Time series at sample boundary points from processed SCHISM files
        - Ocean boundary condition visualization

        Parameters
        ----------
        figsize : Tuple[float, float], optional
            Figure size in inches. Default is (20, 12).
        time_hours : float, optional
            Duration in hours for time series plots. Default is 24.0 hours.
        n_sample_points : int, optional
            Number of sample points for time series. Default is 4.
        plot_type : str, optional
            Type of ocean boundary plot: 'elevation', 'velocity_u', 'velocity_v',
            'velocity_magnitude', 'temperature', 'salinity'. Default is 'elevation'.
        boundary_type : str, optional
            Type of boundary: '2d' or '3d'. Default is '2d'.
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

        Examples
        --------
        >>> plotter = SchismPlotter(config)
        >>> fig, axes = plotter.plot_ocean_boundary_analysis_overview()
        >>> fig.show()
        """
        try:
            # Check if we have ocean boundary data
            if not self._has_ocean_boundary_data():
                fig, ax = plt.subplots(figsize=figsize)
                ax.text(
                    0.5,
                    0.5,
                    "No ocean boundary data available for analysis",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title("Ocean Boundary Analysis Overview - No Data")
                return fig, {"error": ax}

            # Create subplot layout
            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1])

            axes = {}

            # Top row: Ocean boundary spatial maps
            ax1 = fig.add_subplot(gs[0, :2], projection=setup_cartopy_axis())
            ax2 = fig.add_subplot(gs[0, 2:], projection=setup_cartopy_axis())

            try:
                # Plot boundary locations
                self.grid_plotter.plot_boundaries(ax=ax1, **kwargs)
                axes["boundary_locations"] = ax1

                # Plot boundary sample points map
                self._plot_boundary_points_map(ax2, n_sample_points, **kwargs)
                axes["sample_points"] = ax2

            except Exception as e:
                logger.error(f"Error plotting boundary locations: {e}")
                ax1.set_title("Boundary Locations - Error")
                logger.error(f"Error plotting sample points: {e}")
                ax2.set_title("Sample Points - Error")

            # Bottom row: Input vs processed data comparison
            ax3 = fig.add_subplot(gs[1, :2])
            ax4 = fig.add_subplot(gs[1, 2:])

            try:
                # Input ocean boundary data time series
                self.data_plotter.plot_ocean_boundary_inputs_at_points(
                    n_points=n_sample_points,
                    time_hours=time_hours,
                    plot_type=plot_type,
                    boundary_type=boundary_type,
                    ax=ax3,
                    **kwargs,
                )
                ax3.set_title(
                    f"Input {boundary_type.upper()} {plot_type.replace('_', ' ').title()} (from netCDF)",
                    fontweight="bold",
                )
                axes["input_data"] = ax3

                # Processed ocean boundary data time series
                self.data_plotter.plot_processed_ocean_boundary_data(
                    n_points=n_sample_points,
                    time_hours=time_hours,
                    plot_type=plot_type,
                    boundary_type=boundary_type,
                    ax=ax4,
                    **kwargs,
                )
                ax4.set_title(
                    f"Processed {boundary_type.upper()} {plot_type.replace('_', ' ').title()} (from *.th.nc)",
                    fontweight="bold",
                )
                axes["processed_data"] = ax4

            except Exception as e:
                ax3.text(
                    0.5,
                    0.5,
                    f"Error plotting input data:\n{str(e)[:50]}...",
                    ha="center",
                    va="center",
                    transform=ax3.transAxes,
                    fontsize=8,
                )
                ax4.text(
                    0.5,
                    0.5,
                    f"Error plotting processed data:\n{str(e)[:50]}...",
                    ha="center",
                    va="center",
                    transform=ax4.transAxes,
                    fontsize=8,
                )

            plt.tight_layout()
            fig.suptitle(
                f"SCHISM Ocean Boundary Analysis: Input vs Processed Data Comparison ({boundary_type.upper()})",
                fontsize=16,
                fontweight="bold",
                y=0.98,
            )

            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches="tight")
                logger.info(f"Ocean boundary analysis overview saved to {save_path}")

            return fig, axes

        except Exception as e:
            logger.error(f"Error creating ocean boundary analysis overview: {e}")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(
                0.5,
                0.5,
                f"Error creating ocean boundary analysis overview:\n{str(e)}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7),
            )
            ax.set_title("Ocean Boundary Analysis Overview - Error")
            return fig, {"error": ax}

    def _plot_atmospheric_points_map(self, ax, n_points: int = 4, **kwargs):
        """
        Plot atmospheric sample points map showing locations used for time series.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot on (should have cartopy projection)
        n_points : int
            Number of sample points to show
        **kwargs : dict
            Additional plotting parameters
        """
        try:
            # Get sample points from data plotter
            sample_points = self.data_plotter._get_representative_atmospheric_points(
                n_points
            )
            if sample_points is None or len(sample_points) == 0:
                logger.warning(
                    "No atmospheric sample points available for atmospheric points map plot"
                )
                ax.set_title("Atmospheric Sample Points - No Data")
                return
            grid = self.grid_plotter.grid
            self._plot_points_map_with_overlays(
                ax,
                sample_points,
                label_prefix="A",
                grid=grid,
                boundary_overlay=True,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"Error plotting atmospheric sample points: {e}")
            ax.text(
                0.5,
                0.5,
                f"Error plotting atmospheric sample points:\n{str(e)}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
            )
            ax.set_title("Atmospheric Sample Points - Error")

    def _has_ocean_boundary_data(self) -> bool:
        """Check if ocean boundary data is available in configuration."""
        if not self.config:
            return False

        if not (hasattr(self.config, "data") and self.config.data):
            return False

        # Check for boundary conditions data
        if (
            hasattr(self.config.data, "boundary_conditions")
            and self.config.data.boundary_conditions is not None
        ):
            bc = self.config.data.boundary_conditions

            # Check for explicit boundaries definition
        if hasattr(bc, "boundaries") and bc.boundaries and bool(bc.boundaries):
            return True

            # Check for tidal setup (which creates ocean boundaries)
            if (
                hasattr(bc, "setup_type")
                and bc.setup_type in ["tidal", "hybrid"]
                and hasattr(bc, "constituents")
                and bc.constituents
            ):
                return True

            # Check for tidal data files
            if hasattr(bc, "tidal_data") and bc.tidal_data:
                return True

        return False

    def plot_schism_boundary_data(
        self,
        bctides_file: Union[str, Path],
        plot_type: str = "elevation",
        time_hours: float = 24.0,
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        """
        Plot actual SCHISM boundary data from bctides.in file.

        This plots the processed boundary data that SCHISM will actually use,
        rather than the input TPXO data.

        Parameters
        ----------
        bctides_file : Union[str, Path]
            Path to bctides.in file
        plot_type : str, optional
            Type of plot: 'elevation', 'velocity', 'summary'. Default is 'elevation'.
        time_hours : float, optional
            Duration in hours for time series. Default is 24.0 hours.
        **kwargs : dict
            Additional keyword arguments passed to DataPlotter.plot_schism_boundary_data.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes.Axes
            The axes object.
        """
        return self.data_plotter.plot_schism_boundary_data(
            bctides_file=bctides_file,
            plot_type=plot_type,
            time_hours=time_hours,
            **kwargs,
        )

    def plot_gr3_file(
        self,
        property_key: str,  # logical key, e.g., "property_depth", "property_bathymetry"
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        """
        Plot .gr3 property files using logical key (auto-resolved from file_map).

        Parameters
        ----------
        property_key : str
            Logical key for property file (e.g., "property_depth", "property_bathymetry").
        **kwargs : dict
            Additional keyword arguments passed to DataPlotter.plot_gr3_file.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes.Axes
            The axes object.
        """
        if property_key not in self.file_map:
            raise ValueError(
                f"Property file for '{property_key}' not found in file_map."
            )
        return self.data_plotter.plot_gr3_file(self.file_map[property_key], **kwargs)

    def plot_bctides_file(
        self, bctides_key: str = "bctides", **kwargs
    ) -> Tuple[Figure, Axes]:
        """
        Plot bctides.in configuration file using logical key (auto-resolved from file_map).

        Parameters
        ----------
        bctides_key : str, optional
            Logical key for bctides file (default is "bctides").
        **kwargs : dict
            Additional keyword arguments passed to DataPlotter.plot_bctides_file.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes.Axes
            The axes object.
        """
        if bctides_key not in self.file_map:
            raise ValueError(f"bctides file for '{bctides_key}' not found in file_map.")
        return self.data_plotter.plot_bctides_file(self.file_map[bctides_key], **kwargs)

    def _has_atmospheric_data(self) -> bool:
        """Check if atmospheric data is available in configuration."""
        if not self.config:
            return False

        # Check for atmospheric data in different possible locations
        if hasattr(self.config, "data") and self.config.data:
            # Check for sflux-style data
            if hasattr(self.config.data, "sflux") and self.config.data.sflux not in [
                None,
                {},
                [],
                "",
            ]:
                return True

            # Check for atmos-style data (more common in newer configurations)
            if hasattr(self.config.data, "atmos") and self.config.data.atmos not in [
                None,
                {},
                [],
                "",
            ]:
                # Check if any atmospheric data sources are available
                if (
                    hasattr(self.config.data.atmos, "air_1")
                    and self.config.data.atmos.air_1
                ):
                    return True
                if (
                    hasattr(self.config.data.atmos, "air")
                    and self.config.data.atmos.air
                ):
                    return True
                if (
                    hasattr(self.config.data.atmos, "rad_1")
                    and self.config.data.atmos.rad_1
                ):
                    return True
                if (
                    hasattr(self.config.data.atmos, "prc_1")
                    and self.config.data.atmos.prc_1
                ):
                    return True

        return False

    def _has_tidal_data(self) -> bool:
        """Check if tidal data is available in configuration."""
        if not self.config:
            return False
        return (
            hasattr(self.config, "data")
            and hasattr(self.config.data, "boundary_conditions")
            and self.config.data.boundary_conditions is not None
            and hasattr(self.config.data.boundary_conditions, "setup_type")
            and self.config.data.boundary_conditions.setup_type in ["tidal", "hybrid"]
            and hasattr(self.config.data.boundary_conditions, "constituents")
            and self.config.data.boundary_conditions.constituents
        )

    # Animation methods
    def animate_boundary_data(
        self,
        data_file: Union[str, Path],
        variable: str,
        output_file: Optional[Union[str, Path]] = None,
        level_idx: int = 0,
        **kwargs,
    ):
        """
        Create animation of boundary data time evolution.

        Parameters
        ----------
        data_file : Union[str, Path]
            Path to boundary data file (*.th.nc)
        variable : str
            Variable name to animate
        output_file : Optional[Union[str, Path]]
            Path to save animation (MP4 or GIF)
        level_idx : int, optional
            Level index for 3D data. Default is 0.
        **kwargs : dict
            Additional plotting parameters

        Returns
        -------
        animation.FuncAnimation
            Matplotlib animation object
        """
        return self.animation_plotter.animate_boundary_data(
            data_file, variable, output_file, level_idx, **kwargs
        )

    def animate_atmospheric_data(
        self,
        data_file: Union[str, Path],
        variable: str = "air",
        parameter: Optional[str] = None,
        output_file: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        """
        Create animation of atmospheric forcing data progression.

        Parameters
        ----------
        data_file : Union[str, Path]
            Path to atmospheric data file
        variable : str, optional
            Type of atmospheric data ('air', 'rad', 'prc'). Default is 'air'.
        parameter : Optional[str]
            Specific parameter to animate
        output_file : Optional[Union[str, Path]]
            Path to save animation (MP4 or GIF)
        **kwargs : dict
            Additional plotting parameters

        Returns
        -------
        animation.FuncAnimation
            Matplotlib animation object
        """
        return self.animation_plotter.animate_atmospheric_data(
            data_file, variable, parameter, output_file, **kwargs
        )

    def animate_grid_data(
        self,
        data_file: Union[str, Path],
        variable: str,
        output_file: Optional[Union[str, Path]] = None,
        show_grid: bool = True,
        **kwargs,
    ):
        """
        Create animation of grid-based data with spatial-temporal visualization.

        Parameters
        ----------
        data_file : Union[str, Path]
            Path to data file with grid-based temporal data
        variable : str
            Variable name to animate
        output_file : Optional[Union[str, Path]]
            Path to save animation (MP4 or GIF)
        show_grid : bool, optional
            Whether to show computational grid. Default is True.
        **kwargs : dict
            Additional plotting parameters

        Returns
        -------
        animation.FuncAnimation
            Matplotlib animation object
        """
        return self.animation_plotter.animate_grid_data(
            data_file, variable, output_file, show_grid, **kwargs
        )

    def create_multi_variable_animation(
        self,
        data_files: Dict[str, Union[str, Path]],
        variables: Dict[str, str],
        output_file: Optional[Union[str, Path]] = None,
        layout: str = "grid",
        **kwargs,
    ):
        """
        Create multi-panel animation with multiple variables.

        Parameters
        ----------
        data_files : Dict[str, Union[str, Path]]
            Dictionary mapping panel names to data file paths
        variables : Dict[str, str]
            Dictionary mapping panel names to variable names
        output_file : Optional[Union[str, Path]]
            Path to save animation (MP4 or GIF)
        layout : str, optional
            Panel layout: 'grid', 'vertical', 'horizontal'. Default is 'grid'.
        **kwargs : dict
            Additional plotting parameters

        Returns
        -------
        animation.FuncAnimation
            Matplotlib animation object
        """
        return self.animation_plotter.create_multi_variable_animation(
            data_files, variables, output_file, layout, **kwargs
        )

    def stop_animation(self) -> None:
        """Stop the current animation."""
        self.animation_plotter.stop_animation()

    def pause_animation(self) -> None:
        """Pause the current animation."""
        self.animation_plotter.pause_animation()

    def resume_animation(self) -> None:
        """Resume the current animation."""
        self.animation_plotter.resume_animation()


# Convenience functions for direct usage
def plot_schism_overview(
    config: Optional[Any] = None, grid_file: Optional[Union[str, Path]] = None, **kwargs
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
    config: Optional[Any] = None, grid_file: Optional[Union[str, Path]] = None, **kwargs
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
    data_type: str,  # logical key, e.g., "salinity_3d", "temperature_3d", "elevation_2d"
    config: Optional[Any] = None,
    grid_file: Optional[Union[str, Path]] = None,
    variable: Optional[str] = None,
    **kwargs,
) -> Tuple[Figure, Axes]:
    """
    Plot boundary condition data from SCHISM input files using logical key.

    Parameters
    ----------
    data_type : str
        Logical key for boundary data (e.g., "salinity_3d", "temperature_3d", "elevation_2d").
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
    return plotter.plot_boundary_data(data_type, variable, **kwargs)


def plot_gr3_file(
    property_key: str,
    config: Optional[Any] = None,
    grid_file: Optional[Union[str, Path]] = None,
    **kwargs,
) -> Tuple[Figure, Axes]:
    """
    Plot .gr3 property file using logical key (auto-resolved from file_map).

    Parameters
    ----------
    property_key : str
        Logical key for property file (e.g., "property_depth", "property_bathymetry").
    config : Optional[Any]
        SCHISM configuration object.
    grid_file : Optional[Union[str, Path]]
        Path to grid file if config is not provided.
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
    return plotter.plot_gr3_file(property_key, **kwargs)


def plot_bctides_file(
    bctides_key: str = "bctides",
    config: Optional[Any] = None,
    grid_file: Optional[Union[str, Path]] = None,
    **kwargs,
) -> Tuple[Figure, Axes]:
    """
    Plot bctides.in configuration file using logical key (auto-resolved from file_map).

    Parameters
    ----------
    bctides_key : str, optional
        Logical key for bctides file (default is "bctides").
    config : Optional[Any]
        SCHISM configuration object.
    grid_file : Optional[Union[str, Path]]
        Path to grid file if config is not provided.
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
    return plotter.plot_bctides_file(bctides_key, **kwargs)


# Export main interface
__all__ = [
    "SchismPlotter",
    "OverviewPlotter",
    "ValidationPlotter",
    "ModelValidator",
    "ValidationResult",
    "AnimationPlotter",
    "AnimationConfig",
    "plot_schism_overview",
    "plot_grid",
    "plot_boundary_data",
    "plot_gr3_file",
    "plot_bctides_file",
]
