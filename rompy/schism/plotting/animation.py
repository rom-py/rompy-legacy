"""
Animation plotting functionality for SCHISM input files.

This module provides time series animation capabilities for SCHISM input data
including boundary conditions, atmospheric forcing, and grid-based temporal data.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
from matplotlib.artist import Artist

from .core import BasePlotter, PlotConfig
from .utils import (
    detect_file_type,
    get_variable_info,
    load_schism_data,
    setup_colormap,
    add_boundary_overlay,
    get_geographic_extent,
)

logger = logging.getLogger(__name__)


class AnimationConfig(PlotConfig):
    """
    Configuration model for SCHISM animation parameters.

    Extends PlotConfig with animation-specific parameters.
    """

    # Animation timing parameters
    frame_rate: int = 10  # frames per second
    duration: Optional[float] = None  # total duration in seconds
    interval: int = 100  # milliseconds between frames
    repeat: bool = True  # whether to repeat animation

    # Time range parameters
    time_start: Optional[str] = None  # start time (ISO format)
    time_end: Optional[str] = None  # end time (ISO format)
    time_step: Optional[int] = None  # time step size

    # Animation quality parameters
    bitrate: int = 1800  # video bitrate for MP4
    quality: str = "medium"  # animation quality: low, medium, high

    # Animation controls
    show_time_label: bool = True  # show current time on plot
    time_label_format: str = "%Y-%m-%d %H:%M"  # time label format
    time_label_position: Tuple[float, float] = (0.02, 0.98)  # position in axes coords

    # Progress tracking
    show_progress: bool = True  # show progress bar during creation

    @property
    def effective_interval(self) -> int:
        """Calculate effective interval from frame rate."""
        return int(1000 / self.frame_rate) if self.frame_rate > 0 else self.interval


class AnimationPlotter(BasePlotter):
    """
    SCHISM animation plotting functionality.

    This class provides methods for creating time series animations of SCHISM
    input data including boundary conditions, atmospheric forcing, and temporal data.

    Parameters
    ----------
    config : Optional[Any]
        SCHISM configuration object
    grid_file : Optional[Union[str, Path]]
        Path to grid file if config is not provided
    animation_config : Optional[AnimationConfig]
        Animation configuration parameters
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        grid_file: Optional[Union[str, Path]] = None,
        animation_config: Optional[AnimationConfig] = None
    ):
        """Initialize animation plotter."""
        self.animation_config = animation_config or AnimationConfig()
        super().__init__(config, grid_file, self.animation_config)

        # Animation state
        self._current_animation = None
        self._animation_data = None
        self._time_text = None

    def plot(self, *args, **kwargs):
        """
        Implement abstract plot method from BasePlotter.

        This method is not used directly in AnimationPlotter as animations
        are created through specific animation methods.
        """
        raise NotImplementedError(
            "AnimationPlotter uses specific animation methods like "
            "animate_boundary_data, animate_atmospheric_data, etc. "
            "Use these methods instead of the generic plot method."
        )

    def animate_boundary_data(
        self,
        data_file: Union[str, Path],
        variable: str,
        output_file: Optional[Union[str, Path]] = None,
        level_idx: int = 0,
        **kwargs
    ) -> animation.FuncAnimation:
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
        logger.info(f"Creating boundary data animation for {variable}")

        # Load data
        ds = load_schism_data(data_file)

        # Get the correct variable name
        actual_variable = self._get_actual_variable_name(ds, variable)
        data_var = ds[actual_variable]

        if 'time' not in data_var.dims:
            raise ValueError("Data must have time dimension for animation")

        # Get time range
        time_indices = self._get_time_indices(ds)
        times = ds['time'].values[time_indices]

        # Set up figure and axes
        fig, ax = self.create_figure(use_cartopy=True, **kwargs)

        # Initialize plot elements
        self._setup_animation_plot(ax, ds, variable, **kwargs)

        # Create animation function
        def animate_frame(frame_idx: int) -> List[Artist]:
            """Animation function for each frame."""
            return self._animate_boundary_frame(
                ax, ds, data_var, variable, time_indices[frame_idx],
                times[frame_idx], level_idx, **kwargs
            )

        # Create animation
        anim = animation.FuncAnimation(
            fig, animate_frame, frames=len(time_indices),
            interval=self.animation_config.effective_interval,
            repeat=self.animation_config.repeat,
            blit=False  # Disable blitting for complex plots
        )

        # Save animation if requested
        if output_file:
            self._save_animation(anim, output_file)

        self._current_animation = anim
        return anim

    def animate_atmospheric_data(
        self,
        data_file: Union[str, Path],
        variable: str = "air",
        parameter: Optional[str] = None,
        output_file: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> animation.FuncAnimation:
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
        logger.info(f"Creating atmospheric data animation for {variable}")

        # Load data
        ds = load_schism_data(data_file)

        # Determine parameter to plot
        if not parameter:
            if variable == "air":
                parameter = "air_temperature" if "air_temperature" in ds.data_vars else list(ds.data_vars)[0]
            else:
                parameter = list(ds.data_vars)[0]

        if parameter not in ds.data_vars:
            raise ValueError(f"Parameter {parameter} not found in dataset")

        data_var = ds[parameter]
        if 'time' not in data_var.dims:
            raise ValueError("Data must have time dimension for animation")

        # Get time range
        time_indices = self._get_time_indices(ds)
        times = ds['time'].values[time_indices]

        # Set up figure and axes
        fig, ax = self.create_figure(use_cartopy=True, **kwargs)

        # Initialize plot elements
        self._setup_animation_plot(ax, ds, parameter, **kwargs)

        # Create animation function
        def animate_frame(frame_idx: int) -> List[Artist]:
            """Animation function for each frame."""
            return self._animate_atmospheric_frame(
                ax, ds, data_var, parameter, time_indices[frame_idx],
                times[frame_idx], **kwargs
            )

        # Create animation
        anim = animation.FuncAnimation(
            fig, animate_frame, frames=len(time_indices),
            interval=self.animation_config.effective_interval,
            repeat=self.animation_config.repeat,
            blit=False
        )

        # Save animation if requested
        if output_file:
            self._save_animation(anim, output_file)

        self._current_animation = anim
        return anim

    def animate_grid_data(
        self,
        data_file: Union[str, Path],
        variable: str,
        output_file: Optional[Union[str, Path]] = None,
        show_grid: bool = True,
        **kwargs
    ) -> animation.FuncAnimation:
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
        logger.info(f"Creating grid data animation for {variable}")

        # Load data
        ds = load_schism_data(data_file)

        # Get the correct variable name
        actual_variable = self._get_actual_variable_name(ds, variable)
        data_var = ds[actual_variable]
        if 'time' not in data_var.dims:
            raise ValueError("Data must have time dimension for animation")

        # Get time range
        time_indices = self._get_time_indices(ds)
        times = ds['time'].values[time_indices]

        # Set up figure and axes
        fig, ax = self.create_figure(use_cartopy=True, **kwargs)

        # Initialize plot elements
        self._setup_animation_plot(ax, ds, actual_variable, show_grid=show_grid, **kwargs)

        # Create animation function
        def animate_frame(frame_idx: int) -> List[Artist]:
            """Animation function for each frame."""
            return self._animate_grid_frame(
                ax, ds, data_var, actual_variable, time_indices[frame_idx],
                times[frame_idx], show_grid, **kwargs
            )

        # Create animation
        anim = animation.FuncAnimation(
            fig, animate_frame, frames=len(time_indices),
            interval=self.animation_config.effective_interval,
            repeat=self.animation_config.repeat,
            blit=False
        )

        # Save animation if requested
        if output_file:
            self._save_animation(anim, output_file)

        self._current_animation = anim
        return anim

    def create_multi_variable_animation(
        self,
        data_files: Dict[str, Union[str, Path]],
        variables: Dict[str, str],
        output_file: Optional[Union[str, Path]] = None,
        layout: str = "grid",
        **kwargs
    ) -> animation.FuncAnimation:
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
        logger.info("Creating multi-variable animation")

        # Load all datasets
        datasets = {}
        for panel_name, data_file in data_files.items():
            datasets[panel_name] = load_schism_data(data_file)

        # Determine common time range
        time_indices = self._get_common_time_indices(datasets)

        # Set up multi-panel figure
        fig, axes = self._create_multi_panel_figure(len(data_files), layout, **kwargs)

        # Create mapping between panel indices and data file names
        panel_names = list(data_files.keys())

        # Initialize all plot elements
        for i, (panel_key, ax) in enumerate(axes.items()):
            panel_name = panel_names[i]
            ds = datasets[panel_name]
            variable = variables[panel_name]
            actual_variable = self._get_actual_variable_name(ds, variable)
            self._setup_animation_plot(ax, ds, actual_variable, title=panel_name, **kwargs)

        # Create animation function
        def animate_frame(frame_idx: int) -> List[Artist]:
            """Animation function for each frame."""
            artists = []
            for i, (panel_key, ax) in enumerate(axes.items()):
                panel_name = panel_names[i]
                ds = datasets[panel_name]
                variable = variables[panel_name]
                actual_variable = self._get_actual_variable_name(ds, variable)
                data_var = ds[actual_variable]
                time_idx = time_indices[frame_idx]
                time_val = ds['time'].values[time_idx]

                frame_artists = self._animate_multi_frame(
                    ax, ds, data_var, actual_variable, time_idx, time_val, **kwargs
                )
                artists.extend(frame_artists)
            return artists

        # Create animation
        anim = animation.FuncAnimation(
            fig, animate_frame, frames=len(time_indices),
            interval=self.animation_config.effective_interval,
            repeat=self.animation_config.repeat,
            blit=False
        )

        # Save animation if requested
        if output_file:
            self._save_animation(anim, output_file)

        self._current_animation = anim
        return anim

    def _get_time_indices(self, ds: xr.Dataset) -> np.ndarray:
        """Get time indices for animation based on configuration."""
        times = ds['time'].values

        # Apply time range filtering
        if self.animation_config.time_start or self.animation_config.time_end:
            start_idx = 0
            end_idx = len(times)

            if self.animation_config.time_start:
                start_time = np.datetime64(self.animation_config.time_start)
                start_idx = np.searchsorted(times, start_time)

            if self.animation_config.time_end:
                end_time = np.datetime64(self.animation_config.time_end)
                end_idx = np.searchsorted(times, end_time)

            times = times[start_idx:end_idx]
            indices = np.arange(start_idx, end_idx)
        else:
            indices = np.arange(len(times))

        # Apply time step
        if self.animation_config.time_step:
            indices = indices[::self.animation_config.time_step]

        return indices

    def _get_common_time_indices(self, datasets: Dict[str, xr.Dataset]) -> np.ndarray:
        """Get common time indices across multiple datasets."""
        # Find common time range
        all_times = []
        for ds in datasets.values():
            all_times.append(ds['time'].values)

        # Use intersection of all time ranges
        common_start = max(times[0] for times in all_times)
        common_end = min(times[-1] for times in all_times)

        # Get indices for first dataset (assuming similar time grids)
        first_ds = next(iter(datasets.values()))
        times = first_ds['time'].values

        start_idx = np.searchsorted(times, common_start)
        end_idx = np.searchsorted(times, common_end)

        indices = np.arange(start_idx, end_idx + 1)

        # Apply time step
        if self.animation_config.time_step:
            indices = indices[::self.animation_config.time_step]

        return indices

    def _setup_animation_plot(
        self,
        ax: Axes,
        ds: xr.Dataset,
        variable: str,
        show_grid: bool = True,
        title: Optional[str] = None,
        **kwargs
    ) -> None:
        """Set up initial plot elements for animation."""
        # Set title
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"{variable} Animation")

        # Add geographic features if using cartopy
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature

            if hasattr(ax, 'projection'):
                if self.animation_config.add_coastlines:
                    ax.add_feature(cfeature.COASTLINE)
                    ax.add_feature(cfeature.LAND, alpha=0.3)

                if self.animation_config.add_gridlines:
                    ax.gridlines(draw_labels=True, alpha=0.5)

        except ImportError:
            logger.warning("Cartopy not available, skipping geographic features")

        # Add time label if requested
        if self.animation_config.show_time_label:
            self._time_text = ax.text(
                self.animation_config.time_label_position[0],
                self.animation_config.time_label_position[1],
                "", transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
            )

    def _animate_boundary_frame(
        self,
        ax: Axes,
        ds: xr.Dataset,
        data_var: xr.DataArray,
        variable: str,
        time_idx: int,
        time_val: np.datetime64,
        level_idx: int,
        **kwargs
    ) -> List[Artist]:
        """Animate single frame of boundary data."""
        # Clear previous frame
        for collection in ax.collections:
            collection.remove()

        # Get data for current time
        if 'time' in data_var.dims:
            data = data_var.isel(time=time_idx)
        else:
            data = data_var

        # Handle multi-dimensional boundary data (nOpenBndNodes, nLevels, nComponents)
        # Need to select specific level and component for boundary plotting
        if 'nOpenBndNodes' in data.dims:
            # For boundary data, we need to select both level and component
            if 'nLevels' in data.dims and data.sizes['nLevels'] > level_idx:
                data = data.isel(nLevels=level_idx)
            if 'nComponents' in data.dims:
                data = data.isel(nComponents=0)  # Usually only one component
        elif len(data.shape) > 1 and level_idx < data.shape[-1]:
            # For other multi-dimensional data, use original logic
            data = data.isel({data.dims[-1]: level_idx})

        # Filter out figure-level parameters from plotting kwargs
        plot_kwargs = {k: v for k, v in kwargs.items()
                      if k not in ['figsize', 'use_cartopy', 'projection', 'cmap']}

        # Handle boundary data with proper coordinate mapping
        if 'nOpenBndNodes' in data.dims:
            n_nodes = data.sizes['nOpenBndNodes']
            values = data.values.flatten() if data.values.ndim > 1 else data.values

            if self.grid:
                # Get actual boundary coordinates from grid
                try:
                    hgrid = self.grid.pylibs_hgrid

                    # Compute boundaries to get boundary node indices
                    hgrid.compute_bnd()

                    # Get the first open boundary (assuming single boundary for now)
                    if hasattr(hgrid, 'iobn') and len(hgrid.iobn) > 0:
                        boundary_node_indices = hgrid.iobn[0]  # First open boundary

                        if len(boundary_node_indices) == n_nodes:
                            # Get actual coordinates for boundary nodes
                            boundary_x = hgrid.x[boundary_node_indices]
                            boundary_y = hgrid.y[boundary_node_indices]

                            # Create spatial scatter plot with boundary data
                            im = ax.scatter(boundary_x, boundary_y, c=values,
                                          cmap=self.animation_config.cmap, s=50, **plot_kwargs)

                            # Set extent based on boundary coordinates
                            margin = 0.05  # 5% margin
                            x_range = boundary_x.max() - boundary_x.min()
                            y_range = boundary_y.max() - boundary_y.min()
                            ax.set_xlim(boundary_x.min() - margin * x_range,
                                       boundary_x.max() + margin * x_range)
                            ax.set_ylim(boundary_y.min() - margin * y_range,
                                       boundary_y.max() + margin * y_range)

                            ax.set_xlabel('Longitude')
                            ax.set_ylabel('Latitude')

                        else:
                            logger.warning(f"Boundary node count mismatch: grid={len(boundary_node_indices)}, data={n_nodes}")
                            # Fallback to line plot with node indices
                            boundary_indices = np.arange(n_nodes)
                            ax.plot(boundary_indices, values, 'o-', markersize=4, **plot_kwargs)
                            ax.set_xlabel('Boundary Node Index')
                            ax.set_ylabel(variable)
                            ax.grid(True, alpha=0.3)
                    else:
                        logger.warning("No open boundary information found in grid")
                        # Fallback to line plot with node indices
                        boundary_indices = np.arange(n_nodes)
                        ax.plot(boundary_indices, values, 'o-', markersize=4, **plot_kwargs)
                        ax.set_xlabel('Boundary Node Index')
                        ax.set_ylabel(variable)
                        ax.grid(True, alpha=0.3)

                except Exception as e:
                    logger.warning(f"Could not access grid for boundary plotting: {e}")
                    # Fallback: simple index plot
                    boundary_indices = np.arange(n_nodes)
                    ax.plot(boundary_indices, values, 'o-', markersize=4, **plot_kwargs)
                    ax.set_xlabel('Boundary Node Index')
                    ax.set_ylabel(variable)
                    ax.grid(True, alpha=0.3)
            else:
                # No grid available - simple index plot
                boundary_indices = np.arange(n_nodes)
                ax.plot(boundary_indices, values, 'o-', markersize=4, **plot_kwargs)
                ax.set_xlabel('Boundary Node Index')
                ax.set_ylabel(variable)
                ax.grid(True, alpha=0.3)
        else:
            # Non-boundary data - create scatter plot with grid coordinates if available
            if self.grid and hasattr(self.grid, 'pylibs_hgrid'):
                try:
                    hgrid = self.grid.pylibs_hgrid
                    if hasattr(hgrid, 'x') and hasattr(hgrid, 'y'):
                        # Use sequential node indices for non-boundary data
                        node_indices = np.arange(len(data.values))
                        x_coords = hgrid.x[node_indices]
                        y_coords = hgrid.y[node_indices]

                        im = ax.scatter(
                            x_coords, y_coords, c=data.values,
                            cmap=self.animation_config.cmap, s=50, **plot_kwargs
                        )

                except Exception as e:
                    logger.warning(f"Could not plot with grid coordinates: {e}")
                    # Fallback to simple line plot
                    ax.plot(data.values, **plot_kwargs)
            else:
                # Simple line plot fallback
                ax.plot(data.values, **plot_kwargs)

        # Update time label
        if self._time_text:
            time_str = np.datetime_as_string(time_val,
                                           unit='m')  # minute precision
            formatted_time = np.datetime64(time_str).astype('datetime64[s]').astype(str)
            self._time_text.set_text(f"Time: {formatted_time}")

        return [self._time_text] if self._time_text else []

    def _animate_atmospheric_frame(
        self,
        ax: Axes,
        ds: xr.Dataset,
        data_var: xr.DataArray,
        parameter: str,
        time_idx: int,
        time_val: np.datetime64,
        **kwargs
    ) -> List[Artist]:
        """Animate single frame of atmospheric data."""
        # Clear previous frame
        for collection in ax.collections:
            collection.remove()
        for image in ax.images:
            image.remove()

        # Filter out figure-level parameters from plotting kwargs
        plot_kwargs = {k: v for k, v in kwargs.items()
                      if k not in ['figsize', 'use_cartopy', 'projection', 'cmap']}

        # Get data for current time
        data = data_var.isel(time=time_idx)

        # Create spatial plot
        if 'x' in data.coords and 'y' in data.coords:
            im = ax.pcolormesh(
                data.coords['x'], data.coords['y'], data.values,
                cmap=self.animation_config.cmap, **plot_kwargs
            )
        elif 'lon' in data.coords and 'lat' in data.coords:
            im = ax.pcolormesh(
                data.coords['lon'], data.coords['lat'], data.values,
                cmap=self.animation_config.cmap, **plot_kwargs
            )
        else:
            # Fallback to simple contour plot
            im = ax.contourf(data.values, cmap=self.animation_config.cmap, **plot_kwargs)

        # Update time label
        if self._time_text:
            time_str = np.datetime_as_string(time_val, unit='m')
            formatted_time = np.datetime64(time_str).astype('datetime64[s]').astype(str)
            self._time_text.set_text(f"Time: {formatted_time}")

        return [im, self._time_text] if self._time_text else [im]

    def _animate_grid_frame(
        self,
        ax: Axes,
        ds: xr.Dataset,
        data_var: xr.DataArray,
        variable: str,
        time_idx: int,
        time_val: np.datetime64,
        show_grid: bool,
        **kwargs
    ) -> List[Artist]:
        """Animate single frame of grid-based data."""
        # Clear previous frame
        for collection in ax.collections:
            collection.remove()
        for image in ax.images:
            image.remove()

        # Filter out figure-level parameters from plotting kwargs
        plot_kwargs = {k: v for k, v in kwargs.items()
                      if k not in ['figsize', 'use_cartopy', 'projection', 'cmap']}

        # Get data for current time
        data = data_var.isel(time=time_idx)

        # Create grid-based visualization
        if self.grid and hasattr(self.grid, 'pylibs_hgrid'):
            try:
                hgrid = self.grid.pylibs_hgrid
                # Plot data on grid
                if hasattr(hgrid, 'x') and hasattr(hgrid, 'y'):
                    im = ax.scatter(
                        hgrid.x, hgrid.y, c=data.values,
                        cmap=self.animation_config.cmap, s=20, **plot_kwargs
                    )

                    # Add grid overlay if requested
                    if show_grid and hasattr(hgrid, 'triangles'):
                        ax.triplot(hgrid.x, hgrid.y, hgrid.triangles,
                                 alpha=self.animation_config.grid_alpha,
                                 color=self.animation_config.grid_color)

            except Exception as e:
                logger.warning(f"Could not plot grid data: {e}")
                # Fallback visualization
                im = ax.contourf(data.values, cmap=self.animation_config.cmap, **plot_kwargs)
        else:
            # Fallback to contour plot
            im = ax.contourf(data.values, cmap=self.animation_config.cmap, **plot_kwargs)

        # Update time label
        if self._time_text:
            time_str = np.datetime_as_string(time_val, unit='m')
            formatted_time = np.datetime64(time_str).astype('datetime64[s]').astype(str)
            self._time_text.set_text(f"Time: {formatted_time}")

        return [im, self._time_text] if self._time_text else [im]

    def _get_actual_variable_name(self, ds: xr.Dataset, requested_var: str) -> str:
        """
        Get the actual variable name in the dataset.

        SCHISM boundary files often store data in 'time_series' rather than
        descriptive variable names.

        Parameters
        ----------
        ds : xr.Dataset
            The dataset to search
        requested_var : str
            The requested variable name

        Returns
        -------
        str
            The actual variable name in the dataset

        Raises
        ------
        ValueError
            If the requested variable is not found in the dataset
        """
        if requested_var in ds.data_vars:
            return requested_var
        elif 'time_series' in ds.data_vars:
            return 'time_series'
        else:
            # Look for the first non-coordinate variable with time dimension
            for var_name, var in ds.data_vars.items():
                if 'time' in var.dims and var_name not in ['time_step', 'nOpenBndNodes', 'nLevels', 'nComponents', 'one']:
                    return var_name
            # Raise error if no suitable variable found
            available_vars = list(ds.data_vars.keys())
            raise ValueError(f"Variable {requested_var} not found in dataset. Available variables: {available_vars}")

    def _animate_multi_frame(
        self,
        ax: Axes,
        ds: xr.Dataset,
        data_var: xr.DataArray,
        variable: str,
        time_idx: int,
        time_val: np.datetime64,
        **kwargs
    ) -> List[Artist]:
        """Animate single frame for multi-panel animation."""
        # Delegate to appropriate frame animation method
        if 'boundary' in variable.lower() or 'th.nc' in str(ds.encoding.get('source', '')):
            return self._animate_boundary_frame(
                ax, ds, data_var, variable, time_idx, time_val, 0, **kwargs
            )
        elif 'air' in variable.lower() or 'rad' in variable.lower() or 'prc' in variable.lower():
            return self._animate_atmospheric_frame(
                ax, ds, data_var, variable, time_idx, time_val, **kwargs
            )
        else:
            return self._animate_grid_frame(
                ax, ds, data_var, variable, time_idx, time_val, True, **kwargs
            )

    def _create_multi_panel_figure(
        self,
        num_panels: int,
        layout: str,
        **kwargs
    ) -> Tuple[Figure, Dict[str, Axes]]:
        """Create multi-panel figure for animations."""
        # Extract figsize from kwargs or use defaults
        figsize = kwargs.pop('figsize', None)

        if layout == "vertical":
            default_figsize = (12, 4 * num_panels)
            fig, axes_list = plt.subplots(num_panels, 1,
                                        figsize=figsize or default_figsize,
                                        **kwargs)
        elif layout == "horizontal":
            default_figsize = (6 * num_panels, 8)
            fig, axes_list = plt.subplots(1, num_panels,
                                        figsize=figsize or default_figsize,
                                        **kwargs)
        else:  # grid layout
            rows = int(np.ceil(np.sqrt(num_panels)))
            cols = int(np.ceil(num_panels / rows))
            default_figsize = (6 * cols, 6 * rows)
            fig, axes_list = plt.subplots(rows, cols,
                                        figsize=figsize or default_figsize,
                                        **kwargs)

        # Convert to dictionary
        if num_panels == 1:
            axes_dict = {"panel_0": axes_list}
        else:
            axes_flat = axes_list.flatten() if hasattr(axes_list, 'flatten') else axes_list
            axes_dict = {f"panel_{i}": ax for i, ax in enumerate(axes_flat[:num_panels])}

        return fig, axes_dict

    def _save_animation(
        self,
        anim: animation.FuncAnimation,
        output_file: Union[str, Path]
    ) -> None:
        """Save animation to file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving animation to {output_path}")

        if self.animation_config.show_progress:
            try:
                from tqdm import tqdm
                progress_callback = lambda i, n: tqdm.write(f"Frame {i+1}/{n}")
            except ImportError:
                progress_callback = None
        else:
            progress_callback = None

        if output_path.suffix.lower() == '.gif':
            # Save as GIF
            writer = animation.PillowWriter(fps=self.animation_config.frame_rate)
            anim.save(str(output_path), writer=writer, progress_callback=progress_callback)
        elif output_path.suffix.lower() == '.mp4':
            # Save as MP4
            writer = animation.FFMpegWriter(
                fps=self.animation_config.frame_rate,
                bitrate=self.animation_config.bitrate
            )
            anim.save(str(output_path), writer=writer, progress_callback=progress_callback)
        else:
            # Default to MP4
            output_path = output_path.with_suffix('.mp4')
            writer = animation.FFMpegWriter(
                fps=self.animation_config.frame_rate,
                bitrate=self.animation_config.bitrate
            )
            anim.save(str(output_path), writer=writer, progress_callback=progress_callback)

        logger.info(f"Animation saved successfully to {output_path}")

    def stop_animation(self) -> None:
        """Stop the current animation."""
        if self._current_animation:
            self._current_animation.event_source.stop()
            logger.info("Animation stopped")

    def pause_animation(self) -> None:
        """Pause the current animation."""
        if self._current_animation:
            self._current_animation.pause()
            logger.info("Animation paused")

    def resume_animation(self) -> None:
        """Resume the current animation."""
        if self._current_animation:
            self._current_animation.resume()
            logger.info("Animation resumed")
