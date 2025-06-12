"""
Data plotting functionality for SCHISM input files.

This module provides comprehensive visualization capabilities for SCHISM input data
including boundary conditions, atmospheric forcing, and tidal data.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.spatial import cKDTree

from .core import BasePlotter, PlotConfig, PlotValidator
from .utils import (
    detect_file_type,
    get_variable_info,
    load_schism_data,
    create_time_subset,
    setup_colormap,
    add_boundary_overlay,
    get_geographic_extent,
    create_diverging_colormap_levels
)

logger = logging.getLogger(__name__)


class DataPlotter(BasePlotter):
    """
    SCHISM data plotting functionality.
    
    This class provides methods for visualizing SCHISM input data files
    including 3D boundary conditions, atmospheric forcing, and tidal data.
    
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
        """Initialize DataPlotter."""
        super().__init__(config, grid_file, plot_config)

    def plot(self, file_path: Union[str, Path], **kwargs) -> Tuple[Figure, Axes]:
        """
        Main plotting method that auto-detects file type.
        
        Parameters
        ----------
        file_path : Union[str, Path]
            Path to SCHISM data file
        **kwargs : dict
            Additional plotting parameters
            
        Returns
        -------
        fig : Figure
            Figure object
        ax : Axes
            Axes object
        """
        file_type = detect_file_type(file_path)
        
        if file_type == 'salinity_3d':
            return self.plot_salinity_3d(file_path, **kwargs)
        elif file_type == 'temperature_3d':
            return self.plot_temperature_3d(file_path, **kwargs)
        elif file_type == 'velocity_3d':
            return self.plot_velocity_3d(file_path, **kwargs)
        elif file_type == 'elevation_2d':
            return self.plot_elevation_2d(file_path, **kwargs)
        elif file_type == 'atmospheric':
            return self.plot_atmospheric_spatial(file_path=file_path, **kwargs)
        elif file_type == 'gr3':
            return self.plot_gr3_file(file_path, **kwargs)
        elif file_type == 'bctides':
            return self.plot_bctides_file(file_path, **kwargs)
        else:
            return self.plot_boundary_data(file_path, **kwargs)

    def plot_boundary_data(
        self,
        file_path: Union[str, Path],
        variable: Optional[str] = None,
        time_idx: int = 0,
        level_idx: int = 0,
        ax: Optional[Axes] = None,
        plot_type: str = "spatial",
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """
        Plot boundary condition data from SCHISM input files.
        
        Parameters
        ----------
        file_path : Union[str, Path]
            Path to boundary data file
        variable : Optional[str]
            Variable to plot. If None, first available variable is used.
        time_idx : int, optional
            Time index to plot. Default is 0.
        level_idx : int, optional
            Vertical level index for 3D data. Default is 0 (surface).
        ax : Optional[Axes]
            Existing axes to plot on
        plot_type : str, optional
            Type of plot: 'spatial', 'timeseries', 'profile'. Default is 'spatial'.
        **kwargs : dict
            Additional plotting parameters
            
        Returns
        -------
        fig : Figure
            Figure object
        ax : Axes
            Axes object
        """
        ds = load_schism_data(file_path)
        
        # Get variable name
        if variable is None:
            variable = list(ds.data_vars)[0]
        
        PlotValidator.validate_dataset(ds, required_vars=[variable])
        
        if plot_type == "spatial":
            return self._plot_boundary_spatial(ds, variable, time_idx, level_idx, ax, **kwargs)
        elif plot_type == "timeseries":
            return self._plot_boundary_timeseries(ds, variable, level_idx, ax, **kwargs)
        elif plot_type == "profile":
            return self._plot_boundary_profile(ds, variable, time_idx, ax, **kwargs)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")

    def plot_salinity_3d(
        self,
        file_path: Union[str, Path],
        time_idx: int = 0,
        level_idx: int = 0,
        ax: Optional[Axes] = None,
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """
        Plot 3D salinity boundary data.
        
        Parameters
        ----------
        file_path : Union[str, Path]
            Path to SAL_3D.th.nc file
        time_idx : int, optional
            Time index to plot. Default is 0.
        level_idx : int, optional
            Vertical level index. Default is 0 (surface).
        ax : Optional[Axes]
            Existing axes to plot on
        **kwargs : dict
            Additional plotting parameters
            
        Returns
        -------
        fig : Figure
            Figure object
        ax : Axes
            Axes object
        """
        return self.plot_boundary_data(
            file_path, variable="salinity", time_idx=time_idx, 
            level_idx=level_idx, ax=ax, **kwargs
        )

    def plot_temperature_3d(
        self,
        file_path: Union[str, Path],
        time_idx: int = 0,
        level_idx: int = 0,
        ax: Optional[Axes] = None,
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """
        Plot 3D temperature boundary data.
        
        Parameters
        ----------
        file_path : Union[str, Path]
            Path to TEM_3D.th.nc file
        time_idx : int, optional
            Time index to plot. Default is 0.
        level_idx : int, optional
            Vertical level index. Default is 0 (surface).
        ax : Optional[Axes]
            Existing axes to plot on
        **kwargs : dict
            Additional plotting parameters
            
        Returns
        -------
        fig : Figure
            Figure object
        ax : Axes
            Axes object
        """
        return self.plot_boundary_data(
            file_path, variable="temperature", time_idx=time_idx,
            level_idx=level_idx, ax=ax, **kwargs
        )

    def plot_velocity_3d(
        self,
        file_path: Union[str, Path],
        time_idx: int = 0,
        level_idx: int = 0,
        ax: Optional[Axes] = None,
        plot_component: str = "magnitude",
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """
        Plot 3D velocity boundary data.
        
        Parameters
        ----------
        file_path : Union[str, Path]
            Path to uv3D.th.nc file
        time_idx : int, optional
            Time index to plot. Default is 0.
        level_idx : int, optional
            Vertical level index. Default is 0 (surface).
        ax : Optional[Axes]
            Existing axes to plot on
        plot_component : str, optional
            Component to plot: 'magnitude', 'u', 'v', 'vectors'. Default is 'magnitude'.
        **kwargs : dict
            Additional plotting parameters
            
        Returns
        -------
        fig : Figure
            Figure object
        ax : Axes
            Axes object
        """
        ds = load_schism_data(file_path)
        
        # Find velocity components
        u_var = None
        v_var = None
        for var in ds.data_vars:
            if 'u' in var.lower() and 'velocity' in var.lower():
                u_var = var
            elif 'v' in var.lower() and 'velocity' in var.lower():
                v_var = var
        
        if u_var is None or v_var is None:
            # Try common naming conventions
            possible_u = ['u', 'uvel', 'u_velocity', 'eastward_velocity']
            possible_v = ['v', 'vvel', 'v_velocity', 'northward_velocity']
            
            for u_name in possible_u:
                if u_name in ds.data_vars:
                    u_var = u_name
                    break
            
            for v_name in possible_v:
                if v_name in ds.data_vars:
                    v_var = v_name
                    break
        
        if u_var is None or v_var is None:
            raise ValueError("Could not find velocity components in dataset")
        
        if plot_component == "u":
            return self.plot_boundary_data(
                file_path, variable=u_var, time_idx=time_idx,
                level_idx=level_idx, ax=ax, **kwargs
            )
        elif plot_component == "v":
            return self.plot_boundary_data(
                file_path, variable=v_var, time_idx=time_idx,
                level_idx=level_idx, ax=ax, **kwargs
            )
        elif plot_component == "magnitude":
            return self._plot_velocity_magnitude(
                ds, u_var, v_var, time_idx, level_idx, ax, **kwargs
            )
        elif plot_component == "vectors":
            return self._plot_velocity_vectors(
                ds, u_var, v_var, time_idx, level_idx, ax, **kwargs
            )
        else:
            raise ValueError(f"Unknown velocity component: {plot_component}")

    def plot_elevation_2d(
        self,
        file_path: Union[str, Path],
        time_idx: int = 0,
        ax: Optional[Axes] = None,
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """
        Plot 2D elevation boundary data.
        
        Parameters
        ----------
        file_path : Union[str, Path]
            Path to elev2D.th.nc file
        time_idx : int, optional
            Time index to plot. Default is 0.
        ax : Optional[Axes]
            Existing axes to plot on
        **kwargs : dict
            Additional plotting parameters
            
        Returns
        -------
        fig : Figure
            Figure object
        ax : Axes
            Axes object
        """
        return self.plot_boundary_data(
            file_path, variable="elevation", time_idx=time_idx, ax=ax, **kwargs
        )

    def plot_atmospheric_spatial(
        self,
        variable: str = "air",
        parameter: Optional[str] = None,
        time_idx: int = 0,
        level_idx: int = 0,
        ax: Optional[Axes] = None,
        file_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """
        Plot spatial distribution of atmospheric forcing data.
        
        Parameters
        ----------
        variable : str, optional
            Type of atmospheric data ('air', 'rad', 'prc'). Default is 'air'.
        parameter : Optional[str]
            Specific parameter to plot (e.g., 'prmsl', 'uwind', 'vwind').
            Use 'wind_vectors' for wind vector plots.
        time_idx : int, optional
            Time index to plot. Default is 0.
        level_idx : int, optional
            Vertical level index for 3D variables. Default is 0.
        ax : Optional[Axes]
            Existing axes to plot on
        file_path : Optional[Union[str, Path]]
            Path to atmospheric data file. If None, uses config data.
        **kwargs : dict
            Additional plotting parameters. For wind vectors:
            - plot_type: 'quiver' or 'barbs' (default: 'quiver')
            - uwind_name: name of u-wind variable (default: 'uwind')
            - vwind_name: name of v-wind variable (default: 'vwind')
            - vector_density: stride for vector subsampling (default: 1)
            - vector_scale: scale for vectors (default: None)
            - barb_length: length for barbs (default: 5)
            
        Returns
        -------
        fig : Figure
            Figure object
        ax : Axes
            Axes object
        """
        fig, ax = self.create_figure(ax=ax, **kwargs)
        
        # Get dataset
        if file_path:
            ds = load_schism_data(file_path)
        else:
            ds = self._get_atmospheric_dataset(variable)
        
        # Parameter mapping
        parameter_map = {
            "air": ["prmsl", "uwind", "vwind", "stmp", "spfh"],
            "rad": ["dlwrf", "dswrf"],
            "prc": ["prate"],
        }
        
        # Find parameter if not specified
        if parameter is None:
            if variable in parameter_map:
                for param in parameter_map[variable]:
                    if param in ds.data_vars:
                        parameter = param
                        break
            
            if parameter is None and len(ds.data_vars) > 0:
                parameter = list(ds.data_vars)[0]
        
        # Check if we're plotting wind vectors
        if parameter == "wind_vectors" or (variable == "air" and parameter == "air"):
            return self._plot_atmospheric_wind_vectors(
                ds, time_idx, level_idx, ax, fig, **kwargs
            )
        
        if parameter not in ds.data_vars:
            raise ValueError(f"Parameter '{parameter}' not found in dataset")
        
        # Get coordinates
        lons = ds["lon"].values if "lon" in ds else ds["longitude"].values
        lats = ds["lat"].values if "lat" in ds else ds["latitude"].values
        
        # Get data for specified time and level
        data = ds[parameter][time_idx].values
        if len(data.shape) > 2:  # 3D data
            data = data[level_idx]
        
        # Create meshgrid if needed
        if len(lons.shape) == 1 and len(lats.shape) == 1:
            lons, lats = np.meshgrid(lons, lats)
        
        # Filter out figure-related kwargs that shouldn't go to pcolormesh
        plot_kwargs = {k: v for k, v in kwargs.items() 
                      if k not in ['figsize', 'dpi', 'save_path', 'use_cartopy']}
        
        # Plot data
        im = ax.pcolormesh(lons, lats, data, cmap=self.plot_config.cmap, **plot_kwargs)
        
        # Add colorbar
        units = ds[parameter].attrs.get("units", "")
        label = f"{parameter} ({units})" if units else parameter
        cbar = self.add_colorbar(fig, ax, im, label=label)
        
        # Add grid boundaries if available
        if hasattr(self, 'grid') and self.grid is not None:
            try:
                add_boundary_overlay(ax, self.grid)
            except Exception as e:
                logger.warning(f"Could not add grid boundaries: {e}")
        
        # Set extent
        extent = get_geographic_extent(lons, lats)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        
        # Set title
        try:
            time_str = str(ds["time"].values[time_idx])
            title = f"{parameter} - {time_str}"
        except:
            title = parameter
        
        return self.finalize_plot(fig, ax, title=title)

    def plot_atmospheric_timeseries(
        self,
        variable: str = "air",
        parameter: Optional[str] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        location_idx: Optional[int] = None,
        level_idx: int = 0,
        ax: Optional[Axes] = None,
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """
        Plot time series of atmospheric data at a specific location.
        
        Parameters
        ----------
        variable : str, optional
            Type of atmospheric data ('air', 'rad', 'prc'). Default is 'air'.
        parameter : Optional[str]
            Specific parameter to plot. Use 'wind_speed' for wind speed calculation.
        lat : Optional[float]
            Latitude of point to extract.
        lon : Optional[float]
            Longitude of point to extract.
        location_idx : Optional[int]
            Index of location to plot. If None, nearest to (lat, lon) is used.
        level_idx : int, optional
            Vertical level index for 3D variables. Default is 0.
        ax : Optional[Axes]
            Existing axes to plot on
        **kwargs : dict
            Additional plotting parameters. For wind speed:
            - uwind_name: name of u-wind variable (default: 'uwind')
            - vwind_name: name of v-wind variable (default: 'vwind')
            
        Returns
        -------
        fig : Figure
            Figure object
        ax : Axes
            Axes object
        """
        fig, ax = self.create_figure(ax=ax, use_cartopy=False, **kwargs)
        
        # Get dataset
        ds = self._get_atmospheric_dataset(variable)
        
        # Find parameter
        if parameter is None:
            parameter = list(ds.data_vars)[0]
        
        # Check if we're plotting wind speed
        if parameter == "wind_speed" or (variable == "air" and parameter == "air"):
            return self._plot_atmospheric_wind_speed_timeseries(
                ds, lat, lon, location_idx, level_idx, ax, fig, **kwargs
            )
        
        # Get coordinates
        lons = ds["lon"].values if "lon" in ds else ds["longitude"].values
        lats = ds["lat"].values if "lat" in ds else ds["latitude"].values
        
        # Find location index
        if location_idx is None:
            if lat is None or lon is None:
                raise ValueError("Either location_idx or both lat and lon must be provided")
            
            if len(lons.shape) == 1 and len(lats.shape) == 1:
                lon_idx = np.abs(lons - lon).argmin()
                lat_idx = np.abs(lats - lat).argmin()
                location_idx = (lat_idx, lon_idx)
            else:
                points = np.vstack((lons.flatten(), lats.flatten())).T
                tree = cKDTree(points)
                _, location_idx = tree.query([lon, lat])
        
        # Extract time series
        times = ds["time"].values
        
        if isinstance(location_idx, tuple):
            values = ds[parameter].isel(
                {ds[parameter].dims[1]: location_idx[0],
                 ds[parameter].dims[2]: location_idx[1]}
            ).values
        else:
            values = ds[parameter].flatten()[location_idx::len(lons.flatten())]
        
        # Handle 3D data
        if len(values.shape) > 1:
            values = values[:, level_idx]
        
        # Plot time series
        ax.plot(times, values, **kwargs)
        
        # Set labels
        units = ds[parameter].attrs.get("units", "")
        ylabel = f"{parameter} ({units})" if units else parameter
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Time")
        
        # Format dates
        fig.autofmt_xdate()
        
        # Set title
        if isinstance(location_idx, tuple):
            loc_lat = lats[location_idx[0]]
            loc_lon = lons[location_idx[1]]
        else:
            loc_lat = lats.flatten()[location_idx]
            loc_lon = lons.flatten()[location_idx]
        
        title = f"{parameter} at ({loc_lat:.2f}°, {loc_lon:.2f}°)"
        
        return self.finalize_plot(fig, ax, title=title)

    def plot_tidal_boundaries(
        self,
        ax: Optional[Axes] = None,
        colors: Dict[str, str] = None,
        linewidth: float = 2.0,
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """
        Plot tidal boundaries.
        
        Parameters
        ----------
        ax : Optional[Axes]
            Existing axes to plot on
        colors : Dict[str, str], optional
            Colors for different boundary types
        linewidth : float, optional
            Line width for boundaries. Default is 2.0.
        **kwargs : dict
            Additional plotting parameters
            
        Returns
        -------
        fig : Figure
            Figure object
        ax : Axes
            Axes object
        """
        fig, ax = self.create_figure(ax=ax, **kwargs)
        
        if colors is None:
            colors = {"tidal": "blue", "ocean": "red", "land": "green"}
        
        try:
            grid = self.grid
            
            # Plot regular boundaries first
            add_boundary_overlay(ax, grid, boundary_colors=colors, linewidth=linewidth)
            
            # Add tidal-specific boundaries if available in config
            if self.config and hasattr(self.config, 'data') and hasattr(self.config.data, 'tides'):
                tides = self.config.data.tides
                # Implementation would depend on tidal data structure
                logger.info("Tidal boundary plotting from config not yet fully implemented")
            
        except Exception as e:
            logger.error(f"Error plotting tidal boundaries: {e}")
            raise
        
        return self.finalize_plot(fig, ax, title="Tidal Boundaries")

    def _plot_boundary_spatial(
        self,
        ds: xr.Dataset,
        variable: str,
        time_idx: int,
        level_idx: int,
        ax: Optional[Axes],
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """Plot spatial distribution of boundary data."""
        fig, ax = self.create_figure(ax=ax, **kwargs)
        
        # Filter out figure-related kwargs that shouldn't go to plotting functions
        plot_kwargs = {k: v for k, v in kwargs.items() 
                      if k not in ['figsize', 'dpi', 'save_path', 'use_cartopy']}
        
        try:
            # Handle SCHISM boundary files (*.th.nc) which use time_series variable
            if 'time_series' in ds.data_vars:
                data_var = ds['time_series']
                
                # Extract data for specified time and level 
                if 'time' in data_var.dims:
                    data = data_var.isel(time=time_idx)
                else:
                    data = data_var
                
                # Handle multi-dimensional data (levels, components)
                if len(data.shape) > 1:
                    if 'nLevels' in data.dims and data.sizes['nLevels'] > level_idx:
                        data = data.isel(nLevels=level_idx)
                    if 'nComponents' in data.dims:
                        data = data.isel(nComponents=0)  # Take first component
                
                # For boundary data, we need actual coordinates
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
                                                  cmap=self.plot_config.cmap, s=50, **plot_kwargs)
                                    
                                    # Add colorbar
                                    units = ds[variable].attrs.get("units", "") if variable in ds.attrs else ""
                                    if not units and 'time_series' in ds.data_vars:
                                        units = ds['time_series'].attrs.get("units", "")
                                    label = f"{variable} ({units})" if units else variable
                                    cbar = self.add_colorbar(fig, ax, im, label=label)
                                    
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
                                    # Fallback to line plot
                                    boundary_indices = np.arange(n_nodes)
                                    ax.plot(boundary_indices, values, 'o-', markersize=4, **plot_kwargs)
                                    ax.set_xlabel('Boundary Node Index')
                                    ax.set_ylabel(variable)
                                    ax.grid(True, alpha=0.3)
                            else:
                                logger.warning("No open boundary information found in grid")
                                # Fallback to line plot
                                boundary_indices = np.arange(n_nodes)
                                ax.plot(boundary_indices, values, 'o-', markersize=4, **plot_kwargs)
                                ax.set_xlabel('Boundary Node Index')
                                ax.set_ylabel(variable)
                                ax.grid(True, alpha=0.3)
                            
                        except Exception as e:
                            logger.warning(f"Could not access grid for boundary plotting: {e}")
                            # Fallback: simple index plot
                            boundary_indices = np.arange(n_nodes)
                            ax.plot(boundary_indices, values, 'o-', markersize=4)
                            ax.set_xlabel('Boundary Node Index')
                            ax.set_ylabel(variable)
                            ax.grid(True, alpha=0.3)
                    else:
                        # No grid available - simple index plot
                        boundary_indices = np.arange(n_nodes)
                        ax.plot(boundary_indices, values, 'o-', markersize=4)
                        ax.set_xlabel('Boundary Node Index')
                        ax.set_ylabel(variable)
                        ax.grid(True, alpha=0.3)
                        
            else:
                # Handle other variable types with spatial coordinates
                var_data = ds[variable]
                
                # Extract data for specified time and level
                if "time" in var_data.dims:
                    data = var_data.isel(time=time_idx)
                else:
                    data = var_data
                
                if len(data.shape) > 1 and level_idx < data.shape[-1]:
                    data = data.isel({data.dims[-1]: level_idx})
                
                # Try to find spatial coordinates
                if hasattr(data, 'lon') and hasattr(data, 'lat'):
                    x, y = data.lon.values, data.lat.values
                elif 'lon' in ds.coords and 'lat' in ds.coords:
                    x, y = ds.lon.values, ds.lat.values
                else:
                    # Fallback to simple plot
                    ax.plot(data.values, 'o-')
                    ax.set_ylabel(variable)
                    ax.grid(True, alpha=0.3)
                    return self.finalize_plot(fig, ax, title=f"{variable} (Boundary Data)")
                
                # Create spatial plot
                im = ax.scatter(x, y, c=data.values, cmap=self.plot_config.cmap, **plot_kwargs)
                
                # Add colorbar
                units = ds[variable].attrs.get("units", "")
                label = f"{variable} ({units})" if units else variable
                cbar = self.add_colorbar(fig, ax, im, label=label)
                
                # Set extent
                extent = get_geographic_extent(x, y)
                ax.set_xlim(extent[0], extent[1])
                ax.set_ylim(extent[2], extent[3])
                
        except Exception as e:
            logger.error(f"Error plotting boundary spatial data: {e}")
            # Fallback: show error message
            ax.text(0.5, 0.5, f"Error plotting {variable}\n{str(e)}", 
                   ha='center', va='center', transform=ax.transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
            ax.set_title(f"Error: {Path(file_path).name}" if 'file_path' in locals() else f"Error plotting {variable}")
        
        return self.finalize_plot(fig, ax, title=f"{variable} (Boundary Data)")

    def _plot_boundary_timeseries(
        self,
        ds: xr.Dataset,
        variable: str,
        level_idx: int,
        ax: Optional[Axes],
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """Plot time series of boundary data."""
        fig, ax = self.create_figure(ax=ax, use_cartopy=False, **kwargs)
        
        # Get variable data
        var_data = ds[variable]
        
        if "time" not in var_data.dims:
            raise ValueError(f"Variable {variable} does not have time dimension")
        
        # Average over nodes or select first node
        if "node" in var_data.dims:
            data = var_data.mean(dim="node")
        else:
            data = var_data
        
        # Handle vertical levels
        if len(data.shape) > 1 and level_idx < data.shape[-1]:
            data = data.isel({data.dims[-1]: level_idx})
        
        # Plot time series
        times = ds["time"].values
        ax.plot(times, data.values, **kwargs)
        
        # Set labels
        units = ds[variable].attrs.get("units", "")
        ylabel = f"{variable} ({units})" if units else variable
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Time")
        
        # Format dates
        fig.autofmt_xdate()
        
        return self.finalize_plot(fig, ax, title=f"{variable} (Time Series)")

    def _plot_boundary_profile(
        self,
        ds: xr.Dataset,
        variable: str,
        time_idx: int,
        ax: Optional[Axes],
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """Plot vertical profile of boundary data."""
        fig, ax = self.create_figure(ax=ax, use_cartopy=False, **kwargs)
        
        var_data = ds[variable]
        
        # Find depth/level dimension
        depth_dim = None
        for dim in var_data.dims:
            if dim in ["depth", "level", "sigma", "z"]:
                depth_dim = dim
                break
        
        if depth_dim is None:
            raise ValueError(f"Variable {variable} does not have depth dimension")
        
        # Extract data for specified time
        if "time" in var_data.dims:
            data = var_data.isel(time=time_idx)
        else:
            data = var_data
        
        # Average over nodes
        if "node" in data.dims:
            data = data.mean(dim="node")
        
        # Get depth values
        depths = ds[depth_dim].values
        
        # Plot profile
        ax.plot(data.values, depths, **kwargs)
        
        # Invert y-axis for depth
        ax.invert_yaxis()
        
        # Set labels
        units = ds[variable].attrs.get("units", "")
        xlabel = f"{variable} ({units})" if units else variable
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"Depth ({ds[depth_dim].attrs.get('units', 'm')})")
        
        return self.finalize_plot(fig, ax, title=f"{variable} (Profile)")

    def _plot_velocity_magnitude(
        self,
        ds: xr.Dataset,
        u_var: str,
        v_var: str,
        time_idx: int,
        level_idx: int,
        ax: Optional[Axes],
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """Plot velocity magnitude."""
        fig, ax = self.create_figure(ax=ax, **kwargs)
        
        # Get velocity components
        u_data = ds[u_var][time_idx].values
        v_data = ds[v_var][time_idx].values
        
        # Handle 3D data
        if len(u_data.shape) > 1:
            u_data = u_data[level_idx]
            v_data = v_data[level_idx]
        
        # Calculate magnitude
        magnitude = np.sqrt(u_data**2 + v_data**2)
        
        # Plot using grid coordinates if available
        if self.grid:
            try:
                hgrid = self.grid.pylibs_hgrid
                x, y = hgrid.x, hgrid.y
                
                im = ax.scatter(x, y, c=magnitude, cmap=self.plot_config.cmap, **kwargs)
                
                # Add colorbar
                cbar = self.add_colorbar(fig, ax, im, label="Velocity Magnitude (m/s)")
                
                # Set extent
                extent = get_geographic_extent(x, y)
                ax.set_xlim(extent[0], extent[1])
                ax.set_ylim(extent[2], extent[3])
                
            except Exception as e:
                logger.error(f"Error plotting velocity magnitude: {e}")
                raise
        
        return self.finalize_plot(fig, ax, title="Velocity Magnitude")

    def _plot_velocity_vectors(
        self,
        ds: xr.Dataset,
        u_var: str,
        v_var: str,
        time_idx: int,
        level_idx: int,
        ax: Optional[Axes],
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """Plot velocity vectors."""
        fig, ax = self.create_figure(ax=ax, **kwargs)
        
        # Get velocity components
        u_data = ds[u_var][time_idx].values
        v_data = ds[v_var][time_idx].values
        
        # Handle 3D data
        if len(u_data.shape) > 1:
            u_data = u_data[level_idx]
            v_data = v_data[level_idx]
        
        # Plot using grid coordinates if available
        if self.grid:
            try:
                hgrid = self.grid.pylibs_hgrid
                x, y = hgrid.x, hgrid.y
                
                # Subsample for clearer visualization
                stride = kwargs.pop("stride", 5)
                scale = kwargs.pop("scale", None)
                
                q = ax.quiver(
                    x[::stride], y[::stride],
                    u_data[::stride], v_data[::stride],
                    scale=scale, **kwargs
                )
                
                # Add quiver key
                if scale:
                    ax.quiverkey(q, 0.9, 0.9, 1.0, "1 m/s", labelpos="E")
                
                # Set extent
                extent = get_geographic_extent(x, y)
                ax.set_xlim(extent[0], extent[1])
                ax.set_ylim(extent[2], extent[3])
                
            except Exception as e:
                logger.error(f"Error plotting velocity vectors: {e}")
                raise
        
        return self.finalize_plot(fig, ax, title="Velocity Vectors")

    def _plot_atmospheric_wind_vectors(
        self,
        ds: xr.Dataset,
        time_idx: int,
        level_idx: int,
        ax: Axes,
        fig: Figure,
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """
        Plot atmospheric wind vectors.
        
        Parameters
        ----------
        ds : xr.Dataset
            Dataset containing wind data
        time_idx : int
            Time index
        level_idx : int
            Level index
        ax : Axes
            Axes object
        fig : Figure
            Figure object
        **kwargs : dict
            Additional plotting parameters
            
        Returns
        -------
        fig : Figure
            Figure object
        ax : Axes
            Axes object
        """
        # Get wind component names
        uwind_name = kwargs.pop("uwind_name", "uwind")
        vwind_name = kwargs.pop("vwind_name", "vwind")
        plot_type = kwargs.pop("plot_type", "quiver")
        
        if uwind_name not in ds or vwind_name not in ds:
            raise ValueError(
                f"Wind components {uwind_name} and {vwind_name} not found in dataset"
            )
        
        # Get coordinates
        lons = ds["lon"].values if "lon" in ds else ds["longitude"].values
        lats = ds["lat"].values if "lat" in ds else ds["latitude"].values
        
        # Get wind data for specified time and level
        u_data = ds[uwind_name][time_idx].values
        v_data = ds[vwind_name][time_idx].values
        
        # Handle 3D data
        if len(u_data.shape) > 2:
            u_data = u_data[level_idx]
            v_data = v_data[level_idx]
        
        # Create meshgrid if needed
        if len(lons.shape) == 1 and len(lats.shape) == 1:
            lons, lats = np.meshgrid(lons, lats)
        
        # Subsample for clearer visualization
        density = kwargs.pop("vector_density", 1)
        scale = kwargs.pop("vector_scale", None)
        
        # Plot wind vectors
        if plot_type == "quiver":
            q = ax.quiver(
                lons[::density, ::density],
                lats[::density, ::density],
                u_data[::density, ::density],
                v_data[::density, ::density],
                scale=scale,
                **kwargs,
            )
            # Add quiver key if scale is provided
            if scale:
                ax.quiverkey(q, 0.9, 0.9, 10, "10 m/s", labelpos="E")
                
        elif plot_type == "barbs":
            barb_length = kwargs.pop("barb_length", 5)
            barb_density = kwargs.pop("barb_density", 1)
            ax.barbs(
                lons[::barb_density, ::barb_density],
                lats[::barb_density, ::barb_density],
                u_data[::barb_density, ::barb_density],
                v_data[::barb_density, ::barb_density],
                length=barb_length,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported plot_type: {plot_type}")
        
        # Add grid boundaries if available
        if hasattr(self, 'grid') and self.grid is not None:
            try:
                add_boundary_overlay(ax, self.grid)
            except Exception as e:
                logger.warning(f"Could not add grid boundaries: {e}")
        
        # Set extent
        extent = get_geographic_extent(lons, lats)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        
        # Set title
        try:
            time_str = str(ds["time"].values[time_idx])
            title = f"Wind Vectors - {time_str}"
        except:
            title = "Wind Vectors"
        
        return self.finalize_plot(fig, ax, title=title)

    def _plot_atmospheric_wind_speed_timeseries(
        self,
        ds: xr.Dataset,
        lat: Optional[float],
        lon: Optional[float],
        location_idx: Optional[int],
        level_idx: int,
        ax: Optional[Axes],
        fig: Figure,
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """
        Plot wind speed time series at a specific location.
        
        Parameters
        ----------
        ds : xr.Dataset
            Dataset containing wind data
        lat : Optional[float]
            Latitude of point to extract
        lon : Optional[float]
            Longitude of point to extract
        location_idx : Optional[int]
            Index of location to plot
        level_idx : int
            Vertical level index
        ax : Optional[Axes]
            Existing axes to plot on
        fig : Figure
            Figure object
        **kwargs : dict
            Additional plotting parameters
            
        Returns
        -------
        fig : Figure
            Figure object
        ax : Axes
            Axes object
        """
        # Get wind component names
        uwind_name = kwargs.pop("uwind_name", "uwind")
        vwind_name = kwargs.pop("vwind_name", "vwind")
        
        if uwind_name not in ds or vwind_name not in ds:
            raise ValueError(
                f"Wind components {uwind_name} and {vwind_name} not found in dataset"
            )
        
        # Get coordinates
        lons = ds["lon"].values if "lon" in ds else ds["longitude"].values
        lats = ds["lat"].values if "lat" in ds else ds["latitude"].values
        
        # Find location index
        if location_idx is None:
            if lat is None or lon is None:
                raise ValueError("Either location_idx or both lat and lon must be provided")
            
            if len(lons.shape) == 1 and len(lats.shape) == 1:
                lon_idx = np.abs(lons - lon).argmin()
                lat_idx = np.abs(lats - lat).argmin()
                location_idx = (lat_idx, lon_idx)
            else:
                points = np.vstack((lons.flatten(), lats.flatten())).T
                tree = cKDTree(points)
                _, location_idx = tree.query([lon, lat])
        
        # Extract time series for wind components
        times = ds["time"].values
        
        if isinstance(location_idx, tuple):
            u_values = ds[uwind_name].isel(
                {ds[uwind_name].dims[1]: location_idx[0],
                 ds[uwind_name].dims[2]: location_idx[1]}
            ).values
            v_values = ds[vwind_name].isel(
                {ds[vwind_name].dims[1]: location_idx[0],
                 ds[vwind_name].dims[2]: location_idx[1]}
            ).values
        else:
            # Handle flattened index
            u_values = ds[uwind_name].values.reshape(len(times), -1)[:, location_idx]
            v_values = ds[vwind_name].values.reshape(len(times), -1)[:, location_idx]
        
        # Handle 3D data
        if len(u_values.shape) > 1:
            u_values = u_values[:, level_idx]
            v_values = v_values[:, level_idx]
        
        # Compute wind speed
        wind_speed = np.sqrt(u_values**2 + v_values**2)
        
        # Plot time series
        ax.plot(times, wind_speed, **kwargs)
        
        # Set labels
        units = ds[uwind_name].attrs.get("units", "m/s")
        ylabel = f"Wind Speed ({units})" if units else "Wind Speed"
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Time")
        
        # Format dates
        fig.autofmt_xdate()
        
        # Set title
        if isinstance(location_idx, tuple):
            if len(lons.shape) == 1 and len(lats.shape) == 1:
                loc_lat = lats[location_idx[0]]
                loc_lon = lons[location_idx[1]]
            else:
                loc_lat = lats[location_idx[0], location_idx[1]]
                loc_lon = lons[location_idx[0], location_idx[1]]
        else:
            loc_lat = lats.flatten()[location_idx]
            loc_lon = lons.flatten()[location_idx]
        
        title = f"Wind Speed at ({loc_lat:.2f}°, {loc_lon:.2f}°)"
        
        return self.finalize_plot(fig, ax, title=title)

    def plot_gr3_file(
        self,
        file_path: Union[str, Path],
        ax: Optional[Axes] = None,
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """
        Plot .gr3 property files with appropriate colormaps.
        
        Parameters
        ----------
        file_path : Union[str, Path]
            Path to .gr3 file
        ax : Optional[Axes]
            Existing axes to plot on
        **kwargs : dict
            Additional plotting parameters
            
        Returns
        -------
        fig : Figure
            Figure object
        ax : Axes
            Axes object
        """
        from .utils import load_schism_data
        
        fig, ax = self.create_figure(ax=ax, **kwargs)
        
        # Load .gr3 file (assuming it can be loaded as grid with values)
        try:
            # Try to load using pyschism if available
            if hasattr(self, 'grid') and self.grid is not None:
                grid = self.grid
                if hasattr(grid, 'pylibs_hgrid'):
                    hgrid = grid.pylibs_hgrid
                    x, y = hgrid.x, hgrid.y
                    
                    # Read .gr3 file values
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                    
                    # Skip header and node/element counts
                    data_lines = lines[2:]  # Skip first 2 lines (header + counts)
                    
                    # Extract node values
                    values = []
                    for line in data_lines:
                        if line.strip():
                            parts = line.strip().split()
                            if len(parts) >= 4:  # node_id, x, y, value
                                values.append(float(parts[3]))
                    
                    values = np.array(values)
                    
                    # Determine colormap based on filename
                    file_name = Path(file_path).name.lower()
                    if 'bathy' in file_name or 'depth' in file_name:
                        cmap = 'viridis_r'  # Deeper = darker
                        label = 'Depth (m)'
                    elif 'rough' in file_name:
                        cmap = 'plasma'
                        label = 'Roughness'
                    elif 'temp' in file_name:
                        cmap = 'coolwarm'
                        label = 'Temperature (°C)'
                    elif 'sal' in file_name:
                        cmap = 'viridis'
                        label = 'Salinity'
                    else:
                        cmap = 'viridis'
                        label = 'Value'
                    
                    # Create scatter plot
                    im = ax.scatter(x, y, c=values, cmap=cmap, s=1, **kwargs)
                    
                    # Add colorbar
                    cbar = self.add_colorbar(fig, ax, im, label=label)
                    
                    # Set title
                    title = f"{Path(file_path).name} Properties"
                    
                else:
                    raise ValueError("Grid does not have pylibs_hgrid attribute")
            else:
                raise ValueError("No grid available for .gr3 plotting")
                
        except Exception as e:
            logger.error(f"Error plotting .gr3 file: {e}")
            # Fallback to simple text display
            ax.text(0.5, 0.5, f"Error loading .gr3 file:\n{e}", 
                   ha='center', va='center', transform=ax.transAxes)
            title = f"Error: {Path(file_path).name}"
        
        return self.finalize_plot(fig, ax, title=title)

    def plot_bctides_file(
        self,
        file_path: Union[str, Path],
        ax: Optional[Axes] = None,
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """
        Plot bctides.in configuration file.
        
        Parameters
        ----------
        file_path : Union[str, Path]
            Path to bctides.in file
        ax : Optional[Axes]
            Existing axes to plot on
        **kwargs : dict
            Additional plotting parameters
            
        Returns
        -------
        fig : Figure
            Figure object
        ax : Axes
            Axes object
        """
        fig, ax = self.create_figure(ax=ax, use_cartopy=False, **kwargs)
        
        try:
            # Read bctides.in file
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Parse basic information
            info_text = []
            constituent_count = 0
            boundary_count = 0
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line or line.startswith('!'):
                    continue
                    
                if i < 10:  # First few lines contain configuration
                    parts = line.split()
                    if len(parts) >= 1:
                        try:
                            value = float(parts[0])
                            if i == 0:
                                info_text.append(f"Start time: {value}")
                            elif i == 1:
                                info_text.append(f"Time step: {value}")
                            elif i == 2:
                                constituent_count = int(value)
                                info_text.append(f"Constituents: {constituent_count}")
                        except ValueError:
                            if 'M2' in line or 'S2' in line or 'K1' in line or 'O1' in line:
                                info_text.append(f"Tidal constituent: {line}")
            
            # Display information as text
            y_pos = 0.9
            for text in info_text:
                ax.text(0.1, y_pos, text, transform=ax.transAxes, fontsize=12)
                y_pos -= 0.1
            
            # Add summary
            if boundary_count > 0:
                ax.text(0.1, y_pos-0.1, f"Boundary segments: {boundary_count}", 
                       transform=ax.transAxes, fontsize=12, weight='bold')
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            
            title = f"{Path(file_path).name} Configuration"
            
        except Exception as e:
            logger.error(f"Error parsing bctides.in file: {e}")
            ax.text(0.5, 0.5, f"Error reading bctides.in:\n{e}", 
                   ha='center', va='center', transform=ax.transAxes)
            title = f"Error: {Path(file_path).name}"
        
        return self.finalize_plot(fig, ax, title=title)

    def _get_atmospheric_dataset(self, variable: str) -> xr.Dataset:
        """Get atmospheric dataset from configuration."""
        if not self.config:
            raise ValueError("No configuration available for atmospheric data")
        
        if not hasattr(self.config, 'data') or not hasattr(self.config.data, 'atmos'):
            raise ValueError("No atmospheric data available in configuration")
        
        # Get dataset based on variable type
        if variable == "air":
            if hasattr(self.config.data.atmos, "air_1") and self.config.data.atmos.air_1:
                source = self.config.data.atmos.air_1.source
                if hasattr(source, 'dataset'):
                    return source.dataset
                elif hasattr(source, 'get_dataset'):
                    return source.get_dataset()
                elif hasattr(source, 'data'):
                    return source.data
                else:
                    # Load from file if it's a SourceFile
                    import xarray as xr
                    return xr.open_dataset(source.uri)
            elif hasattr(self.config.data.atmos, "air") and self.config.data.atmos.air:
                source = self.config.data.atmos.air.source
                if hasattr(source, 'dataset'):
                    return source.dataset
                elif hasattr(source, 'get_dataset'):
                    return source.get_dataset()
                elif hasattr(source, 'data'):
                    return source.data
                else:
                    import xarray as xr
                    return xr.open_dataset(source.uri)
        elif variable == "rad":
            if hasattr(self.config.data.atmos, "rad_1") and self.config.data.atmos.rad_1:
                source = self.config.data.atmos.rad_1.source
                if hasattr(source, 'dataset'):
                    return source.dataset
                elif hasattr(source, 'get_dataset'):
                    return source.get_dataset()
                elif hasattr(source, 'data'):
                    return source.data
                else:
                    import xarray as xr
                    return xr.open_dataset(source.uri)
        elif variable == "prc":
            if hasattr(self.config.data.atmos, "prc_1") and self.config.data.atmos.prc_1:
                source = self.config.data.atmos.prc_1.source
                if hasattr(source, 'dataset'):
                    return source.dataset
                elif hasattr(source, 'get_dataset'):
                    return source.get_dataset()
                elif hasattr(source, 'data'):
                    return source.data
                else:
                    import xarray as xr
                    return xr.open_dataset(source.uri)
        
        raise ValueError(f"No {variable} data available in configuration")