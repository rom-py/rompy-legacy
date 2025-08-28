"""
Plotting extensions for SCHISMConfig

This module contains plotting methods to be added to SCHISMConfig class.
"""


def plot_sflux_spatial(
    self,
    variable="air",
    parameter=None,
    time_idx=0,
    level_idx=0,
    ax=None,
    figsize=(10, 8),
    cmap="viridis",
    add_colorbar=True,
    title=None,
    **kwargs,
):
    """Plot spatial distribution of a sflux variable.

    This method leverages the grid and data available in the config object
    to create spatial plots of atmospheric forcing data.

    Parameters
    ----------
    variable : str, optional
        Type of sflux data to plot ('air', 'rad', or 'prc'). Default is 'air'.
    parameter : str, optional
        Specific parameter to plot (e.g., 'prmsl', 'uwind', 'vwind', 'stmp' for air;
        'dlwrf', 'dswrf' for rad; 'prate' for prc). If None, a suitable parameter
        is chosen automatically.
    time_idx : int, optional
        Index of the time step to plot. Default is 0 (first time step).
    level_idx : int, optional
        Index of the vertical level to plot for 3D variables. Default is 0 (surface).
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, a new figure is created.
    figsize : tuple, optional
        Figure size for new figure. Default is (10, 8).
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap to use for the plot. Default is 'viridis'.
    add_colorbar : bool, optional
        Whether to add a colorbar to the plot. Default is True.
    title : str, optional
        Title for the plot. If None, a title is generated based on the variable.
    **kwargs : dict
        Additional keyword arguments to pass to pcolormesh or quiver plotting functions.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.
    """
    import logging

    import matplotlib.pyplot as plt
    import numpy as np
    import xarray as xr

    logger = logging.getLogger(__name__)

    # Check if we have atmospheric data in the config
    if not hasattr(self, "data") or not hasattr(self.data, "atmos"):
        raise ValueError("No atmospheric data available in the config object")

    # Get the dataset from the appropriate source based on variable
    if variable == "air":
        if hasattr(self.data.atmos, "air_1") and self.data.atmos.air_1 is not None:
            ds = self.data.atmos.air_1.source.dataset
        elif hasattr(self.data.atmos, "air") and self.data.atmos.air is not None:
            ds = self.data.atmos.air.source.dataset
        else:
            raise ValueError("No air data available in the config object")
    elif variable == "rad":
        if hasattr(self.data.atmos, "rad_1") and self.data.atmos.rad_1 is not None:
            ds = self.data.atmos.rad_1.source.dataset
        else:
            raise ValueError("No radiation data available in the config object")
    elif variable == "prc":
        if hasattr(self.data.atmos, "prc_1") and self.data.atmos.prc_1 is not None:
            ds = self.data.atmos.prc_1.source.dataset
        else:
            raise ValueError("No precipitation data available in the config object")
    else:
        raise ValueError(f"Unsupported variable type: {variable}")

    # If the dataset is a DataArray, convert to Dataset
    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset()

    # Define common parameter names for each variable type
    parameter_map = {
        "air": ["prmsl", "uwind", "vwind", "stmp", "spfh"],
        "rad": ["dlwrf", "dswrf"],
        "prc": ["prate"],
    }

    # Try to find a suitable parameter if not provided
    if parameter is None:
        if variable in parameter_map:
            # Look for the first parameter that exists in the dataset
            for param in parameter_map[variable]:
                if param in ds:
                    parameter = param
                    break

        # If still not found, just use the first data variable
        if parameter is None and len(ds.data_vars) > 0:
            parameter = list(ds.data_vars)[0]

    # Check if we're plotting wind vectors
    if variable == "air" and parameter == "air":
        # Special case - plot wind vectors
        uwind_name = kwargs.pop("uwind_name", "uwind")
        vwind_name = kwargs.pop("vwind_name", "vwind")
        plot_type = kwargs.pop("plot_type", "quiver")

        if uwind_name not in ds or vwind_name not in ds:
            raise ValueError(
                f"Wind components {uwind_name} and {vwind_name} not found in dataset"
            )

        # Create plot
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        # Get coordinates
        lons = ds["lon"].values if "lon" in ds else ds["longitude"].values
        lats = ds["lat"].values if "lat" in ds else ds["latitude"].values

        # Get the data for the current time step
        u_data = ds[uwind_name][time_idx].values
        v_data = ds[vwind_name][time_idx].values

        # Check if we need to extract a specific level
        if len(u_data.shape) > 2:  # 3D data
            u_data = u_data[level_idx]
            v_data = v_data[level_idx]

        # Create meshgrid if needed
        if len(lons.shape) == 1 and len(lats.shape) == 1:
            lons, lats = np.meshgrid(lons, lats)

        # Subsample for clearer plot if needed
        stride = kwargs.pop("vector_density", 1)
        scale = kwargs.pop("vector_scale", None)

        # Plot wind vectors
        if plot_type == "quiver":
            q = ax.quiver(
                lons[::stride, ::stride],
                lats[::stride, ::stride],
                u_data[::stride, ::stride],
                v_data[::stride, ::stride],
                scale=scale,
                **kwargs,
            )
            # Add a key if scale is provided
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
    else:
        # Make sure the parameter exists
        if parameter not in ds:
            raise ValueError(f"Parameter '{parameter}' not found in dataset")

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        # Get coordinates
        lons = ds["lon"].values if "lon" in ds else ds["longitude"].values
        lats = ds["lat"].values if "lat" in ds else ds["latitude"].values

        # Get the data for the current time step
        data = ds[parameter][time_idx].values

        # Check if we need to extract a specific level
        if len(data.shape) > 2:  # 3D data
            data = data[level_idx]

        # Create meshgrid if needed
        if len(lons.shape) == 1 and len(lats.shape) == 1:
            lons, lats = np.meshgrid(lons, lats)

        # Plot the data
        im = ax.pcolormesh(lons, lats, data, cmap=cmap, **kwargs)

        # Add colorbar if requested
        if add_colorbar:
            cbar = fig.colorbar(im, ax=ax)
            # Add units if available
            if "units" in ds[parameter].attrs:
                cbar.set_label(ds[parameter].attrs["units"])

    # Add grid boundaries if available
    if hasattr(self, "grid") and self.grid is not None:
        try:
            # Try to plot the grid boundaries
            self.grid.plot_boundary(ax=ax, color="k", linewidth=1.5)
        except Exception as e:
            logger.warning(f"Could not plot grid boundaries: {e}")

    # Add coastlines if available and requested
    coastlines = kwargs.pop("coastlines", False)
    if coastlines:
        try:
            import cartopy.feature as cfeature

            ax.add_feature(cfeature.COASTLINE, linewidth=1.0)
        except ImportError:
            logger.warning("Cartopy not available, coastlines not added")

    # Set title
    if title is None:
        # Get the time as a string if possible
        try:
            time_str = str(ds["time"].values[time_idx])
            title = f"{parameter} - {time_str}"
        except:
            title = parameter
    ax.set_title(title)

    # Set labels
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Set equal aspect ratio
    ax.set_aspect("equal")

    return fig, ax


def plot_sflux_timeseries(
    self,
    variable="air",
    parameter=None,
    lat=None,
    lon=None,
    location_idx=None,
    level_idx=0,
    ax=None,
    figsize=(12, 6),
    **kwargs,
):
    """Plot time series of a sflux variable at a specific location.

    This method leverages the atmospheric data available in the config object
    to create time series plots at specified locations.

    Parameters
    ----------
    variable : str, optional
        Type of sflux data to plot ('air', 'rad', or 'prc'). Default is 'air'.
    parameter : str, optional
        Specific parameter to plot (e.g., 'prmsl', 'uwind', 'vwind', 'stmp' for air;
        'dlwrf', 'dswrf' for rad; 'prate' for prc). If None, a suitable parameter
        is chosen automatically.
    lat : float, optional
        Latitude of the point to extract. Required if location_idx is None.
    lon : float, optional
        Longitude of the point to extract. Required if location_idx is None.
    location_idx : int or tuple, optional
        Index of the location to plot. Can be a single index or (i,j) tuple for 2D grids.
        If None, the nearest point to (lat, lon) is used.
    level_idx : int, optional
        Index of the vertical level to plot for 3D variables. Default is 0 (surface).
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, a new figure is created.
    figsize : tuple, optional
        Figure size for new figure. Default is (12, 6).
    **kwargs : dict
        Additional keyword arguments to pass to plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    """
    import logging

    import matplotlib.pyplot as plt
    import numpy as np
    import xarray as xr
    from scipy.spatial import cKDTree

    logging.getLogger(__name__)

    # Check if we have atmospheric data in the config
    if not hasattr(self, "data") or not hasattr(self.data, "sflux"):
        raise ValueError("No sflux data available in the config object")

    # Get the dataset from the appropriate source based on variable
    if variable == "air":
        if hasattr(self.data.atmos, "air_1") and self.data.atmos.air_1 is not None:
            ds = self.data.atmos.air_1.source.dataset
        elif hasattr(self.data.atmos, "air") and self.data.atmos.air is not None:
            ds = self.data.atmos.air.source.dataset
        else:
            raise ValueError("No air data available in the config object")
    elif variable == "rad":
        if hasattr(self.data.atmos, "rad_1") and self.data.atmos.rad_1 is not None:
            ds = self.data.atmos.rad_1.source.dataset
        else:
            raise ValueError("No radiation data available in the config object")
    elif variable == "prc":
        if hasattr(self.data.atmos, "prc_1") and self.data.atmos.prc_1 is not None:
            ds = self.data.atmos.prc_1.source.dataset
        else:
            raise ValueError("No precipitation data available in the config object")
    else:
        raise ValueError(f"Unsupported variable type: {variable}")

    # If the dataset is a DataArray, convert to Dataset
    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset()

    # Define common parameter names for each variable type
    parameter_map = {
        "air": ["prmsl", "uwind", "vwind", "stmp", "spfh"],
        "rad": ["dlwrf", "dswrf"],
        "prc": ["prate"],
    }

    # Try to find a suitable parameter if not provided
    if parameter is None:
        if variable in parameter_map:
            # Look for the first parameter that exists in the dataset
            for param in parameter_map[variable]:
                if param in ds:
                    parameter = param
                    break

        # If still not found, just use the first data variable
        if parameter is None and len(ds.data_vars) > 0:
            parameter = list(ds.data_vars)[0]

    # Check if we're plotting wind speed time series
    if variable == "air" and parameter == "air":
        # Special case - compute wind speed
        uwind_name = kwargs.pop("uwind_name", "uwind")
        vwind_name = kwargs.pop("vwind_name", "vwind")

        if uwind_name not in ds or vwind_name not in ds:
            raise ValueError(
                f"Wind components {uwind_name} and {vwind_name} not found in dataset"
            )

        # Will compute wind speed later after extracting time series
    else:
        # Make sure the parameter exists
        if parameter not in ds:
            raise ValueError(f"Parameter '{parameter}' not found in dataset")

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Get coordinates
    lons = ds["lon"].values if "lon" in ds else ds["longitude"].values
    lats = ds["lat"].values if "lat" in ds else ds["latitude"].values

    # Find the location index if not provided
    if location_idx is None:
        if lat is None or lon is None:
            raise ValueError("Either location_idx or both lat and lon must be provided")

        # Find nearest grid point
        if len(lons.shape) == 1 and len(lats.shape) == 1:
            # Regular grid
            lon_idx = np.abs(lons - lon).argmin()
            lat_idx = np.abs(lats - lat).argmin()
            location_idx = (lat_idx, lon_idx)
        else:
            # Unstructured grid
            points = np.vstack((lons.flatten(), lats.flatten())).T
            tree = cKDTree(points)
            _, location_idx = tree.query([lon, lat])

    # Extract time series
    times = ds["time"].values

    if variable == "air" and parameter == "air":
        # For wind, compute speed from u and v components
        if isinstance(location_idx, tuple) and len(location_idx) == 2:
            # 2D location index
            u_values = (
                ds[uwind_name]
                .isel(
                    {
                        ds[uwind_name].dims[1]: location_idx[0],
                        ds[uwind_name].dims[2]: location_idx[1],
                    }
                )
                .values
            )
            v_values = (
                ds[vwind_name]
                .isel(
                    {
                        ds[vwind_name].dims[1]: location_idx[0],
                        ds[vwind_name].dims[2]: location_idx[1],
                    }
                )
                .values
            )
        else:
            # 1D location index
            u_values = (
                ds[uwind_name]
                .isel(
                    {
                        ds[uwind_name].dims[1]: location_idx // lons.shape[1],
                        ds[uwind_name].dims[2]: location_idx % lons.shape[1],
                    }
                )
                .values
            )
            v_values = (
                ds[vwind_name]
                .isel(
                    {
                        ds[vwind_name].dims[1]: location_idx // lons.shape[1],
                        ds[vwind_name].dims[2]: location_idx % lons.shape[1],
                    }
                )
                .values
            )

        # Check if we need to extract a specific level
        if len(u_values.shape) > 1:  # 3D data
            u_values = u_values[:, level_idx]
            v_values = v_values[:, level_idx]

        # Compute wind speed
        values = np.sqrt(u_values**2 + v_values**2)

        # Set default parameter name for title and labels
        if parameter == "air":
            parameter = "wind_speed"
    else:
        # Extract the time series for a single parameter
        if isinstance(location_idx, tuple) and len(location_idx) == 2:
            # 2D location index
            values = (
                ds[parameter]
                .isel(
                    {
                        ds[parameter].dims[1]: location_idx[0],
                        ds[parameter].dims[2]: location_idx[1],
                    }
                )
                .values
            )
        else:
            # 1D location index
            values = (
                ds[parameter]
                .isel(
                    {
                        ds[parameter].dims[1]: location_idx // lons.shape[1],
                        ds[parameter].dims[2]: location_idx % lons.shape[1],
                    }
                )
                .values
            )

        # Check if we need to extract a specific level
        if len(values.shape) > 1:  # 3D data
            values = values[:, level_idx]

    # Plot the time series
    ax.plot(times, values, **kwargs)

    # Add labels and title
    ax.set_xlabel("Time")

    # Set y-label with units if available
    if variable == "air" and parameter == "wind_speed":
        units = ds[uwind_name].attrs.get("units", "m/s")
        ylabel = kwargs.pop(
            "ylabel", f"Wind Speed ({units})" if units else "Wind Speed"
        )
    else:
        units = ds[parameter].attrs.get("units", "")
        ylabel = kwargs.pop("ylabel", f"{parameter} ({units})" if units else parameter)

    ax.set_ylabel(ylabel)

    # Format location for title
    if isinstance(location_idx, tuple):
        if len(lons.shape) == 1 and len(lats.shape) == 1:
            loc_lat = lats[location_idx[0]]
            loc_lon = lons[location_idx[1]]
        else:
            loc_lat = lats[location_idx[0], location_idx[1]]
            loc_lon = lons[location_idx[0], location_idx[1]]
    else:
        if len(lons.shape) == 1 and len(lats.shape) == 1:
            loc_lat = lats[location_idx // lons.shape[0]]
            loc_lon = lons[location_idx % lons.shape[0]]
        else:
            loc_lat = lats.flatten()[location_idx]
            loc_lon = lons.flatten()[location_idx]

    title = kwargs.pop("title", f"{parameter} at ({loc_lat:.2f}°, {loc_lon:.2f}°)")
    ax.set_title(title)

    # Format the x-axis to show dates nicely
    fig.autofmt_xdate()

    return fig, ax
