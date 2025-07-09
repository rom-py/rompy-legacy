"""
Boundary plotting extensions for SCHISMConfig.

This module contains methods for visualizing boundary data in SCHISM models.
"""


def plot_boundary_points(self, variable=None, ax=None, figsize=(10, 8), **kwargs):
    """Plot boundary node points from a boundary dataset.

    Parameters
    ----------
    variable : str, optional
        Variable to color the points by. If None, points are not colored by variable.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, a new figure is created.
    figsize : tuple, optional
        Figure size for new figure. Default is (10, 8).
    **kwargs : dict
        Additional keyword arguments to pass to scatter plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import logging

    logger = logging.getLogger(__name__)

    # Check if we have the necessary components
    if not hasattr(self, "grid") or self.grid is None:
        raise ValueError("No grid available in the config object")
    if not hasattr(self, "data") or self.data is None:
        raise ValueError("No data available in the config object")

    # Check for boundary data in the correct location (under ocean)
    boundary = None
    if (
        hasattr(self.data, "ocean")
        and self.data.ocean is not None
        and hasattr(self.data.ocean, "boundary")
        and self.data.ocean.boundary is not None
    ):
        boundary = self.data.ocean.boundary
    # Fall back to legacy location for backward compatibility
    elif hasattr(self.data, "boundary") and self.data.boundary is not None:
        boundary = self.data.boundary
    else:
        raise ValueError(
            "No boundary data available in the config object (checked both data.ocean.boundary and data.boundary)"
        )

    # Get the grid and boundary dataset
    grid = self.grid
    dataset = boundary.source.dataset

    # Create plot if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Plot the grid boundaries
    try:
        grid.plot_boundary(ax=ax, color="k", linewidth=1.0)
    except Exception as e:
        logger.warning(f"Could not plot grid boundaries: {e}")

    # Get boundary node indices
    boundary_nodes = []
    for i in range(grid.pylibs_hgrid.nob):
        boundary_nodes.extend(grid.pylibs_hgrid.iobn[i])

    # Get boundary node coordinates
    x = grid.pylibs_hgrid.x[boundary_nodes]
    y = grid.pylibs_hgrid.y[boundary_nodes]

    # Color by variable if specified
    if variable is not None and variable in dataset:
        # Get variable values for coloring
        if "node" in dataset.dims and len(dataset[variable].shape) >= 1:
            # Extract values for the first time step if available
            if "time" in dataset.dims and len(dataset[variable].shape) >= 2:
                values = dataset[variable].isel(time=0).values
            else:
                values = dataset[variable].values

            # Plot colored points
            scatter = ax.scatter(x, y, c=values, **kwargs)
            plt.colorbar(scatter, ax=ax, label=variable)
        else:
            logger.warning(
                f"Variable {variable} not suitable for coloring boundary points"
            )
            ax.scatter(x, y, **kwargs)
    else:
        # Plot without variable coloring
        ax.scatter(x, y, **kwargs)

    # Set labels and title
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Boundary Points")
    ax.set_aspect("equal")

    return fig, ax


def plot_boundary_timeseries(
    self, variable=None, boundary_idx=0, node_idx=0, ax=None, figsize=(12, 6), **kwargs
):
    """Plot time series of a boundary variable at a specific boundary node.

    Parameters
    ----------
    variable : str, optional
        Variable to plot. If None, the first variable in the dataset is used.
    boundary_idx : int, optional
        Index of the boundary to plot. Default is 0 (first boundary).
    node_idx : int, optional
        Index of the node within the boundary to plot. Default is 0 (first node).
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
    import matplotlib.pyplot as plt
    import numpy as np
    import logging

    logger = logging.getLogger(__name__)

    # Check if we have the necessary components
    if not hasattr(self, "grid") or self.grid is None:
        raise ValueError("No grid available in the config object")
    if not hasattr(self, "data") or self.data is None:
        raise ValueError("No data available in the config object")

    # Check for boundary data in the correct location (under ocean)
    boundary = None
    if (
        hasattr(self.data, "ocean")
        and self.data.ocean is not None
        and hasattr(self.data.ocean, "boundary")
        and self.data.ocean.boundary is not None
    ):
        boundary = self.data.ocean.boundary
    # Fall back to legacy location for backward compatibility
    elif hasattr(self.data, "boundary") and self.data.boundary is not None:
        boundary = self.data.boundary
    else:
        raise ValueError(
            "No boundary data available in the config object (checked both data.ocean.boundary and data.boundary)"
        )

    # Get the grid and boundary dataset
    grid = self.grid
    dataset = boundary.source.dataset

    # Check if dataset has time dimension
    if "time" not in dataset.dims:
        raise ValueError("Dataset does not have a time dimension")

    # Get first variable if not specified
    if variable is None and len(dataset.data_vars) > 0:
        variable = list(dataset.data_vars)[0]

    if variable not in dataset:
        raise ValueError(f"Variable '{variable}' not found in dataset")

    # Create plot if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Get boundary node indices
    if boundary_idx >= grid.pylibs_hgrid.nob:
        raise ValueError(
            f"Boundary index {boundary_idx} out of range (0-{grid.pylibs_hgrid.nob-1})"
        )

    boundary_nodes = grid.pylibs_hgrid.iobn[boundary_idx]
    if node_idx >= len(boundary_nodes):
        raise ValueError(
            f"Node index {node_idx} out of range (0-{len(boundary_nodes)-1})"
        )

    # Get global node index
    global_node_idx = boundary_nodes[node_idx]

    # Get coordinates of the node
    x = grid.pylibs_hgrid.x[global_node_idx]
    y = grid.pylibs_hgrid.y[global_node_idx]

    # Extract time series
    times = dataset["time"].values

    # Find the appropriate dimension for nodes
    node_dim = None
    for dim in dataset[variable].dims:
        if dim in ["node", "nOpenBndNodes", "nBnd"]:
            node_dim = dim
            break

    if node_dim is None:
        raise ValueError(f"Could not find node dimension in variable {variable}")

    # Map global node index to dataset index if needed
    if "node" in dataset.coords:
        # Check if dataset nodes match grid nodes
        dataset_nodes = dataset["node"].values
        if global_node_idx in dataset_nodes:
            dataset_node_idx = np.where(dataset_nodes == global_node_idx)[0][0]
        else:
            # If not, just use the local node index within boundary
            logger.warning(
                "Node indices in dataset do not match grid node indices. Using local index."
            )
            dataset_node_idx = node_idx
    else:
        # If no node coordinate, assume nodes are in the same order as in the grid
        dataset_node_idx = node_idx

    # Extract values
    if len(dataset[variable].dims) == 2 and node_dim in dataset[variable].dims:
        # 2D variable (time, node)
        values = dataset[variable].isel({node_dim: dataset_node_idx}).values
    elif len(dataset[variable].dims) == 3 and node_dim in dataset[variable].dims:
        # 3D variable (time, node, level) - extract surface level
        values = dataset[variable].isel({node_dim: dataset_node_idx, "level": 0}).values
    else:
        raise ValueError(f"Variable {variable} does not have expected dimensions")

    # Plot the time series
    ax.plot(times, values, **kwargs)

    # Add labels and title
    ax.set_xlabel("Time")
    units = dataset[variable].attrs.get("units", "")
    ax.set_ylabel(f"{variable} ({units})" if units else variable)
    ax.set_title(
        f"{variable} at Boundary {boundary_idx}, Node {node_idx} ({x:.2f}, {y:.2f})"
    )

    # Format the x-axis to show dates nicely
    fig.autofmt_xdate()

    return fig


def plot_boundary_profile(
    self,
    variable=None,
    boundary_idx=0,
    node_idx=0,
    time_idx=0,
    ax=None,
    figsize=(8, 10),
    **kwargs,
):
    """Plot depth profile of a boundary variable at a specific boundary node and time.

    Parameters
    ----------
    variable : str, optional
        Variable to plot. If None, the first 3D variable in the dataset is used.
    boundary_idx : int, optional
        Index of the boundary to plot. Default is 0 (first boundary).
    node_idx : int, optional
        Index of the node within the boundary to plot. Default is 0 (first node).
    time_idx : int, optional
        Index of the time step to plot. Default is 0 (first time step).
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, a new figure is created.
    figsize : tuple, optional
        Figure size for new figure. Default is (8, 10).
    **kwargs : dict
        Additional keyword arguments to pass to plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import logging

    logger = logging.getLogger(__name__)

    # Check if we have the necessary components
    if not hasattr(self, "grid") or self.grid is None:
        raise ValueError("No grid available in the config object")
    if not hasattr(self, "data") or self.data is None:
        raise ValueError("No data available in the config object")

    # Check for boundary data in the correct location (under ocean)
    boundary = None
    if (
        hasattr(self.data, "ocean")
        and self.data.ocean is not None
        and hasattr(self.data.ocean, "boundary")
        and self.data.ocean.boundary is not None
    ):
        boundary = self.data.ocean.boundary
    # Fall back to legacy location for backward compatibility
    elif hasattr(self.data, "boundary") and self.data.boundary is not None:
        boundary = self.data.boundary
    else:
        raise ValueError(
            "No boundary data available in the config object (checked both data.ocean.boundary and data.boundary)"
        )

    # Get the grid and boundary dataset
    grid = self.grid
    dataset = boundary.source.dataset

    # Check if dataset has depth/level dimension
    depth_dim = None
    for dim in dataset.dims:
        if dim in ["depth", "level", "nLevels", "sigma"]:
            depth_dim = dim
            break

    if depth_dim is None:
        raise ValueError("Dataset does not have a depth/level dimension")

    # Find a 3D variable if not specified
    if variable is None:
        for var in dataset.data_vars:
            if len(dataset[var].dims) >= 3 and depth_dim in dataset[var].dims:
                variable = var
                break

    if variable not in dataset or depth_dim not in dataset[variable].dims:
        raise ValueError(
            f"Variable '{variable}' not found in dataset or does not have a depth dimension"
        )

    # Create plot if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Get boundary node indices
    if boundary_idx >= grid.pylibs_hgrid.nob:
        raise ValueError(
            f"Boundary index {boundary_idx} out of range (0-{grid.pylibs_hgrid.nob-1})"
        )

    boundary_nodes = grid.pylibs_hgrid.iobn[boundary_idx]
    if node_idx >= len(boundary_nodes):
        raise ValueError(
            f"Node index {node_idx} out of range (0-{len(boundary_nodes)-1})"
        )

    # Get global node index
    global_node_idx = boundary_nodes[node_idx]

    # Get coordinates of the node
    x = grid.pylibs_hgrid.x[global_node_idx]
    y = grid.pylibs_hgrid.y[global_node_idx]

    # Find the appropriate dimension for nodes
    node_dim = None
    for dim in dataset[variable].dims:
        if dim in ["node", "nOpenBndNodes", "nBnd"]:
            node_dim = dim
            break

    if node_dim is None:
        raise ValueError(f"Could not find node dimension in variable {variable}")

    # Map global node index to dataset index if needed
    if "node" in dataset.coords:
        # Check if dataset nodes match grid nodes
        dataset_nodes = dataset["node"].values
        if global_node_idx in dataset_nodes:
            dataset_node_idx = np.where(dataset_nodes == global_node_idx)[0][0]
        else:
            # If not, just use the local node index within boundary
            logger.warning(
                "Node indices in dataset do not match grid node indices. Using local index."
            )
            dataset_node_idx = node_idx
    else:
        # If no node coordinate, assume nodes are in the same order as in the grid
        dataset_node_idx = node_idx

    # Get depth values
    depths = dataset[depth_dim].values

    # Extract profile values
    if "time" in dataset[variable].dims:
        values = (
            dataset[variable].isel(time=time_idx, **{node_dim: dataset_node_idx}).values
        )
    else:
        values = dataset[variable].isel({node_dim: dataset_node_idx}).values

    # Plot the depth profile
    ax.plot(values, depths, **kwargs)

    # Reverse y-axis to have surface at top
    ax.invert_yaxis()

    # Add labels and title
    units = dataset[variable].attrs.get("units", "")
    ax.set_xlabel(f"{variable} ({units})" if units else variable)
    ax.set_ylabel(f'Depth ({dataset[depth_dim].attrs.get("units", "m")})')

    # Add time to title if available
    title = f"{variable} Profile at Boundary {boundary_idx}, Node {node_idx} ({x:.2f}, {y:.2f})"
    if "time" in dataset:
        try:
            time_str = str(dataset["time"].values[time_idx])
            title += f"\nTime: {time_str}"
        except:
            pass

    ax.set_title(title)

    return fig
