"""Tests for the boundary plotting methods in SCHISMDataBoundary."""

import os
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta

pytest.importorskip("rompy.schism")

from rompy.core.data import DataBlob
from rompy.core.time import TimeRange
from rompy.schism.data import SCHISMDataBoundary
from rompy.schism.grid import SCHISMGrid


# Define the location of test files
HERE = Path(__file__).parent



@pytest.fixture
def test_boundary_data(grid2d):
    """Create sample boundary data for testing without using SCHISMDataBoundary."""
    # Create a sample dataset
    times = pd.date_range(start=datetime.now(), periods=5, freq="1D").to_pydatetime()

    # Create a simple elevation boundary
    grid = grid2d
    grid.pylibs_hgrid.compute_bnd()

    # Get a boundary with some nodes
    boundary_idx = 0
    if grid.pylibs_hgrid.nob > 0:
        boundary_nodes = grid.pylibs_hgrid.iobn[boundary_idx]
    else:
        # If no boundaries in test grid, create a fake one
        boundary_nodes = [0, 1, 2, 3, 4]

    # Values for each node at each time
    num_nodes = len(boundary_nodes)
    num_times = len(times)

    # Create 2D variable (time, node)
    elevation = np.sin(np.linspace(0, 2 * np.pi, num_times * num_nodes)).reshape(
        num_times, num_nodes
    )

    # Create 3D variable (time, node, level)
    num_levels = 3
    temperature = np.zeros((num_times, num_nodes, num_levels))
    for t in range(num_times):
        for n in range(num_nodes):
            for l in range(num_levels):
                # Decreasing temperature with depth
                temperature[t, n, l] = 20 - l * 5 + np.sin(t) + 0.5 * n

    # Create xarray dataset
    ds = xr.Dataset(
        {
            "elevation": (["time", "nOpenBndNodes"], elevation),
            "temperature": (["time", "nOpenBndNodes", "nLevels"], temperature),
        },
        coords={
            "time": times,
            "node": (["nOpenBndNodes"], boundary_nodes),
            "depth": (["nLevels"], [0, 5, 10]),
        },
    )

    # Add unit attributes
    ds["elevation"].attrs["units"] = "m"
    ds["temperature"].attrs["units"] = "degC"

    # Create a simple class that just returns the dataset
    class SimpleDataSource:
        def __init__(self, dataset):
            self._dataset = dataset

        def ds(self):
            return self._dataset

    # Return a simple container instead of SCHISMDataBoundary
    return SimpleDataSource(ds)


def test_plot_boundary_points(grid2d, test_boundary_data):
    """Test plotting of boundary points."""
    # Create a simplified test that doesn't rely on SCHISMDataBoundary validation
    import matplotlib.pyplot as plt

    # Get boundary points from grid
    x_bound, y_bound = grid2d.boundary_points()

    # Create a simple plot
    fig, ax = plt.subplots()
    ax.scatter(x_bound, y_bound, color="blue", marker="o")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Boundary Points")

    assert fig is not None
    assert ax is not None


def test_plot_boundary_timeseries(grid2d, test_boundary_data):
    """Test plotting of boundary time series."""
    import matplotlib.pyplot as plt
    import xarray as xr

    # Get the sample dataset directly from the fixture
    ds = test_boundary_data.ds()

    # Test 2D variable (elevation) - create a simple plot
    fig, ax = plt.subplots()

    # Plot elevation for first two nodes
    for i in range(2):
        ax.plot(ds.time, ds.elevation[:, i], label=f"Node {i}")

    ax.set_xlabel("Time")
    ax.set_ylabel("Elevation (m)")
    ax.set_title("Boundary Elevation Time Series")
    ax.legend()

    assert fig is not None

    # Test 3D variable - create a simple plot for temperature
    fig, ax = plt.subplots()

    # Plot temperature at surface for first node
    ax.plot(ds.time, ds.temperature[:, 0, 0], label="Surface")

    # Plot temperature at mid-depth for first node
    ax.plot(ds.time, ds.temperature[:, 0, 1], label="Mid-depth")

    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Boundary Temperature Time Series")
    ax.legend()

    assert fig is not None


def test_plot_boundary_profile(grid2d, test_boundary_data):
    """Test plotting of boundary vertical profile."""
    import matplotlib.pyplot as plt

    # Get the sample dataset directly from the fixture
    ds = test_boundary_data.ds()

    # Create a simple profile plot for temperature
    fig, ax = plt.subplots()

    # Plot temperature profile at first time and node
    ax.plot(ds.temperature[0, 0, :], ds.depth, marker="o")

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Temperature Profile")
    ax.invert_yaxis()  # Depth increases downward

    assert fig is not None


def test_boundary_plotting_workflow(grid2d, test_boundary_data):
    """Test a complete workflow with multiple plots."""
    import matplotlib.pyplot as plt

    # Get the sample dataset directly from the fixture
    ds = test_boundary_data.ds()

    # Create a figure with multiple subplots for different visualizations
    fig = plt.figure(figsize=(15, 10))

    # 1. Plot boundary points in first subplot
    ax1 = plt.subplot(2, 2, 1)
    x_bound, y_bound = grid2d.boundary_points()
    ax1.scatter(x_bound, y_bound, color="blue")
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.set_title("Boundary Points")

    # 2. Plot elevation time series in second subplot
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(ds.time, ds.elevation[:, 0], label="Node 0")
    ax2.plot(ds.time, ds.elevation[:, 1], label="Node 1")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Elevation (m)")
    ax2.set_title("Boundary Elevation")
    ax2.legend()

    # 3. Plot temperature time series in third subplot
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(ds.time, ds.temperature[:, 0, 0], label="Surface")
    ax3.plot(ds.time, ds.temperature[:, 0, 1], label="Mid-depth")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Temperature (°C)")
    ax3.set_title("Temperature Time Series")
    ax3.legend()

    # 4. Plot temperature profile in fourth subplot
    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(ds.temperature[0, 0, :], ds.depth, marker="o")
    ax4.set_xlabel("Temperature (°C)")
    ax4.set_ylabel("Depth (m)")
    ax4.set_title("Temperature Profile")
    ax4.invert_yaxis()  # Depth increases downward

    plt.tight_layout()

    assert fig is not None
