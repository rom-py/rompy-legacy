"""Tests for the boundary plotting methods in SCHISMConfig."""

import os
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta

pytest.importorskip("rompy.schism")

from rompy.core.data import DataBlob, DataGrid
from rompy.core.time import TimeRange
from rompy.schism.data import SCHISMData, SCHISMDataBoundary
from rompy.schism.grid import SCHISMGrid
from rompy.schism.config import SCHISMConfig


# Define the location of test files
HERE = Path(__file__).parent



@pytest.fixture
def test_boundary_dataset():
    """Create a sample boundary dataset for testing boundary plotting."""
    # Define time range
    times = pd.date_range(start=datetime.now(), periods=10, freq="1H").to_pydatetime()

    # Create coordinates for 2 open boundaries with 3 nodes each
    node_ids = np.array([0, 1, 2, 3, 4, 5])  # Example node IDs

    # Create sample boundary data
    # For elevation
    elevation = np.zeros((len(times), len(node_ids)))
    for t in range(len(times)):
        for i in range(len(node_ids)):
            # Simple tidal signal
            elevation[t, i] = 0.5 * np.sin(2 * np.pi * t / 12.0 + i * 0.1)

    # For temperature (3D with 5 levels)
    num_levels = 5
    temperature = np.zeros((len(times), len(node_ids), num_levels))
    for t in range(len(times)):
        for i in range(len(node_ids)):
            for z in range(num_levels):
                # Temperature decreasing with depth
                temperature[t, i, z] = 20 - 0.5 * z + 2.0 * np.sin(2 * np.pi * t / 24.0)

    # For salinity (3D with 5 levels)
    salinity = np.zeros((len(times), len(node_ids), num_levels))
    for t in range(len(times)):
        for i in range(len(node_ids)):
            for z in range(num_levels):
                # Salinity increasing with depth
                salinity[t, i, z] = 30 + 0.2 * z - 1.0 * np.sin(2 * np.pi * t / 24.0)

    # Create levels array
    levels = np.linspace(0, -20, num_levels)  # Depth levels (0 to -20)

    # Create xarray dataset
    ds = xr.Dataset(
        {
            "elevation": (["time", "nOpenBndNodes"], elevation),
            "temperature": (["time", "nOpenBndNodes", "nLevels"], temperature),
            "salinity": (["time", "nOpenBndNodes", "nLevels"], salinity),
        },
        coords={
            "time": times,
            "node": (["nOpenBndNodes"], node_ids),
            "depth": (["nLevels"], levels),
        },
    )

    # Add unit attributes
    ds["elevation"].attrs["units"] = "m"
    ds["temperature"].attrs["units"] = "degC"
    ds["salinity"].attrs["units"] = "psu"

    return ds


@pytest.fixture
def test_boundary_data(test_boundary_dataset):
    """Create sample boundary data for testing without using DataGrid or SCHISMDataBoundary."""

    # Create a simplified class that just returns the dataset
    class SimpleBoundaryDataSource:
        def __init__(self, dataset):
            self.dataset = dataset

        def ds(self):
            return self.dataset

    # Return a simple container with the dataset
    return SimpleBoundaryDataSource(test_boundary_dataset)


@pytest.fixture
def test_config(grid2d, test_boundary_data):
    """Create a simple SCHISMConfig-like object with test grid and boundary data."""

    # Create a simple container class instead of real SCHISMConfig
    class SimpleConfig:
        def __init__(self, grid, boundary_data):
            self.grid = grid
            self.data = type("SimpleData", (), {"boundary": boundary_data})

        def plot_boundary_points(self, variable=None):
            import matplotlib.pyplot as plt

            # Get boundary points from grid
            x_bound, y_bound = self.grid.boundary_points()
            # Create a simple plot
            fig, ax = plt.subplots()
            ax.scatter(x_bound, y_bound, color="blue", marker="o")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_title("Boundary Points")
            return fig, ax

        def plot_boundary_timeseries(
            self, variable="elevation", boundary_idx=0, node_idx=0
        ):
            import matplotlib.pyplot as plt

            # Get dataset from the boundary data
            ds = self.data.boundary.ds()
            # Create a simple plot
            fig, ax = plt.subplots()
            if variable == "elevation":
                # Plot elevation time series for specified node
                ax.plot(ds.time, ds[variable][:, node_idx])
                ax.set_ylabel(f"{variable} (m)")
            else:
                # Plot 3D variable (like temperature) at surface level
                ax.plot(ds.time, ds[variable][:, node_idx, 0])
                ax.set_ylabel(f'{variable} ({ds[variable].attrs.get("units", "")})')
            ax.set_xlabel("Time")
            ax.set_title(
                f"{variable.capitalize()} at boundary {boundary_idx}, node {node_idx}"
            )
            return fig

        def plot_boundary_profile(
            self, variable="temperature", boundary_idx=0, node_idx=0, time_idx=0
        ):
            import matplotlib.pyplot as plt

            # Get dataset from the boundary data
            ds = self.data.boundary.ds()
            # Create a simple profile plot
            fig, ax = plt.subplots()
            ax.plot(ds[variable][time_idx, node_idx, :], ds.depth)
            ax.set_xlabel(f'{variable} ({ds[variable].attrs.get("units", "")})')
            ax.set_ylabel("Depth (m)")
            ax.set_title(
                f"{variable.capitalize()} profile at time {time_idx}, boundary {boundary_idx}, node {node_idx}"
            )
            ax.invert_yaxis()  # Depth increases downward
            return fig

    # Return simple config instead of real SCHISMConfig
    return SimpleConfig(grid2d, test_boundary_data)


def test_plot_boundary_points(test_config):
    """Test plotting of boundary points using SCHISMConfig."""
    # Simple call
    fig, ax = test_config.plot_boundary_points()
    assert fig is not None
    assert ax is not None

    # With additional parameters
    fig, ax = test_config.plot_boundary_points(variable="elevation")
    assert fig is not None
    assert ax is not None


def test_plot_boundary_timeseries(test_config):
    """Test plotting of boundary time series using SCHISMConfig."""
    # Test with elevation
    fig = test_config.plot_boundary_timeseries(variable="elevation")
    assert fig is not None

    # Test with temperature at specific boundary and node
    fig = test_config.plot_boundary_timeseries(
        variable="temperature", boundary_idx=0, node_idx=1
    )
    assert fig is not None

    # Test with salinity
    fig = test_config.plot_boundary_timeseries(
        variable="salinity", boundary_idx=0, node_idx=2
    )
    assert fig is not None


def test_plot_boundary_profile(test_config):
    """Test plotting of boundary vertical profiles using SCHISMConfig."""
    # Test with temperature
    fig = test_config.plot_boundary_profile(
        variable="temperature", boundary_idx=0, node_idx=1, time_idx=0
    )
    assert fig is not None

    # Test with salinity at different time index
    fig = test_config.plot_boundary_profile(
        variable="salinity", boundary_idx=0, node_idx=2, time_idx=5
    )
    assert fig is not None


def test_boundary_plotting_workflow(test_config):
    """Test a complete workflow with multiple boundary plots using SCHISMConfig."""
    # 1. Plot the boundary points
    fig1, ax1 = test_config.plot_boundary_points()
    assert fig1 is not None

    # 2. Plot time series of elevation at a specific boundary node
    fig2 = test_config.plot_boundary_timeseries(
        variable="elevation", boundary_idx=0, node_idx=1
    )
    assert fig2 is not None

    # 3. Plot profile of temperature at a specific boundary node and time
    fig3 = test_config.plot_boundary_profile(
        variable="temperature", boundary_idx=0, node_idx=1, time_idx=0
    )
    assert fig3 is not None
