"""Tests for the visualization methods in SCHISM Config class."""

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

pytest.importorskip("rompy.schism")

# Import yaml for configuration loading
import yaml

from rompy.core.data import DataBlob, DataGrid
from rompy.core.time import TimeRange
from rompy.model import ModelRun
from rompy.schism import SCHISMGrid
from rompy.schism.config import SCHISMConfig
from rompy.schism.config_plotting import plot_sflux_spatial, plot_sflux_timeseries

# Import the plotting modules directly
from rompy.schism.config_plotting_boundary import (
    plot_boundary_points,
    plot_boundary_profile,
    plot_boundary_timeseries,
)
from rompy.schism.config_plotting_tides import plot_tidal_boundaries, plot_tidal_rose

# Import helper functions from test_adapter
from tests.schism.test_adapter import prepare_test_grid

HERE = Path(__file__).parent
import logging

logging.basicConfig(level=logging.INFO)


@pytest.fixture
def grid():
    """Return a test grid for visualization tests."""
    grid = SCHISMGrid(hgrid=DataBlob(source=HERE / "test_data/hgrid.gr3"), drag=1)
    return prepare_test_grid(grid)


@pytest.fixture
def boundary_ds():
    """Create a sample dataset for boundary visualization tests."""
    # Create a sample dataset
    times = np.array([datetime.now() + timedelta(hours=i) for i in range(5)])
    boundary_nodes = np.array([0, 1, 2, 3, 4])

    # Create 2D variable (time, node)
    elevation = np.sin(np.linspace(0, 2 * np.pi, 5 * 5)).reshape(5, 5)

    # Create 3D variable (time, node, level)
    num_levels = 3
    temperature = np.zeros((5, 5, num_levels))
    for t in range(5):
        for n in range(5):
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

    return ds


@pytest.fixture
def sflux_ds():
    """Create a sample dataset for sflux visualization tests."""
    # Create a sample grid and time dimension
    lon = np.linspace(-75, -70, 20)
    lat = np.linspace(35, 40, 15)
    times = np.array([datetime.now() + timedelta(hours=i) for i in range(5)])

    # Create 2D meshgrid for spatial data
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Create some sample data
    num_times = len(times)
    num_lats = len(lat)
    num_lons = len(lon)

    # Air pressure at mean sea level
    prmsl = np.zeros((num_times, num_lats, num_lons))
    # Wind components
    uwind = np.zeros((num_times, num_lats, num_lons))
    vwind = np.zeros((num_times, num_lats, num_lons))
    # Air temperature
    stmp = np.zeros((num_times, num_lats, num_lons))
    # Specific humidity
    spfh = np.zeros((num_times, num_lats, num_lons))

    # Fill with meaningful data
    for t in range(num_times):
        # Center of low pressure system moving across the domain
        center_lon = -75 + t * 1.0
        center_lat = 35 + t * 0.8

        for i in range(num_lats):
            for j in range(num_lons):
                # Distance from pressure center
                dist = np.sqrt((lon[j] - center_lon) ** 2 + (lat[i] - center_lat) ** 2)

                # Pressure field (low in center, higher outward)
                prmsl[t, i, j] = 101300 - 2000 * np.exp(-0.5 * dist**2)

                # Wind field (circular around low pressure)
                dx = lon[j] - center_lon
                dy = lat[i] - center_lat
                wind_speed = 10 * (1 - np.exp(-0.3 * dist))
                angle = np.arctan2(dy, dx) + np.pi / 2  # Counter-clockwise around low
                uwind[t, i, j] = wind_speed * np.cos(angle)
                vwind[t, i, j] = wind_speed * np.sin(angle)

                # Temperature decreases with latitude and time
                stmp[t, i, j] = 25 - 0.2 * (lat[i] - 35) - 0.5 * t

                # Humidity has a similar but weaker pattern to pressure
                spfh[t, i, j] = 0.015 - 0.005 * np.exp(-0.5 * dist**2)

    # Create xarray dataset
    ds = xr.Dataset(
        {
            "prmsl": (["time", "lat", "lon"], prmsl),
            "uwind": (["time", "lat", "lon"], uwind),
            "vwind": (["time", "lat", "lon"], vwind),
            "stmp": (["time", "lat", "lon"], stmp),
            "spfh": (["time", "lat", "lon"], spfh),
        },
        coords={
            "time": times,
            "lat": lat,
            "lon": lon,
        },
    )

    # Add unit attributes
    ds["prmsl"].attrs["units"] = "Pa"
    ds["uwind"].attrs["units"] = "m/s"
    ds["vwind"].attrs["units"] = "m/s"
    ds["stmp"].attrs["units"] = "degC"
    ds["spfh"].attrs["units"] = "kg/kg"

    return ds


# Create a simple container class for test objects
class SimpleContainer:
    pass


@pytest.fixture
def test_container(grid, boundary_ds, sflux_ds):
    """Create a simple container with all the objects needed for testing."""
    container = SimpleContainer()

    # Add the grid
    container.grid = grid

    # Set up ocean data with boundary information
    # Create ocean boundary structure
    container.boundary_source = SimpleContainer()
    container.boundary_source.dataset = boundary_ds

    # Create ocean data container
    container.ocean = SimpleContainer()
    container.ocean.boundary = SimpleContainer()
    container.ocean.boundary.source = container.boundary_source

    # Set up atmospheric data
    # Create a source object with the atmospheric dataset
    container.air_source = SimpleContainer()
    container.air_source.dataset = sflux_ds

    # Create an air object
    container.air = SimpleContainer()
    container.air.source = container.air_source

    # Create the atmos object
    container.atmos = SimpleContainer()
    container.atmos.air_1 = container.air

    # Create a data object with the SCHISMData structure
    container.data = SimpleContainer()
    container.data.atmos = container.atmos
    container.data.ocean = container.ocean
    container.data.wave = None
    container.data.tides = None

    # Add legacy attributes for backward compatibility with existing plotting code
    # These will be used by the plotting code that still expects 'boundary' and 'sflux'
    container.data.boundary = container.ocean.boundary
    container.data.sflux = container.atmos

    return container


def test_boundary_visualization(test_container, tmp_path):
    """Test the visualization methods for ocean boundaries."""
    # Create a test object that mimics what SCHISMConfig would have
    mock_self = SimpleContainer()
    mock_self.grid = test_container.grid
    mock_self.data = test_container.data

    # Test plot_boundary_points - call the function directly
    fig, ax = plot_boundary_points(mock_self, variable="elevation")
    fig.savefig(tmp_path / "boundary_points.png")
    plt.close(fig)

    # Test plot_boundary_timeseries - call the function directly
    fig = plot_boundary_timeseries(
        mock_self, variable="elevation", boundary_idx=0, node_idx=0
    )
    fig.savefig(tmp_path / "boundary_timeseries.png")
    plt.close(fig)

    # Test plot_boundary_profile - call the function directly
    fig = plot_boundary_profile(
        mock_self, variable="temperature", boundary_idx=0, node_idx=0, time_idx=0
    )
    fig.savefig(tmp_path / "boundary_profile.png")
    plt.close(fig)

    assert (tmp_path / "boundary_points.png").exists()
    assert (tmp_path / "boundary_timeseries.png").exists()
    assert (tmp_path / "boundary_profile.png").exists()


def test_sflux_visualization(test_container, tmp_path):
    """Test the visualization methods for atmospheric forcing."""
    # Create a test object that mimics what SCHISMConfig would have
    mock_self = SimpleContainer()
    mock_self.grid = test_container.grid
    mock_self.data = test_container.data

    # Test plot_sflux_spatial - call the function directly
    fig, ax = plot_sflux_spatial(
        mock_self, variable="air", parameter="prmsl", time_idx=0
    )
    fig.savefig(tmp_path / "sflux_spatial.png")
    plt.close(fig)

    # Get the dataset for coordinates
    atmos_ds = test_container.air_source.dataset

    # Pick a point in the middle of the domain for timeseries
    lon_idx = len(atmos_ds.lon) // 2
    lat_idx = len(atmos_ds.lat) // 2
    lon_val = float(atmos_ds.lon.values[lon_idx])
    lat_val = float(atmos_ds.lat.values[lat_idx])

    # Test plot_sflux_timeseries - call the function directly
    fig, ax = plot_sflux_timeseries(
        mock_self, variable="air", parameter="stmp", lon=lon_val, lat=lat_val
    )
    fig.savefig(tmp_path / "sflux_timeseries.png")
    plt.close(fig)

    assert (tmp_path / "sflux_spatial.png").exists()
    assert (tmp_path / "sflux_timeseries.png").exists()


# Add a comprehensive workflow test that tests multiple plotting functions together
def test_visualization_workflow(test_container, tmp_path):
    """Test a comprehensive visualization workflow with multiple plots."""
    # Create a test object that mimics what SCHISMConfig would have
    mock_self = SimpleContainer()
    mock_self.grid = test_container.grid
    mock_self.data = test_container.data

    # Create a multi-panel figure to demonstrate comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot boundary points in first panel
    plot_boundary_points(mock_self, ax=axes[0, 0])
    axes[0, 0].set_title("Boundary Points")

    # Plot boundary timeseries in second panel
    plot_boundary_timeseries(
        mock_self, variable="elevation", boundary_idx=0, node_idx=0, ax=axes[0, 1]
    )
    axes[0, 1].set_title("Boundary Elevation Timeseries")

    # Plot sflux spatial field in third panel
    plot_sflux_spatial(
        mock_self, variable="air", parameter="prmsl", time_idx=0, ax=axes[1, 0]
    )
    axes[1, 0].set_title("Air Pressure Field")

    # Plot boundary profile in fourth panel
    plot_boundary_profile(
        mock_self,
        variable="temperature",
        boundary_idx=0,
        node_idx=0,
        time_idx=0,
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Temperature Profile")

    # Adjust layout and save
    plt.tight_layout()
    fig.savefig(tmp_path / "visualization_workflow.png")
    plt.close(fig)

    assert (tmp_path / "visualization_workflow.png").exists()


@pytest.mark.slow
@pytest.mark.skip("visualization needs some work")
def test_realistic_visualization(tmp_path):
    """Test visualization using a real configuration file."""
    # Load the configuration from demo_nml.yaml
    config_path = HERE / ".." / ".." / "notebooks" / "schism" / "demo_nml.yaml"
    config_dir = config_path.parent

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Fix relative paths in the configuration
    model_config = config["config"]
    data_config = model_config.get("data", {})

    # Fix wave data paths if applicable
    if (
        "wave" in data_config
        and "source" in data_config["wave"]
        and "catalog_uri" in data_config["wave"]["source"]
    ):
        source = data_config["wave"]["source"]
        if isinstance(source["catalog_uri"], str) and source["catalog_uri"].startswith(
            ".."
        ):
            source["catalog_uri"] = str((config_dir / source["catalog_uri"]).resolve())

    # Initialize ModelRun with the loaded configuration
    model_run = ModelRun(**config)

    # Create output directory for test figures
    output_dir = tmp_path / "realistic_viz"
    output_dir.mkdir(exist_ok=True)

    # Use ModelRun to produce the data
    model_run()

    # Generate various plots using the native plotting methods

    # 1. Atmospheric fields
    if (
        hasattr(model_run.config.data, "atmos")
        and model_run.config.data.atmos is not None
    ):
        # Load the dataset from the source

        fig, ax = model_run.config.plot_sflux_spatial(
            variable="air", parameter="uwind", time_idx=0
        )
        fig.suptitle("Wind U-Component (First Timestep)")
        fig.savefig(output_dir / "air_uwind.png")
        plt.close(fig)

        # Plot air pressure
        fig, ax = model_run.config.plot_sflux_spatial(
            variable="air", parameter="prmsl", time_idx=0
        )
        fig.suptitle("Air Pressure at Mean Sea Level (First Timestep)")
        fig.savefig(output_dir / "air_prmsl.png")
        plt.close(fig)

        # Create a multi-panel atmospheric plot
        fig, axes = plt.subplots(2, 1, figsize=(10, 16))

        # Air pressure spatial field
        fig, ax = model_run.config.plot_sflux_spatial(
            variable="air", parameter="prmsl", time_idx=0, ax=axes[0]
        )
        axes[0].set_title("Air Pressure Field")

        # Wind field - using special wind vectors plot mode
        fig, ax = model_run.config.plot_sflux_spatial(
            variable="air", parameter="wind", time_idx=0, ax=axes[1]
        )
        axes[1].set_title("Wind Field")

        plt.tight_layout()
        fig.savefig(output_dir / "atmospheric_fields.png")
        plt.close(fig)

    # 2. Tidal information if available
    if (
        hasattr(model_run.config.data, "tides")
        and model_run.config.data.tides is not None
    ):
        # Plot tidal constituents at boundaries
        fig, ax = model_run.config.plot_tidal_boundaries()
        fig.savefig(output_dir / "tidal_boundaries.png")
        plt.close(fig)

        # Plot tidal rose for dominant constituents
        fig = model_run.config.plot_tidal_rose()
        fig.savefig(output_dir / "tidal_rose.png")
        plt.close(fig)

    # 3. Boundary data if available
    has_boundary_data = False
    if (
        hasattr(model_run.config.data, "ocean")
        and hasattr(model_run.config.data.ocean, "boundary")
        and model_run.config.data.ocean.boundary is not None
    ):
        has_boundary_data = True
    elif (
        hasattr(model_run.config.data, "boundary")
        and model_run.config.data.boundary is not None
    ):
        has_boundary_data = True

    if has_boundary_data:
        # Plot boundary points
        fig, ax = model_run.config.plot_boundary_points()
        fig.suptitle("Model Boundary Points")
        fig.savefig(output_dir / "boundary_points.png")
        plt.close(fig)

        # Plot boundary timeseries
        fig, ax = model_run.config.plot_boundary_timeseries(boundary_idx=0, node_idx=0)
        fig.savefig(output_dir / "boundary_timeseries.png")
        plt.close(fig)

    # 4. Create a combined multi-panel overview figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Model grid and boundaries
    if has_boundary_data:
        fig, ax = model_run.config.plot_boundary_points(ax=axes[0, 0])
        axes[0, 0].set_title("Model Grid and Boundaries")
    else:
        # If no boundary data, just plot the grid
        model_run.config.grid.plot(ax=axes[0, 0])
        axes[0, 0].set_title("Model Grid")

    # Panel 2: Air pressure field
    if (
        hasattr(model_run.config.data, "atmos")
        and model_run.config.data.atmos is not None
    ):
        fig, ax = model_run.config.plot_sflux_spatial(
            variable="air",
            parameter="prmsl",
            time_idx=0,
            ax=axes[0, 1],
        )
        axes[0, 1].set_title("Air Pressure Field")
    else:
        axes[0, 1].text(
            0.5,
            0.5,
            "Air pressure data not available",
            ha="center",
            va="center",
            transform=axes[0, 1].transAxes,
        )
        axes[0, 1].set_title("Air Pressure Field")

    # Panel 3: Wind field
    if (
        hasattr(model_run.config.data, "atmos")
        and model_run.config.data.atmos is not None
    ):
        fig, ax = model_run.config.plot_sflux_spatial(
            variable="air",
            parameter="wind",
            time_idx=0,
            ax=axes[1, 0],
        )
        axes[1, 0].set_title("Wind Field")
    else:
        axes[1, 0].text(
            0.5,
            0.5,
            "Wind field data not available",
            ha="center",
            va="center",
            transform=axes[1, 0].transAxes,
        )
        axes[1, 0].set_title("Wind Field")

    # Panel 4: Tidal constituents or another relevant plot
    if (
        hasattr(model_run.config.data, "tides")
        and model_run.config.data.tides is not None
    ):
        # If we have tides data, try to make a simple plot of it
        # Note: plot_tidal_rose creates its own figure, so we create a simpler plot here
        model_run.config.grid.plot(ax=axes[1, 1])
        axes[1, 1].set_title("Model Domain")
    elif (
        hasattr(model_run.config.data, "atmos")
        and model_run.config.data.atmos is not None
    ):
        # Fallback to temperature or another field if available
        fig, ax = model_run.config.plot_sflux_spatial(
            variable="air",
            parameter="stmp",
            time_idx=0,
            ax=axes[1, 1],
        )
        axes[1, 1].set_title("Air Temperature")
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "No additional data available",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
        )
        axes[1, 1].set_title("Additional Data")

    plt.tight_layout()
    fig.savefig(output_dir / "model_overview.png")
    plt.close(fig)

    # Verify output files were created
    assert (output_dir / "model_overview.png").exists(), "Overview plot was not created"
    print(f"Generated visualization plots in {output_dir}")
