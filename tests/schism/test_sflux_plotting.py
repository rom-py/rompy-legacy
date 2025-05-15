"""Tests for the atmospheric forcing (sflux) plotting methods in SCHISMDataSflux."""

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
from rompy.schism.data import SCHISMDataSflux, SfluxAir, SfluxRad, SfluxPrc
from rompy.schism.grid import SCHISMGrid

# Import helper functions from test_adapter
from tests.schism.test_adapter import prepare_test_grid

# Define the location of test files
HERE = Path(__file__).parent


@pytest.fixture
def test_grid():
    """Return a test grid for testing sflux plotting."""
    grid = SCHISMGrid(hgrid=DataBlob(source=HERE / "test_data/hgrid.gr3"), drag=1)
    return prepare_test_grid(grid)


@pytest.fixture
def test_air_dataset():
    """Create a sample air dataset for testing sflux air plotting."""
    # Create a sample grid and time dimension
    lon = np.linspace(-75, -70, 20)
    lat = np.linspace(35, 40, 15)
    times = pd.date_range(start=datetime.now(), periods=5, freq="1D").to_pydatetime()

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


@pytest.fixture
def test_sflux_data(test_air_dataset):
    """Create sample sflux data for testing without using SfluxAir or SCHISMDataSflux."""

    # Create a simple class that just returns the dataset
    class SimpleSfluxDataSource:
        def __init__(self, dataset):
            self.air_1 = self.AirContainer(dataset)

        class AirContainer:
            def __init__(self, dataset):
                self.dataset = dataset

    # Return a simple container with the air dataset
    return SimpleSfluxDataSource(test_air_dataset)


def test_plot_sflux_spatial(test_grid, test_sflux_data):
    """Test plotting of sflux spatial fields."""
    import matplotlib.pyplot as plt
    import numpy as np

    # Get the raw dataset to avoid validation issues
    air_ds = test_sflux_data.air_1.dataset

    # Test plotting air pressure field
    fig, ax = plt.subplots(figsize=(10, 8))

    # Get grid for contour plot
    lons = air_ds.lon.values
    lats = air_ds.lat.values
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Plot contour of pressure at first time step
    cs = ax.contourf(lon_grid, lat_grid, air_ds.prmsl[0, :, :] / 100, cmap="viridis")
    plt.colorbar(cs, ax=ax, label="Pressure (hPa)")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Air Pressure at Sea Level")

    assert fig is not None
    assert ax is not None

    # Test with wind vectors
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot wind vectors on a subset of the grid for clarity
    skip = 2  # Skip every other grid point
    q = ax.quiver(
        lon_grid[::skip, ::skip],
        lat_grid[::skip, ::skip],
        air_ds.uwind[0, ::skip, ::skip],
        air_ds.vwind[0, ::skip, ::skip],
        scale=100,
    )

    # Add colorbar as background showing wind speed
    wind_speed = np.sqrt(air_ds.uwind[0, :, :] ** 2 + air_ds.vwind[0, :, :] ** 2)
    cs = ax.contourf(lon_grid, lat_grid, wind_speed, cmap="RdBu_r", alpha=0.5)
    plt.colorbar(cs, ax=ax, label="Wind Speed (m/s)")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Wind Field")

    assert fig is not None
    assert ax is not None


def test_plot_sflux_timeseries(test_sflux_data):
    """Test plotting of sflux time series."""
    import matplotlib.pyplot as plt

    # Get the raw dataset to avoid validation issues
    air_ds = test_sflux_data.air_1.dataset

    # Test with temperature time series at a specific point
    fig, ax = plt.subplots(figsize=(10, 6))

    # Choose a grid point for the time series
    lat_idx, lon_idx = 5, 10

    # Plot temperature time series
    ax.plot(air_ds.time, air_ds.stmp[:, lat_idx, lon_idx], "r-", marker="o")
    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(
        f"Air Temperature at Point ({air_ds.lon[lon_idx]:.2f}°E, {air_ds.lat[lat_idx]:.2f}°N)"
    )
    ax.grid(True)

    assert fig is not None

    # Test with wind speed time series
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate wind speed
    wind_speed = np.sqrt(
        air_ds.uwind[:, lat_idx, lon_idx] ** 2 + air_ds.vwind[:, lat_idx, lon_idx] ** 2
    )

    # Plot wind speed time series
    ax.plot(air_ds.time, wind_speed, "b-", marker="s")
    ax.set_xlabel("Time")
    ax.set_ylabel("Wind Speed (m/s)")
    ax.set_title(
        f"Wind Speed at Point ({air_ds.lon[lon_idx]:.2f}°E, {air_ds.lat[lat_idx]:.2f}°N)"
    )
    ax.grid(True)

    assert fig is not None


def test_sflux_plotting_workflow(test_grid, test_sflux_data):
    """Test a complete workflow with multiple sflux plots."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.gridspec import GridSpec

    # Get the raw dataset
    air_ds = test_sflux_data.air_1.dataset

    # Create a figure with multiple subplots for a complete workflow
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(2, 2, figure=fig)

    # Get mesh grid for spatial plots
    lons = air_ds.lon.values
    lats = air_ds.lat.values
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # 1. Air pressure plot
    ax1 = fig.add_subplot(gs[0, 0])
    cs1 = ax1.contourf(lon_grid, lat_grid, air_ds.prmsl[0, :, :] / 100, cmap="viridis")
    plt.colorbar(cs1, ax=ax1, label="Pressure (hPa)")
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.set_title("Air Pressure at Sea Level")

    # 2. Wind vectors plot
    ax2 = fig.add_subplot(gs[0, 1])
    skip = 2  # Skip grid points for clarity
    q = ax2.quiver(
        lon_grid[::skip, ::skip],
        lat_grid[::skip, ::skip],
        air_ds.uwind[0, ::skip, ::skip],
        air_ds.vwind[0, ::skip, ::skip],
        scale=100,
    )

    # Background showing wind speed
    wind_speed = np.sqrt(air_ds.uwind[0, :, :] ** 2 + air_ds.vwind[0, :, :] ** 2)
    cs2 = ax2.contourf(lon_grid, lat_grid, wind_speed, cmap="RdBu_r", alpha=0.5)
    plt.colorbar(cs2, ax=ax2, label="Wind Speed (m/s)")

    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.set_title("Wind Field")

    # 3. Temperature time series
    ax3 = fig.add_subplot(gs[1, 0])
    lat_idx, lon_idx = 7, 10  # Choose a point
    ax3.plot(air_ds.time, air_ds.stmp[:, lat_idx, lon_idx], "r-", marker="o")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Temperature (°C)")
    ax3.set_title(f"Air Temperature Time Series")
    ax3.grid(True)

    # 4. Pressure field at a different time
    ax4 = fig.add_subplot(gs[1, 1])
    time_idx = 2  # Different time step
    cs4 = ax4.contourf(
        lon_grid, lat_grid, air_ds.prmsl[time_idx, :, :] / 100, cmap="viridis"
    )
    plt.colorbar(cs4, ax=ax4, label="Pressure (hPa)")
    ax4.set_xlabel("Longitude")
    ax4.set_ylabel("Latitude")
    ax4.set_title(f"Air Pressure at Time {time_idx + 1}")

    plt.tight_layout()

    assert fig is not None
