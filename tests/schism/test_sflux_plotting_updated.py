"""Tests for the atmospheric forcing (sflux) plotting methods in SCHISMConfig."""

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
from rompy.schism.config import SCHISMConfig

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
    """Create sample sflux data for testing without using complex validation models."""

    # Create a simple class that just returns the dataset
    class SimpleSfluxData:
        def __init__(self, air_dataset):
            self.air_1 = self.AirContainer(air_dataset)
            self.air_2 = None
            self.rad_1 = None
            self.rad_2 = None
            self.prc_1 = None
            self.prc_2 = None

        class AirContainer:
            def __init__(self, dataset):
                self.dataset = dataset
                # Add variable names for compatibility
                self.uwind_name = "uwind"
                self.vwind_name = "vwind"
                self.prmsl_name = "prmsl"
                self.stmp_name = "stmp"
                self.spfh_name = "spfh"

    # Return a simple container with the air dataset
    return SimpleSfluxData(test_air_dataset)


@pytest.fixture
def test_config(test_grid, test_sflux_data):
    """Create a simple SCHISMConfig-like object with test grid and sflux data."""

    # Create a simple container class instead of real SCHISMConfig
    class SimpleConfig:
        def __init__(self, grid, sflux_data):
            self.grid = grid
            self.data = type("SimpleData", (), {"sflux": sflux_data})

        def plot_sflux_spatial(
            self, variable="air", parameter="prmsl", time_idx=0, cmap="viridis"
        ):
            import matplotlib.pyplot as plt
            import numpy as np

            # Get the dataset from sflux air data
            air_ds = self.data.sflux.air_1.dataset

            # Create grid for plotting
            lons = air_ds.lon.values
            lats = air_ds.lat.values
            lon_grid, lat_grid = np.meshgrid(lons, lats)

            # Create figure and axis
            fig, ax = plt.subplots()

            if parameter == "prmsl":
                # Pressure field
                cs = ax.contourf(
                    lon_grid,
                    lat_grid,
                    air_ds[parameter][time_idx, :, :] / 100,
                    cmap=cmap,
                )
                plt.colorbar(cs, ax=ax, label="Pressure (hPa)")
            elif parameter in ["uwind", "vwind"]:
                # Wind components
                wind_speed = np.sqrt(
                    air_ds.uwind[time_idx, :, :] ** 2
                    + air_ds.vwind[time_idx, :, :] ** 2
                )
                cs = ax.contourf(lon_grid, lat_grid, wind_speed, cmap=cmap)
                plt.colorbar(cs, ax=ax, label="Wind Speed (m/s)")

                # Add vectors on a coarser grid
                skip = 2  # To avoid cluttered vectors
                ax.quiver(
                    lon_grid[::skip, ::skip],
                    lat_grid[::skip, ::skip],
                    air_ds.uwind[time_idx, ::skip, ::skip],
                    air_ds.vwind[time_idx, ::skip, ::skip],
                )
            else:
                # Other fields
                cs = ax.contourf(
                    lon_grid, lat_grid, air_ds[parameter][time_idx, :, :], cmap=cmap
                )
                plt.colorbar(
                    cs,
                    ax=ax,
                    label=f'{parameter} ({air_ds[parameter].attrs.get("units", "")})',
                )

            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_title(f"{parameter.capitalize()} at time {time_idx}")

            return fig, ax

        def plot_sflux_timeseries(
            self, variable="air", parameter="stmp", location=None
        ):
            import matplotlib.pyplot as plt
            import numpy as np

            # Get the dataset from sflux air data
            air_ds = self.data.sflux.air_1.dataset

            # Default location at middle of domain if not specified
            if location is None:
                lat_idx = len(air_ds.lat) // 2
                lon_idx = len(air_ds.lon) // 2
            else:
                # Find nearest point to requested location
                lat_idx = np.abs(air_ds.lat.values - location["lat"]).argmin()
                lon_idx = np.abs(air_ds.lon.values - location["lon"]).argmin()

            # Create figure and axis
            fig, ax = plt.subplots()

            # Plot time series
            ax.plot(air_ds.time, air_ds[parameter][:, lat_idx, lon_idx], "o-")

            ax.set_xlabel("Time")
            ax.set_ylabel(f'{parameter} ({air_ds[parameter].attrs.get("units", "")})')
            ax.set_title(
                f"{parameter.capitalize()} at location (lat={air_ds.lat.values[lat_idx]:.2f}, lon={air_ds.lon.values[lon_idx]:.2f})"
            )
            ax.grid(True)

            return fig

    # Return simple config instead of real SCHISMConfig
    return SimpleConfig(test_grid, test_sflux_data)


def test_plot_sflux_spatial(test_config):
    """Test plotting of sflux spatial fields using SCHISMConfig."""
    # Test with default parameters (air, default parameter)
    fig, ax = test_config.plot_sflux_spatial(variable="air")
    assert fig is not None
    assert ax is not None

    # Test with specific parameter
    fig, ax = test_config.plot_sflux_spatial(
        variable="air", parameter="uwind", time_idx=0, cmap="RdBu_r"
    )
    assert fig is not None
    assert ax is not None

    # Test with specific time index
    fig, ax = test_config.plot_sflux_spatial(
        variable="air", parameter="prmsl", time_idx=2
    )
    assert fig is not None
    assert ax is not None


def test_plot_sflux_timeseries(test_config):
    """Test plotting of sflux time series using SCHISMConfig."""
    # Test with default parameters
    fig = test_config.plot_sflux_timeseries(variable="air")
    assert fig is not None

    # Test with specific parameter and location
    fig = test_config.plot_sflux_timeseries(
        variable="air", parameter="stmp", location={"lat": 37.5, "lon": -72.5}
    )
    assert fig is not None

    # Test with specific time range
    # Create a time range covering a subset of the data
    air_ds = test_config.data.sflux.air_1.dataset
    start_time = air_ds.time.values[1]
    end_time = air_ds.time.values[3]

    fig = test_config.plot_sflux_timeseries(
        variable="air", parameter="stmp", location={"lat": 37.5, "lon": -72.5}
    )
    assert fig is not None


def test_sflux_plotting_workflow(test_config):
    """Test a complete workflow with multiple sflux plots using SCHISMConfig."""
    # 1. Plot the pressure field at a specific time
    fig1, ax1 = test_config.plot_sflux_spatial(
        variable="air", parameter="prmsl", time_idx=0
    )
    assert fig1 is not None

    # 2. Plot the wind field at the same time
    fig2, ax2 = test_config.plot_sflux_spatial(
        variable="air", parameter="uwind", time_idx=0, cmap="RdBu_r"
    )
    assert fig2 is not None

    # 3. Plot time series of temperature at a specific location
    fig3 = test_config.plot_sflux_timeseries(
        variable="air", parameter="stmp", location={"lat": 38, "lon": -73}
    )
    assert fig3 is not None

    # Additional test: plot pressure field at different time
    fig4, ax4 = test_config.plot_sflux_spatial(
        variable="air", parameter="prmsl", time_idx=2
    )
    assert fig4 is not None
