import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from rompy.swan.config import SwanConfig
from rompy.swan.data import Swan_accessor, SwanDataGrid
from rompy.swan.grid import SwanGrid


@pytest.fixture
def nc_bathy():
    # touch temp netcdf file
    tmp_path = tempfile.mkdtemp()
    source = os.path.join(tmp_path, "bathy.nc")
    ds = xr.Dataset(
        {
            "depth": xr.DataArray(
                np.random.rand(10, 10),
                dims=["lat", "lon"],
                coords={
                    "lat": np.arange(0, 10),
                    "lon": np.arange(0, 10),
                },
            ),
        }
    )
    ds.to_netcdf(source)
    return SwanDataGrid(
        id="bottom", path=source, z1="depth", var="BOTTOM", latname="lat", lonname="lon"
    )


@pytest.fixture
def nc_data_source():
    # touch temp netcdf file
    tmp_path = tempfile.mkdtemp()
    source = os.path.join(tmp_path, "test.nc")
    ds = xr.Dataset(
        {
            "u10": xr.DataArray(
                np.random.rand(10, 10, 10),
                dims=["time", "latitude", "longitude"],
                coords={
                    "time": pd.date_range("2000-01-01", periods=10),
                    "latitude": np.arange(0, 10),
                    "longitude": np.arange(0, 10),
                },
            ),
            "v10": xr.DataArray(
                np.random.rand(10, 10, 10),
                dims=["time", "latitude", "longitude"],
                coords={
                    "time": pd.date_range("2000-01-01", periods=10),
                    "latitude": np.arange(0, 10),
                    "longitude": np.arange(0, 10),
                },
            ),
        }
    )
    ds.to_netcdf(source)
    return SwanDataGrid(id="wind", path=source, z1="u10", z2="v10", variable="WIND")


def test_swandata_write(nc_data_source):
    staging_dir = tempfile.mkdtemp()
    swangrid = SwanGrid(x0=0, y0=0, dx=1, dy=1, nx=10, ny=10)
    config_ref = (
        "INPGRID WIND REG 0.0 0.0 0.0 9 9 1.0 1.0 NONSTATION 20000101.000000 24.0 HR\n"
    )
    config_ref += "READINP WIND 1 'wind.grd' 3 0 1 0 FREE\n"
    config = nc_data_source.get(staging_dir, swangrid)
    assert config == config_ref


def test_bathy_write(nc_bathy):
    staging_dir = tempfile.mkdtemp()
    config = nc_bathy.get(staging_dir)
