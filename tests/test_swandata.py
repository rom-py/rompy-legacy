import numpy as np
import pandas as pd
import pytest
import xarray as xr

from rompy.core.source import SourceFile
from rompy.core.types import DatasetCoords
from rompy.swan.data import SwanDataGrid
from rompy.swan.grid import SwanGrid


@pytest.fixture
def nc_bathy(tmp_path):
    # touch temp netcdf file
    source = tmp_path / "bathy.nc"
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
        var="bottom",
        source=SourceFile(uri=source),
        z1="depth",
        coords=DatasetCoords(
            x="lon",
            y="lat",
        ),
    )


@pytest.fixture
def nc_data_source(tmp_path):
    # touch temp netcdf file
    source = tmp_path / "test.nc"
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
    return SwanDataGrid(
        var="wind",
        source=SourceFile(uri=source),
        z1="u10",
        z2="v10",
    )


def test_swandata_write(tmp_path, nc_data_source):
    swangrid = SwanGrid(x0=0, y0=0, dx=1, dy=1, nx=10, ny=10)
    config_ref = "INPGRID WIND REG 0.0 0.0 0.0 9 9 1.0 1.0 EXC -99.0 NONSTATION 20000101.000000 24.00 HR\n"
    config_ref += "READINP WIND 1.0 'wind.grd' 3 0 1 0 FREE\n"
    config = nc_data_source.get(tmp_path, swangrid)
    assert config == config_ref


def test_bathy_write(tmp_path, nc_bathy):
    config = nc_bathy.get(tmp_path)
