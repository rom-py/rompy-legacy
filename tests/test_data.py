import os
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import rompy
from rompy.core import (BaseGrid, DataBlob, DataGrid, DatasetIntake,
                        DatasetXarray, TimeRange)


# create dummy local datasource for testing
@pytest.fixture
def txt_data_source():
    # touch temp text file
    tmp_path = tempfile.mkdtemp()
    source = os.path.join(tmp_path, "test.txt")
    with open(source, "w") as f:
        f.write("hello world")
    return DataBlob(id="test", path=source)


@pytest.fixture
def grid_data_source():
    return DataGrid(
        id="wind",
        catalog=os.path.join(rompy.__path__[0], "catalogs", "oceanum.yaml"),
        dataset="era5_wind10m",
        filter={
            "sort": {"coords": ["latitude"]},
            "crop": {
                "time": slice("2000-01-01", "2000-01-02"),
                "latitude": slice(0, 10),
                "longitude": slice(0, 10),
            },
        },
    )


def test_get(txt_data_source):
    ds = txt_data_source
    output = ds.get("./test.txt")
    assert output.path.is_file()


def test_get_no_path(txt_data_source):
    ds = txt_data_source
    with pytest.raises(TypeError):
        ds.get()


def test_fails_both_path_and_url():
    with pytest.raises(ValueError):
        DataBlob(path="foo", url="bar")


@pytest.mark.skip(reason="not reproducible outside of oceanum")
def test_intake_grid(grid_data_source):
    data = grid_data_source
    assert data.ds.latitude.max() == 10
    assert data.ds.latitude.min() == 0
    assert data.ds.longitude.max() == 10
    assert data.ds.longitude.min() == 0
    downloaded = data.get("simulations")
    assert downloaded.ds.latitude.max() == 10
    assert downloaded.ds.latitude.min() == 0
    assert downloaded.ds.longitude.max() == 10
    assert downloaded.ds.longitude.min() == 0


@pytest.fixture
def nc_data_source():
    # touch temp netcdf file
    tmp_path = tempfile.mkdtemp()
    source = os.path.join(tmp_path, "test.nc")
    ds = xr.Dataset(
        {
            "data": xr.DataArray(
                np.random.rand(10, 10, 10),
                dims=["time", "latitude", "longitude"],
                coords={
                    "time": pd.date_range("2000-01-01", periods=10),
                    "latitude": np.arange(0, 10),
                    "longitude": np.arange(0, 10),
                },
            )
        }
    )
    ds.to_netcdf(source)
    return DataGrid(id="grid", dataset=DatasetXarray(uri=source))


def test_netcdf_grid(nc_data_source):
    data = nc_data_source
    assert data.ds.latitude.max() == 9
    assert data.ds.latitude.min() == 0
    assert data.ds.longitude.max() == 9
    assert data.ds.longitude.min() == 0


def test_grid_filter(nc_data_source):
    grid = BaseGrid(x=np.arange(2, 7), y=np.arange(3, 7))
    nc_data_source._filter_grid(grid)
    assert nc_data_source.ds.latitude.max() == 6
    assert nc_data_source.ds.latitude.min() == 3
    assert nc_data_source.ds.longitude.max() == 6
    assert nc_data_source.ds.longitude.min() == 2


def test_grid_filter_buffer(nc_data_source):
    grid = BaseGrid(x=np.arange(3, 7), y=np.arange(3, 7))
    nc_data_source._filter_grid(grid, buffer=1)
    assert nc_data_source.ds.latitude.max() == 7
    assert nc_data_source.ds.latitude.min() == 2
    assert nc_data_source.ds.longitude.max() == 7
    assert nc_data_source.ds.longitude.min() == 2


def test_time_filter(nc_data_source):
    grid = BaseGrid(x=np.arange(3, 7), y=np.arange(3, 7))
    nc_data_source._filter_grid(grid, buffer=1)
    nc_data_source._filter_time(TimeRange(start="2000-01-02", end="2000-01-03"))
    assert nc_data_source.ds.latitude.max() == 7
    assert nc_data_source.ds.latitude.min() == 2
    assert nc_data_source.ds.longitude.max() == 7
    assert nc_data_source.ds.longitude.min() == 2
    assert nc_data_source.ds.time.max() == np.datetime64("2000-01-03")
    assert nc_data_source.ds.time.min() == np.datetime64("2000-01-02")
