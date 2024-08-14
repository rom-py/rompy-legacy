import os
from pathlib import Path

import intake
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pydantic import ValidationError

from rompy.core import DataBlob, DataGrid, RegularGrid, TimeRange
from rompy.core.data import (SourceDatamesh, SourceDataset, SourceFile,
                             SourceIntake)
from rompy.core.filters import Filter
from rompy.core.types import DatasetCoords

HERE = Path(__file__).parent
DATAMESH_TOKEN = os.environ.get("DATAMESH_TOKEN")


# create dummy local datasource for testing
@pytest.fixture
def txt_data_source(tmp_path):
    # touch temp text file
    source = tmp_path / "test.txt"
    with open(source, "w") as f:
        f.write("hello world")
    return DataBlob(id="test", source=source)


@pytest.fixture
def grid():
    return RegularGrid(x0=2, y0=3, dx=1, dy=1, nx=5, ny=4)


@pytest.fixture
def grid_data_source():
    return DataGrid(
        id="wind",
        source=SourceIntake(
            dataset_id="era5",
            catalog_uri=HERE / "data" / "catalog.yaml",
        ),
        filter={
            "sort": {"coords": ["latitude"]},
            "crop": {
                "time": slice("2023-01-01", "2023-01-02"),
                "latitude": slice(0, 20),
                "longitude": slice(0, 20),
            },
        },
    )


@pytest.fixture
def nc_data_source(tmp_path):
    # touch temp netcdf file
    source = tmp_path / "test.nc"
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
    return DataGrid(id="grid", source=SourceFile(uri=source))


def test_get(tmp_path, txt_data_source):
    ds = txt_data_source
    output = ds.get(tmp_path)
    assert output.is_file()


def test_get_no_path(txt_data_source):
    ds = txt_data_source
    with pytest.raises(TypeError):
        ds.get()


def test_intake_grid(tmp_path, grid_data_source):
    data = grid_data_source
    assert data.ds.latitude.max() == 20
    assert data.ds.latitude.min() == 0
    assert data.ds.longitude.max() == 20
    assert data.ds.longitude.min() == 0
    outfile = downloaded = data.get(tmp_path)
    dset = xr.open_dataset(outfile)
    assert dset.equals(data.ds)


def test_netcdf_grid(nc_data_source):
    data = nc_data_source
    assert data.ds.latitude.max() == 9
    assert data.ds.latitude.min() == 0
    assert data.ds.longitude.max() == 9
    assert data.ds.longitude.min() == 0


def test_grid_filter(nc_data_source, grid):
    nc_data_source._filter_grid(grid)
    assert nc_data_source.ds.latitude.max() == 6
    assert nc_data_source.ds.latitude.min() == 3
    assert nc_data_source.ds.longitude.max() == 6
    assert nc_data_source.ds.longitude.min() == 2


def test_grid_filter_buffer(nc_data_source, grid):
    nc_data_source.buffer = 1.0
    nc_data_source._filter_grid(grid)
    assert nc_data_source.ds.latitude.max() == 7
    assert nc_data_source.ds.latitude.min() == 2
    assert nc_data_source.ds.longitude.max() == 7
    assert nc_data_source.ds.longitude.min() == 1


def test_time_filter(nc_data_source, grid):
    nc_data_source._filter_time(TimeRange(start="2000-01-02", end="2000-01-03"))
    assert nc_data_source.ds.time.max() == np.datetime64("2000-01-03")
    assert nc_data_source.ds.time.min() == np.datetime64("2000-01-02")


def test_source_dataset():
    dset = xr.open_dataset(HERE / "data" / "aus-20230101.nc")
    source = SourceDataset(obj=dset)
    assert isinstance(source.open(), xr.Dataset)


def test_source_file():
    source = SourceFile(uri=HERE / "data" / "aus-20230101.nc")
    assert isinstance(source.open(), xr.Dataset)


def test_source_intake_uri():
    source = SourceIntake(
        dataset_id="ausspec",
        catalog_uri=HERE / "data" / "catalog.yaml",
    )
    assert isinstance(source.open(), xr.Dataset)


def test_source_intake_yaml():
    dataset = intake.open_netcdf(str(HERE / "data/era5-20230101.nc"))
    dataset.name = "era5"
    source = SourceIntake(
        dataset_id="era5",
        catalog_yaml=dataset.yaml(),
    )
    assert isinstance(source.open(), xr.Dataset)


def test_source_intake_uri_or_yaml():
    dataset = intake.open_netcdf(str(HERE / "data/era5-20230101.nc"))
    dataset.name = "era5"
    with pytest.raises(ValidationError):
        SourceIntake(dataset_id="era5")
        SourceIntake(
            dataset_id="era5",
            catalog_uri=HERE / "data" / "catalog.yaml",
            catalog_yaml=dataset.yaml(),
        )


def test_intake_grid_plot(grid_data_source):
    data = grid_data_source
    data.plot(param="u10", isel={"time": 0})


@pytest.mark.skipif(DATAMESH_TOKEN is None, reason="Datamesh token required")
def test_source_datamesh():
    dataset = SourceDatamesh(datasource="era5_wind10m", token=DATAMESH_TOKEN)
    filters = Filter()
    filters.crop.update(dict(time=slice("2000-01-01T00:00:00", "2000-01-01T03:00:00")))
    filters.crop.update(
        dict(longitude=slice(115.5, 116.0), latitude=slice(-33.0, -32.5))
    )
    dset = dataset.open(
        variables=["u10"],
        filters=filters,
        coords=DatasetCoords(x="longitude", y="latitude"),
    )
    assert isinstance(dset, xr.Dataset)
