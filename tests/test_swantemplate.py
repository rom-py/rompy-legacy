from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from tests.utils import compare_files

from rompy.core.source import SourceFile
from rompy.core.time import TimeRange
from rompy.core.types import DatasetCoords
from rompy.model import ModelRun
from rompy.swan import Boundnest1, SwanConfig, SwanDataGrid, SwanGrid

HERE = Path(__file__).parent


@pytest.fixture
def grid():
    return SwanGrid(x0=115.68, y0=-32.76, dx=0.001, dy=0.001, nx=390, ny=150, rot=77)


@pytest.fixture
def time():
    # yield TimeRange(start="2000-01-01T00", end="2000-01-10T12", interval="1d")
    yield TimeRange(
        start=datetime(2020, 2, 21, 4), end=datetime(2020, 2, 24, 4), interval="15M"
    )


@pytest.fixture
def nc_bathy(tmpdir):
    # touch temp netcdf file
    bottom = SwanGrid(
        x0=115.68, y0=-32.76, rot=77, nx=391, ny=151, dx=0.001, dy=0.001, exc=-99.0
    )
    source = tmpdir / "bathy.nc"
    # calculate lat/lon manually due to rounding errors in arange
    lat = []
    for nn in range(bottom.ny):
        lat.append(bottom.y0 + (nn * bottom.dy))
    lon = []
    for nn in range(bottom.nx):
        lon.append(bottom.x0 + (nn * bottom.dx))
    ds = xr.Dataset(
        {
            "depth": xr.DataArray(
                np.random.rand(bottom.ny, bottom.nx),
                dims=["lat", "lon"],
                coords={"lat": lat, "lon": lon},
            ),
        }
    )
    ds.to_netcdf(source)
    return SwanDataGrid(
        id="bottom",
        source=SourceFile(uri=source),
        z1="depth",
        var="BOTTOM",
        coords=DatasetCoords(
            x="lon",
            y="lat",
        ),
    )


@pytest.fixture
def nc_bnd(tmpdir, time):
    # Dummy dataset to cover the same time range
    fname = tmpdir / "aus-boundary.nc"
    dset_in = xr.open_dataset(HERE / "data/aus-20230101.nc")
    dset_out = xr.concat(len(time.date_range) * [dset_in.isel(time=[0])], dim="time")
    dset_out = dset_out.assign_coords({"time": time.date_range})
    dset_out["lon"] = dset_out.lon.isel(time=0, drop=True)
    dset_out["lat"] = dset_out.lat.isel(time=0, drop=True)
    dset_out.to_netcdf(fname)

    bnd = Boundnest1(
        id="boundary",
        source=SourceFile(
            uri=fname,
            kwargs=dict(engine="netcdf4"),
        ),
        sel_method="idw",
        tolerance=2.0,
        rectangle="closed",
    )
    return bnd


@pytest.fixture
def nc_data_source(tmpdir, time):
    # touch temp netcdf file
    # setup to replicate what was already there in the model templates
    wind_grid = SwanGrid(
        x0=115.68, y0=-32.76, rot=77, nx=391, ny=151, dx=0.001, dy=0.001, exc=-99.0
    )
    source = tmpdir / "wind_input.nc"

    # calculate lat/lon manually due to rounding errors in arange
    lat = []
    for nn in range(wind_grid.ny):
        lat.append(wind_grid.y0 + (nn * wind_grid.dy))
    lon = []
    for nn in range(wind_grid.nx):
        lon.append(wind_grid.x0 + (nn * wind_grid.dx))

    nt = len(time.date_range)
    ds = xr.Dataset(
        {
            "u": xr.DataArray(
                np.random.rand(nt, wind_grid.ny, wind_grid.nx),
                dims=["time", "latitude", "longitude"],
                coords={
                    "time": time.date_range,
                    "latitude": lat,
                    "longitude": lon,
                },
            ),
            "v": xr.DataArray(
                np.random.rand(nt, wind_grid.ny, wind_grid.nx),
                dims=["time", "latitude", "longitude"],
                coords={
                    "time": time.date_range,
                    "latitude": lat,
                    "longitude": lon,
                },
            ),
        }
    )
    ds.to_netcdf(source)
    return SwanDataGrid(
        id="wind", var="WIND", source=SourceFile(uri=source), z1="u", z2="v"
    )


@pytest.fixture
def config(grid, nc_data_source, nc_bathy, nc_bnd):
    """Create a SwanConfig object."""
    return SwanConfig(
        grid=grid,
        forcing={"bottom": nc_bathy, "wind": nc_data_source, "boundary": nc_bnd},
    )


@pytest.mark.skip(reason="Locations are not written in the new INPUT, investigate")
def test_swantemplate(tmpdir, config):
    """Test the swantemplate function."""
    runtime = ModelRun(
        model_type="swan",
        run_id="test_swantemplate",
        output_dir=str(tmpdir),
        config=config,
    )
    runtime.generate()
    compare_files(
        HERE / "simulations/test_swan_ref/INPUT_NEW",
        tmpdir / runtime.run_id / "INPUT",
    )
