import pytest
from pathlib import Path
import xarray as xr
from wavespectra import read_swan

from rompy.core.time import TimeRange
from rompy.swan.grid import SwanGrid
from rompy.swan.boundary import DatasetIntake, DatasetXarray, DatasetWavespectra, DataBoundary


HERE = Path(__file__).parent


@pytest.fixture(scope="module")
def time():
    yield TimeRange(start="2023-01-01T00", end="2023-01-01T12", interval="1h")


@pytest.fixture(scope="module")
def grid():
    yield SwanGrid(x0=110, y0=-30, dx=0.5, dy=0.5, nx=10, ny=10, rot=77)


def test_dataset_wavespectra(tmpdir):
    dset = xr.open_dataset(HERE / "data/aus-20230101.nc")
    outfile = tmpdir / "aus-20230101.swn"
    dset.spec.to_swan(outfile)
    dataset = DatasetWavespectra(
        uri=outfile,
        reader="read_swan",
    )
    assert hasattr(dataset.open(), "spec")


def test_dataset_xarray():
    dataset = DatasetXarray(
        uri=HERE / "data/aus-20230101.nc",
        engine="netcdf4",
    )
    assert hasattr(dataset.open(), "spec")


def test_dataset_intake():
    dataset = DatasetIntake(
        dataset_id="ausspec",
        catalog_uri=HERE / "data/catalog.yaml",
    )
    assert hasattr(dataset.open(), "spec")


def test_data_boundary_spacing_from_dataset(tmpdir, time, grid):
    bnd = DataBoundary(
        id="westaus",
        dataset=DatasetXarray(
            uri=HERE / "data/aus-20230101.nc",
            engine="netcdf4",
        ),
        sel_method="idw",
        tolerance=2.0,
        rectangle="closed",
    )
    bnd._filter_time(time=time)
    bnd.get(stage_dir=tmpdir, grid=grid)
    ds = read_swan(tmpdir / "westaus.bnd")
    xbnd, ybnd = bnd._boundary_points(grid)
    assert xbnd == pytest.approx(ds.lon.values)
    assert ybnd == pytest.approx(ds.lat.values)


def test_data_boundary_custom_spacing(tmpdir, time, grid):
    bnd = DataBoundary(
        id="westaus",
        dataset=DatasetXarray(
            uri=HERE / "data/aus-20230101.nc",
            engine="netcdf4",
        ),
        spacing=1,
        sel_method="idw",
        tolerance=2.0,
        rectangle="closed",
    )
    bnd._filter_time(time=time)
    bnd.get(stage_dir=tmpdir, grid=grid)
    ds = read_swan(tmpdir / "westaus.bnd")
    xbnd, ybnd = bnd._boundary_points(grid)
    assert xbnd == pytest.approx(ds.lon.values)
    assert ybnd == pytest.approx(ds.lat.values)


def test_data_boundary_spacing_lt_perimeter(tmpdir, time, grid):
    with pytest.raises(ValueError):
        bnd = DataBoundary(
            id="westaus",
            dataset=DatasetXarray(
                uri=HERE / "data/aus-20230101.nc",
                engine="netcdf4",
            ),
            spacing=100,
            sel_method="idw",
            tolerance=2.0,
            rectangle="closed",
        )
        bnd._filter_time(time=time)
        bnd.get(stage_dir=tmpdir, grid=grid)

