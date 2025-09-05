from pathlib import Path

import pytest

# Import test utilities
from test_utils.logging import get_test_logger

# Initialize logger
logger = get_test_logger(__name__)

import xarray as xr
from wavespectra import read_swan

from rompy.core.source import SourceFile, SourceIntake, SourceWavespectra
from rompy.core.time import TimeRange
from rompy.swan.boundary import Boundnest1, BoundspecSegmentXY, BoundspecSide
from rompy.swan.grid import SwanGrid

HERE = Path(__file__).parent


@pytest.fixture(scope="module")
def time():
    yield TimeRange(start="2023-01-01T00", end="2023-01-01T12", interval="1h")


@pytest.fixture(scope="module")
def grid():
    yield SwanGrid(x0=110, y0=-30, dx=0.5, dy=0.5, nx=10, ny=10, rot=20)


def test_source_wavespectra(tmp_path):
    dset = xr.open_dataset(HERE / "data/aus-20230101.nc")
    outfile = tmp_path / "aus-20230101.swn"
    dset.spec.to_swan(outfile)
    dataset = SourceWavespectra(
        uri=outfile,
        reader="read_swan",
    )
    assert hasattr(dataset.open(), "spec")


def test_source_netcdf_formatted_with_wavespectra_conventions():
    dataset = SourceFile(
        uri=HERE / "data/aus-20230101.nc",
        kwargs=dict(engine="netcdf4"),
    )
    assert hasattr(dataset.open(), "spec")


def test_source_intake():
    dataset = SourceIntake(
        dataset_id="ausspec",
        catalog_uri=HERE / "data/catalog.yaml",
    )
    assert hasattr(dataset.open(), "spec")


def test_data_boundary_spacing_from_dataset(tmp_path, time, grid):
    bnd = Boundnest1(
        id="westaus",
        source=SourceFile(
            uri=HERE / "data/aus-20230101.nc",
            kwargs=dict(engine="netcdf4"),
        ),
        sel_method="idw",
        sel_method_kwargs={"tolerance": 2.0},
        rectangle="closed",
    )
    bnd._filter_time(time=time)
    bnd.get(destdir=tmp_path, grid=grid)
    ds = read_swan(tmp_path / "westaus.bnd")
    xbnd, ybnd = grid.boundary_points()
    assert xbnd == pytest.approx(ds.lon.values)
    assert ybnd == pytest.approx(ds.lat.values)


def test_data_boundary_custom_spacing(tmp_path, time, grid):
    bnd = Boundnest1(
        id="westaus",
        source=SourceFile(
            uri=HERE / "data/aus-20230101.nc",
            kwargs=dict(engine="netcdf4"),
        ),
        spacing=1,
        sel_method="idw",
        sel_method_kwargs={"tolerance": 2.0},
        rectangle="closed",
    )
    bnd._filter_time(time=time)
    bnd.get(destdir=tmp_path, grid=grid)
    ds = read_swan(tmp_path / "westaus.bnd")
    xbnd, ybnd = grid.boundary_points(spacing=1)
    assert xbnd == pytest.approx(ds.lon.values)
    assert ybnd == pytest.approx(ds.lat.values)


def test_data_boundary_spacing_lt_perimeter(tmp_path, time, grid):
    with pytest.raises(ValueError):
        bnd = Boundnest1(
            id="westaus",
            source=SourceFile(
                uri=HERE / "data/aus-20230101.nc",
                kwargs=dict(engine="netcdf4"),
            ),
            spacing=100,
            sel_method="idw",
            sel_method_kwargs={"tolerance": 2.0},
            rectangle="closed",
        )
        bnd._filter_time(time=time)
        bnd.get(destdir=tmp_path, grid=grid)


def test_boundspecside(tmp_path, time, grid):
    bnd = BoundspecSide(
        id="westaus",
        source=SourceFile(
            uri=HERE / "data/aus-20230101.nc",
            kwargs=dict(engine="netcdf4"),
        ),
        sel_method="nearest",
        sel_method_kwargs={"tolerance": 2.0},
        location={"side": "west"},
    )
    bnd.get(destdir=tmp_path, grid=grid, time=time)


def test_boundspecsegmentxy_from_side(tmp_path, time, grid):
    bnd = BoundspecSegmentXY(
        id="westaus",
        source=SourceFile(
            uri=HERE / "data/aus-20230101.nc",
            kwargs=dict(engine="netcdf4"),
        ),
        sel_method="idw",
        sel_method_kwargs={"tolerance": 3.0},
        location={"model_type": "side", "side": "west"},
    )
    bnd.get(destdir=tmp_path, grid=grid, time=time)


def test_boundspecsegmentxy_from_sides(tmp_path, time, grid):
    bnd = BoundspecSegmentXY(
        id="westaus",
        source=SourceFile(
            uri=HERE / "data/aus-20230101.nc",
            kwargs=dict(engine="netcdf4"),
        ),
        sel_method="nearest",
        sel_method_kwargs={"tolerance": 3.0},
        location={
            "model_type": "sides",
            "sides": [{"side": "west"}, {"side": "south"}],
        },
    )
    bnd.get(destdir=tmp_path, grid=grid, time=time)


def test_source_wavespectra_ploting(tmp_path):
    Boundnest1(
        id="westaus",
        source=SourceFile(
            uri=HERE / "data/aus-20230101.nc",
            kwargs=dict(engine="netcdf4"),
        ),
        spacing=1,
        sel_method="idw",
        sel_method_kwargs={"tolerance": 2.0},
        rectangle="closed",
        coords={"x": "lon", "y": "lat"},
    ).plot()
