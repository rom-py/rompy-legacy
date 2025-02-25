import os
from pathlib import Path

import pytest

pytest.importorskip("rompy.schism")
import xarray as xr

from rompy.core import DataBlob, TimeRange
from rompy.core.source import SourceFile, SourceIntake
from rompy.schism import SCHISMGrid
from rompy.schism.data import (
    SCHISMDataBoundary,
    SCHISMDataOcean,
    SCHISMDataSflux,
    SCHISMDataTides,
    SfluxAir,
    TidalDataset,
)

HERE = Path(__file__).parent
DATAMESH_TOKEN = os.environ.get("DATAMESH_TOKEN")
import logging

logging.basicConfig(level=logging.INFO)


@pytest.fixture
def grid2d():
    return SCHISMGrid(hgrid=DataBlob(source=HERE / "test_data/hgrid.gr3"), drag=1)


@pytest.fixture
def grid_atmos_source():
    return SourceIntake(
        dataset_id="era5",
        catalog_uri=HERE / ".." / "data" / "catalog.yaml",
    )


@pytest.fixture
def hycom_bnd2d():
    hycomdata = HERE / "test_data" / "hycom.nc"
    if not hycomdata.exists():
        from utils import download_hycom

        logging.info("Hycom test data not found, downloading...")
        logging.info("This may take a while...only has to be done once.")
        download_hycom(dest=HERE / "test_data", hgrid=HERE / "test_data" / "hgrid.gr3")
    return SCHISMDataBoundary(
        id="hycom",
        source=SourceFile(
            uri=HERE / "test_data" / "hycom.nc",
        ),
        variables=["surf_el"],
        coords={"t": "time", "y": "ylat", "x": "xlon"},
    )


def hycom_bnd3d():
    hycomdata = HERE / "test_data" / "hycom.nc"
    if not hycomdata.exists():
        from utils import download_hycom

        logging.info("Hycom test data not found, downloading...")
        logging.info("This may take a while...only has to be done once.")
        download_hycom(dest=HERE / "test_data", hgrid=HERE / "test_data" / "hgrid.gr3")
    return SCHISMDataBoundary(
        id="hycom",
        source=SourceFile(
            uri=HERE / "test_data" / "hycom.nc",
        ),
        variables=["surf_el"],
        coords={"t": "time", "y": "ylat", "x": "xlon", "z": "depth"},
    )


def test_atmos(tmp_path, grid_atmos_source):
    data = SCHISMDataSflux(
        air_1=SfluxAir(
            id="air_1",
            source=grid_atmos_source,
            uwind_name="u10",
            vwind_name="v10",
            filter={
                "sort": {"coords": ["latitude"]},
                "crop": {
                    "time": slice("2023-01-01", "2023-01-02"),
                    "latitude": slice(0, 20),
                    "longitude": slice(0, 20),
                },
            },
        )
    )
    data.get(tmp_path)


def test_oceandataboundary(tmp_path, grid2d, hycom_bnd2d):
    hycom_bnd2d.get(tmp_path, grid2d)
    with xr.open_dataset(tmp_path / "hycom.th.nc") as bnd:
        assert "one" in bnd.dims
        assert "time" in bnd.dims
        assert "nOpenBndNodes" in bnd.dims
        assert "nLevels" in bnd.dims
        assert "nComponents" in bnd.dims
        assert len(bnd.nOpenBndNodes) == len(grid2d.ocean_boundary()[0])
        assert bnd.time_series.isnull().sum() == 0


def test_oceandata(tmp_path, grid2d, hycom_bnd2d):
    oceandata = SCHISMDataOcean(elev2D=hycom_bnd2d)
    oceandata.get(tmp_path, grid2d)


def test_tidal_boundary(tmp_path, grid2d):
    if not (HERE / "test_data" / "tpxo9-neaus" / "h_m2s2n2.nc").exists():
        from utils import untar_file

        untar_file(HERE / "test_data" / "tpxo9-neaus.tar.gz", HERE / "test_data/")
    from utils import untar_file

    tides = SCHISMDataTides(
        tidal_data=TidalDataset(
            elevations=HERE / "test_data" / "tpxo9-neaus" / "h_m2s2n2.nc",
            velocities=HERE / "test_data" / "tpxo9-neaus" / "u_m2s2n2.nc",
        ),
        constituents=["M2", "S2", "N2"],
    )
    tides.get(
        destdir=tmp_path,
        grid=grid2d,
        time=TimeRange(start="2023-01-01", end="2023-01-02", dt=3600),
    )
