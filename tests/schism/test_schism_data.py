import logging
import os
from pathlib import Path

import pytest
# Import test utilities
from test_utils.logging import get_test_logger

# Initialize logger
logger = get_test_logger(__name__)

pytest.importorskip("rompy.schism")
import xarray as xr

from rompy.core.data import DataBlob
from rompy.core.source import SourceFile, SourceIntake
from rompy.schism import SCHISMGrid
from rompy.schism.data import SCHISMDataBoundary, SCHISMDataSflux, SfluxAir

HERE = Path(__file__).parent
DATAMESH_TOKEN = os.environ.get("DATAMESH_TOKEN")

logging.basicConfig(level=logging.INFO)


@pytest.fixture
def grid2d():
    grid = SCHISMGrid(hgrid=DataBlob(source=HERE / "test_data/hgrid.gr3"), drag=1)
    return grid


@pytest.fixture
def grid3d():
    grid = SCHISMGrid(
        hgrid=DataBlob(source=HERE / "test_data/hgrid.gr3"),
        vgrid=DataBlob(source=HERE / "test_data/vgrid.in"),
        drag=1,
    )
    return grid


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
        from tests.utils import download_hycom

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


@pytest.fixture
def hycom_bnd_temp_3d():
    hycomdata = HERE / "test_data" / "hycom.nc"
    if not hycomdata.exists():
        from tests.utils import download_hycom

        logging.info("Hycom test data not found, downloading...")
        logging.info("This may take a while...only has to be done once.")
        download_hycom(dest=HERE / "test_data", hgrid=HERE / "test_data" / "hgrid.gr3")
    return SCHISMDataBoundary(
        id="hycom",
        source=SourceFile(
            uri=HERE / "test_data" / "hycom.nc",
        ),
        variables=["temperature"],
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
    # Ensure boundary data is formatted correctly for the backend

    logging.info(f"Debug: grid2d type is {type(grid2d)}")
    logging.info(
        f"Debug: grid2d has get_boundary_nodes: {hasattr(grid2d, 'get_boundary_nodes')}"
    )
    logging.info(
        f"Debug: grid2d has ocean_boundary: {hasattr(grid2d, 'ocean_boundary')}"
    )
    logging.info(
        f"Debug: grid2d has boundary_points: {hasattr(grid2d, 'boundary_points')}"
    )

    output_path = hycom_bnd2d.get(tmp_path, grid2d)

    with xr.open_dataset(output_path) as bnd:
        assert "one" in bnd.dims
        assert "time" in bnd.dims
        assert "nOpenBndNodes" in bnd.dims
        assert "nLevels" in bnd.dims
        assert "nComponents" in bnd.dims

        # Instead of comparing length, just log the values
        logging.info(f"NetCDF has {len(bnd.nOpenBndNodes)} boundary nodes")

        assert bnd.nOpenBndNodes.size == grid2d.nobn
        assert grid2d.nvrt is None

        logging.info(f"Grid has {len(bnd.nOpenBndNodes)} boundary nodes")
        assert bnd.time_series.isnull().sum() == 0


def test_oceandataboundary3d(tmp_path, grid3d, hycom_bnd_temp_3d):
    # Ensure boundary data is formatted correctly for the backend

    # Generate the boundary data
    output_path = hycom_bnd_temp_3d.get(tmp_path, grid3d)

    with xr.open_dataset(output_path) as bnd:
        assert "one" in bnd.dims
        assert "time" in bnd.dims
        assert "nOpenBndNodes" in bnd.dims
        assert "nLevels" in bnd.dims
        assert "nComponents" in bnd.dims

        # Log information about boundary nodes
        logging.info(f"NetCDF has {len(bnd.nOpenBndNodes)} boundary nodes in 3D test")
        assert bnd.nOpenBndNodes.size == grid3d.nobn
        assert bnd.nLevels.size == grid3d.nvrt

        logging.info(f"Grid has {len(grid3d.nobn)} boundary nodes in 3D test")

        # Skip node count assertion for now
        # assert len(bnd.nOpenBndNodes) == len(boundary_nodes)

        assert bnd.time_series.isnull().sum() == 0
