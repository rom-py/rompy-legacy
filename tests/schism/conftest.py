"""

Shared fixtures for SCHISM tests.

This module provides reusable pytest fixtures for testing SCHISM functionality.
"""

import logging
from pathlib import Path

import pytest

from rompy.core.data import DataBlob, DataGrid
from rompy.core.filters import Filter
from rompy.core.source import SourceFile
from rompy.core.types import DatasetCoords
from rompy.schism.boundary_core import TidalDataset

# Import directly from the new implementation
from rompy.schism.grid import SCHISMGrid
from rompy.schism.vgrid import VGrid as SchismVGrid

logger = logging.getLogger(__name__)


@pytest.fixture
def test_files_dir():
    """Return path to test files directory (old structure)."""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def hycom_path(test_files_dir):
    """Get the path to HYCOM data."""
    hycomdata = test_files_dir / "hycom.nc"
    if not hycomdata.exists():
        from tests.utils import download_hycom

        logging.info("Hycom test data not found, downloading...")
        logging.info("This may take a while...only has to be done once.")
        download_hycom(dest=test_files_dir, hgrid=test_files_dir / "hgrid.gr3")

    return str(hycomdata)


@pytest.fixture
def hgrid_path(test_files_dir):
    """Return path to a test hgrid.gr3 file."""
    # Look for hgrid.gr3 files in the test data directory
    potential_files = list(test_files_dir.glob("**/hgrid.gr3"))
    if potential_files:
        return potential_files[0]

    # Fall back to the hgrid_20kmto60km file if no hgrid.gr3 found
    fallback = Path(__file__).parent / "hgrid_20kmto60km_rompyschism_testing.gr3"
    if fallback.exists():
        return fallback

    pytest.skip("No suitable hgrid file found for testing")
    return None


@pytest.fixture
def grid2d(test_files_dir):
    """Return a 2D SCHISM grid for testing."""
    # Create a grid with DataBlob for hgrid
    grid = SCHISMGrid(
        hgrid=DataBlob(source=test_files_dir / "hgrid.gr3"),
        drag=1.0,
    )
    return grid


@pytest.fixture
def grid3d(test_files_dir):
    """Return a 3D SCHISM grid with vgrid for testing."""
    vgrid_path = test_files_dir / "vgrid.in"
    if vgrid_path.exists():
        vgrid = DataBlob(source=vgrid_path)
    else:
        # Create a basic vertical grid with default values
        vgrid = SchismVGrid(
            ivcor=2, nvrt=20, h_s=20.0, theta_b=0.5, theta_f=5.0  # LSC2
        )

    # Create the grid with both hgrid and vgrid
    grid = SCHISMGrid(
        hgrid=DataBlob(source=test_files_dir / "hgrid.gr3"),
        vgrid=vgrid,
        drag=1.0,
    )

    return grid


@pytest.fixture
def grid_atmos_source(test_files_dir):
    """Create a source for atmospheric data."""
    return DataGrid(
        source=SourceFile(uri=str(test_files_dir / "air_1.nc")),
        coords=DatasetCoords(t="time", x="lon", y="lat"),
        variables=[
            "air_pressure",
            "air_temperature",
            "specific_humidity",
            "eastward_wind",
            "northward_wind",
        ],
        buffer=0.1,
        filter=Filter(),
        crop_data=True,
    )


@pytest.fixture
def hycom_bnd_elev(test_files_dir, hycom_path):
    """Create a 2D hydrodynamic boundary source."""
    return DataGrid(
        source=SourceFile(uri=hycom_path),
        coords=DatasetCoords(t="time", x="lon", y="lat"),
        variables=["surf_el"],
        buffer=0.1,
        filter=Filter(),
        crop_data=True,
    )


@pytest.fixture
def hycom_bnd_vel(test_files_dir, hycom_path):
    """Create a 2D hydrodynamic boundary source."""
    return DataGrid(
        source=SourceFile(uri=hycom_path),
        coords=DatasetCoords(t="time", x="lon", y="lat"),
        variables=["water_u", "water_v"],
        buffer=0.1,
        filter=Filter(),
        crop_data=True,
    )


@pytest.fixture
def hycom_bnd_temp_3d(test_files_dir, hycom_path):
    """Create a 3D temperature boundary source."""
    return DataGrid(
        source=SourceFile(uri=hycom_path),
        coords=DatasetCoords(t="time", x="lon", y="lat", z="depth"),
        variables=["water_temp"],
        buffer=0.1,
        filter=Filter(),
        crop_data=True,
    )


@pytest.fixture
def hycom_bnd2d(test_files_dir, hycom_path):
    """Create a 3D temperature boundary source."""
    return DataGrid(
        source=SourceFile(uri=hycom_path),
        coords=DatasetCoords(t="time", x="lon", y="lat", z="depth"),
    )


@pytest.fixture
def tidal_data_files(test_files_dir):
    """Return paths to tidal elevation and velocity files for testing."""
    tidal_database = test_files_dir / "tides"
    if not (tidal_database / "oceanum-atlas" / "grid_tpxo9_atlas_30_v2.nc").exists():
        if (tidal_database / "oceanum-atlas.tar.gz").exists():
            import tarfile

            logger.info(f"Unpacking {tidal_database / 'oceanum-atlas.tar.gz'}")
            with tarfile.open(tidal_database / "oceanum-atlas.tar.gz") as tar:
                tar.extractall(path=tidal_database)
    return tidal_database


@pytest.fixture
def tidal_dataset(tidal_data_files):
    """Return a tidal dataset instance for testing."""

    return TidalDataset(
        tidal_database=tidal_data_files,
        constituents=["M2", "S2"],
        tidal_model="OCEANUM-atlas",
    )


@pytest.fixture
def mock_tidal_data():
    """Create mock tidal data for testing."""
    import numpy as np

    # Mock data for testing - enough for any boundary size
    # For elevation: [amplitude, phase]
    # For velocity: [u_amplitude, u_phase, v_amplitude, v_phase]
    def mock_data(self, lons, lats, tnames, data_type="h"):
        num_nodes = len(lons)
        num_constituents = len(tnames)
        if data_type == "h":  # Elevation
            arr = np.zeros((num_nodes, num_constituents, 2))
            arr[..., 0] = 0.5  # amplitude
            arr[..., 1] = 45.0  # phase
            return arr
        elif data_type == "uv":  # Velocity
            arr = np.zeros((num_nodes, num_constituents, 4))
            arr[..., 0] = 0.1  # u_amp
            arr[..., 1] = 30.0  # u_phase
            arr[..., 2] = 0.2  # v_amp
            arr[..., 3] = 60.0  # v_phase
            return arr
        else:
            raise ValueError(f"Unknown data type: {data_type}")

    return mock_data
