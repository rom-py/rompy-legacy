"""

Shared fixtures for SCHISM tests.

This module provides reusable pytest fixtures for testing SCHISM functionality.
"""

import logging
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import pytest
import requests

from rompy.core.data import DataBlob, DataGrid
from rompy.core.filters import Filter
from rompy.core.source import SourceFile
from rompy.core.types import DatasetCoords
from rompy.schism.boundary_core import TidalDataset
# Import directly from the new implementation
from rompy.schism.grid import SCHISMGrid
from rompy.schism.vgrid import VGrid as SchismVGrid

DATA_REPO = "rom-py/rompy-test-data"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
GITHUB_API_RELEASES = f"https://api.github.com/repos/{DATA_REPO}/releases/latest"

# Add the tests directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import test utilities for logging
from test_utils.logging import configure_test_logging

logger = logging.getLogger(__name__)


def pytest_addoption(parser):
    """Add command-line options for pytest."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests",
    )
    parser.addoption(
        "--rompy-log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level for ROMPY tests",
    )


def pytest_configure(config):
    """Configure pytest with plugins and settings, and ensure test data is present."""
    import logging

    # Get log level from command line or use default
    log_level_str = config.getoption("--rompy-log-level")
    log_level = getattr(logging, log_level_str)

    # Configure logging for tests
    configure_test_logging(level=log_level_str)


def download_and_extract_data():
    print("Downloading test data from rompy-test-data repo...")
    # Get latest release info
    resp = requests.get(GITHUB_API_RELEASES)
    resp.raise_for_status()
    release = resp.json()
    # Find the zipball URL
    zip_url = release["zipball_url"]
    # Download the zip
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "data.zip")
        with requests.get(zip_url, stream=True) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        # Extract only the data/ directory
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # Find the top-level directory in the zip
            top_level = zip_ref.namelist()[0].split("/")[0]
            # Extract data/ to DATA_DIR
            for member in zip_ref.namelist():
                if member.startswith(f"{top_level}/data/"):
                    zip_ref.extract(member, tmpdir)
            src_data_dir = os.path.join(tmpdir, top_level, "data")
            if os.path.exists(DATA_DIR):
                shutil.rmtree(DATA_DIR)
            shutil.copytree(src_data_dir, DATA_DIR)
    print("Test data downloaded and extracted.")


# Only download if data dir is missing or empty
if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
    try:
        download_and_extract_data()
    except Exception as e:
        print(f"Failed to download test data: {e}", file=sys.stderr)
        sys.exit(1)
    # --- END: Automatic test data download ---


@pytest.fixture(scope="session", autouse=True)
def setup_logging():
    """Set up logging for all tests.

    This fixture runs once per test session and ensures that logging is properly
    configured for all tests.
    """
    # Import here to avoid circular imports
    from test_utils.logging import configure_test_logging

    # Configure logging with default settings
    configure_test_logging()

    # Return a function to reconfigure logging if needed
    return configure_test_logging


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
