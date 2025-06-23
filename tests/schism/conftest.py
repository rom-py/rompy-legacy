"""

Shared fixtures for SCHISM tests.

This module provides reusable pytest fixtures for testing SCHISM functionality.
"""

import os
from pathlib import Path

import pytest
import xarray as xr

from rompy.core.data import DataBlob, DataGrid
from rompy.core.filters import Filter
from rompy.core.source import SourceFile, SourceIntake
from rompy.core.time import TimeRange
from rompy.core.types import DatasetCoords
from rompy.schism.data import (
    SCHISMDataBoundary,
    SCHISMDataSflux,
    SfluxAir,
)
from rompy.schism.boundary_core import (
    BoundaryHandler,
    TidalBoundary,  # Backward compatibility alias
    TidalDataset
)

# Import directly from the new implementation
from rompy.schism.grid import SCHISMGrid
from rompy.schism.vgrid import VGrid as SchismVGrid


@pytest.fixture
def test_data_dir():
    """Return path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def test_files_dir():
    """Return path to test files directory (old structure)."""
    return Path(__file__).parent / "test_data"


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
def hycom_bnd_elev(test_files_dir):
    """Create a 2D hydrodynamic boundary source."""
    return DataGrid(
        source=SourceFile(uri=str(test_files_dir / "hycom.nc")),
        coords=DatasetCoords(t="time", x="lon", y="lat"),
        variables=["surf_el"],
        buffer=0.1,
        filter=Filter(),
        crop_data=True,
    )

@pytest.fixture
def hycom_bnd_vel(test_files_dir):
    """Create a 2D hydrodynamic boundary source."""
    return DataGrid(
        source=SourceFile(uri=str(test_files_dir / "hycom.nc")),
        coords=DatasetCoords(t="time", x="lon", y="lat"),
        variables=["u", "v"],
        buffer=0.1,
        filter=Filter(),
        crop_data=True,
    )


@pytest.fixture
def hycom_bnd_temp_3d(test_files_dir):
    """Create a 3D temperature boundary source."""
    return DataGrid(
        source=SourceFile(uri=str(test_files_dir / "hycom.nc")),
        coords=DatasetCoords(t="time", x="lon", y="lat", z="depth"),
        variables=["water_temp"],
        buffer=0.1,
        filter=Filter(),
        crop_data=True,
    )


@pytest.fixture
def tidal_data_files(test_files_dir):
    """Return paths to tidal elevation and velocity files for testing."""
    tidal_database = test_files_dir / "tides" 
    return tidal_database


@pytest.fixture
def tidal_dataset(tidal_data_files):
    """Return a tidal dataset instance for testing."""
    from rompy.schism.boundary_core import TidalDataset

    return TidalDataset(
        tidal_database=tidal_data_files,
        constituents=["M2", "S2"],
        model="OCEANUM-atlas"
    )


@pytest.fixture
def mock_tidal_data():
    """Create mock tidal data for testing."""
    import numpy as np

    # Mock data for testing - enough for any boundary size
    # For elevation: [amplitude, phase]
    # For velocity: [u_amplitude, u_phase, v_amplitude, v_phase]
    def mock_data(self, lons, lats, constituent, data_type="h"):
        if data_type == "h":  # Elevation
            return np.array([[0.5, 45.0] for _ in range(len(lons))])
        elif data_type == "uv":  # Velocity
            return np.array([[0.1, 30.0, 0.2, 60.0] for _ in range(len(lons))])
        else:
            raise ValueError(f"Unknown data type: {data_type}")

    return mock_data
