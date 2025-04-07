"""

Shared fixtures for SCHISM tests.

This module provides reusable pytest fixtures for testing SCHISM functionality.
"""
import os
import pytest
from pathlib import Path

import xarray as xr
from rompy.core import DataBlob, DataGrid, TimeRange
from rompy.core.filters import Filter
from rompy.core.types import DatasetCoords
from rompy.core.source import SourceFile, SourceIntake
# Import directly from the new implementation
from rompy.schism.grid import SCHISMGrid
from rompy.schism.vgrid import VGrid as SchismVGrid
from rompy.schism.data import (
    SCHISMDataBoundary,
    SCHISMDataOcean,
    SCHISMDataSflux,
    SCHISMDataTides,
    SfluxAir,
    TidalDataset,
)

# Helper functions imported from test_adapter
from tests.schism.test_adapter import prepare_test_grid, ensure_boundary_data_format, patch_output_file


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
    
    # Prepare the grid using helpers from test_adapter
    grid = prepare_test_grid(grid)
    return grid


@pytest.fixture
def grid3d(test_files_dir):
    """Return a 3D SCHISM grid with vgrid for testing."""
    # Prepare vgrid based on existence
    vgrid_path = test_files_dir / "vgrid.in"
    if vgrid_path.exists():
        vgrid = DataBlob(source=vgrid_path)
    else:
        # Create a basic vertical grid with default values
        vgrid = SchismVGrid(
            ivcor=2,  # LSC2
            nvrt=20,
            h_s=20.0,
            theta_b=0.5,
            theta_f=5.0
        )
    
    # Create the grid with both hgrid and vgrid
    grid = SCHISMGrid(
        hgrid=DataBlob(source=test_files_dir / "hgrid.gr3"),
        vgrid=vgrid,
        drag=1.0,
    )
    
    # Prepare the grid using helpers from test_adapter
    grid = prepare_test_grid(grid)
    return grid


@pytest.fixture
def grid_atmos_source(test_files_dir):
    """Create a source for atmospheric data."""
    return DataGrid(
        source=SourceFile(
            uri=str(test_files_dir / "air_1.nc")
        ),
        coords=DatasetCoords(
            t="time",
            x="lon",
            y="lat"
        ),
        variables=["air_pressure", "air_temperature", "specific_humidity", "eastward_wind", "northward_wind"],
        buffer=0.1,
        filter=Filter(),
        crop_data=True
    )


@pytest.fixture
def hycom_bnd2d(test_files_dir):
    """Create a 2D hydrodynamic boundary source."""
    return DataGrid(
        source=SourceFile(
            uri=str(test_files_dir / "hycom.nc")
        ),
        coords=DatasetCoords(
            t="time",
            x="lon",
            y="lat"
        ),
        variables=["surf_el"],
        buffer=0.1,
        filter=Filter(),
        crop_data=True
    )


@pytest.fixture
def hycom_bnd_temp_3d(test_files_dir):
    """Create a 3D temperature boundary source."""
    return DataGrid(
        source=SourceFile(
            uri=str(test_files_dir / "hycom.nc")
        ),
        coords=DatasetCoords(
            t="time",
            x="lon",
            y="lat",
            z="depth"
        ),
        variables=["water_temp"],
        buffer=0.1,
        filter=Filter(),
        crop_data=True
    )
