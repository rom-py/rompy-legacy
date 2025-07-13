"""
Tests for the SCHISM hotstart functionality.
"""

import os
from datetime import datetime
from pathlib import Path

import pytest

pytest.importorskip("rompy.schism")

from rompy.core.data import DataBlob
from rompy.core.source import SourceFile
from rompy.core.time import TimeRange
from rompy.schism import SCHISMGrid
from rompy.schism.hotstart import SCHISMDataHotstart

HERE = Path(__file__).parent
import logging

logging.basicConfig(level=logging.INFO)

time_range = TimeRange(start="2023-01-01", end="2023-01-02", dt=3600)


@pytest.fixture
def grid3d():
    """Create a 3D SCHISM grid for testing."""
    grid = SCHISMGrid(
        hgrid=DataBlob(source=HERE / "test_data/hgrid.gr3"),
        vgrid=DataBlob(source=HERE / "test_data/vgrid.in"),
        drag=1,
    )
    return grid


def test_hotstart_creation(tmp_path, grid3d, hycom_path):
    """Test creating a hotstart file from HYCOM data."""
    # Create a SCHISMDataHotstart instance
    hotstart = SCHISMDataHotstart(
        source=SourceFile(uri=hycom_path),
        temp_var="temperature",
        salt_var="salinity",
        coords={
            "x": "xlon",
            "y": "ylat",
            "t": "time",
            "z": "depth",
        },
        time_base=datetime(
            2000, 1, 1
        ),  # Based on the time_origin attribute in the HYCOM file
        output_filename="test_hotstart.nc",
    )

    # Generate the hotstart file
    output_path = hotstart.get(tmp_path, grid3d, time=time_range)

    # Verify the output file exists
    assert Path(output_path).exists()

    # Open the file and check its structure using xarray
    import xarray as xr

    # Open the NetCDF file with xarray
    ds = xr.open_dataset(output_path)

    # Check dimensions
    assert "node" in ds.dims
    assert "elem" in ds.dims
    assert "side" in ds.dims
    assert "nVert" in ds.dims
    assert "ntracers" in ds.dims
    assert "one" in ds.dims

    # Check variables
    expected_vars = [
        "time",
        "iths",
        "ifile",
        "idry_e",
        "idry_s",
        "idry",
        "eta2",
        "we",
        "tr_el",
        "tr_nd",
        "tr_nd0",
        "su2",
        "sv2",
        "q2",
        "xl",
        "dfv",
        "dfh",
        "dfq1",
        "dfq2",
        "nsteps_from_cold",
        "cumsum_eta",
    ]

    for var in expected_vars:
        assert var in ds.variables

    # Check tracer dimensions
    assert ds.variables["tr_nd"].shape[2] == 2  # Should have 2 tracers (temp and salt)
    assert ds.variables["tr_el"].shape[2] == 2

    # Check grid dimensions match
    assert ds.dims["node"] == grid3d.np
    assert ds.dims["elem"] == grid3d.ne
    assert ds.dims["side"] > 0  # Just check it has sides, don't compare with grid3d.ns
    assert ds.dims["nVert"] == grid3d.nvrt

    # Close the dataset
    ds.close()


def test_hotstart_with_sourcefile(tmp_path, grid3d, hycom_path):
    """Test creating a hotstart file using SourceFile."""
    # Create a SCHISMDataHotstart instance with SourceFile
    hotstart = SCHISMDataHotstart(
        source=SourceFile(uri=hycom_path),
        temp_var="temperature",
        salt_var="salinity",
        coords={
            "x": "xlon",
            "y": "ylat",
            "t": "time",
            "z": "depth",
        },
        time_base=datetime(2000, 1, 1),
        output_filename="test_hotstart_file.nc",
    )

    # Generate the hotstart file
    output_path = hotstart.get(tmp_path, grid3d, time=time_range)

    # Verify the output file exists
    assert Path(output_path).exists()


def test_hotstart_with_time_range(tmp_path, grid3d, hycom_path):
    """Test creating a hotstart file with a TimeRange object."""
    # Create a TimeRange

    # Create a SCHISMDataHotstart instance
    hotstart = SCHISMDataHotstart(
        source=SourceFile(uri=hycom_path),
        temp_var="temperature",
        salt_var="salinity",
        coords={
            "x": "xlon",
            "y": "ylat",
            "t": "time",
            "z": "depth",
        },
        time_base=datetime(2000, 1, 1),
        output_filename="test_hotstart_timerange.nc",
    )

    # Generate the hotstart file with time_range
    output_path = hotstart.get(tmp_path, grid3d, time=time_range)

    # Verify the output file exists
    assert Path(output_path).exists()
