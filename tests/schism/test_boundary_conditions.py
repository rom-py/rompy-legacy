"""
Test module for the unified boundary conditions in SCHISM.

This module tests the functionality of the new boundary conditions implementation,
including the BoundarySetupWithSource and SCHISMDataBoundaryConditions classes,
as well as the factory functions for common configurations.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from rompy.core.data import DataBlob
from rompy.core.source import SourceFile
from rompy.core.time import TimeRange
from rompy.schism.boundary_conditions import (
    create_hybrid_boundary_config,
    create_nested_boundary_config,
    create_river_boundary_config,
    create_tidal_only_boundary_config,
)
from rompy.schism.boundary_core import (
    ElevationType,
    TidalDataset,
    TracerType,
    VelocityType,
)
from rompy.schism.data import (
    BoundarySetupWithSource,
    SCHISMDataBoundary,
    SCHISMDataBoundaryConditions,
)
from rompy.schism.grid import SCHISMGrid
from rompy.schism.tides_enhanced import TidalDataset


@pytest.fixture
def time_range():
    """Create a time range for testing."""
    return TimeRange(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 5),
    )


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary directory for test outputs."""
    output_dir = tmp_path / "boundary_conditions_test"
    output_dir.mkdir(exist_ok=True)
    return output_dir


class TestBoundarySetupWithSource:
    """Tests for the BoundarySetupWithSource class."""

    def test_basic_initialization(self):
        """Test basic initialization with different boundary types."""
        # Tidal boundary
        tidal_boundary = BoundarySetupWithSource(
            elev_type=ElevationType.HARMONIC,
            vel_type=VelocityType.HARMONIC,
            temp_type=TracerType.NONE,
            salt_type=TracerType.NONE,
        )

        assert tidal_boundary.elev_type == ElevationType.HARMONIC
        assert tidal_boundary.vel_type == VelocityType.HARMONIC
        assert tidal_boundary.temp_type == TracerType.NONE
        assert tidal_boundary.salt_type == TracerType.NONE

        # River boundary
        river_boundary = BoundarySetupWithSource(
            elev_type=ElevationType.NONE,
            vel_type=VelocityType.CONSTANT,
            temp_type=TracerType.CONSTANT,
            salt_type=TracerType.CONSTANT,
            const_flow=-100.0,
            const_temp=15.0,
            const_salt=0.5,
        )

        assert river_boundary.elev_type == ElevationType.NONE
        assert river_boundary.vel_type == VelocityType.CONSTANT
        assert river_boundary.temp_type == TracerType.CONSTANT
        assert river_boundary.salt_type == TracerType.CONSTANT
        assert river_boundary.const_flow == -100.0
        assert river_boundary.const_temp == 15.0
        assert river_boundary.const_salt == 0.5

    def test_with_data_sources(self):
        """Test initialization with data sources."""
        # Create mock data sources
        elev_source = DataBlob(source="path/to/elev2D.th.nc")
        vel_source = DataBlob(source="path/to/uv3D.th.nc")

        # Hybrid boundary with data sources
        hybrid_boundary = BoundarySetupWithSource(
            elev_type=ElevationType.HARMONICEXTERNAL,
            vel_type=VelocityType.HARMONICEXTERNAL,
            temp_type=TracerType.NONE,
            salt_type=TracerType.NONE,
            elev_source=elev_source,
            vel_source=vel_source,
        )

        assert hybrid_boundary.elev_type == ElevationType.HARMONICEXTERNAL
        assert hybrid_boundary.vel_type == VelocityType.HARMONICEXTERNAL
        assert hybrid_boundary.elev_source == elev_source
        assert hybrid_boundary.vel_source == vel_source

    def test_validation_warnings(self, caplog):
        """Test that warnings are logged for missing data sources."""
        # Create a boundary that should have data sources but doesn't
        boundary = BoundarySetupWithSource(
            elev_type=ElevationType.EXTERNAL,
            vel_type=VelocityType.RELAXED,
            temp_type=TracerType.EXTERNAL,
            salt_type=TracerType.EXTERNAL,
            # Missing data sources
        )

        # Check that warnings were logged
        assert "elev_source should be provided for EXTERNAL" in caplog.text
        assert "vel_source should be provided for" in caplog.text
        assert "temp_source should be provided for EXTERNAL" in caplog.text
        assert "salt_source should be provided for EXTERNAL" in caplog.text

    def test_to_boundary_config(self):
        """Test conversion to boundary config."""
        # Create a boundary setup
        boundary = BoundarySetupWithSource(
            elev_type=ElevationType.HARMONIC,
            vel_type=VelocityType.HARMONIC,
            temp_type=TracerType.NONE,
            salt_type=TracerType.NONE,
        )

        # Convert to boundary config
        config = boundary.to_boundary_config()

        # Check the config
        assert config.elev_type == ElevationType.HARMONIC
        assert config.vel_type == VelocityType.HARMONIC
        assert config.temp_type == TracerType.NONE
        assert config.salt_type == TracerType.NONE


class TestSCHISMDataBoundaryConditions:
    """Tests for the SCHISMDataBoundaryConditions class."""

    def test_basic_initialization(self, tidal_dataset):
        """Test basic initialization."""
        # Create a basic boundary conditions object
        bc = SCHISMDataBoundaryConditions(tidal_data=tidal_dataset)

        assert bc.tidal_data.constituents == ["m2", "s2"]
        assert bc.tidal_data.tidal_model == "OCEANUM-atlas"
        assert bc.boundaries == {}

    def test_with_boundaries(self, tidal_dataset):
        """Test initialization with boundary configurations."""
        # Create boundary setups that don't require tidal data
        constant_boundary = BoundarySetupWithSource(
            elev_type=ElevationType.CONSTANT,
            vel_type=VelocityType.CONSTANT,
            const_elev=1.0,
            const_flow=-50.0,
        )

        river_boundary = BoundarySetupWithSource(
            elev_type=ElevationType.NONE,
            vel_type=VelocityType.CONSTANT,
            const_flow=-100.0,
        )

        # Create boundary conditions with multiple boundaries
        bc = SCHISMDataBoundaryConditions(
            tidal_data=tidal_dataset,
            boundaries={0: constant_boundary, 1: river_boundary},
        )

        assert len(bc.boundaries) == 2
        assert bc.boundaries[0].elev_type == ElevationType.CONSTANT
        assert bc.boundaries[1].vel_type == VelocityType.CONSTANT
        assert bc.boundaries[1].const_flow == -100.0

    def test_with_setup_type(self, tidal_dataset):
        """Test initialization with setup type."""
        # Test that tidal setup type fails without tidal data
        with pytest.raises(
            ValueError,
            match="Tidal data is required for HARMONIC or HARMONICEXTERNAL boundary types",
        ):
            bc_tidal = SCHISMDataBoundaryConditions(
                setup_type="tidal"
                # Missing tidal_data should cause validation error
            )

        # Test that tidal setup type works with tidal data
        bc_tidal_valid = SCHISMDataBoundaryConditions(
            tidal_data=tidal_dataset, setup_type="tidal"
        )
        assert bc_tidal_valid.setup_type == "tidal"

        # Test river setup type (should work without tidal data)
        bc_river = SCHISMDataBoundaryConditions(setup_type="river")

        assert bc_river.setup_type == "river"

    def test_validate_tidal_data(self):
        """Test validation of tidal data."""
        # Test that configurations requiring tidal data fail without it
        with pytest.raises(
            ValueError,
            match="Tidal data is required for HARMONIC or HARMONICEXTERNAL boundary types",
        ):
            SCHISMDataBoundaryConditions(
                setup_type="tidal",
                # Missing tidal_data
                boundaries={
                    0: BoundarySetupWithSource(
                        elev_type=ElevationType.HARMONIC, vel_type=VelocityType.HARMONIC
                    )
                },
            )

    def test_tidal_data(self, tidal_dataset):
        """Test with actual tidal dataset."""
        # Create with tidal dataset
        bc = SCHISMDataBoundaryConditions(tidal_data=tidal_dataset, setup_type="tidal")

        assert bc.tidal_data == tidal_dataset

    def test_write_bctides(self, grid2d, time_range, temp_output_dir, tidal_dataset):
        """Test writing bctides.in file."""
        # Create a simple tidal boundary configuration with real data
        bc = SCHISMDataBoundaryConditions(
            tidal_data=tidal_dataset,
            setup_type="tidal",
            boundaries={
                0: BoundarySetupWithSource(
                    elev_type=ElevationType.HARMONIC, vel_type=VelocityType.HARMONIC
                )
            },
        )

        # Get the boundary configuration
        result = bc.get(temp_output_dir, grid2d, time_range)

        # Check that bctides.in path is returned and file exists
        assert "bctides" in result
        bctides_path = Path(result["bctides"])
        assert bctides_path.exists()

        # Verify basic content of the file
        with open(bctides_path, "r") as f:
            content = f.read()
            content_lower = content.lower()
            assert "m2" in content_lower or "s2" in content_lower


@pytest.mark.parametrize(
    "function_name,expected_type",
    [
        ("create_tidal_only_boundary_config", "tidal"),
        ("create_hybrid_boundary_config", "hybrid"),
        ("create_river_boundary_config", "river"),
        ("create_nested_boundary_config", "nested"),
    ],
)
def test_factory_functions_basic(function_name, expected_type):
    """Test basic functionality of factory functions."""
    # Get the factory function
    import rompy.schism.boundary_conditions as bc_module

    factory_func = getattr(bc_module, function_name)

    # All functions should work with minimal arguments during construction
    # Validation happens later during .get() call when data is actually used
    boundary_config = factory_func()

    # Check the result
    assert isinstance(boundary_config, SCHISMDataBoundaryConditions)
    assert boundary_config.setup_type == expected_type


def test_tidal_only_factory(tidal_data_files):
    """Test the tidal-only factory function with real data."""
    # Create configuration with tidal data
    bc = create_tidal_only_boundary_config(
        tidal_database=tidal_data_files,
        tidal_model="OCEANUM-atlas",
        constituents=["M2", "S2", "N2"],
    )

    # Check the configuration
    assert bc.setup_type == "tidal"
    # constituents are normalized to lowercase internally
    assert bc.tidal_data.constituents == ["m2", "s2", "n2"]
    assert bc.tidal_data is not None
    assert bc.tidal_data.tidal_database == tidal_data_files
    assert bc.tidal_data.tidal_model == "OCEANUM-atlas"


def test_hybrid_factory(
    tidal_data_files, grid2d, time_range, temp_output_dir, hycom_bnd_elev
):
    """Test the hybrid factory function with real data."""
    # Skip if the tidal data files don't exist
    # if not os.path.exists(tidal_data_files["elevation"]) or not os.path.exists(
    #     tidal_data_files["velocity"]
    # ):
    #     pytest.skip("Tidal data files not available")

    # # Create a simple DataBlob for sources
    # elev_source = DataBlob(source=tidal_data_files["elevation"])
    # vel_source = DataBlob(source=tidal_data_files["velocity"])
    elev_source = hycom_bnd_elev

    # Create configuration with data sources
    bc = create_hybrid_boundary_config(
        tidal_database=tidal_data_files,
        tidal_model="OCEANUM-atlas",
        constituents=["M2", "S2", "N2"],
        elev_source=elev_source,
    )

    # Check the configuration
    assert bc.setup_type == "hybrid"
    # constituents are normalized to lowercase internally
    assert bc.tidal_data.constituents == ["m2", "s2", "n2"]
    assert bc.tidal_data is not None
    assert bc.tidal_data.tidal_database == tidal_data_files
    assert bc.tidal_data.tidal_model == "OCEANUM-atlas"
    assert len(bc.boundaries) == 1
    assert bc.boundaries[0].elev_type == ElevationType.HARMONICEXTERNAL
    assert bc.boundaries[0].elev_source == elev_source

    # Process the data to verify it works with real files
    result = bc.get(temp_output_dir, grid2d, time_range)
    assert "bctides" in result
    assert os.path.exists(result["bctides"])


def test_river_factory(tidal_data_files):
    """Test the river factory function."""
    # Create configuration with river boundary
    bc = create_river_boundary_config(
        river_boundary_index=1,
        river_flow=-100.0,
        other_boundaries="tidal",
        tidal_database=tidal_data_files,
        tidal_model="OCEANUM-atlas",
        constituents=["M2", "S2", "N2"],
    )

    # Check the configuration
    assert bc.setup_type == "river"
    assert len(bc.boundaries) >= 1
    assert 1 in bc.boundaries
    assert bc.boundaries[1].vel_type == VelocityType.CONSTANT
    assert bc.boundaries[1].const_flow == -100.0


def test_nested_factory(
    tidal_data_files, grid2d, time_range, temp_output_dir, hycom_bnd_elev, hycom_bnd_vel
):
    """Test the nested factory function with real data."""
    # # Skip if the tidal data files don't exist
    # if not os.path.exists(tidal_data_files["elevation"]) or not os.path.exists(
    #     tidal_data_files["velocity"]
    # ):
    #     pytest.skip("Tidal data files not available")

    # Create simple DataBlobs for sources
    elev_source = hycom_bnd_elev
    vel_source = hycom_bnd_vel

    # Create configuration with nested boundary
    bc = create_nested_boundary_config(
        with_tides=True,
        inflow_relax=0.9,
        outflow_relax=0.1,
        tidal_database=tidal_data_files,
        tidal_model="OCEANUM-atlas",
        constituents=["M2", "S2", "N2"],
        elev_source=elev_source,
        vel_source=vel_source,
    )

    # Check the configuration
    assert bc.setup_type == "nested"
    assert len(bc.boundaries) == 1
    assert bc.boundaries[0].vel_type == VelocityType.RELAXED
    assert bc.boundaries[0].inflow_relax == 0.9
    assert bc.boundaries[0].outflow_relax == 0.1
    assert bc.boundaries[0].elev_source == elev_source

    # Process the data to verify it works with real files
    result = bc.get(temp_output_dir, grid2d, time_range)
    assert "bctides" in result
    assert os.path.exists(result["bctides"])


def test_integration_with_schism_data(
    grid2d, time_range, temp_output_dir, tidal_dataset
):
    """Test integration with SCHISMData."""
    from rompy.schism import SCHISMData

    # Create a boundary configuration with real tidal data
    bc = create_tidal_only_boundary_config(
        constituents=tidal_dataset.constituents,
        tidal_database=tidal_dataset.tidal_database,
        tidal_model=tidal_dataset.tidal_model,
    )

    # Create a SCHISMData object with the boundary configuration
    schism_data = SCHISMData(boundary_conditions=bc)

    # Process the data
    result = schism_data.get(temp_output_dir, grid2d, time_range)

    # Check that the processing was successful
    assert result is not None
    assert "boundary_conditions" in result

    # Verify the boundary conditions file was created
    bctides_path = Path(result["boundary_conditions"]["bctides"])
    assert bctides_path.exists()

    # Verify basic content of the file
    with open(bctides_path, "r") as f:
        content = f.read()
        assert len(content) > 0
