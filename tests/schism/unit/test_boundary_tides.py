from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from rompy.schism.bctides import Bctides
from rompy.schism.boundary_core import TidalBoundary  # Backward compatibility alias
from rompy.schism.boundary_core import (
    BoundaryConfig,
    ElevationType,
    TracerType,
    VelocityType,
    create_hybrid_boundary,
    create_nested_boundary,
    create_river_boundary,
    create_tidal_boundary,
)


class TestBoundaryConfig:
    """Tests for the BoundaryConfig class."""

    def test_init_default(self):
        """Test initialization with default values."""
        config = BoundaryConfig()

        assert config.elev_type == ElevationType.NONE
        assert config.vel_type == VelocityType.NONE
        assert config.temp_type == TracerType.NONE
        assert config.salt_type == TracerType.NONE

        assert config.ethconst is None
        assert config.vthconst is None
        assert config.tthconst is None
        assert config.sthconst is None

        assert config.inflow_relax == 0.5
        assert config.outflow_relax == 0.1

        assert config.tobc == 1.0
        assert config.sobc == 1.0

    def test_init_custom(self):
        """Test initialization with custom values."""
        config = BoundaryConfig(
            elev_type=ElevationType.HARMONIC,
            vel_type=VelocityType.HARMONIC,
            temp_type=TracerType.CONSTANT,
            salt_type=TracerType.CONSTANT,
            ethconst=1.0,
            vthconst=-100.0,
            tthconst=15.0,
            sthconst=35.0,
            inflow_relax=0.8,
            outflow_relax=0.2,
        )

        assert config.elev_type == ElevationType.HARMONIC
        assert config.vel_type == VelocityType.HARMONIC
        assert config.temp_type == TracerType.CONSTANT
        assert config.salt_type == TracerType.CONSTANT

        assert config.ethconst == 1.0
        assert config.vthconst == -100.0
        assert config.tthconst == 15.0
        assert config.sthconst == 35.0

        assert config.inflow_relax == 0.8
        assert config.outflow_relax == 0.2

        assert config.tobc == 1.0
        assert config.sobc == 1.0

    def test_flather_validation(self):
        """Test validation for Flather boundary conditions."""
        config = BoundaryConfig(
            elev_type=ElevationType.NONE, vel_type=VelocityType.FLATHER
        )

        # Set eta_mean and vn_mean for Flather boundary
        num_nodes = 5  # Just for testing
        config.eta_mean = np.ones(num_nodes) * 0.1
        config.vn_mean = np.ones(num_nodes) * 0.05

        # Validate that eta_mean and vn_mean are properly set
        assert isinstance(config.eta_mean, np.ndarray)
        assert isinstance(config.vn_mean, np.ndarray)
        assert len(config.eta_mean) == num_nodes
        assert len(config.vn_mean) == num_nodes


@pytest.fixture
def sample_grid_path():
    """Return path to a test grid file."""
    grid_path = Path(__file__).parent.parent / "hgrid_20kmto60km_schism_testing.gr3"
    if not grid_path.exists():
        pytest.skip("Test grid file not found")
    return str(grid_path)


@pytest.fixture
def sample_tidal_files():
    """Return paths to tidal data files."""
    test_data_dir = Path(__file__).parent.parent / "test_data" / "tpxo9-neaus"
    elev_file = test_data_dir / "h_m2s2n2.nc"
    vel_file = test_data_dir / "u_m2s2n2.nc"

    if not elev_file.exists() or not vel_file.exists():
        pytest.skip("Tidal data files not found")

    return {"elevations": str(elev_file), "velocities": str(vel_file)}


class TestTidalBoundary:
    """Tests for the TidalBoundary class."""

    def test_init(self, sample_grid_path):
        """Test initialization with a grid file."""
        boundary = TidalBoundary(
            grid_path=sample_grid_path, constituents=["M2", "S2", "N2"]
        )

        assert boundary.constituents == ["M2", "S2", "N2"]
        assert boundary.tidal_database == "tpxo"
        assert boundary.grid is not None

    def test_set_boundary_config(self, sample_grid_path):
        """Test setting boundary configuration."""
        boundary = TidalBoundary(grid_path=sample_grid_path)

        config = BoundaryConfig(
            elev_type=ElevationType.HARMONIC, vel_type=VelocityType.HARMONIC
        )

        boundary.set_boundary_config(0, config)
        assert 0 in boundary.boundary_configs
        assert boundary.boundary_configs[0].elev_type == ElevationType.HARMONIC
        assert boundary.boundary_configs[0].vel_type == VelocityType.HARMONIC

    def test_set_boundary_type(self, sample_grid_path):
        """Test setting boundary type."""
        boundary = TidalBoundary(grid_path=sample_grid_path)

        boundary.set_boundary_type(
            0,
            elev_type=ElevationType.HARMONIC,
            vel_type=VelocityType.HARMONIC,
            temp_type=TracerType.NONE,
            salt_type=TracerType.NONE,
        )

        assert 0 in boundary.boundary_configs
        assert boundary.boundary_configs[0].elev_type == ElevationType.HARMONIC
        assert boundary.boundary_configs[0].vel_type == VelocityType.HARMONIC
        assert boundary.boundary_configs[0].temp_type == TracerType.NONE
        assert boundary.boundary_configs[0].salt_type == TracerType.NONE

    def test_get_flags_list(self, sample_grid_path):
        """Test getting flags list from configurations."""
        boundary = TidalBoundary(grid_path=sample_grid_path)

        # Add configurations for multiple boundaries
        boundary.set_boundary_type(0, ElevationType.HARMONIC, VelocityType.HARMONIC)
        boundary.set_boundary_type(
            1,
            ElevationType.CONSTANT,
            VelocityType.CONSTANT,
            vthconst=-100.0,
            ethconst=1.0,
        )
        boundary.set_boundary_type(3, ElevationType.NONE, VelocityType.FLATHER)

        flags = boundary.get_flags_list()

        assert len(flags) == 4  # Should include boundary indices 0, 1, 2, 3
        assert flags[0] == [
            int(ElevationType.HARMONIC),
            int(VelocityType.HARMONIC),
            0,
            0,
        ]
        assert flags[1] == [
            int(ElevationType.CONSTANT),
            int(VelocityType.CONSTANT),
            0,
            0,
        ]
        assert flags[2] == [0, 0, 0, 0]  # Default for missing index 2
        assert flags[3] == [int(ElevationType.NONE), int(VelocityType.FLATHER), 0, 0]

    def test_get_constant_values(self, sample_grid_path):
        """Test getting constant values from configurations."""
        boundary = TidalBoundary(grid_path=sample_grid_path)

        # Add configuration with constant values
        boundary.set_boundary_type(
            0,
            ElevationType.CONSTANT,
            VelocityType.CONSTANT,
            ethconst=1.0,
            vthconst=-100.0,
        )

        constants = boundary.get_constant_values()

        assert "ethconst" in constants
        assert "vthconst" in constants
        assert constants["ethconst"][0] == 1.0
        assert constants["vthconst"][0] == -100.0

    def test_create_bctides(self, sample_grid_path, sample_tidal_files, monkeypatch):
        """Test creating a Bctides object."""
        # Skip actual file operations in Bctides
        monkeypatch.setattr(Bctides, "_get_tidal_factors", lambda self: None)

        boundary = TidalBoundary(
            grid_path=sample_grid_path,
            constituents=["M2", "S2", "N2"],
            tidal_elevations=sample_tidal_files["elevations"],
            tidal_velocities=sample_tidal_files["velocities"],
        )

        boundary.set_boundary_type(0, ElevationType.HARMONIC, VelocityType.HARMONIC)

        # Set run parameters
        boundary.set_run_parameters(datetime(2023, 1, 1), 10.0)

        # Create Bctides object
        bctides = boundary.create_bctides()

        assert isinstance(bctides, Bctides)
        assert bctides._start_time == datetime(2023, 1, 1)
        assert bctides._rnday == 10.0
        assert bctides.tnames == ["M2", "S2", "N2"]


class TestFactoryFunctions:
    """Tests for the factory functions."""

    def test_create_tidal_boundary(self, sample_grid_path):
        """Test creating a tidal-only boundary."""
        boundary = create_tidal_boundary(
            grid_path=sample_grid_path, constituents=["M2", "S2", "N2"]
        )

        assert isinstance(boundary, TidalBoundary)
        assert boundary.constituents == ["M2", "S2", "N2"]

        # Check default configuration
        configs = boundary.boundary_configs
        assert 0 in configs
        assert configs[0].elev_type == ElevationType.HARMONIC
        assert configs[0].vel_type == VelocityType.HARMONIC

    def test_create_hybrid_boundary(self, sample_grid_path):
        """Test creating a hybrid boundary."""
        boundary = create_hybrid_boundary(
            grid_path=sample_grid_path, constituents=["M2", "S2", "N2"]
        )

        assert isinstance(boundary, TidalBoundary)

        # Check default configuration
        configs = boundary.boundary_configs
        assert 0 in configs
        assert configs[0].elev_type == ElevationType.HARMONICEXTERNAL
        assert configs[0].vel_type == VelocityType.HARMONICEXTERNAL

    def test_create_river_boundary(self, sample_grid_path):
        """Test creating a river boundary."""
        river_flow = -500.0
        boundary = create_river_boundary(
            grid_path=sample_grid_path, river_flow=river_flow
        )

        assert isinstance(boundary, TidalBoundary)

        # Check river configuration
        configs = boundary.boundary_configs
        assert 0 in configs
        assert configs[0].elev_type == ElevationType.NONE
        assert configs[0].vel_type == VelocityType.CONSTANT
        assert configs[0].vthconst == river_flow

    def test_create_nested_boundary(self, sample_grid_path):
        """Test creating a nested boundary."""
        boundary = create_nested_boundary(
            grid_path=sample_grid_path,
            with_tides=True,
            inflow_relax=0.9,
            outflow_relax=0.8,
            constituents=["M2", "S2", "N2"],
        )

        assert isinstance(boundary, TidalBoundary)

        # Check nested configuration
        configs = boundary.boundary_configs
        assert 0 in configs
        assert configs[0].elev_type == ElevationType.HARMONICEXTERNAL
        assert configs[0].vel_type == VelocityType.RELAXED
        assert configs[0].temp_type == TracerType.EXTERNAL
        assert configs[0].salt_type == TracerType.EXTERNAL
        assert configs[0].inflow_relax == 0.9
        assert configs[0].outflow_relax == 0.8
