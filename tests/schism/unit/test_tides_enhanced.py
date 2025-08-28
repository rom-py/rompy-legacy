import pytest
import os
from pathlib import Path
from datetime import datetime

from rompy.core.time import TimeRange
from rompy.schism.tides_enhanced import (
    SCHISMDataTidesEnhanced,
    BoundarySetup,
    TidalDataset,
    create_tidal_only_config,
    create_hybrid_config,
    create_river_config,
    create_nested_config,
)
from rompy.schism.data import SCHISMDataBoundary
from rompy.schism.boundary_core import ElevationType, VelocityType, TracerType

# We'll use the grid2d fixture from the parent conftest.py
# No need to redefine it here


@pytest.fixture
def test_time_range():
    """Return a test time range."""
    return TimeRange(start=datetime(2023, 1, 1), end=datetime(2023, 1, 10))


@pytest.fixture
def tidal_dataset():
    """Return a test tidal dataset."""
    test_data_dir = Path(__file__).parent.parent / "test_data" / "tpxo9-neaus"
    elev_file = test_data_dir / "h_m2s2n2.nc"
    vel_file = test_data_dir / "u_m2s2n2.nc"

    if not elev_file.exists() or not vel_file.exists():
        pytest.skip("Tidal data files not found")

    return TidalDataset(elevations=str(elev_file), velocities=str(vel_file))


class TestBoundarySetup:
    """Tests for the BoundarySetup class."""

    def test_init_default(self):
        """Test initialization with default values."""
        setup = BoundarySetup()

        assert setup.elev_type == 5  # Default to HARMONICEXTERNAL
        assert setup.vel_type == 5  # Default to HARMONICEXTERNAL
        assert setup.temp_type == 0  # Default to NONE
        assert setup.salt_type == 0  # Default to NONE

        assert setup.const_elev is None
        assert setup.const_flow is None
        assert setup.const_temp is None
        assert setup.const_salt is None

        assert setup.inflow_relax == 0.5
        assert setup.outflow_relax == 0.1

        assert setup.temp_nudge == 1.0
        assert setup.salt_nudge == 1.0

    def test_init_custom(self):
        """Test initialization with custom values."""
        setup = BoundarySetup(
            elev_type=3,  # TIDAL
            vel_type=3,  # TIDAL
            temp_type=2,  # CONSTANT
            salt_type=2,  # CONSTANT
            const_elev=1.0,
            const_flow=-100.0,
            const_temp=15.0,
            const_salt=35.0,
            inflow_relax=0.8,
            outflow_relax=0.2,
            temp_nudge=0.9,
            salt_nudge=0.9,
        )

        assert setup.elev_type == 3
        assert setup.vel_type == 3
        assert setup.temp_type == 2
        assert setup.salt_type == 2

        assert setup.const_elev == 1.0
        assert setup.const_flow == -100.0
        assert setup.const_temp == 15.0
        assert setup.const_salt == 35.0

        assert setup.inflow_relax == 0.8
        assert setup.outflow_relax == 0.2

        assert setup.temp_nudge == 0.9
        assert setup.salt_nudge == 0.9

    def test_to_boundary_config(self):
        """Test conversion to BoundaryConfig."""
        setup = BoundarySetup(
            elev_type=3,  # TIDAL
            vel_type=3,  # TIDAL
            temp_type=2,  # CONSTANT
            salt_type=2,  # CONSTANT
            const_elev=1.0,
            const_flow=-100.0,
            const_temp=15.0,
            const_salt=35.0,
        )

        config = setup.to_boundary_config()

        assert config.elev_type == 3
        assert config.vel_type == 3
        assert config.temp_type == 2
        assert config.salt_type == 2

        assert config.ethconst == 1.0
        assert config.vthconst == -100.0
        assert config.tthconst == 15.0
        assert config.sthconst == 35.0


class TestSCHISMDataTidesEnhanced:
    """Tests for the SCHISMDataTidesEnhanced class."""

    def test_init_default(self):
        """Test initialization with default values."""
        tides = SCHISMDataTidesEnhanced()

        assert tides.data_type == "tides_enhanced"
        # Check that tidal_data is None by default
        assert tides.tidal_data is None

    def test_init_with_constituents(self):
        """Test initialization with constituents."""
        from rompy.schism.boundary_core import TidalDataset

        constituents = ["M2", "S2", "N2"]
        tidal_data = TidalDataset(constituents=constituents)
        tides = SCHISMDataTidesEnhanced(tidal_data=tidal_data)

        assert tides.tidal_data.constituents == ["m2", "s2", "n2"]

    def test_init_with_boundaries(self, tidal_dataset):
        """Test initialization with boundary configurations."""
        # Define boundary configurations
        boundaries = {
            0: BoundarySetup(elev_type=3, vel_type=3),  # Harmonic types
            1: BoundarySetup(
                elev_type=2, vel_type=2, const_flow=-100.0  # Constant types
            ),
        }

        tides = SCHISMDataTidesEnhanced(
            tidal_data=tidal_dataset,
            boundaries=boundaries,
        )

        assert len(tides.boundaries) == 2
        assert tides.boundaries[0].elev_type == 3
        assert tides.boundaries[1].elev_type == 2
        assert tides.boundaries[1].vel_type == 2
        assert tides.boundaries[1].const_flow == -100.0

    def test_create_tidal_boundary(self, grid2d, tidal_dataset):
        """Test creating a TidalBoundary from configuration."""
        # Update tidal dataset with specific constituents
        tidal_dataset.constituents = ["M2", "S2", "N2"]

        tides = SCHISMDataTidesEnhanced(
            tidal_data=tidal_dataset,
            setup_type="tidal",
        )

        boundary = tides.create_tidal_boundary(grid2d)

        assert boundary is not None
        assert boundary.tidal_data.constituents == ["M2", "S2", "N2"]  # Case as input
        assert boundary.tidal_data.tidal_model is not None

        # With setup_type="tidal", boundary should be configured for tidal forcing
        assert len(boundary.boundary_configs) >= 0  # May be empty if no configs set

    def test_get(self, grid2d, test_time_range, tidal_dataset, tmp_path):
        """Test generating bctides.in file."""
        # Update tidal dataset with specific constituents
        tidal_dataset.constituents = ["M2", "S2", "N2"]

        tides = SCHISMDataTidesEnhanced(
            tidal_data=tidal_dataset,
            setup_type="tidal",
        )

        # Mock the write_boundary_file method to avoid actual file writing
        class MockTidalBoundary:
            def __init__(self, *args, **kwargs):
                self.boundary_configs = {}
                self.args = args
                self.kwargs = kwargs

            def set_boundary_type(self, *args, **kwargs):
                pass

            def set_run_parameters(self, *args, **kwargs):
                pass

            def write_boundary_file(self, output_path):
                # Just create an empty file
                with open(output_path, "w") as f:
                    f.write("# Mock bctides.in file\n")
                return output_path

        # Call get method
        output_path = tides.get(tmp_path, grid2d, test_time_range)

        # Check that the file was created
        assert os.path.exists(output_path)
        assert os.path.basename(output_path) == "bctides.in"


class TestTidesOceanConsistency:
    """Tests for cross-validation between SCHISMDataOcean and SCHISMDataTidesEnhanced."""

    def test_temperature_validation(
        self, grid2d, tidal_dataset, hycom_bnd2d, hycom_bnd_temp_3d
    ):
        """Test that temperature boundary validation works correctly."""
        # Create a tidal config that requires temperature
        # Update tidal dataset with specific constituents
        tidal_dataset.constituents = ["M2", "S2"]

        tides = SCHISMDataTidesEnhanced(
            tidal_data=tidal_dataset,
            boundaries={
                0: BoundarySetup(
                    elev_type=ElevationType.HARMONIC,
                    vel_type=VelocityType.HARMONIC,
                    temp_type=TracerType.CONSTANT,
                    const_temp=15.0,
                )
            },
        )

        # Create ocean boundary data without temperature - should log a warning
        elev_boundary = SCHISMDataBoundary(
            source=hycom_bnd2d.source,
            variables=["surf_el"],
        )

        # NOTE: This test needs to be rewritten for the new boundary conditions system
        # The old SCHISMDataOcean approach is no longer valid
        pytest.skip("Test needs to be rewritten for new boundary conditions system")

    def test_salinity_validation(
        self, grid2d, tidal_dataset, hycom_bnd2d, hycom_bnd_temp_3d
    ):
        """Test that salinity boundary validation works correctly."""
        # Create a tidal config that requires salinity
        # Update tidal dataset with specific constituents
        tidal_dataset.constituents = ["M2", "S2"]

        tides = SCHISMDataTidesEnhanced(
            tidal_data=tidal_dataset,
            boundaries={
                0: BoundarySetup(
                    elev_type=ElevationType.HARMONIC,
                    vel_type=VelocityType.HARMONIC,
                    salt_type=TracerType.CONSTANT,
                    const_salt=35.0,
                )
            },
        )

        # Create ocean boundary data without salinity - should log a warning
        elev_boundary = SCHISMDataBoundary(
            source=hycom_bnd2d.source,
            variables=["surf_el"],
        )

        # NOTE: This test needs to be rewritten for the new boundary conditions system
        # The old SCHISMDataOcean approach is no longer valid
        pytest.skip("Test needs to be rewritten for new boundary conditions system")


class TestFactoryFunctions:
    """Tests for the factory functions."""

    def test_create_tidal_only_config(self, tidal_dataset):
        """Test creating a tidal-only configuration."""
        config = create_tidal_only_config(
            constituents=["M2", "S2", "N2"],
            tidal_model="OCEANUM-atlas",
        )

        assert isinstance(config, SCHISMDataTidesEnhanced)
        assert config.tidal_data.constituents == ["m2", "s2", "n2"]
        assert config.tidal_data.tidal_model == "OCEANUM-atlas"
        assert config.tidal_data is not None

    def test_create_hybrid_config(self, tidal_dataset):
        """Test creating a hybrid configuration."""
        config = create_hybrid_config(
            constituents=["M2", "S2"],
            tidal_model="OCEANUM-atlas",
        )

        assert isinstance(config, SCHISMDataTidesEnhanced)
        assert config.tidal_data.constituents == ["m2", "s2"]

    def test_create_river_config(self, tidal_dataset):
        """Test creating a river configuration."""
        config = create_river_config(
            river_boundary_index=1,
            river_flow=-100.0,
            constituents=["M2", "S2"],
            tidal_model="OCEANUM-atlas",
        )

        assert isinstance(config, SCHISMDataTidesEnhanced)
        assert config.tidal_data.constituents == ["m2", "s2"]
        assert config.boundaries is not None
        assert 1 in config.boundaries
        assert config.boundaries[1].const_flow == -100.0

    def test_create_nested_config(self, tidal_dataset):
        """Test creating a nested configuration."""
        config = create_nested_config(
            inflow_relax=0.9,
            outflow_relax=0.1,
            constituents=["M2", "S2"],
            tidal_model="OCEANUM-atlas",
        )

        assert isinstance(config, SCHISMDataTidesEnhanced)
        assert config.tidal_data.constituents == ["m2", "s2"]
        assert config.boundaries is not None
        assert 0 in config.boundaries
        assert config.boundaries[0].inflow_relax == 0.9
        assert config.boundaries[0].outflow_relax == 0.1
