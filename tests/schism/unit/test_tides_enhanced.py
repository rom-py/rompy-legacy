import pytest
import os
from pathlib import Path
import numpy as np
from datetime import datetime

from rompy.core.time import TimeRange
from rompy.schism.grid import SCHISMGrid
from rompy.schism.tides_enhanced import (
    SCHISMDataTidesEnhanced,
    BoundarySetup,
    TidalDataset,
    create_tidal_only_config,
    create_hybrid_config,
    create_river_config,
    create_nested_config
)
from rompy.schism.boundary_tides import (
    ElevationType,
    VelocityType,
    TracerType
)


@pytest.fixture
def test_grid():
    """Return a test grid."""
    grid_path = Path(__file__).parent.parent / "hgrid_20kmto60km_rompyschism_testing.gr3"
    if not grid_path.exists():
        pytest.skip("Test grid file not found")
    
    # Create a grid object
    grid = SCHISMGrid(hgrid=str(grid_path))
    return grid


@pytest.fixture
def test_time_range():
    """Return a test time range."""
    return TimeRange(
        start=datetime(2023, 1, 1),
        end=datetime(2023, 1, 10)
    )


@pytest.fixture
def tidal_dataset():
    """Return a test tidal dataset."""
    test_data_dir = Path(__file__).parent.parent / "test_data" / "tpxo9-neaus"
    elev_file = test_data_dir / "h_m2s2n2.nc"
    vel_file = test_data_dir / "u_m2s2n2.nc"
    
    if not elev_file.exists() or not vel_file.exists():
        pytest.skip("Tidal data files not found")
    
    return TidalDataset(
        elevations=str(elev_file),
        velocities=str(vel_file)
    )


class TestBoundarySetup:
    """Tests for the BoundarySetup class."""
    
    def test_init_default(self):
        """Test initialization with default values."""
        setup = BoundarySetup()
        
        assert setup.elev_type == 5  # Default to TIDALSPACETIME
        assert setup.vel_type == 5   # Default to TIDALSPACETIME
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
            vel_type=3,   # TIDAL
            temp_type=2,  # CONSTANT
            salt_type=2,  # CONSTANT
            const_elev=1.0,
            const_flow=-100.0,
            const_temp=15.0,
            const_salt=35.0,
            inflow_relax=0.8,
            outflow_relax=0.2,
            temp_nudge=0.9,
            salt_nudge=0.9
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
            vel_type=3,   # TIDAL
            temp_type=2,  # CONSTANT
            salt_type=2,  # CONSTANT
            const_elev=1.0,
            const_flow=-100.0,
            const_temp=15.0,
            const_salt=35.0
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
        
        assert tides.data_type == "tides"
        assert tides.tidal_database == "tpxo"
        assert tides.ntip == 0
        assert tides.cutoff_depth == 50.0
    
    def test_init_with_constituents(self):
        """Test initialization with constituents."""
        constituents = ["M2", "S2", "N2"]
        tides = SCHISMDataTidesEnhanced(constituents=constituents)
        
        assert tides.constituents == constituents
    
    def test_init_with_boundaries(self, tidal_dataset):
        """Test initialization with boundary configurations."""
        # Define boundary configurations
        boundaries = {
            0: BoundarySetup(
                elev_type=int(ElevationType.TIDAL),
                vel_type=int(VelocityType.TIDAL)
            ),
            1: BoundarySetup(
                elev_type=int(ElevationType.NONE),
                vel_type=int(VelocityType.CONSTANT),
                const_flow=-100.0
            )
        }
        
        tides = SCHISMDataTidesEnhanced(
            constituents=["M2", "S2", "N2"],
            tidal_data=tidal_dataset,
            boundaries=boundaries
        )
        
        assert tides.boundaries is not None
        assert len(tides.boundaries) == 2
        assert tides.boundaries[0].elev_type == int(ElevationType.TIDAL)
        assert tides.boundaries[1].vel_type == int(VelocityType.CONSTANT)
        assert tides.boundaries[1].const_flow == -100.0
    
    def test_create_tidal_boundary(self, test_grid, tidal_dataset):
        """Test creating a TidalBoundary from configuration."""
        tides = SCHISMDataTidesEnhanced(
            constituents=["M2", "S2", "N2"],
            tidal_data=tidal_dataset,
            setup_type="tidal"
        )
        
        boundary = tides.create_tidal_boundary(test_grid)
        
        assert boundary is not None
        assert boundary.constituents == ["M2", "S2", "N2"]
        assert boundary.tidal_database == "tpxo"
        
        # With setup_type="tidal", all boundaries should be configured as tidal
        for i in range(test_grid.pylibs_hgrid.nob):
            config = boundary.boundary_configs.get(i)
            if config:
                assert config.elev_type == ElevationType.TIDAL
                assert config.vel_type == VelocityType.TIDAL
    
    def test_get(self, test_grid, test_time_range, tidal_dataset, tmp_path):
        """Test generating bctides.in file."""
        tides = SCHISMDataTidesEnhanced(
            constituents=["M2", "S2", "N2"],
            tidal_data=tidal_dataset,
            setup_type="tidal"
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
        
        # Replace the create_tidal_boundary method with our mock
        tides.create_tidal_boundary = lambda grid: MockTidalBoundary()
        
        # Call get method
        output_path = tides.get(tmp_path, test_grid, test_time_range)
        
        # Check that the file was created
        assert os.path.exists(output_path)
        assert os.path.basename(output_path) == "bctides.in"


class TestFactoryFunctions:
    """Tests for the factory functions."""
    
    def test_create_tidal_only_config(self, tidal_dataset):
            """Test creating a tidal-only configuration."""
            config = create_tidal_only_config(
                constituents=["M2", "S2", "N2"],
                tidal_database="tpxo",
                tidal_elevations=tidal_dataset.elevations,
                tidal_velocities=tidal_dataset.velocities,
                ntip=1
            )
        
            assert isinstance(config, SCHISMDataTidesEnhanced)
            assert config.constituents == ["M2", "S2", "N2"]
            assert config.tidal_database == "tpxo"
            assert config.ntip == 1
            assert hasattr(config, "setup_type")
            assert config.tidal_data is not None
    
    def test_create_hybrid_config(self, tidal_dataset):
        """Test creating a hybrid configuration."""
        config = create_hybrid_config(
            constituents=["M2", "S2", "N2"],
            tidal_database="tpxo",
            tidal_elevations=tidal_dataset.elevations,
            tidal_velocities=tidal_dataset.velocities
        )
        
        assert isinstance(config, SCHISMDataTidesEnhanced)
        assert config.constituents == ["M2", "S2", "N2"]
        assert hasattr(config, "setup_type")
    
    def test_create_river_config(self, tidal_dataset):
        """Test creating a river configuration."""
        river_flow = -500.0
        config = create_river_config(
            river_boundary_index=0,
            river_flow=river_flow,
            other_boundaries="tidal",
            constituents=["M2", "S2", "N2"],
            tidal_database="tpxo",
            tidal_elevations=tidal_dataset.elevations,
            tidal_velocities=tidal_dataset.velocities
        )
        
        assert isinstance(config, SCHISMDataTidesEnhanced)
        assert config.constituents == ["M2", "S2", "N2"]
        assert config.boundaries is not None
        assert 0 in config.boundaries
        assert config.boundaries[0].const_flow == river_flow
    
    def test_create_nested_config(self, tidal_dataset):
        """Test creating a nested configuration."""
        config = create_nested_config(
            with_tides=True,
            inflow_relax=0.9,
            outflow_relax=0.8,
            constituents=["M2", "S2", "N2"],
            tidal_database="tpxo",
            tidal_elevations=tidal_dataset.elevations,
            tidal_velocities=tidal_dataset.velocities
        )
        
        assert isinstance(config, SCHISMDataTidesEnhanced)
        assert config.constituents == ["M2", "S2", "N2"]
        assert config.boundaries is not None
        assert 0 in config.boundaries
        assert config.boundaries[0].inflow_relax == 0.9
        assert config.boundaries[0].outflow_relax == 0.8