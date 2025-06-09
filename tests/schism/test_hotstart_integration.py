"""
Tests for integrated hotstart functionality in SCHISM boundary conditions.

This module tests the new integrated hotstart configuration that allows
hotstart file generation using the same data sources as boundary conditions.
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime

from rompy.core.source import SourceFile
from rompy.core.time import TimeRange
from rompy.core.types import DatasetCoords
from rompy.schism.data import (
    HotstartConfig,
    SCHISMDataBoundaryConditions,
    BoundarySetupWithSource,
    SCHISMDataBoundary,
)
from rompy.schism.tides_enhanced import TidalDataset


class TestHotstartConfig:
    """Test the HotstartConfig class."""

    def test_hotstart_config_defaults(self):
        """Test HotstartConfig with default values."""
        config = HotstartConfig()
        
        assert config.enabled is False
        assert config.temp_var == "temperature"
        assert config.salt_var == "salinity"
        assert config.time_offset == 0.0
        assert config.output_filename == "hotstart.nc"

    def test_hotstart_config_custom_values(self):
        """Test HotstartConfig with custom values."""
        config = HotstartConfig(
            enabled=True,
            temp_var="water_temp",
            salt_var="sal",
            time_offset=1.5,
            output_filename="custom_hotstart.nc"
        )
        
        assert config.enabled is True
        assert config.temp_var == "water_temp"
        assert config.salt_var == "sal"
        assert config.time_offset == 1.5
        assert config.output_filename == "custom_hotstart.nc"

    def test_hotstart_config_serialization(self):
        """Test that HotstartConfig can be serialized and deserialized."""
        original_config = HotstartConfig(
            enabled=True,
            temp_var="temperature",
            salt_var="salinity",
            time_offset=0.5,
            output_filename="test.nc"
        )
        
        # Serialize to dict
        config_dict = original_config.model_dump()
        
        # Deserialize from dict
        restored_config = HotstartConfig(**config_dict)
        
        assert restored_config.enabled == original_config.enabled
        assert restored_config.temp_var == original_config.temp_var
        assert restored_config.salt_var == original_config.salt_var
        assert restored_config.time_offset == original_config.time_offset
        assert restored_config.output_filename == original_config.output_filename


class TestBoundaryConditionsHotstartIntegration:
    """Test hotstart integration in SCHISMDataBoundaryConditions."""

    @pytest.fixture
    def time_range(self):
        """Create a test time range."""
        return TimeRange(start="2023-01-01", end="2023-01-01T12", dt=3600)

    @pytest.fixture
    def hycom_coords(self):
        """Create coordinate mapping for HYCOM data."""
        return DatasetCoords(
            t="time",
            x="xlon", 
            y="ylat",
            z="depth"
        )

    @pytest.fixture
    def hycom_source(self, test_files_dir):
        """Create a SourceFile for HYCOM data."""
        return SourceFile(uri=str(test_files_dir / "hycom.nc"))

    @pytest.fixture
    def tidal_dataset(self, test_files_dir):
        """Create a real tidal dataset."""
        tpxo_dir = test_files_dir / "tpxo9-neaus"
        return TidalDataset(
            elevations=str(tpxo_dir / "h_m2s2n2.nc"),
            velocities=str(tpxo_dir / "u_m2s2n2.nc")
        )

    @pytest.fixture
    def boundary_setup_with_sources(self, hycom_source, hycom_coords):
        """Create a boundary setup with temperature and salinity sources."""
        temp_source = SCHISMDataBoundary(
            source=hycom_source,
            variables=["temperature"],
            coords=hycom_coords
        )
        salt_source = SCHISMDataBoundary(
            source=hycom_source,
            variables=["salinity"],
            coords=hycom_coords
        )
        
        return BoundarySetupWithSource(
            elev_type=5,  # TIDALSPACETIME
            vel_type=4,   # SPACETIME
            temp_type=4,  # SPACETIME
            salt_type=4,  # SPACETIME
            temp_source=temp_source,
            salt_source=salt_source
        )

    @pytest.fixture
    def boundary_setup_no_sources(self):
        """Create a boundary setup without temperature and salinity sources."""
        return BoundarySetupWithSource(
            elev_type=1,  # Tidal only
            vel_type=1,
            temp_type=1,
            salt_type=1
        )

    def test_boundary_conditions_without_hotstart(self, boundary_setup_with_sources, tidal_dataset):
        """Test boundary conditions without hotstart configuration."""
        bc = SCHISMDataBoundaryConditions(
            tidal_data=tidal_dataset,
            setup_type="hybrid",
            boundaries={0: boundary_setup_with_sources}
        )
        
        assert bc.hotstart_config is None

    def test_boundary_conditions_with_disabled_hotstart(self, boundary_setup_with_sources, tidal_dataset):
        """Test boundary conditions with disabled hotstart."""
        hotstart_config = HotstartConfig(enabled=False)
        
        bc = SCHISMDataBoundaryConditions(
            tidal_data=tidal_dataset,
            setup_type="hybrid",
            boundaries={0: boundary_setup_with_sources},
            hotstart_config=hotstart_config
        )
        
        assert bc.hotstart_config is not None
        assert bc.hotstart_config.enabled is False

    def test_boundary_conditions_with_enabled_hotstart(self, boundary_setup_with_sources, tidal_dataset):
        """Test boundary conditions with enabled hotstart."""
        hotstart_config = HotstartConfig(
            enabled=True,
            temp_var="temperature",
            salt_var="salinity",
            output_filename="test_hotstart.nc"
        )
        
        bc = SCHISMDataBoundaryConditions(
            tidal_data=tidal_dataset,
            setup_type="hybrid",
            boundaries={0: boundary_setup_with_sources},
            hotstart_config=hotstart_config
        )
        
        assert bc.hotstart_config is not None
        assert bc.hotstart_config.enabled is True
        assert bc.hotstart_config.temp_var == "temperature"
        assert bc.hotstart_config.salt_var == "salinity"
        assert bc.hotstart_config.output_filename == "test_hotstart.nc"

    def test_generate_hotstart_method(self, boundary_setup_with_sources, tidal_dataset, grid3d, time_range):
        """Test the _generate_hotstart method with real data."""
        # Create boundary conditions with enabled hotstart
        hotstart_config = HotstartConfig(
            enabled=True,
            temp_var="temperature",
            salt_var="salinity"
        )
        
        bc = SCHISMDataBoundaryConditions(
            tidal_data=tidal_dataset,
            setup_type="hybrid",
            boundaries={0: boundary_setup_with_sources},
            hotstart_config=hotstart_config
        )
        
        # Test the _generate_hotstart method
        with tempfile.TemporaryDirectory() as tmpdir:
            result = bc._generate_hotstart(tmpdir, grid3d, time_range)
            
            # Verify return value is a path
            assert isinstance(result, str)
            
            # Verify the file exists
            assert Path(result).exists()
            
            # Verify the filename matches configuration
            assert Path(result).name == "hotstart.nc"

    def test_generate_hotstart_no_temp_source(self, boundary_setup_no_sources, tidal_dataset, grid3d, time_range):
        """Test _generate_hotstart raises error when no temperature source available."""
        hotstart_config = HotstartConfig(enabled=True)
        
        bc = SCHISMDataBoundaryConditions(
            tidal_data=tidal_dataset,
            setup_type="hybrid",
            boundaries={0: boundary_setup_no_sources},
            hotstart_config=hotstart_config
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Hotstart generation requires both temperature and salinity sources"):
                bc._generate_hotstart(tmpdir, grid3d, time_range)

    def test_generate_hotstart_no_salt_source(self, hycom_source, hycom_coords, tidal_dataset, grid3d, time_range):
        """Test _generate_hotstart raises error when no salinity source available."""
        # Create boundary setup with only temperature source
        temp_source = SCHISMDataBoundary(
            source=hycom_source,
            variables=["temperature"],
            coords=hycom_coords
        )
        
        boundary_setup = BoundarySetupWithSource(
            elev_type=5,
            vel_type=4,
            temp_type=4,
            salt_type=1,  # No salt source needed for type 1
            temp_source=temp_source
            # No salt_source
        )
        
        hotstart_config = HotstartConfig(enabled=True)
        
        bc = SCHISMDataBoundaryConditions(
            tidal_data=tidal_dataset,
            setup_type="hybrid",
            boundaries={0: boundary_setup},
            hotstart_config=hotstart_config
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Hotstart generation requires both temperature and salinity sources"):
                bc._generate_hotstart(tmpdir, grid3d, time_range)

    def test_multiple_boundaries_hotstart_source_selection(self, hycom_source, hycom_coords, tidal_dataset, grid3d, time_range):
        """Test that hotstart uses sources from any boundary that has both temp and salt."""
        # Create first boundary without temp/salt sources
        boundary_0 = BoundarySetupWithSource(
            elev_type=1,
            vel_type=1,
            temp_type=1,
            salt_type=1
        )
        
        # Create second boundary with temp/salt sources
        temp_source = SCHISMDataBoundary(
            source=hycom_source,
            variables=["temperature"],
            coords=hycom_coords
        )
        salt_source = SCHISMDataBoundary(
            source=hycom_source,
            variables=["salinity"],
            coords=hycom_coords
        )
        
        boundary_1 = BoundarySetupWithSource(
            elev_type=5,
            vel_type=4,
            temp_type=4,
            salt_type=4,
            temp_source=temp_source,
            salt_source=salt_source
        )
        
        hotstart_config = HotstartConfig(enabled=True)
        
        bc = SCHISMDataBoundaryConditions(
            tidal_data=tidal_dataset,
            setup_type="hybrid",
            boundaries={0: boundary_0, 1: boundary_1},
            hotstart_config=hotstart_config
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = bc._generate_hotstart(tmpdir, grid3d, time_range)
            
            # Verify the file was created successfully
            assert Path(result).exists()

    def test_hotstart_with_custom_variable_names(self, boundary_setup_with_sources, tidal_dataset, grid3d, time_range):
        """Test hotstart generation with custom variable names."""
        hotstart_config = HotstartConfig(
            enabled=True,
            temp_var="temperature",  # Should match the variable in HYCOM data
            salt_var="salinity",     # Should match the variable in HYCOM data
            output_filename="custom_hotstart.nc"
        )
        
        bc = SCHISMDataBoundaryConditions(
            tidal_data=tidal_dataset,
            setup_type="hybrid",
            boundaries={0: boundary_setup_with_sources},
            hotstart_config=hotstart_config
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = bc._generate_hotstart(tmpdir, grid3d, time_range)
            
            # Verify the custom filename was used
            assert Path(result).name == "custom_hotstart.nc"
            assert Path(result).exists()

    def test_hotstart_file_structure(self, boundary_setup_with_sources, tidal_dataset, grid3d, time_range):
        """Test that the generated hotstart file has the correct structure."""
        hotstart_config = HotstartConfig(enabled=True)
        
        bc = SCHISMDataBoundaryConditions(
            tidal_data=tidal_dataset,
            setup_type="hybrid",
            boundaries={0: boundary_setup_with_sources},
            hotstart_config=hotstart_config
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = bc._generate_hotstart(tmpdir, grid3d, time_range)
            
            # Check file structure using xarray
            import xarray as xr
            
            ds = xr.open_dataset(result)
            
            # Check basic dimensions exist
            expected_dims = ["node", "elem", "side", "nVert", "ntracers", "one"]
            for dim in expected_dims:
                assert dim in ds.dims
            
            # Check tracer variables exist
            assert "tr_nd" in ds.variables  # Node tracers
            assert "tr_el" in ds.variables  # Element tracers
            
            # Check tracer dimensions (should have 2 tracers: temp and salt)
            assert ds.variables["tr_nd"].shape[2] == 2
            assert ds.variables["tr_el"].shape[2] == 2
            
            ds.close()

    @pytest.mark.slow
    def test_end_to_end_boundary_conditions_with_hotstart(self, boundary_setup_with_sources, tidal_dataset, grid3d, time_range):
        """Test end-to-end boundary conditions generation with hotstart enabled."""
        hotstart_config = HotstartConfig(
            enabled=True,
            temp_var="temperature",
            salt_var="salinity",
            output_filename="e2e_hotstart.nc"
        )
        
        bc = SCHISMDataBoundaryConditions(
            tidal_data=tidal_dataset,
            setup_type="hybrid",
            boundaries={0: boundary_setup_with_sources},
            hotstart_config=hotstart_config
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # This will generate boundary conditions AND hotstart
            result = bc.get(tmpdir, grid3d, time_range)
            
            # Verify hotstart was included in results
            assert "hotstart" in result
            assert Path(result["hotstart"]).exists()
            
            # Verify other boundary files were also created
            assert len(result) > 1  # Should have more than just hotstart

    def test_boundary_conditions_hotstart_disabled_no_generation(self, boundary_setup_with_sources, tidal_dataset, grid3d, time_range):
        """Test that hotstart is not generated when disabled."""
        hotstart_config = HotstartConfig(enabled=False)
        
        bc = SCHISMDataBoundaryConditions(
            tidal_data=tidal_dataset,
            setup_type="hybrid",
            boundaries={0: boundary_setup_with_sources},
            hotstart_config=hotstart_config
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = bc.get(tmpdir, grid3d, time_range)
            
            # Verify hotstart was not included in results
            assert "hotstart" not in result