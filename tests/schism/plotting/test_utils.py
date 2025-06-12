"""
Test suite for SCHISM plotting utilities.

This module tests the utility functions used across the plotting module
including file detection, data loading, and plotting helpers.
"""

import pytest
import numpy as np
import xarray as xr
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile

from rompy.schism.plotting.utils import (
    validate_file_exists,
    setup_cartopy_axis,
    get_geographic_extent,
    setup_colormap,
    add_grid_overlay,
    add_boundary_overlay,
    detect_file_type,
    load_schism_data,
    get_variable_info,
    create_time_subset,
    format_scientific_notation,
    create_diverging_colormap_levels,
    save_plot
)


class TestValidateFileExists:
    """Test validate_file_exists function."""
    
    def test_existing_file(self, tmp_path):
        """Test with existing file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        
        assert validate_file_exists(test_file) is True
        assert validate_file_exists(str(test_file)) is True
    
    def test_nonexistent_file(self):
        """Test with non-existent file."""
        assert validate_file_exists("nonexistent.txt") is False
        assert validate_file_exists(Path("nonexistent.txt")) is False


class TestSetupCartopyAxis:
    """Test setup_cartopy_axis function."""
    
    @patch('cartopy.crs.PlateCarree')
    def test_cartopy_available(self, mock_platecarree):
        """Test when cartopy is available."""
        mock_projection = Mock()
        mock_platecarree.return_value = mock_projection
        
        result = setup_cartopy_axis()
        
        assert result == mock_projection
        mock_platecarree.assert_called_once()
    
    def test_cartopy_unavailable(self):
        """Test when cartopy is not available."""
        with patch('builtins.__import__', side_effect=lambda name, *args, **kwargs: exec('raise ImportError()') if name == 'cartopy.crs' else __import__(name, *args, **kwargs)):
            result = setup_cartopy_axis()
            
            assert result is None


class TestGetGeographicExtent:
    """Test get_geographic_extent function."""
    
    def test_basic_extent(self):
        """Test basic geographic extent calculation."""
        lons = np.array([10, 20, 30])
        lats = np.array([40, 50, 60])
        
        extent = get_geographic_extent(lons, lats)
        
        expected_lon_range = 20  # 30 - 10
        expected_lat_range = 20  # 60 - 40
        expected_buffer = 0.1  # default buffer
        
        assert extent[0] == 10 - expected_lon_range * expected_buffer  # lon_min
        assert extent[1] == 30 + expected_lon_range * expected_buffer  # lon_max
        assert extent[2] == 40 - expected_lat_range * expected_buffer  # lat_min
        assert extent[3] == 60 + expected_lat_range * expected_buffer  # lat_max
    
    def test_custom_buffer(self):
        """Test with custom buffer."""
        lons = np.array([0, 10])
        lats = np.array([0, 10])
        buffer = 0.2
        
        extent = get_geographic_extent(lons, lats, buffer=buffer)
        
        expected_buffer_val = 10 * buffer  # range is 10
        
        assert extent[0] == 0 - expected_buffer_val
        assert extent[1] == 10 + expected_buffer_val
        assert extent[2] == 0 - expected_buffer_val
        assert extent[3] == 10 + expected_buffer_val
    
    def test_with_nan_values(self):
        """Test with NaN values in coordinates."""
        lons = np.array([10, np.nan, 30])
        lats = np.array([40, 50, np.nan])
        
        extent = get_geographic_extent(lons, lats)
        
        # Should handle NaN values properly
        assert not np.isnan(extent[0])
        assert not np.isnan(extent[1])
        assert not np.isnan(extent[2])
        assert not np.isnan(extent[3])


class TestSetupColormap:
    """Test setup_colormap function."""
    
    def test_basic_setup(self):
        """Test basic colormap setup."""
        data = np.array([1, 2, 3, 4, 5])
        
        cmap_name, norm, levels = setup_colormap(data)
        
        assert cmap_name == "viridis"  # default
        assert norm.vmin == 1
        assert norm.vmax == 5
        assert levels is None
    
    def test_custom_parameters(self):
        """Test with custom parameters."""
        data = np.array([1, 2, 3, 4, 5])
        
        cmap_name, norm, levels = setup_colormap(
            data, cmap="plasma", vmin=0, vmax=10, levels=5
        )
        
        assert cmap_name == "plasma"
        assert norm.vmin == 0
        assert norm.vmax == 10
        assert len(levels) == 5
        assert levels[0] == 0
        assert levels[-1] == 10
    
    def test_custom_levels_list(self):
        """Test with custom levels as list."""
        data = np.array([1, 2, 3, 4, 5])
        custom_levels = [1, 2, 3, 4, 5]
        
        cmap_name, norm, levels = setup_colormap(data, levels=custom_levels)
        
        np.testing.assert_array_equal(levels, custom_levels)


class TestDetectFileType:
    """Test detect_file_type function."""
    
    def test_gr3_files(self):
        """Test .gr3 file detection."""
        assert detect_file_type("test.gr3") == "gr3"
        assert detect_file_type("depth.GR3") == "gr3"  # case insensitive
    
    def test_th_nc_files(self):
        """Test .th.nc file detection."""
        assert detect_file_type("SAL_3D.th.nc") == "salinity_3d"
        assert detect_file_type("TEM_3D.th.nc") == "temperature_3d"
        assert detect_file_type("uv3D.th.nc") == "velocity_3d"
        assert detect_file_type("elev2D.th.nc") == "elevation_2d"
        assert detect_file_type("other.th.nc") == "boundary_th"
    
    def test_bctides_files(self):
        """Test bctides.in file detection."""
        assert detect_file_type("bctides.in") == "bctides"
        assert detect_file_type("/path/to/bctides.in") == "bctides"
    
    def test_sflux_files(self):
        """Test atmospheric forcing file detection."""
        assert detect_file_type("sflux_air_1.nc") == "atmospheric"
        assert detect_file_type("sflux_rad_1.nc") == "atmospheric"
    
    def test_unknown_files(self):
        """Test unknown file types."""
        assert detect_file_type("unknown.txt") == "unknown"
        assert detect_file_type("data.nc") == "unknown"


class TestLoadSchismData:
    """Test load_schism_data function."""
    
    def create_test_netcdf(self, file_path):
        """Create a test NetCDF file."""
        times = np.arange(5)
        nodes = np.arange(3)
        
        ds = xr.Dataset(
            {
                "temperature": (["time", "node"], np.random.rand(5, 3)),
                "salinity": (["time", "node"], np.random.rand(5, 3)),
            },
            coords={
                "time": times,
                "node": nodes,
            }
        )
        
        ds.to_netcdf(file_path)
        return ds
    
    def test_load_netcdf_file(self, tmp_path):
        """Test loading NetCDF file."""
        file_path = tmp_path / "test.th.nc"
        original_ds = self.create_test_netcdf(file_path)
        
        loaded_ds = load_schism_data(file_path)
        
        assert isinstance(loaded_ds, xr.Dataset)
        assert "temperature" in loaded_ds.data_vars
        assert "salinity" in loaded_ds.data_vars
        
        # Clean up
        loaded_ds.close()
    
    def test_nonexistent_file(self):
        """Test with non-existent file."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            load_schism_data("nonexistent.nc")
    
    def test_unsupported_file_type(self, tmp_path):
        """Test with unsupported file type."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("not a netcdf file")
        
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_schism_data(file_path)


class TestGetVariableInfo:
    """Test get_variable_info function."""
    
    def create_test_dataset(self):
        """Create test dataset with various variables."""
        times = np.arange(10)
        nodes = np.arange(5)
        depths = np.arange(3)
        
        # Create deterministic data with known bounds
        temp_data = np.full((10, 5, 3), 20.0)  # Fill with middle value
        temp_data[0, 0, 0] = 15.5  # Set known min
        temp_data[5, 2, 1] = 25.8  # Set known max
        # Add some variation but keep within bounds
        for i in range(10):
            for j in range(5):
                for k in range(3):
                    if not (i == 0 and j == 0 and k == 0) and not (i == 5 and j == 2 and k == 1):
                        temp_data[i, j, k] = 16.0 + (i + j + k) * 0.5  # Predictable variation
        
        ds = xr.Dataset(
            {
                "temperature": (
                    ["time", "node", "depth"], 
                    temp_data,
                    {"units": "°C", "long_name": "Sea Water Temperature"}
                ),
                "salinity": (
                    ["time", "node"], 
                    np.random.rand(10, 5),
                    {"units": "psu", "description": "Sea water salinity"}
                ),
            },
            coords={
                "time": times,
                "node": nodes,
                "depth": depths,
                "lon": (["node"], np.random.rand(5)),
                "lat": (["node"], np.random.rand(5)),
            }
        )
        
        return ds
    
    def test_3d_variable_info(self):
        """Test getting info for 3D variable."""
        ds = self.create_test_dataset()
        
        info = get_variable_info(ds, "temperature")
        
        assert info["name"] == "temperature"
        assert info["dims"] == ("time", "node", "depth")
        assert info["shape"] == (10, 5, 3)
        assert info["units"] == "°C"
        assert info["long_name"] == "Sea Water Temperature"
        assert info["has_time"] is True
        assert info["has_depth"] is True
        assert "node" in info["spatial_dims"]
        assert info["min_value"] == 15.5
        assert info["max_value"] == 25.8
    
    def test_2d_variable_info(self):
        """Test getting info for 2D variable."""
        ds = self.create_test_dataset()
        
        info = get_variable_info(ds, "salinity")
        
        assert info["name"] == "salinity"
        assert info["dims"] == ("time", "node")
        assert info["shape"] == (10, 5)
        assert info["units"] == "psu"
        assert info["description"] == "Sea water salinity"
        assert info["has_time"] is True
        assert info["has_depth"] is False
        assert "node" in info["spatial_dims"]
    
    def test_nonexistent_variable(self):
        """Test with non-existent variable."""
        ds = self.create_test_dataset()
        
        with pytest.raises(ValueError, match="Variable nonexistent not found"):
            get_variable_info(ds, "nonexistent")


class TestCreateTimeSubset:
    """Test create_time_subset function."""
    
    def create_time_dataset(self):
        """Create dataset with time dimension."""
        times = np.arange(np.datetime64('2023-01-01'), np.datetime64('2023-01-11'))
        nodes = np.arange(3)
        
        ds = xr.Dataset(
            {
                "temperature": (["time", "node"], np.random.rand(10, 3)),
            },
            coords={
                "time": times,
                "node": nodes,
            }
        )
        
        return ds
    
    def test_no_time_dimension(self):
        """Test with dataset without time dimension."""
        ds = xr.Dataset({"temp": (["x"], [1, 2, 3])})
        
        result = create_time_subset(ds)
        
        assert result is ds  # Should return original dataset
    
    def test_time_index_subset(self):
        """Test subsetting by time index."""
        ds = self.create_time_dataset()
        
        result = create_time_subset(ds, time_idx=5)
        
        assert "time" not in result.dims  # Should be single time point
        assert result["temperature"].shape == (3,)  # Only node dimension left
    
    def test_time_range_subset(self):
        """Test subsetting by time range."""
        ds = self.create_time_dataset()
        
        result = create_time_subset(ds, start_time='2023-01-03', end_time='2023-01-07')
        
        assert len(result["time"]) == 5  # 5 days inclusive
        assert result["temperature"].shape == (5, 3)
    
    def test_no_subset_parameters(self):
        """Test with no subset parameters."""
        ds = self.create_time_dataset()
        
        result = create_time_subset(ds)
        
        assert result is ds  # Should return original dataset


class TestFormatScientificNotation:
    """Test format_scientific_notation function."""
    
    def test_small_numbers(self):
        """Test formatting of small numbers."""
        assert format_scientific_notation(0.001) == "1.00e-03"
        assert format_scientific_notation(0.0001) == "1.00e-04"
    
    def test_large_numbers(self):
        """Test formatting of large numbers."""
        assert format_scientific_notation(10000) == "1.00e+04"
        assert format_scientific_notation(100000) == "1.00e+05"
    
    def test_normal_numbers(self):
        """Test formatting of normal-range numbers."""
        assert format_scientific_notation(1.234) == "1.23"
        assert format_scientific_notation(12.345) == "12.35"
        assert format_scientific_notation(123.456) == "123.46"
    
    def test_custom_precision(self):
        """Test custom precision."""
        assert format_scientific_notation(1.23456, precision=3) == "1.235"
        assert format_scientific_notation(0.001, precision=1) == "1.0e-03"


class TestCreateDivergingColormapLevels:
    """Test create_diverging_colormap_levels function."""
    
    def test_symmetric_data(self):
        """Test with symmetric data around center."""
        data = np.array([-5, -2, 0, 3, 5])
        
        levels = create_diverging_colormap_levels(data, num_levels=11, center=0)
        
        assert len(levels) == 11
        assert levels[0] == -5
        assert levels[-1] == 5
        assert levels[5] == 0  # Middle level should be center
    
    def test_asymmetric_data(self):
        """Test with asymmetric data."""
        data = np.array([-2, 0, 1, 5, 8])
        
        levels = create_diverging_colormap_levels(data, num_levels=9, center=0)
        
        max_dev = max(abs(-2 - 0), abs(8 - 0))  # 8 is max deviation
        assert len(levels) == 9
        assert levels[0] == -max_dev
        assert levels[-1] == max_dev
        assert levels[4] == 0  # Middle level should be center
    
    def test_custom_center(self):
        """Test with custom center value."""
        data = np.array([8, 10, 12, 14, 16])
        center = 12
        
        levels = create_diverging_colormap_levels(data, num_levels=5, center=center)
        
        max_dev = max(abs(8 - 12), abs(16 - 12))  # 4 is max deviation
        assert len(levels) == 5
        assert levels[0] == center - max_dev
        assert levels[-1] == center + max_dev
        assert levels[2] == center


class TestSavePlot:
    """Test save_plot function."""
    
    @patch('matplotlib.pyplot.Figure')
    def test_successful_save(self, mock_figure_class, tmp_path):
        """Test successful plot saving."""
        mock_fig = Mock()
        output_file = tmp_path / "test_plot.png"
        
        save_plot(mock_fig, output_file)
        
        mock_fig.savefig.assert_called_once_with(
            output_file, dpi=300, bbox_inches='tight'
        )
    
    @patch('matplotlib.pyplot.Figure')
    def test_save_with_custom_parameters(self, mock_figure_class, tmp_path):
        """Test saving with custom parameters."""
        mock_fig = Mock()
        output_file = tmp_path / "test_plot.png"
        
        save_plot(mock_fig, output_file, dpi=150, bbox_inches='standard')
        
        mock_fig.savefig.assert_called_once_with(
            output_file, dpi=150, bbox_inches='standard'
        )
    
    @patch('matplotlib.pyplot.Figure')
    def test_save_error_handling(self, mock_figure_class, tmp_path):
        """Test error handling during save."""
        mock_fig = Mock()
        mock_fig.savefig.side_effect = Exception("Save failed")
        output_file = tmp_path / "test_plot.png"
        
        with pytest.raises(Exception, match="Save failed"):
            save_plot(mock_fig, output_file)


if __name__ == "__main__":
    pytest.main([__file__])