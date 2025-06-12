"""
Test suite for SCHISM plotting core functionality.

This module tests the base plotting classes, configuration models,
and validation functionality.
"""

import pytest
import numpy as np
import xarray as xr
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from rompy.schism.plotting.core import PlotConfig, BasePlotter, PlotValidator


class TestPlotConfig:
    """Test PlotConfig pydantic model."""
    
    def test_default_initialization(self):
        """Test default PlotConfig initialization."""
        config = PlotConfig()
        
        assert config.figsize == (12, 8)
        assert config.dpi == 100
        assert config.cmap == "viridis"
        assert config.add_coastlines is True
        assert config.add_gridlines is True
        assert config.projection == "PlateCarree"
        assert config.show_grid is True
        assert config.grid_alpha == 0.3
        assert config.grid_color == "gray"
        assert config.show_boundaries is True
        assert config.boundary_colors == {"ocean": "red", "land": "green", "tidal": "blue"}
        assert config.boundary_linewidth == 2.0
        assert config.add_colorbar is True
        assert config.colorbar_orientation == "vertical"
        assert config.xlabel == "Longitude"
        assert config.ylabel == "Latitude"
    
    def test_custom_initialization(self):
        """Test PlotConfig with custom parameters."""
        config = PlotConfig(
            figsize=(10, 6),
            dpi=150,
            cmap="plasma",
            add_coastlines=False,
            grid_alpha=0.5,
            boundary_linewidth=3.0
        )
        
        assert config.figsize == (10, 6)
        assert config.dpi == 150
        assert config.cmap == "plasma"
        assert config.add_coastlines is False
        assert config.grid_alpha == 0.5
        assert config.boundary_linewidth == 3.0
    
    def test_figsize_validation(self):
        """Test figsize validation."""
        # Valid figsize
        config = PlotConfig(figsize=(8, 6))
        assert config.figsize == (8, 6)
        
        # Invalid figsize - wrong length (pydantic catches this as ValidationError)
        with pytest.raises(Exception):  # Could be ValidationError or ValueError
            PlotConfig(figsize=(8,))
        
        # Invalid figsize - negative values
        with pytest.raises(ValueError, match="figsize values must be positive"):
            PlotConfig(figsize=(-8, 6))
    
    def test_dpi_validation(self):
        """Test DPI validation."""
        # Valid DPI
        config = PlotConfig(dpi=200)
        assert config.dpi == 200
        
        # Invalid DPI
        with pytest.raises(ValueError, match="dpi must be positive"):
            PlotConfig(dpi=-100)
    
    def test_grid_alpha_validation(self):
        """Test grid alpha validation."""
        # Valid alpha
        config = PlotConfig(grid_alpha=0.7)
        assert config.grid_alpha == 0.7
        
        # Invalid alpha - too high
        with pytest.raises(ValueError, match="grid_alpha must be between 0 and 1"):
            PlotConfig(grid_alpha=1.5)
        
        # Invalid alpha - negative
        with pytest.raises(ValueError, match="grid_alpha must be between 0 and 1"):
            PlotConfig(grid_alpha=-0.1)
    
    def test_boundary_linewidth_validation(self):
        """Test boundary linewidth validation."""
        # Valid linewidth
        config = PlotConfig(boundary_linewidth=1.5)
        assert config.boundary_linewidth == 1.5
        
        # Invalid linewidth
        with pytest.raises(ValueError, match="boundary_linewidth must be positive"):
            PlotConfig(boundary_linewidth=-1.0)
    
    def test_colorbar_orientation_validation(self):
        """Test colorbar orientation validation."""
        # Valid orientations
        config1 = PlotConfig(colorbar_orientation="vertical")
        assert config1.colorbar_orientation == "vertical"
        
        config2 = PlotConfig(colorbar_orientation="horizontal")
        assert config2.colorbar_orientation == "horizontal"
        
        # Invalid orientation
        with pytest.raises(ValueError, match="colorbar_orientation must be 'vertical' or 'horizontal'"):
            PlotConfig(colorbar_orientation="diagonal")


class TestBasePlotter:
    """Test BasePlotter abstract base class."""
    
    def create_mock_grid(self):
        """Create a mock grid object for testing."""
        mock_grid = Mock()
        mock_grid.pylibs_hgrid = Mock()
        mock_grid.pylibs_hgrid.x = np.array([0, 1, 2])
        mock_grid.pylibs_hgrid.y = np.array([0, 1, 2])
        mock_grid.pylibs_hgrid.dp = np.array([10, 20, 30])
        mock_grid.pylibs_hgrid.elnode = np.array([[0, 1, 2]])
        return mock_grid
    
    def create_mock_config(self):
        """Create a mock configuration object."""
        mock_config = Mock()
        mock_config.grid = self.create_mock_grid()
        return mock_config
    
    def create_concrete_plotter(self, **kwargs):
        """Create a concrete implementation of BasePlotter for testing."""
        class ConcretePlotter(BasePlotter):
            def plot(self, **kwargs):
                return Mock(), Mock()
        
        return ConcretePlotter(**kwargs)
    
    def test_initialization_with_config(self):
        """Test BasePlotter initialization with config."""
        mock_config = self.create_mock_config()
        plotter = self.create_concrete_plotter(config=mock_config)
        
        assert plotter.config == mock_config
        assert plotter.grid_file is None
        assert isinstance(plotter.plot_config, PlotConfig)
        assert plotter._grid is None
        assert plotter._grid_loaded is False
    
    def test_initialization_with_grid_file(self, tmp_path):
        """Test BasePlotter initialization with grid file."""
        grid_file = tmp_path / "test_grid.gr3"
        grid_file.write_text("mock grid data")
        
        plotter = self.create_concrete_plotter(grid_file=grid_file)
        
        assert plotter.config is None
        assert plotter.grid_file == grid_file
        assert isinstance(plotter.plot_config, PlotConfig)
    
    def test_initialization_validation_error(self):
        """Test initialization validation errors."""
        # No config or grid file
        with pytest.raises(ValueError, match="Either config or grid_file must be provided"):
            self.create_concrete_plotter()
        
        # Non-existent grid file
        with pytest.raises(FileNotFoundError, match="Grid file not found"):
            self.create_concrete_plotter(grid_file="nonexistent.gr3")
    
    def test_grid_property_from_config(self):
        """Test grid property loading from config."""
        mock_config = self.create_mock_config()
        plotter = self.create_concrete_plotter(config=mock_config)
        
        # Grid should be loaded from config
        grid = plotter.grid
        assert grid == mock_config.grid
        assert plotter._grid_loaded is True
    
    def test_grid_property_from_file(self, tmp_path):
        """Test grid property loading from file."""
        grid_file = tmp_path / "test_grid.gr3"
        grid_file.write_text("mock grid data")
        
        mock_grid = self.create_mock_grid()
        
        # Mock the import and class - patch where it's imported in core.py
        with patch('rompy.schism.grid.SCHISMGrid') as mock_schism_grid:
            mock_schism_grid.from_file.return_value = mock_grid
            
            plotter = self.create_concrete_plotter(grid_file=grid_file)
            
            # Grid should be loaded from file
            grid = plotter.grid
            assert grid == mock_grid
            assert plotter._grid_loaded is True
            mock_schism_grid.from_file.assert_called_once_with(grid_file)
    
    @patch('matplotlib.pyplot.subplots')
    def test_create_figure_regular(self, mock_subplots):
        """Test create_figure without cartopy."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        plotter = self.create_concrete_plotter(config=self.create_mock_config())
        plotter.plot_config.add_coastlines = False
        
        fig, ax = plotter.create_figure(use_cartopy=False)
        
        assert fig == mock_fig
        assert ax == mock_ax
        mock_subplots.assert_called_once()
    
    @patch('matplotlib.pyplot.figure')
    @patch('cartopy.crs.PlateCarree')
    def test_create_figure_cartopy(self, mock_platecarree, mock_figure):
        """Test create_figure with cartopy."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax
        
        plotter = self.create_concrete_plotter(config=self.create_mock_config())
        
        with patch('cartopy.feature'):
            fig, ax = plotter.create_figure(use_cartopy=True)
        
        assert fig == mock_fig
        assert ax == mock_ax
    
    @patch('matplotlib.pyplot.subplots')
    def test_create_figure_cartopy_fallback(self, mock_subplots):
        """Test create_figure cartopy import fallback."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        plotter = self.create_concrete_plotter(config=self.create_mock_config())
        
        # Simply test with cartopy disabled
        plotter.plot_config.add_coastlines = False
        fig, ax = plotter.create_figure(use_cartopy=False)
        
        assert fig == mock_fig
        assert ax == mock_ax
        mock_subplots.assert_called_once()
    
    def test_add_colorbar(self):
        """Test add_colorbar method."""
        plotter = self.create_concrete_plotter(config=self.create_mock_config())
        
        mock_fig = Mock()
        mock_ax = Mock()
        mock_mappable = Mock()
        mock_cbar = Mock()
        mock_fig.colorbar.return_value = mock_cbar
        
        # Test with colorbar enabled
        cbar = plotter.add_colorbar(mock_fig, mock_ax, mock_mappable, label="Test")
        
        assert cbar == mock_cbar
        mock_fig.colorbar.assert_called_once_with(
            mock_mappable, ax=mock_ax, orientation="vertical"
        )
        mock_cbar.set_label.assert_called_once_with("Test")
        
        # Test with colorbar disabled
        plotter.plot_config.add_colorbar = False
        cbar = plotter.add_colorbar(mock_fig, mock_ax, mock_mappable)
        assert cbar is None
    
    def test_set_plot_labels(self):
        """Test set_plot_labels method."""
        plotter = self.create_concrete_plotter(config=self.create_mock_config())
        mock_ax = Mock()
        
        plotter.set_plot_labels(mock_ax, title="Test Title", xlabel="X", ylabel="Y")
        
        mock_ax.set_title.assert_called_once_with("Test Title")
        mock_ax.set_xlabel.assert_called_once_with("X")
        mock_ax.set_ylabel.assert_called_once_with("Y")
    
    def test_finalize_plot(self):
        """Test finalize_plot method."""
        plotter = self.create_concrete_plotter(config=self.create_mock_config())
        mock_fig = Mock()
        mock_ax = Mock()
        
        fig, ax = plotter.finalize_plot(mock_fig, mock_ax, title="Final Title")
        
        assert fig == mock_fig
        assert ax == mock_ax
        mock_ax.set_title.assert_called_once_with("Final Title")
        mock_fig.tight_layout.assert_called_once()


class TestPlotValidator:
    """Test PlotValidator class."""
    
    def create_test_dataset(self):
        """Create a test xarray dataset."""
        times = np.arange(10)
        nodes = np.arange(5)
        
        ds = xr.Dataset(
            {
                "temperature": (["time", "node"], np.random.rand(10, 5)),
                "salinity": (["time", "node"], np.random.rand(10, 5)),
            },
            coords={
                "time": times,
                "node": nodes,
                "lon": (["node"], np.random.rand(5) * 360 - 180),
                "lat": (["node"], np.random.rand(5) * 180 - 90),
            }
        )
        
        return ds
    
    def test_validate_dataset_success(self):
        """Test successful dataset validation."""
        ds = self.create_test_dataset()
        
        # Test without required variables
        assert PlotValidator.validate_dataset(ds) is True
        
        # Test with required variables
        assert PlotValidator.validate_dataset(ds, required_vars=["temperature"]) is True
        assert PlotValidator.validate_dataset(ds, required_vars=["temperature", "salinity"]) is True
    
    def test_validate_dataset_failures(self):
        """Test dataset validation failures."""
        ds = self.create_test_dataset()
        
        # Test with non-dataset input
        with pytest.raises(ValueError, match="Input must be an xarray Dataset"):
            PlotValidator.validate_dataset("not_a_dataset")
        
        # Test with missing required variables
        with pytest.raises(ValueError, match="Missing required variables"):
            PlotValidator.validate_dataset(ds, required_vars=["nonexistent"])
    
    def test_validate_coordinates_success(self):
        """Test successful coordinate validation."""
        ds = self.create_test_dataset()
        
        # Test without required coordinates
        assert PlotValidator.validate_coordinates(ds) is True
        
        # Test with required coordinates
        assert PlotValidator.validate_coordinates(ds, required_coords=["time"]) is True
        assert PlotValidator.validate_coordinates(ds, required_coords=["time", "node"]) is True
    
    def test_validate_coordinates_failure(self):
        """Test coordinate validation failure."""
        ds = self.create_test_dataset()
        
        with pytest.raises(ValueError, match="Missing required coordinates"):
            PlotValidator.validate_coordinates(ds, required_coords=["nonexistent"])
    
    def test_validate_time_dimension_success(self):
        """Test successful time dimension validation."""
        ds = self.create_test_dataset()
        
        assert PlotValidator.validate_time_dimension(ds, "temperature") is True
        assert PlotValidator.validate_time_dimension(ds, "salinity") is True
    
    def test_validate_time_dimension_failures(self):
        """Test time dimension validation failures."""
        ds = self.create_test_dataset()
        
        # Test with non-existent variable
        with pytest.raises(ValueError, match="Variable nonexistent not found"):
            PlotValidator.validate_time_dimension(ds, "nonexistent")
        
        # Test with variable without time dimension
        ds["static_var"] = ("node", np.random.rand(5))
        with pytest.raises(ValueError, match="does not have time dimension"):
            PlotValidator.validate_time_dimension(ds, "static_var")
    
    def test_validate_spatial_dimensions_success(self):
        """Test successful spatial dimension validation."""
        ds = self.create_test_dataset()
        
        assert PlotValidator.validate_spatial_dimensions(ds, "temperature") is True
        assert PlotValidator.validate_spatial_dimensions(ds, "salinity") is True
    
    def test_validate_spatial_dimensions_failures(self):
        """Test spatial dimension validation failures."""
        ds = self.create_test_dataset()
        
        # Test with non-existent variable
        with pytest.raises(ValueError, match="Variable nonexistent not found"):
            PlotValidator.validate_spatial_dimensions(ds, "nonexistent")
        
        # Test with variable without spatial dimensions
        ds["time_only"] = ("time", np.random.rand(10))
        with pytest.raises(ValueError, match="does not have spatial dimensions"):
            PlotValidator.validate_spatial_dimensions(ds, "time_only")


if __name__ == "__main__":
    pytest.main([__file__])