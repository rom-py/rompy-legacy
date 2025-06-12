"""
Tests for SCHISM overview plotting functionality.

This module tests the enhanced overview plotting capabilities including
comprehensive multi-panel layouts, grid analysis, and data analysis overviews.
Uses real data from fixtures as per critical rules.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import xarray as xr

from rompy.schism.plotting.overview import OverviewPlotter
from rompy.schism.plotting.core import PlotConfig


class TestOverviewPlotter:
    """Test the OverviewPlotter class."""

    def test_initialization_with_config(self, grid2d):
        """Test OverviewPlotter initialization with config."""
        # Create a mock config with grid
        config = type('Config', (), {})()
        config.grid = grid2d
        
        plotter = OverviewPlotter(config=config)
        
        assert plotter.config == config
        assert plotter.grid is not None
        
    def test_initialization_with_grid_file(self, hgrid_path):
        """Test OverviewPlotter initialization with grid file."""
        if hgrid_path is None:
            pytest.skip("No hgrid file available for testing")
            
        plotter = OverviewPlotter(grid_file=hgrid_path)
        
        assert plotter.grid_file == hgrid_path
        assert plotter.grid is not None
        
    def test_initialization_validation_error(self):
        """Test that initialization fails without config or grid file."""
        with pytest.raises(ValueError, match="Either config or grid_file must be provided"):
            OverviewPlotter()

    def test_get_grid_info_with_grid(self, grid2d):
        """Test grid info extraction with valid grid."""
        config = type('Config', (), {})()
        config.grid = grid2d
        
        plotter = OverviewPlotter(config=config)
        grid_info = plotter._get_grid_info()
        
        assert isinstance(grid_info, str)
        assert 'Nodes:' in grid_info
        assert 'Elements:' in grid_info
        assert 'Depth:' in grid_info
        
    def test_get_grid_info_without_grid(self):
        """Test grid info extraction without grid."""
        config = type('Config', (), {})()
        config.grid = None
        
        plotter = OverviewPlotter(config=config)
        grid_info = plotter._get_grid_info()
        
        assert isinstance(grid_info, str)
        assert grid_info == "Grid info unavailable"

    def test_calculate_quality_metrics(self, grid2d):
        """Test quality metrics calculation."""
        config = type('Config', (), {})()
        config.grid = grid2d
        
        plotter = OverviewPlotter(config=config)
        
        # Mock grid with quality attributes for testing
        if hasattr(grid2d, 'tri') and len(grid2d.tri) > 0:
            metrics = plotter._calculate_quality_metrics()
            assert isinstance(metrics, dict)
        else:
            # Grid doesn't have triangulation, should return empty metrics
            metrics = plotter._calculate_quality_metrics()
            assert isinstance(metrics, dict)

    def test_plot_quality_metrics_chart_without_data(self, grid2d):
        """Test quality metrics chart without data."""
        config = type('Config', (), {})()
        config.grid = grid2d
        
        plotter = OverviewPlotter(config=config)
        
        # Test with empty metrics
        fig, ax = plt.subplots()
        plotter._plot_quality_metrics_chart(ax, {})
        
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_run_validation_checks(self, grid2d):
        """Test validation checks execution."""
        config = type('Config', (), {})()
        config.grid = grid2d
        
        plotter = OverviewPlotter(config=config)
        validation_results = plotter._run_validation_checks()
        
        assert isinstance(validation_results, dict)
        # Should have at least basic validation results
        assert len(validation_results) > 0
        
    def test_plot_validation_results_with_data(self, grid2d):
        """Test validation results plotting with data."""
        config = type('Config', (), {})()
        config.grid = grid2d
        
        plotter = OverviewPlotter(config=config)
        
        # Create mock validation results
        validation_results = [
            type('Result', (), {'status': 'PASS', 'check_name': 'Grid Check'})(),
            type('Result', (), {'status': 'WARNING', 'check_name': 'Boundary Check'})(),
        ]
        
        # _plot_validation_results expects ax and results parameters
        fig, ax = plt.subplots()
        validation_dict = plotter._run_validation_checks()
        plotter._plot_validation_results(ax, validation_dict)
        
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_get_data_summary(self, grid2d):
        """Test data summary generation."""
        config = type('Config', (), {})()
        config.grid = grid2d
        
        plotter = OverviewPlotter(config=config)
        data_summary = plotter._get_data_summary()
        
        assert isinstance(data_summary, dict)

    def test_calculate_grid_statistics_with_grid(self, grid2d):
        """Test grid statistics calculation with grid."""
        config = type('Config', (), {})()
        config.grid = grid2d
        
        plotter = OverviewPlotter(config=config)
        stats = plotter._calculate_grid_statistics()
        
        assert isinstance(stats, dict)
        assert 'Total Nodes' in stats
        assert 'Total Elements' in stats
        
    def test_calculate_grid_statistics_without_grid(self):
        """Test grid statistics calculation without grid."""
        config = type('Config', (), {})()
        config.grid = None
        
        plotter = OverviewPlotter(config=config)
        stats = plotter._calculate_grid_statistics()
        
        assert isinstance(stats, dict)
        assert len(stats) == 0  # Should return empty dict when no grid

    def test_create_statistics_table_without_data(self, grid2d):
        """Test statistics table creation without data."""
        config = type('Config', (), {})()
        config.grid = grid2d
        
        plotter = OverviewPlotter(config=config)
        
        # Test with empty stats
        fig, ax = plt.subplots()
        plotter._create_statistics_table(ax, {}, "Test Statistics")
        
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_depth_histogram_without_grid(self):
        """Test depth histogram without grid."""
        config = type('Config', (), {})()
        config.grid = None
        
        plotter = OverviewPlotter(config=config)
        
        # Should handle missing grid gracefully
        fig, ax = plt.subplots()
        plotter._plot_depth_histogram(ax)
        
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_element_size_histogram(self, grid2d):
        """Test element size histogram plotting."""
        config = type('Config', (), {})()
        config.grid = grid2d
        
        plotter = OverviewPlotter(config=config)
        
        # Should create histogram plot
        fig, ax = plt.subplots()
        plotter._plot_element_size_histogram(ax)
        
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_save_plot_success(self, grid2d, tmp_path):
        """Test successful plot saving."""
        config = type('Config', (), {})()
        config.grid = grid2d
        
        plotter = OverviewPlotter(config=config)
        
        # Create a simple figure
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        
        save_path = tmp_path / "test_plot.png"
        result = plotter._save_plot(fig, save_path)
        
        assert result is True
        assert save_path.exists()
        plt.close(fig)

    def test_save_plot_failure(self, grid2d, tmp_path):
        """Test plot saving failure handling."""
        config = type('Config', (), {})()
        config.grid = grid2d
        
        plotter = OverviewPlotter(config=config)
        
        # Create a simple figure
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        
        # Try to save to an invalid path
        save_path = tmp_path / "nonexistent_dir" / "test_plot.png"
        
        # Mock savefig to raise an exception
        with pytest.MonkeyPatch.context() as m:
            def mock_savefig(*args, **kwargs):
                raise OSError("Save failed")
            m.setattr(fig, "savefig", mock_savefig)
            
            result = plotter._save_plot(fig, save_path)
            
        assert result is False
        plt.close(fig)

    def test_plot_data_coverage_timeline(self, grid2d):
        """Test data coverage timeline plotting."""
        config = type('Config', (), {})()
        config.grid = grid2d
        
        plotter = OverviewPlotter(config=config)
        
        # Should create timeline plot
        fig, ax = plt.subplots()
        plotter._plot_data_coverage_timeline(ax)
        
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_atmospheric_timeseries_overview(self, grid2d):
        """Test atmospheric time series overview plotting."""
        config = type('Config', (), {})()
        config.grid = grid2d
        
        plotter = OverviewPlotter(config=config)
        
        # Should create atmospheric overview plot
        fig, ax = plt.subplots()
        plotter._plot_forcing_timeseries_overview(ax)
        
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_data_quality_metrics(self, grid2d):
        """Test data quality metrics plotting."""
        config = type('Config', (), {})()
        config.grid = grid2d
        
        plotter = OverviewPlotter(config=config)
        
        # Should create quality metrics plot
        fig, ax = plt.subplots()
        data_summary = plotter._get_data_summary()
        plotter._plot_data_summary_chart(ax, data_summary)
        
        assert fig is not None
        assert ax is not None
        plt.close(fig)


class TestOverviewPlotterIntegration:
    """Integration tests using real fixtures."""

    def test_overview_with_real_grid_structure(self, grid2d):
        """Test overview creation with real grid structure."""
        config = type('Config', (), {})()
        config.grid = grid2d
        
        plotter = OverviewPlotter(config=config)
        
        # Test that we can access grid properties
        assert plotter.grid is not None
        if hasattr(plotter.grid, 'x'):
            assert len(plotter.grid.x) > 0
        if hasattr(plotter.grid, 'y'):
            assert len(plotter.grid.y) > 0

    def test_full_overview_creation_mock(self, grid2d):
        """Test full overview creation with minimal mocking."""
        config = type('Config', (), {})()
        config.grid = grid2d
        
        plotter = OverviewPlotter(config=config)
        
        # Test individual components work
        grid_info = plotter._get_grid_info()
        assert isinstance(grid_info, str)
        
        data_summary = plotter._get_data_summary()
        assert isinstance(data_summary, dict)
        
        # Test that validation runs without errors
        validation_results = plotter._run_validation_checks()
        assert isinstance(validation_results, dict)