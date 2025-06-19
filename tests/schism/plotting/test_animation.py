"""
Tests for SCHISM animation plotting functionality.

This module tests the AnimationPlotter class and animation methods
in the SchismPlotter interface.
"""

import logging
import unittest.mock as mock
from pathlib import Path
from typing import Dict, Any
import tempfile

import numpy as np
import pytest
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from rompy.schism.plotting.animation import AnimationPlotter, AnimationConfig
from rompy.schism.plotting import SchismPlotter


class TestAnimationConfig:
    """Test AnimationConfig model."""
    
    def test_default_config(self):
        """Test default animation configuration."""
        config = AnimationConfig()
        
        assert config.frame_rate == 10
        assert config.interval == 100
        assert config.repeat is True
        assert config.bitrate == 1800
        assert config.quality == "medium"
        assert config.show_time_label is True
        assert config.time_label_format == "%Y-%m-%d %H:%M"
        assert config.time_label_position == (0.02, 0.98)
        assert config.show_progress is True
        
    def test_effective_interval(self):
        """Test effective interval calculation."""
        config = AnimationConfig(frame_rate=20)
        assert config.effective_interval == 50  # 1000/20
        
        config = AnimationConfig(frame_rate=5, interval=200)
        assert config.effective_interval == 200  # 1000/5
        
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid configuration
        config = AnimationConfig(
            frame_rate=15,
            duration=10.0,
            time_start="2023-01-01T00:00:00",
            time_end="2023-01-02T00:00:00"
        )
        assert config.frame_rate == 15
        assert config.duration == 10.0


class TestAnimationPlotter:
    """Test AnimationPlotter class."""
    
    @pytest.fixture
    def mock_dataset(self):
        """Create mock xarray dataset for testing."""
        times = np.arange('2023-01-01', '2023-01-02', dtype='datetime64[h]')
        data = np.random.random((len(times), 10))
        
        ds = xr.Dataset({
            'time_series': (['time', 'node'], data),
            'temperature': (['time', 'node'], data + 273.15),
            'time': times
        })
        return ds
        
    @pytest.fixture
    def mock_atmospheric_dataset(self):
        """Create mock atmospheric dataset."""
        times = np.arange('2023-01-01', '2023-01-02', dtype='datetime64[h]')
        lons = np.linspace(-180, 180, 20)
        lats = np.linspace(-90, 90, 10)
        
        data = np.random.random((len(times), len(lats), len(lons)))
        
        ds = xr.Dataset({
            'air_temperature': (['time', 'lat', 'lon'], data),
            'time': times,
            'lat': lats,
            'lon': lons
        })
        return ds
        
    @pytest.fixture
    def animation_plotter(self):
        """Create AnimationPlotter instance."""
        config = AnimationConfig(frame_rate=5, show_progress=False)
        with mock.patch('rompy.schism.plotting.core.BasePlotter._validate_initialization'):
            return AnimationPlotter(animation_config=config)
        
    def test_init(self):
        """Test AnimationPlotter initialization."""
        with mock.patch('rompy.schism.plotting.core.BasePlotter._validate_initialization'):
            plotter = AnimationPlotter()
        assert isinstance(plotter.animation_config, AnimationConfig)
        assert plotter._current_animation is None
        assert plotter._animation_data is None
        assert plotter._time_text is None
        
    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = AnimationConfig(frame_rate=20, repeat=False)
        with mock.patch('rompy.schism.plotting.core.BasePlotter._validate_initialization'):
            plotter = AnimationPlotter(animation_config=config)
        assert plotter.animation_config.frame_rate == 20
        assert plotter.animation_config.repeat is False
        
    @mock.patch('rompy.schism.plotting.animation.load_schism_data')
    @mock.patch('matplotlib.pyplot.subplots')
    def test_animate_boundary_data(self, mock_subplots, mock_load_data, 
                                 animation_plotter, mock_dataset):
        """Test boundary data animation."""
        # Setup mocks
        mock_load_data.return_value = mock_dataset
        mock_fig, mock_ax = mock.MagicMock(), mock.MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        # Mock animation creation
        with mock.patch('matplotlib.animation.FuncAnimation') as mock_anim:
            mock_anim_instance = mock.MagicMock()
            mock_anim.return_value = mock_anim_instance
            
            # Test animation creation
            result = animation_plotter.animate_boundary_data(
                'test_boundary.th.nc', 
                'time_series'
            )
            
            # Verify calls
            mock_load_data.assert_called_once_with('test_boundary.th.nc')
            mock_anim.assert_called_once()
            assert result == mock_anim_instance
            
    @mock.patch('rompy.schism.plotting.animation.load_schism_data')
    @mock.patch('matplotlib.pyplot.subplots')
    def test_animate_atmospheric_data(self, mock_subplots, mock_load_data,
                                    animation_plotter, mock_atmospheric_dataset):
        """Test atmospheric data animation."""
        # Setup mocks
        mock_load_data.return_value = mock_atmospheric_dataset
        mock_fig, mock_ax = mock.MagicMock(), mock.MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        # Mock animation creation
        with mock.patch('matplotlib.animation.FuncAnimation') as mock_anim:
            mock_anim_instance = mock.MagicMock()
            mock_anim.return_value = mock_anim_instance
            
            # Test animation creation
            result = animation_plotter.animate_atmospheric_data(
                'test_atmospheric.nc',
                'air',
                'air_temperature'
            )
            
            # Verify calls
            mock_load_data.assert_called_once_with('test_atmospheric.nc')
            mock_anim.assert_called_once()
            assert result == mock_anim_instance
            
    @mock.patch('rompy.schism.plotting.animation.load_schism_data')
    @mock.patch('matplotlib.pyplot.subplots')
    def test_animate_grid_data(self, mock_subplots, mock_load_data,
                             animation_plotter, mock_dataset):
        """Test grid data animation."""
        # Setup mocks
        mock_load_data.return_value = mock_dataset
        mock_fig, mock_ax = mock.MagicMock(), mock.MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        # Mock animation creation
        with mock.patch('matplotlib.animation.FuncAnimation') as mock_anim:
            mock_anim_instance = mock.MagicMock()
            mock_anim.return_value = mock_anim_instance
            
            # Test animation creation
            result = animation_plotter.animate_grid_data(
                'test_grid.nc',
                'temperature'
            )
            
            # Verify calls
            mock_load_data.assert_called_once_with('test_grid.nc')
            mock_anim.assert_called_once()
            assert result == mock_anim_instance
            
    def test_get_time_indices_full_range(self, animation_plotter, mock_dataset):
        """Test time indices extraction for full range."""
        indices = animation_plotter._get_time_indices(mock_dataset)
        expected_length = len(mock_dataset['time'])
        assert len(indices) == expected_length
        assert np.array_equal(indices, np.arange(expected_length))
        
    def test_get_time_indices_with_step(self, mock_dataset):
        """Test time indices with time step."""
        config = AnimationConfig(time_step=2)
        with mock.patch('rompy.schism.plotting.core.BasePlotter._validate_initialization'):
            plotter = AnimationPlotter(animation_config=config)
        
        indices = plotter._get_time_indices(mock_dataset)
        expected = np.arange(0, len(mock_dataset['time']), 2)
        assert np.array_equal(indices, expected)
        
    def test_get_time_indices_with_range(self, mock_dataset):
        """Test time indices with time range."""
        config = AnimationConfig(
            time_start="2023-01-01T06:00:00",
            time_end="2023-01-01T18:00:00"
        )
        with mock.patch('rompy.schism.plotting.core.BasePlotter._validate_initialization'):
            plotter = AnimationPlotter(animation_config=config)
        
        indices = plotter._get_time_indices(mock_dataset)
        # Should have subset of indices
        assert len(indices) < len(mock_dataset['time'])
        assert indices[0] >= 0
        assert indices[-1] < len(mock_dataset['time'])
        
    def test_get_common_time_indices(self, animation_plotter, mock_dataset, 
                                   mock_atmospheric_dataset):
        """Test common time indices across datasets."""
        datasets = {
            'boundary': mock_dataset,
            'atmospheric': mock_atmospheric_dataset
        }
        
        indices = animation_plotter._get_common_time_indices(datasets)
        
        # Should return valid indices
        assert len(indices) > 0
        assert all(idx >= 0 for idx in indices)
        
    def test_setup_animation_plot(self, animation_plotter):
        """Test animation plot setup."""
        fig, ax = plt.subplots()
        mock_ds = xr.Dataset({'temp': (['time'], [1, 2, 3])})
        
        # Test basic setup
        animation_plotter._setup_animation_plot(ax, mock_ds, 'temp')
        
        # Check title was set
        assert ax.get_title() == 'temp Animation'
        
        # Test with custom title
        animation_plotter._setup_animation_plot(ax, mock_ds, 'temp', title='Custom')
        assert ax.get_title() == 'Custom'
        
        plt.close(fig)
        
    def test_create_multi_panel_figure(self, animation_plotter):
        """Test multi-panel figure creation."""
        # Test vertical layout
        fig, axes = animation_plotter._create_multi_panel_figure(2, 'vertical')
        assert len(axes) == 2
        assert 'panel_0' in axes
        assert 'panel_1' in axes
        plt.close(fig)
        
        # Test horizontal layout
        fig, axes = animation_plotter._create_multi_panel_figure(3, 'horizontal')
        assert len(axes) == 3
        plt.close(fig)
        
        # Test grid layout
        fig, axes = animation_plotter._create_multi_panel_figure(4, 'grid')
        assert len(axes) == 4
        plt.close(fig)
        
    def test_animation_controls(self, animation_plotter):
        """Test animation control methods."""
        # Mock animation object
        mock_anim = mock.MagicMock()
        mock_event_source = mock.MagicMock()
        mock_anim.event_source = mock_event_source
        
        animation_plotter._current_animation = mock_anim
        
        # Test stop
        animation_plotter.stop_animation()
        mock_event_source.stop.assert_called_once()
        
        # Test pause
        animation_plotter.pause_animation()
        mock_anim.pause.assert_called_once()
        
        # Test resume
        animation_plotter.resume_animation()
        mock_anim.resume.assert_called_once()
        
    @mock.patch('rompy.schism.plotting.animation.Path')
    def test_save_animation_gif(self, mock_path, animation_plotter):
        """Test saving animation as GIF."""
        # Mock animation and path
        mock_anim = mock.MagicMock()
        mock_path_instance = mock.MagicMock()
        mock_path_instance.suffix.lower.return_value = '.gif'
        mock_path_instance.parent.mkdir = mock.MagicMock()
        mock_path.return_value = mock_path_instance
        
        # Mock PillowWriter
        with mock.patch('matplotlib.animation.PillowWriter') as mock_writer:
            writer_instance = mock.MagicMock()
            mock_writer.return_value = writer_instance
            
            animation_plotter._save_animation(mock_anim, 'test.gif')
            
            mock_writer.assert_called_once_with(fps=animation_plotter.animation_config.frame_rate)
            mock_anim.save.assert_called_once()
            
    @mock.patch('rompy.schism.plotting.animation.Path')
    def test_save_animation_mp4(self, mock_path, animation_plotter):
        """Test saving animation as MP4."""
        # Mock animation and path
        mock_anim = mock.MagicMock()
        mock_path_instance = mock.MagicMock()
        mock_path_instance.suffix.lower.return_value = '.mp4'
        mock_path_instance.parent.mkdir = mock.MagicMock()
        mock_path.return_value = mock_path_instance
        
        # Mock FFMpegWriter
        with mock.patch('matplotlib.animation.FFMpegWriter') as mock_writer:
            writer_instance = mock.MagicMock()
            mock_writer.return_value = writer_instance
            
            animation_plotter._save_animation(mock_anim, 'test.mp4')
            
            mock_writer.assert_called_once_with(
                fps=animation_plotter.animation_config.frame_rate,
                bitrate=animation_plotter.animation_config.bitrate
            )
            mock_anim.save.assert_called_once()
            
    def test_animate_frame_errors(self, animation_plotter, mock_dataset):
        """Test error handling in animation frames."""
        fig, ax = plt.subplots()
        
        # Test with invalid variable (create dataset without time_series or the variable)
        ds_no_vars = xr.Dataset({'other_var': (['node'], [1, 2, 3])})
        with mock.patch('rompy.schism.plotting.animation.load_schism_data', return_value=ds_no_vars):
            with pytest.raises(ValueError, match="Variable invalid not found"):
                animation_plotter.animate_boundary_data('test.nc', 'invalid')
            
        # Test with data without time dimension
        ds_no_time = xr.Dataset({'static': (['node'], [1, 2, 3])})
        
        with mock.patch('rompy.schism.plotting.animation.load_schism_data', return_value=ds_no_time):
            with pytest.raises(ValueError, match="Data must have time dimension"):
                animation_plotter.animate_boundary_data('test.nc', 'static')
                
        plt.close(fig)


class TestSchismPlotterAnimationIntegration:
    """Test animation integration in SchismPlotter."""
    
    @pytest.fixture
    def schism_plotter(self):
        """Create SchismPlotter instance."""
        config = AnimationConfig(frame_rate=5, show_progress=False)
        with mock.patch('rompy.schism.plotting.utils.validate_file_exists', return_value=True), \
             mock.patch('rompy.schism.plotting.core.BasePlotter._validate_initialization'), \
             mock.patch('pathlib.Path.exists', return_value=True):
            plotter = SchismPlotter(grid_file='mock_grid.gr3', animation_config=config)
        return plotter
        
    def test_animation_plotter_initialization(self, schism_plotter):
        """Test that animation plotter is properly initialized."""
        assert hasattr(schism_plotter, 'animation_plotter')
        assert isinstance(schism_plotter.animation_plotter, AnimationPlotter)
        assert schism_plotter.animation_plotter.animation_config.frame_rate == 5
        
    @mock.patch('rompy.schism.plotting.animation.AnimationPlotter.animate_boundary_data')
    def test_animate_boundary_data_delegation(self, mock_animate, schism_plotter):
        """Test boundary data animation delegation."""
        mock_animate.return_value = mock.MagicMock()
        
        result = schism_plotter.animate_boundary_data(
            'test.th.nc', 'temperature', 'output.mp4', level_idx=1
        )
        
        mock_animate.assert_called_once_with(
            'test.th.nc', 'temperature', 'output.mp4', 1
        )
        
    @mock.patch('rompy.schism.plotting.animation.AnimationPlotter.animate_atmospheric_data')
    def test_animate_atmospheric_data_delegation(self, mock_animate, schism_plotter):
        """Test atmospheric data animation delegation."""
        mock_animate.return_value = mock.MagicMock()
        
        result = schism_plotter.animate_atmospheric_data(
            'test.nc', 'air', 'temperature', 'output.gif'
        )
        
        mock_animate.assert_called_once_with(
            'test.nc', 'air', 'temperature', 'output.gif'
        )
        
    @mock.patch('rompy.schism.plotting.animation.AnimationPlotter.animate_grid_data')
    def test_animate_grid_data_delegation(self, mock_animate, schism_plotter):
        """Test grid data animation delegation."""
        mock_animate.return_value = mock.MagicMock()
        
        result = schism_plotter.animate_grid_data(
            'test.nc', 'salinity', 'output.mp4', show_grid=False
        )
        
        mock_animate.assert_called_once_with(
            'test.nc', 'salinity', 'output.mp4', False
        )
        
    @mock.patch('rompy.schism.plotting.animation.AnimationPlotter.create_multi_variable_animation')
    def test_create_multi_variable_animation_delegation(self, mock_animate, schism_plotter):
        """Test multi-variable animation delegation."""
        mock_animate.return_value = mock.MagicMock()
        
        data_files = {'temp': 'temp.nc', 'sal': 'sal.nc'}
        variables = {'temp': 'temperature', 'sal': 'salinity'}
        
        result = schism_plotter.create_multi_variable_animation(
            data_files, variables, 'output.mp4', 'vertical'
        )
        
        mock_animate.assert_called_once_with(
            data_files, variables, 'output.mp4', 'vertical'
        )
        
    @mock.patch('rompy.schism.plotting.animation.AnimationPlotter.stop_animation')
    def test_stop_animation_delegation(self, mock_stop, schism_plotter):
        """Test stop animation delegation."""
        schism_plotter.stop_animation()
        mock_stop.assert_called_once()
        
    @mock.patch('rompy.schism.plotting.animation.AnimationPlotter.pause_animation')
    def test_pause_animation_delegation(self, mock_pause, schism_plotter):
        """Test pause animation delegation."""
        schism_plotter.pause_animation()
        mock_pause.assert_called_once()
        
    @mock.patch('rompy.schism.plotting.animation.AnimationPlotter.resume_animation')
    def test_resume_animation_delegation(self, mock_resume, schism_plotter):
        """Test resume animation delegation."""
        schism_plotter.resume_animation()
        mock_resume.assert_called_once()


class TestAnimationPerformance:
    """Test animation performance and memory usage."""
    
    @pytest.fixture
    def large_dataset(self):
        """Create large dataset for performance testing."""
        # Create dataset with many time steps
        times = np.arange('2023-01-01', '2023-01-08', dtype='datetime64[h]')  # 168 hours
        nodes = 1000
        data = np.random.random((len(times), nodes))
        
        ds = xr.Dataset({
            'temperature': (['time', 'node'], data),
            'salinity': (['time', 'node'], data * 35),
            'time': times
        })
        return ds
        
    def test_memory_efficient_time_indexing(self, large_dataset):
        """Test that time indexing doesn't load entire dataset."""
        config = AnimationConfig(time_step=10)  # Every 10th time step
        with mock.patch('rompy.schism.plotting.core.BasePlotter._validate_initialization'):
            plotter = AnimationPlotter(animation_config=config)
        
        indices = plotter._get_time_indices(large_dataset)
        
        # Should be much smaller than full dataset
        assert len(indices) < len(large_dataset['time']) / 5
        assert len(indices) == len(large_dataset['time'][::10])
        
    def test_frame_rate_calculation(self):
        """Test frame rate calculations for different configurations."""
        # High frame rate
        config = AnimationConfig(frame_rate=30)
        assert config.effective_interval == 33  # ~1000/30
        
        # Low frame rate
        config = AnimationConfig(frame_rate=1)
        assert config.effective_interval == 1000
        
        # Custom interval override
        config = AnimationConfig(frame_rate=10, interval=50)
        assert config.effective_interval == 100  # Uses frame_rate calculation


class TestAnimationIntegration:
    """Integration tests for animation functionality."""
    
    def test_animation_config_inheritance(self):
        """Test that AnimationConfig properly inherits from PlotConfig."""
        config = AnimationConfig(
            figsize=(16, 10),
            cmap='coolwarm',
            frame_rate=15,
            show_time_label=False
        )
        
        # Check PlotConfig attributes
        assert config.figsize == (16, 10)
        assert config.cmap == 'coolwarm'
        
        # Check AnimationConfig attributes
        assert config.frame_rate == 15
        assert config.show_time_label is False
        
    def test_animation_figure_creation(self):
        """Test that animation creates figures properly."""
        config = AnimationConfig(figsize=(14, 8), dpi=150)
        with mock.patch('rompy.schism.plotting.core.BasePlotter._validate_initialization'):
            plotter = AnimationPlotter(animation_config=config)
        
        # Create figure through base plotter method
        fig, ax = plotter.create_figure(use_cartopy=False)
        
        # Should have animation config parameters applied
        assert plotter.animation_config.figsize == (14, 8)
        assert plotter.animation_config.dpi == 150
        
        # Clean up
        import matplotlib.pyplot as plt
        plt.close(fig)
        
    def test_error_handling_missing_data(self):
        """Test error handling for missing data files."""
        with mock.patch('rompy.schism.plotting.core.BasePlotter._validate_initialization'):
            plotter = AnimationPlotter()
        
        with mock.patch('rompy.schism.plotting.animation.load_schism_data', 
                       side_effect=FileNotFoundError("File not found")):
            with pytest.raises(FileNotFoundError):
                plotter.animate_boundary_data('nonexistent.nc', 'temperature')
                
    def test_error_handling_invalid_time_range(self):
        """Test error handling for invalid time ranges."""
        # Create test dataset
        times = np.arange('2023-07-01', '2023-07-02', dtype='datetime64[h]')
        data = np.random.random((len(times), 10))
        mock_dataset = xr.Dataset({
            'time_series': (['time', 'node'], data),
            'time': times
        })
        
        config = AnimationConfig(
            time_start="2025-01-01T00:00:00",  # Future date
            time_end="2025-01-02T00:00:00"
        )
        with mock.patch('rompy.schism.plotting.core.BasePlotter._validate_initialization'):
            plotter = AnimationPlotter(animation_config=config)
        
        # Should handle gracefully and return empty or minimal indices
        indices = plotter._get_time_indices(mock_dataset)
        # Depending on implementation, might be empty or handle edge case
        assert isinstance(indices, np.ndarray)


if __name__ == '__main__':
    pytest.main([__file__])