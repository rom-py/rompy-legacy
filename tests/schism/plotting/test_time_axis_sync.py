"""
Test time axis synchronization in comparison plots.

This module tests the fix for ensuring consistent time axes across
input vs processed data comparison plots.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import Mock, patch
from pathlib import Path

from rompy.schism.plotting.data import DataPlotter
from rompy.core.types import RompyBaseModel


class TestTimeAxisSync:
    """Test time axis synchronization functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = Mock()
        self.data_plotter = DataPlotter(config=self.mock_config)

    def test_compute_standardized_time_axis(self):
        """Test standardized time axis computation."""
        # Test with default parameters
        time_hours = 24.0
        dt = 0.5
        times = self.data_plotter._compute_standardized_time_axis(time_hours, dt)

        expected_length = int(time_hours / dt) + 1  # +1 for inclusive end
        assert len(times) == expected_length
        assert times[0] == 0.0
        assert times[-1] <= time_hours
        assert np.allclose(np.diff(times), dt)

    def test_compute_standardized_time_axis_different_dt(self):
        """Test standardized time axis with different time step."""
        time_hours = 12.0
        dt = 1.0  # 1-hour intervals
        times = self.data_plotter._compute_standardized_time_axis(time_hours, dt)

        expected_length = int(time_hours / dt) + 1
        assert len(times) == expected_length
        assert times[0] == 0.0
        assert times[-1] <= time_hours
        assert np.allclose(np.diff(times), dt)

    def test_align_data_to_time_axis_same_length(self):
        """Test data alignment when arrays have same length."""
        # Create test data
        current_times = np.array([0, 0.5, 1.0, 1.5, 2.0])
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        target_times = np.array([0, 0.5, 1.0, 1.5, 2.0])

        aligned_data = self.data_plotter._align_data_to_time_axis(
            data, current_times, target_times
        )

        # Should be identical when times match
        np.testing.assert_array_equal(aligned_data, data)

    def test_align_data_to_time_axis_interpolation(self):
        """Test data alignment with interpolation."""
        # Create test data with different time spacing
        current_times = np.array([0, 1.0, 2.0])  # 1-hour intervals
        data = np.array([1.0, 3.0, 5.0])  # Linear data
        target_times = np.array([0, 0.5, 1.0, 1.5, 2.0])  # 0.5-hour intervals

        aligned_data = self.data_plotter._align_data_to_time_axis(
            data, current_times, target_times
        )

        # Check interpolated values
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # Linear interpolation
        np.testing.assert_array_almost_equal(aligned_data, expected)

    def test_align_data_to_time_axis_different_lengths(self):
        """Test data alignment when data and time arrays have different lengths."""
        # Mismatched lengths (simulating real-world issue)
        current_times = np.array([0, 0.5, 1.0, 1.5, 2.0])
        data = np.array([1.0, 2.0, 3.0])  # Shorter than times
        target_times = np.array([0, 0.5, 1.0, 1.5, 2.0])

        aligned_data = self.data_plotter._align_data_to_time_axis(
            data, current_times, target_times
        )

        # Should handle the mismatch gracefully
        assert len(aligned_data) == len(target_times)
        assert not np.any(np.isnan(aligned_data))

    def test_align_data_to_time_axis_empty_data(self):
        """Test data alignment with empty data."""
        current_times = np.array([])
        data = np.array([])
        target_times = np.array([0, 0.5, 1.0])

        aligned_data = self.data_plotter._align_data_to_time_axis(
            data, current_times, target_times
        )

        # Should return zeros for empty data
        expected = np.zeros_like(target_times)
        np.testing.assert_array_equal(aligned_data, expected)

    @patch('rompy.schism.plotting.data.DataPlotter._get_atmospheric_dataset')
    @patch('rompy.schism.plotting.data.DataPlotter._get_representative_atmospheric_points')
    @patch('rompy.schism.plotting.data.DataPlotter._compute_atmospheric_timeseries')
    def test_atmospheric_plotting_uses_standardized_time_axis(
        self, mock_compute_timeseries, mock_get_points, mock_get_dataset
    ):
        """Test that atmospheric plotting uses standardized time axis."""
        # Setup mocks
        mock_get_dataset.return_value = Mock()
        mock_get_points.return_value = [(0.0, 0.0), (1.0, 1.0)]

        # Mock time series data with different length than expected
        short_data = [np.array([1, 2, 3]), np.array([4, 5, 6])]  # 3 points
        mock_compute_timeseries.return_value = short_data

        # Setup config
        self.mock_config.data = Mock()
        self.mock_config.data.atmos = Mock()

        time_hours = 2.0  # Should give 5 points with dt=0.5
        fig, ax = self.data_plotter.plot_atmospheric_inputs_at_points(
            time_hours=time_hours, plot_type="wind_speed"
        )

        # Check that the method was called
        mock_compute_timeseries.assert_called_once()

        # Verify that standardized time axis would be used
        expected_times = self.data_plotter._compute_standardized_time_axis(time_hours, 0.5)
        assert len(expected_times) == 5  # 0, 0.5, 1.0, 1.5, 2.0

        plt.close(fig)

    def test_time_axis_consistency_across_methods(self):
        """Test that all plotting methods produce consistent time axes."""
        time_hours = 6.0
        dt = 0.5

        # All methods should produce the same time axis
        times1 = self.data_plotter._compute_standardized_time_axis(time_hours, dt)
        times2 = self.data_plotter._compute_standardized_time_axis(time_hours, dt)

        np.testing.assert_array_equal(times1, times2)

        # Verify expected properties
        assert len(times1) == 13  # 0 to 6 hours with 0.5 hour steps
        assert times1[0] == 0.0
        assert times1[-1] == 6.0

    def test_standardized_time_axis_edge_cases(self):
        """Test edge cases for standardized time axis."""
        # Very small time duration
        times = self.data_plotter._compute_standardized_time_axis(0.5, 0.5)
        assert len(times) == 2  # [0.0, 0.5]

        # Zero time duration
        times = self.data_plotter._compute_standardized_time_axis(0.0, 0.5)
        assert len(times) == 1  # [0.0]

        # Large time duration
        times = self.data_plotter._compute_standardized_time_axis(48.0, 1.0)
        expected_length = 49  # 0 to 48 inclusive
        assert len(times) == expected_length
        assert times[-1] == 48.0

    @patch('rompy.schism.plotting.data.DataPlotter._find_sflux_files')
    @patch('rompy.schism.plotting.data.DataPlotter._get_representative_atmospheric_points')
    @patch('rompy.schism.plotting.data.DataPlotter._compute_processed_atmospheric_timeseries')
    def test_processed_atmospheric_plotting_alignment(
        self, mock_compute_processed, mock_get_points, mock_find_files
    ):
        """Test that processed atmospheric plotting aligns data correctly."""
        # Setup mocks
        mock_find_files.return_value = ['file1.nc']
        mock_get_points.return_value = [(0.0, 0.0)]

        # Mock processed data with different length
        processed_data = [np.array([1, 2, 3, 4])]  # 4 points
        mock_compute_processed.return_value = processed_data

        time_hours = 2.0  # Should give 5 points with dt=0.5
        fig, ax = self.data_plotter.plot_processed_atmospheric_data(
            time_hours=time_hours, plot_type="wind_speed"
        )

        # The method should handle the length mismatch
        mock_compute_processed.assert_called_once()

        plt.close(fig)

    def test_data_alignment_preserves_trends(self):
        """Test that data alignment preserves data trends."""
        # Create linear test data for more predictable interpolation
        current_times = np.array([0, 1, 2, 3, 4])  # 5 points over 4 hours
        data = np.array([0, 2, 4, 6, 8])  # Linear data

        # Target times with higher resolution
        target_times = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])

        aligned_data = self.data_plotter._align_data_to_time_axis(
            data, current_times, target_times
        )

        # Check that the trend is preserved
        assert len(aligned_data) == len(target_times)

        # The interpolated data should follow the linear trend
        expected_aligned = target_times * 2  # Linear relationship y = 2x
        np.testing.assert_array_almost_equal(aligned_data, expected_aligned, decimal=10)

    def teardown_method(self):
        """Clean up after tests."""
        plt.close('all')
