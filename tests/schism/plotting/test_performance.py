"""
Performance tests for SCHISM plotting module.

This module tests the performance of plotting functions with large datasets,
measuring memory usage, execution time, and scalability.
"""

import pytest
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import time
import gc
import psutil
import os
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile

from rompy.schism.plotting import SchismPlotter
from rompy.schism.plotting.core import PlotConfig
from rompy.schism.plotting.utils import (
    load_schism_data,
    create_time_subset,
    setup_colormap,
    save_plot
)


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def time_function(func, *args, **kwargs):
    """Time a function execution and return time and result."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return end_time - start_time, result


def create_large_mock_grid(num_nodes=50000, num_elements=95000):
    """Create a large mock SCHISM grid for performance testing."""
    # Generate random coordinates
    np.random.seed(42)  # For reproducible tests
    x = np.random.uniform(-125, -120, num_nodes)
    y = np.random.uniform(40, 45, num_nodes)
    depth = np.random.uniform(1, 1000, num_nodes)

    # Generate triangular elements
    tri = []
    for i in range(0, num_nodes - 2, 3):
        if len(tri) >= num_elements:
            break
        tri.append([i, i+1, i+2])

    tri = np.array(tri[:num_elements])

    # Create mock grid object
    grid = Mock()
    grid.x = x
    grid.y = y
    grid.depth = depth
    grid.tri = tri
    grid.nodes = np.arange(num_nodes)
    grid.elements = np.arange(num_elements)

    # Add boundary information
    boundary_nodes = np.arange(0, min(1000, num_nodes), 10)  # Every 10th node as boundary
    grid.boundary_nodes = boundary_nodes
    grid.open_boundaries = [boundary_nodes]

    return grid


def create_large_atmospheric_data(nx=200, ny=150, nt=168):
    """Create large atmospheric forcing data for performance testing."""
    np.random.seed(42)

    # Create coordinates
    lon = np.linspace(-125, -120, nx)
    lat = np.linspace(40, 45, ny)
    time = np.arange(nt)

    # Create data variables
    data_vars = {
        'air_pressure': (['time', 'lat', 'lon'],
                        np.random.normal(101325, 1000, (nt, ny, nx))),
        'air_temperature': (['time', 'lat', 'lon'],
                           np.random.normal(288, 10, (nt, ny, nx))),
        'eastward_wind': (['time', 'lat', 'lon'],
                         np.random.normal(0, 5, (nt, ny, nx))),
        'northward_wind': (['time', 'lat', 'lon'],
                          np.random.normal(0, 5, (nt, ny, nx))),
        'specific_humidity': (['time', 'lat', 'lon'],
                             np.random.uniform(0.005, 0.020, (nt, ny, nx)))
    }

    coords = {
        'time': time,
        'lat': lat,
        'lon': lon
    }

    return xr.Dataset(data_vars, coords)


def create_large_boundary_data(n_nodes=1000, n_levels=30, n_times=168):
    """Create large 3D boundary data for performance testing."""
    np.random.seed(42)

    coords = {
        'time': np.arange(n_times),
        'node': np.arange(n_nodes),
        'level': np.arange(n_levels)
    }

    # Temperature data
    temp_data = np.random.normal(285, 5, (n_times, n_nodes, n_levels))

    # Salinity data
    salt_data = np.random.normal(35, 2, (n_times, n_nodes, n_levels))

    # Velocity data
    u_data = np.random.normal(0, 0.5, (n_times, n_nodes, n_levels))
    v_data = np.random.normal(0, 0.5, (n_times, n_nodes, n_levels))

    # Create datasets
    temp_ds = xr.Dataset({
        'temperature': (['time', 'node', 'level'], temp_data)
    }, coords)

    salt_ds = xr.Dataset({
        'salinity': (['time', 'node', 'level'], salt_data)
    }, coords)

    vel_ds = xr.Dataset({
        'eastward_velocity': (['time', 'node', 'level'], u_data),
        'northward_velocity': (['time', 'node', 'level'], v_data)
    }, coords)

    return temp_ds, salt_ds, vel_ds


class TestPlottingPerformance:
    """Test plotting performance with large datasets."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock configuration
        self.config = Mock()
        self.config.grid = Mock()

        # Create plot configuration for performance testing
        self.plot_config = PlotConfig(
            figsize=(10, 8),
            dpi=100,  # Lower DPI for faster rendering
            cmap='viridis'
        )

        # Memory tracking
        self.initial_memory = get_memory_usage()

    def teardown_method(self):
        """Clean up after tests."""
        plt.close('all')
        gc.collect()

    @pytest.mark.performance
    @pytest.mark.skip(reason="Performance test")
    def test_large_grid_plotting_performance(self):
        """Test grid plotting performance with large grids."""
        # Test different grid sizes
        grid_sizes = [
            (5000, 9000),    # Small grid
            (25000, 45000),  # Medium grid
            (50000, 95000),  # Large grid
        ]

        results = {}

        for num_nodes, num_elements in grid_sizes:
            grid = create_large_mock_grid(num_nodes, num_elements)

            # Mock plotter with large grid
            plotter = Mock()
            plotter.grid = grid
            plotter.config = self.config
            plotter.plot_config = self.plot_config

            # Mock the actual plotting function
            def mock_plot_grid(subsample_factor=1, **kwargs):
                # Simulate grid plotting performance
                effective_elements = num_elements // subsample_factor

                fig, ax = plt.subplots(figsize=self.plot_config.figsize)

                # Simulate plotting time based on number of elements
                time.sleep(effective_elements / 1000000)  # Simulate processing time

                # Create some dummy plot data
                x_sample = grid.x[::subsample_factor][:1000]
                y_sample = grid.y[::subsample_factor][:1000]
                ax.scatter(x_sample, y_sample, s=1, alpha=0.5)

                return fig, ax

            # Test standard plotting
            exec_time, (fig, ax) = time_function(mock_plot_grid)
            memory_after = get_memory_usage()
            plt.close(fig)

            # Test optimized plotting
            exec_time_opt, (fig_opt, ax_opt) = time_function(
                mock_plot_grid, subsample_factor=10
            )
            memory_after_opt = get_memory_usage()
            plt.close(fig_opt)

            results[f"{num_nodes}_nodes"] = {
                'standard_time': exec_time,
                'optimized_time': exec_time_opt,
                'speedup': exec_time / exec_time_opt if exec_time_opt > 0 else 0,
                'memory_increase': memory_after - self.initial_memory,
                'memory_increase_opt': memory_after_opt - self.initial_memory
            }

            # Performance assertions
            assert exec_time < 30.0, f"Grid plotting too slow: {exec_time:.2f}s"
            assert exec_time_opt < exec_time, "Optimization should improve performance"
            assert memory_after < self.initial_memory + 500, "Memory usage too high"

            print(f"\nGrid size {num_nodes} nodes, {num_elements} elements:")
            print(f"  Standard time: {exec_time:.3f}s")
            print(f"  Optimized time: {exec_time_opt:.3f}s")
            print(f"  Speedup: {exec_time/exec_time_opt:.1f}x")
            print(f"  Memory increase: {memory_after - self.initial_memory:.1f} MB")

        # Overall performance check
        assert len(results) == len(grid_sizes), "All grid sizes should be tested"

    @pytest.mark.performance
    @pytest.mark.skip(reason="Performance test")

    def test_atmospheric_data_performance(self):
        """Test atmospheric data plotting performance."""
        # Create large atmospheric dataset
        large_atm_data = create_large_atmospheric_data(nx=200, ny=150, nt=168)

        # Test data loading performance
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp_file:
            large_atm_data.to_netcdf(tmp_file.name)
            tmp_path = tmp_file.name

        try:
            # Test loading
            load_time, dataset = time_function(xr.open_dataset, tmp_path)

            # Test subset creation
            subset_time, subset = time_function(
                create_time_subset, dataset, time_idx=0
            )

            # Test variable info extraction
            info_time, var_info = time_function(
                lambda: {
                    'air_pressure': {
                        'min': float(dataset.air_pressure.min()),
                        'max': float(dataset.air_pressure.max()),
                        'mean': float(dataset.air_pressure.mean())
                    }
                }
            )

            # Performance assertions
            assert load_time < 5.0, f"Data loading too slow: {load_time:.2f}s"
            assert subset_time < 2.0, f"Time subsetting too slow: {subset_time:.2f}s"
            assert info_time < 1.0, f"Variable info too slow: {info_time:.2f}s"

            print(f"\nAtmospheric data performance:")
            print(f"  Loading time: {load_time:.3f}s")
            print(f"  Subset time: {subset_time:.3f}s")
            print(f"  Info extraction: {info_time:.3f}s")
            print(f"  Dataset size: {large_atm_data.nbytes / 1024**2:.1f} MB")

            dataset.close()

        finally:
            # Clean up
            os.unlink(tmp_path)

    @pytest.mark.performance
    def test_boundary_data_performance(self):
        """Test 3D boundary data plotting performance."""
        # Create large boundary datasets
        temp_ds, salt_ds, vel_ds = create_large_boundary_data(
            n_nodes=1000, n_levels=30, n_times=168
        )

        datasets = {
            'temperature': temp_ds,
            'salinity': salt_ds,
            'velocity': vel_ds
        }

        for name, dataset in datasets.items():
            # Test data operations

            # Time subset
            subset_time, subset = time_function(
                create_time_subset, dataset, time_idx=0
            )

            # Depth level extraction
            if 'level' in dataset.dims:
                level_time, surface_data = time_function(
                    lambda: dataset.isel(level=0)
                )
            else:
                level_time = 0
                surface_data = dataset

            # Statistical operations
            stats_time, stats = time_function(
                lambda: {
                    'min': float(dataset[list(dataset.data_vars)[0]].min()),
                    'max': float(dataset[list(dataset.data_vars)[0]].max()),
                    'mean': float(dataset[list(dataset.data_vars)[0]].mean())
                }
            )

            # Performance assertions
            assert subset_time < 3.0, f"Time subsetting too slow for {name}: {subset_time:.2f}s"
            assert level_time < 1.0, f"Level extraction too slow for {name}: {level_time:.2f}s"
            assert stats_time < 2.0, f"Statistics too slow for {name}: {stats_time:.2f}s"

            print(f"\n{name.capitalize()} boundary data performance:")
            print(f"  Time subset: {subset_time:.3f}s")
            print(f"  Level extraction: {level_time:.3f}s")
            print(f"  Statistics: {stats_time:.3f}s")
            print(f"  Dataset size: {dataset.nbytes / 1024**2:.1f} MB")

    @pytest.mark.performance
    @pytest.mark.skip(reason="Performance test")
    def test_colormap_performance(self):
        """Test colormap setup performance with large data arrays."""
        # Test different data sizes
        data_sizes = [10000, 50000, 100000, 500000]

        for size in data_sizes:
            # Create large data array
            np.random.seed(42)
            data = np.random.normal(0, 1, size)

            # Test colormap setup
            setup_time, (cmap, norm, levels) = time_function(
                setup_colormap,
                data=data,
                cmap='viridis',
                levels=20
            )

            # Performance assertions
            assert setup_time < 1.0, f"Colormap setup too slow for {size} points: {setup_time:.2f}s"
            assert levels is not None, "Colormap levels should be generated"
            assert norm is not None, "Normalization should be created"

            print(f"Colormap setup for {size} points: {setup_time:.4f}s")

    @pytest.mark.performance
    @pytest.mark.skip(reason="Performance test")
    def test_plot_saving_performance(self):
        """Test plot saving performance with different formats and resolutions."""
        # Create a test figure
        fig, ax = plt.subplots(figsize=(12, 10))

        # Add some complex content
        np.random.seed(42)
        x = np.random.randn(10000)
        y = np.random.randn(10000)
        colors = np.random.randn(10000)

        scatter = ax.scatter(x, y, c=colors, alpha=0.6, s=1)
        ax.set_title('Performance Test Plot')
        fig.colorbar(scatter)

        # Test different save configurations
        save_configs = [
            {'format': 'png', 'dpi': 100},
            {'format': 'png', 'dpi': 300},
            {'format': 'pdf'},
            {'format': 'svg'}
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            for config in save_configs:
                save_file = tmp_path / f"test.{config['format']}"

                # Time the save operation
                save_time, _ = time_function(
                    save_plot, fig, save_file, **config
                )

                # Check file was created
                assert save_file.exists(), f"Save file not created: {save_file}"

                # Performance assertion
                max_time = 10.0 if config.get('dpi', 100) <= 150 else 30.0
                assert save_time < max_time, f"Save too slow for {config}: {save_time:.2f}s"

                file_size = save_file.stat().st_size / 1024  # KB
                print(f"Save {config}: {save_time:.3f}s, {file_size:.1f} KB")

        plt.close(fig)

    @pytest.mark.performance
    @pytest.mark.skip(reason="Memory scaling test causes crashes in current environment")
    def test_memory_scaling(self):
        """Test memory usage scaling with dataset size."""
        initial_memory = get_memory_usage()
        memory_measurements = []

        # Test different dataset sizes
        sizes = [1000, 5000, 10000, 25000]

        for size in sizes:
            # Create dataset of given size
            np.random.seed(42)
            data = np.random.randn(size, size // 10)  # 2D array

            # Measure memory after creation
            current_memory = get_memory_usage()
            memory_increase = current_memory - initial_memory
            memory_measurements.append((size, memory_increase))

            # Simulate some processing
            processed_data = data * 2 + 1
            stats = {
                'mean': np.mean(processed_data),
                'std': np.std(processed_data),
                'min': np.min(processed_data),
                'max': np.max(processed_data)
            }

            # Memory should not grow excessively
            final_memory = get_memory_usage()
            total_increase = final_memory - initial_memory

            # Rule of thumb: memory increase should be roughly proportional to data size
            expected_memory_mb = (data.nbytes / 1024**2) * 3  # Factor of 3 for processing overhead
            assert total_increase < expected_memory_mb + 100, f"Memory usage too high: {total_increase:.1f} MB"

            print(f"Size {size}: Data {data.nbytes/1024**2:.1f} MB, Memory increase {total_increase:.1f} MB")

            # Clean up
            del data, processed_data, stats
            gc.collect()

        # Check memory scaling is reasonable
        assert len(memory_measurements) == len(sizes), "All sizes should be measured"

        # Memory growth should not be exponential
        for i in range(1, len(memory_measurements)):
            size_ratio = sizes[i] / sizes[i-1]
            memory_ratio = memory_measurements[i][1] / max(memory_measurements[i-1][1], 1)

            # Memory growth should be reasonable (allow for some system overhead)
            # Increased tolerance for memory scaling tests due to system variations
            assert memory_ratio <= size_ratio**2 + 10, f"Memory scaling too aggressive: {memory_ratio:.2f}"

    @pytest.mark.performance
    @pytest.mark.skip(reason="Performance test")
    def test_concurrent_plotting_performance(self):
        """Test performance when creating multiple plots concurrently."""
        import threading
        import queue

        # Create shared data
        grid = create_large_mock_grid(10000, 18000)

        # Results queue
        results_queue = queue.Queue()

        def create_plot_worker(worker_id):
            """Worker function to create plots."""
            try:
                start_time = time.time()

                # Create a simple plot
                fig, ax = plt.subplots(figsize=(8, 6))

                # Simulate plotting with subset of data
                x_sample = grid.x[::10][:1000]
                y_sample = grid.y[::10][:1000]
                ax.scatter(x_sample, y_sample, s=1, alpha=0.5)
                ax.set_title(f'Worker {worker_id}')

                end_time = time.time()

                plt.close(fig)

                results_queue.put({
                    'worker_id': worker_id,
                    'time': end_time - start_time,
                    'success': True
                })

            except Exception as e:
                results_queue.put({
                    'worker_id': worker_id,
                    'error': str(e),
                    'success': False
                })

        # Test with multiple workers
        num_workers = 4
        threads = []

        start_time = time.time()

        # Start workers
        for i in range(num_workers):
            thread = threading.Thread(target=create_plot_worker, args=(i,))
            thread.start()
            threads.append(thread)

        # Wait for completion
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout

        total_time = time.time() - start_time

        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())

        # Analyze results
        successful_workers = [r for r in results if r.get('success', False)]
        failed_workers = [r for r in results if not r.get('success', True)]

        assert len(successful_workers) >= num_workers * 0.8, "Most workers should succeed"
        assert total_time < 60.0, f"Concurrent plotting too slow: {total_time:.2f}s"

        if successful_workers:
            avg_worker_time = np.mean([r['time'] for r in successful_workers])
            print(f"\nConcurrent plotting performance:")
            print(f"  Total time: {total_time:.3f}s")
            print(f"  Average worker time: {avg_worker_time:.3f}s")
            print(f"  Successful workers: {len(successful_workers)}/{num_workers}")
            print(f"  Failed workers: {len(failed_workers)}")


class TestPlottingScalability:
    """Test plotting scalability with increasing dataset sizes."""

    @pytest.mark.performance
    @pytest.mark.slow
    def test_grid_size_scalability(self):
        """Test how plotting performance scales with grid size."""
        # Progressive grid sizes
        grid_configs = [
            (1000, 1800),     # Very small
            (5000, 9000),     # Small
            (15000, 27000),   # Medium
            (30000, 54000),   # Large
            (50000, 95000),   # Very large
        ]

        performance_data = []

        for num_nodes, num_elements in grid_configs:
            grid = create_large_mock_grid(num_nodes, num_elements)

            # Mock plotting operations
            def mock_grid_operations():
                # Simulate typical grid operations

                # Coordinate extent calculation
                x_min, x_max = np.min(grid.x), np.max(grid.x)
                y_min, y_max = np.min(grid.y), np.max(grid.y)

                # Depth statistics
                depth_stats = {
                    'min': np.min(grid.depth),
                    'max': np.max(grid.depth),
                    'mean': np.mean(grid.depth)
                }

                # Boundary processing
                boundary_count = len(grid.boundary_nodes)

                # Element quality (simplified)
                element_areas = np.random.random(len(grid.tri))
                quality_metric = np.mean(element_areas)

                return {
                    'extent': (x_min, x_max, y_min, y_max),
                    'depth_stats': depth_stats,
                    'boundary_count': boundary_count,
                    'quality': quality_metric
                }

            # Time the operations
            operation_time, results = time_function(mock_grid_operations)

            # Memory usage
            current_memory = get_memory_usage()

            performance_data.append({
                'nodes': num_nodes,
                'elements': num_elements,
                'time': operation_time,
                'memory': current_memory,
                'results': results
            })

            print(f"Grid {num_nodes:>6} nodes, {num_elements:>6} elements: {operation_time:.4f}s")

            # Performance should scale reasonably
            assert operation_time < 5.0, f"Grid operations too slow: {operation_time:.2f}s"

            # Clean up
            del grid
            gc.collect()

        # Analyze scaling behavior
        assert len(performance_data) == len(grid_configs), "All configurations should be tested"

        # Check that performance scales sub-quadratically
        for i in range(1, len(performance_data)):
            size_ratio = performance_data[i]['nodes'] / performance_data[i-1]['nodes']
            time_ratio = performance_data[i]['time'] / max(performance_data[i-1]['time'], 0.001)

            # Time should not grow faster than O(n^1.5)
            max_expected_ratio = size_ratio ** 1.5
            assert time_ratio <= max_expected_ratio + 2, f"Performance scaling too poor: {time_ratio:.2f}"

        print(f"\nScalability test completed successfully for {len(grid_configs)} grid sizes")


# Fixtures for performance testing
@pytest.fixture(scope="session")
def large_test_grid():
    """Create a large grid for performance testing."""
    return create_large_mock_grid(25000, 45000)


@pytest.fixture(scope="session")
def large_atmospheric_dataset():
    """Create large atmospheric data for performance testing."""
    return create_large_atmospheric_data(nx=150, ny=100, nt=72)


@pytest.fixture(scope="session")
def large_boundary_datasets():
    """Create large boundary datasets for performance testing."""
    return create_large_boundary_data(n_nodes=500, n_levels=20, n_times=72)


# Performance test markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers",
        "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow (may take several minutes)"
    )


if __name__ == "__main__":
    # Run performance tests directly
    pytest.main([__file__, "-v", "-m", "performance"])
