SCHISM Plotting Documentation
=============================

The SCHISM plotting module provides comprehensive visualization capabilities for SCHISM model inputs, grid structures, boundary conditions, atmospheric forcing, and model validation. This documentation covers the complete API and usage examples.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   overview
   api_reference
   examples
   tutorials

Quick Start
-----------

The SCHISM plotting system is designed around a unified interface that can create various types of plots:

.. code-block:: python

   from rompy.schism.plotting import SchismPlotter
   
   # Initialize with a SCHISM configuration
   plotter = SchismPlotter(config=schism_config)
   
   # Create basic plots
   fig, ax = plotter.plot_grid()
   fig, ax = plotter.plot_bathymetry()
   fig, ax = plotter.plot_boundaries()
   
   # Create comprehensive overview
   fig, axes = plotter.plot_comprehensive_overview()

Key Features
------------

Grid Visualization
~~~~~~~~~~~~~~~~~~

* **Grid Structure**: Visualize SCHISM unstructured grid with nodes and elements
* **Bathymetry**: Plot depth contours and topography
* **Grid Quality**: Assess element quality, aspect ratios, and connectivity
* **Boundaries**: Display open boundaries, land boundaries, and island boundaries

Data Visualization
~~~~~~~~~~~~~~~~~~

* **Atmospheric Forcing**: Wind vectors, pressure fields, temperature, humidity
* **Boundary Data**: 3D temperature, salinity, and velocity boundary conditions
* **Tidal Data**: Tidal constituents and harmonic analysis
* **Time Series**: Temporal evolution of forcing data

Model Validation
~~~~~~~~~~~~~~~~

* **Setup Validation**: 20+ validation checks across grid, boundaries, forcing
* **Quality Assessment**: Grid quality metrics and data integrity checks
* **Validation Reports**: Comprehensive validation summaries with radar charts

Advanced Features
~~~~~~~~~~~~~~~~~

* **Multi-panel Overviews**: Comprehensive 8-panel model summaries
* **Real-time Validation**: Interactive validation with detailed feedback
* **Export Capabilities**: High-resolution plot export with customizable formats
* **Cartopy Integration**: Geographic projections and coordinate systems

Main Classes
------------

.. autosummary::
   :toctree: _autosummary

   ~rompy.schism.plotting.SchismPlotter
   ~rompy.schism.plotting.core.BasePlotter
   ~rompy.schism.plotting.core.PlotConfig
   ~rompy.schism.plotting.grid.GridPlotter
   ~rompy.schism.plotting.data.DataPlotter
   ~rompy.schism.plotting.overview.OverviewPlotter
   ~rompy.schism.plotting.validation.ModelValidator
   ~rompy.schism.plotting.validation.ValidationPlotter

Core Configuration
------------------

.. autoclass:: rompy.schism.plotting.core.PlotConfig
   :members:
   :undoc-members:

Main Interface
--------------

.. autoclass:: rompy.schism.plotting.SchismPlotter
   :members:
   :undoc-members:

Usage Examples
--------------

Basic Grid Plotting
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rompy.schism.plotting import SchismPlotter
   from rompy.model import ModelRun
   
   # Load SCHISM configuration
   config = ModelRun.from_yaml("schism_config.yaml")
   
   # Initialize plotter
   plotter = SchismPlotter(config=config.config)
   
   # Create grid plot
   fig, ax = plotter.plot_grid(
       figsize=(12, 10),
       show_elements=True,
       show_boundaries=True
   )
   fig.show()

Bathymetry Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Plot bathymetry with custom colormap
   fig, ax = plotter.plot_bathymetry(
       figsize=(14, 10),
       colormap='ocean',
       contour_levels=20,
       show_contours=True
   )
   fig.show()

Atmospheric Forcing
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Plot atmospheric data overview
   fig, ax = plotter.plot_atmospheric_data(
       figsize=(16, 12),
       variables=['wind_speed', 'air_pressure'],
       time_range=(0, 24)  # First 24 hours
   )
   fig.show()

Comprehensive Overview
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create comprehensive 8-panel overview
   fig, axes = plotter.plot_comprehensive_overview(
       figsize=(20, 16),
       include_validation=True,
       include_quality_metrics=True
   )
   
   # Save high-resolution plot
   fig.savefig('model_overview.png', dpi=300, bbox_inches='tight')

Model Validation
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Run model validation
   validation_results = plotter.run_model_validation()
   
   # Create validation summary plot
   fig, axes = plotter.plot_validation_summary(
       figsize=(14, 10)
   )
   
   # Print validation summary
   for result in validation_results:
       print(f"{result.check_name}: {result.status} - {result.message}")

Working with Real Data
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Initialize with grid file for existing model
   plotter = SchismPlotter(grid_file="hgrid.gr3")
   
   # Plot specific boundary data file
   fig, ax = plotter.plot_boundary_data(
       "SAL_3D.th.nc", 
       variable="salinity",
       time_index=0
   )
   
   # Create data analysis overview
   fig, axes = plotter.plot_data_analysis_overview()

Configuration Options
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rompy.schism.plotting.core import PlotConfig
   
   # Custom plot configuration
   plot_config = PlotConfig(
       figsize=(16, 12),
       dpi=150,
       colormap='viridis',
       grid_alpha=0.3,
       boundary_linewidth=2.0,
       colorbar_orientation='horizontal'
   )
   
   # Use custom configuration
   plotter = SchismPlotter(
       config=schism_config,
       plot_config=plot_config
   )

Advanced Features
-----------------

Custom Plotting Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~

The plotting system is extensible. You can access individual plotter components:

.. code-block:: python

   # Access individual plotters
   grid_plotter = plotter.grid_plotter
   data_plotter = plotter.data_plotter
   overview_plotter = plotter.overview_plotter
   
   # Use specialized plotting methods
   fig, ax = grid_plotter.plot_grid_quality_metrics()
   fig, ax = data_plotter.plot_3d_boundary_profiles("TEM_3D.th.nc")

Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~

For large datasets, consider these performance optimizations:

.. code-block:: python

   # Subsample large grids for faster plotting
   fig, ax = plotter.plot_grid(
       subsample_factor=10,  # Plot every 10th element
       simplify_boundaries=True
   )
   
   # Time subset for atmospheric data
   fig, ax = plotter.plot_atmospheric_data(
       time_range=(0, 48),  # First 48 hours only
       spatial_subsample=5  # Every 5th grid point
   )

Error Handling
~~~~~~~~~~~~~~

The plotting system includes comprehensive error handling:

.. code-block:: python

   try:
       fig, ax = plotter.plot_bathymetry()
   except FileNotFoundError:
       print("Grid file not found")
   except ValueError as e:
       print(f"Configuration error: {e}")
   except Exception as e:
       print(f"Plotting error: {e}")

Integration with Other Tools
----------------------------

Jupyter Notebooks
~~~~~~~~~~~~~~~~~~

The plotting system works seamlessly in Jupyter notebooks:

.. code-block:: python

   # Enable inline plotting
   %matplotlib inline
   
   # Create interactive plots
   fig, ax = plotter.plot_grid()
   # Plot displays automatically in notebook

Saving and Export
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Save individual plots
   fig, ax = plotter.plot_bathymetry()
   fig.savefig('bathymetry.png', dpi=300, bbox_inches='tight')
   
   # Batch export multiple plots
   plot_dir = Path("plots")
   plot_dir.mkdir(exist_ok=True)
   
   # Grid plots
   fig, ax = plotter.plot_grid()
   fig.savefig(plot_dir / "grid.png", dpi=150)
   
   # Boundary plots
   fig, ax = plotter.plot_boundaries()
   fig.savefig(plot_dir / "boundaries.png", dpi=150)

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Import Errors**

.. code-block:: python

   # If cartopy is not available
   import warnings
   warnings.filterwarnings('ignore', message='Cartopy not available')

**Memory Issues with Large Grids**

.. code-block:: python

   # Use subsampling for very large grids
   fig, ax = plotter.plot_grid(subsample_factor=20)

**Missing Data Files**

.. code-block:: python

   # Check file existence before plotting
   from rompy.schism.plotting.utils import validate_file_exists
   
   if validate_file_exists("boundary_data.nc"):
       fig, ax = plotter.plot_boundary_data("boundary_data.nc")
   else:
       print("Boundary data file not found")

Getting Help
~~~~~~~~~~~~

* Check the examples in ``examples/schism/``
* Review the comprehensive test suite in ``tests/schism/plotting/``
* See the real data demonstration in ``examples/schism/schism_real_data_plotting_demo.py``

Next Steps
----------

* :doc:`api_reference` - Complete API documentation
* :doc:`examples` - Detailed examples and use cases  
* :doc:`tutorials` - Step-by-step tutorials for common workflows