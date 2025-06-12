API Reference
=============

This section provides detailed API documentation for all SCHISM plotting classes and functions.

Main Interface
--------------

SchismPlotter
~~~~~~~~~~~~~

.. autoclass:: rompy.schism.plotting.SchismPlotter
   :members:
   :undoc-members:
   :show-inheritance:

Core Classes
------------

BasePlotter
~~~~~~~~~~~

.. autoclass:: rompy.schism.plotting.core.BasePlotter
   :members:
   :undoc-members:
   :show-inheritance:

PlotConfig
~~~~~~~~~~

.. autoclass:: rompy.schism.plotting.core.PlotConfig
   :members:
   :undoc-members:
   :show-inheritance:

PlotValidator
~~~~~~~~~~~~~

.. autoclass:: rompy.schism.plotting.core.PlotValidator
   :members:
   :undoc-members:
   :show-inheritance:

Specialized Plotters
--------------------

GridPlotter
~~~~~~~~~~~

.. autoclass:: rompy.schism.plotting.grid.GridPlotter
   :members:
   :undoc-members:
   :show-inheritance:

DataPlotter
~~~~~~~~~~~

.. autoclass:: rompy.schism.plotting.data.DataPlotter
   :members:
   :undoc-members:
   :show-inheritance:

OverviewPlotter
~~~~~~~~~~~~~~~

.. autoclass:: rompy.schism.plotting.overview.OverviewPlotter
   :members:
   :undoc-members:
   :show-inheritance:

Validation Classes
------------------

ModelValidator
~~~~~~~~~~~~~~

.. autoclass:: rompy.schism.plotting.validation.ModelValidator
   :members:
   :undoc-members:
   :show-inheritance:

ValidationPlotter
~~~~~~~~~~~~~~~~~

.. autoclass:: rompy.schism.plotting.validation.ValidationPlotter
   :members:
   :undoc-members:
   :show-inheritance:

ValidationResult
~~~~~~~~~~~~~~~~

.. autoclass:: rompy.schism.plotting.validation.ValidationResult
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
-----------------

File and Data Operations
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: rompy.schism.plotting.utils.validate_file_exists

.. autofunction:: rompy.schism.plotting.utils.detect_file_type

.. autofunction:: rompy.schism.plotting.utils.load_schism_data

.. autofunction:: rompy.schism.plotting.utils.get_variable_info

.. autofunction:: rompy.schism.plotting.utils.create_time_subset

Plotting Utilities
~~~~~~~~~~~~~~~~~~

.. autofunction:: rompy.schism.plotting.utils.setup_cartopy_axis

.. autofunction:: rompy.schism.plotting.utils.get_geographic_extent

.. autofunction:: rompy.schism.plotting.utils.setup_colormap

.. autofunction:: rompy.schism.plotting.utils.add_grid_overlay

.. autofunction:: rompy.schism.plotting.utils.add_boundary_overlay

.. autofunction:: rompy.schism.plotting.utils.format_scientific_notation

.. autofunction:: rompy.schism.plotting.utils.create_diverging_colormap_levels

.. autofunction:: rompy.schism.plotting.utils.save_plot

Type Definitions
----------------

The plotting module uses several type aliases for better code documentation:

.. code-block:: python

    from typing import Dict, List, Optional, Tuple, Union
    from pathlib import Path
    import matplotlib.pyplot as plt
    import numpy as np
    import xarray as xr

    # Common type aliases used throughout the plotting module
    Figure = plt.Figure
    Axes = plt.Axes
    PathLike = Union[str, Path]
    NumericArray = Union[np.ndarray, List[float]]
    DataArray = xr.DataArray
    Dataset = xr.Dataset

Configuration Parameters
------------------------

PlotConfig Parameters
~~~~~~~~~~~~~~~~~~~~~

The :class:`~rompy.schism.plotting.core.PlotConfig` class accepts the following parameters:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Parameter
     - Type
     - Description
   * - figsize
     - Tuple[float, float]
     - Figure size in inches (width, height). Default: (10, 8)
   * - dpi
     - int
     - Figure resolution in dots per inch. Default: 100
   * - colormap
     - str
     - Default colormap name. Default: 'viridis'
   * - grid_alpha
     - float
     - Transparency for grid overlays (0-1). Default: 0.5
   * - boundary_linewidth
     - float
     - Line width for boundary plots. Default: 1.5
   * - colorbar_orientation
     - str
     - Colorbar orientation ('vertical' or 'horizontal'). Default: 'vertical'
   * - save_format
     - str
     - Default file format for saving plots. Default: 'png'
   * - tight_layout
     - bool
     - Whether to use tight layout. Default: True

Validation Categories
~~~~~~~~~~~~~~~~~~~~~

The :class:`~rompy.schism.plotting.validation.ModelValidator` performs checks in these categories:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Category
     - Validation Checks
   * - Grid Quality
     - Connectivity, element quality, depth validity, geographic extent
   * - Boundaries
     - Boundary node count, boundary types, tidal constituents
   * - Forcing Data
     - Atmospheric forcing coverage, boundary data availability, time consistency
   * - Configuration
     - Parameter validity, file paths, model setup consistency
   * - Time Stepping
     - Time step stability, CFL conditions, simulation period
   * - File Integrity
     - File existence, format validation, data completeness

Return Types
------------

Most plotting functions return a tuple of ``(fig, ax)`` or ``(fig, axes)``:

**Single Plot Functions**
  Return ``Tuple[Figure, Axes]`` for functions that create single plots:
  
  - :meth:`~rompy.schism.plotting.SchismPlotter.plot_grid`
  - :meth:`~rompy.schism.plotting.SchismPlotter.plot_bathymetry`
  - :meth:`~rompy.schism.plotting.SchismPlotter.plot_boundaries`
  - :meth:`~rompy.schism.plotting.SchismPlotter.plot_atmospheric_data`

**Multi-Panel Functions**
  Return ``Tuple[Figure, Dict[str, Axes]]`` for functions that create multiple subplots:
  
  - :meth:`~rompy.schism.plotting.SchismPlotter.plot_comprehensive_overview`
  - :meth:`~rompy.schism.plotting.SchismPlotter.plot_grid_analysis_overview`
  - :meth:`~rompy.schism.plotting.SchismPlotter.plot_data_analysis_overview`
  - :meth:`~rompy.schism.plotting.SchismPlotter.plot_validation_summary`

**Validation Functions**
  Return ``List[ValidationResult]`` for validation operations:
  
  - :meth:`~rompy.schism.plotting.SchismPlotter.run_model_validation`
  - :meth:`~rompy.schism.plotting.validation.ModelValidator.run_all_validations`

Exception Handling
------------------

The plotting module defines custom exceptions and handles common error conditions:

**Common Exceptions**

.. code-block:: python

    # File not found errors
    try:
        plotter = SchismPlotter(grid_file="missing_file.gr3")
    except FileNotFoundError:
        print("Grid file not found")

    # Configuration validation errors  
    try:
        config = PlotConfig(figsize=(-1, -1))  # Invalid size
    except ValueError as e:
        print(f"Configuration error: {e}")

    # Data validation errors
    try:
        fig, ax = plotter.plot_bathymetry()
    except ValueError as e:
        print(f"Data validation error: {e}")

    # Import errors (missing optional dependencies)
    try:
        from rompy.schism.plotting.utils import setup_cartopy_axis
    except ImportError:
        print("Cartopy not available - geographic projections disabled")

Performance Notes
-----------------

**Memory Usage**
  Large grids may require significant memory. Consider using subsampling options for visualization.

**Rendering Speed**
  Complex plots with many elements can be slow to render. Use the ``subsample_factor`` parameter for faster preview plots.

**File I/O**
  Loading large NetCDF files can be time-consuming. The plotting system includes caching mechanisms for repeated operations.

**Cartopy Performance**
  Geographic projections add overhead. Disable cartopy features if not needed for better performance.

Examples
--------

**Basic Usage**

.. code-block:: python

    from rompy.schism.plotting import SchismPlotter
    
    # Initialize with configuration
    plotter = SchismPlotter(config=my_config)
    
    # Create basic plots
    fig, ax = plotter.plot_grid()
    fig, ax = plotter.plot_bathymetry()

**Advanced Configuration**

.. code-block:: python

    from rompy.schism.plotting.core import PlotConfig
    
    # Custom configuration
    config = PlotConfig(
        figsize=(16, 12),
        dpi=150,
        colormap='ocean',
        grid_alpha=0.3
    )
    
    plotter = SchismPlotter(config=my_config, plot_config=config)

**Error Handling**

.. code-block:: python

    try:
        fig, ax = plotter.plot_atmospheric_data()
    except FileNotFoundError:
        print("Atmospheric data files not found")
    except ValueError as e:
        print(f"Data validation error: {e}")

See Also
--------

* :doc:`index` - Main plotting documentation
* :doc:`examples` - Detailed usage examples
* :doc:`tutorials` - Step-by-step tutorials