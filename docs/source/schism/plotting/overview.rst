SCHISM Plotting Overview
========================

The SCHISM plotting module provides a comprehensive suite of visualization tools for SCHISM (Semi-implicit Cross-scale Hydroscience Integrated System Model) input data, grid structures, boundary conditions, atmospheric forcing, and model validation.

Purpose and Scope
-----------------

This module is designed to assist SCHISM modelers in:

* **Model Setup Visualization**: Examining grid structure, bathymetry, and boundaries before running simulations
* **Input Data Validation**: Verifying boundary conditions, atmospheric forcing, and other input data
* **Quality Assessment**: Evaluating grid quality, data integrity, and model configuration
* **Model Documentation**: Creating publication-quality figures and animations for reports and presentations
* **Workflow Integration**: Seamless integration with ROMPY model setup workflows

Key Components
--------------

Core Plotting System
~~~~~~~~~~~~~~~~~~~~

The plotting system is built around a unified interface with specialized components:

* **SchismPlotter**: Main interface for all plotting functionality
* **PlotConfig**: Pydantic-based configuration system for consistent styling
* **BasePlotter**: Abstract base class providing common functionality

Specialized Plotters
~~~~~~~~~~~~~~~~~~~~

* **GridPlotter**: Grid structure, bathymetry, boundaries, and quality metrics
* **DataPlotter**: Boundary data, atmospheric forcing, and temporal analysis
* **OverviewPlotter**: Multi-panel comprehensive overviews
* **ValidationPlotter**: Model validation summaries and quality assessment
* **AnimationPlotter**: Time series animations and dynamic visualizations

Visualization Types
-------------------

Static Visualizations
~~~~~~~~~~~~~~~~~~~~

Grid and Bathymetry
  * Unstructured grid visualization with elements and nodes
  * Bathymetry contours and depth coloring
  * Grid quality metrics and element statistics
  * Boundary identification and classification

Boundary Conditions
  * 3D temperature, salinity, and velocity boundary data
  * Spatial distribution of boundary values
  * Time series analysis at boundary nodes
  * Vertical profile visualization

Atmospheric Forcing
  * Wind field vectors and magnitude
  * Pressure, temperature, and humidity fields
  * Precipitation and radiation data
  * Temporal evolution and spatial patterns

Model Validation
  * Grid quality assessment with metrics
  * Data integrity checks and summaries
  * Configuration validation results
  * Comprehensive validation reports

Dynamic Visualizations
~~~~~~~~~~~~~~~~~~~~~

Time Series Animations
  * Boundary data evolution over time
  * Atmospheric forcing progression
  * Grid-based temporal data
  * Multi-variable synchronized animations

Interactive Features
  * Animation playback controls
  * Configurable frame rates and quality
  * Multiple output formats (MP4, GIF)
  * Geographic projections and overlays

Architecture and Design
----------------------

Modular Structure
~~~~~~~~~~~~~~~~

The plotting system follows a modular architecture:

.. code-block:: text

   rompy.schism.plotting/
   ├── __init__.py              # Main interface
   ├── core.py                  # Base classes and configuration
   ├── grid.py                  # Grid-specific plotting
   ├── data.py                  # Data visualization
   ├── overview.py              # Multi-panel overviews
   ├── validation.py            # Validation and quality assessment
   ├── animation.py             # Time series animations
   ├── utils.py                 # Utility functions
   └── validation.py            # Input validation

Configuration System
~~~~~~~~~~~~~~~~~~~

The plotting system uses Pydantic models for configuration:

* **Consistent Parameters**: Standardized styling across all plots
* **Validation**: Input validation and type checking
* **Extensibility**: Easy addition of new parameters
* **Documentation**: Self-documenting configuration options

Integration Points
~~~~~~~~~~~~~~~~~

* **ROMPY Workflows**: Seamless integration with model setup processes
* **Cartopy**: Geographic projections and coordinate systems
* **Matplotlib**: High-quality static and animated visualizations
* **XArray**: Efficient handling of multidimensional data

Usage Patterns
--------------

Quick Start Workflow
~~~~~~~~~~~~~~~~~~~~

1. **Initialize Plotter**: Create SchismPlotter with model configuration
2. **Create Visualizations**: Use plot methods for specific visualization types
3. **Customize Appearance**: Apply styling and formatting options
4. **Export Results**: Save figures or animations for documentation

Typical Use Cases
~~~~~~~~~~~~~~~~

Model Development
  * Verify grid quality and structure
  * Examine boundary condition setup
  * Validate atmospheric forcing data
  * Create diagnostic plots during development

Model Documentation
  * Generate publication-quality figures
  * Create comprehensive model overviews
  * Produce animations for presentations
  * Export high-resolution graphics

Quality Assurance
  * Run validation checks before simulation
  * Assess grid and data quality metrics
  * Generate validation reports
  * Identify potential model issues

Integration Examples
~~~~~~~~~~~~~~~~~~~

Command Line Tools
  * Demo scripts for testing and examples
  * Batch processing capabilities
  * Automated plot generation

Jupyter Notebooks
  * Interactive exploration and analysis
  * Inline plotting and visualization
  * Educational and tutorial content

Automated Workflows
  * Integration with CI/CD pipelines
  * Automated quality checks
  * Batch processing of multiple models

Technical Features
-----------------

Performance Optimizations
~~~~~~~~~~~~~~~~~~~~~~~~~

* **Memory Efficiency**: Optimized handling of large datasets
* **Rendering Speed**: Efficient matplotlib usage patterns
* **Scalability**: Support for high-resolution grids and long time series
* **Caching**: Intelligent caching of computed results

Quality and Standards
~~~~~~~~~~~~~~~~~~~~

* **Code Quality**: Comprehensive test suite with high coverage
* **Documentation**: Detailed API documentation and examples
* **Standards Compliance**: PEP8 formatting and type hints
* **Error Handling**: Robust error handling and user feedback

Extensibility
~~~~~~~~~~~~

* **Plugin Architecture**: Easy addition of new plot types
* **Custom Styling**: Flexible theming and appearance customization
* **Data Sources**: Support for various input data formats
* **Output Formats**: Multiple export options for different use cases

Getting Started
---------------

Basic Usage
~~~~~~~~~~

.. code-block:: python

   from rompy.schism.plotting import SchismPlotter

   # Initialize with model configuration
   plotter = SchismPlotter(config=schism_config)

   # Create basic visualizations
   fig, ax = plotter.plot_grid()
   fig, ax = plotter.plot_bathymetry()
   fig, ax = plotter.plot_boundaries()

Advanced Features
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create comprehensive overview
   fig, axes = plotter.plot_comprehensive_overview()

   # Generate validation report
   results = plotter.run_model_validation()
   fig, axes = plotter.plot_validation_summary()

   # Create time series animation
   from rompy.schism.plotting.animation import AnimationConfig

   anim_config = AnimationConfig(frame_rate=15, quality='high')
   plotter = SchismPlotter(grid_file="hgrid.gr3", animation_config=anim_config)

   anim = plotter.animate_boundary_data(
       "SAL_3D.th.nc", "salinity", "salinity_animation.mp4"
   )

Next Steps
----------

* :doc:`api_reference` - Detailed API documentation
* :doc:`examples` - Comprehensive examples and use cases
* :doc:`animations` - Time series animation capabilities
* :doc:`tutorials` - Step-by-step tutorials and workflows

The SCHISM plotting module provides a powerful and flexible foundation for visualizing SCHISM model data, supporting everything from quick diagnostic plots to publication-quality figures and dynamic animations.
