SCHISM Plotting Tutorials
=========================

This section provides step-by-step tutorials for common SCHISM plotting workflows, from basic visualization to advanced multi-panel overviews and animations.

.. contents::
   :local:
   :depth: 2

Tutorial 1: Basic Model Visualization
-------------------------------------

This tutorial covers the fundamental plotting capabilities for visualizing SCHISM model setup.

Prerequisites
~~~~~~~~~~~~~

* SCHISM model configuration file
* Basic familiarity with Python and matplotlib

Setting Up
~~~~~~~~~~

.. code-block:: python

   from rompy.schism.plotting import SchismPlotter
   from rompy.model import ModelRun
   import matplotlib.pyplot as plt

   # Load your SCHISM configuration
   config = ModelRun.from_yaml("my_schism_config.yaml")

   # Initialize the plotter
   plotter = SchismPlotter(config=config.config)

Creating Basic Plots
~~~~~~~~~~~~~~~~~~~~

**Step 1: Grid Visualization**

.. code-block:: python

   # Plot the computational grid
   fig, ax = plotter.plot_grid(
       figsize=(12, 10),
       show_elements=True,
       show_boundaries=True
   )

   # Customize the plot
   ax.set_title('SCHISM Computational Grid', fontsize=16)
   fig.tight_layout()
   fig.show()

**Step 2: Bathymetry**

.. code-block:: python

   # Plot bathymetry with contours
   fig, ax = plotter.plot_bathymetry(
       figsize=(14, 10),
       colormap='ocean',
       contour_levels=20,
       show_contours=True
   )

   ax.set_title('Model Bathymetry', fontsize=16)
   fig.show()

**Step 3: Boundary Conditions**

.. code-block:: python

   # Plot model boundaries
   fig, ax = plotter.plot_boundaries(
       figsize=(12, 8),
       show_open_boundaries=True,
       show_land_boundaries=True
   )

   ax.set_title('Model Boundaries', fontsize=16)
   fig.show()

Saving Results
~~~~~~~~~~~~~

.. code-block:: python

   # Save high-quality plots
   fig, ax = plotter.plot_grid()
   fig.savefig('grid.png', dpi=300, bbox_inches='tight')

   fig, ax = plotter.plot_bathymetry()
   fig.savefig('bathymetry.png', dpi=300, bbox_inches='tight')

Tutorial 2: Working with Real Data
----------------------------------

This tutorial demonstrates how to work with actual SCHISM model output files.

Data Setup
~~~~~~~~~~

For this tutorial, you'll need:

* Grid file (``hgrid.gr3``)
* Boundary data files (``SAL_3D.th.nc``, ``TEM_3D.th.nc``)
* Atmospheric forcing files (optional)

Loading Real Data
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pathlib import Path

   # Initialize with existing grid file
   grid_file = Path("path/to/hgrid.gr3")
   plotter = SchismPlotter(grid_file=grid_file)

Plotting Boundary Data
~~~~~~~~~~~~~~~~~~~~~

**Step 1: Examine Data Structure**

.. code-block:: python

   import xarray as xr

   # Load and examine boundary data
   sal_data = xr.open_dataset("SAL_3D.th.nc")
   print(sal_data)
   print(f"Time range: {sal_data.time.values[0]} to {sal_data.time.values[-1]}")

**Step 2: Plot Boundary Data**

.. code-block:: python

   # Plot salinity boundary conditions
   fig, ax = plotter.plot_boundary_data(
       "SAL_3D.th.nc",
       variable="salinity",
       time_index=0,  # First time step
       figsize=(12, 8)
   )

   ax.set_title('Salinity Boundary Conditions (t=0)', fontsize=14)
   fig.show()

**Step 3: Time Series Analysis**

.. code-block:: python

   # Plot time series at specific boundary nodes
   fig, ax = plotter.plot_boundary_time_series(
       "SAL_3D.th.nc",
       variable="salinity",
       node_indices=[0, 10, 20],  # Sample nodes
       figsize=(12, 6)
   )

   ax.set_title('Salinity Time Series at Boundary Nodes')
   fig.show()

Tutorial 3: Comprehensive Model Overview
----------------------------------------

Create multi-panel overviews for complete model documentation.

Basic Overview
~~~~~~~~~~~~~

.. code-block:: python

   # Create 4-panel basic overview
   fig, axes = plotter.plot_overview(
       figsize=(16, 12),
       include_validation=True
   )

   # Add overall title
   fig.suptitle('SCHISM Model Overview', fontsize=18, y=0.95)
   fig.tight_layout()
   fig.show()

Comprehensive Overview
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create detailed 8-panel overview
   fig, axes = plotter.plot_comprehensive_overview(
       figsize=(20, 16),
       include_validation=True,
       include_quality_metrics=True,
       include_data_analysis=True
   )

   # Save for documentation
   fig.savefig('model_comprehensive_overview.png',
               dpi=150, bbox_inches='tight')

Specialized Overviews
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Grid-focused analysis
   fig, axes = plotter.plot_grid_analysis_overview(
       figsize=(16, 12)
   )

   # Data-focused analysis
   fig, axes = plotter.plot_data_analysis_overview(
       figsize=(16, 12)
   )

Tutorial 4: Model Validation Workflow
-------------------------------------

Implement a comprehensive validation workflow for model quality assurance.

Running Validation
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Run complete model validation
   validation_results = plotter.run_model_validation()

   # Print summary
   for result in validation_results:
       status_icon = "✓" if result.status == "PASS" else "✗"
       print(f"{status_icon} {result.check_name}: {result.message}")

Validation Visualization
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create validation summary plot
   fig, axes = plotter.plot_validation_summary(
       figsize=(14, 10)
   )

   fig.suptitle('Model Validation Results', fontsize=16)
   fig.show()

Custom Validation Checks
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rompy.schism.plotting.validation import ModelValidator

   # Create custom validator
   validator = ModelValidator(config=plotter.config, grid=plotter.grid)

   # Run specific validation categories
   grid_results = validator.validate_grid()
   boundary_results = validator.validate_boundaries()
   forcing_results = validator.validate_forcing()

   # Combine results
   all_results = grid_results + boundary_results + forcing_results

Tutorial 5: Time Series Animations
----------------------------------

Create dynamic animations showing temporal evolution of model data.

Basic Animation Setup
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rompy.schism.plotting.animation import AnimationConfig

   # Configure animation parameters
   anim_config = AnimationConfig(
       frame_rate=15,
       show_time_label=True,
       show_progress=True,
       quality='medium'
   )

   # Initialize plotter with animation support
   plotter = SchismPlotter(
       grid_file="hgrid.gr3",
       animation_config=anim_config
   )

Boundary Data Animation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create salinity boundary animation
   anim = plotter.animate_boundary_data(
       data_file="SAL_3D.th.nc",
       variable="salinity",
       output_file="salinity_animation.mp4",
       level_idx=0,  # surface level
       figsize=(12, 8),
       cmap='viridis'
   )

   print("Animation saved to salinity_animation.mp4")

Multi-Variable Animation
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Define multiple data sources
   data_files = {
       'Salinity': 'SAL_3D.th.nc',
       'Temperature': 'TEM_3D.th.nc'
   }

   variables = {
       'Salinity': 'salinity',
       'Temperature': 'temperature'
   }

   # Create synchronized multi-panel animation
   anim = plotter.create_multi_variable_animation(
       data_files=data_files,
       variables=variables,
       output_file="multi_variable_animation.mp4",
       layout='grid',
       figsize=(16, 10)
   )

Custom Animation Settings
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # High-quality animation for presentations
   high_quality_config = AnimationConfig(
       frame_rate=30,
       bitrate=3600,
       quality='high',
       figsize=(16, 12),
       time_label_format="%Y-%m-%d %H:%M",
       add_coastlines=True,
       add_gridlines=True
   )

   plotter = SchismPlotter(grid_file="hgrid.gr3",
                          animation_config=high_quality_config)

   anim = plotter.animate_boundary_data(
       "SAL_3D.th.nc", "salinity", "high_quality_animation.mp4"
   )

Tutorial 6: Batch Processing and Automation
-------------------------------------------

Automate plot generation for multiple models or datasets.

Batch Plot Generation
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pathlib import Path
   import glob

   # Define plot types to generate
   plot_functions = [
       ('grid', lambda p: p.plot_grid()),
       ('bathymetry', lambda p: p.plot_bathymetry()),
       ('boundaries', lambda p: p.plot_boundaries()),
       ('overview', lambda p: p.plot_overview())
   ]

   # Process multiple model configurations
   config_files = glob.glob("configs/*.yaml")

   for config_file in config_files:
       model_name = Path(config_file).stem
       output_dir = Path(f"plots/{model_name}")
       output_dir.mkdir(parents=True, exist_ok=True)

       # Load configuration and create plotter
       config = ModelRun.from_yaml(config_file)
       plotter = SchismPlotter(config=config.config)

       # Generate all plot types
       for plot_name, plot_func in plot_functions:
           try:
               fig, ax = plot_func(plotter)
               fig.savefig(output_dir / f"{plot_name}.png",
                          dpi=150, bbox_inches='tight')
               plt.close(fig)
               print(f"Generated {model_name}/{plot_name}.png")
           except Exception as e:
               print(f"Error generating {model_name}/{plot_name}: {e}")

Automated Validation
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Automated validation for multiple models
   validation_summary = {}

   for config_file in config_files:
       model_name = Path(config_file).stem

       try:
           config = ModelRun.from_yaml(config_file)
           plotter = SchismPlotter(config=config.config)

           # Run validation
           results = plotter.run_model_validation()

           # Summarize results
           passed = sum(1 for r in results if r.status == "PASS")
           total = len(results)

           validation_summary[model_name] = {
               'passed': passed,
               'total': total,
               'success_rate': passed / total if total > 0 else 0
           }

           # Generate validation plot
           fig, axes = plotter.plot_validation_summary()
           fig.savefig(f"validation/{model_name}_validation.png",
                      dpi=150, bbox_inches='tight')
           plt.close(fig)

       except Exception as e:
           print(f"Validation failed for {model_name}: {e}")
           validation_summary[model_name] = {'error': str(e)}

   # Print summary report
   print("\nValidation Summary:")
   print("-" * 50)
   for model_name, summary in validation_summary.items():
       if 'error' in summary:
           print(f"{model_name}: ERROR - {summary['error']}")
       else:
           rate = summary['success_rate'] * 100
           print(f"{model_name}: {summary['passed']}/{summary['total']} "
                 f"({rate:.1f}% success)")

Tutorial 7: Integration with Jupyter Notebooks
----------------------------------------------

Effective use of SCHISM plotting in interactive Jupyter environments.

Notebook Setup
~~~~~~~~~~~~~

.. code-block:: python

   # Notebook cell 1: Setup
   %matplotlib inline
   import matplotlib.pyplot as plt

   from rompy.schism.plotting import SchismPlotter
   from rompy.model import ModelRun

   # Configure matplotlib for notebooks
   plt.rcParams['figure.dpi'] = 100
   plt.rcParams['savefig.dpi'] = 150

Interactive Plotting
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Notebook cell 2: Load and plot
   config = ModelRun.from_yaml("model_config.yaml")
   plotter = SchismPlotter(config=config.config)

   # Interactive grid plot
   fig, ax = plotter.plot_grid(figsize=(12, 8))
   plt.show()  # Displays inline automatically

Widget Integration
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Notebook cell 3: Interactive widgets
   from ipywidgets import interact, IntSlider
   import xarray as xr

   # Load boundary data
   sal_data = xr.open_dataset("SAL_3D.th.nc")
   max_time = len(sal_data.time) - 1

   # Interactive time step selection
   @interact(time_step=IntSlider(min=0, max=max_time, step=1, value=0))
   def plot_boundary_time_step(time_step):
       fig, ax = plotter.plot_boundary_data(
           "SAL_3D.th.nc",
           variable="salinity",
           time_index=time_step,
           figsize=(10, 6)
       )
       ax.set_title(f'Salinity at Time Step {time_step}')
       plt.show()

Export for Reports
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Notebook cell 4: High-quality export
   # Create publication-quality figure
   fig, axes = plotter.plot_comprehensive_overview(figsize=(16, 12))

   # Save for inclusion in reports
   fig.savefig('model_overview.png', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')

   # Also save as PDF for vector graphics
   fig.savefig('model_overview.pdf', bbox_inches='tight')

Tutorial 8: Custom Styling and Themes
-------------------------------------

Create custom visual styles and themes for consistent documentation.

Custom Plot Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rompy.schism.plotting.core import PlotConfig

   # Define custom styling
   custom_config = PlotConfig(
       figsize=(14, 10),
       dpi=150,
       colormap='viridis',
       grid_alpha=0.2,
       boundary_linewidth=2.5,
       boundary_color='red',
       colorbar_orientation='horizontal',
       title_fontsize=16,
       label_fontsize=12
   )

   # Apply to plotter
   plotter = SchismPlotter(
       config=schism_config,
       plot_config=custom_config
   )

Publication Theme
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Scientific publication theme
   publication_config = PlotConfig(
       figsize=(12, 9),
       dpi=300,
       colormap='coolwarm',
       grid_alpha=0.3,
       boundary_linewidth=1.5,
       colorbar_orientation='vertical',
       title_fontsize=14,
       label_fontsize=11,
       tick_labelsize=10,
       colorbar_labelsize=10
   )

Presentation Theme
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Presentation theme with high contrast
   presentation_config = PlotConfig(
       figsize=(16, 12),
       dpi=150,
       colormap='plasma',
       grid_alpha=0.4,
       boundary_linewidth=3.0,
       boundary_color='white',
       title_fontsize=20,
       label_fontsize=16,
       tick_labelsize=14
   )

Troubleshooting Common Issues
----------------------------

Memory Management
~~~~~~~~~~~~~~~~

.. code-block:: python

   # For large datasets, manage memory carefully
   import gc

   # Close figures after saving
   fig, ax = plotter.plot_grid()
   fig.savefig('grid.png')
   plt.close(fig)

   # Force garbage collection for large processing loops
   for i in range(many_plots):
       # ... create plot ...
       plt.close('all')
       if i % 10 == 0:  # Every 10 plots
           gc.collect()

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Optimize for large grids
   fig, ax = plotter.plot_grid(
       subsample_factor=5,  # Plot every 5th element
       simplify_boundaries=True,
       show_elements=False  # Skip element rendering
   )

Error Handling
~~~~~~~~~~~~~

.. code-block:: python

   # Robust plotting with error handling
   def safe_plot(plot_func, filename, *args, **kwargs):
       try:
           fig, ax = plot_func(*args, **kwargs)
           fig.savefig(filename, dpi=150, bbox_inches='tight')
           plt.close(fig)
           print(f"Successfully created {filename}")
           return True
       except Exception as e:
           print(f"Error creating {filename}: {e}")
           return False

   # Use with any plotting function
   safe_plot(plotter.plot_grid, 'grid.png', figsize=(12, 8))
   safe_plot(plotter.plot_bathymetry, 'bathymetry.png')

Next Steps
----------

After completing these tutorials, you should be able to:

* Create basic and advanced SCHISM visualizations
* Work with real model data and boundary conditions
* Generate comprehensive model overviews
* Implement validation workflows
* Create time series animations
* Automate plot generation processes
* Integrate plotting into Jupyter workflows
* Apply custom styling and themes

For more information:

* :doc:`api_reference` - Complete API documentation
* :doc:`examples` - Additional examples and use cases
* :doc:`animations` - Detailed animation documentation
* :doc:`index` - Main plotting documentation

Happy plotting!
