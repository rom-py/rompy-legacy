Examples and Use Cases
======================

This section provides detailed examples demonstrating various SCHISM plotting capabilities with real-world use cases.

.. contents::
   :local:
   :depth: 2

Basic Plotting Examples
-----------------------

Grid Visualization
~~~~~~~~~~~~~~~~~~

**Simple Grid Plot**

.. code-block:: python

   from rompy.schism.plotting import SchismPlotter
   from rompy.model import ModelRun
   
   # Load SCHISM configuration
   config = ModelRun.from_yaml("my_schism_config.yaml")
   plotter = SchismPlotter(config=config.config)
   
   # Basic grid plot
   fig, ax = plotter.plot_grid()
   fig.show()

**Grid with Custom Styling**

.. code-block:: python

   # Grid plot with custom appearance
   fig, ax = plotter.plot_grid(
       figsize=(14, 10),
       show_elements=True,
       show_boundaries=True,
       element_alpha=0.3,
       boundary_color='red',
       boundary_linewidth=2.0
   )
   
   # Add title and save
   ax.set_title('SCHISM Model Grid', fontsize=16)
   fig.savefig('grid_overview.png', dpi=300, bbox_inches='tight')

**Grid Quality Assessment**

.. code-block:: python

   # Plot grid with quality metrics
   fig, ax = plotter.plot_grid_quality(
       figsize=(12, 10),
       quality_metric='aspect_ratio',
       colormap='RdYlBu_r',
       show_colorbar=True
   )
   
   # Highlight poor quality elements
   ax.set_title('Grid Element Quality (Aspect Ratio)')
   fig.show()

Bathymetry Plotting
~~~~~~~~~~~~~~~~~~~

**Basic Bathymetry**

.. code-block:: python

   # Simple bathymetry plot
   fig, ax = plotter.plot_bathymetry(
       figsize=(12, 10),
       colormap='ocean',
       show_contours=True,
       contour_levels=20
   )
   fig.show()

**Bathymetry with Custom Levels**

.. code-block:: python

   import numpy as np
   
   # Define custom depth contours
   depth_levels = np.array([0, 5, 10, 20, 50, 100, 200, 500, 1000])
   
   fig, ax = plotter.plot_bathymetry(
       figsize=(14, 10),
       colormap='ocean_r',
       contour_levels=depth_levels,
       show_contours=True,
       contour_colors='black',
       contour_linewidth=0.5
   )
   
   # Customize colorbar
   ax.set_title('Bathymetry with Custom Depth Contours')
   fig.show()

**Bathymetry Analysis**

.. code-block:: python

   # Bathymetry with statistical analysis
   fig, ax = plotter.plot_bathymetry_analysis(
       figsize=(16, 10),
       include_histogram=True,
       include_statistics=True,
       depth_ranges=[0, 10, 50, 100, 500, 2000]
   )
   fig.show()

Boundary Condition Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Boundary Overview**

.. code-block:: python

   # Plot all boundaries with types
   fig, ax = plotter.plot_boundaries(
       figsize=(12, 10),
       show_boundary_types=True,
       color_by_type=True,
       show_node_numbers=False
   )
   
   # Add legend for boundary types
   ax.legend(title='Boundary Types')
   fig.show()

**Detailed Boundary Analysis**

.. code-block:: python

   # Boundary plot with detailed information
   fig, ax = plotter.plot_boundary_analysis(
       figsize=(14, 12),
       show_elevations=True,
       show_tracers=True,
       show_forcing_locations=True,
       annotate_segments=True
   )
   fig.show()

Data Visualization Examples
---------------------------

Atmospheric Forcing
~~~~~~~~~~~~~~~~~~~

**Wind Field Visualization**

.. code-block:: python

   # Plot wind vectors and speed
   fig, ax = plotter.plot_atmospheric_data(
       figsize=(16, 10),
       variables=['wind_speed', 'wind_direction'],
       plot_type='vectors',
       vector_scale=20,
       vector_alpha=0.7,
       subsample_factor=5  # Plot every 5th point for clarity
   )
   
   ax.set_title('Wind Field - First Time Step')
   fig.show()

**Atmospheric Time Series**

.. code-block:: python

   # Create atmospheric forcing overview
   fig, axes = plotter.plot_atmospheric_timeseries(
       figsize=(16, 12),
       variables=['air_pressure', 'air_temperature', 'wind_speed'],
       time_range=(0, 72),  # First 72 hours
       location='center'    # Sample at grid center
   )
   
   # Customize each subplot
   axes['pressure'].set_ylabel('Pressure (Pa)')
   axes['temperature'].set_ylabel('Temperature (K)')
   axes['wind'].set_ylabel('Wind Speed (m/s)')
   
   fig.suptitle('Atmospheric Forcing Time Series')
   fig.show()

**Pressure and Temperature Fields**

.. code-block:: python

   # Multi-variable atmospheric plot
   fig, axes = plotter.plot_atmospheric_multivar(
       figsize=(18, 12),
       variables=['air_pressure', 'air_temperature', 'specific_humidity'],
       time_index=0,
       layout='row'  # Arrange in a row
   )
   
   # Customize each panel
   for var, ax in axes.items():
       ax.set_title(f'{var.replace("_", " ").title()}')
   
   fig.show()

Boundary Data Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**3D Temperature Data**

.. code-block:: python

   # Plot temperature boundary data
   fig, ax = plotter.plot_boundary_data(
       'TEM_3D.th.nc',
       variable='temperature',
       time_index=0,
       depth_level=0,  # Surface
       figsize=(12, 10),
       colormap='thermal'
   )
   
   ax.set_title('Surface Temperature at Ocean Boundaries')
   fig.show()

**Salinity Profiles**

.. code-block:: python

   # Plot vertical salinity profiles
   fig, ax = plotter.plot_boundary_profiles(
       'SAL_3D.th.nc',
       variable='salinity',
       locations=[0, 10, 20],  # Boundary node indices
       time_range=(0, 24),
       figsize=(10, 8)
   )
   
   ax.set_xlabel('Salinity (PSU)')
   ax.set_ylabel('Depth (m)')
   ax.legend(title='Boundary Locations')
   fig.show()

**Velocity Boundary Conditions**

.. code-block:: python

   # Plot velocity boundary data
   fig, ax = plotter.plot_velocity_boundaries(
       'uv3D.th.nc',
       time_index=0,
       depth_level=0,
       figsize=(14, 10),
       vector_scale=10,
       show_magnitude=True
   )
   
   ax.set_title('Velocity Boundary Conditions (Surface)')
   fig.show()

**Elevation Boundary Time Series**

.. code-block:: python

   # Plot elevation time series at boundaries
   fig, ax = plotter.plot_elevation_timeseries(
       'elev2D.th.nc',
       boundary_nodes=[0, 20, 40, 60, 80],  # Sample nodes
       time_range=(0, 168),  # One week
       figsize=(14, 8)
   )
   
   ax.set_xlabel('Time (hours)')
   ax.set_ylabel('Elevation (m)')
   ax.set_title('Tidal Elevation at Boundary Nodes')
   fig.show()

Advanced Visualization Examples
-------------------------------

Comprehensive Overview Plots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**8-Panel Model Overview**

.. code-block:: python

   # Create comprehensive overview
   fig, axes = plotter.plot_comprehensive_overview(
       figsize=(20, 16),
       include_validation=True,
       include_quality_metrics=True,
       include_data_summary=True
   )
   
   # Save high-resolution overview
   fig.savefig('model_comprehensive_overview.png', 
               dpi=300, bbox_inches='tight')
   fig.show()

**Grid-Focused Analysis**

.. code-block:: python

   # Grid analysis overview
   fig, axes = plotter.plot_grid_analysis_overview(
       figsize=(16, 12),
       quality_metrics=['aspect_ratio', 'skewness', 'area'],
       include_statistics=True
   )
   
   fig.suptitle('Grid Quality Analysis', fontsize=16)
   fig.show()

**Data-Focused Analysis**

.. code-block:: python

   # Data analysis overview
   fig, axes = plotter.plot_data_analysis_overview(
       figsize=(16, 12),
       atmospheric_variables=['wind_speed', 'air_pressure'],
       boundary_files=['TEM_3D.th.nc', 'SAL_3D.th.nc'],
       time_range=(0, 48)
   )
   
   fig.suptitle('Input Data Analysis', fontsize=16)
   fig.show()

Model Validation and Quality Assessment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Complete Model Validation**

.. code-block:: python

   # Run comprehensive validation
   validation_results = plotter.run_model_validation()
   
   # Print summary
   status_counts = {'PASS': 0, 'WARNING': 0, 'FAIL': 0}
   for result in validation_results:
       status_counts[result.status] += 1
       print(f"{result.check_name}: {result.status}")
       if result.status != 'PASS':
           print(f"  Details: {result.message}")
   
   print(f"\nValidation Summary:")
   print(f"  PASS: {status_counts['PASS']}")
   print(f"  WARNING: {status_counts['WARNING']}")
   print(f"  FAIL: {status_counts['FAIL']}")

**Validation Visualization**

.. code-block:: python

   # Create validation summary plots
   fig, axes = plotter.plot_validation_summary(
       figsize=(14, 10),
       include_radar_chart=True,
       include_timeline=True,
       include_details_table=True
   )
   
   fig.suptitle('Model Validation Summary')
   fig.show()

**Quality Assessment Radar Chart**

.. code-block:: python

   # Quality assessment visualization
   fig, ax = plotter.plot_quality_assessment(
       figsize=(10, 10),
       categories=[
           'Grid Quality', 'Boundary Setup', 'Forcing Data',
           'Configuration', 'Time Stepping', 'File Integrity'
       ],
       show_scores=True,
       highlight_issues=True
   )
   
   ax.set_title('Model Setup Quality Assessment')
   fig.show()

Custom Analysis Examples
~~~~~~~~~~~~~~~~~~~~~~~~

**Multi-Time Step Analysis**

.. code-block:: python

   import matplotlib.pyplot as plt
   
   # Plot data evolution over time
   time_steps = [0, 24, 48, 72]  # Hours
   fig, axes = plt.subplots(2, 2, figsize=(16, 12))
   axes = axes.flatten()
   
   for i, time_step in enumerate(time_steps):
       # Plot atmospheric pressure at each time step
       ax = axes[i]
       fig_temp, ax = plotter.plot_atmospheric_data(
           variables=['air_pressure'],
           time_index=time_step,
           figsize=(8, 6),
           ax=ax
       )
       ax.set_title(f'Pressure at t={time_step}h')
       plt.close(fig_temp)  # Close temporary figure
   
   plt.tight_layout()
   plt.show()

**Cross-Section Analysis**

.. code-block:: python

   # Define cross-section line
   start_point = (-125.0, 45.0)  # Longitude, Latitude
   end_point = (-124.0, 46.0)
   
   # Extract cross-section data
   fig, ax = plotter.plot_cross_section(
       variable='temperature',
       start_point=start_point,
       end_point=end_point,
       data_file='TEM_3D.th.nc',
       time_index=0,
       figsize=(12, 8)
   )
   
   ax.set_xlabel('Distance along section (km)')
   ax.set_ylabel('Depth (m)')
   ax.set_title('Temperature Cross-Section')
   fig.show()

**Seasonal Analysis**

.. code-block:: python

   # Compare different seasons
   seasonal_configs = {
       'Winter': 'winter_config.yaml',
       'Summer': 'summer_config.yaml'
   }
   
   fig, axes = plt.subplots(1, 2, figsize=(16, 8))
   
   for i, (season, config_file) in enumerate(seasonal_configs.items()):
       config = ModelRun.from_yaml(config_file)
       plotter = SchismPlotter(config=config.config)
       
       # Plot seasonal forcing
       fig_temp, ax = plotter.plot_atmospheric_data(
           variables=['air_temperature'],
           time_index=0,
           ax=axes[i]
       )
       axes[i].set_title(f'{season} Temperature')
       plt.close(fig_temp)
   
   plt.tight_layout()
   plt.show()

Production Workflow Examples
---------------------------

Automated Report Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pathlib import Path
   import datetime
   
   def generate_model_report(config_file, output_dir):
       """Generate complete model validation report."""
       
       # Initialize plotter
       config = ModelRun.from_yaml(config_file)
       plotter = SchismPlotter(config=config.config)
       
       # Create output directory
       output_dir = Path(output_dir)
       output_dir.mkdir(exist_ok=True)
       
       # 1. Grid overview
       fig, ax = plotter.plot_grid()
       fig.savefig(output_dir / 'grid_overview.png', dpi=150)
       plt.close(fig)
       
       # 2. Bathymetry
       fig, ax = plotter.plot_bathymetry()
       fig.savefig(output_dir / 'bathymetry.png', dpi=150)
       plt.close(fig)
       
       # 3. Boundaries
       fig, ax = plotter.plot_boundaries()
       fig.savefig(output_dir / 'boundaries.png', dpi=150)
       plt.close(fig)
       
       # 4. Comprehensive overview
       fig, axes = plotter.plot_comprehensive_overview()
       fig.savefig(output_dir / 'comprehensive_overview.png', dpi=150)
       plt.close(fig)
       
       # 5. Validation
       validation_results = plotter.run_model_validation()
       fig, axes = plotter.plot_validation_summary()
       fig.savefig(output_dir / 'validation_summary.png', dpi=150)
       plt.close(fig)
       
       # Generate text report
       report_file = output_dir / 'validation_report.txt'
       with open(report_file, 'w') as f:
           f.write(f"SCHISM Model Validation Report\n")
           f.write(f"Generated: {datetime.datetime.now()}\n")
           f.write(f"Config: {config_file}\n\n")
           
           for result in validation_results:
               f.write(f"{result.check_name}: {result.status}\n")
               if result.message:
                   f.write(f"  {result.message}\n")
       
       print(f"Report generated in: {output_dir}")

   # Usage
   generate_model_report('my_config.yaml', 'model_report')

Batch Processing
~~~~~~~~~~~~~~~~

.. code-block:: python

   def process_multiple_configs(config_dir, output_base):
       """Process multiple SCHISM configurations."""
       
       config_dir = Path(config_dir)
       output_base = Path(output_base)
       
       for config_file in config_dir.glob('*.yaml'):
           print(f"Processing: {config_file.name}")
           
           try:
               # Create output directory for this config
               config_output = output_base / config_file.stem
               config_output.mkdir(exist_ok=True)
               
               # Initialize plotter
               config = ModelRun.from_yaml(config_file)
               plotter = SchismPlotter(config=config.config)
               
               # Generate key plots
               plots = {
                   'grid': plotter.plot_grid,
                   'bathymetry': plotter.plot_bathymetry,
                   'boundaries': plotter.plot_boundaries
               }
               
               for plot_name, plot_func in plots.items():
                   fig, ax = plot_func()
                   fig.savefig(config_output / f'{plot_name}.png', dpi=150)
                   plt.close(fig)
                   
               print(f"  Completed: {config_output}")
               
           except Exception as e:
               print(f"  Error processing {config_file.name}: {e}")

   # Usage
   process_multiple_configs('configs/', 'batch_output/')

Performance Optimization Examples
---------------------------------

Large Grid Visualization
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # For very large grids (>100k elements)
   fig, ax = plotter.plot_grid(
       figsize=(12, 10),
       subsample_factor=20,     # Plot every 20th element
       simplify_boundaries=True, # Simplify boundary lines
       show_elements=False,     # Don't show individual elements
       show_nodes=False,        # Don't show node markers
       alpha=0.5               # Make partially transparent
   )

High-Resolution Output
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Generate publication-quality figures
   fig, ax = plotter.plot_bathymetry(
       figsize=(16, 12),
       colormap='ocean',
       contour_levels=50,
       show_contours=True
   )
   
   # High-resolution save with multiple formats
   fig.savefig('bathymetry_hires.png', dpi=300, bbox_inches='tight')
   fig.savefig('bathymetry_hires.pdf', bbox_inches='tight')
   fig.savefig('bathymetry_hires.svg', bbox_inches='tight')

Memory Management
~~~~~~~~~~~~~~~~

.. code-block:: python

   import gc
   
   # Process large datasets with memory management
   for time_step in range(0, 168, 6):  # Every 6 hours for a week
       fig, ax = plotter.plot_atmospheric_data(
           time_index=time_step,
           variables=['wind_speed']
       )
       
       # Save and clean up
       fig.savefig(f'wind_t{time_step:03d}.png', dpi=150)
       plt.close(fig)
       gc.collect()  # Force garbage collection

Integration Examples
-------------------

Jupyter Notebook Usage
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # In Jupyter notebook cell
   %matplotlib inline
   
   from rompy.schism.plotting import SchismPlotter
   
   # Interactive plotting
   plotter = SchismPlotter(config=config)
   
   # Plot displays automatically
   fig, ax = plotter.plot_grid()
   # No need for fig.show() in notebooks

Web Application Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import io
   import base64
   
   def create_plot_for_web(config, plot_type='grid'):
       """Create plot for web display."""
       
       plotter = SchismPlotter(config=config)
       
       # Generate plot
       if plot_type == 'grid':
           fig, ax = plotter.plot_grid()
       elif plot_type == 'bathymetry':
           fig, ax = plotter.plot_bathymetry()
       
       # Convert to base64 for web display
       img_buffer = io.BytesIO()
       fig.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
       img_buffer.seek(0)
       img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
       plt.close(fig)
       
       return f"data:image/png;base64,{img_base64}"

Error Handling Examples
----------------------

Robust Plotting Function
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def safe_plot_overview(config_file, output_dir):
       """Create overview plot with comprehensive error handling."""
       
       try:
           # Load configuration
           config = ModelRun.from_yaml(config_file)
           plotter = SchismPlotter(config=config.config)
           
           # Create overview with fallbacks
           try:
               fig, axes = plotter.plot_comprehensive_overview()
               save_path = Path(output_dir) / 'comprehensive_overview.png'
               
           except Exception as e:
               print(f"Comprehensive overview failed: {e}")
               print("Falling back to basic overview...")
               
               fig, ax = plotter.plot_overview()
               save_path = Path(output_dir) / 'basic_overview.png'
           
           # Save plot
           fig.savefig(save_path, dpi=150, bbox_inches='tight')
           print(f"Overview saved to: {save_path}")
           
           return fig, save_path
           
       except FileNotFoundError:
           print(f"Configuration file not found: {config_file}")
           return None, None
           
       except Exception as e:
           print(f"Unexpected error: {e}")
           return None, None

Validation with Fallbacks
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def validate_with_fallbacks(plotter):
       """Run validation with graceful fallbacks."""
       
       try:
           # Try full validation
           results = plotter.run_model_validation()
           print(f"Full validation completed: {len(results)} checks")
           
       except Exception as e:
           print(f"Full validation failed: {e}")
           print("Running basic validation...")
           
           # Basic validation fallbacks
           basic_checks = []
           
           try:
               # Check if grid is available
               if plotter.grid is not None:
                   basic_checks.append("Grid: PASS")
               else:
                   basic_checks.append("Grid: FAIL - No grid data")
           except:
               basic_checks.append("Grid: ERROR")
           
           try:
               # Check if config is available
               if plotter.config is not None:
                   basic_checks.append("Config: PASS")
               else:
                   basic_checks.append("Config: FAIL - No configuration")
           except:
               basic_checks.append("Config: ERROR")
           
           print("Basic validation results:")
           for check in basic_checks:
               print(f"  {check}")
           
           return basic_checks

See Also
--------

* :doc:`index` - Main plotting documentation
* :doc:`api_reference` - Complete API reference
* :doc:`tutorials` - Step-by-step tutorials
* :ref:`schism-real-data-demo` - Real data demonstration script