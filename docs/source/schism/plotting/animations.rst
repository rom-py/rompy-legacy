SCHISM Animation Documentation
==============================

The SCHISM plotting module includes comprehensive time series animation capabilities for creating dynamic visualizations of model input data, boundary conditions, atmospheric forcing, and grid-based temporal data.

.. contents::
   :local:
   :depth: 3

Overview
--------

The animation system enables creation of:

* **Boundary Data Animations**: Time evolution of boundary conditions (temperature, salinity, velocity)
* **Atmospheric Data Animations**: Progression of atmospheric forcing fields
* **Grid Data Animations**: Spatial-temporal visualization of grid-based data
* **Multi-Variable Animations**: Synchronized multi-panel animations comparing multiple variables

Key features include:

* High-quality MP4 and web-friendly GIF output formats
* Configurable frame rates, quality settings, and time ranges
* Geographic projections with coastlines and gridlines
* Time label overlays and progress tracking
* Animation playback controls (pause/resume/stop)
* Integration with existing SCHISM plotting workflows

Quick Start
-----------

Basic Animation Setup
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rompy.schism.plotting import SchismPlotter
   from rompy.schism.plotting.animation import AnimationConfig

   # Create animation configuration
   anim_config = AnimationConfig(
       frame_rate=15,           # frames per second
       show_time_label=True,    # display current time
       show_progress=True,      # show progress during creation
       quality='high'           # animation quality
   )

   # Initialize plotter with animation config
   plotter = SchismPlotter(
       grid_file="hgrid.gr3",
       animation_config=anim_config
   )

Simple Boundary Animation
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create boundary data animation
   anim = plotter.animate_boundary_data(
       data_file="SAL_3D.th.nc",
       variable="salinity",
       output_file="salinity_animation.mp4",
       level_idx=0,  # surface level
       figsize=(12, 8),
       cmap='viridis'
   )

   # Display animation
   import matplotlib.pyplot as plt
   plt.show()

Animation Types
---------------

Boundary Data Animations
~~~~~~~~~~~~~~~~~~~~~~~~

Visualize time evolution of boundary conditions with geographic coordinate mapping:

.. code-block:: python

   # Temperature boundary animation
   anim = plotter.animate_boundary_data(
       data_file="TEM_3D.th.nc",
       variable="temperature",
       output_file="temperature_boundary.mp4",
       level_idx=0,      # surface level
       figsize=(12, 8),
       cmap='RdYlBu_r'
   )

   # Salinity with custom time range
   config = AnimationConfig(
       time_start="2023-07-01T00:00:00",
       time_end="2023-07-02T00:00:00",
       frame_rate=20
   )

   plotter = SchismPlotter(grid_file="hgrid.gr3", animation_config=config)
   anim = plotter.animate_boundary_data(
       "SAL_3D.th.nc", "salinity", "salinity_subset.mp4"
   )

Atmospheric Data Animations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create animations of atmospheric forcing data progression:

.. code-block:: python

   # Air temperature animation
   anim = plotter.animate_atmospheric_data(
       data_file="sflux_air_1.nc",
       variable="air",
       parameter="air_temperature",
       output_file="air_temperature.mp4",
       figsize=(14, 10),
       cmap='RdYlBu_r'
   )

   # Wind field animation
   anim = plotter.animate_atmospheric_data(
       data_file="sflux_wind_1.nc",
       variable="wind",
       parameter="wind_speed",
       output_file="wind_animation.gif",
       figsize=(12, 10)
   )

Grid Data Animations
~~~~~~~~~~~~~~~~~~~~

Spatial-temporal visualization of grid-based data:

.. code-block:: python

   # Elevation data animation with grid overlay
   anim = plotter.animate_grid_data(
       data_file="elevation.nc",
       variable="elevation",
       output_file="elevation_animation.mp4",
       show_grid=True,
       figsize=(12, 12),
       cmap='coolwarm'
   )

   # Velocity magnitude animation
   anim = plotter.animate_grid_data(
       data_file="velocity.nc",
       variable="velocity_magnitude",
       output_file="velocity.mp4",
       show_grid=False,
       figsize=(14, 10)
   )

Multi-Variable Animations
~~~~~~~~~~~~~~~~~~~~~~~~~

Create synchronized multi-panel animations:

.. code-block:: python

   # Define data files and variables
   data_files = {
       'Temperature': 'TEM_3D.th.nc',
       'Salinity': 'SAL_3D.th.nc',
       'Elevation': 'elevation.nc'
   }

   variables = {
       'Temperature': 'temperature',
       'Salinity': 'salinity',
       'Elevation': 'elevation'
   }

   # Create multi-panel animation
   anim = plotter.create_multi_variable_animation(
       data_files=data_files,
       variables=variables,
       output_file="multi_variable.mp4",
       layout='grid',      # 'grid', 'vertical', 'horizontal'
       figsize=(18, 12)
   )

Animation Configuration
-----------------------

AnimationConfig Class
~~~~~~~~~~~~~~~~~~~~~

The ``AnimationConfig`` class provides comprehensive control over animation parameters:

.. autoclass:: rompy.schism.plotting.animation.AnimationConfig
   :members:
   :undoc-members:

Configuration Examples
~~~~~~~~~~~~~~~~~~~~~

**Basic Configuration**

.. code-block:: python

   from rompy.schism.plotting.animation import AnimationConfig

   # Standard quality animation
   config = AnimationConfig(
       frame_rate=15,
       show_time_label=True,
       show_progress=True,
       quality='medium'
   )

**High Quality Configuration**

.. code-block:: python

   # High quality for presentation
   config = AnimationConfig(
       frame_rate=30,
       bitrate=3600,       # higher bitrate
       quality='high',
       figsize=(16, 12),
       dpi=150,
       show_time_label=True,
       time_label_format="%Y-%m-%d %H:%M:%S"
   )

**Performance Optimized Configuration**

.. code-block:: python

   # Optimized for large datasets
   config = AnimationConfig(
       frame_rate=8,       # lower frame rate
       time_step=3,        # use every 3rd time step
       quality='low',
       show_progress=False,
       figsize=(10, 8)
   )

**Custom Time Range**

.. code-block:: python

   # Specific time period
   config = AnimationConfig(
       time_start="2023-07-01T12:00:00",
       time_end="2023-07-02T12:00:00",
       time_step=2,        # every 2nd time step
       frame_rate=20,
       duration=30.0       # 30 second animation
   )

Advanced Features
-----------------

Animation Controls
~~~~~~~~~~~~~~~~~

Control animation playback programmatically:

.. code-block:: python

   # Create animation
   anim = plotter.animate_boundary_data("data.nc", "temperature")

   # Animation controls
   plotter.pause_animation()   # pause current animation
   plotter.resume_animation()  # resume paused animation
   plotter.stop_animation()    # stop current animation

Custom Styling
~~~~~~~~~~~~~~

Apply custom visual styling to animations:

.. code-block:: python

   # Custom colormap and styling
   anim = plotter.animate_boundary_data(
       "SAL_3D.th.nc",
       "salinity",
       output_file="custom_style.mp4",
       figsize=(14, 10),
       cmap='plasma',
       vmin=30.0,          # custom value range
       vmax=36.0,
       add_coastlines=True,
       add_gridlines=True,
       grid_alpha=0.3
   )

Geographic Projections
~~~~~~~~~~~~~~~~~~~~~

Use geographic projections with cartopy:

.. code-block:: python

   # Mercator projection with geographic features
   anim = plotter.animate_atmospheric_data(
       "atmospheric_data.nc",
       "air_temperature",
       output_file="geo_animation.mp4",
       projection='mercator',
       add_coastlines=True,
       add_borders=True,
       add_gridlines=True,
       extent=[-180, 180, -90, 90]  # global extent
   )

Memory Management
~~~~~~~~~~~~~~~~

Handle large datasets efficiently:

.. code-block:: python

   # Memory-efficient animation for large datasets
   config = AnimationConfig(
       time_step=5,        # reduce temporal resolution
       frame_rate=10,      # moderate frame rate
       quality='medium'    # balance quality vs memory
   )

   # Process data in chunks if needed
   plotter = SchismPlotter(grid_file="hgrid.gr3", animation_config=config)

   # Use time subsetting for very large datasets
   anim = plotter.animate_boundary_data(
       "large_dataset.nc",
       "temperature",
       "subset_animation.mp4"
   )

Demo Scripts and Examples
-------------------------

Command Line Demo Script
~~~~~~~~~~~~~~~~~~~~~~~~

The package includes a comprehensive demo script for testing and examples:

.. code-block:: bash

   # Basic boundary animation with synthetic data
   python examples/schism_animation_demo.py --mode boundary --demo-data --show

   # High quality atmospheric animation as GIF
   python examples/schism_animation_demo.py \
       --mode atmospheric \
       --format gif \
       --frame-rate 20 \
       --quality high \
       --output weather.gif \
       --demo-data

   # Multi-variable animation
   python examples/schism_animation_demo.py \
       --mode multi \
       --demo-data \
       --frame-rate 15 \
       --output multi_var.mp4

   # Custom duration and frame rate
   python examples/schism_animation_demo.py \
       --mode grid \
       --frame-rate 30 \
       --duration 10.0 \
       --demo-data

Demo Script Options
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Animation modes
   --mode {boundary,atmospheric,grid,multi}

   # Output options
   --output OUTPUT_FILE              # output file path
   --format {mp4,gif}                # output format
   --show                           # display instead of saving

   # Quality settings
   --frame-rate RATE                # frames per second
   --duration SECONDS               # animation duration
   --quality {low,medium,high}      # quality preset

   # Data options
   --demo-data                      # use synthetic demo data
   --boundary-file FILE             # specify boundary data file
   --atmospheric-file FILE          # specify atmospheric file
   --grid-data-file FILE           # specify grid data file
   --grid-file FILE                # specify grid file
   --variable VARIABLE             # variable to animate

   # General options
   --verbose                       # verbose logging

Real Data Usage
~~~~~~~~~~~~~~

Working with actual SCHISM model output:

.. code-block:: bash

   # Auto-detect files in model run directory
   cd path/to/schism/model/run
   python /path/to/schism_animation_demo.py --mode boundary

   # Specify files explicitly
   python examples/schism_animation_demo.py \
       --mode boundary \
       --boundary-file SAL_3D.th.nc \
       --grid-file hgrid.gr3 \
       --output salinity_real_data.mp4

   # Multi-variable with real data
   python examples/schism_animation_demo.py \
       --mode multi \
       --output comprehensive_animation.mp4

Integration Examples
-------------------

Jupyter Notebook Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use animations in Jupyter notebooks:

.. code-block:: python

   # Enable inline plotting
   %matplotlib inline

   from rompy.schism.plotting import SchismPlotter
   from rompy.schism.plotting.animation import AnimationConfig

   # Create animation configuration
   config = AnimationConfig(frame_rate=15, show_time_label=True)
   plotter = SchismPlotter(grid_file="hgrid.gr3", animation_config=config)

   # Create and display animation
   anim = plotter.animate_boundary_data("data.nc", "temperature")

   # Animation displays automatically in notebook cell

Model Workflow Integration
~~~~~~~~~~~~~~~~~~~~~~~~~

Integrate animations into model setup workflows:

.. code-block:: python

   from pathlib import Path
   from rompy.model import ModelRun
   from rompy.schism.plotting import SchismPlotter
   from rompy.schism.plotting.animation import AnimationConfig

   # Generate model run
   config = {...}  # Your SCHISM configuration
   model_run = ModelRun(**config)
   staging_dir = model_run.generate()

   # Create animations from generated files
   animation_config = AnimationConfig(frame_rate=15, quality='high')
   plotter = SchismPlotter(
       grid_file=staging_dir / "hgrid.gr3",
       animation_config=animation_config
   )

   # Animate boundary conditions
   boundary_files = {
       'Salinity': staging_dir / 'SAL_3D.th.nc',
       'Temperature': staging_dir / 'TEM_3D.th.nc'
   }

   for name, file_path in boundary_files.items():
       if file_path.exists():
           anim = plotter.animate_boundary_data(
               file_path,
               name.lower(),
               f"{name.lower()}_animation.mp4"
           )

Batch Processing
~~~~~~~~~~~~~~~

Create multiple animations automatically:

.. code-block:: python

   import os
   from pathlib import Path

   # Define animation parameters
   animations = [
       {
           'file': 'SAL_3D.th.nc',
           'variable': 'salinity',
           'output': 'salinity.mp4',
           'cmap': 'viridis'
       },
       {
           'file': 'TEM_3D.th.nc',
           'variable': 'temperature',
           'output': 'temperature.mp4',
           'cmap': 'RdYlBu_r'
       }
   ]

   # Create output directory
   output_dir = Path("animations")
   output_dir.mkdir(exist_ok=True)

   # Generate animations
   for anim_config in animations:
       if Path(anim_config['file']).exists():
           anim = plotter.animate_boundary_data(
               anim_config['file'],
               anim_config['variable'],
               output_dir / anim_config['output'],
               cmap=anim_config['cmap']
           )
           print(f"Created {anim_config['output']}")

Performance Optimization
------------------------

Large Dataset Handling
~~~~~~~~~~~~~~~~~~~~~~

Strategies for handling large datasets efficiently:

.. code-block:: python

   # 1. Reduce temporal resolution
   config = AnimationConfig(
       time_step=5,        # use every 5th time step
       frame_rate=10       # moderate frame rate
   )

   # 2. Subset time ranges
   config = AnimationConfig(
       time_start="2023-07-01T00:00:00",
       time_end="2023-07-01T12:00:00",  # 12 hours only
       frame_rate=15
   )

   # 3. Lower quality for preview
   config = AnimationConfig(
       quality='low',
       figsize=(8, 6),     # smaller figure size
       frame_rate=8
   )

Memory Management
~~~~~~~~~~~~~~~~

Best practices for memory efficiency:

.. code-block:: python

   # Close figures after animation creation
   anim = plotter.animate_boundary_data("data.nc", "temperature")
   plt.close('all')  # Free memory

   # Use context managers for batch processing
   with plt.ioff():  # Turn off interactive plotting
       for data_file in data_files:
           anim = plotter.animate_boundary_data(data_file, "variable")
           plt.close('all')

Quality vs Performance Trade-offs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Preview quality (fast)
   preview_config = AnimationConfig(
       frame_rate=8,
       quality='low',
       figsize=(8, 6),
       time_step=3
   )

   # Production quality (slower)
   production_config = AnimationConfig(
       frame_rate=30,
       quality='high',
       figsize=(16, 12),
       bitrate=3600,
       dpi=150
   )

Troubleshooting
--------------

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Memory Errors with Large Datasets**

.. code-block:: python

   # Solution: Reduce temporal resolution
   config = AnimationConfig(time_step=10, quality='low')

**Slow Rendering Performance**

.. code-block:: python

   # Solution: Lower frame rate and figure size
   config = AnimationConfig(frame_rate=8, figsize=(8, 6))

**Missing Geographic Features**

.. code-block:: bash

   # Solution: Install cartopy
   pip install cartopy

**Animation Not Saving**

.. code-block:: python

   # Check file permissions and disk space
   import os
   output_file = "animation.mp4"
   output_dir = os.path.dirname(output_file)

   if not os.path.exists(output_dir):
       os.makedirs(output_dir)

   if not os.access(output_dir, os.W_OK):
       print("No write permission for output directory")

**FFmpeg Not Found (for MP4)**

.. code-block:: bash

   # Install ffmpeg
   conda install ffmpeg
   # or
   pip install ffmpeg-python

**Coordinate Mismatch Errors**

This error typically occurs with boundary data files that have multi-dimensional structure. The animation system automatically handles this by properly slicing 3D boundary data.

.. code-block:: python

   # The system automatically handles:
   # - Time dimension selection
   # - Level selection for 3D data
   # - Component selection for multi-component data

   # If you encounter issues, specify the level explicitly:
   anim = plotter.animate_boundary_data(
       "SAL_3D.th.nc",
       "salinity",
       level_idx=0  # surface level
   )

Debugging Tips
~~~~~~~~~~~~~

.. code-block:: python

   # Enable verbose logging
   import logging
   logging.basicConfig(level=logging.DEBUG)

   # Test with synthetic data first
   python examples/schism_animation_demo.py --mode boundary --demo-data --show

   # Check data file structure
   import xarray as xr
   ds = xr.open_dataset("your_data_file.nc")
   print(ds)  # Examine dimensions and variables

   # Verify grid file compatibility
   from rompy.schism.plotting.utils import validate_file_exists
   print(validate_file_exists("hgrid.gr3"))

Dependencies and Requirements
----------------------------

Required Packages
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Core dependencies
   pip install matplotlib numpy xarray pydantic

   # Optional but recommended
   pip install cartopy          # geographic projections
   pip install ffmpeg-python    # MP4 encoding
   pip install pillow          # GIF creation
   pip install tqdm            # progress bars

System Requirements
~~~~~~~~~~~~~~~~~~

* **Python**: 3.8 or higher
* **Memory**: 4GB+ recommended for large datasets
* **Disk Space**: Sufficient space for output videos (can be large)
* **Graphics**: No special requirements (uses matplotlib backend)

Installation Verification
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Test basic functionality
   from rompy.schism.plotting.animation import AnimationConfig, AnimationPlotter

   # Test with demo data
   python examples/schism_animation_demo.py --mode boundary --demo-data --show

API Reference
-------------

Core Classes
~~~~~~~~~~~~

.. autoclass:: rompy.schism.plotting.animation.AnimationPlotter
   :members:
   :undoc-members:

.. autoclass:: rompy.schism.plotting.animation.AnimationConfig
   :members:
   :undoc-members:

Integration Methods
~~~~~~~~~~~~~~~~~~

Animation methods are integrated into the main SchismPlotter class:

.. automethod:: rompy.schism.plotting.SchismPlotter.animate_boundary_data

.. automethod:: rompy.schism.plotting.SchismPlotter.animate_atmospheric_data

.. automethod:: rompy.schism.plotting.SchismPlotter.animate_grid_data

.. automethod:: rompy.schism.plotting.SchismPlotter.create_multi_variable_animation

Best Practices
--------------

Animation Design
~~~~~~~~~~~~~~~

1. **Choose Appropriate Frame Rates**
   - 8-12 fps: Sufficient for most scientific animations
   - 15-20 fps: Smooth for presentation
   - 30+ fps: Only for high-quality production

2. **Select Suitable Colormaps**
   - Use perceptually uniform colormaps (viridis, plasma, cividis)
   - Consider colorblind-friendly options
   - Match colormap to data type (diverging for differences, sequential for magnitudes)

3. **Include Context Information**
   - Enable time labels to show temporal progression
   - Add geographic features (coastlines, gridlines) for spatial context
   - Use descriptive titles and axis labels

File Management
~~~~~~~~~~~~~~

1. **Organize Output Files**
   - Use descriptive filenames with variable and time range
   - Create separate directories for different animation types
   - Include metadata in filenames (e.g., "salinity_2023-07_surface.mp4")

2. **Choose Appropriate Formats**
   - MP4: High quality, smaller file sizes, good for archival
   - GIF: Web-friendly, larger files, good for presentations

3. **Consider Storage Requirements**
   - High-quality animations can be large (100MB+ for complex multi-variable animations)
   - Plan storage capacity accordingly
   - Consider compression settings vs quality trade-offs

Quality Control
~~~~~~~~~~~~~~

1. **Preview Before Production**
   - Create low-quality previews first to verify animation content
   - Test with subset time ranges
   - Verify all data displays correctly

2. **Validate Data Consistency**
   - Check that time series progression makes physical sense
   - Verify coordinate systems are correct
   - Ensure boundary data aligns with grid

3. **Test Different Viewing Conditions**
   - Preview animations at different sizes
   - Test on different devices/screens
   - Consider audience viewing environment

Further Reading
--------------

* :doc:`index` - Main SCHISM plotting documentation
* :doc:`examples` - Additional plotting examples
* :doc:`api_reference` - Complete API documentation
* **Demo Scripts**: ``examples/schism_animation_demo.py`` - Comprehensive demonstration
* **Real Data Examples**: ``docs/ANIMATION_REAL_DATA_USAGE.md`` - Working with actual model output
* **Implementation Details**: ``docs/ANIMATION_IMPLEMENTATION_SUMMARY.md`` - Technical implementation overview
