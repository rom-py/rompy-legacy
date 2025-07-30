===============================
SCHISM Hotstart Configuration
===============================

Overview
========

The SCHISM hotstart system provides a unified way to generate initial condition files (hotstart.nc) for SCHISM simulations. The hotstart functionality is logically integrated with the boundary conditions system, allowing you to generate initial conditions from the same ocean data sources you're already using for boundary forcing.

A hotstart file contains initial values for temperature, salinity, and other model variables at every grid point and vertical level, allowing SCHISM to begin with realistic ocean conditions rather than starting from rest or uniform values.

Key Features
============

* **Integrated Configuration** - Hotstart generation is configured alongside boundary conditions
* **Data Source Reuse** - Automatically uses temperature and salinity sources from boundary conditions
* **No Duplication** - Eliminates the need to specify ocean data sources twice
* **Optional Generation** - Completely optional feature that doesn't interfere with existing workflows
* **Flexible Variables** - Configurable variable names to match different ocean model outputs

Architecture
============

The hotstart functionality is implemented through:

* ``HotstartConfig`` - Configuration class for hotstart parameters
* Integration with ``SCHISMDataBoundaryConditions`` - Hotstart configured alongside boundaries
* Automatic data source detection - Finds temperature and salinity sources from boundary configurations
* ``SCHISMDataHotstart`` backend - Handles the actual file generation and interpolation

HotstartConfig Class
====================

The ``HotstartConfig`` class defines all parameters needed for hotstart file generation:

.. autoclass:: rompy.schism.data.HotstartConfig
   :members:
   :undoc-members:
   :show-inheritance:

Configuration Parameters
-------------------------

=================== ============ ===================================================
Parameter           Default      Description
=================== ============ ===================================================
``enabled``         ``False``    Whether to generate hotstart file
``temp_var``        ``"temperature"`` Name of temperature variable in source dataset
``salt_var``        ``"salinity"``    Name of salinity variable in source dataset  
``time_offset``     ``0.0``      Offset to add to source time values (in days)
``time_base``       ``2000-01-01`` Base time for source time calculations
``output_filename`` ``"hotstart.nc"`` Name of the output hotstart file
=================== ============ ===================================================

Usage Examples
==============

Basic Configuration
-------------------

The simplest way to enable hotstart generation is to add a ``hotstart_config`` section to your boundary conditions:

.. code-block:: yaml

   data:
     boundary_conditions:
       setup_type: "hybrid"
       hotstart_config:
         enabled: true
       boundaries:
         0:
           elev_type: 5  # TIDALSPACETIME
           vel_type: 4   # SPACETIME  
           temp_type: 4  # SPACETIME
           salt_type: 4  # SPACETIME
           temp_source:
             data_type: boundary
             source:
               model_type: file
               uri: ocean_data.nc
             variables: [temperature]
             coords:
               t: time
               x: lon
               y: lat
               z: depth
           salt_source:
             data_type: boundary
             source:
               model_type: file
               uri: ocean_data.nc
             variables: [salinity]
             coords:
               t: time
               x: lon
               y: lat
               z: depth

Custom Variable Names
---------------------

If your ocean data uses different variable names, you can specify them:

.. code-block:: yaml

   hotstart_config:
     enabled: true
     temp_var: "water_temp"
     salt_var: "sal"
     output_filename: "initial_conditions.nc"

Custom Time Configuration
--------------------------

For datasets with specific time reference systems:

.. code-block:: yaml

   hotstart_config:
     enabled: true
     time_base: "1990-01-01"
     time_offset: 0.5  # Add 0.5 days to source times

Python API Usage
================

You can also configure hotstart generation programmatically:

.. code-block:: python

   from rompy.schism.data import SCHISMDataBoundaryConditions, HotstartConfig
   from rompy.schism.boundary_conditions import create_hybrid_boundary_config

   # Create hotstart configuration
   hotstart_config = HotstartConfig(
       enabled=True,
       temp_var="temperature",
       salt_var="salinity",
       output_filename="hotstart.nc"
   )

   # Create boundary conditions with hotstart
   boundary_conditions = create_hybrid_boundary_config(
       tidal_data=tidal_dataset,
       ocean_source=ocean_source,
       hotstart_config=hotstart_config
   )

   # Generate files (including hotstart if enabled)
   result = boundary_conditions.get(destdir, grid, time_range)
   
   # Check if hotstart was generated
   if "hotstart" in result:
       print(f"Hotstart file created: {result['hotstart']}")

Integration with Factory Functions
===================================

All boundary condition factory functions support hotstart configuration:

.. code-block:: python

   from rompy.schism.boundary_conditions import create_hybrid_boundary_config

   # Using factory function with hotstart
   boundary_config = create_hybrid_boundary_config(
       tidal_constituents=["M2", "S2", "N2"],
       tidal_database="tpxo",
       ocean_source=hycom_source,
       hotstart_config=HotstartConfig(enabled=True)
   )

Generated Files
===============

When hotstart generation is enabled, the boundary conditions system will create:

Standard Boundary Files
-----------------------
* ``bctides.in`` - Tidal boundary configuration
* ``elev2D.th.nc`` - Elevation boundary data  
* ``uv3D.th.nc`` - Velocity boundary data
* ``TEM_3D.th.nc`` - Temperature boundary data
* ``SAL_3D.th.nc`` - Salinity boundary data

Hotstart File
-------------
* ``hotstart.nc`` - Initial conditions file containing:
  
  * Temperature and salinity at all grid nodes and vertical levels
  * Zero initial velocities and turbulence variables
  * Proper SCHISM NetCDF format with all required variables

File Structure
--------------

The generated hotstart.nc file contains the standard SCHISM hotstart format:

=================== ========================================
Variable            Description
=================== ========================================
``tr_nd``           Node-based tracers (temperature, salinity)
``tr_el``           Element-based tracers  
``eta2``            Surface elevation
``we``              Vertical velocity
``su2``, ``sv2``    Horizontal velocities at sides
``q2``, ``xl``      Turbulence variables
``dfv``, ``dfh``    Diffusivity variables
``idry``            Dry/wet flags
=================== ========================================

Data Source Requirements
========================

For hotstart generation to work, your boundary conditions must include:

Required Sources
----------------
* **Temperature source** - A boundary data source with temperature variables
* **Salinity source** - A boundary data source with salinity variables

The sources can be:
* Separate files for temperature and salinity
* Same file containing both variables
* Any combination that provides both temperature and salinity data

Coordinate Requirements
-----------------------
* **Time dimension** - For selecting the appropriate time slice
* **Horizontal coordinates** - Longitude/latitude or x/y for spatial interpolation
* **Vertical coordinate** - Depth or sigma levels for vertical interpolation

Example Ocean Data Sources
---------------------------

HYCOM Global Model:

.. code-block:: yaml

   temp_source:
     source:
       model_type: file
       uri: "hycom_global.nc"
     variables: ["water_temp"]
     coords:
       t: time
       x: lon
       y: lat
       z: depth

ROMS Regional Model:

.. code-block:: yaml

   temp_source:
     source:
       model_type: file  
       uri: "roms_output.nc"
     variables: ["temp"]
     coords:
       t: ocean_time
       x: lon_rho
       y: lat_rho
       z: s_rho

Error Handling
==============

The hotstart system includes comprehensive error checking:

Missing Data Sources
--------------------
If hotstart is enabled but temperature or salinity sources are not available:

.. code-block:: text

   ValueError: Hotstart generation requires both temperature and salinity sources 
   to be configured in boundary conditions

Variable Name Mismatches
------------------------
If the specified variable names don't exist in the source data, the system will attempt to find alternative names or report an error with suggestions.

Time Range Issues
-----------------
If the requested time is outside the available data range, the system will use the closest available time and issue a warning.

Best Practices
==============

1. **Use Same Data Sources**
   
   Configure hotstart to use the same ocean model data you're using for boundary conditions to ensure consistency.

2. **Check Variable Names**
   
   Verify that ``temp_var`` and ``salt_var`` match the actual variable names in your ocean data files.

3. **Time Alignment**
   
   Ensure your hotstart time aligns with your simulation start time for optimal initial conditions.

4. **Grid Resolution**
   
   Higher resolution ocean data will provide better interpolated initial conditions, especially in coastal areas.

5. **Validation**
   
   Always check the generated hotstart.nc file to ensure reasonable temperature and salinity ranges for your domain.

Troubleshooting
===============

Common Issues and Solutions
---------------------------

**Hotstart file not generated**
   Check that ``enabled: true`` is set in hotstart_config

**Variable not found errors**
   Verify variable names match your ocean data using ``temp_var`` and ``salt_var`` parameters

**Interpolation warnings**
   Normal for coastal areas - the system will use nearest neighbor interpolation for missing data

**Large file sizes**
   Hotstart files can be large for high-resolution grids - this is normal

**Time coordinate issues**
   Adjust ``time_base`` and ``time_offset`` to match your ocean data's time reference

Migration from Legacy Hotstart
===============================

If you were previously using the standalone ``SCHISMDataHotstart`` class:

Old Configuration:

.. code-block:: yaml

   data:
     hotstart:
       source:
         model_type: file
         uri: ocean_data.nc
       temp_var: temperature
       salt_var: salinity
       coords:
         t: time
         x: lon
         y: lat
         z: depth

New Integrated Configuration:

.. code-block:: yaml

   data:
     boundary_conditions:
       hotstart_config:
         enabled: true
         temp_var: temperature
         salt_var: salinity
       boundaries:
         0:
           temp_source:
             source:
               model_type: file
               uri: ocean_data.nc
             variables: [temperature]
             coords:
               t: time
               x: lon
               y: lat
               z: depth
           salt_source:
             source:
               model_type: file
               uri: ocean_data.nc
             variables: [salinity]
             coords:
               t: time
               x: lon
               y: lat
               z: depth

The new approach eliminates data source duplication and creates a more logical configuration structure.

See Also
========

* :doc:`enhanced_tides` - Boundary conditions documentation
* :doc:`../core_concepts` - Core ROMPY concepts
* `SCHISM Manual <https://schism-dev.github.io/schism/master/index.html>`_ - Official SCHISM documentation