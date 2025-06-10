=============================
SCHISM Boundary Conditions
=============================

Overview
========

The SCHISM boundary conditions system provides a unified interface for configuring all types of boundary conditions in SCHISM simulations. This system replaces the previous separate tidal and ocean configurations with a single, flexible approach that supports:

- **Harmonic boundaries** - Pure harmonic tidal forcing using tidal constituents
- **Hybrid boundaries** - Combined harmonic and external data forcing  
- **River boundaries** - Constant or time-varying river inputs
- **Nested boundaries** - Coupling with parent model outputs
- **Custom configurations** - Flexible mixing of different boundary types

Key Classes
===========

SCHISMDataBoundaryConditions
-----------------------------

The main class for configuring boundary conditions. This unified interface handles all boundary types and their associated data sources.

.. autoclass:: rompy.schism.data.SCHISMDataBoundaryConditions
   :members:
   :undoc-members:
   :show-inheritance:

BoundarySetupWithSource
-----------------------

Configures individual boundary segments with their data sources and boundary condition types.

.. autoclass:: rompy.schism.data.BoundarySetupWithSource
   :members:
   :undoc-members:
   :show-inheritance:

BoundaryHandler
---------------

Core boundary handler that extends BoundaryData and supports all SCHISM boundary types.

.. autoclass:: rompy.schism.boundary_core.BoundaryHandler
   :members:
   :undoc-members:
   :show-inheritance:

BoundaryConfig
--------------

Configuration for individual boundary segments.

.. autoclass:: rompy.schism.boundary_core.BoundaryConfig
   :members:
   :undoc-members:
   :show-inheritance:

Boundary Type Enums
====================

ElevationType
-------------

.. autoclass:: rompy.schism.boundary_core.ElevationType
   :members:
   :undoc-members:

VelocityType
------------

.. autoclass:: rompy.schism.boundary_core.VelocityType
   :members:
   :undoc-members:

TracerType
----------

.. autoclass:: rompy.schism.boundary_core.TracerType
   :members:
   :undoc-members:

Factory Functions
=================

The boundary conditions module provides convenient factory functions for creating common boundary configurations. These functions return ``SCHISMDataBoundaryConditions`` objects that can be directly used in SCHISM simulations.

High-Level Configuration Functions
----------------------------------

create_tidal_only_boundary_config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: rompy.schism.boundary_conditions.create_tidal_only_boundary_config

**Example Usage:**

.. code-block:: python

    from rompy.schism.boundary_conditions import create_tidal_only_boundary_config
    
    # Basic tidal configuration
    bc = create_tidal_only_boundary_config(
        constituents=["M2", "S2", "N2", "K1", "O1"],
        tidal_elevations="/path/to/h_tpxo9.nc",
        tidal_velocities="/path/to/u_tpxo9.nc"
    )
    
    # With earth tidal potential
    bc = create_tidal_only_boundary_config(
        constituents=["M2", "S2", "K1", "O1"],
        ntip=1  # Enable earth tidal potential
    )

create_hybrid_boundary_config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: rompy.schism.boundary_conditions.create_hybrid_boundary_config

**Example Usage:**

.. code-block:: python

    from rompy.schism.boundary_conditions import create_hybrid_boundary_config
    from rompy.core.data import DataBlob
    
    # Hybrid configuration with external data
    bc = create_hybrid_boundary_config(
        constituents=["M2", "S2"],
        tidal_elevations="/path/to/h_tpxo9.nc",
        tidal_velocities="/path/to/u_tpxo9.nc",
        elev_source=DataBlob(source="/path/to/elev2D.th.nc"),
        vel_source=DataBlob(source="/path/to/uv3D.th.nc"),
        temp_source=DataBlob(source="/path/to/TEM_3D.th.nc"),
        salt_source=DataBlob(source="/path/to/SAL_3D.th.nc")
    )

create_river_boundary_config
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: rompy.schism.boundary_conditions.create_river_boundary_config

**Example Usage:**

.. code-block:: python

    from rompy.schism.boundary_conditions import create_river_boundary_config
    
    # River boundary with tidal forcing on other boundaries
    bc = create_river_boundary_config(
        river_boundary_index=1,
        river_flow=-500.0,  # 500 m³/s inflow
        river_temp=15.0,    # 15°C
        river_salt=0.1,     # 0.1 PSU (fresh water)
        other_boundaries="tidal",
        constituents=["M2", "S2", "N2"]
    )
    
    # River-only configuration
    bc = create_river_boundary_config(
        river_boundary_index=0,
        river_flow=-200.0,
        other_boundaries="none"
    )

create_nested_boundary_config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: rompy.schism.boundary_conditions.create_nested_boundary_config

**Example Usage:**

.. code-block:: python

    from rompy.schism.boundary_conditions import create_nested_boundary_config
    from rompy.schism.data import SCHISMDataBoundary
    from rompy.core.source import SourceFile
    
    # Nested boundary with tides and parent model data
    bc = create_nested_boundary_config(
        with_tides=True,
        inflow_relax=0.9,
        outflow_relax=0.1,
        constituents=["M2", "S2"],
        elev_source=SCHISMDataBoundary(
            source=SourceFile(uri="/path/to/parent_model.nc"),
            variables=["ssh"]
        ),
        vel_source=SCHISMDataBoundary(
            source=SourceFile(uri="/path/to/parent_model.nc"),
            variables=["u", "v"]
        )
    )
    
    # Nested boundary without tides
    bc = create_nested_boundary_config(
        with_tides=False,
        inflow_relax=0.8,
        outflow_relax=0.2,
        elev_source=elev_data,
        vel_source=vel_data
    )

Low-Level Boundary Creation Functions
-------------------------------------

These functions create ``BoundaryHandler`` objects for direct grid-based boundary manipulation:

.. autofunction:: rompy.schism.boundary_core.create_tidal_boundary
.. autofunction:: rompy.schism.boundary_core.create_hybrid_boundary
.. autofunction:: rompy.schism.boundary_core.create_river_boundary
.. autofunction:: rompy.schism.boundary_core.create_nested_boundary

Usage Examples
==============

Tidal-Only Configuration
-------------------------

For simulations with purely tidal forcing:

.. code-block:: python

    from rompy.schism.boundary_conditions import create_tidal_only_boundary_config
    from rompy.schism.data import SCHISMData
    
    # Create tidal-only boundary configuration
    boundary_conditions = create_tidal_only_boundary_config(
        constituents=["M2", "S2", "N2", "K1", "O1"],
        tidal_database="tpxo",
        tidal_elevations="path/to/tidal_elevations.nc",
        tidal_velocities="path/to/tidal_velocities.nc",
    )
    
    # Use in SCHISM configuration
    schism_data = SCHISMData(
        boundary_conditions=boundary_conditions,
    )

Hybrid Tidal + Ocean Data
--------------------------

For simulations combining tidal forcing with external ocean data:

.. code-block:: python

    from rompy.schism.boundary_conditions import create_hybrid_boundary_config
    from rompy.core.data import DataBlob
    
    # Create hybrid boundary configuration
    boundary_conditions = create_hybrid_boundary_config(
        constituents=["M2", "S2"],
        tidal_elevations="path/to/tidal_elevations.nc",
        tidal_velocities="path/to/tidal_velocities.nc",
        # Add ocean data sources
        elev_source=DataBlob(source="path/to/elev2D.th.nc"),
        vel_source=DataBlob(source="path/to/uv3D.th.nc"),
        temp_source=DataBlob(source="path/to/TEM_3D.th.nc"),
        salt_source=DataBlob(source="path/to/SAL_3D.th.nc"),
    )

River Boundary Configuration
----------------------------

For simulations with river inputs:

.. code-block:: python

    from rompy.schism.boundary_conditions import create_river_boundary_config
    
    # Create river boundary configuration
    boundary_conditions = create_river_boundary_config(
        river_boundary_index=1,     # Index of the river boundary
        river_flow=-100.0,          # Negative for inflow (m³/s)
        other_boundaries="tidal",   # Other boundaries are tidal
        constituents=["M2", "S2"],
        tidal_elevations="path/to/tidal_elevations.nc",
        tidal_velocities="path/to/tidal_velocities.nc",
    )

Nested Model Configuration
--------------------------

For simulations nested within a larger model:

.. code-block:: python

    from rompy.schism.boundary_conditions import create_nested_boundary_config
    from rompy.schism.data import SCHISMDataBoundary
    from rompy.core.source import SourceFile
    
    # Create nested boundary configuration
    boundary_conditions = create_nested_boundary_config(
        with_tides=True,
        inflow_relax=0.8,
        outflow_relax=0.2,
        constituents=["M2", "S2"],
        tidal_elevations="path/to/tidal_elevations.nc",
        tidal_velocities="path/to/tidal_velocities.nc",
        # Add parent model data sources
        elev_source=SCHISMDataBoundary(
            source=SourceFile(uri="path/to/parent_model.nc"),
            variables=["ssh"],
        ),
        vel_source=SCHISMDataBoundary(
            source=SourceFile(uri="path/to/parent_model.nc"),
            variables=["u", "v"],
        ),
    )

Direct Boundary Handler Usage
-----------------------------

For maximum control, use the BoundaryHandler class directly:

.. code-block:: python

    from rompy.schism.boundary_core import (
        BoundaryHandler, 
        ElevationType, 
        VelocityType, 
        TracerType
    )
    
    # Create boundary handler
    boundary = BoundaryHandler(
        grid_path="path/to/hgrid.gr3",
        constituents=["M2", "S2", "K1", "O1"],
        tidal_database="tpxo",
        tidal_elevations="path/to/h_tpxo9.nc",
        tidal_velocities="path/to/uv_tpxo9.nc"
    )
    
    # Configure different boundary types
    boundary.set_boundary_type(
        0,  # Ocean boundary with tides
        elev_type=ElevationType.HARMONIC,
        vel_type=VelocityType.HARMONIC
    )
    
    boundary.set_boundary_type(
        1,  # River boundary
        elev_type=ElevationType.NONE,
        vel_type=VelocityType.CONSTANT,
        vthconst=-500.0  # River inflow
    )
    
    # Set simulation parameters and write output
    boundary.set_run_parameters(start_time, run_days)
    boundary.write_boundary_file("path/to/bctides.in")

Custom Boundary Configuration
-----------------------------

For complex scenarios with mixed boundary types:

.. code-block:: python

    from rompy.schism.data import SCHISMDataBoundaryConditions, BoundarySetupWithSource
    from rompy.schism.boundary_core import ElevationType, VelocityType, TracerType
    from rompy.core.data import DataBlob
    
    # Create custom boundary configuration
    boundary_conditions = SCHISMDataBoundaryConditions(
        constituents=["M2", "S2"],
        tidal_database="tpxo",
        boundaries={
            # Ocean boundary (harmonic + external data)
            0: BoundarySetupWithSource(
                elev_type=ElevationType.HARMONICEXTERNAL,
                vel_type=VelocityType.HARMONICEXTERNAL,
                temp_type=TracerType.EXTERNAL,
                salt_type=TracerType.EXTERNAL,
                elev_source=DataBlob(source="path/to/elev2D.th.nc"),
                vel_source=DataBlob(source="path/to/uv3D.th.nc"),
                temp_source=DataBlob(source="path/to/TEM_3D.th.nc"),
                salt_source=DataBlob(source="path/to/SAL_3D.th.nc"),
            ),
            # River boundary (constant flow)
            1: BoundarySetupWithSource(
                elev_type=ElevationType.NONE,
                vel_type=VelocityType.CONSTANT,
                temp_type=TracerType.CONSTANT,
                salt_type=TracerType.CONSTANT,
                const_flow=-100.0,  # m³/s, negative for inflow
                const_temp=15.0,    # °C
                const_salt=0.5,     # PSU
            ),
        }
    )

Boundary Types
==============

The system supports various boundary condition types for different variables:

Elevation Types
---------------

- **NONE** - No elevation boundary condition
- **TIMEHIST** - Time history from elev.th
- **CONSTANT** - Constant elevation
- **HARMONIC** - Pure harmonic tidal elevation using tidal constituents
- **EXTERNAL** - Time-varying elevation from external data (elev2D.th.nc)
- **HARMONICEXTERNAL** - Combined harmonic and external elevation data

Velocity Types
--------------

- **NONE** - No velocity boundary condition
- **TIMEHIST** - Time history from flux.th
- **CONSTANT** - Constant velocity/flow rate
- **HARMONIC** - Pure harmonic tidal velocity using tidal constituents
- **EXTERNAL** - Time-varying velocity from external data (uv3D.th.nc)
- **HARMONICEXTERNAL** - Combined harmonic and external velocity data
- **FLATHER** - Flather type radiation boundary
- **RELAXED** - Relaxation boundary condition (for nesting)

Tracer Types
------------

- **NONE** - No tracer boundary condition
- **TIMEHIST** - Time history from temp/salt.th
- **CONSTANT** - Constant tracer value
- **INITIAL** - Initial profile for inflow
- **EXTERNAL** - Time-varying tracer from external data

Data Sources
============

The system supports multiple data source types:

DataBlob
--------

Simple file-based data source for pre-processed SCHISM input files:

.. code-block:: python

    from rompy.core.data import DataBlob
    
    elev_source = DataBlob(source="path/to/elev2D.th.nc")

SCHISMDataBoundary
------------------

Advanced data source with variable mapping and coordinate transformation:

.. code-block:: python

    from rompy.schism.data import SCHISMDataBoundary
    from rompy.core.source import SourceFile
    
    vel_source = SCHISMDataBoundary(
        source=SourceFile(uri="path/to/ocean_model.nc"),
        variables=["u", "v"],
        crop_coords={"lon": [-180, 180], "lat": [-90, 90]},
    )

Configuration Files
===================

The boundary conditions can also be configured via YAML files:

**Tidal-Only Configuration:**

.. code-block:: yaml

    boundary_conditions:
      data_type: boundary_conditions
      constituents: ["M2", "S2", "N2", "K1", "O1"]
      tidal_database: tpxo
      tidal_data:
        elevations: path/to/h_tpxo9.nc
        velocities: path/to/u_tpxo9.nc
      setup_type: tidal

**Hybrid Configuration:**

.. code-block:: yaml

    boundary_conditions:
      data_type: boundary_conditions
      constituents: ["M2", "S2", "N2", "K1", "O1"]
      tidal_database: tpxo
      tidal_data:
        elevations: path/to/h_tpxo9.nc
        velocities: path/to/u_tpxo9.nc
      setup_type: hybrid
      boundaries:
        0:
          elev_type: HARMONICEXTERNAL
          vel_type: HARMONICEXTERNAL
          temp_type: EXTERNAL
          salt_type: EXTERNAL
          elev_source:
            data_type: blob
            source: path/to/elev2D.th.nc
          vel_source:
            data_type: blob
            source: path/to/uv3D.th.nc
          temp_source:
            data_type: blob
            source: path/to/TEM_3D.th.nc
          salt_source:
            data_type: blob
            source: path/to/SAL_3D.th.nc

**River Configuration:**

.. code-block:: yaml

    boundary_conditions:
      data_type: boundary_conditions
      constituents: ["M2", "S2"]
      tidal_database: tpxo
      setup_type: river
      boundaries:
        0:  # Tidal boundary
          elev_type: HARMONIC
          vel_type: HARMONIC
          temp_type: NONE
          salt_type: NONE
        1:  # River boundary
          elev_type: NONE
          vel_type: CONSTANT
          temp_type: CONSTANT
          salt_type: CONSTANT
          const_flow: -500.0
          const_temp: 15.0
          const_salt: 0.1

**Nested Configuration:**

.. code-block:: yaml

    boundary_conditions:
      data_type: boundary_conditions
      constituents: ["M2", "S2"]
      tidal_database: tpxo
      tidal_data:
        elevations: path/to/h_tpxo9.nc
        velocities: path/to/u_tpxo9.nc
      setup_type: nested
      boundaries:
        0:
          elev_type: HARMONICEXTERNAL
          vel_type: RELAXED
          temp_type: EXTERNAL
          salt_type: EXTERNAL
          inflow_relax: 0.8
          outflow_relax: 0.2
          elev_source:
            data_type: schism_boundary
            source:
              data_type: source_file
              uri: path/to/parent_model.nc
            variables: ["ssh"]
          vel_source:
            data_type: schism_boundary
            source:
              data_type: source_file
              uri: path/to/parent_model.nc
            variables: ["u", "v"]



Benefits of the New System
==========================

1. **Unified Interface** - Single configuration object for all boundary types
2. **Flexible Configuration** - Mix different boundary types per segment
3. **Factory Functions** - Simplified setup for common scenarios
4. **Better Validation** - Comprehensive validation of boundary configurations
5. **Data Source Integration** - Seamless integration with data processing pipeline
6. **Backward Compatibility** - Maintains compatibility with existing workflows where possible
7. **Clear Naming** - Module and class names reflect actual functionality
8. **Consolidated Code** - Eliminates duplication between modules

Advanced Features
=================

Factory Function Parameters
---------------------------

All factory functions support additional parameters for fine-tuning:

**Common Parameters:**

- ``constituents``: List of tidal constituents (e.g., ["M2", "S2", "N2", "K1", "O1"])
- ``tidal_database``: Database identifier ("tpxo", "fes2014", "got")
- ``tidal_elevations``: Path to tidal elevation NetCDF file
- ``tidal_velocities``: Path to tidal velocity NetCDF file

**Tidal Potential:**

.. code-block:: python

    bc = create_tidal_only_boundary_config(
        constituents=["M2", "S2", "K1", "O1"],
        ntip=1,          # Enable tidal potential
        tip_dp=1.0,      # Depth threshold
        cutoff_depth=50.0,  # Cutoff depth
    )

**Relaxation Parameters:**

.. code-block:: python

    bc = create_nested_boundary_config(
        with_tides=True,
        inflow_relax=0.8,   # Strong relaxation for inflow
        outflow_relax=0.2,  # Weak relaxation for outflow
    )

**Multiple Tidal Databases:**

.. code-block:: python

    bc = create_tidal_only_boundary_config(
        tidal_database="fes2014",  # Alternative: "tpxo", "got"
        constituents=["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1"],
    )

**Custom Boundary Types:**

.. code-block:: python

    from rompy.schism.data import BoundarySetupWithSource
    from rompy.schism.boundary_core import ElevationType, VelocityType, TracerType
    
    # Custom boundary with specific types
    custom_boundary = BoundarySetupWithSource(
        elev_type=ElevationType.HARMONICEXTERNAL,
        vel_type=VelocityType.HARMONICEXTERNAL,
        temp_type=TracerType.EXTERNAL,
        salt_type=TracerType.EXTERNAL,
        inflow_relax=0.9,
        outflow_relax=0.1
    )

Flather Radiation Boundaries
----------------------------

Configure Flather radiation boundaries using the low-level BoundaryHandler:

.. code-block:: python

    from rompy.schism.boundary_core import BoundaryHandler, ElevationType, VelocityType
    
    # Create boundary handler
    boundary = BoundaryHandler(grid_path="path/to/hgrid.gr3")
    
    # Configure Flather boundary
    boundary.set_boundary_type(
        boundary_index=1,
        elev_type=ElevationType.NONE,
        vel_type=VelocityType.FLATHER,
        eta_mean=[0.0, 0.0, 0.0],        # Mean elevation at each node
        vn_mean=[[0.1], [0.1], [0.1]]    # Mean normal velocity at each node
    )

Common Tidal Constituents
-------------------------

**Major Constituents (recommended for most applications):**

- M2, S2, N2, K1, O1

**Semi-diurnal:**

- M2 (Principal lunar), S2 (Principal solar), N2 (Lunar elliptic), K2 (Lunisolar)

**Diurnal:**

- K1 (Lunar diurnal), O1 (Lunar principal), P1 (Solar principal), Q1 (Larger lunar elliptic)

**Long Period:**

- Mf (Lunisolar fortnightly), Mm (Lunar monthly), Ssa (Solar semiannual)

**Full Set Example:**

.. code-block:: python

    bc = create_tidal_only_boundary_config(
        constituents=[
            "M2", "S2", "N2", "K2",     # Semi-diurnal
            "K1", "O1", "P1", "Q1",     # Diurnal  
            "Mf", "Mm", "Ssa"           # Long period
        ]
    )

Best Practices
--------------

1. **Start Simple**: Begin with tidal-only configurations using major constituents
2. **Validate Data**: Ensure tidal and external data files cover your model domain and time period
3. **Check Units**: River flows are in m³/s (negative for inflow)
4. **Relaxation Values**: Use 0.8-1.0 for strong nudging, 0.1-0.3 for weak nudging
5. **File Formats**: Use NetCDF files for better performance and metadata
6. **Coordinate Systems**: Ensure all data sources use consistent coordinate systems
7. **Time Coverage**: External data must cover the entire simulation period plus spin-up

See Also
========

- :doc:`../core/data` - Core data handling classes
- :doc:`../core/boundary` - Base boundary condition classes
- :class:`rompy.schism.data.SCHISMData` - Main SCHISM configuration class
- :class:`rompy.schism.grid.SCHISMGrid` - SCHISM grid handling
- :doc:`hotstart` - Hotstart configuration documentation