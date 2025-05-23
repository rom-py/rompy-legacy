Enhanced Tidal Boundary Handling in SCHISM
======================================

Overview
--------

The enhanced tides module in ROMPY provides an improved approach to handling SCHISM tidal data with comprehensive support for all boundary condition types specified in the SCHISM documentation. This module extends the functionality of the standard tides implementation with more flexible configuration options and better support for complex boundary setups.

Key Features
-----------

* Support for all SCHISM boundary types (tidal, constant, space-time, hybrid, etc.)
* Simplified configuration for common boundary scenarios
* Factory functions for creating standard boundary configurations
* Improved handling of multiple boundary segments with different types
* Support for Flather boundary conditions with mean elevation and velocity
* Enhanced relaxation parameter control for nested model boundaries

Enhanced Tides vs. Standard Tides
---------------------------------

The enhanced tides implementation differs from the standard tides implementation in several key ways:

+---------------------------+-------------------------------------+-------------------------------------+
| Feature                   | Standard Tides                      | Enhanced Tides                      |
+===========================+=====================================+=====================================+
| Boundary Type Support     | Limited to basic tidal boundaries   | All boundary types supported        |
+---------------------------+-------------------------------------+-------------------------------------+
| Configuration Flexibility | Simple configuration                | Highly customizable                 |
+---------------------------+-------------------------------------+-------------------------------------+
| Multiple Boundaries       | Limited support                     | Full support for different types    |
+---------------------------+-------------------------------------+-------------------------------------+
| Factory Functions         | Not available                       | Available for common configurations |
+---------------------------+-------------------------------------+-------------------------------------+
| Flather Boundaries        | Limited support                     | Full support with mean values       |
+---------------------------+-------------------------------------+-------------------------------------+
| Relaxation Parameters     | Basic support                       | Advanced control                    |
+---------------------------+-------------------------------------+-------------------------------------+
| Configuration Approach    | Numeric flags                       | Enum-based types                    |
+---------------------------+-------------------------------------+-------------------------------------+

From Flags to Enum Types
------------------------

One of the significant changes in the enhanced tides implementation is the replacement of the numeric flags system with a more intuitive enum-based approach:

**Standard Tides (Old Approach)**

The standard tides implementation used numeric flags to specify boundary types:

.. code-block:: python

    from rompy.schism.data import SCHISMDataTides
    
    # Using numeric flags
    tides = SCHISMDataTides(
        constituents=["M2", "S2"],
        elev_flags=[5],  # 5 = tidal elevation
        vel_flags=[5]    # 5 = tidal velocity
    )

This approach required users to remember what each numeric flag represented, which could be error-prone and less intuitive.

**Enhanced Tides (New Approach)**

The enhanced tides implementation replaces numeric flags with enum types for better readability and type safety:

.. code-block:: python

    from rompy.schism.tides_enhanced import SCHISMDataTidesEnhanced, BoundarySetup
    from rompy.schism.boundary_tides import ElevationType, VelocityType, TracerType
    
    # Using enum types
    tides = SCHISMDataTidesEnhanced(
        constituents=["M2", "S2"],
        boundaries={
            0: BoundarySetup(
                elev_type=ElevationType.TIDAL,  # Instead of 5
                vel_type=VelocityType.TIDAL     # Instead of 5
            )
        }
    )

The enum types provide several advantages:

1. **Self-documenting code**: The enum names clearly indicate what each boundary type does.
2. **Type safety**: The compiler/interpreter can catch invalid types at development time.
3. **IDE support**: Modern IDEs can provide auto-completion and documentation for enum values.
4. **Extensibility**: New boundary types can be added to the enums without changing existing code.

Boundary Type Enumerations
-------------------------

The enhanced tides implementation uses enum classes to represent boundary types, replacing the numeric flags used in the standard implementation. These enum classes are defined in ``boundary_tides.py``:

**ElevationType**: Represents elevation boundary condition types

.. code-block:: python

    class ElevationType(IntEnum):
        NONE = 0            # Not specified
        TIMEHIST = 1        # Time history from elev.th
        CONSTANT = 2        # Constant elevation
        TIDAL = 3           # Tidal elevation
        SPACETIME = 4       # Space and time-varying from elev2D.th.nc
        TIDALSPACETIME = 5  # Combination of tide and external file

**VelocityType**: Represents velocity boundary condition types

.. code-block:: python

    class VelocityType(IntEnum):
        NONE = 0            # Not specified
        TIMEHIST = 1        # Time history from flux.th
        CONSTANT = 2        # Constant discharge
        TIDAL = 3           # Tidal velocity
        SPACETIME = 4       # Space and time-varying from uv3D.th.nc
        TIDALSPACETIME = 5  # Combination of tide and external file
        FLATHER = -1        # Flather type radiation boundary
        RELAXED = -4        # 3D input with relaxation

**TracerType**: Represents temperature/salinity boundary condition types

.. code-block:: python

    class TracerType(IntEnum):
        NONE = 0        # Not specified
        TIMEHIST = 1    # Time history from temp/salt.th
        CONSTANT = 2    # Constant temperature/salinity
        INITIAL = 3     # Initial profile for inflow
        SPACETIME = 4   # 3D input

These enum types provide better code readability and type safety compared to the numeric flags used in the standard implementation. When working with the enhanced tides module, you should import and use these enum types directly:

.. code-block:: python

    from rompy.schism.boundary_tides import ElevationType, VelocityType, TracerType
    
    # Configure a boundary with specific types
    boundary_setup = BoundarySetup(
        elev_type=ElevationType.TIDAL,
        vel_type=VelocityType.RELAXED,
        temp_type=TracerType.SPACETIME,
        salt_type=TracerType.SPACETIME
    )

**Mapping Between Old Flags and New Enum Types**

For users familiar with the old flags system, here's how the numeric flags map to the new enum types:

*Elevation Types:*

+-------+-------------------------+---------------------------+
| Flag  | Description             | Enum Value                |
+=======+=========================+===========================+
| 0     | No elevation BC         | ElevationType.NONE        |
+-------+-------------------------+---------------------------+
| 1     | Time history            | ElevationType.TIMEHIST    |
+-------+-------------------------+---------------------------+
| 2     | Constant elevation      | ElevationType.CONSTANT    |
+-------+-------------------------+---------------------------+
| 3     | Tidal elevation         | ElevationType.TIDAL       |
+-------+-------------------------+---------------------------+
| 4     | Space-time elevation    | ElevationType.SPACETIME   |
+-------+-------------------------+---------------------------+
| 5     | Tidal + space-time      | ElevationType.TIDALSPACETIME |
+-------+-------------------------+---------------------------+

*Velocity Types:*

+-------+-------------------------+---------------------------+
| Flag  | Description             | Enum Value                |
+=======+=========================+===========================+
| -4    | Relaxed velocity        | VelocityType.RELAXED      |
+-------+-------------------------+---------------------------+
| -1    | Flather velocity        | VelocityType.FLATHER      |
+-------+-------------------------+---------------------------+
| 0     | No velocity BC          | VelocityType.NONE         |
+-------+-------------------------+---------------------------+
| 1     | Time history            | VelocityType.TIMEHIST     |
+-------+-------------------------+---------------------------+
| 2     | Constant discharge      | VelocityType.CONSTANT     |
+-------+-------------------------+---------------------------+
| 3     | Tidal velocity          | VelocityType.TIDAL        |
+-------+-------------------------+---------------------------+
| 4     | Space-time velocity     | VelocityType.SPACETIME    |
+-------+-------------------------+---------------------------+
| 5     | Tidal + space-time      | VelocityType.TIDALSPACETIME |
+-------+-------------------------+---------------------------+

*Tracer Types (Temperature and Salinity):*

+-------+-------------------------+---------------------------+
| Flag  | Description             | Enum Value                |
+=======+=========================+===========================+
| 0     | No tracer BC            | TracerType.NONE           |
+-------+-------------------------+---------------------------+
| 1     | Time history            | TracerType.TIMEHIST       |
+-------+-------------------------+---------------------------+
| 2     | Constant value          | TracerType.CONSTANT       |
+-------+-------------------------+---------------------------+
| 3     | Initial profile         | TracerType.INITIAL        |
+-------+-------------------------+---------------------------+
| 4     | Space-time tracer       | TracerType.SPACETIME      |
+-------+-------------------------+---------------------------+

**Simplified Configuration with Setup Types**

For users who prefer a simpler approach without dealing with individual boundary types, the enhanced tides module provides the `setup_type` parameter:

.. code-block:: python

    # Simple configuration using setup_type
    tides = SCHISMDataTidesEnhanced(
        constituents=["M2", "S2"],
        setup_type="tidal"  # Automatically sets all boundaries to tidal
    )

Available setup types include "tidal", "hybrid", "river", and "nested", each configuring the appropriate boundary types automatically.

Key Components
-------------

TidalDataset
~~~~~~~~~~~

The ``TidalDataset`` class is used to define the tidal dataset for use in SCHISM simulations:

.. code-block:: python

    from rompy.schism.tides_enhanced import TidalDataset
    
    tidal_data = TidalDataset(
        elevations="/path/to/tidal_elevations.nc",
        velocities="/path/to/tidal_velocities.nc"
    )

BoundarySetup
~~~~~~~~~~~~

The ``BoundarySetup`` class provides a flexible configuration for individual boundary segments:

.. code-block:: python

    from rompy.schism.tides_enhanced import BoundarySetup
    from rompy.schism.boundary_tides import ElevationType, VelocityType, TracerType
    
    # Example: Tidal boundary configuration
    tidal_boundary = BoundarySetup(
        elev_type=ElevationType.TIDAL,
        vel_type=VelocityType.TIDAL,
        temp_type=TracerType.NONE,
        salt_type=TracerType.NONE
    )
    
    # Example: River boundary configuration
    river_boundary = BoundarySetup(
        elev_type=ElevationType.NONE,
        vel_type=VelocityType.CONSTANT,
        const_flow=-100.0  # Negative for inflow
    )
    
    # Example: Nested model boundary with relaxation
    nested_boundary = BoundarySetup(
        elev_type=ElevationType.SPACETIME,
        vel_type=VelocityType.RELAXED,
        temp_type=TracerType.SPACETIME,
        salt_type=TracerType.SPACETIME,
        inflow_relax=0.8,
        outflow_relax=0.2
    )

SCHISMDataTidesEnhanced
~~~~~~~~~~~~~~~~~~~~~~~

The main class for enhanced tidal handling is ``SCHISMDataTidesEnhanced``, which provides comprehensive support for all boundary types:

.. code-block:: python

    from rompy.schism.tides_enhanced import SCHISMDataTidesEnhanced
    
    # Create enhanced tidal data
    tides = SCHISMDataTidesEnhanced(
        constituents=["M2", "S2", "N2", "K1", "O1"],
        tidal_database="tpxo",
        tidal_data=tidal_data,
        setup_type="tidal",  # Options: "tidal", "hybrid", "river", "nested"
        boundaries={
            0: tidal_boundary,  # First boundary segment
            1: river_boundary,  # Second boundary segment
            2: nested_boundary  # Third boundary segment
        }
    )

Factory Functions
----------------

The enhanced tides module provides several factory functions in ``tides_enhanced.py`` to simplify the creation of common boundary configurations. These functions handle the details of setting up the appropriate boundary types and parameters, allowing users to focus on their specific requirements.

Tidal-Only Configuration
~~~~~~~~~~~~~~~~~~~~~~~

Creates a configuration where all open boundaries are treated as tidal boundaries:

.. code-block:: python

    from rompy.schism.tides_enhanced import create_tidal_only_config
    
    tides = create_tidal_only_config(
        constituents=["M2", "S2", "N2", "K1", "O1"],  # Tidal constituents to use
        tidal_database="tpxo",                       # Database source
        tidal_elevations="/path/to/elevations.nc",   # Elevation data file
        tidal_velocities="/path/to/velocities.nc",   # Velocity data file
        ntip=0                                      # Nodal factor method (0=standard)
    )

Internally, this function:
1. Creates an instance of ``SCHISMDataTidesEnhanced``
2. Sets all boundaries to use ``ElevationType.TIDAL`` and ``VelocityType.TIDAL``
3. Configures the tidal database and constituents
4. Returns the configured instance ready for use

Hybrid Configuration
~~~~~~~~~~~~~~~~~~~

Creates a configuration for hybrid tidal + external data boundaries:

.. code-block:: python

    from rompy.schism.tides_enhanced import create_hybrid_config
    
    tides = create_hybrid_config(
        constituents=["M2", "S2", "N2", "K1", "O1"],
        tidal_database="tpxo",
        tidal_elevations="/path/to/elevations.nc",
        tidal_velocities="/path/to/velocities.nc"
    )

This configuration sets all boundaries to use ``ElevationType.TIDALSPACETIME`` and ``VelocityType.TIDALSPACETIME``, which combines tidal harmonics with external data.

River Configuration
~~~~~~~~~~~~~~~~~

Creates a configuration with a designated river boundary and optional tidal boundaries:

.. code-block:: python

    from rompy.schism.tides_enhanced import create_river_config
    
    tides = create_river_config(
        river_boundary_index=1,       # Index of the river boundary
        river_flow=-100.0,            # Flow rate (negative for inflow)
        other_boundaries="tidal",     # Other boundaries are tidal
        constituents=["M2", "S2"],    # Optional - only used if other_boundaries="tidal"
        tidal_elevations="/path/to/elevations.nc",  # Only if other_boundaries="tidal"
        tidal_velocities="/path/to/velocities.nc"   # Only if other_boundaries="tidal"
    )

This function:
1. Sets the specified boundary to use ``VelocityType.CONSTANT`` with the given flow rate
2. Configures other boundaries based on the ``other_boundaries`` parameter
3. Returns a fully configured ``SCHISMDataTidesEnhanced`` instance

Nested Configuration
~~~~~~~~~~~~~~~~~~

Creates a configuration for nested model boundaries with external data:

.. code-block:: python

    from rompy.schism.tides_enhanced import create_nested_config
    
    tides = create_nested_config(
        with_tides=True,              # Include tidal components
        inflow_relax=0.8,             # Relaxation parameter for inflow
        outflow_relax=0.2,            # Relaxation parameter for outflow
        constituents=["M2", "S2"],    # Only used if with_tides=True
        tidal_elevations="/path/to/elevations.nc",  # Only if with_tides=True
        tidal_velocities="/path/to/velocities.nc"   # Only if with_tides=True
    )

This function creates a nested boundary configuration that:
1. Sets boundaries to use ``ElevationType.SPACETIME`` (or ``TIDALSPACETIME`` if with_tides=True)
2. Uses ``VelocityType.RELAXED`` with the specified relaxation parameters
3. Configures tracer types to use ``TracerType.SPACETIME`` for temperature and salinity

Usage in SCHISM Configuration
----------------------------

The enhanced tides can be used in a SCHISM configuration as follows:

.. code-block:: python

    from rompy.schism.config import SCHISMConfig
    from rompy.schism.data import SCHISMData
    from rompy.schism.grid import SCHISMGrid
    from rompy.schism.tides_enhanced import create_tidal_only_config
    
    # Create grid
    grid = SCHISMGrid(hgrid="/path/to/hgrid.gr3", drag=0.0025)
    
    # Create enhanced tidal data
    tides = create_tidal_only_config(
        constituents=["M2", "S2", "N2", "K1", "O1"],
        tidal_elevations="/path/to/elevations.nc",
        tidal_velocities="/path/to/velocities.nc"
    )
    
    # Create SCHISM data with enhanced tides
    data = SCHISMData(tides=tides)
    
    # Create SCHISM configuration
    config = SCHISMConfig(grid=grid, data=data)
    
    # Generate files
    config(runtime)

Advanced Usage: Multiple Boundary Types
--------------------------------------

One of the key advantages of the enhanced tides module is the ability to configure different boundary types for different segments:

.. code-block:: python

    from rompy.schism.tides_enhanced import SCHISMDataTidesEnhanced, BoundarySetup
    from rompy.schism.boundary_tides import ElevationType, VelocityType, TracerType
    
    # Create enhanced tidal data with multiple boundary types
    tides = SCHISMDataTidesEnhanced(
        constituents=["M2", "S2", "N2"],
        tidal_database="tpxo",
        tidal_data=tidal_dataset,
        boundaries={
            0: BoundarySetup(  # Ocean boundary: tidal
                elev_type=ElevationType.TIDAL,
                vel_type=VelocityType.TIDAL
            ),
            1: BoundarySetup(  # River boundary: constant flow
                elev_type=ElevationType.NONE,
                vel_type=VelocityType.CONSTANT,
                const_flow=-100.0  # Inflow
            ),
            2: BoundarySetup(  # Nested boundary: relaxed
                elev_type=ElevationType.SPACETIME,
                vel_type=VelocityType.RELAXED,
                inflow_relax=0.8,
                outflow_relax=0.2
            )
        }
    )

Backward Compatibility and Legacy Integration
------------------------------------------

The enhanced tides module is designed to be backward compatible with existing SCHISM configurations that use the standard ``SCHISMDataTides`` class. This compatibility is implemented in several practical ways:

Legacy Flags Support
~~~~~~~~~~~~~~~~~~~~

One of the most important backward compatibility features is support for the legacy flags system. In the standard tides implementation, boundary types were configured using numeric flags:

**YAML Configuration (Legacy Format):**

.. code-block:: yaml

    tides:
      constituents:
      - M2
      - S2
      - N2
      cutoff_depth: 50.0
      flags: 
        - [5, 3, 4, 4]  # [elev_type, vel_type, temp_type, salt_type]
      tidal_data:
        data_type: tidal_dataset
        elevations: path/to/elevations.nc
        velocities: path/to/velocities.nc

**Python Code (Legacy Format):**

.. code-block:: python

    from rompy.schism.data import SCHISMDataTides
    
    # Using numeric flags
    tides = SCHISMDataTides(
        constituents=["M2", "S2"],
        flags=[[5, 3, 4, 4]]  # [elev_type, vel_type, temp_type, salt_type]
    )

The enhanced tides implementation continues to support this format for backward compatibility:

.. code-block:: python

    from rompy.schism.tides_enhanced import SCHISMDataTidesEnhanced
    
    # Using legacy flags format
    tides = SCHISMDataTidesEnhanced(
        constituents=["M2", "S2"],
        flags=[[5, 3, 4, 4]]  # Still works with the enhanced implementation
    )

Internally, the enhanced implementation converts these numeric flags to the corresponding enum types:

.. code-block:: python

    # Internal conversion (happens automatically)
    config = BoundaryConfig(
        elev_type=flags[i][0],  # 5 -> ElevationType.TIDAL
        vel_type=flags[i][1],   # 3 -> VelocityType.SPACETIDAL
        temp_type=flags[i][2],  # 4 -> TracerType.SPACETIME
        salt_type=flags[i][3]   # 4 -> TracerType.SPACETIME
    )

YAML Configuration Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here's how the configuration looks in both formats:

**Legacy Format:**

.. code-block:: yaml

    tides:
      constituents:
      - M2
      - S2
      - N2
      cutoff_depth: 50.0
      flags: 
        - [5, 3, 4, 4]
      tidal_data:
        data_type: tidal_dataset
        elevations: path/to/elevations.nc
        velocities: path/to/velocities.nc

**Enhanced Format:**

.. code-block:: yaml

    tides:
      data_type: tides_enhanced
      constituents: [M2, S2, N2]
      tidal_database: "tpxo"
      ntip: 0
      cutoff_depth: 50.0
      tidal_data:
        data_type: tidal_dataset
        elevations: path/to/elevations.nc
        velocities: path/to/velocities.nc
      boundaries:
        0:  # Applied to all open boundaries
          elev_type: 5  # TIDAL: Tidal elevation
          vel_type: 3   # SPACETIDAL: Space-time + harmonic velocity
          temp_type: 4  # SPACETIME: External data for temperature
          salt_type: 4  # SPACETIME: External data for salinity

Both configurations will work with the enhanced tides implementation, allowing for a gradual transition to the new format.

Integration with the Bctides Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The enhanced tides implementation builds upon and extends the functionality of the ``Bctides`` module, which is a direct implementation of SCHISM tidal boundary conditions based on the PyLibs approach. The relationship between these components is as follows:

1. **Bctides**: Low-level module that handles the actual generation of the bctides.in file, including tidal factor calculation, interpolation of tidal data, and file writing.

2. **TidalBoundary**: Mid-level class that uses Bctides internally but provides a more object-oriented interface with enum-based configuration.

3. **SCHISMDataTidesEnhanced**: High-level class that integrates with the overall SCHISM configuration system and provides additional convenience features.

The enhanced tides implementation uses the ``TidalBoundary`` class, which in turn uses the ``Bctides`` class internally for the actual file generation. This layered approach allows for both backward compatibility and enhanced functionality.

.. code-block:: text

    +----------------------------+     +-------------------------+     +------------------------+
    | SCHISMDataTidesEnhanced    |     | TidalBoundary           |     | Bctides               |
    | (High-level interface)     |     | (Mid-level)             |     | (Low-level)           |
    +----------------------------+     +-------------------------+     +------------------------+
    | - Validates configuration  | --> | - Manages boundary types| --> | - Calculates factors  |
    | - Converts between formats |     | - Sets up parameters    |     | - Interpolates data   |
    | - Integrates with SCHISM   |     | - Writes boundary files |     | - Writes bctides.in   |
    +----------------------------+     +-------------------------+     +------------------------+

The ``Bctides`` module is not redundant - it still handles the core functionality of generating the bctides.in file according to the SCHISM format specification. The enhanced tides implementation provides a more user-friendly interface on top of this core functionality.

Other Compatibility Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Drop-in Replacement**: The ``SCHISMDataTidesEnhanced`` class implements the same interface as ``SCHISMDataTides``, allowing it to be used as a direct replacement in existing code:

   .. code-block:: python

       # Old code with standard tides
       from rompy.schism.data import SCHISMDataTides
       tides = SCHISMDataTides(constituents=["M2", "S2"])
       
       # New code with enhanced tides
       from rompy.schism.tides_enhanced import SCHISMDataTidesEnhanced
       tides = SCHISMDataTidesEnhanced(constituents=["M2", "S2"])

2. **Automatic Type Detection**: When using the ``SCHISMData`` class, it automatically handles both standard and enhanced tides objects:

   .. code-block:: python

       from rompy.schism.data import SCHISMData
       
       # Works with both standard and enhanced tides
       data = SCHISMData(tides=tides)

3. **Simplified Setup Types**: The enhanced tides module provides a ``setup_type`` parameter that automatically configures the boundaries based on common patterns:

   .. code-block:: python

       # Equivalent to standard tides behavior
       tides = SCHISMDataTidesEnhanced(
           constituents=["M2", "S2"],
           setup_type="tidal"  # Uses tidal boundaries for all segments
       )

These compatibility features ensure that existing SCHISM workflows can be gradually migrated to use the enhanced tides functionality without breaking changes, while still allowing access to the more advanced features when needed.

Why Not Rename to Bctides?
~~~~~~~~~~~~~~~~~~~~~~~~~

While the enhanced tides implementation is closely related to the ``Bctides`` module, keeping them as separate components with different names serves several important purposes:

1. **Separation of Concerns**: The ``Bctides`` module focuses specifically on generating the bctides.in file, while the enhanced tides implementation provides a higher-level interface that integrates with the overall SCHISM configuration system.

2. **Backward Compatibility**: Maintaining separate names allows for a gradual transition from the standard tides implementation to the enhanced implementation without breaking existing code.

3. **Conceptual Clarity**: The name "SCHISMDataTidesEnhanced" clearly indicates that this is an enhanced version of the standard "SCHISMDataTides" class, making it easier for users to understand the relationship between these components.

Data Flow in the Enhanced Tides Implementation
--------------------------------------------

Understanding how data flows through the enhanced tides implementation is important for both users and developers. Here's a detailed overview of the process:

1. **Configuration Input**:
   Users provide configuration either directly to ``SCHISMDataTidesEnhanced`` or through one of the factory functions.

2. **Configuration Validation**:
   ``SCHISMDataTidesEnhanced`` validates the input configuration, checking for required data and parameter consistency.

3. **Boundary Setup Creation**:
   For each boundary segment, a ``BoundarySetup`` instance is created (either directly or converted from legacy flags).

4. **TidalBoundary Creation**:
   When needed, ``SCHISMDataTidesEnhanced.create_tidal_boundary()`` creates a ``TidalBoundary`` instance with the configured parameters.

5. **Bctides Creation**:
   ``TidalBoundary.create_bctides()`` creates a ``Bctides`` instance, which is the low-level handler for tidal boundary conditions.

6. **File Generation**:
   The ``Bctides`` instance handles the details of generating the bctides.in file, including tidal factor calculation and data interpolation.

This process can be visualized as follows:

.. code-block:: text

    User Input
        |
        v
    +----------------------------+
    | SCHISMDataTidesEnhanced    |
    | - validate_tidal_data()    |
    | - validate_setup_type()    |
    +----------------------------+
        |
        v
    +----------------------------+
    | TidalBoundary              |
    | - set_boundary_config()    |
    | - set_boundary_type()      |
    | - get_flags_list()         |
    +----------------------------+
        |
        v
    +----------------------------+
    | Bctides                    |
    | - _get_tidal_factors()     |
    | - _interpolate_tidal_data()|
    | - write_bctides()          |
    +----------------------------+
        |
        v
    bctides.in file

This hierarchical approach provides a clean separation of concerns, making the code easier to maintain and extend.

Data Input Requirements
---------------------

Each boundary type in the enhanced tides implementation has specific data input requirements. This section details what data is required for each type of boundary configuration and the ROMPy data objects that provide this data.

### Elevation Types

.. list-table::
   :header-rows: 1
   :widths: 15 10 25 25 25

   * - Type
     - Value
     - Description
     - Required Data
     - ROMPy Data Objects
   * - ``NONE``
     - 0
     - Not specified
     - None
     - None
   * - ``TIMEHIST``
     - 1
     - Time history
     - ``elev.th`` file
     - ``SCHISMDataOcean`` with ``boundary_data``
   * - ``CONSTANT``
     - 2
     - Constant elevation
     - ``const_elev`` parameter
     - ``BoundarySetup`` with ``const_elev`` set
   * - ``TIDAL``
     - 3
     - Tidal elevation
     - Tidal elevation file
     - ``TidalDataset`` with ``elevations`` path
   * - ``SPACETIME``
     - 4
     - Space and time-varying
     - ``elev2D.th.nc`` file
     - ``SCHISMDataOcean`` with ``boundary_data``
   * - ``TIDALSPACETIME``
     - 5
     - Tidal + external data
     - Both tidal and external files
     - ``TidalDataset`` + ``SCHISMDataOcean``

Velocity Types
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 10 25 25 25

   * - Type
     - Value
     - Description
     - Required Data
     - ROMPy Data Objects
   * - ``NONE``
     - 0
     - Not specified
     - None
     - None
   * - ``TIMESERIES``
     - 1
     - Time history
     - ``flux.th`` file
     - ``SCHISMDataOcean`` with ``boundary_data``
   * - ``CONSTANT``
     - 2
     - Constant discharge
     - ``const_flow`` parameter
     - ``BoundarySetup`` with ``const_flow`` set
   * - ``TIDAL``
     - 3
     - Tidal velocity
     - Tidal velocity file
     - ``TidalDataset`` with ``velocities`` path
   * - ``SPACETIME``
     - 4
     - Space and time-varying
     - ``uv3D.th.nc`` file
     - ``SCHISMDataOcean`` with ``boundary_data``
   * - ``TIDALSPACETIME``
     - 5
     - Tidal + external data
     - Both tidal and external files
     - ``TidalDataset`` + ``SCHISMDataOcean``
   * - ``FLATHER``
     - -1
     - Flather radiation
     - ``mean_elev`` and ``mean_flow``
     - ``BoundarySetup`` with these parameters
   * - ``RELAXED``
     - -4
     - 3D with relaxation
     - Relaxation parameters
     - ``BoundarySetup`` with relaxation parameters

Tracer Types (Temperature and Salinity)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 10 25 25 25

   * - Type
     - Value
     - Description
     - Required Data
     - ROMPy Data Objects
   * - ``NONE``
     - 0
     - Not specified
     - None
     - None
   * - ``TIMESERIES``
     - 1
     - Time history
     - ``temp.th`` or ``salt.th``
     - ``SCHISMDataOcean`` with ``boundary_data``
   * - ``CONSTANT``
     - 2
     - Constant value
     - ``const_temp`` or ``const_salt``
     - ``BoundarySetup`` with these parameters
   * - ``INITIAL``
     - 3
     - Initial profile
     - Initial conditions
     - ``SCHISMDataHotstart``
   * - ``SPACETIME``
     - 4
     - 3D input
     - ``temp3D.th.nc`` or ``salt3D.th.nc``
     - ``SCHISMDataOcean`` with ``boundary_data``

Common Configurations and Their Requirements
------------------------------------------

YAML Configuration Examples
------------------------

The enhanced tides implementation supports various configuration formats in YAML files:

Pure Tidal Configuration
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    tides:
      data_type: tides_enhanced
      constituents: [M2, S2, K1, O1]
      tidal_database: "tpxo"
      setup_type: "tidal"
      tidal_data:
        data_type: tidal_dataset
        elevations: path/to/elevations.nc  # Required
        velocities: path/to/velocities.nc  # Required

**Required Data:**

* Tidal elevation file (``elevations``)
* Tidal velocity file (``velocities``)
* List of constituents

Hybrid Boundary (``setup_type="hybrid``")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    tides:
      data_type: tides_enhanced
      constituents: [M2, S2, K1, O1]
      tidal_database: "tpxo"
      setup_type: "hybrid"
      tidal_data:
        data_type: tidal_dataset
        elevations: path/to/elevations.nc  # Required
        velocities: path/to/velocities.nc  # Required

**Required Data:**

* Tidal elevation file (``elevations``)
* Tidal velocity file (``velocities``)
* List of constituents
* External data files in the run directory:

  * ``elev2D.th.nc`` for elevation
  * ``uv3D.th.nc`` for velocity

River Boundary (``setup_type="river``")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    tides:
      data_type: tides_enhanced
      setup_type: "river"
      boundaries:
        0:  # Only applied to boundary 0
          const_flow: -100.0  # Negative for inflow, required

**Required Data:**

* Constant flow value (``const_flow``)

Nested Boundary (``setup_type="nested``")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    tides:
      data_type: tides_enhanced
      setup_type: "nested"
      boundaries:
        0:  # Applied to all open boundaries
          inflow_relax: 0.8  # Required
          outflow_relax: 0.8  # Required

**Required Data:**

* Relaxation factors (``inflow_relax`` and ``outflow_relax``)
* External data files in the run directory:

  * ``elev2D.th.nc`` for elevation
  * ``uv3D.th.nc`` for velocity
  * ``temp3D.th.nc`` for temperature (if used)
  * ``salt3D.th.nc`` for salinity (if used)

Mixed Boundary Types
~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    tides:
      data_type: tides_enhanced
      constituents: [M2, S2, K1, O1]
      tidal_database: "tpxo"
      tidal_data:
        data_type: tidal_dataset
        elevations: path/to/elevations.nc
        velocities: path/to/velocities.nc
      boundaries:
        0:  # Ocean boundary
          elev_type: 3  # TIDAL
          vel_type: 3   # TIDAL
        1:  # River boundary
          elev_type: 0  # NONE
          vel_type: 2   # CONSTANT
          const_flow: -100.0
        2:  # Nested boundary
          elev_type: 4  # SPACETIME
          vel_type: -4  # RELAXED
          inflow_relax: 0.8
          outflow_relax: 0.2


Conclusion
---------

The enhanced tides implementation provides a more flexible and intuitive way to configure tidal boundary conditions in SCHISM. By replacing the numeric flags with enum types, it improves code readability, maintainability, and type safety. The implementation is backward compatible with existing SCHISM configurations, making it easy to transition from the standard tides implementation to the enhanced one while taking advantage of the more comprehensive support for all boundary types.
