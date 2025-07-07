"""
Boundary Conditions Factory Functions for SCHISM

This module provides factory functions for creating boundary condition configurations
for SCHISM simulations. The main classes (BoundarySetupWithSource and 
SCHISMDataBoundaryConditions) are defined in rompy.schism.data.

Key Features:
- Factory functions for creating common boundary condition setups
- Support for all SCHISM boundary types: tidal, river, nested, and hybrid configurations
- Simplified configuration creation with sensible defaults

Factory Functions:
- create_tidal_only_boundary_config: Creates a configuration with tidal boundaries
- create_hybrid_boundary_config: Creates a configuration with hybrid tidal + external data
- create_river_boundary_config: Creates a configuration with river boundaries
- create_nested_boundary_config: Creates a configuration for nested model boundaries

Example Usage:
    ```python
    from rompy.schism.boundary_conditions import create_tidal_only_boundary_config
    from rompy.schism.data import SCHISMData
    from rompy.core.data import DataBlob
    
    # Simple tidal configuration
    bc = create_tidal_only_boundary_config(
        constituents=["M2", "S2", "N2", "K1", "O1"],
        tidal_elevations="/path/to/elevations.nc",
        tidal_velocities="/path/to/velocities.nc"
    )

    # Hybrid configuration with data sources
    bc = create_hybrid_boundary_config(
        constituents=["M2", "S2"],
        tidal_elevations="/path/to/elevations.nc",
        tidal_velocities="/path/to/velocities.nc",
        elev_source=DataBlob(path="/path/to/elev2D.th.nc")
    )

    # Using in SCHISMData
    schism_data = SCHISMData(
        boundary_conditions=bc,
        atmos=atmos_data
    )
    ```
"""

__all__ = [
    "create_tidal_only_boundary_config",
    "create_hybrid_boundary_config",
    "create_river_boundary_config",
    "create_nested_boundary_config",
    # Re-export core components
    "ElevationType",
    "VelocityType",
    "TracerType",
    "BoundaryConfig",
    "BoundaryHandler",
    "create_tidal_boundary",
    "create_hybrid_boundary",
    "create_river_boundary",
    "create_nested_boundary",
]

import logging

# Import factory functions and core components from boundary_core
from rompy.schism.boundary_core import (
    ElevationType,
    VelocityType,
    TracerType,
    BoundaryConfig,
    BoundaryHandler,
    create_tidal_boundary,
    create_hybrid_boundary,
    create_river_boundary,
    create_nested_boundary,
    create_tidal_only_boundary_config,
    create_hybrid_boundary_config,
    create_river_boundary_config,
    create_nested_boundary_config,
)

logger = logging.getLogger(__name__)

# All factory functions are now imported from boundary_core
# This module serves as a high-level interface with documentation and re-exports
