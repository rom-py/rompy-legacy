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
]

import logging
from typing import List, Literal, Optional, Union

from rompy.core.data import DataBlob
from rompy.schism.boundary_tides import ElevationType, TracerType, VelocityType
from rompy.schism.data import (
    SCHISMDataBoundary,
    BoundarySetupWithSource,
    SCHISMDataBoundaryConditions,
)
from rompy.schism.tides_enhanced import TidalDataset

logger = logging.getLogger(__name__)


# Factory functions for common configurations


def create_tidal_only_boundary_config(
    constituents: List[str] = ["M2", "S2", "N2", "K1", "O1"],
    tidal_database: str = "tpxo",
    tidal_elevations: Optional[str] = None,
    tidal_velocities: Optional[str] = None,
    ntip: int = 0,
) -> SCHISMDataBoundaryConditions:
    """
    Create a configuration where all open boundaries are treated as tidal boundaries.

    Parameters
    ----------
    constituents : List[str]
        Tidal constituents to include
    tidal_database : str
        Tidal database to use
    tidal_elevations : str, optional
        Path to tidal elevation data
    tidal_velocities : str, optional
        Path to tidal velocity data
    ntip : int
        Nodal factor method (0=standard)

    Returns
    -------
    SCHISMDataBoundaryConditions
        Configured boundary conditions
    """
    # Create tidal dataset if both paths are provided
    tidal_data = None
    if tidal_elevations and tidal_velocities:
        tidal_data = TidalDataset(
            elevations=tidal_elevations, velocities=tidal_velocities
        )

    # Create the config with tidal setup
    config = SCHISMDataBoundaryConditions(
        constituents=constituents,
        tidal_database=tidal_database,
        tidal_data=tidal_data,
        ntip=ntip,
        setup_type="tidal",
        boundaries={},
        hotstart_config=None,
    )

    return config


def create_hybrid_boundary_config(
    constituents: List[str] = ["M2", "S2", "N2", "K1", "O1"],
    tidal_database: str = "tpxo",
    tidal_elevations: Optional[str] = None,
    tidal_velocities: Optional[str] = None,
    elev_source: Optional[Union[DataBlob, SCHISMDataBoundary]] = None,
    vel_source: Optional[Union[DataBlob, SCHISMDataBoundary]] = None,
    temp_source: Optional[Union[DataBlob, SCHISMDataBoundary]] = None,
    salt_source: Optional[Union[DataBlob, SCHISMDataBoundary]] = None,
) -> SCHISMDataBoundaryConditions:
    """
    Create a configuration for hybrid tidal + external data boundaries.

    Parameters
    ----------
    constituents : List[str]
        Tidal constituents to include
    tidal_database : str
        Tidal database to use
    tidal_elevations : str, optional
        Path to tidal elevation data
    tidal_velocities : str, optional
        Path to tidal velocity data
    elev_source : Union[DataBlob, SCHISMDataBoundary], optional
        Data source for elevation
    vel_source : Union[DataBlob, SCHISMDataBoundary], optional
        Data source for velocity
    temp_source : Union[DataBlob, SCHISMDataBoundary], optional
        Data source for temperature
    salt_source : Union[DataBlob, SCHISMDataBoundary], optional
        Data source for salinity

    Returns
    -------
    SCHISMDataBoundaryConditions
        Configured boundary conditions
    """
    # Create tidal dataset if both paths are provided
    tidal_data = None
    if tidal_elevations and tidal_velocities:
        tidal_data = TidalDataset(
            elevations=tidal_elevations, velocities=tidal_velocities
        )

    # Create the config with hybrid setup
    config = SCHISMDataBoundaryConditions(
        constituents=constituents,
        tidal_database=tidal_database,
        tidal_data=tidal_data,
        setup_type="hybrid",
        boundaries={
            0: BoundarySetupWithSource(
                elev_type=ElevationType.TIDALSPACETIME,
                vel_type=VelocityType.TIDALSPACETIME,
                temp_type=TracerType.SPACETIME if temp_source else TracerType.NONE,
                salt_type=TracerType.SPACETIME if salt_source else TracerType.NONE,
                elev_source=elev_source,
                vel_source=vel_source,
                temp_source=temp_source,
                salt_source=salt_source,
            )
        },
        hotstart_config=None,
    )

    return config


def create_river_boundary_config(
    river_boundary_index: int = 0,
    river_flow: float = -100.0,  # Negative for inflow
    other_boundaries: Literal["tidal", "hybrid", "none"] = "tidal",
    constituents: List[str] = ["M2", "S2"],
    tidal_elevations: Optional[str] = None,
    tidal_velocities: Optional[str] = None,
) -> SCHISMDataBoundaryConditions:
    """
    Create a configuration with a designated river boundary and optional tidal boundaries.

    Parameters
    ----------
    river_boundary_index : int
        Index of the river boundary
    river_flow : float
        Flow rate (negative for inflow)
    other_boundaries : str
        How to treat other boundaries ("tidal", "hybrid", or "none")
    constituents : List[str]
        Tidal constituents to include (only used if other_boundaries="tidal" or "hybrid")
    tidal_elevations : str, optional
        Path to tidal elevation data (only if other_boundaries="tidal" or "hybrid")
    tidal_velocities : str, optional
        Path to tidal velocity data (only if other_boundaries="tidal" or "hybrid")

    Returns
    -------
    SCHISMDataBoundaryConditions
        Configured boundary conditions
    """
    # Create tidal dataset if both paths are provided and needed
    tidal_data = None
    if (
        other_boundaries in ["tidal", "hybrid"]
        and tidal_elevations
        and tidal_velocities
    ):
        tidal_data = TidalDataset(
            elevations=tidal_elevations, velocities=tidal_velocities
        )

    # Create the basic config
    config = SCHISMDataBoundaryConditions(
        constituents=constituents if other_boundaries in ["tidal", "hybrid"] else [],
        tidal_database="tpxo" if other_boundaries in ["tidal", "hybrid"] else "",
        tidal_data=tidal_data,
        setup_type="river",
        hotstart_config=None,
    )

    # Add the river boundary
    config.boundaries[river_boundary_index] = BoundarySetupWithSource(
        elev_type=ElevationType.NONE,
        vel_type=VelocityType.CONSTANT,
        temp_type=TracerType.NONE,
        salt_type=TracerType.NONE,
        const_flow=river_flow,
    )

    return config


def create_nested_boundary_config(
    with_tides: bool = True,
    inflow_relax: float = 0.8,
    outflow_relax: float = 0.2,
    constituents: List[str] = ["M2", "S2"],
    tidal_elevations: Optional[str] = None,
    tidal_velocities: Optional[str] = None,
    elev_source: Optional[Union[DataBlob, SCHISMDataBoundary]] = None,
    vel_source: Optional[Union[DataBlob, SCHISMDataBoundary]] = None,
    temp_source: Optional[Union[DataBlob, SCHISMDataBoundary]] = None,
    salt_source: Optional[Union[DataBlob, SCHISMDataBoundary]] = None,
) -> SCHISMDataBoundaryConditions:
    """
    Create a configuration for nested model boundaries with external data.

    Parameters
    ----------
    with_tides : bool
        Include tidal components
    inflow_relax : float
        Relaxation parameter for inflow (0-1)
    outflow_relax : float
        Relaxation parameter for outflow (0-1)
    constituents : List[str]
        Tidal constituents to include (only used if with_tides=True)
    tidal_elevations : str, optional
        Path to tidal elevation data (only if with_tides=True)
    tidal_velocities : str, optional
        Path to tidal velocity data (only if with_tides=True)
    elev_source : Union[DataBlob, SCHISMDataBoundary], optional
        Data source for elevation
    vel_source : Union[DataBlob, SCHISMDataBoundary], optional
        Data source for velocity
    temp_source : Union[DataBlob, SCHISMDataBoundary], optional
        Data source for temperature
    salt_source : Union[DataBlob, SCHISMDataBoundary], optional
        Data source for salinity

    Returns
    -------
    SCHISMDataBoundaryConditions
        Configured boundary conditions
    """
    # Create tidal dataset if both paths are provided and needed
    tidal_data = None
    if with_tides and tidal_elevations and tidal_velocities:
        tidal_data = TidalDataset(
            elevations=tidal_elevations, velocities=tidal_velocities
        )

    # Create the basic config
    config = SCHISMDataBoundaryConditions(
        constituents=constituents if with_tides else [],
        tidal_database="tpxo" if with_tides else "",
        tidal_data=tidal_data,
        setup_type="nested",
        hotstart_config=None,
    )

    # Determine elevation type based on tides setting
    elev_type = ElevationType.TIDALSPACETIME if with_tides else ElevationType.SPACETIME

    # Add the nested boundary configuration
    config.boundaries[0] = BoundarySetupWithSource(
        elev_type=elev_type,
        vel_type=VelocityType.RELAXED,
        temp_type=TracerType.SPACETIME if temp_source else TracerType.NONE,
        salt_type=TracerType.SPACETIME if salt_source else TracerType.NONE,
        inflow_relax=inflow_relax,
        outflow_relax=outflow_relax,
        elev_source=elev_source,
        vel_source=vel_source,
        temp_source=temp_source,
        salt_source=salt_source,
    )

    return config
