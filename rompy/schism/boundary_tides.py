import logging
import os
import sys
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union, Any

import numpy as np
from pydantic import ConfigDict, Field, BaseModel

# Ensure path to pylibs is available if needed
if "/home/tdurrant/source/pylibs" not in sys.path:
    sys.path.append("/home/tdurrant/source/pylibs")

# Import PyLibs functions if available
try:
    from pylib import *
    from src.schism_file import read_schism_hgrid, loadz
except ImportError:
    logging.warning("PyLibs not found, some functionality may be limited")

# Import from local modules
from .boundary import BoundaryData
from .bctides import Bctides

logger = logging.getLogger(__name__)


class ElevationType(IntEnum):
    """Elevation boundary condition types."""
    NONE = 0  # Not specified
    TIMESERIES = 1  # Time history from elev.th
    CONSTANT = 2  # Constant elevation
    TIDAL = 3  # Tidal elevation
    SPACETIME = 4  # Space and time-varying from elev2D.th.nc
    TIDALSPACETIME = 5  # Combination of tide and external file


class VelocityType(IntEnum):
    """Velocity boundary condition types."""
    NONE = 0  # Not specified
    TIMESERIES = 1  # Time history from flux.th
    CONSTANT = 2  # Constant discharge
    TIDAL = 3  # Tidal velocity
    SPACETIME = 4  # Space and time-varying from uv3D.th.nc
    TIDALSPACETIME = 5  # Combination of tide and external file
    FLATHER = -1  # Flather type radiation boundary
    RELAXED = -4  # 3D input with relaxation


class TracerType(IntEnum):
    """Temperature/salinity boundary condition types."""
    NONE = 0  # Not specified
    TIMESERIES = 1  # Time history from temp/salt.th
    CONSTANT = 2  # Constant temperature/salinity
    INITIAL = 3  # Initial profile for inflow
    SPACETIME = 4  # 3D input


class TidalSpecies(IntEnum):
    """Tidal species types."""
    LONG_PERIOD = 0  # Long period (declinational)
    DIURNAL = 1  # Diurnal
    SEMI_DIURNAL = 2  # Semi-diurnal


class BoundaryConfig(BaseModel):
    """Configuration for a single boundary."""

    elev_type: ElevationType = Field(
        ElevationType.NONE, description="Elevation boundary condition type"
    )
    vel_type: VelocityType = Field(
        VelocityType.NONE, description="Velocity boundary condition type"
    )
    temp_type: TracerType = Field(
        TracerType.NONE, description="Temperature boundary condition type"
    )
    salt_type: TracerType = Field(
        TracerType.NONE, description="Salinity boundary condition type"
    )

    # Parameters for constant values (types 2)
    ethconst: Optional[float] = Field(None, description="Constant elevation value")
    vthconst: Optional[float] = Field(None, description="Constant velocity value")
    tthconst: Optional[float] = Field(None, description="Constant temperature value")
    sthconst: Optional[float] = Field(None, description="Constant salinity value")

    # Parameters for relaxation (type -4)
    inflow_relax: float = Field(
        0.5, description="Relaxation factor for inflow (0-1, 1 is strongest nudging)"
    )
    outflow_relax: float = Field(
        0.1, description="Relaxation factor for outflow (0-1, 1 is strongest nudging)"
    )

    # Parameters for Flather (type -1)
    eta_mean: Optional[List[float]] = Field(
        None, description="Mean elevation values for each node"
    )
    vn_mean: Optional[List[List[float]]] = Field(
        None, description="Mean normal velocity values for each node at each level"
    )

    # Parameters for temperature/salinity nudging
    tobc: float = Field(
        1.0, description="Temperature nudging factor (0-1, 1 is strongest nudging)"
    )
    sobc: float = Field(
        1.0, description="Salinity nudging factor (0-1, 1 is strongest nudging)"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class TidalBoundary(BoundaryData):
    """Handler for SCHISM tidal boundary conditions.
    
    This class extends BoundaryData to specifically handle tidal boundaries
    with full support for all SCHISM boundary condition types.
    """

    def __init__(
        self,
        grid_path: Union[str, Path],
        constituents: Union[str, List[str]] = "major",
        tidal_database: str = "tpxo",
        boundary_configs: Optional[Dict[int, BoundaryConfig]] = None,
        tidal_elevations: Optional[str] = None,
        tidal_velocities: Optional[str] = None,
        ntip: int = 0,
        tip_dp: float = 1.0,
        cutoff_depth: float = 50.0,
        *args, **kwargs
    ):
        """Initialize the tidal boundary handler.

        Parameters
        ----------
        grid_path : str or Path
            Path to the SCHISM grid file
        constituents : str or list of str, optional
            Tidal constituents to use, by default "major"
        tidal_database : str, optional
            Tidal database to use, by default "tpxo"
        boundary_configs : dict, optional
            Configuration for each boundary, keyed by boundary index
        tidal_elevations : str, optional
            Path to tidal elevations file
        tidal_velocities : str, optional
            Path to tidal velocities file
        ntip : int, optional
            Number of earth tidal potential regions, by default 0
        tip_dp : float, optional
            Depth threshold for tidal potential, by default 1.0
        cutoff_depth : float, optional
            Cutoff depth for tides, by default 50.0
        """
        super().__init__(grid_path, *args, **kwargs)
        
        self.constituents = constituents
        self.tidal_database = tidal_database if tidal_database is not None else "none"
        self.boundary_configs = boundary_configs or {}
        self.tidal_elevations = tidal_elevations
        self.tidal_velocities = tidal_velocities
        self.ntip = ntip
        self.tip_dp = tip_dp
        self.cutoff_depth = cutoff_depth
        
        # For storing start time and run duration
        self._start_time = None
        self._rnday = None
        
    def set_boundary_config(self, boundary_index: int, config: BoundaryConfig):
        """Set configuration for a specific boundary.

        Parameters
        ----------
        boundary_index : int
            Index of the boundary
        config : BoundaryConfig
            Configuration for the boundary
        """
        self.boundary_configs[boundary_index] = config
        
    def set_boundary_type(
        self, 
        boundary_index: int,
        elev_type: ElevationType,
        vel_type: VelocityType,
        temp_type: TracerType = TracerType.NONE,
        salt_type: TracerType = TracerType.NONE,
        **kwargs
    ):
        """Set boundary types for a specific boundary.

        Parameters
        ----------
        boundary_index : int
            Index of the boundary
        elev_type : ElevationType
            Elevation boundary condition type
        vel_type : VelocityType
            Velocity boundary condition type
        temp_type : TracerType, optional
            Temperature boundary condition type
        salt_type : TracerType, optional
            Salinity boundary condition type
        **kwargs
            Additional parameters for the boundary configuration
        """
        config = BoundaryConfig(
            elev_type=elev_type,
            vel_type=vel_type,
            temp_type=temp_type,
            salt_type=salt_type,
            **kwargs
        )
        self.set_boundary_config(boundary_index, config)
        
    def set_run_parameters(self, start_time, run_days):
        """Set start time and run duration.

        Parameters
        ----------
        start_time : datetime or list
            Start time for the simulation
        run_days : float
            Duration of the simulation in days
        """
        self._start_time = start_time
        self._rnday = run_days
        
    def get_flags_list(self) -> List[List[int]]:
        """Get list of boundary flags for Bctides.

        Returns
        -------
        list of list of int
            Boundary flags for each boundary
        """
        if not self.boundary_configs:
            return [[5, 5, 0, 0]]  # Default to tidal
            
        # Find max boundary without using default parameter
        if self.boundary_configs:
            # Convert keys to list and find max
            boundary_keys = list(self.boundary_configs.keys())
            max_boundary = max(boundary_keys) if boundary_keys else -1
        else:
            max_boundary = -1
            
        flags = []
        
        for i in range(int(max_boundary) + 1):
            if i in self.boundary_configs:
                config = self.boundary_configs[i]
                flags.append([
                    int(config.elev_type),
                    int(config.vel_type),
                    int(config.temp_type),
                    int(config.salt_type)
                ])
            else:
                flags.append([0, 0, 0, 0])
                
        return flags
    
    def get_constant_values(self) -> Dict[str, List[float]]:
        """Get constant values for boundaries.

        Returns
        -------
        dict
            Dictionary of constant values for each boundary type
        """
        result = {
            "ethconst": [],
            "vthconst": [],
            "tthconst": [],
            "sthconst": [],
            "tobc": [],
            "sobc": [],
            "inflow_relax": [],
            "outflow_relax": [],
        }
        
        if not self.boundary_configs:
            return result
            
        # Find max boundary without using default parameter
        if self.boundary_configs:
            # Convert keys to list and find max
            boundary_keys = list(self.boundary_configs.keys())
            max_boundary = max(boundary_keys) if boundary_keys else -1
        else:
            max_boundary = -1
        
        for i in range(int(max_boundary) + 1):
            if i in self.boundary_configs:
                config = self.boundary_configs[i]
                
                # Handle type 2 (constant) boundaries
                if config.elev_type == ElevationType.CONSTANT:
                    result["ethconst"].append(config.ethconst if config.ethconst is not None else 0.0)
                else:
                    result["ethconst"].append(0.0)
                    
                if config.vel_type == VelocityType.CONSTANT:
                    result["vthconst"].append(config.vthconst if config.vthconst is not None else 0.0)
                else:
                    result["vthconst"].append(0.0)
                    
                if config.temp_type == TracerType.CONSTANT:
                    result["tthconst"].append(config.tthconst if config.tthconst is not None else 0.0)
                else:
                    result["tthconst"].append(0.0)
                    
                if config.salt_type == TracerType.CONSTANT:
                    result["sthconst"].append(config.sthconst if config.sthconst is not None else 0.0)
                else:
                    result["sthconst"].append(0.0)
                
                # Nudging factors for temperature and salinity
                result["tobc"].append(config.tobc)
                result["sobc"].append(config.sobc)
                
                # Relaxation factors for velocity
                if config.vel_type == VelocityType.RELAXED:
                    result["inflow_relax"].append(config.inflow_relax)
                    result["outflow_relax"].append(config.outflow_relax)
                else:
                    result["inflow_relax"].append(0.5)  # Default values
                    result["outflow_relax"].append(0.1)
            else:
                # Default values for missing boundaries
                result["ethconst"].append(0.0)
                result["vthconst"].append(0.0)
                result["tthconst"].append(0.0)
                result["sthconst"].append(0.0)
                result["tobc"].append(1.0)
                result["sobc"].append(1.0)
                result["inflow_relax"].append(0.5)
                result["outflow_relax"].append(0.1)
                
        return result
    
    def create_bctides(self) -> Bctides:
        """Create a Bctides instance from this configuration.

        Returns
        -------
        Bctides
            Configured Bctides instance
        """
        flags = self.get_flags_list()
        constants = self.get_constant_values()
        
        # Clean up lists to avoid None values
        ethconst = constants["ethconst"] if constants["ethconst"] else None
        vthconst = constants["vthconst"] if constants["vthconst"] else None
        tthconst = constants["tthconst"] if constants["tthconst"] else None
        sthconst = constants["sthconst"] if constants["sthconst"] else None
        tobc = constants["tobc"] if constants["tobc"] else None
        sobc = constants["sobc"] if constants["sobc"] else None
        
        # Create Bctides object
        bctides = Bctides(
            hgrid=self.grid,
            flags=flags,
            constituents=self.constituents,
            tidal_database=self.tidal_database,
            ntip=self.ntip,
            tip_dp=self.tip_dp,
            cutoff_depth=self.cutoff_depth,
            ethconst=ethconst,
            vthconst=vthconst,
            tthconst=tthconst,
            sthconst=sthconst,
            tobc=tobc,
            sobc=sobc,
            tidal_elevations=self.tidal_elevations,
            tidal_velocities=self.tidal_velocities
        )
        
        # Set start time and run duration
        if self._start_time and self._rnday is not None:
            bctides._start_time = self._start_time
            bctides._rnday = self._rnday
            
        return bctides
    
    def write_boundary_file(self, output_path: Union[str, Path]) -> Path:
        """Write the bctides.in file.

        Parameters
        ----------
        output_path : str or Path
            Path to write the file

        Returns
        -------
        Path
            Path to the written file

        Raises
        ------
        ValueError
            If start_time and rnday are not set
        """
        if not self._start_time or self._rnday is None:
            raise ValueError("start_time and rnday must be set before writing boundary file")
            
        # Create Bctides object
        bctides = self.create_bctides()
        
        # Write file
        output_path = Path(output_path)
        bctides.write_bctides(output_path)
        
        return output_path


# Factory functions for common configurations

def create_tidal_boundary(
    grid_path: Union[str, Path],
    constituents: Union[str, List[str]] = "major",
    tidal_database: str = "tpxo",
    tidal_elevations: Optional[str] = None,
    tidal_velocities: Optional[str] = None
) -> TidalBoundary:
    """Create a tidal-only boundary.

    Parameters
    ----------
    grid_path : str or Path
        Path to SCHISM grid
    constituents : str or list, optional
        Tidal constituents, by default "major"
    tidal_database : str, optional
        Tidal database, by default "tpxo"
    tidal_elevations : str, optional
        Path to tidal elevation file
    tidal_velocities : str, optional
        Path to tidal velocity file

    Returns
    -------
    TidalBoundary
        Configured tidal boundary
    """
    boundary = TidalBoundary(
        grid_path=grid_path,
        constituents=constituents,
        tidal_database=tidal_database,
        tidal_elevations=tidal_elevations,
        tidal_velocities=tidal_velocities
    )
    
    # Set default configuration for all boundaries: pure tidal
    boundary.set_boundary_type(
        0,  # Will be applied to all boundaries
        elev_type=ElevationType.TIDAL,
        vel_type=VelocityType.TIDAL
    )
    
    return boundary


def create_hybrid_boundary(
    grid_path: Union[str, Path],
    constituents: Union[str, List[str]] = "major",
    tidal_database: str = "tpxo",
    tidal_elevations: Optional[str] = None,
    tidal_velocities: Optional[str] = None
) -> TidalBoundary:
    """Create a hybrid boundary with tides + external data.

    Parameters
    ----------
    grid_path : str or Path
        Path to SCHISM grid
    constituents : str or list, optional
        Tidal constituents, by default "major"
    tidal_database : str, optional
        Tidal database, by default "tpxo"
    tidal_elevations : str, optional
        Path to tidal elevation file
    tidal_velocities : str, optional
        Path to tidal velocity file

    Returns
    -------
    TidalBoundary
        Configured hybrid boundary
    """
    boundary = TidalBoundary(
        grid_path=grid_path,
        constituents=constituents,
        tidal_database=tidal_database,
        tidal_elevations=tidal_elevations,
        tidal_velocities=tidal_velocities
    )
    
    # Set default configuration for all boundaries: tidal + spacetime
    boundary.set_boundary_type(
        0,  # Will be applied to all boundaries
        elev_type=ElevationType.TIDALSPACETIME,
        vel_type=VelocityType.TIDALSPACETIME
    )
    
    return boundary


def create_river_boundary(
    grid_path: Union[str, Path],
    river_flow: float = -100.0,  # Negative for inflow
    river_boundary_index: int = 0
) -> TidalBoundary:
    """Create a river boundary with constant flow.

    Parameters
    ----------
    grid_path : str or Path
        Path to SCHISM grid
    river_flow : float, optional
        River flow value (negative for inflow), by default -100.0
    river_boundary_index : int, optional
        Index of the river boundary, by default 0

    Returns
    -------
    TidalBoundary
        Configured river boundary
    """
    boundary = TidalBoundary(grid_path=grid_path)
    
    # Set river boundary
    boundary.set_boundary_type(
        river_boundary_index,
        elev_type=ElevationType.NONE,  # No elevation specified
        vel_type=VelocityType.CONSTANT,  # Constant flow
        vthconst=river_flow  # Flow value
    )
    
    return boundary


def create_nested_boundary(
    grid_path: Union[str, Path],
    with_tides: bool = False,
    inflow_relax: float = 0.8,
    outflow_relax: float = 0.8,
    constituents: Union[str, List[str]] = "major",
    tidal_database: str = "tpxo",
    tidal_elevations: Optional[str] = None,
    tidal_velocities: Optional[str] = None
) -> TidalBoundary:
    """Create a nested boundary with optional tides.

    Parameters
    ----------
    grid_path : str or Path
        Path to SCHISM grid
    with_tides : bool, optional
        Whether to include tides, by default False
    inflow_relax : float, optional
        Relaxation factor for inflow, by default 0.8
    outflow_relax : float, optional
        Relaxation factor for outflow, by default 0.8
    constituents : str or list, optional
        Tidal constituents if with_tides=True, by default "major"
    tidal_database : str, optional
        Tidal database if with_tides=True, by default "tpxo"
    tidal_elevations : str, optional
        Path to tidal elevation file if with_tides=True
    tidal_velocities : str, optional
        Path to tidal velocity file if with_tides=True

    Returns
    -------
    TidalBoundary
        Configured nested boundary
    """
    boundary = TidalBoundary(
        grid_path=grid_path,
        constituents=constituents if with_tides else None,
        tidal_database=tidal_database if with_tides else None,
        tidal_elevations=tidal_elevations if with_tides else None,
        tidal_velocities=tidal_velocities if with_tides else None
    )
    
    if with_tides:
        # Tides + external data with relaxation
        boundary.set_boundary_type(
            0,  # Will be applied to all boundaries
            elev_type=ElevationType.TIDALSPACETIME,
            vel_type=VelocityType.RELAXED,
            temp_type=TracerType.SPACETIME,
            salt_type=TracerType.SPACETIME,
            inflow_relax=inflow_relax,
            outflow_relax=outflow_relax
        )
    else:
        # Just external data with relaxation
        boundary.set_boundary_type(
            0,  # Will be applied to all boundaries
            elev_type=ElevationType.SPACETIME,
            vel_type=VelocityType.RELAXED,
            temp_type=TracerType.SPACETIME,
            salt_type=TracerType.SPACETIME,
            inflow_relax=inflow_relax,
            outflow_relax=outflow_relax
        )
    
    return boundary