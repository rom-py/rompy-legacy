"""
Core SCHISM Boundary Conditions Module

This module provides the core infrastructure for handling all types of SCHISM boundary 
conditions including tidal, river, nested, and hybrid configurations. It serves as the 
foundation for the SCHISM boundary conditions system.

Key Components:
- Boundary condition type enums (ElevationType, VelocityType, TracerType)
- BoundaryConfig class for individual boundary configuration
- BoundaryHandler class for comprehensive boundary management
- Factory functions for creating common boundary setups
- High-level configuration functions for complex scenarios

This module was formerly named boundary_tides.py but was renamed to better reflect
its role as the core boundary handling module for all boundary types, not just tidal.

Example Usage:
    ```python
    from rompy.schism.boundary_core import (
        BoundaryHandler, 
        ElevationType, 
        VelocityType,
        create_tidal_boundary
    )
    
    # Create a boundary handler
    boundary = BoundaryHandler(grid_path="hgrid.gr3")
    
    # Configure tidal boundary
    boundary.set_boundary_type(
        0, 
        elev_type=ElevationType.HARMONIC,
        vel_type=VelocityType.HARMONIC
    )
    
    # Or use factory function
    tidal_boundary = create_tidal_boundary(
        grid_path="hgrid.gr3",
        constituents=["M2", "S2", "K1", "O1"]
    )
    ```
"""

import logging
import os
import sys
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union, Any

import numpy as np
from pydantic import ConfigDict, Field, BaseModel, field_validator


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
    TIMEHIST = 1  # Time history from elev.th
    CONSTANT = 2  # Constant elevation
    HARMONIC = 3  # Harmonic tidal constituents
    EXTERNAL = 4  # External model data from elev2D.th.nc
    HARMONICEXTERNAL = 5  # Combination of harmonic and external data


class VelocityType(IntEnum):
    """Velocity boundary condition types."""

    NONE = 0  # Not specified
    TIMEHIST = 1  # Time history from flux.th
    CONSTANT = 2  # Constant discharge
    HARMONIC = 3  # Harmonic tidal constituents
    EXTERNAL = 4  # External model data from uv3D.th.nc
    HARMONICEXTERNAL = 5  # Combination of harmonic and external data
    FLATHER = -1  # Flather type radiation boundary
    RELAXED = -4  # 3D input with relaxation


class TracerType(IntEnum):
    """Temperature/salinity boundary condition types."""

    NONE = 0  # Not specified
    TIMEHIST = 1  # Time history from temp/salt.th
    CONSTANT = 2  # Constant temperature/salinity
    INITIAL = 3  # Initial profile for inflow
    EXTERNAL = 4  # External model 3D input


class TidalSpecies(IntEnum):
    """Tidal species types."""

    LONG_PERIOD = 0  # Long period (declinational)
    DIURNAL = 1  # Diurnal
    SEMI_DIURNAL = 2  # Semi-diurnal


class TidalDataset(BaseModel):
    """
    This class is used to define the tidal dataset to use from an available pyTMD tidal database. 
    Custom databases can be configured by providing a database.json file in the tidal database directory.
    see https://pytmd.readthedocs.io/en/latest/getting_started/Getting-Started.html
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tidal_database: Optional[Path] = Field(
        None, description="Path to pyTMD tidal database directory. If None, defaults to pyTMD default."
    )
    tidal_model: Optional[str] = Field(
        'FES2014', description="Name of the pyTMD tidal model to use (e.g., 'FES2014')"
    )

    # Basic tidal configuration
    constituents: Union[List[str], str] = Field(
        default_factory=lambda: ["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1"],
        description="Tidal constituents to include",
    )

    # Earth tidal potential settings
    tidal_potential: bool = Field(
        default=False,
        description="Apply Earth tidal potential loading to the model. The coefficients used any selected constituents with species 0, 1, 2.",
    )
    cutoff_depth: float = Field(default=50.0, description="Cutoff depth for Earth tidal potential loading to the model")

    # Nodal corrections
    nodal_corrections: bool = Field(
        default=False,
        description="Apply nodal corrections to tidal constituents",
    )

    tide_interpolation_method: str = Field(
        default='bilinear',
        description="Method for tidal interpolation. see https://pytmd.readthedocs.io/en/latest/api_reference/interpolate.html.",
    )

    extrapolate_tides: bool = Field(
        default=False,
        description="Extrapolate tidal constituents outside the domain. If False, will raise an error if any constituent is outside the domain.",
    )

    extrapolation_distance: float = Field(
        default=50.0,
        description="Distance in kilometre to extrapolate tidal constituents outside the tidal model. Only used if extrapolate_tides is True.",
    )

    extra_databases: Optional[List[Path]] = Field(
        default=[], description="Extra tidal databases loaded from database.json if present"
    )

    def get(self) -> Dict[str, Any]:
        """Get the tidal dataset as a dictionary."""

        # Ensure extra_databases is a list of paths to JSON files
        extra_databases = self.extra_databases
        if self.tidal_database:
            db_json = Path(self.tidal_database) / "database.json"
            if db_json.exists():
                if extra_databases is None:
                    extra_databases = [db_json]
                elif isinstance(extra_databases, list):
                    if db_json not in extra_databases:
                        extra_databases.append(db_json)

        if len(extra_databases) > 0:
            logger.info(f"Loading extra tidal databases from {extra_databases}")

        return {
            "constituents": self.constituents,
            "tidal_database": self.tidal_database,
            "tidal_model": self.tidal_model,
            "tidal_potential": self.tidal_potential,
            "cutoff_depth": self.cutoff_depth,
            "nodal_corrections": self.nodal_corrections,
            "tide_interpolation_method": self.tide_interpolation_method,
            "extra_databases": extra_databases,
        }

    @field_validator("tidal_potential", "nodal_corrections", "extrapolate_tides", mode="before")
    @classmethod
    def ensure_python_bool(cls, v):
        return bool(v)
   

class BoundaryConfig(BaseModel):
    """Configuration for a single SCHISM boundary segment."""

    # Required fields with default values
    elev_type: ElevationType = Field(
        default=ElevationType.NONE, description="Elevation boundary condition type"
    )
    vel_type: VelocityType = Field(
        default=VelocityType.NONE, description="Velocity boundary condition type"
    )
    temp_type: TracerType = Field(
        default=TracerType.NONE, description="Temperature boundary condition type"
    )
    salt_type: TracerType = Field(
        default=TracerType.NONE, description="Salinity boundary condition type"
    )

    # Optional fields for specific boundary types
    # Elevation constants (for ElevationType.CONSTANT)
    ethconst: Optional[float] = Field(
        default=None, description="Constant elevation value (for CONSTANT type)"
    )

    # Velocity/flow constants (for VelocityType.CONSTANT)
    vthconst: Optional[float] = Field(
        default=None, description="Constant velocity/flow value (for CONSTANT type)"
    )

    # Temperature constants and parameters
    tthconst: Optional[float] = Field(
        default=None, description="Constant temperature value (for CONSTANT type)"
    )
    tobc: Optional[float] = Field(
        default=1.0,
        description="Temperature nudging factor (0-1, 1 is strongest nudging)",
    )
    temp_th_path: Optional[str] = Field(
        default=None, description="Path to temperature time history file (for type 1)"
    )
    temp_3d_path: Optional[str] = Field(
        default=None, description="Path to 3D temperature file (for type 4)"
    )

    # Salinity constants and parameters
    sthconst: Optional[float] = Field(
        default=None, description="Constant salinity value (for CONSTANT type)"
    )
    sobc: Optional[float] = Field(
        default=1.0, description="Salinity nudging factor (0-1, 1 is strongest nudging)"
    )
    salt_th_path: Optional[str] = Field(
        default=None, description="Path to salinity time history file (for type 1)"
    )
    salt_3d_path: Optional[str] = Field(
        default=None, description="Path to 3D salinity file (for type 4)"
    )

    # Velocity/flow time history parameters (for VelocityType.TIMEHIST)
    flow_th_path: Optional[str] = Field(
        default=None, description="Path to flow time history file (for type 1)"
    )

    # Relaxation parameters for velocity (for VelocityType.RELAXED)
    inflow_relax: Optional[float] = Field(
        default=0.5,
        description="Relaxation factor for inflow (0-1, 1 is strongest nudging)",
    )
    outflow_relax: Optional[float] = Field(
        default=0.1,
        description="Relaxation factor for outflow (0-1, 1 is strongest nudging)",
    )

    # Flather boundary values (for VelocityType.FLATHER)
    eta_mean: Optional[List[float]] = Field(
        default=None, description="Mean elevation profile for Flather boundary"
    )
    vn_mean: Optional[List[List[float]]] = Field(
        default=None, description="Mean velocity profile for Flather boundary"
    )

    # Space-time parameters
    elev_st_path: Optional[str] = Field(
        default=None,
        description="Path to space-time elevation file (for SPACETIME type)",
    )
    vel_st_path: Optional[str] = Field(
        default=None,
        description="Path to space-time velocity file (for SPACETIME type)",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __str__(self):
        """String representation of the boundary configuration."""
        return (
            f"BoundaryConfig(elev_type={self.elev_type}, vel_type={self.vel_type}, "
            f"temp_type={self.temp_type}, salt_type={self.salt_type})"
        )
 

class BoundaryHandler(BoundaryData):
    """Handler for SCHISM boundary conditions.

    This class extends BoundaryData to handle all SCHISM boundary condition types
    including tidal, river, nested, and hybrid configurations.
    """

    def __init__(
        self,
        grid_path: Union[str, Path],
        tidal_data: Optional[TidalDataset] = None,
        boundary_configs: Optional[Dict[int, BoundaryConfig]] = None,
        *args,
        **kwargs,
    ):
        """Initialize the boundary handler.

        Parameters
        ----------
        grid_path : str or Path
            Path to the SCHISM grid file
        tidal_data : TidalDataset, optional
            Tidal dataset containing specification of tidal forcing
        boundary_configs : dict, optional
            Configuration for each boundary, keyed by boundary index

        """
        super().__init__(grid_path, *args, **kwargs)

        self.tidal_data = tidal_data 
        self.boundary_configs = boundary_configs if boundary_configs is not None else {}

        # For storing start time and run duration
        self._start_time = None
        self._rnday = None

        # Additional file paths for various boundary types
        self.temp_th_path = None  # Temperature time history
        self.temp_3d_path = None  # 3D temperature
        self.salt_th_path = None  # Salinity time history
        self.salt_3d_path = None  # 3D salinity
        self.flow_th_path = None  # Flow time history
        self.elev_st_path = None  # Space-time elevation
        self.vel_st_path = None  # Space-time velocity

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
        **kwargs,
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
            **kwargs,
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
                flags.append(
                    [
                        int(config.elev_type),
                        int(config.vel_type),
                        int(config.temp_type),
                        int(config.salt_type),
                    ]
                )
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
            "eta_mean": [],
            "vn_mean": [],
            "temp_th_path": [],
            "temp_3d_path": [],
            "salt_th_path": [],
            "salt_3d_path": [],
            "flow_th_path": [],
            "elev_st_path": [],
            "vel_st_path": [],
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
                    result["ethconst"].append(
                        config.ethconst if config.ethconst is not None else 0.0
                    )
                else:
                    result["ethconst"].append(0.0)

                if config.vel_type == VelocityType.CONSTANT:
                    result["vthconst"].append(
                        config.vthconst if config.vthconst is not None else 0.0
                    )
                else:
                    result["vthconst"].append(0.0)

                if config.temp_type == TracerType.CONSTANT:
                    result["tthconst"].append(
                        config.tthconst if config.tthconst is not None else 0.0
                    )
                else:
                    result["tthconst"].append(0.0)

                if config.salt_type == TracerType.CONSTANT:
                    result["sthconst"].append(
                        config.sthconst if config.sthconst is not None else 0.0
                    )
                else:
                    result["sthconst"].append(0.0)

                # Nudging factors for temperature and salinity
                result["tobc"].append(config.tobc if config.tobc is not None else 1.0)
                result["sobc"].append(config.sobc if config.sobc is not None else 1.0)

                # Temperature and salinity file paths
                result["temp_th_path"].append(config.temp_th_path)
                result["temp_3d_path"].append(config.temp_3d_path)
                result["salt_th_path"].append(config.salt_th_path)
                result["salt_3d_path"].append(config.salt_3d_path)

                # Flow time history path
                result["flow_th_path"].append(config.flow_th_path)

                # Space-time file paths
                result["elev_st_path"].append(config.elev_st_path)
                result["vel_st_path"].append(config.vel_st_path)

                # Relaxation factors for velocity
                if config.vel_type == VelocityType.RELAXED:
                    result["inflow_relax"].append(
                        config.inflow_relax if config.inflow_relax is not None else 0.5
                    )
                    result["outflow_relax"].append(
                        config.outflow_relax
                        if config.outflow_relax is not None
                        else 0.1
                    )
                else:
                    result["inflow_relax"].append(0.5)  # Default values
                    result["outflow_relax"].append(0.1)

                # Handle Flather boundaries
                if config.vel_type == VelocityType.FLATHER:
                    # Create default values if none provided
                    if config.eta_mean is None:
                        # For testing, create a simple array of zeros with size = num nodes on this boundary
                        # In practice, this should be filled with actual mean elevation values
                        num_nodes = (
                            self.grid.nobn[i]
                            if hasattr(self.grid, "nobn") and i < len(self.grid.nobn)
                            else 1
                        )
                        eta_mean = [0.0] * num_nodes
                    else:
                        eta_mean = config.eta_mean

                    if config.vn_mean is None:
                        # For testing, create a simple array of arrays with zeros
                        num_nodes = (
                            self.grid.nobn[i]
                            if hasattr(self.grid, "nobn") and i < len(self.grid.nobn)
                            else 1
                        )
                        # Assume 5 vertical levels for testing
                        vn_mean = [[0.0] * 5 for _ in range(num_nodes)]
                    else:
                        vn_mean = config.vn_mean

                    result["eta_mean"].append(eta_mean)
                    result["vn_mean"].append(vn_mean)
                else:
                    result["eta_mean"].append(None)
                    result["vn_mean"].append(None)
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
                result["eta_mean"].append(None)
                result["vn_mean"].append(None)
                result["temp_th_path"].append(None)
                result["temp_3d_path"].append(None)
                result["salt_th_path"].append(None)
                result["salt_3d_path"].append(None)
                result["flow_th_path"].append(None)
                result["elev_st_path"].append(None)
                result["vel_st_path"].append(None)

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
        inflow_relax = constants["inflow_relax"] if constants["inflow_relax"] else None
        outflow_relax = (
            constants["outflow_relax"] if constants["outflow_relax"] else None
        )

        # Add flow and flux boundary information
        ncbn = 0
        nfluxf = 0

        # Count the number of flow and flux boundaries
        for i, config in self.boundary_configs.items():
            # Count flow boundaries - both CONSTANT type with non-zero flow value
            # and type 1 (time history) are considered flow boundaries
            if (
                config.vel_type == VelocityType.CONSTANT and config.vthconst is not None
            ) or (config.vel_type == VelocityType.TIMEHIST):
                ncbn += 1

            # Count flux boundaries - type 3 with flux specified
            if config.vel_type == VelocityType.HARMONIC:
                nfluxf += 1

        # Extract file paths
        temp_th_path = (
            constants.get("temp_th_path", [None])[0]
            if constants.get("temp_th_path")
            else None
        )
        temp_3d_path = (
            constants.get("temp_3d_path", [None])[0]
            if constants.get("temp_3d_path")
            else None
        )
        salt_th_path = (
            constants.get("salt_th_path", [None])[0]
            if constants.get("salt_th_path")
            else None
        )
        salt_3d_path = (
            constants.get("salt_3d_path", [None])[0]
            if constants.get("salt_3d_path")
            else None
        )
        flow_th_path = (
            constants.get("flow_th_path", [None])[0]
            if constants.get("flow_th_path")
            else None
        )
        elev_st_path = (
            constants.get("elev_st_path", [None])[0]
            if constants.get("elev_st_path")
            else None
        )
        vel_st_path = (
            constants.get("vel_st_path", [None])[0]
            if constants.get("vel_st_path")
            else None
        )

        # Extract Flather boundary data if available
        eta_mean = (
            constants.get("eta_mean", [None]) if constants.get("eta_mean") else None
        )
        vn_mean = constants.get("vn_mean", [None]) if constants.get("vn_mean") else None

        # Ensure grid boundaries are computed before creating Bctides
        if self.grid is not None:
            if hasattr(self.grid, "compute_bnd") and not hasattr(self.grid, "nob"):
                logger.info("Computing grid boundaries for Bctides")
                self.grid.compute_bnd()
            elif not hasattr(self.grid, "nob") and hasattr(self.grid, "compute_all"):
                logger.info(
                    "Running compute_all to ensure grid boundaries are available"
                )
                self.grid.compute_all()

            # Verify boundaries were computed
            if not hasattr(self.grid, "nob"):
                logger.error(
                    "Failed to compute grid boundaries - grid has no 'nob' attribute"
                )
                raise AttributeError("Grid boundaries could not be computed")

        # Create Bctides object with all the enhanced parameters
        bctides = Bctides(
            hgrid=self.grid,
            flags=flags,
            constituents=self.tidal_data.constituents,
            tidal_database=self.tidal_data.tidal_database,
            tidal_model=self.tidal_data.tidal_model,
            tidal_potential=self.tidal_data.tidal_potential,
            cutoff_depth=self.tidal_data.cutoff_depth,
            nodal_corrections=self.tidal_data.nodal_corrections,
            tide_interpolation_method=self.tidal_data.tide_interpolation_method,
            extrapolate_tides=self.tidal_data.extrapolate_tides,
            extrapolation_distance=self.tidal_data.extrapolation_distance,
            extra_databases=self.tidal_data.extra_databases,
            ethconst=ethconst,
            vthconst=vthconst,
            tthconst=tthconst,
            sthconst=sthconst,
            tobc=tobc,
            sobc=sobc,
            relax=constants.get("inflow_relax", []),  # For backward compatibility
            inflow_relax=inflow_relax,
            outflow_relax=outflow_relax,
            ncbn=ncbn,
            nfluxf=nfluxf,
            elev_th_path=None,  # Time history of elevation is not handled by this path yet
            elev_st_path=elev_st_path,
            flow_th_path=flow_th_path,
            vel_st_path=vel_st_path,
            temp_th_path=temp_th_path,
            temp_3d_path=temp_3d_path,
            salt_th_path=salt_th_path,
            salt_3d_path=salt_3d_path,
        )

        # Set additional properties for Flather boundaries
        if eta_mean and any(x is not None for x in eta_mean):
            bctides.eta_mean = eta_mean
        if vn_mean and any(x is not None for x in vn_mean):
            bctides.vn_mean = vn_mean

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
            raise ValueError(
                "start_time and rnday must be set before writing boundary file"
            )

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
    tidal_database: Union[str, Path] = None,
    tidal_model: Optional[str] = 'FES2014',
    nodal_corrections: bool = True,
    tidal_potential: bool = True,
    cutoff_depth: float = 50.0,
    tide_interpolation_method: str = "bilinear",
) -> BoundaryHandler:
    """Create a tidal-only boundary.

    Parameters
    ----------
    grid_path : str or Path
        Path to SCHISM grid
    constituents : str or list, optional
        Tidal constituents, by default "major"
    tidal_database : str or Path, optional
        Tidal database path for pyTMD to use, by default None
    tidal_model : str, optional
        Tidal model to use, by default 'FES2014'
    nodal_corrections : bool, optional
        Whether to apply nodal corrections, by default True
    tidal_potential : bool, optional
        Whether to include tidal potential, by default True
    cutoff_depth : float, optional
        Depth threshold for tidal potential, by default 50.0

    Returns
    -------
    BoundaryHandler
        Configured tidal boundary
    """

    tidal_data = TidalDataset(
        constituents=constituents,
        tidal_database=tidal_database,
        tidal_model=tidal_model,
        nodal_corrections=nodal_corrections,
        tidal_potential=tidal_potential,
        cutoff_depth=cutoff_depth,
        tide_interpolation_method=tide_interpolation_method,)

    boundary = BoundaryHandler(
        grid_path=grid_path,
        constituents=constituents,
        tidal_data=tidal_data,
    )

    # Set default configuration for all boundaries: pure tidal
    boundary.set_boundary_type(
        0,  # Will be applied to all boundaries
        elev_type=ElevationType.HARMONIC,
        vel_type=VelocityType.HARMONIC,
    )

    return boundary


# High-level factory functions for creating boundary configurations


def create_tidal_only_boundary_config(
    constituents: Union[str, List[str]] = "major",
    tidal_database: Union[str, Path] = None,
    tidal_model: Optional[str] = 'FES2014',
    nodal_corrections: bool = True,
    tidal_potential: bool = True,
    cutoff_depth: float = 50.0,
    tide_interpolation_method: str = "bilinear",
):
    """
    Create a configuration where all open boundaries are treated as tidal boundaries.

    Parameters
    ----------
    constituents : str or list, optional
        Tidal constituents to include, by default "major"
    tidal_database : str or Path, optional
        Path to tidal database for pyTMD, by default None
    tidal_model : str, optional
        Tidal model to use, by default 'FES2014'
    nodal_corrections : bool, optional
        Whether to apply nodal corrections, by default True
    tidal_potential : bool, optional
        Whether to include tidal potential, by default True
    cutoff_depth : float, optional
        Depth threshold for tidal potential, by default 50.0
    tide_interpolation_method : str, optional
        Method for tide interpolation, by default "bilinear"

    Returns
    -------
    SCHISMDataBoundaryConditions
        Configured boundary conditions
    """
    from rompy.schism.data import SCHISMDataBoundaryConditions
    
    # Create tidal dataset 
    tidal_data = TidalDataset(
        constituents=constituents,
        tidal_database=tidal_database,
        tidal_model=tidal_model,
        nodal_corrections=nodal_corrections,
        tidal_potential=tidal_potential,
        cutoff_depth=cutoff_depth,
        tide_interpolation_method=tide_interpolation_method,
    )

    # Create the config with tidal setup
    config = SCHISMDataBoundaryConditions(
        tidal_data=tidal_data,
        setup_type="tidal",
        boundaries={},
        hotstart_config=None,
    )

    return config


def create_hybrid_boundary_config(
    constituents: Union[str, List[str]] = "major",
    tidal_database: Union[str, Path] = None,
    tidal_model: Optional[str] = 'FES2014',
    nodal_corrections: bool = True,
    tidal_potential: bool = True,
    cutoff_depth: float = 50.0,
    tide_interpolation_method: str = "bilinear",
    elev_source: Optional[Union[Any, Any]] = None,
    vel_source: Optional[Union[Any, Any]] = None,
    temp_source: Optional[Union[Any, Any]] = None,
    salt_source: Optional[Union[Any, Any]] = None,
):
    """
    Create a configuration for hybrid harmonic + external data boundaries.

    Parameters
    ----------
    constituents : str or list, optional
        Tidal constituents to include, by default "major"
    tidal_database : str or Path, optional
        Path to tidal database for pyTMD, by default None
    tidal_model : str, optional
        Tidal model to use, by default 'FES2014'
    nodal_corrections : bool, optional
        Whether to apply nodal corrections, by default True
    tidal_potential : bool, optional
        Whether to include tidal potential, by default True
    cutoff_depth : float, optional
        Depth threshold for tidal potential, by default 50.0
    tide_interpolation_method : str, optional
        Method for tide interpolation, by default "bilinear"
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
    from rompy.schism.data import SCHISMDataBoundaryConditions, BoundarySetupWithSource
    from rompy.schism.tides_enhanced import TidalDataset
    
    # Create tidal dataset 
    tidal_data = TidalDataset(
        constituents=constituents,
        tidal_database=tidal_database,
        tidal_model=tidal_model,
        nodal_corrections=nodal_corrections,
        tidal_potential=tidal_potential,
        cutoff_depth=cutoff_depth,
        tide_interpolation_method=tide_interpolation_method,
    )

    # Create the config with hybrid setup
    config = SCHISMDataBoundaryConditions(
        tidal_data=tidal_data,
        setup_type="hybrid",
        boundaries={
            0: BoundarySetupWithSource(
                elev_type=ElevationType.HARMONICEXTERNAL,
                vel_type=VelocityType.HARMONICEXTERNAL,
                temp_type=TracerType.EXTERNAL if temp_source else TracerType.INITIAL,
                salt_type=TracerType.EXTERNAL if salt_source else TracerType.INITIAL,
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
    constituents: Union[str, List[str]] = "major",
    tidal_database: Union[str, Path] = None,
    tidal_model: Optional[str] = 'FES2014',
    nodal_corrections: bool = True,
    tidal_potential: bool = True,
    cutoff_depth: float = 50.0,
    tide_interpolation_method: str = "bilinear",
):
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
    constituents : str or list, optional
        Tidal constituents to include, by default "major"
    tidal_database : str or Path, optional
        Path to tidal database for pyTMD, by default None
    tidal_model : str, optional
        Tidal model to use, by default 'FES2014'
    nodal_corrections : bool, optional
        Whether to apply nodal corrections, by default True
    tidal_potential : bool, optional
        Whether to include tidal potential, by default True
    cutoff_depth : float, optional
        Depth threshold for tidal potential, by default 50.0
    tide_interpolation_method : str, optional
        Method for tide interpolation, by default "bilinear"

    Returns
    -------
    SCHISMDataBoundaryConditions
        Configured boundary conditions
    """
    from rompy.schism.data import SCHISMDataBoundaryConditions, BoundarySetupWithSource
    from rompy.schism.tides_enhanced import TidalDataset
    
    # Create tidal dataset if both paths are provided and needed
    tidal_data = None
    if (
        other_boundaries in ["tidal", "hybrid"]
    ):
        tidal_data = TidalDataset(
            constituents=constituents,
            tidal_database=tidal_database,
            tidal_model=tidal_model,
            nodal_corrections=nodal_corrections,
            tidal_potential=tidal_potential,
            cutoff_depth=cutoff_depth,
            tide_interpolation_method=tide_interpolation_method,
        )

    # Create the basic config
    config = SCHISMDataBoundaryConditions(
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
    elev_source: Optional[Union[Any, Any]] = None,
    vel_source: Optional[Union[Any, Any]] = None,
    temp_source: Optional[Union[Any, Any]] = None,
    salt_source: Optional[Union[Any, Any]] = None,
    constituents: Union[str, List[str]] = "major",
    tidal_database: Union[str, Path] = None,
    tidal_model: Optional[str] = 'FES2014',
    nodal_corrections: bool = True,
    tidal_potential: bool = True,
    cutoff_depth: float = 50.0,
    tide_interpolation_method: str = "bilinear",
):
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
    elev_source : Union[DataBlob, SCHISMDataBoundary], optional
        Data source for elevation
    vel_source : Union[DataBlob, SCHISMDataBoundary], optional
        Data source for velocity
    temp_source : Union[DataBlob, SCHISMDataBoundary], optional
        Data source for temperature
    salt_source : Union[DataBlob, SCHISMDataBoundary], optional
        Data source for salinity
    constituents : str or list, optional
        Tidal constituents to include, by default "major"
    tidal_database : str or Path, optional
        Path to tidal database for pyTMD, by default None
    tidal_model : str, optional
        Tidal model to use, by default 'FES2014'
    nodal_corrections : bool, optional
        Whether to apply nodal corrections, by default True
    tidal_potential : bool, optional
        Whether to include tidal potential, by default True
    cutoff_depth : float, optional
        Depth threshold for tidal potential, by default 50.0
    tide_interpolation_method : str, optional
        Method for tide interpolation, by default "bilinear"

    Returns
    -------
    SCHISMDataBoundaryConditions
        Configured boundary conditions
    """
    from rompy.schism.data import SCHISMDataBoundaryConditions, BoundarySetupWithSource
    from rompy.schism.tides_enhanced import TidalDataset
    
    # Create tidal dataset if both paths are provided and needed
    tidal_data = None
    if with_tides:
        tidal_data = TidalDataset(
            constituents=constituents,
            tidal_database=tidal_database,
            tidal_model=tidal_model,
            nodal_corrections=nodal_corrections,
            tidal_potential=tidal_potential,
            cutoff_depth=cutoff_depth,
            tide_interpolation_method=tide_interpolation_method,
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
    elev_type = ElevationType.HARMONICEXTERNAL if with_tides else ElevationType.EXTERNAL

    # Add the nested boundary configuration
    config.boundaries[0] = BoundarySetupWithSource(
        elev_type=elev_type,
        vel_type=VelocityType.RELAXED,
        temp_type=TracerType.EXTERNAL if temp_source else TracerType.NONE,
        salt_type=TracerType.EXTERNAL if salt_source else TracerType.NONE,
        inflow_relax=inflow_relax,
        outflow_relax=outflow_relax,
        elev_source=elev_source,
        vel_source=vel_source,
        temp_source=temp_source,
        salt_source=salt_source,
    )

    return config


# Backward compatibility alias
TidalBoundary = BoundaryHandler


def create_hybrid_boundary(
    grid_path: Union[str, Path],
    constituents: Union[str, List[str]] = "major",
    tidal_database: Union[str, Path] = None,
    tidal_model: Optional[str] = 'FES2014',
    nodal_corrections: bool = True,
    tidal_potential: bool = True,
    cutoff_depth: float = 50.0,
    tide_interpolation_method: str = "bilinear",
) -> BoundaryHandler:
    """Create a hybrid boundary with tides + external data.

    Parameters
    ----------
    grid_path : str or Path
        Path to SCHISM grid
    constituents : str or list, optional
        Tidal constituents to include, by default "major"
    tidal_database : str or Path, optional
        Path to tidal database for pyTMD, by default None
    tidal_model : str, optional
        Tidal model to use, by default 'FES2014'
    nodal_corrections : bool, optional
        Whether to apply nodal corrections, by default True
    tidal_potential : bool, optional
        Whether to include tidal potential, by default True
    cutoff_depth : float, optional
        Depth threshold for tidal potential, by default 50.0
    tide_interpolation_method : str, optional
        Method for tide interpolation, by default "bilinear"

    Returns
    -------
    BoundaryHandler
        Configured hybrid boundary
    """

    tidal_data = TidalDataset(
        constituents=constituents,
        tidal_database=tidal_database,
        tidal_model=tidal_model,
        nodal_corrections=nodal_corrections,
        tidal_potential=tidal_potential,
        cutoff_depth=cutoff_depth,
        tide_interpolation_method=tide_interpolation_method,
    )

    boundary = BoundaryHandler(
        grid_path=grid_path,
        tidal_data=tidal_data
    )

    # Set default configuration for all boundaries: tidal + spacetime
    boundary.set_boundary_type(
        0,  # Will be applied to all boundaries
        elev_type=ElevationType.HARMONICEXTERNAL,
        vel_type=VelocityType.HARMONICEXTERNAL,
    )

    return boundary


def create_river_boundary(
    grid_path: Union[str, Path],
    river_flow: float = -100.0,  # Negative for inflow
    river_boundary_index: int = 0,
) -> BoundaryHandler:
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
    BoundaryHandler
        Configured river boundary
    """
    boundary = BoundaryHandler(grid_path=grid_path)

    # Set river boundary
    boundary.set_boundary_type(
        river_boundary_index,
        elev_type=ElevationType.NONE,  # No elevation specified
        vel_type=VelocityType.CONSTANT,  # Constant flow
        vthconst=river_flow,  # Flow value
    )

    return boundary


def create_nested_boundary(
    grid_path: Union[str, Path],
    with_tides: bool = False,
    inflow_relax: float = 0.8,
    outflow_relax: float = 0.8,
    constituents: Union[str, List[str]] = "major",
    tidal_database: Union[str, Path] = None,
    tidal_model: Optional[str] = 'FES2014',
    nodal_corrections: bool = True,
    tidal_potential: bool = True,
    cutoff_depth: float = 50.0,
    tide_interpolation_method: str = "bilinear",
) -> BoundaryHandler:
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
        Tidal constituents to include, by default "major"
    tidal_database : str or Path, optional
        Path to tidal database for pyTMD, by default None
    tidal_model : str, optional
        Tidal model to use, by default 'FES2014'
    nodal_corrections : bool, optional
        Whether to apply nodal corrections, by default True
    tidal_potential : bool, optional
        Whether to include tidal potential, by default True
    cutoff_depth : float, optional
        Depth threshold for tidal potential, by default 50.0
    tide_interpolation_method : str, optional
        Method for tide interpolation, by default "bilinear"

    Returns
    -------
    BoundaryHandler
        Configured nested boundary
    """

    tidal_data = None
    if with_tides:
        tidal_data = TidalDataset(
            constituents=constituents,
            tidal_database=tidal_database,
            tidal_model=tidal_model,
            nodal_corrections=nodal_corrections,
            tidal_potential=tidal_potential,
            cutoff_depth=cutoff_depth,
            tide_interpolation_method=tide_interpolation_method,
        )

    boundary = BoundaryHandler(
        grid_path=grid_path,
        constituents=constituents if with_tides else None,
        tidal_data=tidal_data,
    )

    if with_tides:
        # Tides + external data with relaxation
        boundary.set_boundary_type(
            0,  # Will be applied to all boundaries
            elev_type=ElevationType.HARMONICEXTERNAL,
            vel_type=VelocityType.RELAXED,
            temp_type=TracerType.EXTERNAL,
            salt_type=TracerType.EXTERNAL,
            inflow_relax=inflow_relax,
            outflow_relax=outflow_relax,
        )
    else:
        # Just external data with relaxation
        boundary.set_boundary_type(
            0,  # Will be applied to all boundaries
            elev_type=ElevationType.EXTERNAL,
            vel_type=VelocityType.RELAXED,
            temp_type=TracerType.EXTERNAL,
            salt_type=TracerType.EXTERNAL,
            inflow_relax=inflow_relax,
            outflow_relax=outflow_relax,
        )

    return boundary
