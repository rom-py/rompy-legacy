"""
Enhanced implementation of SCHISM tidal boundary conditions.

This module provides a more comprehensive and flexible approach to
handling SCHISM boundary conditions, with full support for all boundary
types specified in the SCHISM documentation.
"""

import logging
import os
import sys
import tempfile
import subprocess
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, cast

import numpy as np
import xarray as xr
from pydantic import ConfigDict, Field, model_validator, field_validator, BaseModel

# Ensure PyLibs is in path
sys.path.append("/home/tdurrant/source/pylibs")

# Import PyLibs functions directly
from pylib import *
from src.schism_file import read_schism_hgrid, loadz

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


class ConstituentSet(IntEnum):
    """Predefined constituent sets."""

    MAJOR = 0  # Major tidal constituents
    ALL = 1  # All tidal constituents (major + minor)
    CUSTOM = 2  # Custom set of constituents


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

    @field_validator("eta_mean")
    def validate_eta_mean(cls, v, values):
        """Validate that eta_mean is provided if needed."""
        vel_type = values.data.get("vel_type")
        if vel_type == VelocityType.FLATHER and not v:
            logger.warning("eta_mean should be provided for Flather boundary conditions")
        return v or []

    @field_validator("vn_mean")
    def validate_vn_mean(cls, v, values):
        """Validate that vn_mean is provided if needed."""
        vel_type = values.data.get("vel_type")
        if vel_type == VelocityType.FLATHER and not v:
            logger.warning("vn_mean should be provided for Flather boundary conditions")
        return v or [[0.0]]


class BctidesConfig(BaseModel):
    """Configuration for bctides.in generation."""

    tidal_database: str = Field("tpxo", description="Tidal database to use")
    constituents: Union[str, List[str]] = Field("major", description="Tidal constituents")
    boundaries: Dict[int, BoundaryConfig] = Field(
        default_factory=dict, description="Configuration for each boundary"
    )
    ntip: int = Field(0, description="Number of earth tidal potential regions (0 to disable)")
    tip_dp: float = Field(1.0, description="Depth threshold for tidal potential")
    cutoff_depth: float = Field(50.0, description="Cutoff depth for tides")

    # Paths to tidal database files
    tidal_elevations: Optional[str] = Field(
        None, description="Path to tidal elevations file (TPXO format)"
    )
    tidal_velocities: Optional[str] = Field(
        None, description="Path to tidal velocities file (TPXO format)"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_boundary_flags(self) -> List[List[int]]:
        """Convert boundary configurations to flag lists for Bctides."""
        if not self.boundaries:
            return [[5, 5, 0, 0]]  # Default: tidal elevations and velocities

        max_boundary = max(self.boundaries.keys())
        flags = []

        for i in range(max_boundary + 1):
            if i in self.boundaries:
                config = self.boundaries[i]
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
        """Get constant values for boundaries."""
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
        
        if not self.boundaries:
            return result
            
        max_boundary = max(self.boundaries.keys())
        
        for i in range(max_boundary + 1):
            if i in self.boundaries:
                config = self.boundaries[i]
                result["ethconst"].append(config.ethconst if config.ethconst is not None else 0.0)
                result["vthconst"].append(config.vthconst if config.vthconst is not None else 0.0)
                result["tthconst"].append(config.tthconst if config.tthconst is not None else 0.0)
                result["sthconst"].append(config.sthconst if config.sthconst is not None else 0.0)
                result["tobc"].append(config.tobc)
                result["sobc"].append(config.sobc)
                result["inflow_relax"].append(config.inflow_relax)
                result["outflow_relax"].append(config.outflow_relax)
            else:
                result["ethconst"].append(0.0)
                result["vthconst"].append(0.0)
                result["tthconst"].append(0.0)
                result["sthconst"].append(0.0)
                result["tobc"].append(1.0)
                result["sobc"].append(1.0)
                result["inflow_relax"].append(0.5)
                result["outflow_relax"].append(0.1)
                
        return result
    
    def create_bctides(self, hgrid) -> "Bctides":
        """Create a Bctides instance from this configuration."""
        flags = self.get_boundary_flags()
        constants = self.get_constant_values()
        
        # Create and return Bctides object
        return Bctides(
            hgrid=hgrid,
            flags=flags,
            constituents=self.constituents,
            tidal_database=self.tidal_database,
            ntip=self.ntip,
            tip_dp=self.tip_dp,
            cutoff_depth=self.cutoff_depth,
            ethconst=constants["ethconst"],
            vthconst=constants["vthconst"],
            tthconst=constants["tthconst"],
            sthconst=constants["sthconst"],
            tobc=constants["tobc"],
            sobc=constants["sobc"],
            tidal_elevations=self.tidal_elevations,
            tidal_velocities=self.tidal_velocities,
            inflow_relax=constants["inflow_relax"],
            outflow_relax=constants["outflow_relax"],
            # Add boundary-specific configurations
            boundaries=self.boundaries
        )


class Bctides:
    """Enhanced implementation of SCHISM tidal boundary conditions.
    
    This class provides comprehensive support for all SCHISM boundary condition types
    as described in the documentation, with improved handling of different boundary types
    and configuration options.
    """

    def __init__(
        self,
        hgrid,
        flags=None,
        constituents="major",
        tidal_database="tpxo",
        ntip=0,
        tip_dp=1.0,
        cutoff_depth=50.0,
        ethconst=None,
        vthconst=None,
        tthconst=None,
        sthconst=None,
        tobc=None,
        sobc=None,
        relax=None,
        tidal_elevations=None,  # Path to tidal elevations file (TPXO format)
        tidal_velocities=None,  # Path to tidal velocities file (TPXO format)
        inflow_relax=None,      # Relaxation factors for inflow (type -4)
        outflow_relax=None,     # Relaxation factors for outflow (type -4)
        boundaries=None,        # Comprehensive boundary configurations
    ):
        """Initialize enhanced Bctides handler.

        Parameters
        ----------
        hgrid : Grid or str
            SCHISM horizontal grid
        flags : list of lists, optional
            Boundary condition flags
        constituents : str or list, optional
            Tidal constituents to use, by default "major"
        tidal_database : str, optional
            Tidal database to use, by default "tpxo"
        ntip : int, optional
            Number of earth tidal potential regions (0 to disable), by default 0
        tip_dp : float, optional
            Depth threshold for tidal potential, by default 1.0
        cutoff_depth : float, optional
            Cutoff depth for tides, by default 50.0
        ethconst : list, optional
            Constant elevation for each boundary
        vthconst : list, optional
            Constant velocity for each boundary
        tthconst : list, optional
            Constant temperature for each boundary
        sthconst : list, optional
            Constant salinity for each boundary
        tobc : list, optional
            Temperature OBC values
        sobc : list, optional
            Salinity OBC values
        relax : list, optional
            Relaxation parameters
        tidal_elevations : str, optional
            Path to tidal elevations file
        tidal_velocities : str, optional
            Path to tidal velocities file
        inflow_relax : list, optional
            Relaxation factors for inflow (type -4)
        outflow_relax : list, optional
            Relaxation factors for outflow (type -4)
        boundaries : dict, optional
            Comprehensive boundary configurations
        """
        self.flags = flags or [[5, 5, 0, 0]]  # Default to tidal elevation and velocity
        self.ntip = ntip
        self.tip_dp = tip_dp
        self.cutoff_depth = cutoff_depth
        self.ethconst = ethconst or []
        self.vthconst = vthconst or []
        self.tthconst = tthconst or []
        self.sthconst = sthconst or []
        self.tobc = tobc or [1.0]
        self.sobc = sobc or [1.0]
        self.relax = relax or []
        self.tidal_database = tidal_database
        self.inflow_relax = inflow_relax or [0.5]  # Default moderate relaxation
        self.outflow_relax = outflow_relax or [0.1]  # Default weak relaxation
        self.boundaries = boundaries or {}

        # Store tidal file paths
        self.tidal_elevations = tidal_elevations
        self.tidal_velocities = tidal_velocities

        # Store start time and run duration (will be set by SCHISMDataTides.get())
        self._start_time = None
        self._rnday = None

        # Load grid from file or object
        if isinstance(hgrid, str) or isinstance(hgrid, Path):
            hgrid_path = str(hgrid)
            if hgrid_path.endswith(".npz"):
                self.gd = loadz(hgrid_path).hgrid
                self.gd.x = self.gd.lon
                self.gd.y = self.gd.lat
            else:
                self.gd = read_schism_hgrid(hgrid_path)
        else:
            # Assume it's already a grid object
            self.gd = hgrid

        # Define constituent sets
        self.major_constituents = ["O1", "K1", "Q1", "P1", "M2", "S2", "K2", "N2"]
        self.minor_constituents = ["MM", "Mf", "M4", "MN4", "MS4", "2N2", "S1"]

        # Determine which constituents to use
        if isinstance(constituents, str):
            if constituents.lower() == "major":
                self.tnames = self.major_constituents
            elif constituents.lower() == "all":
                self.tnames = self.major_constituents + self.minor_constituents
            else:
                # Assume it's a comma-separated string
                self.tnames = [c.strip() for c in constituents.split(",")]
        elif isinstance(constituents, list):
            self.tnames = constituents
        else:
            # Default to major constituents
            self.tnames = self.major_constituents

        # For storing tidal factors
        self.amp = []
        self.freq = []
        self.nodal = []
        self.tear = []
        self.species = []

        # Pre-defined frequencies and factors for common constituents
        # These will be used if we can't find tide_fac_const.npz
        # Values from tide_fac_const.npz via loadz('/sciclone/data10/wangzg/FES2014/tide_fac_const/tide_fac_const.npz')
        self.default_factors = {
            # name: [amplitude, frequency(cycles/second), species_type]
            "M2": [0.242334, 0.0000140519, 2],  # Semi-diurnal
            "S2": [0.112743, 0.0000145444, 2],  # Semi-diurnal
            "N2": [0.046398, 0.0000137880, 2],  # Semi-diurnal
            "K2": [0.030704, 0.0000145444, 2],  # Semi-diurnal
            "K1": [0.141565, 0.0000072921, 1],  # Diurnal
            "O1": [0.100514, 0.0000067598, 1],  # Diurnal
            "P1": [0.046843, 0.0000072521, 1],  # Diurnal
            "Q1": [0.019256, 0.0000064959, 1],  # Diurnal
            "MF": [0.042041, 0.0000005323, 0],  # Long period
            "MM": [0.022191, 0.0000002639, 0],  # Long period
            "SSA": [0.019669, 0.0000000639, 0],  # Long period
        }

    def start_date(self, dt=None):
        """Set or get the start date."""
        if dt is not None:
            self._start_time = dt
        return self._start_time
    
    def _get_boundary_config(self, ibnd):
        """Get configuration for a specific boundary."""
        return self.boundaries.get(ibnd) if self.boundaries else None
    
    def _get_boundary_flag(self, ibnd, field_idx):
        """Get a specific boundary flag value."""
        if ibnd < len(self.flags):
            flags = self.flags[ibnd]
            if field_idx < len(flags):
                return flags[field_idx]
        return 0  # Default to "not specified"
    
    def _get_tidal_factors(self):
        """Get tidal amplitude, frequency, and species for constituents.

        Uses constituent information from TPXO files or default factors if needed.
        """
        # Check if we already have tidal factors
        if hasattr(self, "amp") and len(self.amp) > 0:
            return

        logger.info("Computing tidal factors")

        # Initialize arrays
        self.amp = []
        self.freq = []
        self.species = []

        # Try to get tidal constituent information directly from TPXO file
        if self.tidal_elevations and os.path.exists(self.tidal_elevations):
            try:
                logger.info(f"Getting tidal constituents from {self.tidal_elevations}")
                # Open the TPXO elevation file
                nc = ReadNC(self.tidal_elevations, 1)
                # Get list of tidal constituents from the file
                if hasattr(nc, "variables") and "con" in nc.variables:
                    cons = nc.variables["con"][:]
                    # Convert constituents to strings
                    file_constituents = []
                    for i in range(len(cons)):
                        const_name = "".join(
                            [c.decode("utf-8") for c in cons[i]]
                        ).strip()
                        file_constituents.append(const_name)

                    logger.info(f"Found constituents in TPXO file: {file_constituents}")

                    # Check if our requested constituents are in the file
                    for tname in self.tnames:
                        if tname.upper() in [c.upper() for c in file_constituents]:
                            # Use default factors for these constituents
                            tname_upper = tname.upper()
                            if tname_upper in self.default_factors:
                                default_factors = self.default_factors[tname_upper]
                                self.amp.append(default_factors[0])
                                self.freq.append(default_factors[1])
                                self.species.append(default_factors[2])
                                logger.info(
                                    f"Using default factors for {tname}: amp={default_factors[0]}, freq={default_factors[1]}, species={default_factors[2]}"
                                )
                            else:
                                # If no default factors, use generic values
                                logger.warning(
                                    f"No default factors for {tname}, using generic values"
                                )
                                species_type = self._determine_species(tname)
                                self.amp.append(0.1)
                                self.freq.append(0.00001)
                                self.species.append(species_type)
                        else:
                            logger.warning(
                                f"Requested constituent {tname} not found in TPXO file"
                            )
                            raise ValueError(
                                f"Constituent {tname} not found in TPXO file {self.tidal_elevations}"
                            )
                nc.close()
            except Exception as e:
                logger.error(f"Error reading constituents from TPXO file: {e}")
                raise
        else:
            # If no TPXO file, use default factors
            logger.warning(
                "No TPXO elevation file provided, using default tidal factors"
            )
            for tname in self.tnames:
                tname_upper = tname.upper()
                if tname_upper in self.default_factors:
                    default_factors = self.default_factors[tname_upper]
                    self.amp.append(default_factors[0])
                    self.freq.append(default_factors[1])
                    self.species.append(default_factors[2])
                    logger.info(
                        f"Using default factors for {tname}: amp={default_factors[0]}, freq={default_factors[1]}, species={default_factors[2]}"
                    )
                else:
                    # If no default factors, use generic values based on name
                    logger.warning(
                        f"No default factors for {tname}, using generic values"
                    )
                    species_type = self._determine_species(tname)
                    self.amp.append(0.1)
                    self.freq.append(0.00001)
                    self.species.append(species_type)

        # Set default nodal factors if tide_fac_improved isn't available
        self.nodal = [1.0] * len(self.tnames)
        self.tear = [0.0] * len(self.tnames)

        # Try to get nodal factors using tide_fac_improved if it's available
        try:
            self._compute_nodal_factors()
        except Exception as e:
            logger.warning(
                f"Could not compute nodal factors using tide_fac_improved: {e}"
            )
            logger.warning("Using default nodal factors of 1.0 and earth tear of 0.0")
    
    def _determine_species(self, tname):
        """Determine the tidal species based on constituent name."""
        tname = tname.upper()
        if tname in ["O1", "K1", "P1", "Q1"]:
            return TidalSpecies.DIURNAL
        elif tname in ["MM", "Mm", "Mf", "MF"]:
            return TidalSpecies.LONG_PERIOD
        else:
            return TidalSpecies.SEMI_DIURNAL  # Default to semi-diurnal

    def _compute_nodal_factors(self):
        """Compute nodal factors using tide_fac_improved.

        If tide_fac_improved is not available or fails, nodal factors
        will default to 1.0 and tear to 0.0.
        """
        if not self._start_time or not self._rnday:
            logger.warning(
                "start_time and rnday must be set before computing nodal factors"
            )
            return

        # Initialize nodal factors with default values
        # These will be used if tide_fac_improved is not available
        self.nodal = [1.0] * len(self.tnames)
        self.tear = [0.0] * len(self.tnames)

        # Try to find the tide_fac executable
        try:
            tide_fac_exe = self._find_tide_fac_exe()
        except FileNotFoundError as e:
            logger.warning(f"tide_fac executable not found: {e}")
            logger.warning("Using default nodal factors of 1.0 and earth tear of 0.0")
            return

        # Ensure we have the required parameters
        if isinstance(self._start_time, datetime):
            year = self._start_time.year
            month = self._start_time.month
            day = self._start_time.day
            hour = self._start_time.hour
        else:
            # Assume it's a list [year, month, day, hour]
            year, month, day, hour = self._start_time

        # Create input file for tide_fac_improved
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".in") as fid:
            fid.write(f"{self._rnday}\n{hour} {day} {month} {year}\n0\n")
            tide_fac_in = fid.name

        tide_fac_out = tide_fac_in.replace(".in", ".out")

        try:
            # Run the tidal factor calculator
            cmd = f"{tide_fac_exe} < {tide_fac_in} > {tide_fac_out}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to run {tide_fac_exe}: {result.stderr}")

            # Read nodal factors
            with open(tide_fac_out, "r") as f:
                lines = [i for i in f.readlines() if len(i.split()) == 3]

            for i, tname in enumerate(self.tnames):
                found = False
                for line in lines:
                    if line.strip().startswith(tname.upper()):
                        parts = line.strip().split()
                        self.nodal[i] = float(parts[1])
                        self.tear[i] = float(parts[2])
                        found = True
                        break

                if not found:
                    logger.warning(
                        f"Constituent {tname} not found in tide_fac_out, using default values"
                    )
        except Exception as e:
            logger.warning(f"Error computing nodal factors: {e}")
            logger.warning("Using default nodal factors of 1.0 and earth tear of 0.0")
        finally:
            # Clean up temporary files
            for fname in [tide_fac_in, tide_fac_out]:
                if os.path.exists(fname):
                    try:
                        os.remove(fname)
                    except:
                        pass

    def _find_tide_fac_exe(self):
        """Find tide_fac_improved executable.

        Searches in common locations and PATH.

        Returns
        -------
        str
            Path to tide_fac_improved executable

        Raises
        ------
        FileNotFoundError
            If tide_fac_improved cannot be found
        """
        # Common locations for tide_fac_improved
        common_locations = [
            # Local directories
            "./tide_fac_improved",
            "./tide_fac",
            # System directories
            "/usr/bin/tide_fac_improved",
            "/usr/local/bin/tide_fac_improved",
            # Home directory
            os.path.expanduser("~/tide_fac_improved"),
            os.path.expanduser("~/bin/tide_fac_improved"),
            # If installed with conda
            os.path.join(os.environ.get("CONDA_PREFIX", ""), "bin", "tide_fac_improved"),
        ]

        # Check common locations first
        for loc in common_locations:
            if os.path.exists(loc) and os.access(loc, os.X_OK):
                logger.info(f"Found tide_fac_improved at {loc}")
                return loc

        # Check if tide_fac_improved is in PATH
        try:
            result = subprocess.run(
                ["which", "tide_fac_improved"], capture_output=True, text=True
            )
            if result.returncode == 0:
                exe_path = result.stdout.strip()
                logger.info(f"Found tide_fac_improved in PATH at {exe_path}")
                return exe_path
        except:
            pass

        # Try to search for it in common bin directories
        search_dirs = [
            "/usr/bin",
            "/usr/local/bin",
            "/opt/bin",
            os.path.expanduser("~/bin"),
        ]
        if "PATH" in os.environ:
            search_dirs.extend(os.environ["PATH"].split(":"))

        for directory in search_dirs:
            candidate = os.path.join(directory, "tide_fac_improved")
            if os.path.exists(candidate) and os.access(candidate, os.X_OK):
                logger.info(f"Found tide_fac_improved at {candidate}")
                return candidate

            # Also try without the "_improved" suffix
            candidate = os.path.join(directory, "tide_fac")
            if os.path.exists(candidate) and os.access(candidate, os.X_OK):
                logger.info(f"Found tide_fac at {candidate}")
                return candidate

        # Try to download it
        try:
            logger.info("Attempting to download tide_fac_improved...")
            import tempfile
            import shutil
            import urllib.request

            # URLs for potential downloads
            urls = [
                "https://github.com/schism-dev/schism/raw/master/src/Utility/Tides/tide_fac_improved.f90",
                "https://github.com/schism-dev/schism/raw/master/src/Utility/Tides/tide_fac.f90",
            ]

            # Try to download and compile
            with tempfile.TemporaryDirectory() as tmpdir:
                for url in urls:
                    try:
                        # Download source code
                        source_file = os.path.join(tmpdir, os.path.basename(url))
                        urllib.request.urlretrieve(url, source_file)

                        # Try to compile
                        exe_name = (
                            "tide_fac_improved"
                            if "improved" in url
                            else "tide_fac"
                        )
                        exe_path = os.path.join(tmpdir, exe_name)
                        compile_cmd = f"gfortran -o {exe_path} {source_file}"
                        result = subprocess.run(
                            compile_cmd, shell=True, capture_output=True, text=True
                        )

                        if result.returncode == 0 and os.path.exists(exe_path):
                            # Copy to user's bin directory
                            user_bin = os.path.expanduser("~/bin")
                            os.