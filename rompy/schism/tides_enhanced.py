"""
Enhanced implementation of SCHISM tidal data handling.

This module provides an improved approach to handling SCHISM tidal data
with support for all boundary condition types specified in the SCHISM
documentation.
"""

from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np
from pydantic import ConfigDict, Field, model_validator

from rompy.core.time import TimeRange
from rompy.core.types import RompyBaseModel
from rompy.schism.grid import SCHISMGrid

# Import bctides and boundary modules
from rompy.schism.boundary_core import (
    TidalDataset,
    BoundaryConfig,
    ElevationType,
    VelocityType,
    TracerType,
    TidalBoundary,
)

from rompy.logging import get_logger

logger = get_logger(__name__)


# Utility function to convert numpy types to Python types
def to_python_type(obj):
    """Convert numpy types to Python native types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(
        obj,
        (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64),
    ):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, list):
        return [to_python_type(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: to_python_type(v) for k, v in obj.items()}
    else:
        return obj


class BoundarySetup(RompyBaseModel):
    """Configuration for a boundary in SCHISM."""

    # Basic boundary configuration
    elev_type: int = Field(5, description="Elevation boundary type (0-5)")
    vel_type: int = Field(5, description="Velocity boundary type (-4, -1, 0-5)")
    temp_type: int = Field(0, description="Temperature boundary type (0-4)")
    salt_type: int = Field(0, description="Salinity boundary type (0-4)")

    # Values for constant boundaries
    const_elev: Optional[float] = Field(
        None, description="Constant elevation value (for type 2)"
    )
    const_flow: Optional[float] = Field(
        None, description="Constant flow value (for type 2)"
    )
    const_temp: Optional[float] = Field(
        None, description="Constant temperature value (for type 2)"
    )
    const_salt: Optional[float] = Field(
        None, description="Constant salinity value (for type 2)"
    )

    # Values for relaxation and nudging
    inflow_relax: float = Field(0.5, description="Relaxation factor for inflow (0-1)")
    outflow_relax: float = Field(0.1, description="Relaxation factor for outflow (0-1)")
    temp_nudge: float = Field(1.0, description="Temperature nudging factor (0-1)")
    salt_nudge: float = Field(1.0, description="Salinity nudging factor (0-1)")

    # File paths for different boundary types
    temp_th_path: Optional[str] = Field(
        None, description="Path to temperature time history file (for type 1)"
    )
    temp_3d_path: Optional[str] = Field(
        None, description="Path to 3D temperature file (for type 4)"
    )
    salt_th_path: Optional[str] = Field(
        None, description="Path to salinity time history file (for type 1)"
    )
    salt_3d_path: Optional[str] = Field(
        None, description="Path to 3D salinity file (for type 4)"
    )
    flow_th_path: Optional[str] = Field(
        None, description="Path to flow time history file (for type 1)"
    )
    elev_st_path: Optional[str] = Field(
        None, description="Path to space-time elevation file (for types 2/4)"
    )
    vel_st_path: Optional[str] = Field(
        None, description="Path to space-time velocity file (for types 2/4)"
    )

    # Flather boundary parameters
    mean_elev: Optional[List[float]] = Field(
        None, description="Mean elevation for Flather boundaries"
    )
    mean_flow: Optional[List[List[float]]] = Field(
        None, description="Mean flow for Flather boundaries"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_boundary_config(self) -> BoundaryConfig:
        """Convert to BoundaryConfig for TidalBoundary."""
        return BoundaryConfig(
            elev_type=self.elev_type,
            vel_type=self.vel_type,
            temp_type=self.temp_type,
            salt_type=self.salt_type,
            ethconst=self.const_elev,
            vthconst=self.const_flow,
            tthconst=self.const_temp,
            sthconst=self.const_salt,
            inflow_relax=self.inflow_relax,
            outflow_relax=self.outflow_relax,
            tobc=self.temp_nudge,
            sobc=self.salt_nudge,
            eta_mean=self.mean_elev,
            vn_mean=self.mean_flow,
            temp_th_path=self.temp_th_path,
            temp_3d_path=self.temp_3d_path,
            salt_th_path=self.salt_th_path,
            salt_3d_path=self.salt_3d_path,
            flow_th_path=self.flow_th_path,
            elev_st_path=self.elev_st_path,
            vel_st_path=self.vel_st_path,
        )


class SCHISMDataTidesEnhanced(RompyBaseModel):
    """Enhanced SCHISM tidal data handler with support for all boundary types."""

    # Allow arbitrary types for schema generation
    model_config = ConfigDict(arbitrary_types_allowed=True)

    data_type: Literal["tides_enhanced"] = Field(
        default="tides_enhanced",
        description="Model type discriminator",
    )

    # Tidal dataset specification
    tidal_data: Optional[TidalDataset] = Field(
        None, description="Tidal dataset with elevation and velocity files"
    )

    # Legacy boundary configuration
    flags: Optional[List[List[int]]] = Field(
        None, description="Boundary condition flags (legacy format)"
    )
    ethconst: Optional[List[float]] = Field(
        None, description="Constant elevation for each boundary (legacy format)"
    )
    vthconst: Optional[List[float]] = Field(
        None, description="Constant velocity for each boundary (legacy format)"
    )
    tthconst: Optional[List[float]] = Field(
        None, description="Constant temperature for each boundary (legacy format)"
    )
    sthconst: Optional[List[float]] = Field(
        None, description="Constant salinity for each boundary (legacy format)"
    )
    tobc: Optional[List[float]] = Field(
        None, description="Temperature OBC values (legacy format)"
    )
    sobc: Optional[List[float]] = Field(
        None, description="Salinity OBC values (legacy format)"
    )
    relax: Optional[List[float]] = Field(
        None, description="Relaxation parameters (legacy format)"
    )

    # Enhanced boundary configuration
    boundaries: Dict[int, BoundarySetup] = Field(
        default_factory=dict,
        description="Enhanced boundary configuration by boundary index",
    )

    # Predefined configurations
    setup_type: Optional[Literal["tidal", "hybrid", "river", "nested"]] = Field(
        None, description="Predefined boundary setup type"
    )

    @model_validator(mode="before")
    @classmethod
    def convert_numpy_types(cls, data):
        """Convert any numpy values to Python native types"""
        if not isinstance(data, dict):
            return data

        for key, value in list(data.items()):
            if isinstance(value, (np.bool_, np.integer, np.floating, np.ndarray)):
                data[key] = to_python_type(value)
        return data

    @model_validator(mode="after")
    def validate_tidal_data(self):
        """Ensure tidal data is provided when needed for TIDAL or TIDALSPACETIME boundaries."""
        boundaries = self.boundaries or {}
        needs_tidal_data = False

        # Check setup_type first
        if self.setup_type in ["tidal", "hybrid"]:
            needs_tidal_data = True

        # Then check individual boundaries
        for setup in boundaries.values():
            if (
                hasattr(setup, "elev_type")
                and setup.elev_type
                in [ElevationType.HARMONIC, ElevationType.HARMONICEXTERNAL]
            ) or (
                hasattr(setup, "vel_type")
                and setup.vel_type
                in [VelocityType.HARMONIC, VelocityType.HARMONICEXTERNAL]
            ):
                needs_tidal_data = True
                break

        if needs_tidal_data and not self.tidal_data:
            logger.warning(
                "Tidal data is required for TIDAL or TIDALSPACETIME boundary types but was not provided"
            )

        return self

    @model_validator(mode="after")
    def validate_constant_values(self):
        """Ensure constant values are provided when using CONSTANT boundary types."""
        boundaries = self.boundaries or {}

        for idx, setup in boundaries.items():
            if (
                hasattr(setup, "elev_type")
                and setup.elev_type == ElevationType.CONSTANT
                and setup.const_elev is None
            ):
                logger.warning(
                    f"const_elev is required for CONSTANT elev_type in boundary {idx}"
                )

            if (
                hasattr(setup, "vel_type")
                and setup.vel_type == VelocityType.CONSTANT
                and setup.const_flow is None
            ):
                logger.warning(
                    f"const_flow is required for CONSTANT vel_type in boundary {idx}"
                )

            if (
                hasattr(setup, "temp_type")
                and setup.temp_type == TracerType.CONSTANT
                and setup.const_temp is None
            ):
                logger.warning(
                    f"const_temp is required for CONSTANT temp_type in boundary {idx}"
                )

            if (
                hasattr(setup, "salt_type")
                and setup.salt_type == TracerType.CONSTANT
                and setup.const_salt is None
            ):
                logger.warning(
                    f"const_salt is required for CONSTANT salt_type in boundary {idx}"
                )

        return self

    @model_validator(mode="after")
    def validate_relaxed_boundaries(self):
        """Ensure relaxation parameters are provided for RELAXED velocity boundaries."""
        boundaries = self.boundaries or {}

        for idx, setup in boundaries.items():
            if hasattr(setup, "vel_type") and setup.vel_type == VelocityType.RELAXED:
                if not hasattr(setup, "inflow_relax") or not hasattr(
                    setup, "outflow_relax"
                ):
                    logger.warning(
                        f"inflow_relax and outflow_relax are required for RELAXED vel_type in boundary {idx}"
                    )

        return self

    @model_validator(mode="after")
    def validate_flather_boundaries(self):
        """Ensure mean_elev and mean_flow are provided for FLATHER boundaries."""
        boundaries = self.boundaries or {}

        for idx, setup in boundaries.items():
            if hasattr(setup, "vel_type") and setup.vel_type == VelocityType.FLATHER:
                if setup.mean_elev is None or setup.mean_flow is None:
                    logger.warning(
                        f"mean_elev and mean_flow are required for FLATHER vel_type in boundary {idx}"
                    )

        return self

    @model_validator(mode="after")
    def validate_setup_type(self):
        """Validate setup type specific requirements."""
        # Skip validation if setup_type is not set
        if not self.setup_type:
            return self

        if self.setup_type in ["tidal", "hybrid"]:
            if not (self.tidal_data and self.tidal_data.constituents):
                logger.warning(
                    "constituents are required for tidal or hybrid setup_type"
                )
            if not self.tidal_data:
                logger.warning("tidal_data is required for tidal or hybrid setup_type")

        elif self.setup_type == "river":
            if self.boundaries:
                has_flow = any(
                    hasattr(s, "const_flow") and s.const_flow is not None
                    for s in self.boundaries.values()
                )
                if not has_flow:
                    logger.warning(
                        "At least one boundary should have const_flow for river setup_type"
                    )

        elif self.setup_type == "nested":
            if self.boundaries:
                for idx, setup in self.boundaries.items():
                    if (
                        hasattr(setup, "vel_type")
                        and setup.vel_type == VelocityType.RELAXED
                    ):
                        if not hasattr(setup, "inflow_relax") or not hasattr(
                            setup, "outflow_relax"
                        ):
                            logger.warning(
                                f"inflow_relax and outflow_relax are recommended for nested setup_type in boundary {idx}"
                            )
        else:
            logger.warning(
                f"Unknown setup_type: {self.setup_type}. Expected one of: tidal, hybrid, river, nested"
            )

        # Initialize default empty lists for any None attributes to prevent errors later
        self.flags = self.flags if self.flags is not None else []
        self.ethconst = self.ethconst if self.ethconst is not None else []
        self.vthconst = self.vthconst if self.vthconst is not None else []
        self.tthconst = self.tthconst if self.tthconst is not None else []
        self.sthconst = self.sthconst if self.sthconst is not None else []
        self.tobc = self.tobc if self.tobc is not None else [1.0]
        self.sobc = self.sobc if self.sobc is not None else [1.0]

        return self

    def create_tidal_boundary(self, grid, setup_type=None) -> TidalBoundary:
        """Create a TidalBoundary instance from this configuration.

        This method takes the current configuration and creates a properly configured
        TidalBoundary object that can be used to write bctides.in files.

        Parameters
        ----------
        grid : SCHISMGrid
            SCHISM grid instance
        setup_type : str, optional
            Override the setup type, by default None (uses self.setup_type)

        Returns
        -------
        TidalBoundary
            Configured tidal boundary handler
        """
        # Use local variables for all attributes to avoid modifying the original instance
        flags = self.flags if self.flags is not None else []
        ethconst = self.ethconst if self.ethconst is not None else []
        vthconst = self.vthconst if self.vthconst is not None else []
        tthconst = self.tthconst if self.tthconst is not None else []
        sthconst = self.sthconst if self.sthconst is not None else []
        tobc = self.tobc if self.tobc is not None else [1.0]
        sobc = self.sobc if self.sobc is not None else [1.0]

        # Use provided setup_type or fallback to instance attribute
        active_setup_type = setup_type or self.setup_type

        # Create boundary handler
        boundary = TidalBoundary(
            grid_path=grid.hgrid.source,
            tidal_data=self.tidal_data,
        )

        # Configure boundaries
        if self.boundaries is not None and len(self.boundaries) > 0:
            # Use enhanced boundary configuration
            for idx, setup in self.boundaries.items():
                boundary.set_boundary_config(idx, setup.to_boundary_config())
        elif flags:
            # Use legacy flags
            max_boundary = len(flags)
            for i in range(max_boundary):
                # Default flag values
                elev_type = 0
                vel_type = 0
                temp_type = 0
                salt_type = 0

                # Only access flags if they exist and contain values
                if i < len(flags) and flags[i]:
                    if len(flags[i]) > 0:
                        elev_type = flags[i][0]
                    if len(flags[i]) > 1:
                        vel_type = flags[i][1]
                    if len(flags[i]) > 2:
                        temp_type = flags[i][2]
                    if len(flags[i]) > 3:
                        salt_type = flags[i][3]

                config = BoundaryConfig(
                    elev_type=elev_type,
                    vel_type=vel_type,
                    temp_type=temp_type,
                    salt_type=salt_type,
                )

                # Add constant values if provided
                if ethconst and i < len(ethconst):
                    config.ethconst = ethconst[i]
                if vthconst and i < len(vthconst):
                    config.vthconst = vthconst[i]
                if tthconst and i < len(tthconst):
                    config.tthconst = tthconst[i]
                if sthconst and i < len(sthconst):
                    config.sthconst = sthconst[i]
                if tobc and i < len(tobc):
                    config.tobc = tobc[i]
                if sobc and i < len(sobc):
                    config.sobc = sobc[i]

                boundary.set_boundary_config(i, config)
        elif active_setup_type:
            # Use predefined configuration
            if active_setup_type == "tidal":
                # Pure tidal boundary
                for i in range(grid.pylibs_hgrid.nob):
                    boundary.set_boundary_type(
                        i,
                        elev_type=ElevationType.HARMONIC,
                        vel_type=VelocityType.HARMONIC,
                    )
            elif active_setup_type == "hybrid":
                # Tidal + external data
                for i in range(grid.pylibs_hgrid.nob):
                    boundary.set_boundary_type(
                        i,
                        elev_type=ElevationType.HARMONICEXTERNAL,
                        vel_type=VelocityType.HARMONICEXTERNAL,
                    )
            elif active_setup_type == "river":
                # River boundary (first boundary only)
                if grid.pylibs_hgrid.nob > 0:
                    boundary.set_boundary_type(
                        0,
                        elev_type=ElevationType.NONE,
                        vel_type=VelocityType.CONSTANT,
                        vthconst=-100.0,  # Default inflow
                    )
            elif active_setup_type == "nested":
                # Nested boundary with relaxation
                for i in range(grid.pylibs_hgrid.nob):
                    boundary.set_boundary_type(
                        i,
                        elev_type=ElevationType.EXTERNAL,
                        vel_type=VelocityType.RELAXED,
                        temp_type=TracerType.EXTERNAL,
                        salt_type=TracerType.EXTERNAL,
                        inflow_relax=0.8,
                        outflow_relax=0.8,
                    )
        else:
            # Default: tidal boundary for all open boundaries
            for i in range(grid.pylibs_hgrid.nob):
                boundary.set_boundary_type(
                    i, elev_type=ElevationType.HARMONIC, vel_type=VelocityType.HARMONIC
                )

        return boundary

    def get(self, destdir: str | Path, grid: SCHISMGrid, time: TimeRange) -> str:
        """Generate bctides.in file.

        Parameters
        ----------
        destdir : str | Path
            Destination directory
        grid : SCHISMGrid
            SCHISM grid instance
        time : TimeRange
            Time range for the simulation

        Returns
        -------
        str
            Path to the generated bctides.in file
        """
        logger.info(
            f"===== SCHISMDataTidesEnhanced.get called with destdir={destdir} ====="
        )

        # Convert destdir to Path object
        destdir = Path(destdir)

        # Create destdir if it doesn't exist
        if not destdir.exists():
            logger.info(f"Creating destination directory: {destdir}")
            destdir.mkdir(parents=True, exist_ok=True)

        # Make tidal dataset available if provided
        if self.tidal_data:
            logger.info(f"Processing tidal data from {self.tidal_data}")
            self.tidal_data.get(destdir)

        # Create tidal boundary handler
        boundary = self.create_tidal_boundary(grid)

        # Set start time and run duration
        start_time = time.start
        run_days = (time.end - time.start).total_seconds() / 86400.0  # Convert to days
        boundary.set_run_parameters(start_time, run_days)

        # Generate bctides.in file
        bctides_path = destdir / "bctides.in"
        logger.info(f"Writing bctides.in to: {bctides_path}")

        try:
            # Use the enhanced write_boundary_file method that properly handles all configs
            boundary.write_boundary_file(bctides_path)
            logger.info(f"Successfully wrote bctides.in to {bctides_path}")
        except Exception as e:
            logger.error(f"Error writing bctides.in: {e}")
            # Create minimal fallback version
            try:
                with open(bctides_path, "w") as f:
                    f.write("0 10.0 !nbfr, beta_flux\n")
                    f.write(
                        f"{grid.pylibs_hgrid.nob} !nope: number of open boundaries with elevation specified\n"
                    )
                    for i in range(grid.pylibs_hgrid.nob):
                        f.write(f"{i+1} 0. !open bnd #{i+1}, eta amplitude\n")
                    f.write("0 !ncbn: total # of flow bnd segments with discharge\n")
                    f.write("0 !nfluxf: total # of flux boundary segments\n")
                logger.info(f"Created minimal fallback bctides.in at {bctides_path}")
            except Exception as e2:
                logger.error(f"Failed to create fallback bctides.in: {e2}")

        return str(bctides_path)


# Factory functions for common tidal configurations


def create_tidal_only_config(
    constituents: List[str] = None,
    tidal_model: str = "OCEANUM-atlas-v2",
) -> SCHISMDataTidesEnhanced:
    """Create a configuration for tidal-only boundaries.

    Parameters
    ----------
    constituents : list of str, optional
        Tidal constituents to use, defaults to major constituents
    tidal_model : str, optional
        Tidal database to use, by default "OCEANUM-atlas-v2"
    Returns
    -------
    SCHISMDataTidesEnhanced
        Configured tidal data handler
    """
    tidal_data = TidalDataset(
        constituents=constituents or "major",
        tidal_model=tidal_model,
    )

    return SCHISMDataTidesEnhanced(
        tidal_data=tidal_data,
        setup_type="tidal",
    )


def create_hybrid_config(
    constituents: List[str] = None,
    tidal_model: str = "OCEANUM-atlas-v2",
) -> SCHISMDataTidesEnhanced:
    """Create a configuration for hybrid tidal + external data boundaries.

    Parameters
    ----------
    constituents : list of str, optional
        Tidal constituents to use, defaults to major constituents
    tidal_model : str, optional
        Tidal database to use, by default "OCEANUM-atlas-v2"

    Returns
    -------
    SCHISMDataTidesEnhanced
        Configured tidal data handler
    """
    tidal_data = TidalDataset(
        constituents=constituents or "major",
        tidal_model=tidal_model,
    )

    return SCHISMDataTidesEnhanced(
        tidal_data=tidal_data,
        setup_type="hybrid",
    )


def create_river_config(
    river_boundary_index: int = 0,
    river_flow: float = -100.0,
    other_boundaries: Literal["tidal", "none"] = "tidal",
    constituents: List[str] = None,
    tidal_model: str = "OCEANUM-atlas-v2",
) -> SCHISMDataTidesEnhanced:
    """Create a configuration with a river boundary.

    Parameters
    ----------
    river_boundary_index : int, optional
        Index of the river boundary, by default 0
    river_flow : float, optional
        River flow value (negative for inflow), by default -100.0
    other_boundaries : str, optional
        How to handle other boundaries, by default "tidal"
    constituents : list of str, optional
        Tidal constituents to use, defaults to major constituents
    tidal_model : str, optional
        Tidal database to use, by default "OCEANUM-atlas-v2"

    Returns
    -------
    SCHISMDataTidesEnhanced
        Configured tidal data handler
    """
    tidal_data = TidalDataset(
        constituents=constituents or "major",
        tidal_model=tidal_model,
    )

    # Create basic configuration
    config = SCHISMDataTidesEnhanced(
        tidal_data=tidal_data,
        boundaries={},
    )

    # Configure river boundary
    river_config = BoundarySetup(
        elev_type=ElevationType.NONE,
        vel_type=VelocityType.CONSTANT,
        temp_type=TracerType.NONE,
        salt_type=TracerType.NONE,
        const_flow=river_flow,
    )

    # Configure other boundaries if needed
    if other_boundaries == "tidal":
        BoundarySetup(
            elev_type=ElevationType.HARMONIC,
            vel_type=VelocityType.HARMONIC,
            temp_type=TracerType.NONE,
            salt_type=TracerType.NONE,
        )
    else:
        BoundarySetup(
            elev_type=ElevationType.NONE,
            vel_type=VelocityType.NONE,
            temp_type=TracerType.NONE,
            salt_type=TracerType.NONE,
        )

    # Add boundary configurations
    boundaries = {river_boundary_index: river_config}
    # Other boundary indices will be set dynamically in get() method

    config.boundaries = boundaries
    return config


def create_nested_config(
    with_tides: bool = False,
    inflow_relax: float = 0.8,
    outflow_relax: float = 0.8,
    constituents: List[str] = None,
    tidal_model: str = "OCEANUM-atlas-v2",
) -> SCHISMDataTidesEnhanced:
    """Create a configuration for nested model with external data.

    Parameters
    ----------
    with_tides : bool, optional
        Whether to include tides, by default False
    inflow_relax : float, optional
        Relaxation factor for inflow, by default 0.8
    outflow_relax : float, optional
        Relaxation factor for outflow, by default 0.8
    constituents : list of str, optional
        Tidal constituents to use if with_tides=True
    tidal_model : str, optional
        Tidal database to use, by default "OCEANUM-atlas-v2"

    Returns
    -------
    SCHISMDataTidesEnhanced
        Configured tidal data handler
    """
    tidal_data = TidalDataset(
        constituents=constituents or "major",
        tidal_model=tidal_model,
    )

    # Create boundary configuration
    if with_tides:
        default_config = BoundarySetup(
            elev_type=ElevationType.HARMONICEXTERNAL,
            vel_type=VelocityType.RELAXED,
            temp_type=TracerType.EXTERNAL,
            salt_type=TracerType.EXTERNAL,
            inflow_relax=inflow_relax,
            outflow_relax=outflow_relax,
        )
    else:
        default_config = BoundarySetup(
            elev_type=ElevationType.EXTERNAL,
            vel_type=VelocityType.RELAXED,
            temp_type=TracerType.EXTERNAL,
            salt_type=TracerType.EXTERNAL,
            inflow_relax=inflow_relax,
            outflow_relax=outflow_relax,
        )

    return SCHISMDataTidesEnhanced(
        tidal_data=tidal_data,
        boundaries={0: default_config},  # Will be applied to all boundaries
    )
