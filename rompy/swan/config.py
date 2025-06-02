"""
SWAN Configuration Module

This module provides configuration classes for the SWAN model within the ROMPY framework.
"""

# Standard library imports
import logging
from pathlib import Path
from typing import Annotated, Literal, Optional, Union

# Third-party imports
from pydantic import Field, model_validator

# Local imports
from rompy.core.config import BaseConfig
from rompy.core.logging import get_logger
from rompy.formatting import get_formatted_box, get_formatted_header_footer

from rompy.swan.interface import (
    DataInterface,
    BoundaryInterface,
    OutputInterface,
    LockupInterface,
)

from rompy.swan.legacy import ForcingData, SwanSpectrum, SwanPhysics, Outputs

from rompy.swan.components import boundary, cgrid, numerics
from rompy.swan.components.group import STARTUP, INPGRIDS, PHYSICS, OUTPUT, LOCKUP

from rompy.swan.grid import SwanGrid


# Initialize the logger
logger = get_logger(__name__)

HERE = Path(__file__).parent

DEFAULT_TEMPLATE = str(Path(__file__).parent.parent / "templates" / "swan")


class SwanConfig(BaseConfig):
    """SWAN configuration"""

    grid: SwanGrid = Field(description="The model grid for the SWAN run")
    model_type: Literal["swan"] = Field("swan", description="The model type for SWAN.")
    spectral_resolution: SwanSpectrum = Field(
        SwanSpectrum(), description="The spectral resolution for SWAN."
    )
    forcing: ForcingData = Field(
        ForcingData(), description="The forcing data for SWAN."
    )
    physics: SwanPhysics = Field(
        SwanPhysics(), description="The physics options for SWAN."
    )
    outputs: Outputs = Field(Outputs(), description="The outputs for SWAN.")
    spectra_file: str = Field("boundary.spec", description="The spectra file for SWAN.")
    template: str = Field(DEFAULT_TEMPLATE, description="The template for SWAN.")
    _datefmt: Annotated[str, Field(description="The date format for SWAN.")] = (
        "%Y%m%d.%H%M%S"
    )
    # subnests: List[SwanConfig] = Field([], description="The subnests for SWAN.") # uncomment if needed

    @property
    def domain(self):
        output = f"CGRID {self.grid.cgrid} {self.spectral_resolution.cmd}\n"
        output += f"{self.grid.cgrid_read}\n"
        return output

    def __call__(self, runtime) -> str:
        # Use formatting utilities imported at the top of the file

        # Log the process beginning
        # Use the log_box utility function
        from rompy.formatting import log_box

        log_box(title="PROCESSING SWAN CONFIGURATION", logger=logger)

        # Setup configuration
        ret = {}
        if not self.outputs.grid.period:
            self.outputs.grid.period = runtime.period
        if not self.outputs.spec.period:
            self.outputs.spec.period = runtime.period
        ret["grid"] = f"{self.domain}"
        ret["forcing"] = self.forcing.get(
            self.grid, runtime.period, runtime.staging_dir
        )
        ret["physics"] = f"{self.physics.cmd}"
        ret["outputs"] = self.outputs.cmd

        # Log completion
        from rompy.formatting import log_box

        log_box(title="SWAN CONFIGURATION PROCESSING COMPLETE", logger=logger)
        ret["output_locs"] = self.outputs.spec.locations
        return ret



STARTUP_TYPE = Annotated[STARTUP, Field(description="Startup components")]
INITIAL_TYPE = Annotated[boundary.INITIAL, Field(description="Initial component")]
PHYSICS_TYPE = Annotated[PHYSICS, Field(description="Physics components")]
PROP_TYPE = Annotated[numerics.PROP, Field(description="Propagation components")]
NUMERIC_TYPE = Annotated[numerics.NUMERIC, Field(description="Numerics components")]
OUTPUT_TYPE = Annotated[OUTPUT, Field(description="Output components")]
LOCKUP_TYPE = Annotated[LOCKUP, Field(description="Output components")]
CGRID_TYPES = Annotated[
    Union[cgrid.REGULAR, cgrid.CURVILINEAR, cgrid.UNSTRUCTURED],
    Field(description="Cgrid component", discriminator="model_type"),
]
INPGRID_TYPES = Annotated[
    Union[INPGRIDS, DataInterface],
    Field(description="Input grid components", discriminator="model_type"),
]
BOUNDARY_TYPES = Annotated[
    Union[
        boundary.BOUNDSPEC,
        boundary.BOUNDNEST1,
        boundary.BOUNDNEST2,
        boundary.BOUNDNEST3,
        BoundaryInterface,
    ],
    Field(description="Boundary component", discriminator="model_type"),
]


logger = logging.getLogger(__name__)


class SwanConfigComponents(BaseConfig):
    """SWAN config class.

    TODO: Combine boundary and inpgrid into a single input type.

    Note
    ----
    The `cgrid` is the only required field since it is used to define the swan grid
    object which is passed to other components.

    """

    model_type: Literal["swanconfig", "SWANCONFIG"] = Field(
        default="swanconfig",
        description="Model type discriminator",
    )
    template: str = Field(
        default=str(HERE.parent / "templates" / "swancomp"),
        description="The template for SWAN.",
    )
    cgrid: CGRID_TYPES
    startup: Optional[STARTUP_TYPE] = Field(default=None)
    inpgrid: Optional[INPGRID_TYPES] = Field(default=None)
    boundary: Optional[BOUNDARY_TYPES] = Field(default=None)
    initial: Optional[INITIAL_TYPE] = Field(default=None)
    physics: Optional[PHYSICS_TYPE] = Field(default=None)
    prop: Optional[PROP_TYPE] = Field(default=None)
    numeric: Optional[NUMERIC_TYPE] = Field(default=None)
    output: Optional[OUTPUT_TYPE] = Field(default=None)
    lockup: Optional[LOCKUP_TYPE] = Field(default=None)

    @model_validator(mode="after")
    def no_nor_if_spherical(self) -> "SwanConfigComponents":
        """Ensure SET nor is not prescribed when using spherical coordinates."""
        return self

    @model_validator(mode="after")
    def no_repeating_if_setup(self) -> "SwanConfigComponents":
        """Ensure COORD repeating not set when using set-up."""
        return self

    @model_validator(mode="after")
    def alp_is_zero_if_spherical(self) -> "SwanConfigComponents":
        """Ensure alp is zero when using spherical coordinates."""
        return self

    @model_validator(mode="after")
    def cgrid_contain_inpgrids(self) -> "SwanConfigComponents":
        """Ensure all inpgrids are inside the cgrid area."""
        return self

    @model_validator(mode="after")
    def layer_defined_if_no_mud_inpgrid(self) -> "SwanConfigComponents":
        """Ensure layer is set in MUD command if not defined with INPGRID MUD."""
        return self

    model_validator(mode="after")

    def transm_msc_mdc(self) -> "SwanConfigComponents":
        """Ensure the number of transmission coefficients match msc and mdc."""
        return self

    @model_validator(mode="after")
    def locations_2d(self) -> "SwanConfigComponents":
        """Ensure Location components not used in 1D mode."""
        # FRAME, GROUP, RAY, ISOLINE and NGRID not in 1D
        # BLOCK and NESTOUT not in 1D
        # GROUP not in unstructured
        return self

    @model_validator(mode="after")
    def group_within_cgrid(self) -> "SwanConfigComponents":
        """Ensure group indices are contained in computational grid."""
        return self

    @model_validator(mode="after")
    def not_curvilinear_if_ray(self) -> "SwanConfigComponents":
        """Ensure bottom and water level grids are not curvilinear for RAY."""
        return self

    @property
    def grid(self):
        """Define a SwanGrid from the cgrid field."""
        return SwanGrid.from_component(self.cgrid.grid)

    def _format_value(self, obj):
        """Custom formatter for SwanConfigComponents values.

        This method provides special formatting for specific types used in
        SwanConfigComponents such as grid, boundary, and output components.

        Args:
            obj: The object to format

        Returns:
            A formatted string or None to use default formatting
        """
        # Import specific types if needed
        from rompy.swan.grid import SwanGrid
        from rompy.core.logging import LoggingConfig

        # Get ASCII mode setting from LoggingConfig
        logging_config = LoggingConfig()
        USE_ASCII_ONLY = logging_config.use_ascii

        # Format SwanConfigComponents (self-formatting)
        if isinstance(obj, SwanConfigComponents):
            header, footer, bullet = get_formatted_header_footer(
                title="SWAN COMPONENTS CONFIGURATION", use_ascii=USE_ASCII_ONLY
            )

            lines = [header]

            # Add computational grid info if available
            if hasattr(obj, "cgrid") and obj.cgrid:
                grid_name = type(obj.cgrid).__name__
                lines.append(f"  {bullet} Computational Grid: {grid_name}")
                # Try to add grid details
                if hasattr(obj.cgrid, "grid"):
                    grid = obj.cgrid.grid
                    if hasattr(grid, "mx") and hasattr(grid, "my"):
                        lines.append(f"      Resolution: {grid.mx}x{grid.my} cells")
                    if hasattr(grid, "xp") and hasattr(grid, "yp"):
                        lines.append(f"      Origin: ({grid.xp}, {grid.yp})")
                    if hasattr(grid, "alp"):
                        lines.append(f"      Rotation: {grid.alp}°")
                    if hasattr(grid, "xlen") and hasattr(grid, "ylen"):
                        lines.append(f"      Size: {grid.xlen}x{grid.ylen}")

            # List all non-None components
            components = {
                "Startup": obj.startup,
                "Input Grid": obj.inpgrid,
                "Boundary": obj.boundary,
                "Initial": obj.initial,
                "Physics": obj.physics,
                "Propagation": obj.prop,
                "Numerics": obj.numeric,
                "Output": obj.output,
                "Lock-up": obj.lockup,
            }

            for name, component in components.items():
                if component is not None:
                    if name == "Input Grid" and isinstance(component, list):
                        lines.append(f"  {bullet} {name}: {len(component)} grid(s)")
                        for i, ingrid in enumerate(component):
                            lines.append(f"      Grid {i+1}: {type(ingrid).__name__}")
                            # Try to add more details for each input grid
                            var_name = getattr(ingrid, "var", "unknown")
                            lines.append(f"          Variable: {var_name}")
                    else:
                        lines.append(f"  {bullet} {name}: {type(component).__name__}")
                        # Add details for physics if available
                        if name == "Physics" and hasattr(component, "gen"):
                            gen_type = type(component.gen).__name__
                            lines.append(f"      Generation: {gen_type}")
                            if hasattr(component, "breaking"):
                                break_type = type(component.breaking).__name__
                                lines.append(f"      Breaking: {break_type}")
                            if hasattr(component, "friction"):
                                fric_type = type(component.friction).__name__
                                lines.append(f"      Friction: {fric_type}")
                        # Add details for output if available
                        if name == "Output" and hasattr(component, "quantity"):
                            if hasattr(component.quantity, "quantities"):
                                qtys = component.quantity.quantities
                                qty_count = len(qtys) if isinstance(qtys, list) else 1
                                lines.append(
                                    f"      Quantities: {qty_count} output group(s)"
                                )
                            if hasattr(component, "block"):
                                lines.append(f"      Block output: Yes")
                            if hasattr(component, "specout"):
                                lines.append(f"      Spectral output: Yes")

            # Add template info if available
            if hasattr(obj, "template"):
                template_path = obj.template
                if len(template_path) > 50:  # Truncate long paths
                    template_path = "..." + template_path[-47:]
                lines.append(f"\n  {bullet} Template: {template_path}")

            lines.append(footer)
            return "\n".join(lines)

        # Format SwanGrid with relevant grid details
        if (
            hasattr(obj, "grid")
            and hasattr(obj, "cgrid")
            and hasattr(obj.cgrid, "grid")
        ):
            grid = obj.cgrid.grid

            header, footer, _ = get_formatted_header_footer(
                title="COMPUTATIONAL GRID", use_ascii=USE_ASCII_ONLY
            )

            return (
                f"{header}\n"
                f"  Type:       {getattr(grid, 'grid_type', 'REG')}\n"
                f"  Resolution: {getattr(grid, 'mx', 0)}x{getattr(grid, 'my', 0)} cells\n"
                f"  Origin:     ({getattr(grid, 'xp', 0.0)}, {getattr(grid, 'yp', 0.0)})\n"
                f"  Rotation:   {getattr(grid, 'alp', 0.0)}°\n"
                f"  Size:       {getattr(grid, 'xlen', 0.0)}x{getattr(grid, 'ylen', 0.0)}\n"
                f"{footer}"
            )

        # Format CGRID component directly
        from rompy.swan.components.cgrid import REGULAR

        if isinstance(obj, REGULAR):
            grid = obj.grid

            header, footer, bullet = get_formatted_header_footer(
                title="GRID CONFIGURATION", use_ascii=USE_ASCII_ONLY
            )

            lines = [header]
            lines.append(f"  {bullet} Type:       {getattr(grid, 'grid_type', 'REG')}")
            lines.append(
                f"  {bullet} Resolution: {getattr(grid, 'mx', 0)}x{getattr(grid, 'my', 0)} cells"
            )
            lines.append(
                f"  {bullet} Origin:     ({getattr(grid, 'xp', 0.0)}, {getattr(grid, 'yp', 0.0)})"
            )
            lines.append(f"  {bullet} Rotation:   {getattr(grid, 'alp', 0.0)}°")
            lines.append(
                f"  {bullet} Size:       {getattr(grid, 'xlen', 0.0)}x{getattr(grid, 'ylen', 0.0)}"
            )

            # Add spectrum details if available
            if hasattr(obj, "spectrum"):
                spectrum = obj.spectrum
                lines.append("")
                lines.append(f"  {bullet} Spectrum:")
                if hasattr(spectrum, "mdc"):
                    lines.append(f"      Direction bins: {spectrum.mdc}")
                if hasattr(spectrum, "flow") and hasattr(spectrum, "fhigh"):
                    lines.append(
                        f"      Frequency range: {spectrum.flow} - {spectrum.fhigh} Hz"
                    )

            lines.append(footer)
            return "\n".join(lines)

        # Format grid directly
        from rompy.swan.grid import SwanGrid

        if isinstance(obj, SwanGrid):
            header, footer, _ = get_formatted_header_footer(
                title="SWAN GRID", use_ascii=USE_ASCII_ONLY
            )

            # Try to get values with fallback to None
            mx = getattr(obj, "mx", None)
            my = getattr(obj, "my", None)
            xp = getattr(obj, "xp", None)
            yp = getattr(obj, "yp", None)
            alp = getattr(obj, "alp", None)

            lines = [header]
            if mx and my:
                lines.append(f"  Resolution: {mx}x{my} cells")
            if xp and yp:
                lines.append(f"  Origin:     ({xp}, {yp})")
            if alp is not None:
                lines.append(f"  Rotation:   {alp}°")
            lines.append(footer)
            return "\n".join(lines)

        # Format boundary components
        if hasattr(obj, "boundary") and obj.boundary is not None:
            count = 1
            if hasattr(obj.boundary, "boundaries"):
                count = len(obj.boundary.boundaries)

            header, footer, _ = get_formatted_header_footer(
                title="BOUNDARY CONDITIONS", use_ascii=USE_ASCII_ONLY
            )

            boundary_type = getattr(obj.boundary, "type", "spectral")
            return (
                f"{header}\n"
                f"  Type:     {boundary_type}\n"
                f"  Segments: {count}\n"
                f"{footer}"
            )

        # Format output components
        if hasattr(obj, "output") and obj.output is not None:
            locations = []
            if hasattr(obj.output, "locations"):
                locations = obj.output.locations

            header, footer, bullet = get_formatted_header_footer(
                title="OUTPUT CONFIGURATION", use_ascii=USE_ASCII_ONLY
            )

            lines = [header]
            lines.append(f"  {bullet} Locations: {len(locations)}")

            if hasattr(obj.output, "format"):
                output_format = getattr(obj.output, "format", "default")
                lines.append(f"  {bullet} Format:    {output_format}")

            if hasattr(obj.output, "variables"):
                variables = getattr(obj.output, "variables", [])
                if variables:
                    lines.append(
                        f"  {bullet} Variables: {', '.join(variables) if len(variables) < 5 else f'{len(variables)} variables'}"
                    )

            # Add detailed output info if available
            if hasattr(obj.output, "quantity"):
                lines.append(f"  {bullet} Output quantities configuration available")

            if hasattr(obj.output, "block"):
                lines.append(f"  {bullet} Block output configuration available")

            if hasattr(obj.output, "specout"):
                lines.append(f"  {bullet} Spectral output configuration available")

            lines.append(footer)
            return "\n".join(lines)

        # Format output component directly
        if hasattr(obj, "model_type") and getattr(obj, "model_type") == "output":
            header, footer, bullet = get_formatted_header_footer(
                title="OUTPUT CONFIGURATION", use_ascii=USE_ASCII_ONLY
            )

            lines = [header]

            # Points output
            if hasattr(obj, "points"):
                points = obj.points
                sname = getattr(points, "sname", "unknown")
                xp = getattr(points, "xp", [])
                yp = getattr(points, "yp", [])
                if isinstance(xp, list) and isinstance(yp, list):
                    num_points = min(len(xp), len(yp))
                else:
                    num_points = 1

                lines.append(f"  {bullet} Output Points: {num_points}")
                lines.append(f"      Name: {sname}")

            # Output quantities
            if hasattr(obj, "quantity"):
                qty = obj.quantity
                if hasattr(qty, "quantities") and isinstance(qty.quantities, list):
                    lines.append(
                        f"  {bullet} Output Quantities: {len(qty.quantities)} groups"
                    )
                    for i, group in enumerate(qty.quantities):
                        if hasattr(group, "output") and len(group.output) > 0:
                            outputs = group.output
                            if len(outputs) < 5:
                                lines.append(f"      Group {i+1}: {', '.join(outputs)}")
                            else:
                                lines.append(
                                    f"      Group {i+1}: {len(outputs)} variables"
                                )

            # Table output
            if hasattr(obj, "table"):
                table = obj.table
                sname = getattr(table, "sname", "unknown")
                fname = getattr(table, "fname", "unknown")
                lines.append(f"  {bullet} Table Output:")
                lines.append(f"      Name: {sname}")
                lines.append(f"      File: {fname}")
                if hasattr(table, "output"):
                    outputs = table.output
                    if len(outputs) < 5:
                        lines.append(f"      Variables: {', '.join(outputs)}")
                    else:
                        lines.append(f"      Variables: {len(outputs)} variables")

            # Block output
            if hasattr(obj, "block"):
                block = obj.block
                sname = getattr(block, "sname", "unknown")
                fname = getattr(block, "fname", "unknown")
                lines.append(f"  {bullet} Block Output:")
                lines.append(f"      Name: {sname}")
                lines.append(f"      File: {fname}")
                if hasattr(block, "output"):
                    outputs = block.output
                    if len(outputs) < 5:
                        lines.append(f"      Variables: {', '.join(outputs)}")
                    else:
                        lines.append(f"      Variables: {len(outputs)} variables")

            # Spectral output
            return None

        # Use the new formatting framework
        from rompy.formatting import format_value

        return format_value(obj)

    def __call__(self, runtime) -> str:
        # Use the new LoggingConfig for logging settings
        from rompy.core.logging import LoggingConfig

        logging_config = LoggingConfig()
        USE_ASCII_ONLY = logging_config.use_ascii
        SIMPLE_LOGS = logging_config.format == "simple"

        # Use the log_box utility function
        from rompy.formatting import log_box

        log_box(title="PROCESSING SWAN CONFIGURATION", logger=logger)

        period = runtime.period
        staging_dir = runtime.staging_dir

        # Log configuration components
        logger.info("Configuration components:")
        if self.cgrid:
            if SIMPLE_LOGS:
                logger.info(f"CGRID: {type(self.cgrid).__name__}")
            else:
                logger.info(f"  - CGRID: {type(self.cgrid).__name__}")
            # Log grid details using _format_value
            grid_str = self._format_value(self.cgrid)
            if grid_str:
                for line in grid_str.split("\n"):
                    if SIMPLE_LOGS:
                        logger.info(f"{line}")
                    else:
                        logger.info(f"    {line}")
        if self.startup:
            if SIMPLE_LOGS:
                logger.info(f"Startup: {type(self.startup).__name__}")
            else:
                logger.info(f"  - Startup: {type(self.startup).__name__}")
            # Log startup details using _format_value
            startup_str = self._format_value(self.startup)
            if startup_str:
                for line in startup_str.split("\n"):
                    if SIMPLE_LOGS:
                        logger.info(f"{line}")
                    else:
                        logger.info(f"    {line}")
        if self.inpgrid:
            if isinstance(self.inpgrid, list):
                if SIMPLE_LOGS:
                    logger.info(f"Input Grids: {len(self.inpgrid)} grid(s)")
                else:
                    logger.info(f"  - Input Grids: {len(self.inpgrid)} grid(s)")
                # Log details for each input grid
                for i, inpgrid in enumerate(self.inpgrid):
                    if SIMPLE_LOGS:
                        logger.info(f"Input Grid {i+1}: {type(inpgrid).__name__}")
                    else:
                        logger.info(f"    Input Grid {i+1}: {type(inpgrid).__name__}")
                    inpgrid_str = self._format_value(inpgrid)
                    if inpgrid_str:
                        for line in inpgrid_str.split("\n"):
                            if SIMPLE_LOGS:
                                logger.info(f"  {line}")
                            else:
                                logger.info(f"      {line}")
            else:
                if SIMPLE_LOGS:
                    logger.info(f"Input Grid: {type(self.inpgrid).__name__}")
                else:
                    logger.info(f"  - Input Grid: {type(self.inpgrid).__name__}")
                # Log input grid details using _format_value
                inpgrid_str = self._format_value(self.inpgrid)
                if inpgrid_str:
                    for line in inpgrid_str.split("\n"):
                        if SIMPLE_LOGS:
                            logger.info(f"  {line}")
                        else:
                            logger.info(f"    {line}")
        if self.boundary:
            if SIMPLE_LOGS:
                logger.info(f"Boundary: {type(self.boundary).__name__}")
            else:
                logger.info(f"  - Boundary: {type(self.boundary).__name__}")
            # Log boundary details using _format_value
            boundary_str = self._format_value(self.boundary)
            if boundary_str:
                for line in boundary_str.split("\n"):
                    if SIMPLE_LOGS:
                        logger.info(f"{line}")
                    else:
                        logger.info(f"    {line}")
        if self.physics:
            if SIMPLE_LOGS:
                logger.info(f"Physics: {type(self.physics).__name__}")
            else:
                logger.info(f"  - Physics: {type(self.physics).__name__}")
            # Log physics details using _format_value
            physics_str = self._format_value(self.physics)
            if physics_str:
                for line in physics_str.split("\n"):
                    if SIMPLE_LOGS:
                        logger.info(f"{line}")
                    else:
                        logger.info(f"    {line}")
        if self.output:
            if SIMPLE_LOGS:
                logger.info(f"Output: {type(self.output).__name__}")
            else:
                logger.info(f"  - Output: {type(self.output).__name__}")
            # Log output details using _format_value
            output_str = self._format_value(self.output)
            if output_str:
                for line in output_str.split("\n"):
                    if SIMPLE_LOGS:
                        logger.info(f"{line}")
                    else:
                        logger.info(f"    {line}")

        # Interface the runtime with components that require times
        if self.output:

            logger.debug("Configuring output interface with period")
            self.output = OutputInterface(group=self.output, period=period).group
        if self.lockup:

            logger.debug("Configuring lockup interface with period")
            self.lockup = LockupInterface(group=self.lockup, period=period).group

        # Render each group component before passing to template
        logger.info("Rendering SWAN configuration components")
        logger.debug("Rendering CGRID configuration")
        ret = {"cgrid": self.cgrid.render()}
        if self.startup:
            logger.debug("Rendering startup configuration")
            ret["startup"] = self.startup.render()
        if self.initial:
            logger.debug("Rendering initial configuration")
            ret["initial"] = self.initial.render()
        if self.physics:
            logger.debug("Rendering physics configuration")
            ret["physics"] = self.physics.render()
        if self.prop:
            logger.debug("Rendering propagation configuration")
            ret["prop"] = self.prop.render()
        if self.numeric:
            logger.debug("Rendering numeric configuration")
            ret["numeric"] = self.numeric.render()
        if self.output:
            logger.debug("Rendering output configuration")
            ret["output"] = self.output.render()
        if self.lockup:
            logger.debug("Rendering lockup configuration")
            ret["lockup"] = self.lockup.render()

        # inpgrid / boundary may use the Interface api so we need passing the args
        if self.inpgrid and isinstance(self.inpgrid, DataInterface):
            logger.debug("Rendering inpgrid configuration with data interface")
            ret["inpgrid"] = self.inpgrid.render(staging_dir, self.grid, period)
        elif self.inpgrid:
            logger.debug("Rendering inpgrid configuration")
            ret["inpgrid"] = self.inpgrid.render()
        if self.boundary and isinstance(self.boundary, BoundaryInterface):
            logger.debug("Rendering boundary configuration with boundary interface")
            ret["boundary"] = self.boundary.render(staging_dir, self.grid, period)
        elif self.boundary:
            logger.debug("Rendering boundary configuration")
            ret["boundary"] = self.boundary.render()

        # Use formatting utilities imported at the top of the file

        # Use the log_box utility function
        from rompy.formatting import log_box

        log_box(title="SWAN CONFIGURATION RENDERING COMPLETE", logger=logger)

        return ret
