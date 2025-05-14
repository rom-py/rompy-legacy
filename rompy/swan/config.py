import logging
from pathlib import Path
from typing import Annotated, Literal, Optional, Union

from pydantic import Field, model_validator

from rompy.core.config import BaseConfig

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


logger = logging.getLogger(__name__)


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
        ret["output_locs"] = self.outputs.spec.locations
        return ret

    def __str__(self):
        """Return a formatted string representation of the SwanConfig.

        This provides a human-readable representation of the configuration
        that can be used in logs and other output.
        """
        # Use helper function to avoid circular imports
        from rompy import ROMPY_ASCII_MODE
        USE_ASCII_ONLY = ROMPY_ASCII_MODE()

        # Format header
        lines = []
        if USE_ASCII_ONLY:
            lines.append("+------------------------------------------------------------------------+")
            lines.append("|                       SWAN MODEL CONFIGURATION                         |")
            lines.append("+------------------------------------------------------------------------+")

            # Format grid info
            lines.append("+------------------------------------------------------------------------+")
            lines.append("| GRID CONFIGURATION                                                     |")
            lines.append("+------------------------------------------------------------------------+")
        else:
            lines.append("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
            lines.append("┃                       SWAN MODEL CONFIGURATION                    ┃")
            lines.append("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")

            # Format grid info
            lines.append("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
            lines.append("┃ GRID CONFIGURATION                                                ┃")
            lines.append("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")

        # Extract grid attributes
        grid_attrs = {}
        for attr_name in dir(self.grid):
            if not attr_name.startswith('_') and not callable(getattr(self.grid, attr_name)):
                grid_attrs[attr_name] = getattr(self.grid, attr_name)

        # Format grid attributes
        for attr, value in grid_attrs.items():
            if isinstance(value, (int, float, str, bool)):
                bullet = "*" if USE_ASCII_ONLY else "•"
                lines.append(f"   {bullet} {attr:<15} : {str(value)}")

        # Format spectral resolution
        lines.append("")
        if USE_ASCII_ONLY:
            lines.append("+------------------------------------------------------------------------+")
            lines.append("| SPECTRAL RESOLUTION                                                   |")
            lines.append("+------------------------------------------------------------------------+")
        else:
            lines.append("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
            lines.append("┃ SPECTRAL RESOLUTION                                               ┃")
            lines.append("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")

        spec_attrs = {}
        for attr_name in dir(self.spectral_resolution):
            if not attr_name.startswith('_') and not callable(getattr(self.spectral_resolution, attr_name)):
                spec_attrs[attr_name] = getattr(self.spectral_resolution, attr_name)

        # Format spectral attributes
        for attr, value in spec_attrs.items():
            if isinstance(value, (int, float, str, bool)):
                bullet = "*" if USE_ASCII_ONLY else "•"
                lines.append(f"   {bullet} {attr:<15} : {str(value)}")

        # Format forcing data
        lines.append("")
        if USE_ASCII_ONLY:
            lines.append("+------------------------------------------------------------------------+")
            lines.append("| FORCING DATA                                                           |")
            lines.append("+------------------------------------------------------------------------+")
        else:
            lines.append("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
            lines.append("┃ FORCING DATA                                                      ┃")
            lines.append("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")

        bullet = "*" if USE_ASCII_ONLY else "•"
        if hasattr(self.forcing, "bottom") and self.forcing.bottom:
            lines.append(f"   {bullet} Bottom       : {getattr(self.forcing.bottom, 'source', 'Enabled')}")
        if hasattr(self.forcing, "current") and self.forcing.current:
            lines.append(f"   {bullet} Current      : {getattr(self.forcing.current, 'source', 'Enabled')}")
        if hasattr(self.forcing, "wind") and self.forcing.wind:
            lines.append(f"   {bullet} Wind         : {getattr(self.forcing.wind, 'source', 'Enabled')}")
        if hasattr(self.forcing, "boundary") and self.forcing.boundary:
            lines.append(f"   {bullet} Boundary     : {getattr(self.forcing.boundary, 'source', 'Enabled')}")

        # Format physics settings
        lines.append("")
        if USE_ASCII_ONLY:
            lines.append("+------------------------------------------------------------------------+")
            lines.append("| PHYSICS SETTINGS                                                       |")
            lines.append("+------------------------------------------------------------------------+")
        else:
            lines.append("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
            lines.append("┃ PHYSICS SETTINGS                                                  ┃")
            lines.append("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")

        phys_attrs = {}
        for attr_name in dir(self.physics):
            if not attr_name.startswith('_') and not callable(getattr(self.physics, attr_name)):
                phys_attrs[attr_name] = getattr(self.physics, attr_name)

        # Format physics attributes
        for attr, value in phys_attrs.items():
            if isinstance(value, (int, float, str, bool)):
                bullet = "*" if USE_ASCII_ONLY else "•"
                lines.append(f"   {bullet} {attr:<15} : {str(value)}")

        # Format outputs
        lines.append("")
        if USE_ASCII_ONLY:
            lines.append("+------------------------------------------------------------------------+")
            lines.append("| OUTPUT CONFIGURATION                                                   |")
            lines.append("+------------------------------------------------------------------------+")
        else:
            lines.append("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
            lines.append("┃ OUTPUT CONFIGURATION                                              ┃")
            lines.append("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")

        bullet = "*" if USE_ASCII_ONLY else "•"
        sub_bullet = "-" if USE_ASCII_ONLY else "•"

        # Add grid output info
        if hasattr(self.outputs, "grid"):
            lines.append("   Grid Outputs:")
            grid_out_attrs = {}
            for attr_name in dir(self.outputs.grid):
                if not attr_name.startswith('_') and not callable(getattr(self.outputs.grid, attr_name)):
                    grid_out_attrs[attr_name] = getattr(self.outputs.grid, attr_name)

            for attr, value in grid_out_attrs.items():
                if isinstance(value, (int, float, str, bool)):
                    lines.append(f"     {sub_bullet} {attr:<13} : {str(value)}")

        # Add spec output info
        if hasattr(self.outputs, "spec"):
            lines.append("   Spectral Outputs:")
            spec_out_attrs = {}
            for attr_name in dir(self.outputs.spec):
                if not attr_name.startswith('_') and not callable(getattr(self.outputs.spec, attr_name)):
                    spec_out_attrs[attr_name] = getattr(self.outputs.spec, attr_name)

            for attr, value in spec_out_attrs.items():
                if isinstance(value, (int, float, str, bool)) and attr != "locations":
                    lines.append(f"     {sub_bullet} {attr:<13} : {str(value)}")

        # Format template information
        lines.append("")
        if USE_ASCII_ONLY:
            lines.append("+------------------------------------------------------------------------+")
            lines.append("| TEMPLATE INFORMATION                                                   |")
            lines.append("+------------------------------------------------------------------------+")
        else:
            lines.append("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
            lines.append("┃ TEMPLATE INFORMATION                                              ┃")
            lines.append("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
        template_path = self.template
        if len(template_path) > 70:
            template_path = "..." + template_path[-67:]
        lines.append(f"   {template_path}")

        return "\n".join(lines)


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

    def __str__(self):
        """Return a formatted string representation of the SwanConfigComponents.

        This provides a human-readable representation that can be used in logs and other output.
        """
        # Use helper function to avoid circular imports
        from rompy import ROMPY_ASCII_MODE
        USE_ASCII_ONLY = ROMPY_ASCII_MODE()

        # Format header
        lines = []
        if USE_ASCII_ONLY:
            lines.append("+------------------------------------------------------------------------+")
            lines.append("|                     SWAN COMPONENTS CONFIGURATION                      |")
            lines.append("+------------------------------------------------------------------------+")

            # Add grid information - formatted as a table
            lines.append("+-----------------------------+-------------------------------------+")
            lines.append(f"| COMPUTATIONAL GRID          | {type(self.cgrid).__name__:<35} |")
            lines.append("+-----------------------------+-------------------------------------+")
        else:
            lines.append("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
            lines.append("┃                    SWAN COMPONENTS CONFIGURATION                  ┃")
            lines.append("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")

            # Add grid information - formatted as a table
            lines.append("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
            lines.append(f"┃ COMPUTATIONAL GRID          ┃ {type(self.cgrid).__name__:<35} ┃")
            lines.append("┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫")

        # Extract grid attributes
        grid = self.cgrid.grid
        grid_attrs = {}
        for attr_name in dir(grid):
            if not attr_name.startswith('_') and not callable(getattr(grid, attr_name)):
                grid_attrs[attr_name] = getattr(grid, attr_name)

        # Format grid attributes as table rows
        for attr, value in grid_attrs.items():
            import ipdb
            if isinstance(value, (int, float, str, bool)):
                if USE_ASCII_ONLY:
                    lines.append(f"| {attr:<27} | {str(value):<35} |")
                else:
                    lines.append(f"┃ {attr:<27} ┃ {str(value):<35} ┃")

        if USE_ASCII_ONLY:
            lines.append("+-----------------------------+-------------------------------------+")
        else:
            lines.append("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")

        # Add information for each component if present - formatted as sections
        components = {
            "STARTUP": self.startup,
            "INPUT GRID": self.inpgrid,
            "BOUNDARY": self.boundary,
            "INITIAL CONDITION": self.initial,
            "PHYSICS": self.physics,
            "PROPAGATION": self.prop,
            "NUMERICS": self.numeric,
            "OUTPUT": self.output,
            "LOCK-UP": self.lockup
        }

        for name, component in components.items():
            if component is not None:
                # Add component header
                lines.append("")
                if USE_ASCII_ONLY:
                    lines.append("+-----------------------------+-------------------------------------+")
                    lines.append(f"| {name:<28} | {type(component).__name__:<35} |")
                    lines.append("+-----------------------------+-------------------------------------+")
                else:
                    lines.append(f"┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
                    lines.append(f"┃ {name:<27} ┃ {type(component).__name__:<35} ┃")
                    lines.append(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")

                # Get component attributes
                comp_attrs = {}
                for attr_name in dir(component):
                    if not attr_name.startswith('_') and not callable(getattr(component, attr_name)):
                        attr_value = getattr(component, attr_name)
                        if not isinstance(attr_value, (dict, list)) or attr_name == "model_type":
                            comp_attrs[attr_name] = attr_value

                # Format component attributes
                if comp_attrs:
                    max_attr_len = max(len(attr) for attr in comp_attrs.keys())
                    for attr, value in comp_attrs.items():
                        if isinstance(value, (int, float, str, bool)):
                            if attr != "model_type":  # Skip model_type as it's in the header
                                bullet = "*" if USE_ASCII_ONLY else "•"
                                lines.append(f"   {bullet} {attr:<{max_attr_len}} : {str(value)}")

        # Add template info
        lines.append("")
        if USE_ASCII_ONLY:
            lines.append("+------------------------------------------------------------------------+")
            lines.append("| TEMPLATE INFORMATION                                                  |")
            lines.append("+------------------------------------------------------------------------+")
        else:
            lines.append("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
            lines.append("┃ TEMPLATE INFORMATION                                              ┃")
            lines.append("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")

        template_path = self.template
        if len(template_path) > 70:
            template_path = "..." + template_path[-67:]
        lines.append(f"   {template_path}")

        return "\n".join(lines)

    def __call__(self, runtime) -> str:
        period = runtime.period
        staging_dir = runtime.staging_dir

        # Interface the runtime with components that require times
        if self.output:
            self.output = OutputInterface(group=self.output, period=period).group
        if self.lockup:
            self.lockup = LockupInterface(group=self.lockup, period=period).group

        # Render each group component before passing to template
        ret = {"cgrid": self.cgrid.render()}
        if self.startup:
            ret["startup"] = self.startup.render()
        if self.initial:
            ret["initial"] = self.initial.render()
        if self.physics:
            ret["physics"] = self.physics.render()
        if self.prop:
            ret["prop"] = self.prop.render()
        if self.numeric:
            ret["numeric"] = self.numeric.render()
        if self.output:
            ret["output"] = self.output.render()
        if self.lockup:
            ret["lockup"] = self.lockup.render()

        # inpgrid / boundary may use the Interface api so we need passing the args
        if self.inpgrid and isinstance(self.inpgrid, DataInterface):
            ret["inpgrid"] = self.inpgrid.render(staging_dir, self.grid, period)
        elif self.inpgrid:
            ret["inpgrid"] = self.inpgrid.render()
        if self.boundary and isinstance(self.boundary, BoundaryInterface):
            ret["boundary"] = self.boundary.render(staging_dir, self.grid, period)
        elif self.boundary:
            ret["boundary"] = self.boundary.render()

        return ret
