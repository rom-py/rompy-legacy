import logging
from pathlib import Path
from typing import Annotated, Dict, Literal, Optional, Union

from pydantic import Field, model_validator

from rompy.core.config import BaseConfig, BaseConfigResponse

from rompy.swan.components import boundary, cgrid, numerics
from rompy.swan.components.group import INPGRIDS, LOCKUP, OUTPUT, PHYSICS, STARTUP
from rompy.swan.grid import SwanGrid
from rompy.swan.interface import (
    BoundaryInterface,
    DataInterface,
    LockupInterface,
    OutputInterface,
)
from rompy.swan.legacy import ForcingData, Outputs, SwanPhysics, SwanSpectrum

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
        ret = f"grid: \n\t{self.grid}\n"
        ret += f"spectral_resolution: \n\t{self.spectral_resolution}\n"
        ret += f"forcing: \n{self.forcing}\n"
        ret += f"physics: \n\t{self.physics}\n"
        ret += f"outputs: \n{self.outputs}\n"
        ret += f"template: \n\t{self.template}\n"
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
        response = SwanResponse(
            outputs=dict(
                wave_spectra=self.output.specout.fname,
                wave_parameters=self.output.block.fname,
                wave_hotfiles=self.lockup.hotfile.fname,
            )
        )
        ret["response"] = response

        return ret


class SwanResponse(BaseConfigResponse):
    """Response model for SWAN model outputs.

    Includes paths to wave spectra and wave parameters output files.
    """
    # Class attribute to identify the model type this response is for
    _model_type = "swan"


class SwanConfigResponse(BaseConfigResponse):
    """Response model for SWAN model outputs.

    Includes paths to wave spectra and wave parameters output files.
    """
