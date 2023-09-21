import logging
from pathlib import Path
from typing import Annotated, Literal, Optional, Union

from pydantic import field_validator, Field, model_validator

from rompy.core import BaseConfig, Coordinate, RompyBaseModel, Spectrum, TimeRange
from rompy.swan.boundary import DataBoundary
from rompy.swan.components import boundary, cgrid, inpgrid, physics, startup

from .data import SwanDataGrid
from .grid import SwanGrid

logger = logging.getLogger(__name__)

HERE = Path(__file__).parent

DEFAULT_TEMPLATE = str(Path(__file__).parent.parent / "templates" / "swan")

PROJECT_TYPES = startup.PROJECT
SET_TYPES = startup.SET
MODE_TYPES = startup.MODE
COORDINATES_TYPES = startup.COORDINATES
CGRID_TYPES = Union[cgrid.REGULAR, cgrid.CURVILINEAR, cgrid.UNSTRUCTURED]
INPGRID_TYPES = inpgrid.INPGRIDS
BOUNDARY_TYPES = Union[
    boundary.BOUNDSPEC, boundary.BOUNDNEST1, boundary.BOUNDNEST2, boundary.BOUNDNEST3
]
INITIAL_TYPES = boundary.INITIAL
PHYSICS_TYPES = physics.PHYSICS


class OutputLocs(RompyBaseModel):
    """Output locations"""

    coords: list[Coordinate] = Field(
        [], description="list of coordinates to output spectra"
    )
    # coords: list[Coordinate] = [["115.61", "-32.618"], ["115.686067", "-32.532381"]]

    def __repr__(self):
        ret = __class__.__name__ + "\n"
        for coord in self.coords:
            ret += f"  {coord.lat} {coord.lon}\n"
        return ret

    def __str__(self):
        ret = ""
        for coord in self.coords:
            ret += f"  {coord.lat} {coord.lon}\n"
        return ret


class ForcingData(RompyBaseModel):
    """SWAN forcing data."""
    bottom: SwanDataGrid | None = Field(None, description="Bathymetry data for SWAN")
    wind: SwanDataGrid | None = Field(None, description="The wind data for SWAN.")
    current: SwanDataGrid | None = Field(None, description="The current data for SWAN.")
    boundary: DataBoundary | None = Field(
        None, description="The boundary data for SWAN."
    )

    def get(self, grid: SwanGrid, period: TimeRange, staging_dir: Path):
        forcing = []
        boundary = []
        for source in self:
            if source[1]:
                logger.info(f"\t Processing {source[0]} forcing")
                source[1]._filter_grid(grid)
                source[1]._filter_time(period)
                if source[0] == "boundary":
                    boundary.append(source[1].get(staging_dir, grid))
                else:
                    forcing.append(source[1].get(staging_dir, grid))
        return dict(forcing="\n".join(forcing), boundary="\n".join(boundary))

    def __str__(self):
        ret = ""
        for forcing in self:
            if forcing[1]:
                ret += f"\t{forcing[0]}: {forcing[1].source}\n"
        return ret


class SwanSpectrum(Spectrum):
    """SWAN Spectrum"""

    @property
    def cmd(self):
        return f"CIRCLE {self.ndirs} {self.fmin} {self.fmax} {self.nfreqs}"


class SwanPhysics(RompyBaseModel):
    """Container class represting configuraable SWAN physics options"""

    friction: str = Field(
        default="MAD",
        description="The type of friction, either MAD, COLL, JON or RIP",
    )
    friction_coeff: float = Field(
        default=0.1,
        description="The coefficient of friction for the given surface and object.",
    )

    @field_validator("friction")
    @classmethod
    def validate_friction(cls, v):
        if v not in ["JON", "COLL", "MAD", "RIP"]:
            raise ValueError(
                "friction must be one of JON, COLL, MAD or RIP"
            )  # TODO Raf to add actual friction options
        return v

    @field_validator("friction_coeff")
    @classmethod
    def validate_friction_coeff(cls, v):
        # TODO Raf to add sensible friction coeff range
        if float(v) > 1:
            raise ValueError("friction_coeff must be less than 1")
        if float(v) < 0:
            raise ValueError("friction_coeff must be greater than 0")
        return v

    @property
    def cmd(self):
        ret = ""
        ret += f"GEN3 WESTH 0.000075 0.00175\n"
        ret += f"BREAKING\n"
        ret += f"FRICTION {self.friction} {self.friction_coeff}\n"
        ret += "\n"
        ret += f"TRIADS\n"
        ret += "\n"
        ret += f"PROP BSBT\n"
        ret += f"NUM ACCUR 0.02 0.02 0.02 95 NONSTAT 20\n"
        return ret


class GridOutput(RompyBaseModel):
    """Gridded outputs for SWAN"""

    period: TimeRange | None = None
    variables: list[str] = [
        "DEPTH",
        "UBOT",
        "HSIGN",
        "HSWELL",
        "DIR",
        "TPS",
        "TM01",
        "WIND",
    ]

    def __str__(self):
        ret = "\tGrid:\n"
        if self.period:
            ret += f"\t\tperiod: {self.period}\n"
        ret += f"\tvariables: {' '.join(self.variables)}\n"
        return ret


class SpecOutput(RompyBaseModel):
    """Spectral outputs for SWAN"""
    period: Optional[TimeRange] = Field(
        None, description="Time range for which the spectral outputs are requested"
    )
    locations: Optional[OutputLocs] = Field(
        OutputLocs(coords=[]),
        description="Output locations for which the spectral outputs are requested",
    )

    def __str__(self):
        ret = "\tSpec\n"
        if self.period:
            ret += f"\t\tperiod: {self.period}\n"
        ret += f"\t\tlocations: {self.locations}\n"
        return ret


class Outputs(RompyBaseModel):
    """Outputs for SWAN"""

    grid: GridOutput = GridOutput()
    spec: SpecOutput = SpecOutput()
    _datefmt: str = "%Y%m%d.%H%M%S"

    @property
    def cmd(self):
        out_intvl = "1.0 HR"  # Hardcoded for now, need to get from time object too TODO
        ret = "OUTPUT OPTIONS BLOCK 8\n"
        ret += f"BLOCK 'COMPGRID' HEADER 'outputs/swan_out.nc' LAYOUT 1 {' '.join(self.grid.variables)} OUT {self.grid.period.start.strftime(self._datefmt)} {out_intvl}\n"
        ret += "\n"
        if len(self.spec.locations.coords) > 0:
            ret += f"POINTs 'pts' FILE 'out.loc'\n"
            ret += f"SPECout 'pts' SPEC2D ABS 'outputs/spec_out.nc' OUTPUT {self.spec.period.start.strftime(self._datefmt)} {out_intvl}\n"
            ret += f"TABle 'pts' HEADer 'outputs/tab_out.nc' TIME XP YP HS TPS TM01 DIR DSPR WIND OUTPUT {self.grid.period.start.strftime(self._datefmt)} {out_intvl}\n"
        return ret

    def __str__(self):
        ret = ""
        ret += f"{self.grid}"
        ret += f"{self.spec}"
        return ret


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
    _datefmt: Annotated[
        str, Field(description="The date format for SWAN.")
    ] = "%Y%m%d.%H%M%S"
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


class SwanConfigComponents(BaseConfig):
    """SWAN config class."""

    model_type: Literal["swancomp"] = Field(
        default="swancomp",
        description="Model type discriminator",
    )
    template: str = Field(
        default=str(Path(__file__).parent.parent / "templates" / "swancomp"),
        description="The template for SWAN.",
    )
    project: Optional[PROJECT_TYPES] = Field(
        default=None,
        description="SWAN PROJECT component",
    )
    set: Optional[SET_TYPES] = Field(
        default=None,
        description="SWAN SET component",
    )
    mode: Optional[MODE_TYPES] = Field(
        default=None,
        description="SWAN MODE component",
    )
    coordinates: Optional[COORDINATES_TYPES] = Field(
        default=None,
        description="SWAN COORDINATES component",
    )
    cgrid: Optional[CGRID_TYPES] = Field(
        default=None,
        description="SWAN CGRID component",
        discriminator="model_type",
    )
    inpgrid: Optional[INPGRID_TYPES] = Field(
        default=None, description="SWAN INPGRID components"
    )
    boundary: Optional[BOUNDARY_TYPES] = Field(
        default=None,
        description="SWAN BOUNDARY component",
        discriminator="model_type",
    )
    initial: Optional[INITIAL_TYPES] = Field(
        default=None,
        description="SWAN INITIAL component",
    )
    physics: Optional[PHYSICS_TYPES] = Field(
        default=None,
        description="SWAN PHYSICS component",
    )

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
