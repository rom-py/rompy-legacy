import logging
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, root_validator, validator

from rompy.core import (BaseConfig, Coordinate, RompyBaseModel, Spectrum,
                        TimeRange)
from rompy.swan.boundary import DataBoundary
from rompy.swan.components import base, boundary, cgrid, inpgrid, startup

from .data import SwanDataGrid
from .grid import SwanGrid

logger = logging.getLogger(__name__)

HERE = Path(__file__).parent

COMPONENTS = {
    "project": startup.PROJECT | base.BaseComponent,
    "set": startup.SET | base.BaseComponent,
    "mode": startup.MODE | base.BaseComponent,
    "coordinates": startup.COORDINATES | base.BaseComponent,
    "cgrid": cgrid.REGULAR
    | cgrid.CURVILINEAR
    | cgrid.UNSTRUCTURED
    | base.BaseComponent,
    "inpgrid": list[
        inpgrid.REGULAR
        | inpgrid.CURVILINEAR
        | inpgrid.UNSTRUCTURED
        | base.BaseComponent
    ],
    "boundary": boundary.BOUNDSPEC
    | boundary.BOUNDNEST1
    | boundary.BOUNDNEST2
    | boundary.BOUNDNEST3
    | base.BaseComponent,
    "initial": boundary.INITIAL | base.BaseComponent,
}

DEFAULT_TEMPLATE = str(Path(__file__).parent.parent / "templates" / "swan")


class OutputLocs(RompyBaseModel):
    """Output locations"""

    coords: list[Coordinate] = Field(
        [], description="list of coordinates to output spectra"
    )
    # coords: list[Coordinate] = [["115.61", "-32.618"], ["115.686067", "-32.532381"]]

    def __repr__(self):
        ret = __class__.__name__ + "\n"
        for coord in self.coords:
            ret += f"  {coord[0]} {coord[1]}\n"
        return ret

    def __str__(self):
        ret = ""
        for coord in self.coords:
            ret += f"  {coord[0]} {coord[1]}\n"
        return ret


class ForcingData(RompyBaseModel):
    bottom: SwanDataGrid | None = Field(
        None, description="Bathymetry data for SWAN"
    )  # TODO Raf should probably be required?
    wind: SwanDataGrid | None = Field(None, description="The wind data for SWAN.")
    current: SwanDataGrid | None = Field(None, description="The current data for SWAN.")
    boundary: DataBoundary | None = Field(
        None, description="The boundary data for SWAN."
    )

    def get(self, grid, runtime):
        forcing = []
        boundary = []
        for source in self:
            if source[1]:
                logger.info(f"\t Processing {source[0]} forcing")
                source[1]._filter_grid(grid)
                source[1]._filter_time(runtime.period)
                if source[0] == "boundary":
                    boundary.append(source[1].get(runtime.staging_dir, grid))
                else:
                    forcing.append(source[1].get(runtime.staging_dir, grid))
        return dict(forcing="\n".join(forcing), boundary="\n".join(boundary))

    # below is broken need to move to bnd and forcing to common base object
    # def __str__(self):
    #     ret = ""
    #     for forcing in self:
    #         if forcing[1]:
    #             ret += f"{forcing[0]}:"
    #             if forcing[1].url:
    #                 ret += f" {forcing[1].url}\n"
    #             if forcing[1].path:
    #                 ret += f" {forcing[1].path}\n"
    #             if forcing[1].catalog:
    #                 ret += f" {forcing[1].catalog}"
    #                 ret += f" {forcing[1].dataset}\n"
    #     return ret


class SwanSpectrum(Spectrum):
    """SWAN Spectrum"""

    @property
    def cmd(self):
        return f"CIRCLE {self.ndirs} {self.fmin} {self.fmax} {self.nfreqs}"


class SwanPhysics(RompyBaseModel):
    """Container class represting configuraable SWAN physics options"""

    friction: str = Field(
        default="MAD", description="The type of friction, either 'MAD' or 'TODO'."
    )
    friction_coeff: float = Field(
        default=0.1,
        description="The coefficient of friction for the given surface and object.",
    )

    @validator("friction")
    def validate_friction(cls, v):
        if v not in ["MAD", "OTHER", "ANDANOTHER"]:
            raise ValueError(
                "friction must be one of MAD, OTHER or ANDANOTHER"
            )  # TODO Raf to add actual friction options
        return v

    @validator("friction_coeff")
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
        ret += f"\t\tvariables: {' '.join(self.variables)}\n"
        return ret


class SpecOutput(RompyBaseModel):
    """Spectral outputs for SWAN"""

    period: TimeRange | None = None
    locations: OutputLocs = OutputLocs(coords=[])  # TODO change to None

    def __str__(self):
        ret = "\tSpec\n"
        if self.period:
            ret += f"\t\tperiod: {self.period}\n"
        ret += f"\t\tlocations: {self.locations}\n"
        return ret


class SpecOutput(RompyBaseModel):
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
        if self.spec.locations:
            ret += f"POINTs 'pts' FILE 'out.loc'\n"
        ret += f"SPECout 'pts' SPEC2D ABS 'outputs/spec_out.nc' OUTPUT {self.spec.period.start.strftime(self._datefmt)} {out_intvl}\n"
        ret += f"TABle 'pts' HEADer 'outputs/tab_out.nc' TIME XP YP HS TPS TM01 DIR DSPR WIND OUTPUT {self.grid.period.start.strftime(self._datefmt)} {out_intvl}\n"
        return ret

    def __str__(self):
        ret = "Outputs:\n"
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
    _datefmt: str = Field("%Y%m%d.%H%M%S", description="The date format for SWAN.")
    # subnests: List[SwanConfig] = Field([], description="The subnests for SWAN.") # uncomment if needed

    @property
    def domain(self):
        output = f"CGRID {self.grid.cgrid} {self.spectral_resolution.cmd}\n"
        output += f"{self.grid.cgrid_read}\n"
        return output

    def _get_grid(self, key=None):
        # TODO - Is this functionality needed?
        from intake.source.utils import reverse_format

        grid_spec = self.bottom_grid
        fmt = "{gridtype} {x0:f} {y0:f} {rot:f} {nx:d} {ny:d} {dx:f} {dy:f}"
        if "EXC" in grid_spec:  # append excluded value string
            fmt += " EXC {exc:f}"
        grid_params = reverse_format(fmt, grid_spec)
        return SwanGrid(**grid_params)

    def __call__(self, runtime) -> str:
        ret = {}
        if not self.outputs.grid.period:
            self.outputs.grid.period = runtime.period
        if not self.outputs.spec.period:
            self.outputs.spec.period = runtime.period
        ret["grid"] = f"{self.domain}"
        ret["forcing"] = self.forcing.get(self.grid, runtime)
        ret["physics"] = f"{self.physics.cmd}"
        # TODO raf to complete boundary bit
        # ret["remaining"] = f"BOUND NEST '{self.spectra_file}' CLOSED\n"
        ret["outputs"] = self.outputs.cmd
        ret["output_locs"] = self.outputs.spec.locations
        return ret

    def __str__(self):
        ret = f"grid: {self.grid}\n"
        ret += f"spectral_resolution: {self.spectral_resolution}\n"
        ret += f"forcing: {self.forcing}\n"
        ret += f"physics: {self.physics}\n"
        ret += f"outputs: {self.outputs}\n"
        return ret


class SwanConfigPydantic(BaseConfig):
    """SWAN config class.

    Parameters
    ----------
    model_type: Literal["swan"]
        Model type discriminator.
    cgrid : CGRID
        The computational grid SWAN component.
    inpgrid: INPGRID
        The input grid SWAN component.

    Note
    ----
    - BaseComponent types render empty strings and can be used to skip a certain
      component from rendering to the cmd file.

    TODO: Implement discriminator for inpgrid which is a list of comopnents.

    """

    model_type: Literal["swan"] = "swan"
    project: COMPONENTS.get("project") = Field(..., discriminator="model_type")
    set: COMPONENTS.get("set") = Field(..., discriminator="model_type")
    mode: COMPONENTS.get("mode") = Field(..., discriminator="model_type")
    coordinates: COMPONENTS.get("coordinates") = Field(..., discriminator="model_type")
    cgrid: COMPONENTS.get("cgrid") = Field(..., discriminator="model_type")
    inpgrid: COMPONENTS.get("inpgrid")
    boundary: COMPONENTS.get("boundary") = Field(..., discriminator="model_type")
    initial: COMPONENTS.get("initial") = Field(..., discriminator="model_type")

    @root_validator
    def no_nor_if_spherical(cls, values):
        """Ensure SET nor is not prescribed when using spherical coordinates."""
        return values

    @root_validator
    def no_repeating_if_setup(cls, values):
        """Ensure COORD repeating not set when using set-up."""
        return values

    @root_validator
    def alp_is_zero_if_spherical(cls, values):
        """Ensure alp is zero when using spherical coordinates."""
        return values
