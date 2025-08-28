"""
SWAN Legacy Module

This module contains legacy components for backward compatibility with older versions
of the SWAN model configuration in the ROMPY framework.
"""

from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator

from rompy.core.time import TimeRange
from rompy.core.types import Coordinate, RompyBaseModel, Spectrum
from rompy.logging import get_logger
from rompy.swan.boundary import Boundnest1
from rompy.swan.data import SwanDataGrid
from rompy.swan.grid import SwanGrid

logger = get_logger(__name__)


class ForcingData(RompyBaseModel):
    """SWAN forcing data.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.forcing import ForcingData

    """

    bottom: Optional[SwanDataGrid] = Field(default=None, description="Bathymetry data")
    wind: Optional[SwanDataGrid] = Field(default=None, description="Wind input data")
    current: Optional[SwanDataGrid] = Field(
        default=None, description="Current input data"
    )
    boundary: Optional[Boundnest1] = Field(
        default=None, description="Boundary input data"
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
        ret += "GEN3 WESTH 0.000075 0.00175\n"
        ret += "BREAKING\n"
        ret += f"FRICTION {self.friction} {self.friction_coeff}\n"
        ret += "\n"
        ret += "TRIADS\n"
        ret += "\n"
        ret += "PROP BSBT\n"
        ret += "NUM ACCUR 0.02 0.02 0.02 95 NONSTAT 20\n"
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
            ret += "POINTs 'pts' FILE 'out.loc'\n"
            ret += f"SPECout 'pts' SPEC2D ABS 'outputs/spec_out.nc' OUTPUT {self.spec.period.start.strftime(self._datefmt)} {out_intvl}\n"
            ret += f"TABle 'pts' HEADer 'outputs/tab_out.nc' TIME XP YP HS TPS TM01 DIR DSPR WIND OUTPUT {self.grid.period.start.strftime(self._datefmt)} {out_intvl}\n"
        return ret

    def __str__(self):
        ret = ""
        ret += f"{self.grid}"
        ret += f"{self.spec}"
        return ret
