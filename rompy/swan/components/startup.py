"""
SWAN Startup Components

This module contains components for initializing and configuring SWAN model runs,
including project settings, coordinate systems, and run modes.
"""

# Standard library imports
from typing import Literal, Optional

# Third-party imports
from pydantic import field_validator, Field

# Local imports
from rompy.core.logging import get_logger
from rompy.swan.components.base import BaseComponent
from rompy.swan.subcomponents.startup import CARTESIAN, SPHERICAL

# Initialize the logger
logger = get_logger(__name__)


class PROJECT(BaseComponent):
    """SWAN Project.

    .. code-block:: text

        PROJECT 'name' 'nr' 'title' 'title2 'title3'

    With this required command the user defines a number of strings to identify the
    SWAN run (project name e.g., an engineering project) in the print and plot file.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.startup import PROJECT
        proj = PROJECT(nr="01")
        print(proj.render())
        proj = PROJECT(
            name="waus",
            nr="001",
            title1="Western Australia",
            title2="Perth Nest"
        )
        print(proj.render())

    """

    model_type: Literal["project", "PROJECT"] = Field(
        default="project",
        description="Model type discriminator",
    )
    name: Optional[str] = Field(
        default=None,
        description="Is the name of the project, at most 16 characters long",
        max_length=16,
    )
    nr: str = Field(
        description=(
            "Is the run identification (to be provided as a character string; e.g. "
            "the run number) to distinguish this run among other runs for the same "
            "project; it is at most 4 characters long. It is the only required "
            "information in this command."
        ),
        max_length=4,
    )
    title1: Optional[str] = Field(
        default=None,
        description=(
            "A string of at most 72 characters provided by the user to appear in the "
            "output of the program for the user's convenience (SWAN default: blanks)"
        ),
        max_length=72,
    )
    title2: Optional[str] = Field(
        default=None, description="Same as 'title1'", max_length=72
    )
    title3: Optional[str] = Field(
        default=None, description="Same as 'title1'", max_length=72
    )

    def cmd(self) -> str:
        repr = "PROJECT"
        if self.name is not None:
            repr += f" name='{self.name}'"
        repr += f" nr='{self.nr}'"
        if self.title1 is not None:
            repr += f" title1='{self.title1}'"
        if self.title2 is not None:
            repr += f" title2='{self.title2}'"
        if self.title3 is not None:
            repr += f" title3='{self.title3}'"
        return repr


class SET(BaseComponent):
    """SWAN setting commands.

    .. code-block:: text

        SET [level] [nor] [depmin] [maxmes] [maxerr] [grav] [rho] [cdcap] &
            [inrhog] [hsrerr] NAUTICAL|->CARTESIAN [pwtail] [froudmax] [icewind]

    With this optional command the user assigns values to various general parameters.

    Notes
    -----
    The error level `maxerr` is coded as follows:

    * 1: warnings
    * 2: errors (possibly automatically repaired or repairable by SWAN)
    * 3: severe errors

    Default values for `pwtail` depend on formulations of physics:

    * command GEN1: `pwtail = 5`
    * command GEN2: `pwtail = 5`
    * command GEN3 KOMEN: `pwtail = 4`
    * command GEN3 WESTH: `pwtail = 4`
    * command GEN3 JANSSEN: `pwtail = 5`

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.startup import SET
        set = SET(level=0.5, direction_convention="nautical")
        print(set.render())
        set = SET(
            level=-1.0,
            nor=90,
            depmin=0.01,
            maxerr=3,
            grav=9.81,
            rho=1025,
            cdcap=2.5e-3,
            inrhog=0,
            hsrerr=0.1,
            direction_convention="nautical",
        )
        print(set.render())

    """

    model_type: Literal["set", "SET"] = Field(
        default="set", description="Model type discriminator"
    )
    level: Optional[float] = Field(
        default=None,
        description=(
            "Increase in water level that is constant in space and time can be given "
            "with this option, `level` is the value of this increase (in m). For a "
            "variable water level reference is made to the commands "
            "INPGRID and READINP (SWAN default: 0)"
        ),
        examples=[0],
    )
    nor: Optional[float] = Field(
        default=None,
        description=(
            "Direction of North with respect to the x-axis (measured "
            "counterclockwise); default `nor = 90`, i.e. x-axis of the problem "
            "coordinate system points East. When spherical coordinates are used "
            "(see command COORD) the value of `nor` may not be modified"
        ),
        ge=-360.0,
        le=360.0,
    )
    depmin: Optional[float] = Field(
        default=None,
        description=(
            "Threshold depth (in m). In the computation any positive depth smaller "
            "than `depmin` is made equal to `depmin` (SWAN default: 0.05)"
        ),
        ge=0.0,
    )
    maxmes: Optional[int] = Field(
        default=None,
        description=(
            "Maximum number of error messages during the computation at which the "
            "computation is terminated. During the computational process messages are "
            "written to the print file (SWAN default: 200)"
        ),
        ge=0,
    )
    maxerr: Optional[Literal[1, 2, 3]] = Field(
        default=None,
        description=(
            "During pre-processing SWAN checks input data. Depending on the severity "
            "of the errors encountered during this pre-processing, SWAN does not "
            "start computations. The user can influence the error level above which "
            "SWAN will  not start computations (at the level indicated the "
            "computations will continue) (SWAN default: 1)"
        ),
    )
    grav: Optional[float] = Field(
        default=None,
        description="The gravitational acceleration (in m/s2) (SWAN default: 9.81)",
        ge=0.0,
    )
    rho: Optional[float] = Field(
        default=None,
        description="The water density (in kg/m3) (SWAN default: 1025)",
        ge=0.0,
    )
    cdcap: Optional[float] = Field(
        default=None,
        description=(
            "The maximum value for the wind drag coefficient. A value of 99999 means"
            "no cutting off the drag coefficient. A suggestion for this parameter is "
            "`cdcap = 2.5x 10-3` (SWAN default: 99999) "
        ),
        ge=0.0,
    )
    inrhog: Optional[Literal[0, 1]] = Field(
        default=None,
        description=(
            "To indicate whether the user requires output based on variance or based "
            "on true energy (see Section 2.5). `inrhog` = 0: output based on variance, "
            "`inrhog` = 1: output based on true energy (SWAN default: 0)"
        ),
    )
    hsrerr: Optional[float] = Field(
        default=None,
        description=(
            "The relative difference between the user imposed significant wave height "
            "and the significant wave height computed by SWAN (anywhere along the "
            "computational grid boundary) above which a warning will be given. This "
            "relative difference is the difference normalized with the user provided "
            "significant wave height. This warning will be given for each boundary "
            "grid point where the problem occurs (with its x- and y-index number of "
            "the computational grid). The cause of the difference is explained in "
            "Section 2.6.3. To suppress these warnings (in particular for "
            "nonstationary computations), set `hsrerr` at a very high value or use "
            "command OFF BNDCHK (SWAN default: 0.10) (ONLY MEANT FOR STRUCTURED GRIDS)"
        ),
        ge=0.0,
    )
    direction_convention: Literal["nautical", "cartesian"] = Field(
        description=(
            "Direction convention: `nautical` indicates that the Nautical convention "
            "for wind and wave direction (SWAN input and output) will be used, "
            "`cartesian` indicates that the Cartesian convention for wind and wave "
            "direction will be used. For definition, see Section 2.5 or Appendix A "
            "(SWAN default: `cartesian`)"
        ),
    )
    pwtail: Optional[int] = Field(
        default=None,
        description=(
            "Power of high frequency tail; defines the shape of the spectral tail "
            "above the highest prognostic frequency `fhigh` (see command CGRID). "
            "The energy density is assumed to be proportional to frequency to the "
            "power `pwtail`. If the user wishes to use another value, then this SET "
            "command should be located in the command file after GEN1, GEN2 or GEN3 "
            "command (these will override the SET command with respect to `pwtail`)"
        ),
        ge=0,
    )
    froudmax: Optional[float] = Field(
        default=None,
        description=(
            "Is the maximum Froude number (`U/√gd` with `U` the current and `d` the "
            "water depth). The currents taken from a circulation model may mismatch "
            "with given water depth `d` in the sense that the Froude number becomes "
            "larger than 1. For this, the current velocities will be maximized by "
            "Froude number times `sqrt(gh)` (SWAN default: 0.8)"
        ),
        ge=0.0,
    )
    icewind: Optional[Literal[0, 1]] = Field(
        default=None,
        description=(
            "Controls the scaling of wind input by open water fraction. Default value "
            "of zero corresponds to the case where wind input is scaled by the open "
            "water fraction. If `icewind = 1` then sea ice does not affect wind input "
            "directly. (Though there is still indirect effect via the sea ice sink "
            "term; see command SICE) (SWAN default: 0)"
        ),
    )

    @field_validator("pwtail")
    @classmethod
    def pwtail_after_gen(cls, v):
        if v is not None:
            logger.warning("pwtail only has effect if set after GEN command")
        return v

    def cmd(self) -> str:
        repr = "SET"
        if self.level is not None:
            repr += f" level={self.level}"
        if self.nor is not None:
            repr += f" nor={self.nor}"
        if self.depmin is not None:
            repr += f" depmin={self.depmin}"
        if self.maxmes is not None:
            repr += f" maxmes={self.maxmes}"
        if self.maxerr is not None:
            repr += f" maxerr={self.maxerr}"
        if self.grav is not None:
            repr += f" grav={self.grav}"
        if self.rho is not None:
            repr += f" rho={self.rho}"
        if self.cdcap is not None:
            repr += f" cdcap={self.cdcap}"
        if self.inrhog is not None:
            repr += f" inrhog={self.inrhog}"
        if self.hsrerr is not None:
            repr += f" hsrerr={self.hsrerr}"
        if self.direction_convention is not None:
            repr += f" {self.direction_convention.upper()}"
        if self.pwtail is not None:
            repr += f" pwtail={self.pwtail}"
        if self.froudmax is not None:
            repr += f" froudmax={self.froudmax}"
        if self.icewind is not None:
            repr += f" icewind={self.icewind}"
        return repr


class MODE(BaseComponent):
    """SWAN Mode.

    .. code-block:: text

        MODE ->STATIONARY|NONSTATIONARY ->TWODIMENSIONAL|ONEDIMENSIONAL

    With this optional command the user indicates that the run will be either
    stationary or nonstationary and one-dimensional (1D-mode) or two-dimensional
    (2D-mode). Nonstationary means either (see command COMPUTE):

    * (a) one nonstationary computations or
    * (b) a sequence of stationary computations or
    * (c) a mix of (a) and (b)

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.startup import MODE
        mode = MODE()
        print(mode.render())
        mode = MODE(kind="nonstationary", dim="twodimensional")
        print(mode.render())

    """

    model_type: Literal["mode", "MODE"] = Field(
        default="mode", description="Model type discriminator."
    )
    kind: Literal["stationary", "nonstationary"] = Field(
        default="stationary",
        description="Indicates if run will be stationary or nonstationary",
    )
    dim: Literal["onedimensional", "twodimensional"] = Field(
        default="twodimensional",
        description=(
            "Indicates that the run will be either one-dimensional (1D-mode) or "
            "two-dimensional (2D-mode)"
        ),
    )

    def cmd(self) -> str:
        return f"MODE {self.kind.upper()} {self.dim.upper()}"


class COORDINATES(BaseComponent):
    """SWAN Coordinates.

    .. code-block:: text

        COORDINATES ->CARTESIAN|SPHERICAL REPEATING

    Command to choose between Cartesian and spherical coordinates (see Section 2.5).
    A nested SWAN run must use the same coordinate system as the coarse grid SWAN run.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.startup import COORDINATES
        coords = COORDINATES()
        print(coords.render())
        coords = COORDINATES(
            kind=dict(model_type="spherical", projection="ccm"),
            reapeating=True,
        )
        print(coords.render())

    """

    model_type: Literal["coordinates", "COORDINATES"] = Field(
        default="coordinates",
        description="Model type discriminator",
    )
    kind: CARTESIAN | SPHERICAL = Field(
        default_factory=CARTESIAN,
        description="Coordinates kind",
    )
    reapeating: bool = Field(
        default=False,
        description=(
            "This option is only for academic cases. It means that wave energy "
            "leaving at one end of the domain (in computational x-direction) enter at "
            "the other side; it is as if the wave field repeats itself in x-direction "
            "with the length of the domain in x-direction. This option cannot be used "
            "in combination with computation of set-up (see command SETUP). This "
            "option is available only with regular grids"
        ),
    )

    def cmd(self) -> str:
        repr = f"COORDINATES {self.kind.render()}"
        if self.reapeating:
            repr += " REPEATING"
        return repr
