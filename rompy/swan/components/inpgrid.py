"""Input grid for SWAN."""
from typing import Literal, Union, Annotated, Optional
from pathlib import Path
from pydantic import Field, model_validator
from abc import ABC

from rompy.swan.components.base import BaseComponent
from rompy.swan.subcomponents.time import NONSTATIONARY
from rompy.swan.subcomponents.readgrid import READINP
from rompy.swan.types import GridOptions


HERE = Path(__file__).parent


class INPGRID(BaseComponent, ABC):
    """SWAN input grid.

    .. code-block:: text

        INPGRID [grid_type] ->REGULAR|CURVILINEAR|UNSTRUCTURED (EXCEPTION [excval]) &
            (NONSTATIONARY [tbeginp] [deltinp] ->SEC|MIN|HR|DAY [tendinp])

    This is the base class for all input grids. It is not meant to be used directly.

    """

    model_type: Literal["inpgrid", "INPGRID"] = Field(
        default="inpgrid",
        description="Model type discriminator",
    )
    grid_type: GridOptions = Field(
        description="Type of the swan input grid, e.g, 'bottom', 'wind', etc",
    )
    excval: Optional[float] = Field(
        default=None,
        description=(
            "Exception value to allow identifying and ignoring certain point inside "
            "the given grid during the computation. If `fac` != 1, `excval` must be "
            "given as `fac` times the exception value"
        ),
    )
    nonstationary: Optional[NONSTATIONARY] = Field(
        default=None,
        description="Nonstationary time specification",
    )
    readinp: READINP = Field(
        description="SWAN input grid file reader specification",
    )

    @model_validator(mode="after")
    def set_nonstat_suffix(self) -> "INPGRID":
        """Set the nonstationary suffix."""
        if self.nonstationary is not None:
            self.nonstationary.suffix = "inp"
        if self.grid_type is not None:
            self.readinp.grid_type = self.grid_type
        return self

    def cmd(self) -> str:
        return f"INPGRID {self.grid_type.upper()}"


class REGULAR(INPGRID):
    """SWAN regular input grid group component.

    .. code-block:: text

        INPGRID [grid_type] REGULAR [xpinp] [ypinp] [alpinp] [mxinp] [myinp] &
            [dxinp] [dyinp] (EXCEPTION [excval]) &
            (NONSTATIONARY [tbeginp] [deltinp] ->SEC|MIN|HR|DAY [tendinp])
        READGRID [grid_type] [fac] 'fname1' [idla] [nhedf] ([nhedt]) ([nhedvec]) &
            ->FREE|FORMAT|UNFORMATTED ('form'|[idfm])

    This is a group component that includes an `INPGRID` and a `READGRID` component.

    Examples
    --------

    .. ipython:: python
        :okwarning:
        :okexcept:

        from rompy.swan.components.inpgrid import REGULAR
        inpgrid = REGULAR(
            grid_type="bottom",
            excval=-99.0,
            xpinp=172.0,
            ypinp=-41.0,
            alpinp=0.0,
            mxinp=99,
            myinp=99,
            dxinp=0.005,
            dyinp=0.005,
            readinp=dict(fname1="bottom.txt"),
        )
        print(inpgrid.render())
        inpgrid = REGULAR(
            grid_type="wind",
            excval=-99.0,
            xpinp=172.0,
            ypinp=-41.0,
            alpinp=0.0,
            mxinp=99,
            myinp=99,
            dxinp=0.005,
            dyinp=0.005,
            readinp=dict(fname1="wind.txt"),
            nonstationary=dict(
                tbeg="2019-01-01T00:00:00",
                tend="2019-01-07 00:00:00",
                delt=3600,
                deltfmt="hr",
            ),
        )
        print(inpgrid.render())

    """

    model_type: Literal["regular", "REGULAR"] = Field(
        default="regular",
        description="Model type discriminator",
    )
    xpinp: float = Field(
        description=(
            "Geographic location (x-coordinate) of the origin of the input grid in "
            "problem coordinates (in m) if Cartesian coordinates are used or in "
            "degrees if spherical coordinates are used. In case of spherical "
            "coordinates there is no default"
        ),
    )
    ypinp: float = Field(
        description=(
            "Geographic location (y-coordinate) of the origin of the input grid in "
            "problem coordinates (in m) if Cartesian coordinates are used or in "
            "degrees if spherical coordinates are used. In case of spherical "
            "coordinates there is no default"
        ),
    )
    alpinp: Optional[float] = Field(
        default=0.0,
        description=(
            "Direction of the positive x-axis of the input grid "
            "(in degrees, Cartesian convention)"
        ),
    )
    mxinp: int = Field(
        description=(
            "Number of meshes in x-direction of the input grid (this number is one "
            "less than the number of grid points in this direction)"
        ),
    )
    myinp: int = Field(
        description=(
            "Number of meshes in y-direction of the input grid (this number is one "
            "less than the number of grid points in this direction). In 1D-mode, "
            "`myinp` should be 0"
        ),
    )
    dxinp: float = Field(
        description=(
            "Mesh size in x-direction of the input grid, in m in case of Cartesian "
            "coordinates or in degrees if spherical coordinates are used"
        ),
    )
    dyinp: float = Field(
        description=(
            "Mesh size in y-direction of the input grid, in m in case of Cartesian "
            "coordinates or in degrees if spherical coordinates are used. "
            "In 1D-mode, `dyinp` may have any value"
        ),
    )

    def cmd(self) -> list:
        repr = (
            f"{super().cmd()} REGULAR xpinp={self.xpinp} ypinp={self.ypinp} "
            f"alpinp={self.alpinp} mxinp={self.mxinp} myinp={self.myinp} "
            f"dxinp={self.dxinp} dyinp={self.dyinp}"
        )
        if self.excval is not None:
            repr += f" EXCEPTION excval={self.excval}"
        if self.nonstationary is not None:
            repr += f" {self.nonstationary.render()}"
        repr = [repr] + [self.readinp.render()]
        return repr


class CURVILINEAR(INPGRID):
    """SWAN curvilinear input grid group component.

    .. code-block:: text

        INPGRID [grid_type] CURVILINEAR [stagrx] [stagry] [mxinp] [myinp] &
            (EXCEPTION [excval]) &
            (NONSTATIONARY [tbeginp] [deltinp] ->SEC|MIN|HR|DAY [tendinp])
        READGRID [grid_type] [fac] 'fname1' [idla] [nhedf] ([nhedt]) ([nhedvec]) &
            ->FREE|FORMAT|UNFORMATTED ('form'|[idfm])

    This is a group component that includes an `INPGRID` and a `READGRID` component.

    Examples
    --------

    .. ipython:: python
        :okwarning:
        :okexcept:

        from rompy.swan.components.inpgrid import CURVILINEAR
        inpgrid = CURVILINEAR(
            grid_type="wind",
            stagrx=0.0,
            stagry=0.0,
            mxinp=199,
            myinp=199,
            excval=-99.0,
            readinp=dict(fname1="wind.txt"),
            nonstationary=dict(
                tbeg="2019-01-01T00:00:00",
                tend="2019-01-07 00:00:00",
                delt=3600,
                deltfmt="hr",
            ),
        )
        print(inpgrid.render())

    TODO: Handle (or not) setting default values for mxinp and myinp from cgrid.

    """

    model_type: Literal["curvilinear", "CURVILINEAR"] = Field(
        default="curvilinear", description="Model type discriminator"
    )
    stagrx: float = Field(
        default=0.0,
        description=(
            "Staggered x'-direction with respect to computational grid, e.g., "
            "`stagrx=0.5` means that the input grid points are shifted a half step "
            "in x'-direction; in many flow models x-velocities are defined in points "
            "shifted a half step in x'-direction"
        ),
    )
    stagry: float = Field(
        default=0.0,
        description=(
            "Staggered y'-direction with respect to computational grid, e.g., "
            "`stagry=0.5` means that the input grid points are shifted a half step "
            "in y'-direction; in many flow models y-velocities are defined in points "
            "shifted a half step in y'-direction"
        ),
    )
    mxinp: int = Field(
        description=(
            "Number of meshes in ξ-direction of the input grid (this number is one "
            "less than the number of grid points in this direction)"
        ),
    )
    myinp: int = Field(
        description=(
            "Number of meshes in η-direction of the input grid (this number is one "
            "less than the number of grid points in this direction)"
        ),
    )

    def cmd(self) -> str:
        repr = (
            f"{super().cmd()} CURVILINEAR stagrx={self.stagrx} "
            f"stagry={self.stagry} mxinp={self.mxinp} myinp={self.myinp} "
        )
        if self.excval is not None:
            repr += f" EXCEPTION excval={self.excval}"
        if self.nonstationary is not None:
            repr += f" {self.nonstationary.render()}"
        repr = [repr] + [self.readinp.render()]
        return repr


class UNSTRUCTURED(INPGRID):
    """SWAN unstructured input grid group component.

    .. code-block:: text

        INPGRID [grid_type] UNSTRUCTURED EXCEPTION [excval]) &
            (NONSTATIONARY [tbeginp] [deltinp] ->SEC|MIN|HR|DAY [tendinp])
        READGRID [grid_type] [fac] 'fname1' [idla] [nhedf] ([nhedt]) ([nhedvec]) &
            ->FREE|FORMAT|UNFORMATTED ('form'|[idfm])

    This is a group component that includes an `INPGRID` and a `READGRID` component.

    Examples
    --------

    .. ipython:: python
        :okwarning:
        :okexcept:

        from rompy.swan.components.inpgrid import UNSTRUCTURED
        inpgrid = UNSTRUCTURED(
            grid_type="bottom",
            excval=-99.0,
            readinp=dict(fname1="bottom.txt"),
            nonstationary=dict(
                tbeg="2019-01-01T00:00:00",
                tend="2019-01-07 00:00:00",
                delt=3600,
                deltfmt="hr",
            ),
        )
        print(inpgrid.render())

    """

    model_type: Literal["unstructured", "UNSTRUCTURED"] = Field(
        default="unstructured", description="Model type discriminator"
    )

    def cmd(self) -> str:
        repr = f"{super().cmd()} UNSTRUCTURED"
        if self.excval is not None:
            repr += f" EXCEPTION excval={self.excval}"
        if self.nonstationary is not None:
            repr += f" {self.nonstationary.render()}"
        repr = [repr] + [self.readinp.render()]
        return repr


INPGRID_TYPES = Annotated[
    Union[REGULAR, CURVILINEAR, UNSTRUCTURED],
    Field(discriminator="model_type"),
]


class INPGRIDS(BaseComponent):
    """SWAN input grids group component.

    .. code-block:: text

        INPGRID ...
        READGRID ...

        INPGRID ...
        READGRID ...

        ...

    This group component is a convenience to allow defining and rendering
    a list of input grid components.

    Examples
    --------

    .. ipython:: python
        :okwarning:
        :okexcept:

        from rompy.swan.components.inpgrid import REGULAR, INPGRIDS
        inpgrid_bottom = REGULAR(
            grid_type="bottom",
            excval=-99.0,
            xpinp=172.0,
            ypinp=-41.0,
            alpinp=0.0,
            mxinp=99,
            myinp=99,
            dxinp=0.005,
            dyinp=0.005,
            readinp=dict(fname1="bottom.txt"),
        )
        inpgrid_wind = REGULAR(
            grid_type="wind",
            excval=-99.0,
            xpinp=172.0,
            ypinp=-41.0,
            alpinp=0.0,
            mxinp=99,
            myinp=99,
            dxinp=0.005,
            dyinp=0.005,
            readinp=dict(fname1="wind.txt"),
            nonstationary=dict(
                tbeg="2019-01-01T00:00:00",
                tend="2019-01-07 00:00:00",
                delt=3600,
                deltfmt="hr",
            ),
        )
        inpgrids = INPGRIDS(inpgrids=[inpgrid_bottom, inpgrid_wind])
        print(inpgrids.render())

    """

    model_type: Literal["inpgrids"] = Field(
        default="inpgrids", description="Model type discriminator"
    )
    inpgrids: list[INPGRID_TYPES] = Field(
        min_length=1,
        description="List of input grid components",
    )

    def cmd(self) -> str | list:
        repr = []
        for inpgrid in self.inpgrids:
            repr += [inpgrid.cmd()]
        return repr

    def render(self) -> str:
        """Override base class to allow rendering list of components."""
        cmds = []
        for cmd in self.cmd():
            cmds.append(super().render(cmd))
        return "\n\n".join(cmds)
