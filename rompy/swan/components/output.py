"""
SWAN Output Components

This module contains components for configuring output locations and parameters
for SWAN model results, including spatial points, curves, and nested grids.
"""

from abc import ABC
from typing import Annotated, Literal, Optional, Union

from pydantic import Field, field_validator, model_validator

from rompy.logging import get_logger
from rompy.swan.components.base import BaseComponent, MultiComponents
from rompy.swan.subcomponents.base import IJ, XY
from rompy.swan.subcomponents.output import ABS, REL, SPEC1D, SPEC2D
from rompy.swan.subcomponents.readgrid import GRIDREGULAR
from rompy.swan.subcomponents.time import TimeRangeOpen
from rompy.swan.types import IDLA, BlockOptions

logger = get_logger(__name__)


SPECIAL_NAMES = ["BOTTGRID", "COMPGRID", "BOUNDARY", "BOUND_"]


# =====================================================================================
# Locations
# =====================================================================================
class BaseLocation(BaseComponent, ABC):
    """Base class for SWAN output locations.

    .. code-block:: text

        {MODEL_TYPE} sname='sname'

    This is the base class for all locations components. It is not meant to be used
    directly.

    Note
    ----
    The name of the set of output locations `sname` cannot be longer than 8 characters
    and must not match any SWAN special names such as `BOTTGRID` (define output over
    the bottom/current grid) or `COMPGRID` (define output over the computational grid).

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.output import BaseLocation
        loc = BaseLocation(sname="outsites")
        print(loc.render())

    """

    model_type: Literal["locations", "LOCATIONS"] = Field(
        default="locations",
        description="Model type discriminator",
    )
    sname: str = Field(
        description="Name of the set of output locations defined by this command",
        max_length=8,
    )

    @field_validator("sname")
    @classmethod
    def not_special_name(cls, sname: str) -> str:
        """Ensure sname is not defined as one of the special names."""
        for name in SPECIAL_NAMES:
            if sname.upper().startswith(name):
                raise ValueError(f"sname {sname} is a special name and cannot be used")
        return sname

    def cmd(self) -> str:
        """Command file string for this component."""
        return f"{self.model_type.upper()} sname='{self.sname}'"


class FRAME(BaseLocation):
    """Output locations on a regular grid.

    .. code-block:: text

        FRAME 'sname' [xpfr] [ypfr] [alpfr] [xlenfr] [ylenfr] [mxfr] [myfr]

    With this optional command the user defines output on a rectangular, uniform grid
    in a regular frame.

    If the set of output locations is identical to a part of the computational grid,
    then the user can use the alternative command GROUP.

    Note
    ----
    Cannot be used in 1D-mode.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.output import FRAME
        loc = FRAME(
            sname="outgrid",
            grid=dict(xp=173, yp=-40, xlen=2, ylen=2, mx=19, my=19),
        )
        print(loc.render())

    """

    model_type: Literal["frame", "FRAME"] = Field(
        default="frame", description="Model type discriminator"
    )
    grid: GRIDREGULAR = Field(description="Frame grid definition")

    @field_validator("grid")
    @classmethod
    def grid_suffix(cls, grid: GRIDREGULAR) -> GRIDREGULAR:
        grid.suffix = "fr"
        return grid

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = f"{super().cmd()} {self.grid.render()}"
        return repr


class GROUP(BaseLocation):
    """Output locations on subset of a grid.

    .. code-block:: text

        GROUP 'sname' SUBGRID [ix1] [ix2] [iy1] [iy2]

    With this optional command the user defines a group of output locations on a
    rectangular or curvilinear grid that is identical with (part of) the computational
    grid (rectilinear or curvilinear). Such a group may be convenient for the user to
    obtain output that is not affected by interpolation errors.

    The subgrid contains those points (`ix`,`iy`) of the computational grid for which:
    `ix1` <= `ix` <= `ix2` and `iy1` <= `iy` <= `iy2` (The origin of the computational
    grid is `ix=0`, `iy=0`)

    Limitations: `ix1>=0`, `ix2<=mxc`, `iy1>=0`, `iy2<=myc` (`mxc` and `myc` as
    defined in the command `CGRID` which should always precede this command `GROUP`)

    Note
    ----
    Cannot be used in 1D-mode or in case of unstructured grids.

    Note
    ----
    Regular and curvilinear grids are supported.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.output import GROUP
        loc = GROUP(sname="subgrid", ix1=20, iy1=0, ix2=50, iy2=100)
        print(loc.render())

    """

    model_type: Literal["group", "GROUP"] = Field(
        default="group", description="Model type discriminator"
    )
    ix1: int = Field(
        description="Lowest index of the computational grid in the ix-direction",
        ge=0,
    )
    iy1: int = Field(
        description="Lowest index of the computational grid in the iy-direction",
        ge=0,
    )
    ix2: int = Field(
        description="Highest index of the computational grid in the ix-direction",
    )
    iy2: int = Field(
        description="Highest index of the computational grid in the ix-direction",
    )

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = f"{super().cmd()}"
        repr += f" SUBGRID ix1={self.ix1} iy1={self.iy1} ix2={self.ix2} iy2={self.iy2}"
        return repr


class CURVE(BaseLocation):
    """Output locations along a curve.

    .. code-block:: text

        CURVE 'sname' [xp1] [yp1] < [int] [xp] [yp] >

    With this optional command the user defines output along a curved line. Actually
    this curve is a broken line, defined by the user with its corner points. The values
    of the output quantities along the curve are interpolated from the computational
    grid. This command may be used more than once to define more curves.

    Note
    ----
    The following pre-defined curves are available and could be used instead of a CURVE
    component: 'BOUNDARY' and `BOUND_0N` where `N` is boundary part number.

    Note
    ----
    All coordinates and distances should be given in m when Cartesian coordinates are
    used or degrees when Spherical coordinates are used (see command COORD).

    Note
    ----
    Repeat the group `< int xp yp` > in proper order if there are more corner points
    on the curve.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.output import CURVE
        loc = CURVE(
            sname="outcurve",
            xp1=172,
            yp1=-40,
            npts=[3, 3],
            xp=[172.0, 174.0],
            yp=[-38.0, -38.0],
        )
        print(loc.render())

    """

    model_type: Literal["curve", "CURVE"] = Field(
        default="curve", description="Model type discriminator"
    )
    xp1: float = Field(
        description=(
            "Problem coordinate of the first point of the curve in the x-direction"
        ),
    )
    yp1: float = Field(
        description=(
            "Problem coordinate of the first point of the curve in the y-direction"
        ),
    )
    npts: list[int] = Field(
        description=(
            "The `int` CURVE parameter, SWAN will generate `npts-1` equidistant "
            "locations between two subsequent corner points of the curve "
            "including the two corner points"
        ),
        min_length=1,
    )
    xp: list[float] = Field(
        description=(
            "problem coordinates of a corner point of the curve in the x-direction"
        ),
        min_length=1,
    )
    yp: list[float] = Field(
        description=(
            "problem coordinates of a corner point of the curve in the y-direction"
        ),
        min_length=1,
    )

    @model_validator(mode="after")
    def ensure_equal_size(self) -> "CURVE":
        for key in ["xp", "yp"]:
            if len(getattr(self, key)) != len(self.npts):
                raise ValueError(f"Size of npts and {key} must be the same")
        return self

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = f"{super().cmd()} xp1={self.xp1} yp1={self.yp1}"
        for npts, xp, yp in zip(self.npts, self.xp, self.yp):
            repr += f"\nint={npts} xp={xp} yp={yp}"
        return repr


class CURVES(BaseComponent):
    """Output locations along multiple curves.

    .. code-block:: text

        CURVE 'sname1' [xp1] [yp1] < [int] [xp] [yp] >
        CURVE 'sname2' [xp1] [yp1] < [int] [xp] [yp] >
        ..

    This component can be used to prescribe and render multiple CURVE components.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.output import CURVE, CURVES
        loc1 = CURVE(
            sname="c1", xp1=7, yp1=-40, npts=[3, 3], xp=[7, 9], yp=[-38, -38],
        )
        loc2 = CURVE(
            sname="c2", xp1=3, yp1=-37, npts=[5, 5], xp=[4, 5], yp=[-37, -36],
        )
        locs = CURVES(curves=[loc1, loc2])
        print(locs.render())

    """

    model_type: Literal["curves", "CURVES"] = Field(
        default="curves", description="Model type discriminator"
    )
    curves: list[CURVE] = Field(description="CURVE components")

    @property
    def sname(self) -> list[str]:
        return [curve.sname for curve in self.curves]

    def cmd(self) -> list[str]:
        repr = []
        for curve in self.curves:
            repr += [curve.cmd()]
        return repr


class RAY(BaseComponent):
    """Output locations along a depth contour.

    .. code-block:: text

        RAY 'rname' [xp1] [yp1] [xq1] [yq1] < [int] [xp] [yp] [xq] [yq] >

    With this optional command the user provides SWAN with information to determine
    output locations along the depth contour line(s) defined subsequently in command
    `ISOLINE` (see below).

    These locations are determined by SWAN as the intersections of the depth contour
    line(s) and the set of straight rays defined in this command RAY. These rays are
    characterized by a set of master rays defined by their start and end positions
    (`xp`,`yp`) and (`xq`,`yq`). Between each pair of sequential master rays thus
    defined SWAN generates `int-1` intermediate rays by linear interpolation of the
    start and end positions.

    Rays defined by this component have nothing in common with wave rays (e.g. as
    obtained from conventional refraction computations).

    Note
    ----
    Cannot be used in 1D-mode.

    Note
    ----
    All coordinates and distances should be given in m when Cartesian coordinates are
    used or degrees when Spherical coordinates are used (see command `COORD`).

    Note
    ----
    When using rays the input grid for bottom and water level should not be curvilinear.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.output import RAY
        loc = RAY(
            rname="outray",
            xp1=171.9,
            yp1=-40.1,
            xq1=172.1,
            yq1=-39.9,
            npts=[3, 3],
            xp=[171.9, 173.9],
            yp=[-38.1, -38.1],
            xq=[172.1, 174.1],
            yq=[-37.9, -37.9],
        )
        print(loc.render())

    """

    model_type: Literal["ray", "RAY"] = Field(
        default="ray", description="Model type discriminator"
    )
    rname: str = Field(
        description="Name of the set of rays defined by this command",
        max_length=32,
    )
    xp1: float = Field(
        description=(
            "Problem coordinate of the begin point of the first master ray "
            "in the x-direction"
        ),
    )
    yp1: float = Field(
        description=(
            "Problem coordinate of the begin point of the first master ray "
            "in the y-direction"
        ),
    )
    xq1: float = Field(
        description=(
            "Problem coordinate of the end point of the first master ray "
            "in the x-direction"
        ),
    )
    yq1: float = Field(
        description=(
            "Problem coordinate of the end point of the first master ray "
            "in the y-direction"
        ),
    )
    npts: list[int] = Field(
        description=(
            "The `int` RAY parameter, number of subdivisions between the previous "
            "master ray and the following master ray defined by the following data "
            "(number of subdivisions is one morethan the number of interpolated rays)"
        ),
        min_length=1,
    )
    xp: list[float] = Field(
        description=(
            "problem coordinates of the begin of each subsequent master ray in the "
            "x-direction"
        ),
        min_length=1,
    )
    yp: list[float] = Field(
        description=(
            "problem coordinates of the begin of each subsequent master ray in the "
            "y-direction"
        ),
        min_length=1,
    )
    xq: list[float] = Field(
        description=(
            "problem coordinates of the end of each subsequent master ray in the "
            "x-direction"
        ),
        min_length=1,
    )
    yq: list[float] = Field(
        description=(
            "problem coordinates of the end of each subsequent master ray in the "
            "y-direction"
        ),
        min_length=1,
    )

    @model_validator(mode="after")
    def ensure_equal_size(self) -> "CURVE":
        for key in ["xp", "yp", "xq", "yq"]:
            if len(getattr(self, key)) != len(self.npts):
                raise ValueError(f"Size of npts and {key} must be the same")
        return self

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = f"RAY rname='{self.rname}'"
        repr += f" xp1={self.xp1} yp1={self.yp1} xq1={self.xq1} yq1={self.yq1}"
        for npts, xp, yp, xq, yq in zip(self.npts, self.xp, self.yp, self.xq, self.yq):
            repr += f"\nint={npts} xp={xp} yp={yp} xq={xq} yq={yq}"
        return repr


class ISOLINE(BaseLocation):
    """Output locations along a depth contour.

    .. code-block:: text

        ISOLINE 'sname' 'rname' DEPTH|BOTTOM [dep]

    With this optional command the user defines a set of output locations along one
    depth or bottom level contour line (in combination with command RAY).

    Note
    ----
    Cannot be used in 1D-mode.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.output import ISOLINE
        loc = ISOLINE(sname="outcurve", rname="outray", dep_type="depth", dep=12.0)
        print(loc.render())

    """

    model_type: Literal["isoline", "ISOLINE"] = Field(
        default="isoline", description="Model type discriminator"
    )
    rname: str = Field(
        description="Name of the set of rays defined by this command",
        max_length=8,
    )
    dep: float = Field(
        description=(
            "The depth (in m) of the depth contour line along which output locations "
            "are generated by SWAN."
        ),
    )
    dep_type: Optional[Literal["depth", "bottom"]] = Field(
        default=None,
        description=(
            "Define if the depth contour is extracted from the DEPTH output (the "
            "stationary water depth) or from the BOTTOM output (the depth relative "
            "to the datum level with the water level ignored) (SWAN default: DEPTH)"
        ),
    )

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = f"{super().cmd()} rname='{self.rname}'"
        if self.dep_type is not None:
            repr += f" {self.dep_type.upper()}"
        repr += f" dep={self.dep}"
        return repr


class POINTS(BaseLocation):
    """Isolated output locations.

    .. code-block:: text

        POINTS 'sname' < [xp] [yp] >

    With this optional command the user defines a set of individual output point
    locations.

    Note
    ----
    All coordinates and distances should be given in m when Cartesian coordinates are
    used or degrees when Spherical coordinates are used (see command COORD).

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.output import POINTS
        loc = POINTS(sname="outpts", xp=[172.3, 172.4], yp=[-39, -39])
        print(loc.render())

    """

    model_type: Literal["points", "POINTS"] = Field(
        default="points", description="Model type discriminator"
    )
    xp: Optional[list[float]] = Field(
        description="problem coordinates of the points in the x-direction",
        min_length=1,
    )
    yp: Optional[list[float]] = Field(
        description="problem coordinates of the points in the y-direction",
        min_length=1,
    )

    @model_validator(mode="after")
    def ensure_equal_size(self) -> "POINTS":
        if len(self.xp) != len(self.yp):
            raise ValueError(f"xp and yp must be the same size")
        return self

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = f"{super().cmd()}"
        for xp, yp in zip(self.xp, self.yp):
            repr += f"\nxp={xp} yp={yp}"
        return repr


class POINTS_FILE(BaseLocation):
    """Isolated output locations.

    .. code-block:: text

        POINTS 'sname' FILE 'fname'

    With this optional command the user defines a set of individual output point
    locations from text file. The file should have one point per row with x-coordinates
    and y-coordinates in the first and second columns respectively.

    Note
    ----
    All coordinates and distances should be given in m when Cartesian coordinates are
    used or degrees when Spherical coordinates are used (see command COORD).

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.output import POINTS_FILE
        loc = POINTS_FILE(sname="outpts", fname="./output_locations.txt")
        print(loc.render())

    """

    model_type: Literal["points_file", "POINTS_FILE"] = Field(
        default="points_file", description="Model type discriminator"
    )
    fname: str = Field(
        description="Name of the file containing the output locations",
        max_length=36,
    )

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = f"POINTS sname='{self.sname}' fname='{self.fname}'"
        return repr


class NGRID(BaseLocation):
    """Output locations for a nested grid.

    .. code-block:: text

        NGRID 'sname' [xpn] [ypn] [alpn] [xlenn] [ylenn] [mxn] [myn]

    If the user wishes to carry out nested SWAN runs, a separate coarse-grid SWAN run
    is required. With this optional command `NGRID`, the user defines in the present
    coarse-grid run, a set of output locations along the boundary of the subsequent
    nested computational grid. The set of output locations thus defined is of the type
    NGRID.

    Note
    ----
    Command `NESTOUT` is required after this command `NGRID` to generate some data for
    the (subsequent) nested run.

    Note
    ----
    Cannot be used in 1D-mode.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.output import NGRID
        loc = NGRID(
            sname="outnest",
            grid=dict(xp=173, yp=-40, xlen=2, ylen=2, mx=19, my=19),
        )
        print(loc.render())

    """

    model_type: Literal["ngrid", "NGRID"] = Field(
        default="ngrid", description="Model type discriminator"
    )
    grid: GRIDREGULAR = Field(description="NGRID grid definition")

    @field_validator("grid")
    @classmethod
    def grid_suffix(cls, grid: GRIDREGULAR) -> GRIDREGULAR:
        grid.suffix = "n"
        return grid

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = f"{super().cmd()} {self.grid.render()}"
        return repr


class NGRID_UNSTRUCTURED(BaseLocation):
    """Output locations for a nested grid.

    .. code-block:: text

        NGRID 'sname' UNSTRUCTURED ->TRIANGLE|EASYMESH 'fname'

    If the user wishes to carry out nested SWAN runs, a separate coarse-grid SWAN run
    is required. With this optional command `NGRID`, the user defines in the present
    coarse-grid run, a set of output locations along the boundary of the subsequent
    nested computational grid. The set of output locations thus defined is of the type
    NGRID.

    With this option the user indicates that the subsequent nested grid is unstructured
    Only grids generated by Triangle and Easymesh are supported by SWAN.

    Note
    ----
    Command `NESTOUT` is required after this command `NGRID` to generate some data for
    the (subsequent) nested run.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.output import NGRID_UNSTRUCTURED
        loc = NGRID_UNSTRUCTURED(sname="outnest", kind="triangle", fname="ngrid.txt")
        print(loc.render())

    """

    model_type: Literal["ngrid_unstructured", "NGRID_UNSTRUCTURED"] = Field(
        default="ngrid_unstructured", description="Model type discriminator"
    )
    kind: Optional[Literal["triangle", "easymesh"]] = Field(
        default="triangle",
        description=(
            "Indicate if nested grid is generated by Triangle or Easymesh. The base "
            "name of the grid specified in the `fname` parameter is used internally "
            "by SWAN to define the `.node` and `.ele` files in case of the former or "
            "the `.n` and `.e` files in case of the latter."
        ),
    )
    fname: str = Field(
        description="Basename of the required files, i.e. without extension",
        max_length=32,
    )

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = f"NGRID sname='{self.sname}' UNSTRUCTURED"
        if self.kind is not None:
            repr += f" {self.kind.upper()}"
        repr += f" fname='{self.fname}'"
        return repr


# =====================================================================================
# Settings
# =====================================================================================
class QUANTITY(BaseComponent):
    """Define output settings.

    .. code-block:: text

        QUANTITY < output > 'short' 'long' [lexp] [hexp] [excv] [power] [ref] &
            [fswell] [fmin] [fmax] ->PROBLEMCOORD|FRAME

        Examples:
        ---------
        QUANTITY Xp hexp=100.
        QUANTITY HS TM01 RTMM10 excv=-9.
        QUANTITY HS TM02 FSPR fmin=0.03 fmax=0.5
        QUANTITY Hswell fswell=0.08
        QUANTITY Per short='Tm-1,0' power=0.
        QUANTITY Transp Force Frame

    With this command the user can influence:

    * The naming of output quantities
    * The accuracy of writing output quantities
    * The definition of some output quantities
    * Reference direction for vectors

    Note
    ----
    The following data are accepted only in combination with some specific quantities:

    * power
    * ref
    * fswell
    * fmin
    * fmax
    * PROBLEMCOORD
    * FRAME

    Note
    ----
    **PROBLEMCOORD**: Vector components are relative to the x- and y-axes of the
    problem coordinate system (see command `SET`):

    * Directions are counterclockwise relative to the positive x-axis of the problem
      coordinate system if Cartesian direction convention is used.
    * Directions are relative to North (clockwise) if Nautical direction convention is
      used.

    Note
    ----
    **FRAME**: If output is requested on sets created by command FRAME or
    automatically (see command `SET`) (`COMPGRID` or `BOTTGRID`):

    * Vector components are relative to the x- and y-axes of the frame coordinate
      system.
    * Directions are counterclockwise relative to the positive x-axis of the frame
      coordinate system if Cartesian direction convention is used.
    * Directions are relative to North (clockwise) if Nautical direction convention
      is used.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.output import QUANTITY
        quant = QUANTITY(output=["xp"], hexp=100)
        print(quant.render())
        quant = QUANTITY(output=["hsign", "tm01", "rtmm10"], excv=-9)
        print(quant.render())
        quant = QUANTITY(output=["hsign", "tm02", "fspr"], fmin=0.03, fmax=0.5)
        print(quant.render())
        quant = QUANTITY(output=["hsign"], fswell=0.08)
        print(quant.render())
        quant = QUANTITY(output=["per"], short="Tm-1,0", power=0)
        print(quant.render())
        quant = QUANTITY(output=["transp", "force"], coord="frame")
        print(quant.render())

    """

    model_type: Literal["quantity", "QUANTITY"] = Field(
        default="quantity", description="Model type discriminator"
    )
    output: list[BlockOptions] = Field(
        description="The output variables to define settings for",
        min_length=1,
    )
    short: Optional[str] = Field(
        default=None,
        description=(
            "Short name of the output quantity (e.g. the name in the heading of a "
            "table written by SWAN). If this option is not used, SWAN will use a "
            "realistic name"
        ),
        max_length=16,
    )
    long: Optional[str] = Field(
        default=None,
        description=(
            "Long name of the output quantity (e.g. the name in the heading of a "
            "block output written by SWAN). If this option is not used, SWAN will "
            "use a realistic name"
        ),
        max_length=16,
    )
    lexp: Optional[float] = Field(
        default=None,
        description="Lowest expected value of the output quantity",
    )
    hexp: Optional[float] = Field(
        default=None,
        description=(
            "Highest expected value of the output quantity; the highest expected "
            "value is used by SWAN to determine the number of decimals in a table "
            "with heading. So the `QUANTITY` command can be used in case the default "
            "number of decimals in a table is unsatisfactory"
        ),
    )
    excv: Optional[float] = Field(
        default=None,
        description=(
            "In case there is no valid value (e.g. wave height in a dry point) this "
            "exception value is written in a table or block output"
        ),
    )
    power: Optional[float] = Field(
        default=None,
        description=(
            "power `p` appearing in the definition of `PER`, `RPER` and `WLEN`. Note "
            "that the value for `power` given for `PER` affects also the value of "
            "`RPER`; the power for `WLEN` is independent of that of `PER` or `RPER` "
            "(SWAN default: 1)"
        ),
    )
    ref: Optional[str] = Field(
        default=None,
        description=(
            "Reference time used for the quantity `TSEC`. Default value: starting "
            "time of the first computation, except in cases where this is later than "
            "the time of the earliest input. In these cases, the time of the earliest "
            "input is used"
        ),
    )
    fswell: Optional[float] = Field(
        default=None,
        description=(
            "Upper limit of frequency range used for computing the quantity HSWELL "
            "(SWAN default: 0.1 Hz)"
        ),
    )
    noswll: Optional[int] = Field(
        default=None,
        description=("Number of swells to output for watershed quantities "),
    )
    fmin: Optional[float] = Field(
        default=None,
        description=(
            "Lower limit of frequency range used for computing integral parameters "
            "(SWAN Default: 0.0 Hz)"
        ),
    )
    fmax: Optional[float] = Field(
        default=None,
        description=(
            "Upper limit of frequency range used for computing integral parameters "
            "(SWAN default: 1000.0 Hz, acts as infinity)"
        ),
    )
    coord: Optional[Literal["problemcoord", "frame"]] = Field(
        default=None,
        description=(
            "Define if vectors and directions refer to the problem coordinate system "
            "('problemcoord') or sets created by command FRAME ('frame') "
            "(SWAN default: problemcoord)"
        ),
    )

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = "QUANTITY"
        for output in self.output:
            repr += f" {output.upper()}"
        if self.short is not None:
            repr += f" short='{self.short}'"
        if self.long is not None:
            repr += f" long='{self.long}'"
        if self.lexp is not None:
            repr += f" lexp={self.lexp}"
        if self.hexp is not None:
            repr += f" hexp={self.hexp}"
        if self.excv is not None:
            repr += f" excv={self.excv}"
        if self.power is not None:
            repr += f" power={self.power}"
        if self.ref is not None:
            repr += f" ref='{self.ref}'"
        if self.fswell is not None:
            repr += f" fswell={self.fswell}"
        if self.noswll is not None:
            repr += f" noswll={self.noswll}"
        if self.fmin is not None:
            repr += f" fmin={self.fmin}"
        if self.fmax is not None:
            repr += f" fmax={self.fmax}"
        if self.coord is not None:
            repr += f" {self.coord.upper()}"
        return repr


class QUANTITIES(BaseComponent):
    """Define output settings for multiple output.

    .. code-block:: text

        QUANTITY < output > ...
        QUANTITY < output > ...
        ..

    This component can be used to prescribe and render multiple QUANTITY components.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.output import QUANTITY, QUANTITIES
        q1 = QUANTITY(output=["xp"], hexp=100)
        q2 = QUANTITY(output=["hsign", "tm01", "rtmm10"], excv=-9)
        q3 = QUANTITY(output=["hsign", "tm02", "fspr"], fmin=0.03, fmax=0.5)
        q4 = QUANTITY(output=["hsign"], fswell=0.08)
        q5 = QUANTITY(output=["per"], short="Tm-1,0", power=0)
        q6 = QUANTITY(output=["transp", "force"], coord="frame")
        quantities = QUANTITIES(quantities=[q1, q2, q3, q4, q5, q6])
        print(quantities.render())

    """

    model_type: Literal["quantities", "QUANTITIES"] = Field(
        default="quantities", description="Model type discriminator"
    )
    quantities: list[QUANTITY] = Field(description="QUANTITY components")

    def cmd(self) -> list:
        repr = []
        for quantity in self.quantities:
            repr += [quantity.cmd()]
        return repr


class OUTPUT_OPTIONS(BaseComponent):
    """Set format of output.

    .. code-block:: text

        OUTPUT OPTIons 'comment' (TABLE [field]) (BLOCK [ndec] [len]) (SPEC [ndec])

    This command enables the user to influence the format of block, table and spectral
    output.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.output import OUTPUT_OPTIONS
        opts = OUTPUT_OPTIONS(
            comment="!", field=10, ndec_block=4, len=20, ndec_spec=6,
        )
        print(opts.render())

    """

    model_type: Literal["block", "BLOCK"] = Field(
        default="block", description="Model type discriminator"
    )
    comment: Optional[str] = Field(
        default=None,
        description=(
            "Comment character used in comment lines in the output (SWAN default: %)"
        ),
        min_length=1,
        max_length=1,
    )
    field: Optional[int] = Field(
        default=None,
        description="Length of one data field in a table (SWAN default: 12)",
        ge=8,
        le=16,
    )
    ndec_block: Optional[int] = Field(
        default=None,
        description="Number of decimals in block output (SWAN default: 4)",
        ge=0,
        le=9,
    )
    len: Optional[int] = Field(
        default=None,
        description="Number of data on one line of block output (SWAN default: 6)",
        ge=1,
        le=9999,
    )
    ndec_spec: Optional[int] = Field(
        default=None,
        description="Number of decimals in spectra output (SWAN default: 4)",
        ge=0,
        le=9,
    )

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = "OUTPUT OPTIONS"
        if self.comment is not None:
            repr += f" comment='{self.comment}'"
        if self.field is not None:
            repr += f" TABLE field={self.field}"
        if self.ndec_block is not None or self.len is not None:
            repr += f" BLOCK"
            if self.ndec_block is not None:
                repr += f" ndec={self.ndec_block}"
            if self.len is not None:
                repr += f" len={self.len}"
        if self.ndec_spec is not None:
            repr += f" SPEC ndec={self.ndec_spec}"
        return repr


# =====================================================================================
# Write
# =====================================================================================
class BaseWrite(BaseComponent, ABC):
    """Base class for SWAN output writing.

    .. code-block:: text

        {MODEL_TYPE} sname='sname'

    This is the base class for all write components. It is not meant to be used
    directly.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.output import BaseWrite
        write = BaseWrite(
            sname="outgrid",
            fname="./output-grid.nc",
            times=dict(
                tbeg="2012-01-01T00:00:00",
                delt="PT30M",
                tfmt=1,
                dfmt="min",
                suffix="",
            )
        )
        print(write.render())

    """

    model_type: Literal["write", "WRITE"] = Field(
        default="write",
        description="Model type discriminator",
    )
    sname: str = Field(
        description=(
            "Name of the set of output locations in which the output is to be written"
        ),
        max_length=8,
    )
    fname: str = Field(
        description=(
            "Name of the data file where the output is written to The file format is "
            "defined by the file extension, use `.mat` for MATLAB binary (single "
            "precision) or `.nc` for netCDF format. If any other extension is used "
            "the ASCII format is assumed"
        ),
    )
    times: Optional[TimeRangeOpen] = Field(
        default=None,
        description=(
            "Time specification if the user requires output at various times. If this "
            "option is not specified data will be written for the last time step of "
            "the computation"
        ),
    )

    @model_validator(mode="after")
    def validate_special_names(self) -> "BaseWrite":
        special_names = ("COMPGRID", "BOTTGRID")
        snames = self.sname if isinstance(self.sname, list) else [self.sname]
        for sname in snames:
            if sname in special_names and self.model_type.upper() != "BLOCK":
                raise ValueError(f"Special name {sname} is only supported with BLOCK")
        return self

    @model_validator(mode="after")
    def validate_times(self) -> "BaseWrite":
        if self.times is not None:
            self.times.suffix = self.suffix
        return self

    @property
    def suffix(self) -> str:
        return ""

    def cmd(self) -> str:
        """Command file string for this component."""
        return ""


class BLOCK(BaseWrite):
    """Write spatial distributions.

    .. code-block:: text

        BLOCK 'sname' ->HEADER|NOHEADER 'fname' (LAYOUT [idla]) < output > &
            [unit] (OUTPUT [tbegblk] [deltblk]) SEC|MIN|HR|DAY

    With this optional command the user indicates that one or more spatial
    distributions should be written to a file.

    Note
    ----
    The SWAN special frames 'BOTTGRID' or 'COMPGRID' can be set with the `sname` field.

    Note
    ----
    The text of the header indicates run identification (see command `PROJECT`), time,
    frame or group name ('sname'), variable and unit. The number of header lines is 8.

    Note
    ----
    Cannot be used in 1D-mode.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.output import BLOCK
        block = BLOCK(sname="outgrid", fname="./depth-frame.nc", output=["depth"])
        print(block.render())
        block = BLOCK(
            sname="COMPGRID",
            header=False,
            fname="./output-grid.nc",
            idla=3,
            output=["hsign", "hswell", "dir", "tps", "tm01", "watlev", "qp"],
            times=dict(
                tbeg="2012-01-01T00:00:00",
                delt="PT30M",
                tfmt=1,
                dfmt="min",
                suffix="",
            )
        )
        print(block.render())

    """

    model_type: Literal["block", "BLOCK"] = Field(
        default="block", description="Model type discriminator"
    )
    header: Optional[bool] = Field(
        default=None,
        description=(
            "Indicate if the output should be written to a file with header lines "
            "(SWAN default: True)"
        ),
    )
    idla: Optional[IDLA] = Field(
        default=None,
        description=(
            "Prescribe the lay-out of the output to file (supported options here are "
            "1, 3, 4). Option 4 is recommended for postprocessing an ASCII file by "
            "MATLAB, however option 3 is recommended in case of binary MATLAB output "
            "(SWAN default: 1)"
        ),
    )
    output: list[BlockOptions] = Field(
        description="The output variables to output to block file",
        min_length=1,
    )
    unit: Optional[float] = Field(
        default=None,
        description=(
            "Controls the scaling of the output. The program divides computed values "
            "by `unit` before writing to file, so the user should multiply the "
            "written value by `unit` to obtain the proper value. By default, if "
            "HEADER is selected, value is written as a 5 position integer. SWAN takes "
            "`unit` such that the largest number occurring in the block can be "
            "printed. If NOHEADER is selected, values are printed in floating-point "
            "format by default (`unit=1`)"
        ),
    )

    @field_validator("idla")
    @classmethod
    def validate_idla(cls, idla: IDLA) -> IDLA:
        if idla is not None and idla not in (1, 3, 4):
            raise ValueError(
                f"Only IDLA options (1, 3, 4) are supported in BLOCK, got {idla}"
            )
        return idla

    @property
    def suffix(self) -> str:
        return "blk"

    @property
    def _header(self) -> str:
        """Render the header instruction."""
        if self.header:
            return "HEADER"
        else:
            return "NOHEADER"

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = f"BLOCK sname='{self.sname}'"
        if self.header is not None:
            repr += f" {self._header}"
        repr += f" fname='{self.fname}'"
        if self.idla is not None:
            repr += f" LAYOUT idla={self.idla}"
        for output in self.output:
            if len(self.output) > 1:
                repr += "\n"
            else:
                repr += " "
            repr += f"{output.upper()}"
        if self.unit is not None:
            repr += f"\nunit={self.unit}"
        if self.times is not None:
            repr += f"\nOUTPUT {self.times.render()}"
        return repr


class BLOCKS(MultiComponents):
    """Write multiple spatial distributions.

    .. code-block:: text

        BLOCK 'sname' ->HEADER|NOHEADER 'fname1' (LAYOUT [idla]) < output > &
            [unit] (OUTPUT [tbegblk] [deltblk]) SEC|MIN|HR|DAY
        BLOCK 'sname' ->HEADER|NOHEADER 'fname2' (LAYOUT [idla]) < output > &
            [unit] (OUTPUT [tbegblk] [deltblk]) SEC|MIN|HR|DAY
        ..

    This component can be used to prescribe and render multiple BLOCK components.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.output import BLOCK, BLOCKS
        block1 = BLOCK(sname="outgrid", fname="./depth.txt", output=["depth"])
        block2 = BLOCK(sname="outgrid", fname="./output.nc", output=["hsign", "hswell"])
        blocks = BLOCKS(components=[block1, block2])
        print(blocks.render())

    """

    model_type: Literal["blocks"] = Field(
        default="blocks", description="Model type discriminator"
    )
    components: list[BLOCK] = Field(description="BLOCK components")

    @property
    def sname(self) -> list[str]:
        return [component.sname for component in self.components]


class TABLE(BaseWrite):
    """Write spatial distributions.

    .. code-block:: text

        TABLE 'sname' ->HEADER|NOHEADER|INDEXED 'fname'  < output > &
            (OUTPUT [tbegblk] [deltblk]) SEC|MIN|HR|DAY

    With this optional command the user indicates that for each location of the output
    location set 'sname' (see commands `POINTS`, `CURVE`, `FRAME` or `GROUP`) one or
    more variables should be written to a file. The keywords `HEADER` and `NOHEADER`
    determine the appearance of the table; the filename determines the destination of
    the data.

    Note
    ----
    **HEADER**:
    output is written in fixed format to file with headers giving name of variable
    and unit per column (numbers too large to be written will be shown as `****`.
    The number of header lines is 4.

    **NOHEADER**:
    output is written in floating point format to file and has no headers.

    **INDEXED**:
    output compatible with GIS tools such as ARCVIEW, ARCINFO, etc. The user should
    give two TABLE commands, one to produce one file with `XP` and `YP` as output
    quantities, the other with `HS`, `RTM01` or other output quantities.


    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.output import TABLE
        table = TABLE(
            sname="outpts",
            format="noheader",
            fname="./output_table.nc",
            output=["hsign", "hswell", "dir", "tps", "tm01", "watlev", "qp"],
            times=dict(tbeg="2012-01-01T00:00:00", delt="PT30M", dfmt="min"),
        )
        print(table.render())

    """

    model_type: Literal["table", "TABLE"] = Field(
        default="table", description="Model type discriminator"
    )
    format: Optional[Literal["header", "noheader", "indexed"]] = Field(
        default=None,
        description=(
            "Indicate if the table should be written to a file as a HEADER, NOHEADER "
            "or INDEXED table format (SWAN default: HEADER)"
        ),
    )
    output: list[BlockOptions] = Field(
        description="The output variables to output to block file",
        min_length=1,
    )

    @property
    def suffix(self) -> str:
        return "tbl"

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = f"TABLE sname='{self.sname}'"
        if self.format is not None:
            repr += f" {self.format.upper()}"
        repr += f" fname='{self.fname}'"
        for output in self.output:
            if len(self.output) > 1:
                repr += "\n"
            else:
                repr += " "
            repr += f"{output.upper()}"
        if self.times is not None:
            repr += f"\nOUTPUT {self.times.render()}"
        return repr


DIM_TYPE = Annotated[
    Union[SPEC1D, SPEC2D],
    Field(
        description="Choose between 1D or 2D spectra output",
        discriminator="model_type",
    ),
]
FREQ_TYPE = Annotated[
    Union[ABS, REL],
    Field(
        description="Choose between absolute or relative frequency spectra",
        discriminator="model_type",
    ),
]


class SPECOUT(BaseWrite):
    """Write to data file the wave spectra.

    .. code-block:: text

        SPECOUT 'sname' SPEC1D|->SPEC2D ->ABS|REL 'fname' &
            (OUTPUT [tbeg] [delt] SEC|MIN|HR|DAY)

    With this optional command the user indicates that for each location of the output
    location set 'sname' (see commands `POINTS`, `CURVE`, `FRAME` or `GROUP`) the 1D
    or 2D variance / energy (see command `SET`) density spectrum (either the relative
    frequency or the absolute frequency spectrum) is to be written to a data file.

    Note
    ----
    This write command supports the following location types: `POINTS`, `CURVE`,
    `FRAME` and `GROUP`.

    Note
    ----
    This output file can be used for defining boundary conditions for subsequent SWAN
    runs (command `BOUNDSPEC`).

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.output import SPECOUT
        out = SPECOUT(sname="outpts", fname="./specout.swn")
        print(out.render())
        out = SPECOUT(
            sname="outpts",
            dim=dict(model_type="spec2d"),
            freq=dict(model_type="rel"),
            fname="./specout.nc",
            times=dict(tbeg="2012-01-01T00:00:00", delt="PT30M", dfmt="min"),
        )
        print(out.render())

    """

    model_type: Literal["specout", "SPECOUT"] = Field(
        default="specout", description="Model type discriminator"
    )
    dim: Optional[DIM_TYPE] = Field(default=None)
    freq: Optional[FREQ_TYPE] = Field(default=None)

    @property
    def suffix(self) -> str:
        return "spc"

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = f"SPECOUT sname='{self.sname}'"
        if self.dim is not None:
            repr += f" {self.dim.render()}"
        if self.freq is not None:
            repr += f" {self.freq.render()}"
        repr += f" fname='{self.fname}'"
        if self.times is not None:
            repr += f" OUTPUT {self.times.render()}"
        return repr


class NESTOUT(BaseWrite):
    """Write to 2D boundary spectra.

    .. code-block:: text

        NESTOUT 'sname' 'fname' (OUTPUT [tbegnst] [deltnst] ->SEC|MIN|HR|DAY)

    Write to data file two-dimensional action density spectra (relative frequency)
    along the boundary of a nested grid (see command NGRID) to be used in a subsequent
    SWAN run.

    Note
    ----
    Cannot be used in 1D-mode.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.output import NESTOUT
        out = NESTOUT(
            sname="outnest",
            fname="./nestout.swn",
            times=dict(tbeg="2012-01-01T00:00:00", delt="PT30M", dfmt="min"),
        )
        print(out.render())

    """

    model_type: Literal["nestout", "NESTOUT"] = Field(
        default="nestout", description="Model type discriminator"
    )

    @property
    def suffix(self) -> str:
        return "nst"

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = f"NESTOUT sname='{self.sname}' fname='{self.fname}'"
        if self.times is not None:
            repr += f" OUTPUT {self.times.render()}"
        return repr


# =====================================================================================
# Write or plot intermediate results
# =====================================================================================
class TEST(BaseComponent):
    """Write intermediate results.

    .. code-block:: text

        TEST [itest] [itrace] POINTS XY|IJ (PAR 'fname') (S1D 'fname') (S2D 'fname')

    Note
    ----
    The 6 source terms written due to the presence of the keyword S1D or S2D are: wind
    input, whitecapping, bottom friction, breaking, 3- and 4- wave interactions.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.output import TEST
        test = TEST(
            itest=10,
            points=dict(model_type="ij", i=[0, 0], j=[10, 20]),
            fname_par="integral_parameters.test",
            fname_s1d="1d_variance_density.test",
            fname_s2d="2d_variance_density.test",
        )
        print(test.render())
        import numpy as np
        test = TEST(
            points=dict(
                model_type="xy",
                x=np.linspace(172.5, 174.0, 25),
                y=25*[-38],
            ),
            fname_s2d="2d_variance_density.test",
        )
        print(test.render())

    TODO: Support `k` in POINTS IJ.

    """

    model_type: Literal["test", "TEST"] = Field(
        default="test", description="Model type discriminator"
    )
    itest: Optional[int] = Field(
        default=None,
        description=(
            "The level of test output, for values under 100 the amount is usually "
            "reasonable, for values above 200 it can be very large. Values of up to "
            "50 can be interpreted by the user (SWAN default: 1)"
        ),
    )
    itrace: Optional[int] = Field(
        default=None,
        description=(
            "SWAN writes a message (name of subroutine) to the PRINT file at the "
            "first `itrace` entries of each subroutine (SWAN default: 0)"
        ),
    )
    points: Union[XY, IJ] = Field(
        description="Points where detailed print output is produced (max of 50 points)",
        discriminator="model_type",
    )
    fname_par: Optional[str] = Field(
        default=None,
        description="Name of the file where the integral parameters are written to",
    )
    fname_par: Optional[str] = Field(
        default=None,
        description="Name of the file where the integral parameters are written to",
    )
    fname_s1d: Optional[str] = Field(
        default=None,
        description=(
            "Name of the file where the 1D variance density and 6 source terms are "
            "written to"
        ),
    )
    fname_s2d: Optional[str] = Field(
        default=None,
        description=(
            "Name of the file where the 2D variance density and 6 source terms are "
            "written to"
        ),
    )

    @field_validator("points")
    @classmethod
    def validate_points(cls, points: Union[XY, IJ]) -> Union[XY, IJ]:
        if points.size > 50:
            raise ValueError(f"Maximum of 50 points allowed in TEST, got {points.size}")
        return points

    @model_validator(mode="after")
    def at_least_one(self) -> "TEST":
        """Warns if no test file is being specified."""
        if all(v is None for v in [self.fname_par, self.fname_s1d, self.fname_s2d]):
            logger.warning(
                "TEST command prescribed with no output files, please ensure at least "
                "one of ()`fname_par`, `fname_s1d` or `fname_s2d`) is specified"
            )
        return self

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = "TEST"
        if self.itest is not None:
            repr += f" itest={self.itest}"
        if self.itrace is not None:
            repr += f" itrace={self.itrace}"
        repr += f" POINTS {self.points.model_type.upper()}{self.points.render()}"
        if self.fname_par is not None:
            repr += f"PAR fname='{self.fname_par}' "
        if self.fname_s1d is not None:
            repr += f"S1D fname='{self.fname_s1d}' "
        if self.fname_s2d is not None:
            repr += f"S2D fname='{self.fname_s2d}' "
        return repr.rstrip()
