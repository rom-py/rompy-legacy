"""
SWAN Boundary Subcomponents

This module contains subcomponents for defining boundary conditions in SWAN,
including boundary segments, spectral parameters, and initial conditions.
"""

from typing import Annotated, Literal, Optional, Union

from pydantic import Field, model_validator

from rompy.core.logging import get_logger
from rompy.swan.subcomponents.base import IJ, XY, BaseSubComponent

logger = get_logger(__name__)


class SIDE(BaseSubComponent):
    """Boundary over one side of computational domain.

    .. code-block:: text

        SIDE NORTH|NW|WEST|SW|SOUTH|SE|E|NE CCW|CLOCKWISE

    The boundary is one full side of the computational grid (in 1D cases either of the
    two ends of the 1D-grid).

    Note
    ----
    Should not be used in case of CURVILINEAR grids.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.boundary import SIDE
        side = SIDE(side="west", direction="ccw")
        print(side.render())

    """

    model_type: Literal["side", "SIDE"] = Field(
        default="side",
        description="Model type discriminator",
    )
    side: Literal["north", "nw", "west", "sw", "south", "se", "east", "ne"] = Field(
        description="The side of the grid to apply the boundary to",
    )
    direction: Literal["ccw", "clockwise"] = Field(
        default="ccw",
        description="The direction to apply the boundary in",
    )

    def cmd(self) -> str:
        repr = f"SIDE {self.side.upper()} {self.direction.upper()} "
        return repr


class SIDES(BaseSubComponent):
    """Boundary over multiple side of computational domain.

    .. code-block:: text

        SIDE NORTH|NW|WEST|SW|SOUTH|SE|E|NE CCW|CLOCKWISE
        SIDE NORTH|NW|WEST|SW|SOUTH|SE|E|NE CCW|CLOCKWISE
        ...

    Note
    ----
    Should not be used in case of CURVILINEAR grids.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.boundary import SIDE, SIDES
        side1 = SIDE(side="west", direction="ccw")
        side2 = SIDE(side="north", direction="ccw")
        sides = SIDES(sides=[side1, side2])
        print(sides.render())

    """

    model_type: Literal["sides", "SIDES"] = Field(
        default="sides",
        description="Model type discriminator",
    )
    sides: list[SIDE] = Field(
        description="The sides of the grid to apply the boundary to",
    )

    def cmd(self) -> str:
        repr = []
        for side in self.sides:
            repr += f"SIDE {side.side.upper()} {side.direction.upper()} "
        return repr


class SEGMENT(BaseSubComponent):
    """Boundary over a segment defined from points.

    .. code-block:: text

        SEGMENT XY < [x] [y] >
        SEGMENT IJ < [i] [j] >

    The segment is defined either by means of a series of points in terms of problem
    coordinates (`XY`) or by means of a series of points in terms of grid indices
    (`IJ`). The points do not have to include all or coincide with actual grid points.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.boundary import SEGMENT
        seg = SEGMENT(
            points=dict(
                model_type="xy",
                x=[172, 172, 172, 172.5, 173],
                y=[-41, -40.5, -40, -40, -40],
                fmt="0.2f",
            ),
        )
        print(seg.render())
        seg = SEGMENT(
            points=dict(
                model_type="ij",
                i=[0, 0, 5],
                j=[0, 19, 19],
            ),
        )
        print(seg.render())

    """

    model_type: Literal["segment", "SEGMENT"] = Field(
        default="segment",
        description="Model type discriminator",
    )
    points: Union[XY, IJ] = Field(
        description="Points to define the segment",
        discriminator="model_type",
    )

    def cmd(self) -> str:
        return f"SEGMENT {self.points.model_type.upper()} {self.points.render()}"


class PAR(BaseSubComponent):
    """Spectral parameters.

    .. code-block:: text

        PAR [hs] [per] [dir] [dd]

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.boundary import PAR
        par = PAR(hs=1.5, per=8.1, dir=225)
        print(par.render())

    """

    model_type: Literal["par"] = Field(
        default="par",
        description="Model type discriminator",
    )
    hs: float = Field(
        description="The significant wave height (m)",
        gt=0.0,
    )
    per: float = Field(
        description=(
            "The characteristic period (s) of the energy spectrum (relative "
            "frequency; which is equal to absolute frequency in the absence of "
            "currents); `per` is the value of the peak period if option PEAK is "
            "chosen in command BOUND SHAPE or `per` is the value of the mean period, "
            "if option MEAN was chosen in command BOUND SHAPE."
        ),
        gt=0.0,
    )
    dir: float = Field(
        description=(
            "The peak wave direction thetapeak (degree), constant over frequencies"
        ),
        ge=-360.0,
        le=360.0,
    )
    dd: Optional[float] = Field(
        default=None,
        description=(
            "Coefficient of directional spreading; a `cos^m(θ)` distribution is "
            "assumed. `dd` is interpreted as the directional standard deviation in "
            "degrees, if the option DEGREES is chosen in the command BOUND SHAPE "
            "(SWAN default: 30). `dd` is interpreted as the power `m`, if the option "
            "POWER is chosen in the command BOUND SHAPE (SWAN default: 2)"
        ),
        ge=0.0,
        le=360.0,
    )

    def cmd(self) -> str:
        """Render subcomponent cmd."""
        repr = f"PAR hs={self.hs} per={self.per} dir={self.dir}"
        if self.dd is not None:
            repr += f" dd={self.dd}"
        return repr


class CONSTANTPAR(PAR):
    """Constant spectral parameters.

    .. code-block:: text

        CONSTANT PAR [hs] [per] [dir] ([dd])

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.boundary import CONSTANTPAR
        par = CONSTANTPAR(hs=1.5, per=8.1, dir=225)
        print(par.render())

    """

    model_type: Literal["constantpar", "CONSTANTPAR"] = Field(
        default="constantpar",
        description="Model type discriminator",
    )

    def cmd(self) -> str:
        """Render subcomponent cmd."""
        return f"CONSTANT {super().cmd()}"


class VARIABLEPAR(BaseSubComponent):
    """Variable spectral parameter.

    .. code-block:: text

        VARIABLE PAR < [len] [hs] [per] [dir] [dd] >

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.boundary import VARIABLEPAR
        par = VARIABLEPAR(
            hs=[1.5, 1.4, 1.1],
            per=[8.1, 8.0, 8.1],
            dir=[225, 226, 228],
            dd=[25, 22, 23],
            len=[0, 0.5, 1.0],
        )
        print(par.render())

    """

    model_type: Literal["variablepar", "VARIABLEPAR"] = Field(
        default="variablepar",
        description="Model type discriminator",
    )
    hs: list[Annotated[float, Field(ge=0.0)]] = Field(
        description="The significant wave height (m)",
    )
    per: list[Annotated[float, Field(ge=0.0)]] = Field(
        description=(
            "The characteristic period (s) of the energy spectrum (relative "
            "frequency; which is equal to absolute frequency in the absence of "
            "currents); `per` is the value of the peak period if option PEAK is "
            "chosen in command BOUND SHAPE or `per` is the value of the mean period, "
            "if option MEAN was chosen in command BOUND SHAPE"
        ),
    )
    dir: list[Annotated[float, Field(ge=-360.0, le=360.0)]] = Field(
        description=(
            "The peak wave direction thetapeak (degrees), constant over frequencies"
        ),
    )
    dd: list[Annotated[float, Field(ge=0.0, le=360.0)]] = Field(
        description=(
            "Coefficient of directional spreading; a $cos^m(θ)$ distribution is "
            "assumed. `dd` is interpreted as the directional standard deviation in "
            "degrees, if the option DEGREES is chosen in the command BOUND SHAPE "
            "(SWAN default: 30). `dd` is interpreted as the power `m`, if the option "
            "POWER is chosen in the command BOUND SHAPE (SWAN default: 2)"
        ),
    )
    dist: list[Annotated[float, Field(ge=0)]] = Field(
        alias="len",
        description=(
            "Is the distance from the first point of the side or segment to the point "
            "along the side or segment for which the incident wave spectrum is "
            "prescribed. Note: these points do no have to coincide with grid points of "
            "the computational grid. `len` is the distance in m or degrees in the case "
            "of spherical coordinates, not in grid steps. The values of `len` should "
            "be given in ascending order. The length along a SIDE is measured in "
            "clockwise or counterclockwise direction, depending on the options CCW or "
            "CLOCKWISE (see above). The option CCW is default. In case of a SEGMENT "
            "the length is measured from the indicated begin point of the segment"
        ),
    )

    @model_validator(mode="after")
    def ensure_equal_size(self) -> "VARIABLEPAR":
        for key in ["hs", "per", "dir", "dd"]:
            if len(getattr(self, key)) != len(self.dist):
                raise ValueError(f"Size of dist and {key} must be the same")
        return self

    def cmd(self) -> str:
        """Render subcomponent cmd."""
        repr = "VARIABLE PAR"
        for dist, hs, per, dir, dd in zip(
            self.dist, self.hs, self.per, self.dir, self.dd
        ):
            repr += f" &\n\tlen={dist} hs={hs} per={per} dir={dir} dd={dd}"
        return repr


class CONSTANTFILE(BaseSubComponent):
    """Constant file specification.

    .. code-block:: text

        CONSTANT FILE 'fname' [seq]

    There are three types of files:

    - TPAR files containing nonstationary wave parameters
    - files containing stationary or nonstationary 1D spectra
      (usually from measurements)
    - files containing stationary or nonstationary 2D spectra
      (from other computer programs or other SWAN runs)

    A TPAR file is for only one location; it has the string TPAR on the first
    line of the file and a number of lines which each contain 5 numbers, i.e.:
    Time (ISO-notation), Hs, Period (average or peak period depending on the
    choice given in command BOUND SHAPE), Peak Direction (Nautical or Cartesian,
    depending on command SET), Directional spread (in degrees or as power of cos
    depending on the choice given in command BOUND SHAPE).

    Note
    ----
    Example of a TPAR file:

    .. code-block:: text

        TPAR
        19920516.130000 4.2 12. -110. 22.
        19920516.180000 4.2 12. -110. 22.
        19920517.000000 1.2 8. -110. 22.
        19920517.120000 1.4 8.5 -80. 26
        19920517.200000 0.9 6.5 -95. 28

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.boundary import CONSTANTFILE
        par = CONSTANTFILE(fname="tpar.txt")
        print(par.render())

    """

    model_type: Literal["constantfile", "CONSTANTFILE"] = Field(
        default="constantfile",
        description="Model type discriminator",
    )
    fname: str = Field(
        description="Name of the file containing the boundary condition.",
        max_length=36,
    )
    seq: Optional[int] = Field(
        default=None,
        description=(
            "sequence number of geographic location in the file (see Appendix D); "
            "useful for files which contain spectra for more than one location. "
            "Note: a TPAR file always contains only one location so in this case "
            "`seq` must always be 1"
        ),
        ge=1,
    )

    def cmd(self) -> str:
        """Render subcomponent cmd."""
        repr = f"CONSTANT FILE fname='{self.fname}'"
        if self.seq is not None:
            repr += f" seq={self.seq}"
        return repr


class VARIABLEFILE(BaseSubComponent):
    """Variable file specification.

    .. code-block:: text

        VARIABLE FILE < [len] 'fname' [seq] >

    There are three types of files:

    - TPAR files containing nonstationary wave parameters
    - files containing stationary or nonstationary 1D spectra
      (usually from measurements)
    - files containing stationary or nonstationary 2D spectra
      (from other computer programs or other SWAN runs)

    A TPAR file is for only one location; it has the string TPAR on the first
    line of the file and a number of lines which each contain 5 numbers, i.e.:
    Time (ISO-notation), Hs, Period (average or peak period depending on the
    choice given in command BOUND SHAPE), Peak Direction (Nautical or Cartesian,
    depending on command SET), Directional spread (in degrees or as power of cos
    depending on the choice given in command BOUND SHAPE).

    Note
    ----
    Example of a TPAR file:

    .. code-block:: text

        TPAR
        19920516.130000 4.2 12. -110. 22.
        19920516.180000 4.2 12. -110. 22.
        19920517.000000 1.2 8. -110. 22.
        19920517.120000 1.4 8.5 -80. 26
        19920517.200000 0.9 6.5 -95. 28

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.boundary import VARIABLEFILE
        par = VARIABLEFILE(
            fname=["tpar1.txt", "tpar2.txt", "tpar3.txt"],
            len=[0.0, 0.5, 1.0],
        )
        print(par.render())

    """

    model_type: Literal["variablefile", "VARIABLEFILE"] = Field(
        default="variablefile",
        description="Model type discriminator",
    )
    fname: list[Annotated[str, Field(max_length=36)]] = Field(
        description="Names of the files containing the boundary condition",
    )
    seq: Optional[list[Annotated[int, Field(ge=1)]]] = Field(
        default=None,
        description=(
            "sequence number of geographic location in the file (see Appendix D); "
            "useful for files which contain spectra for more than one location. "
            "Note: a TPAR file always contains only one location so in this case "
            "[seq] must always be 1."
        ),
    )
    dist: list[Annotated[float, Field(ge=0)]] = Field(
        alias="len",
        description=(
            "Is the distance from the first point of the side or segment to the point "
            "along the side or segment for which the incident wave spectrum is "
            "prescribed. Note: these points do no have to coincide with grid points "
            "of the computational grid. [len] is the distance in m or degrees in the "
            "case of spherical coordinates, not in grid steps. The values of `len` "
            "should be given in ascending order. The length along a SIDE is measured "
            "in clockwise or counterclockwise direction, depending on the options CCW "
            "or CLOCKWISE (see above). The option CCW is default. In case of a "
            "SEGMENT the length is measured from the indicated begin point of the "
            "segment."
        ),
    )

    @model_validator(mode="after")
    def ensure_equal_size(self) -> "VARIABLEFILE":
        for key in ["fname", "seq"]:
            attr = getattr(self, key)
            if attr is not None and len(attr) != len(self.dist):
                raise ValueError(f"Size of dist and {key} must be the same")
        if self.seq is None:
            self.seq = [1] * len(self.dist)
        return self

    def cmd(self) -> str:
        """Render subcomponent cmd."""
        repr = "VARIABLE FILE"
        for dist, fname, seq in zip(self.dist, self.fname, self.seq):
            repr += f" &\n\tlen={dist} fname='{fname}' seq={seq}"
        return repr


class DEFAULT(BaseSubComponent):
    """Default initial conditions.

    .. code-block:: text

        DEFAULT

    The initial spectra are computed from the local wind velocities, using the
    deep-water growth curve of Kahma and Calkoen (1992), cut off at values of
    significant wave height and peak frequency from Pierson and Moskowitz (1964).
    The average (over the model area) spatial step size is used as fetch with local
    wind. The shape of the spectrum is default JONSWAP with a cos2-directional
    distribution (options are available: see command BOUND SHAPE).

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.boundary import DEFAULT
        init = DEFAULT()
        print(init.render())

    """

    model_type: Literal["default", "DEFAULT"] = Field(
        default="default",
        description="Model type discriminator",
    )


class ZERO(BaseSubComponent):
    """Zero initial conditions.

    .. code-block:: text

        ZERO

    The initial spectral densities are all 0; note that if waves are generated in the
    model only by wind, waves can become non-zero only by the presence of the
    ”A” term in the growth model; see the keyword AGROW in command GEN3.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.boundary import ZERO
        init = ZERO()
        print(init.render())

    """

    model_type: Literal["zero", "ZERO"] = Field(
        default="zero",
        description="Model type discriminator",
    )


class HOTSINGLE(BaseSubComponent):
    """Hotstart single initial conditions.

    .. code-block:: text

        HOTSTART SINGLE fname='fname' FREE|UNFORMATTED

    Initial wave field is read from file; this file was generated in a previous SWAN
    run by means of the HOTFILE command. If the previous run was nonstationary,
    the time found on the file will be assumed to be the initial time of computation. It
    can also be used for stationary computation as first guess. The computational grid
    (both in geographical space and in spectral space) must be identical to the one in
    the run in which the initial wave field was computed

    Input will be read from a single (concatenated) hotfile. In the case of a previous
    parallel MPI run, the concatenated hotfile can be created from a set of multiple
    hotfiles using the program hcat.exe, see Implementation Manual.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.boundary import HOTSINGLE
        init = HOTSINGLE(fname="hotstart.swn", format="free")
        print(init.render())

    """

    model_type: Literal["hotsingle", "HOTSINGLE"] = Field(
        default="hotsingle",
        description="Model type discriminator",
    )
    fname: str = Field(
        description="Name of the file containing the initial wave field",
        max_length=36,
    )
    format: Literal["free", "unformatted"] = Field(
        default="free",
        description=(
            "Format of the file containing the initial wave field. "
            "FREE: free format, UNFORMATTED: binary format"
        ),
    )

    def cmd(self) -> str:
        """Render subcomponent cmd."""
        return f"HOTSTART SINGLE fname='{self.fname}' {self.format.upper()}"


class HOTMULTIPLE(BaseSubComponent):
    """Hotstart multiple initial conditions.

    .. code-block:: text

        HOTSTART MULTIPLE fname='fname' FREE|UNFORMATTED

    Initial wave field is read from file; this file was generated in a previous SWAN
    run by means of the HOTFILE command. If the previous run was nonstationary,
    the time found on the file will be assumed to be the initial time of computation. It
    can also be used for stationary computation as first guess. The computational grid
    (both in geographical space and in spectral space) must be identical to the one in
    the run in which the initial wave field was computed

    Input will be read from multiple hotfiles obtained from a previous parallel MPI run.
    The number of files equals the number of processors. Hence, for the present run the
    same number of processors must be chosen.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.boundary import HOTMULTIPLE
        init = HOTMULTIPLE(fname="hotstart.swn", format="free")
        print(init.render())

    """

    model_type: Literal["hotmultiple", "HOTMULTIPLE"] = Field(
        default="hotmultiple",
        description="Model type discriminator",
    )
    fname: str = Field(
        description="Name of the file containing the initial wave field",
        max_length=36,
    )
    format: Literal["free", "unformatted"] = Field(
        default="free",
        description=(
            "Format of the file containing the initial wave field. "
            "FREE: free format, UNFORMATTED: binary format"
        ),
    )

    def cmd(self) -> str:
        """Render subcomponent cmd."""
        return f"HOTSTART MULTIPLE fname='{self.fname}' {self.format.upper()}"
