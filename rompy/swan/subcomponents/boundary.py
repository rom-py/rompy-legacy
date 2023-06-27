"""Subcomponents to be rendered inside of components."""
import logging
from typing import Optional, Literal
from pydantic import Field, root_validator, confloat, constr, conint

from rompy.swan.subcomponents.base import BaseSubComponent


logger = logging.getLogger(__name__)


class SIDE(BaseSubComponent):
    """Boundary over one side of computational domain.

    `SIDE NORTH|NW|WEST|SW|SOUTH|SE|E|NE CCW|CLOCKWISE`

    The boundary is one full side of the computational grid (in 1D cases either of the
    two ends of the 1D-grid). SHOULD NOT BE USED IN CASE OF CURVILINEAR GRIDS!

    """

    model_type: Literal["side"] = Field(
        default="side", description="Model type discriminator",
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


class SEGMENTXY(BaseSubComponent):
    """Boundary over a segment defined from point coordinates.

    `SEGMENT XY < [x] [y] >`

    The segment is defined by means of a series of points in terms of problem
    coordinates; these points do not have to coincide with grid points. The (straight)
    line connecting two points must be close to grid lines of the computational grid
    (the maximum distance is one hundredth of the length of the straight line).

    """

    model_type: Literal["segmentxy"] = Field(
        default="segmentxy", description="Model type discriminator",
    )
    points: list[tuple[float, float]] = Field(
        description="Pairs of (x, y) values to define the segment",
    )
    float_format: str = Field(
        default="0.8f",
        description="The format to use for the floats in the points",
    )

    def cmd(self) -> str:
        repr = f"SEGMENT XY &"
        for point in self.points:
            repr += (
                f"\n\t{point[0]:{self.float_format}} {point[1]:{self.float_format}} &"
            )
        return repr + "\n\t"


class SEGMENTIJ(BaseSubComponent):
    """Boundary over a segment defined from grid indices.

    `SEGMENT IJ < [i] [j] >`

    The segment is defined by means of a series of computational grid points given in
    terms of grid indices (origin at 0,0); not all grid points on the segment have to
    be mentioned. If two points are on the same grid line, intermediate points are
    assumed to be on the segment as well.

    """

    model_type: Literal["segmentij"] = Field(
        default="segmentij", description="Model type discriminator",
    )
    points: list[tuple[int, int]] = Field(
        description="Pairs of (i, j) values to define the segment",
    )

    def cmd(self) -> str:
        repr = f"SEGMENT IJ &"
        for point in self.points:
            repr += f"\n\t{point[0]} {point[1]} &"
        return repr + "\n\t"


class PAR(BaseSubComponent):
    """Spectral parameters.

    `PAR [hs] [per] [dir] [dd]`

    """

    model_type: Literal["par"] = Field(
        default="par", description="Model type discriminator",
    )
    hs: confloat(gt=0.0) = Field(
        description="The significant wave height (m)",
    )
    per: confloat(gt=0.0) = Field(
        description=(
            "The characteristic period (s) of the energy spectrum (relative "
            "frequency; which is equal to absolute frequency in the absence of "
            "currents); `per` is the value of the peak period if option PEAK is "
            "chosen in command BOUND SHAPE or `per` is the value of the mean period, "
            "if option MEAN was chosen in command BOUND SHAPE."
        ),
    )
    dir: confloat(ge=-360.0, le=360.0) = Field(
        description=(
            "The peak wave direction θpeak (degrees), constant over frequencies"
        ),
    )
    dd: confloat(ge=0.0, le=360.0) = Field(
        description=(
            "Coefficient of directional spreading; a $cos^m(θ)$ distribution is "
            "assumed. `dd` is interpreted as the directional standard deviation in "
            "degrees, if the option DEGREES is chosen in the command BOUND SHAPE. "
            "Default: `dd=30`. `dd` is interpreted as the power `m`, if the option "
            "POWER is chosen in the command BOUND SHAPE. Default: `dd=2`."
        ),
    )

    def cmd(self) -> str:
        """Render subcomponent cmd."""
        return f"PAR hs={self.hs} per={self.per} dir={self.dir} dd={self.dd}"


class CONSTANTPAR(PAR):
    """Constant spectral parameters.

    `CONSTANT PAR [hs] [per] [dir] [dd]`

    """

    model_type: Literal["constantpar"] = Field(
        default="constantpar", description="Model type discriminator",
    )

    def cmd(self) -> str:
        """Render subcomponent cmd."""
        return f"CONSTANT {super().cmd()}"


class VARIABLEPAR(BaseSubComponent):
    """Variable spectral parameter.

    `VARIABLE PAR < [len] [hs] [per] [dir] [dd] >`

    """

    model_type: Literal["variablepar"] = Field(
        default="variablepar", description="Model type discriminator",
    )
    hs: list[confloat(ge=0.0)] = Field(
        description="The significant wave height (m)",
    )
    per: list[confloat(ge=0.0)] = Field(
        description=(
            "The characteristic period (s) of the energy spectrum (relative "
            "frequency; which is equal to absolute frequency in the absence of "
            "currents); `per` is the value of the peak period if option PEAK is "
            "chosen in command BOUND SHAPE or `per` is the value of the mean period, "
            "if option MEAN was chosen in command BOUND SHAPE."
        ),
    )
    dir: list[confloat(ge=-360.0, le=360.0)] = Field(
        description="The peak wave direction θpeak (degrees), constant over frequencies"
    )
    dd: list[confloat(ge=0.0, le=360.0)] = Field(
        description=(
            "Coefficient of directional spreading; a $cos^m(θ)$ distribution is "
            "assumed. `dd` is interpreted as the directional standard deviation in "
            "degrees, if the option DEGREES is chosen in the command BOUND SHAPE. "
            "Default: `dd=30`. `dd` is interpreted as the power `m`, if the option "
            "POWER is chosen in the command BOUND SHAPE (SWAN default: `dd=2`)."
        ),
    )
    dist: list[confloat(ge=0)] = Field(
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
            "the length is measured from the indicated begin point of the segment."
        ),
    )

    @root_validator
    def ensure_equal_size(cls, values):
        for key in ["hs", "per", "dir", "dd"]:
            if len(values[key]) != len(values["dist"]):
                raise ValueError(f"Sizes of dist and {key} must be the size")
        return values

    def cmd(self) -> str:
        """Render subcomponent cmd."""
        repr = "VARIABLE PAR"
        for dist, hs, per, dir, dd in zip(
            self.dist, self.hs, self.per, self.dir, self.dd
        ):
            repr += f" &\n\t\tlen={dist} hs={hs} per={per} dir={dir} dd={dd}"
        return repr


class CONSTANTFILE(BaseSubComponent):
    """Constant file specification.

    `CONSTANT FILE 'fname' [seq]`

    Note
    ----
    There are three types of files:
    - TPAR files containing nonstationary wave parameters.
    - files containing stationary or nonstationary 1D spectra
      (usually from measurements).
    - files containing stationary or nonstationary 2D spectra
      (from other computer programs or other SWAN runs).

    A TPAR file is for only one location; it has the string TPAR on the first
    line of the file and a number of lines which each contain 5 numbers, i.e.:
    Time (ISO-notation), Hs, Period (average or peak period depending on the
    choice given in command BOUND SHAPE), Peak Direction (Nautical or Cartesian,
    depending on command SET), Directional spread (in degrees or as power of cos
    depending on the choice given in command BOUND SHAPE).

    Example of a TPAR file
    ----------------------
    TPAR
    19920516.130000 4.2 12. -110. 22.
    19920516.180000 4.2 12. -110. 22.
    19920517.000000 1.2 8. -110. 22.
    19920517.120000 1.4 8.5 -80. 26
    19920517.200000 0.9 6.5 -95. 28

    """

    model_type: Literal["constantfile"] = Field(
        default="constantfile", description="Model type discriminator",
    )
    fname: constr(max_length=40) = Field(
        description="Name of the file containing the boundary condition.",
    )
    seq: Optional[conint(ge=1)] = Field(
        description=(
            "sequence number of geographic location in the file (see Appendix D); "
            "useful for files which contain spectra for more than one location. "
            "Note: a TPAR file always contains only one location so in this case "
            "[seq] must always be 1."
        ),
    )

    def cmd(self) -> str:
        """Render subcomponent cmd."""
        repr = f"CONSTANT FILE fname='{self.fname}'"
        if self.seq is not None:
            repr += f" seq={self.seq}"
        return repr


class VARIABLEFILE(BaseSubComponent):
    """Variable file specification.

    `VARIABLE FILE < [len] 'fname' [seq] >`

    Note
    ----
    There are three types of files:
    - TPAR files containing nonstationary wave parameters.
    - files containing stationary or nonstationary 1D spectra
      (usually from measurements).
    - files containing stationary or nonstationary 2D spectra
      (from other computer programs or other SWAN runs).

    A TPAR file is for only one location; it has the string TPAR on the first
    line of the file and a number of lines which each contain 5 numbers, i.e.:
    Time (ISO-notation), Hs, Period (average or peak period depending on the
    choice given in command BOUND SHAPE), Peak Direction (Nautical or Cartesian,
    depending on command SET), Directional spread (in degrees or as power of cos
    depending on the choice given in command BOUND SHAPE).

    Example of a TPAR file
    ----------------------
    TPAR
    19920516.130000 4.2 12. -110. 22.
    19920516.180000 4.2 12. -110. 22.
    19920517.000000 1.2 8. -110. 22.
    19920517.120000 1.4 8.5 -80. 26
    19920517.200000 0.9 6.5 -95. 28

    """

    model_type: Literal["variablefile"] = Field(
        default="variablefile", description="Model type discriminator",
    )
    fname: list[constr(max_length=40)] = Field(
        description="Names of the file containing the boundary condition",
    )
    seq: Optional[list[conint(ge=1)]] = Field(
        description=(
            "sequence number of geographic location in the file (see Appendix D); "
            "useful for files which contain spectra for more than one location. "
            "Note: a TPAR file always contains only one location so in this case "
            "[seq] must always be 1."
        ),
    )
    dist: list[confloat(ge=0)] = Field(
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

    @root_validator
    def ensure_equal_size(cls, values):
        for key in ["fname", "seq"]:
            if values.get(key) is not None and len(values[key]) != len(values["dist"]):
                raise ValueError(f"Sizes of dist and {key} must be the size")
        if values.get("seq") is None:
            values["seq"] = [1] * len(values["dist"])
        return values

    def cmd(self) -> str:
        """Render subcomponent cmd."""
        repr = "VARIABLE FILE"
        for dist, fname, seq in zip(self.dist, self.fname, self.seq):
            repr += f" &\n\t\tlen={dist} fname='{fname}' seq={seq}"
        return repr


class DEFAULT(BaseSubComponent):
    """Default initial conditions.

    `DEFAULT`

    The initial spectra are computed from the local wind velocities, using the
    deep-water growth curve of Kahma and Calkoen (1992), cut off at values of
    significant wave height and peak frequency from Pierson and Moskowitz (1964).
    The average (over the model area) spatial step size is used as fetch with local
    wind. The shape of the spectrum is default JONSWAP with a cos2-directional
    distribution (options are available: see command BOUND SHAPE).

    """

    model_type: Literal["default"] = Field(
        default="default", description="Model type discriminator",
    )


class ZERO(BaseSubComponent):
    """Zero initial conditions.

    `ZERO`

    The initial spectral densities are all 0; note that if waves are generated in the
    model only by wind, waves can become non-zero only by the presence of the
    ”A” term in the growth model; see the keyword AGROW in command GEN3.

    """

    model_type: Literal["zero"] = Field(
        default="zero", description="Model type discriminator",
    )


class HOTSINGLE(BaseSubComponent):
    """Hotstart single initial conditions.

    `HOTSTART SINGLE fname='fname' FREE|UNFORMATTED`

    Initial wave field is read from file; this file was generated in a previous SWAN
    run by means of the HOTFILE command. If the previous run was nonstationary,
    the time found on the file will be assumed to be the initial time of computation. It
    can also be used for stationary computation as first guess. The computational grid
    (both in geographical space and in spectral space) must be identical to the one in
    the run in which the initial wave field was computed

    Input will be read from a single (concatenated) hotfile. In the case of a previous
    parallel MPI run, the concatenated hotfile can be created from a set of multiple
    hotfiles using the program hcat.exe, see Implementation Manual.

    """

    model_type: Literal["hotsingle"] = Field(
        default="hotsingle", description="Model type discriminator",
    )
    fname: constr(max_length=85) = Field(
        description="Name of the file containing the initial wave field",
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

    Initial wave field is read from file; this file was generated in a previous SWAN
    run by means of the HOTFILE command. If the previous run was nonstationary,
    the time found on the file will be assumed to be the initial time of computation. It
    can also be used for stationary computation as first guess. The computational grid
    (both in geographical space and in spectral space) must be identical to the one in
    the run in which the initial wave field was computed

    Input will be read from multiple hotfiles obtained from a previous parallel MPI run.
    The number of files equals the number of processors. Hence, for the present run the
    same number of processors must be chosen.

    """

    model_type: Literal["hotmultiple"] = Field(
        default="hotmultiple", description="Model type discriminator",
    )
    fname: constr(max_length=85) = Field(
        description="Name of the file containing the initial wave field",
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
