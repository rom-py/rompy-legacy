"""Boundary for SWAN."""
from typing import Literal, Optional
from pathlib import Path
from pydantic import Field

from rompy.swan.components.base import BaseComponent
from rompy.swan.subcomponents.spectrum import SHAPESPEC
from rompy.swan.subcomponents.boundary import (
    DEFAULT,
    ZERO,
    HOTSINGLE,
    HOTMULTIPLE,
    SIDE,
    SEGMENTXY,
    SEGMENTIJ,
    PAR,
    CONSTANTPAR,
    VARIABLEPAR,
    CONSTANTFILE,
    VARIABLEFILE,
)


HERE = Path(__file__).parent


class INITIAL(BaseComponent):
    """Initial conditions.

    .. code-block:: text

        INITIAL -> DEFAULT|ZERO|PAR|HOTSTART

    This command can be used to specify the initial values for a stationary (INITIAL
    HOTSTART only) or nonstationary computation. The initial values thus specified
    override the default initialization (see Section 2.6.3). Note that it is possible
    to obtain an initial state by carrying out a previous stationary or nonstationary
    computation.

    Examples
    --------

    .. ipython:: python
        :okwarning:
        :okexcept:

        from rompy.swan.components.boundary import INITIAL
        init = INITIAL()
        print(init.render())
        init = INITIAL(
            kind=dict(model_type="hotmultiple", fname="hotstart.swn", format="free")
        )
        print(init.render())

    """

    model_type: Literal["initial", "INITIAL"] = Field(
        default="initial", description="Model type discriminator",
    )
    kind: DEFAULT | ZERO | PAR | HOTSINGLE | HOTMULTIPLE = Field(
        default_factory=DEFAULT, description="Initial condition type",
    )

    def cmd(self) -> str:
        repr = f"INITIAL {self.kind.render()}"
        return repr


class BOUNDSPEC(BaseComponent):
    """Parametric or spectra boundary along sides or segments.

    .. code-block:: text

        BOUNDSPEC ->SIDE|SEGMENT CONSTANT|VARIABLE PAR|FILE

    This command BOUNDSPEC defines parametric spectra at the boundary. It consists of
    two parts, the first part defines the boundary side or segment where the spectra
    will be given, the second part defines the spectral parameters of these spectra.
    Note that in fact only the incoming wave components of these spectra are used by
    SWAN. The fact that complete spectra are calculated at the model boundaries from
    the spectral parameters should not be misinterpreted. Only the incoming components
    are effective in the computation.

    TODO: Add support for unstructured grid (k).

    Examples
    --------

    .. ipython:: python
        :okwarning:
        :okexcept:

        from rompy.swan.components.boundary import BOUNDSPEC
        boundary = BOUNDSPEC(
            shapespec=dict(model_type="shapespec", shape=dict(model_type="pm")),
            location=dict(model_type="side", side="west", direction="ccw"),
            data=dict(model_type="constantpar", hs=2, per=8, dir=270, dd=30),
        )
        print(boundary.render())

    """

    model_type: Literal["boundspec", "BOUNDSPEC"] = Field(
        default="boundspec", description="Model type discriminator",
    )
    shapespec: SHAPESPEC = Field(
        default_factory=SHAPESPEC, description="Spectral shape specification",
    )
    location: SIDE | SEGMENTXY | SEGMENTIJ = Field(
        description="Location to apply the boundary",
    )
    data: CONSTANTPAR | CONSTANTFILE | VARIABLEPAR | VARIABLEFILE = Field(
        description="Spectral data",
    )

    def cmd(self) -> list:
        repr = [f"{self.shapespec.render()}"]
        repr += [f"BOUNDSPEC {self.location.render()}{self.data.render()}"]
        return repr


class BOUNDNEST1(BaseComponent):
    """Boundary spectra from a coarser SWAN nest.

    .. code-block:: text

        BOUNDNEST1 NEST 'fname' ->CLOSED|OPEN

    With this optional command a nested SWAN run can be carried out with the boundary
    conditions obtained from a coarse grid SWAN run (generated in that previous SWAN
    run with command NESTOUT). The spectral frequencies and directions of the coarse
    grid run do not have to coincide with the frequencies and directions used in the
    nested SWAN run; SWAN will interpolate to these frequencies and directions in the
    nested run (see Section 2.6.3). To generate the nest boundary in the coarse grid
    run, use command NGRID. For the nested run, use the command CGRID with identical
    geographical information except the number of meshes (which will be much higher for
    the nested run). This BOUNDNEST1 command is not available for 1D computations; in
    such cases the commands SPECOUT and BOUNDSPEC can be used for the same purpose. A
    nested SWAN run must use the same coordinate system as the coarse grid SWAN run.
    For a curvilinear grid, it is advised to use the commands POINTS or CURVE and
    SPECOUT instead of NGRID and NESTOUT.

    Examples
    --------

    .. ipython:: python
        :okwarning:
        :okexcept:

        from rompy.swan.components.boundary import BOUNDNEST1
        boundary = BOUNDNEST1(fname="boundary.swn", rectangle="closed")
        print(boundary.render())

    """

    model_type: Literal["boundnest1", "BOUNDNEST1"] = Field(
        default="boundnest1", description="Model type discriminator",
    )
    fname: str = Field(
        description=(
            "Name of the file containing the boundary conditions for the present run, "
            "created by the previous SWAN coarse grid run. This file is structured "
            "according to the rules given in Appendix D for 2D spectra"
        ),
        min_length=1,
        max_length=98,
    )
    rectangle: Literal["closed", "open"] = Field(
        default="closed",
        description=(
            "Boundary is defined over a closed (default) or an open rectangle. "
            "Boundary generated from the NESTOUT command is aways closed"
        ),
    )

    def cmd(self) -> str:
        return f"BOUNDNEST1 NEST fname='{self.fname}' {self.rectangle.upper()}"


class BOUNDNEST2(BaseComponent):
    """Boundary spectra from WAM.

    .. code-block:: text

        BOUNDNEST2 WAMNEST 'fname' FREE|UNFORMATTED ->CRAY|WKSTAT [xgc] [ygc] [lwdate]

    With this optional command (not fully tested) a nested SWAN run can be carried out
    with the boundary conditions obtained from a coarse grid WAM run (WAM Cycle 4.5,
    source code as distributed by the Max Planck Institute in Hamburg). The spectral
    frequencies and directions of the coarse grid run do not have to coincide with the
    frequencies and directions used in the nested SWAN run; SWAN will interpolate to
    these frequencies and directions in the nested run (see Section 2.6.3). Note that
    SWAN will accept output of a WAM output location only if the SWAN grid point on the
    nest boundary lies within a rectangle between two consecutive WAM output locations
    with a width equal to 0.1 times the distance between these output locations on
    either side of the line between these WAM output locations. This BOUNDNEST2 command
    is not available for 1D computations. Only boundary conditions generated by WAM
    Cycle 4.5 can be read properly by SWAN. A nested SWAN run may use either Cartesian
    or spherical coordinates. A curvilinear grid may be used in the nested grid but the
    boundaries of this nest should conform to the rectangular course grid nest
    boundaries. WAM output files are unformatted (binary); this usually implies that
    WAM and SWAN have to run on the same computer. For those cases where WAM and SWAN
    run on different types of machines (binary files do not transfer properly), the
    option FREE is available in this command. The distributed version of WAM does not
    support the required free format nesting output; WAM users who modify WAM such that
    it can make formatted output, must modify WAM such that the files made by WAM can
    be read in free format, i.e. with at least a blank or comma between numbers.

    Note
    ----
    the contents of 'fname' file could look like:

    .. code-block:: text

        CBO9212010000
        CBO9212020000
        CBO9212030000

    Examples
    --------

    .. ipython:: python
        :okwarning:
        :okexcept:

        from rompy.swan.components.boundary import BOUNDNEST2
        boundary = BOUNDNEST2(fname="boundary.wam", format="cray", lwdate=12)
        print(boundary.render())

    """

    model_type: Literal["boundnest2", "BOUNDNEST2"] = Field(
        default="boundnest2", description="Model type discriminator",
    )
    fname: str = Field(
        description=(
            "A file name that contains all the names of WAM files containing the "
            "nested boundary conditions in time-sequence (usually one file per day)"
        ),
        min_length=1,
        max_length=48,
    )
    format: Literal["cray", "wkstat", "free"] = Field(
        description=(
            "Format of the WAM file. `cray`: CRAY version of WAM, `wkstat`: "
            "WORKSTATION version of WAM, `free`: Free format (these files are not "
            "generated standard by WAM)"
        ),
    )
    xgc: Optional[float] = Field(
        default=None,
        description=(
            "If SWAN is used with Cartesian coordinates: longitude of south-west "
            "corner of SWAN computational grid (in degrees); if the south-west "
            "corner of the nest in the WAM computation is on land this value is "
            "required. If SWAN is used with spherical coordinates then `xgc` is "
            "ignored by SWAN (SWAN default: the location of the first spectrum "
            "encountered in the nest file"
        ),
    )
    ygc: Optional[float] = Field(
        default=None,
        description=(
            "If SWAN is used with Cartesian coordinates: latitude of south-west "
            "corner of SWAN computational grid (in degrees); if the south-west "
            "corner of the nest in the WAM computation is on land this value is "
            "required. If SWAN is used with spherical coordinates then `ygc` is "
            "ignored by SWAN (SWAN default: the location of the first spectrum "
            "encountered in the nest file"
        ),
    )
    lwdate: Literal[10, 12, 14] = Field(
        default=12,
        description=(
            "Length of character string for date-time as used in the WAM files. "
            "Possible values are: 10 (i.e. YYMMDDHHMM), 12 (i.e. YYMMDDHHMMSS) "
            "or 14 (i.e. YYYYMMDDHHMMSS) (SWAN default: `lwdate` = 12)"
        ),
    )

    @property
    def format_str(self):
        if self.format == "cray":
            return "UNFORMATTED CRAY"
        elif self.format == "wkstat":
            return "UNFORMATTED WKSTAT"
        elif self.format == "free":
            return "FREE"
        else:
            raise ValueError(f"Unknown format {self.format}")

    def cmd(self) -> str:
        repr = f"BOUNDNEST2 WAMNEST fname='{self.fname}' {self.format_str}"
        if self.xgc is not None:
            repr += f" xgc={self.xgc}"
        if self.ygc is not None:
            repr += f" ygc={self.ygc}"
        repr += f" lwdate={self.lwdate}"
        return repr


class BOUNDNEST3(BaseComponent):
    """Boundary spectra from WAVEWATCHIII.

    .. code-block:: text

        BOUNDNEST3 WW3 'fname' FREE|UNFORMATTED ->CLOSED|OPEN [xgc] [ygc]

    With this optional command a nested SWAN run can be carried out with the boundary
    conditions obtained from a coarse grid WAVEWATCH III run. The spectral frequencies
    and directions of the coarse grid run do not have to coincide with the frequencies
    and directions used in the nested SWAN run; SWAN will interpolate to these
    frequencies and directions in the nested run (see Section 2.6.3). The output files
    of WAVEWATCH III have to be created with the post-processor of WAVEWATCH III as
    output transfer files (formatted or unformatted) with WW_3 OUTP (output type 1 sub
    type 3) at the locations along the nest boundary (i.e. computational grid points in
    WAVEWATCH III). These locations are equal to the corner points of the SWAN nested
    grid and optionally also distributed between the corner points of the SWAN nested
    grid (the boundary of the WAVEWATCH III nested grid need not be closed and may
    cover land). The locations should be output by WAVEWATCH III in sequence (going
    along the nest boundary, clockwise or counterclockwise). Note that SWAN will accept
    output of a WAVEWATCH III output location only if the SWAN grid point on the nest
    boundary lies within a rectangle between two consecutive WAVEWATCH III output
    locations with a width equal to 0.1 times the distance between these output
    locations on either side of the line between these WAVEWATCH III output locations.
    This BOUNDNEST3 command is not available for 1D computations. A nested SWAN run may
    use either Cartesian or spherical coordinates. A curvilinear grid may be used in
    the nested grid but the boundaries of this nest should conform to the rectangular
    course grid nest boundaries.

    Examples
    --------

    .. ipython:: python
        :okwarning:
        :okexcept:

        from rompy.swan.components.boundary import BOUNDNEST3
        boundary = BOUNDNEST3(
            fname="boundary.ww3",
            format="free",
            rectangle="closed",
        )
        print(boundary.render())

    """

    model_type: Literal["boundnest3", "BOUNDNEST3"] = Field(
        default="boundnest3", description="Model type discriminator",
    )
    fname: str = Field(
        description=(
            "The name of the file that contains the spectra computed by WAVEWATCH III"
        ),
        min_length=1,
        max_length=62,
    )
    format: Literal["unformatted", "free"] = Field(
        description=(
            "Format of the WW3 file. `unformatted`: The input WW3 files are binary, "
            "`free`: The input WW3 files are formatted"
        ),
    )
    rectangle: Literal["closed", "open"] = Field(
        default="closed",
        description=(
            "Boundary is defined over a closed (default) or an open rectangle. "
            "Boundary generated from the NESTOUT command is aways closed"
        ),
    )
    xgc: Optional[float] = Field(
        default=None,
        description=(
            "If SWAN is used with Cartesian coordinates: longitude of south-west "
            "corner of SWAN computational grid (in degrees); if the south-west "
            "corner of the nest in the WAM computation is on land this value is "
            "required. If SWAN is used with spherical coordinates then `xgc` is "
            "ignored by SWAN (SWAN default: the location of the first spectrum "
            "encountered in the nest file. "
        ),
    )
    ygc: Optional[float] = Field(
        default=None,
        description=(
            "If SWAN is used with Cartesian coordinates: latitude of south-west "
            "corner of SWAN computational grid (in degrees); if the south-west "
            "corner of the nest in the WAM computation is on land this value is "
            "required. If SWAN is used with spherical coordinates then `ygc` is "
            "ignored by SWAN (SWAN default: the location of the first spectrum "
            "encountered in the nest file. "
        ),
    )

    def cmd(self) -> str:
        repr = f"BOUNDNEST3 WW3 fname='{self.fname}' {self.format.upper()} "
        repr += f"{self.rectangle.upper()}"
        if self.xgc is not None:
            repr += f" xgc={self.xgc}"
        if self.ygc is not None:
            repr += f" ygc={self.ygc}"
        return repr
