"""
SWAN Read Grid Subcomponents

This module contains subcomponents for reading grid data in SWAN,
including regular grids and various input formats.
"""

from abc import ABC
from typing import Literal, Optional, Union

from pydantic import Field, field_validator, model_validator

from rompy.core.logging import get_logger
from rompy.swan.subcomponents.base import BaseSubComponent
from rompy.swan.types import IDLA, GridOptions

logger = get_logger(__name__)


class GRIDREGULAR(BaseSubComponent):
    """SWAN Regular Grid subcomponent.

    .. code-block:: text

        xp yp alp xlen ylen mx my

    Note
    ----
    The direction of the x-axis `alp` must be 0 in case of spherical coordinates

    Note
    ----
    All coordinates and distances should be given in m when Cartesian coordinates are
    used or degrees when Spherical coordinates are used (see command COORD).

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.readgrid import GRIDREGULAR
        kwargs = dict(xp=173, yp=-40, alp=0, xlen=2, ylen=2, mx=199, my=199)
        grid = GRIDREGULAR(suffix="c", **kwargs)
        print(grid.render())
        grid = GRIDREGULAR(suffix="inp", **kwargs)
        print(grid.render())

    """

    model_type: Literal["gridregular", "GRIDREGULAR"] = Field(
        default="gridregular", description="Model type discriminator"
    )
    xp: float = Field(
        description="The x-coordinate of the origin in problem coordinates",
    )
    yp: float = Field(
        description="The y-coordinate of the origin in problem coordinates",
    )
    alp: Optional[float] = Field(
        default=0.0,
        description="Direction of the xaxis in degrees",
    )
    xlen: float = Field(
        description="Length of the computational grid in the x-direction"
    )
    ylen: float = Field(
        description="Length of the computational grid in the y-direction"
    )
    mx: int = Field(
        description=(
            "Number of meshes in computational grid in x-direction (this number is "
            "one less than the number of grid points in this domain)"
        ),
    )
    my: int = Field(
        description=(
            "Number of meshes in computational grid in y-direction (this number is "
            "one less than the number of grid points in this domain)"
        ),
    )
    suffix: Optional[str] = Field(
        default="", description="Suffix for rendering with each output grid parameter."
    )

    @property
    def dx(self):
        """Grid spacing in x-direction."""
        return self.xlen / self.mx

    @property
    def dy(self):
        """Grid spacing in y-direction."""
        return self.ylen / self.my

    def cmd(self) -> str:
        """Command file string for this subcomponent."""
        repr = f"xp{self.suffix}={self.xp}"
        repr += f" yp{self.suffix}={self.yp}"
        repr += f" alp{self.suffix}={self.alp}"
        repr += f" xlen{self.suffix}={self.xlen}"
        repr += f" ylen{self.suffix}={self.ylen}"
        repr += f" mx{self.suffix}={self.mx}"
        repr += f" my{self.suffix}={self.my}"
        return repr


class READGRID(BaseSubComponent, ABC):
    """SWAN grid reader abstract class.

    .. code-block:: text

        READGRID [grid_type] [fac] 'fname1' [idla] [nhedf] ([nhedt]) ([nhedvec]) &
            ->FREE|FORMAT|UNFORMATTED ('form'|[idfm])

    This is the base class for all input grids. It is not meant to be used directly.

    Note
    ----

    File format identifier:

    * 1: Format according to BODKAR convention (a standard of the Ministry of
      Transport and Public Works in the Netherlands). Format string: (10X,12F5.0)
    * 5: Format (16F5.0), an input line consists of 16 fields of 5 places each
    * 6: Format (12F6.0), an input line consists of 12 fields of 6 places each
    * 8: Format (10F8.0), an input line consists of 10 fields of 8 places each

    """

    model_type: Literal["readgrid", "READGRID"] = Field(
        default="readgrid", description="Model type discriminator"
    )
    grid_type: Union[GridOptions, Literal["coordinates"]] = Field(
        description="Type of the SWAN grid file",
    )
    fac: float = Field(
        default=1.0,
        description=(
            "SWAN multiplies all values that are read from file by `fac`. For instance "
            "if the values are given in unit decimeter, one should make `fac=0.1` to "
            "obtain values in m. To change sign use a negative `fac`"
        ),
        gt=0.0,
    )
    idla: IDLA = Field(
        default=1,
        description=(
            "Prescribes the order in which the values of bottom levels "
            "and other fields should be given in the file"
        ),
    )
    nhedf: int = Field(
        default=0,
        description=(
            "The number of header lines at the start of the file. The text in the "
            "header lines is reproduced in the print file created by SWAN . The file "
            "may start with more header lines than `nhedf` because the start of the "
            "file is often also the start of a time step and possibly also of a "
            "vector variable (each having header lines, see `nhedt` and `nhedvec`)"
        ),
        ge=0,
    )
    nhedvec: int = Field(
        default=0,
        description=(
            "For each vector variable: number of header lines in the file "
            "at the start of each component (e.g., x- or y-component)"
        ),
        ge=0,
    )
    format: Literal["free", "fixed", "unformatted"] = Field(
        default="free",
        description=(
            "File format, one of 'free', 'fixed' or 'unformatted'. If 'free', the "
            "file is assumed to use the FREE FORTRAN format. If 'fixed', the file is "
            "assumed to use a fixed format that must be specified by (only) one of "
            "'form' or 'idfm' arguments. Use 'unformatted' to read unformatted "
            "(binary) files (not recommended for ordinary use)"
        ),
    )
    form: Optional[str] = Field(
        default=None,
        description=(
            "A user-specified format string in Fortran convention, e.g., '(10X,12F5.0)'."
            "Only used if `format='fixed'`, do not use it if `idfm` is specified"
        ),
    )
    idfm: Optional[Literal[1, 5, 6, 8]] = Field(
        default=None,
        description=("File format identifier, only used if `format='fixed'`"),
    )

    @model_validator(mode="after")
    def check_format_definition(self) -> "READGRID":
        """Check the arguments specifying the file format are specified correctly."""
        if self.format == "free" and any([self.form, self.idfm]):
            logger.warn(f"FREE format, ignoring form={self.form} idfm={self.idfm}")
        elif self.format == "unformatted" and any([self.form, self.idfm]):
            logger.warn(
                f"UNFORMATTED format, ignoring form={self.form} idfm={self.idfm}"
            )
        elif self.format == "fixed" and not any([self.form, self.idfm]):
            raise ValueError(
                "FIXED format requires one of form or idfm to be specified"
            )
        elif self.format == "fixed" and all([self.form, self.idfm]):
            raise ValueError("FIXED format accepts only one of form or idfm")
        return self

    @property
    def format_repr(self):
        if self.format == "free":
            repr = "FREE"
        elif self.format == "fixed" and self.form:
            repr = f"FORMAT form='{self.form}'"
        elif self.format == "fixed" and self.idfm:
            repr = f"FORMAT idfm={self.idfm}"
        elif self.format == "unformatted":
            repr = "UNFORMATTED"
        return repr


class READCOORD(READGRID):
    """SWAN coordinates reader.

    .. code-block:: text

        READGRID COORDINATES [fac] 'fname' [idla] [nhedf] [nhedvec] &
            FREE|FORMAT ('form'|idfm)

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.readgrid import READCOORD
        readcoord = READCOORD(
            fac=1.0,
            fname="coords.txt",
            idla=3,
            format="free",
        )
        print(readcoord.render())

    """

    model_type: Literal["readcoord", "READCOORD"] = Field(
        default="readcoord", description="Model type discriminator"
    )
    grid_type: Literal["coordinates"] = Field(
        default="coordinates", description="Type of the SWAN grid file"
    )
    fname: str = Field(description="Name of the SWAN coordinates file")

    def cmd(self) -> str:
        repr = (
            f"READGRID COORDINATES fac={self.fac} fname='{self.fname}' "
            f"idla={self.idla} nhedf={self.nhedf} nhedvec={self.nhedvec} "
            f"{self.format_repr}"
        )
        return repr


class READINP(READGRID):
    """SWAN input grid reader.

    .. code-block:: text

        READINP GRID_TYPE [fac] ('fname1' | SERIES 'fname2') [idla] [nhedf] &
            ([nhedt]) [nhedvec] FREE|FORMAT ('form'|idfm)|UNFORMATTED`

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.readgrid import READINP
        readinp = READINP(
            grid_type="wind",
            fname1="wind.txt",
            fac=1.0,
            idla=3,
            format="free",
        )
        print(readinp.render())

    """

    model_type: Literal["readinp", "READINP"] = Field(
        default="readinp", description="Model type discriminator"
    )
    grid_type: Optional[GridOptions] = Field(
        default=None, description="Type of the SWAN grid file"
    )
    fname1: str = Field(
        description="Name of the file with the values of the variable.",
    )
    fname2: Optional[str] = Field(
        default=None,
        description=(
            "Name of file that contains the names of the files where the variables "
            "are given when the SERIES option is used. These names are to be given in "
            "proper time sequence. SWAN reads the next file when the previous file "
            "end has been encountered. In these files the input should be given in "
            "the same format as in the above file 'fname1' (that implies that a file "
            "should start with the start of an input time step)"
        ),
    )
    nhedt: int = Field(
        default=0,
        description=(
            "Only if variable is time dependent: number of header lines in the file "
            "at the start of each time level. A time step may start with more header "
            "lines than `nhedt` because the variable may be a vector variable which "
            "has its own header lines (see `nhedvec`)"
        ),
        ge=0,
    )

    @field_validator("grid_type")
    @classmethod
    def set_undefined(cls, v: str | None) -> str:
        """Allow for undefined value so it can be redefined in INPGRID components."""
        if v is None:
            return "undefined"
        return v

    def cmd(self) -> str:
        repr = f"READINP {self.grid_type.upper()} fac={self.fac} fname1='{self.fname1}'"
        if self.fname2:
            repr += f" SERIES fname2='{self.fname2}'"
        repr += f" idla={self.idla} nhedf={self.nhedf} nhedt={self.nhedt} nhedvec={self.nhedvec} {self.format_repr}"
        return repr
