"""Computational grid for SWAN."""

import logging
from abc import ABC, abstractmethod
from typing import Literal, Optional

from pydantic import Field, model_validator

from rompy.swan.components.base import BaseComponent
from rompy.swan.subcomponents.readgrid import GRIDREGULAR, READCOORD
from rompy.swan.subcomponents.spectrum import SPECTRUM

logger = logging.getLogger(__name__)


class CGRID(BaseComponent, ABC):
    """SWAN computational grid abstract component.

    .. code-block:: text

        CGRID ->REGULAR|CURVILINEAR|UNSTRUCTURED &
            CIRCLE|SECTOR [mdc] [flow] [fhigh] [msc]

    This class should not be used directly.

    """

    model_type: Literal["cgrid", "CGRID"] = Field(
        default="cgrid", description="Model type discriminator"
    )
    spectrum: SPECTRUM = Field(description="Spectrum subcomponent")

    @abstractmethod
    def cmd(self):
        pass


class REGULAR(CGRID):
    """SWAN regular computational grid.

    .. code-block:: text

        CGRID REGULAR [xpc] [ypc] [alpc] [xlenc] [ylenc] [mxc] [myc] &
            ->CIRCLE|SECTOR [mdc] [flow] [fhigh] [msc]

    This is a group component that includes a `CGRID` and a `READGRID` component.

    Note
    ----
    In 1D-mode, `alpc` should be equal to the direction `alpinp`.


    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.cgrid import REGULAR
        cgrid = REGULAR(
            grid=dict(xp=0, yp=0, alp=0, xlen=2000, ylen=1300, mx=100, my=100),
            spectrum=dict(mdc=36, flow=0.04, fhigh=1.0),
        )
        print(cgrid.render())

    """

    model_type: Literal["regular", "REGULAR"] = Field(
        default="regular", description="Model type discriminator"
    )
    grid: GRIDREGULAR = Field(description="Computational grid definition")

    @model_validator(mode="after")
    def grid_suffix(self) -> "REGULAR":
        """Set expected grid suffix."""
        if self.grid.suffix != "c":
            logger.debug(f"Set grid suffix 'c' instead of {self.grid.suffix}")
            self.grid.suffix = "c"
        return self

    def cmd(self) -> str:
        repr = f"CGRID REGULAR {self.grid.render()} {self.spectrum.render()}"
        return repr


class CURVILINEAR(CGRID):
    """SWAN curvilinear computational grid.

    .. code-block:: text

        CGRID CURVILINEAR [mxc] [myc] (EXCEPTION [xexc] [yexc])
            ->CIRCLE|SECTOR [mdc] [flow] [fhigh] [msc]
        READGRID COORDINATES [fac] 'fname' [idla] [nhedf] [nhedvec] &
            FREE|FORMAT ('form'|[idfm])

    This is a group component that includes a `CGRID` and a `READGRID` component.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.cgrid import CURVILINEAR
        cgrid = CURVILINEAR(
            mxc=199,
            myc=199,
            readcoord=dict(fname="./coords.txt"),
            spectrum=dict(mdc=36, flow=0.04, fhigh=1.0),
        )
        print(cgrid.render())

    """

    model_type: Literal["curvilinear", "CURVILINEAR"] = Field(
        default="curvilinear", description="Model type discriminator"
    )
    mxc: int = Field(
        description=(
            "Number of meshes in computational grid in ξ-direction (this number is "
            "one less than the number of grid points in this domain)."
        ),
    )
    myc: int = Field(
        description=(
            "Number of meshes in computational grid in η-direction (this number is "
            "one less than the number of grid points in this domain)."
        ),
    )
    xexc: Optional[float] = Field(
        default=None,
        description=(
            "the value which the user uses to indicate that a grid point is to be "
            "ignored in the computations (this value is provided by the user at the "
            "location of the x-coordinate considered in the file of the "
            "x-coordinates, see command READGRID COOR)."
        ),
    )
    yexc: Optional[float] = Field(
        default=None,
        description=(
            "the value which the user uses to indicate that a grid point is to be "
            "ignored in the computations (this value is provided by the user at the "
            "location of the y-coordinate considered in the file of the "
            "y-coordinates, see command READGRID COOR)."
        ),
    )
    readcoord: READCOORD = Field(
        description="Grid coordinates reader.",
    )

    @model_validator(mode="after")
    def xexc_and_yexc_or_neither(self) -> "CURVILINEAR":
        if [self.xexc, self.yexc].count(None) == 1:
            raise ValueError("xexc and yexc must be specified together")
        return self

    @property
    def exception(self):
        if self.xexc is not None:
            return f"EXCEPTION xexc={self.xexc} xexc={self.yexc}"
        else:
            return ""

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

    def cmd(self) -> str:
        repr = f"CGRID CURVILINEAR mxc={self.mxc} myc={self.myc}"
        if self.exception:
            repr += f" {self.exception}"
        repr += f" {self.spectrum.render()}"
        repr = [repr] + [self.readcoord.render()]
        return repr


class UNSTRUCTURED(CGRID):
    """SWAN unstructured computational grid.

    .. code-block:: text

        CGRID UNSTRUCTURED CIRCLE|SECTOR [mdc] [flow] [fhigh] [msc]
        READGRID UNSTRUCTURED [grid_type] ('fname')

    This is a group component that includes a `CGRID` and a `READGRID` component.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.cgrid import UNSTRUCTURED
        cgrid = UNSTRUCTURED(
            grid_type="adcirc",
            spectrum=dict(mdc=36, flow=0.04, fhigh=1.0),
        )
        print(cgrid.render())

    """

    model_type: Literal["unstructured"] = Field(
        default="unstructured", description="Model type discriminator"
    )
    grid_type: Literal["adcirc", "triangle", "easymesh"] = Field(
        default="adcirc",
        description="Unstructured grid type",
    )
    fname: Optional[str] = Field(
        default=None,
        description="Name of the file containing the unstructured grid",
        max_length=36,
    )

    @model_validator(mode="after")
    def check_fname_required(self) -> "UNSTRUCTURED":
        """Check that fname needs to be provided."""
        if self.grid_type == "adcirc" and self.fname is not None:
            raise ValueError("fname must not be specified for ADCIRC grid")
        elif self.grid_type != "adcirc" and self.fname is None:
            raise ValueError(f"fname must be specified for {self.grid_type} grid")
        return self

    def cmd(self) -> str:
        repr = [f"CGRID UNSTRUCTURED {self.spectrum.cmd()}"]
        repr += [f"READGRID UNSTRUCTURED {self.grid_type.upper()}"]
        if self.grid_type in ["triangle", "easymesh"]:
            repr[-1] += f" fname='{self.fname}'"
        return repr
