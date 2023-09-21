"""Computational grid for SWAN."""
import logging
from pydantic import Field, model_validator
from typing import Literal, Optional
from abc import ABC, abstractmethod

from rompy.swan.components.base import BaseComponent
from rompy.swan.subcomponents.spectrum import SPECTRUM
from rompy.swan.subcomponents.readgrid import READCOORD


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

    Examples
    --------

    .. ipython:: python
        :okwarning:
        :okexcept:

        from rompy.swan.components.cgrid import REGULAR
        cgrid = REGULAR(
            xpc=0.0,
            ypc=0.0,
            alpc=0.0,
            xlenc=2000.0,
            ylenc=1300.0,
            mxc=100,
            myc=100,
            spectrum=dict(mdc=36, flow=0.04, fhigh=1.0),
        )
        print(cgrid.render())

    """

    model_type: Literal["regular", "REGULAR"] = Field(
        default="regular", description="Model type discriminator"
    )
    xpc: float = Field(
        default=0.0,
        description=(
            "Geographic location of the origin of the computational grid in the "
            "problem coordinate system (x-coordinate, in m)"
        ),
    )
    ypc: float = Field(
        default=0.0,
        description=(
            "Geographic location of the origin of the computational grid in the "
            "problem coordinate system (y-coordinate, in m)"
        ),
    )
    alpc: float = Field(
        default=0.0,
        description=(
            "Direction of the positive x-axis of the computational grid (in degrees, "
            "Cartesian convention). In 1D-mode, `alpc` should be equal to the "
            "direction `alpinp`"
        ),
    )
    xlenc: float = Field(
        description=(
            "Length of the computational grid in x-direction (in m). In case of "
            "spherical coordinates `xlenc` is in degrees"
        ),
    )
    ylenc: float = Field(
        description=(
            "Length of the computational grid in y-direction (in m). In 1D-mode, "
            "`ylenc` should be 0. In case of spherical coordinates `ylenc` is in "
            "degrees"
        ),
    )
    mxc: int = Field(
        description=(
            "Number of meshes in computational grid in x-direction (this number is "
            "one less than the number of grid points in this domain)"
        ),
    )
    myc: int = Field(
        description=(
            "Number of meshes in computational grid in y-direction (this number is "
            "one less than the number of grid points in this domain). In 1D-mode, "
            "`myc` should be 0"
        ),
    )

    def cmd(self) -> str:
        repr = (
            f"CGRID REGULAR xpc={self.xpc} ypc={self.ypc} alpc={self.alpc} "
            f"xlenc={self.xlenc} ylenc={self.ylenc} mxc={self.mxc} myc={self.myc} "
            f"{self.spectrum.render()}"
        )
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
        :okexcept:

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
        :okexcept:

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
        max_length=80,
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
