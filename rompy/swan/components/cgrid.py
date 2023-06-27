"""Computational grid for SWAN."""
import logging
from pydantic import Field, root_validator, constr
from typing import Literal, Optional
from abc import ABC, abstractmethod

from rompy.swan.components.base import BaseComponent
from rompy.swan.subcomponents.spectrum import SPECTRUM
from rompy.swan.subcomponents.readgrid import READCOORD


logger = logging.getLogger(__name__)


class CGRID(BaseComponent, ABC):
    """SWAN computational grid base component."""

    model_type: Literal["cgrid"] = Field(
        default="cgrid", description="Model type discriminator"
    )
    spectrum: SPECTRUM = Field(description="Spectrum subcomponent")

    @abstractmethod
    def cmd(self):
        pass


class REGULAR(CGRID):
    """SWAN regular computational grid.

    `CGRID REGULAR [xpc] [ypc] [alpc] [xlenc] [ylenc] [mxc] [myc] CIRCLE|SECTOR ([dir1] [dir2]) [mdc] [flow] [fhigh] [msc]`

    """

    model_type: Literal["regular"] = Field(
        default="regular", description="Model type discriminator"
    )
    xpc: float = Field(
        default=0.0,
        description=(
            "Geographic location of the origin of the computational grid in the "
            "problem coordinate system (x-coordinate, in m)."
        ),
    )
    ypc: float = Field(
        default=0.0,
        description=(
            "Geographic location of the origin of the computational grid in the "
            "problem coordinate system (y-coordinate, in m)."
        ),
    )
    alpc: float = Field(
        default=0.0,
        description=(
            "Direction of the positive x-axis of the computational grid (in degrees, "
            "Cartesian convention). In 1D-mode, `alpc` should be equal to the "
            "direction `alpinp`."
        ),
    )
    xlenc: float = Field(
        description=(
            "Length of the computational grid in x-direction (in m). In case of "
            "spherical coordinates `xlenc` is in degrees."
        ),
    )
    ylenc: float = Field(
        description=(
            "Length of the computational grid in y-direction (in m). In 1D-mode, "
            "`ylenc` should be 0. In case of spherical coordinates `ylenc` is in "
            "degrees."
        ),
    )
    mxc: int = Field(
        description=(
            "Number of meshes in computational grid in x-direction (this number is "
            "one less than the number of grid points in this domain)."
        ),
    )
    myc: int = Field(
        description=(
            "Number of meshes in computational grid in y-direction (this number is "
            "one less than the number of grid points in this domain). In 1D-mode, "
            "`myc` should be 0."
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

    `CGRID CURVILINEAR [mxc] [myc] (EXCEPTION [xexc] [yexc]) CIRCLE|SECTOR ([dir1] [dir2]) [mdc] [flow] [fhigh] [msc]`
    `READGRID COORDINATES [fac] 'fname' [idla] [nhedf] [nhedvec] FREE|FORMAT ('form'|[idfm])`

    """

    model_type: Literal["curvilinear"] = Field(
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
        description=(
            "the value which the user uses to indicate that a grid point is to be "
            "ignored in the computations (this value is provided by the user at the "
            "location of the x-coordinate considered in the file of the "
            "x-coordinates, see command READGRID COOR)."
        ),
    )
    yexc: Optional[float] = Field(
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

    @root_validator
    def check_exception_definition(cls, values: dict) -> dict:
        """Check exception values are prescribed correctly."""
        xexc = values.get("xexc")
        yexc = values.get("yexc")
        if (xexc is None and yexc is not None) or (yexc is None and xexc is not None):
            raise ValueError("xexc and yexc must be specified together")
        return values

    @root_validator
    def check_format_definition(cls, values: dict) -> dict:
        """Check the arguments specifying the file format are specified correctly."""
        format = values.get("format")
        form = values.get("form")
        idfm = values.get("idfm")
        if format == "free" and any([form, idfm]):
            logger.warn(f"FREE format specified, ignoring form={form} idfm={idfm}")
        elif format == "unformatted" and any([form, idfm]):
            logger.warn(
                f"UNFORMATTED format specified, ignoring form={form} idfm={idfm}"
            )
        elif format == "fixed" and not any([form, idfm]):
            raise ValueError(
                "FIXED format requires one of form or idfm to be specified"
            )
        elif format == "fixed" and all([form, idfm]):
            raise ValueError("FIXED format accepts only one of form or idfm")
        return values

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
        repr += f"\n{self.readcoord.render()}"
        return repr


class UNSTRUCTURED(CGRID):
    """SWAN unstructured computational grid.

    `CGRID UNSTRUCTURED CIRCLE|SECTOR ([dir1] [dir2]) [mdc] [flow] [fhigh] [msc]`
    `READGRID UNSTRUCTURED [grid_type] ('fname')`

    """

    model_type: Literal["unstructured"] = Field(
        default="unstructured", description="Model type discriminator"
    )
    grid_type: Literal["adcirc", "triangle", "easymesh"] = Field(
        default="adcirc",
        description="Unstructured grid type",
    )
    fname: Optional[constr(max_length=80)] = Field(
        description="Name of the file containing the unstructured grid",
    )

    @root_validator
    def check_fname_required(cls, values: dict) -> dict:
        """Check that fname needs to be provided."""
        grid_type = values.get("grid_type")
        fname = values.get("fname")
        if grid_type == "adcirc" and fname is not None:
            raise ValueError("fname must not be specified for ADCIRC grid")
        elif grid_type != "adcirc" and fname is None:
            raise ValueError(f"fname must be specified for {grid_type} grid")
        return values

    def cmd(self) -> str:
        repr = f"CGRID UNSTRUCTURED {self.spectrum.cmd()}"
        repr += f"\nREADGRID UNSTRUCTURED {self.grid_type.upper()}"
        if self.grid_type in ["triangle", "easymesh"]:
            repr += f" fname='{self.fname}'"
        return repr
