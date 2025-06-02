"""
SWAN Subcomponents Base Module

This module provides the base classes for SWAN subcomponents in the ROMPY framework.
"""

# Standard library imports
from typing import Literal, Optional
from abc import ABC

# Third-party imports
from pydantic import ConfigDict, Field, model_validator

# Local imports
from rompy.core.logging import get_logger
from rompy.core.types import RompyBaseModel

# Initialize the logger
logger = get_logger(__name__)


class BaseSubComponent(RompyBaseModel, ABC):
    """Base class for SWAN sub-components.

    This class is not intended to be used directly, but to be subclassed by other
    SWAN sub-components to implement the following common behaviour:

    * Define a `render()` method to render a CMD string from the subcomponent
    * Forbid extra arguments so only implemented fields must be specified

    """

    model_type: Literal["subcomponent"] = Field(description="Model type discriminator")
    model_config = ConfigDict(extra="forbid")

    def cmd(self) -> str:
        return self.model_type.upper()

    def render(self) -> str:
        """Render the sub-component to a string."""
        return self.cmd()


class XY(BaseSubComponent):
    """Points in problem coordinates.

    .. code-block:: text

        < [x] [y] >

    Note
    ----
    Coordinates should be given in m when Cartesian coordinates are used or degrees
    when Spherical coordinates are used (see command `COORD`).

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.base import XY
        points = XY(
            x=[172, 172, 172, 172.5, 173],
            y=[-41, -40.5, -40, -40, -40],
            fmt="0.2f",
        )
        print(points.render())

    """

    model_type: Literal["xy", "XY"] = Field(
        default="xy",
        description="Model type discriminator",
    )
    x: list[float] = Field(description="Problem x-coordinate values")
    y: list[float] = Field(description="Problem y-coordinate values")
    fmt: str = Field(
        default="0.8f",
        description="The format to render floats values",
    )

    @model_validator(mode="after")
    def validate_size(self) -> "XY":
        if len(self.x) != len(self.y):
            raise ValueError(f"x and y must be the same size")
        return self

    @property
    def size(self):
        return len(self.x)

    def cmd(self) -> str:
        repr = ""
        for x, y in zip(self.x, self.y):
            repr += f"\n{x:{self.fmt}} {y:{self.fmt}}"
        return repr + "\n"


class IJ(BaseSubComponent):
    """Points in grid indices coordinates.

    .. code-block:: text

        < [x] [y] >

    Note
    ----
    Coordinates should be given in m when Cartesian coordinates are used or degrees
    when Spherical coordinates are used (see command `COORD`).

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.base import IJ
        points = IJ(i=[0, 0, 5], j=[0, 19, 19])
        print(points.render())

    """

    model_type: Literal["ij", "IJ"] = Field(
        default="ij",
        description="Model type discriminator",
    )
    i: list[int] = Field(description="i-index values")
    j: list[int] = Field(description="j-index values")

    @model_validator(mode="after")
    def validate_size(self) -> "IJ":
        if len(self.i) != len(self.j):
            raise ValueError(f"i and j must be the same size")
        return self

    @property
    def size(self):
        return len(self.i)

    def cmd(self) -> str:
        repr = ""
        for i, j in zip(self.i, self.j):
            repr += f"\ni={i} j={j}"
        return repr + "\n"
