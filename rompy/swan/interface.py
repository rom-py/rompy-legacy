"""
SWAN Interface Module

This module provides interface classes for SWAN model components in the ROMPY framework.
"""

# Standard library imports
from pathlib import Path
from typing import Any, Literal, Optional, Union

# Third-party imports
from pydantic import Field, ValidationInfo, field_validator, model_validator

# Local imports
from rompy.core.types import RompyBaseModel
from rompy.core.time import TimeRange
from rompy.core.logging import get_logger
from rompy.swan.boundary import Boundnest1, BoundspecSegmentXY, BoundspecSide
from rompy.swan.data import SwanDataGrid
from rompy.swan.grid import SwanGrid
from rompy.swan.subcomponents.time import NONSTATIONARY, STATIONARY, TimeRangeOpen

# Initialize the logger
logger = get_logger(__name__)


class DataInterface(RompyBaseModel):
    """SWAN forcing data interface.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.interface import DataInterface

    """

    model_type: Literal["data_interface", "DATA_INTERFACE"] = Field(
        default="data_interface", description="Model type discriminator"
    )
    bottom: Optional[SwanDataGrid] = Field(default=None, description="Bathymetry data")
    input: list[SwanDataGrid] = Field(default=[], description="Input grid data")

    @field_validator("input")
    @classmethod
    def ensure_unique_var(
        cls, input: list[SwanDataGrid], info: ValidationInfo
    ) -> list[SwanDataGrid]:
        """Ensure that each input var is unique."""
        vars = []
        if info.data["bottom"] is not None:
            vars.append(info.data["bottom"].var)
        vars.extend([inp.var for inp in input])
        if len(vars) != len(set(vars)):
            raise ValueError("Each var must be unique in input")
        return input

    def get(self, staging_dir: Path, grid: SwanGrid, period: TimeRange):
        inputs = []
        if self.bottom is not None:
            inputs.append(self.bottom)
        inputs.extend(self.input)
        cmds = []
        for input in inputs:
            cmds.append(input.get(destdir=staging_dir, grid=grid, time=period))
        return "\n".join(cmds)

    def render(self, *args, **kwargs):
        """Make this class consistent with the components API."""
        return self.get(*args, **kwargs)


class BoundaryInterface(RompyBaseModel):
    """SWAN forcing boundary interface.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.interface import BoundaryInterface

    """

    model_type: Literal["boundary_interface", "BOUNDARY_INTERFACE"] = Field(
        default="boundary_interface", description="Model type discriminator"
    )
    kind: Union[Boundnest1, BoundspecSide, BoundspecSegmentXY] = Field(
        default=None, description="Boundary data object"
    )

    def get(self, staging_dir: Path, grid: SwanGrid, period: TimeRange):
        filename, cmd = self.kind.get(destdir=staging_dir, grid=grid, time=period)
        logger.info(f"Generating boundary file: {filename}")
        return cmd

    def render(self, *args, **kwargs):
        """Make this class consistent with the components API."""
        return self.get(*args, **kwargs)


class TimeInterface(RompyBaseModel):
    """Base interface to pass time to group components.

    This class is used to set consistent time parameters in a group component by
    redefining existing `times` component attribute based on the `period` field.

    """

    model_type: Literal["interface", "INTERFACE"] = Field(
        default="interface", description="Model type discriminator"
    )
    group: Any = Field(description="Group component to set times to")
    period: TimeRange = Field(description="Time period to write the output over")


class OutputInterface(TimeInterface):
    """Output group component with consistent times."""

    model_type: Literal["outputinterface", "OUTPUTINTERFACE"] = Field(
        default="outputinterface", description="Model type discriminator"
    )

    @model_validator(mode="after")
    def time_interface(self) -> "OutputInterface":
        """Set the time parameter for all WRITE components."""
        for component in self.group._write_fields:
            obj = getattr(self.group, component)
            if obj is not None:
                times = obj.times or TimeRangeOpen()
                obj.times = self._timerange(times.tfmt, times.dfmt, obj.suffix)

    def _timerange(self, tfmt: int, dfmt: str, suffix: str) -> TimeRangeOpen:
        """Convert generic TimeRange into the Swan TimeRangeOpen subcomponent."""
        return TimeRangeOpen(
            tbeg=self.period.start,
            delt=self.period.interval,
            tfmt=tfmt,
            dfmt=dfmt,
            suffix=suffix,
        )


class LockupInterface(TimeInterface):
    """Lockup group component with consistent times."""

    model_type: Literal["lockupinterface", "LOCKUPINTERFACE"] = Field(
        default="lockupinterface", description="Model type discriminator"
    )

    def _nonstationary(self, tfmt: str, dfmt: str) -> NONSTATIONARY:
        return NONSTATIONARY(
            tbeg=self.period.start,
            tend=self.period.end,
            delt=self.period.interval,
            tfmt=tfmt,
            dfmt=dfmt,
            suffix="c",
        )

    def _stationary(self, tfmt: str) -> STATIONARY:
        return STATIONARY(time=self.period.start, tfmt=tfmt)

    @model_validator(mode="after")
    def time_interface(self) -> "LockupInterface":
        """Set the time parameter for COMPUTE components."""
        times = self.group.compute.times or NONSTATIONARY()
        if isinstance(times, NONSTATIONARY):
            times = self._nonstationary(times.tfmt, times.dfmt)
        elif isinstance(times, STATIONARY):
            times = self._stationary(times.tfmt)
        else:
            raise ValueError(f"Unknown time type {type(times)}")
        self.group.compute.times = times
