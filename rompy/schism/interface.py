"""SCHISM interface objects.

These objects are used to define the interface between rompy functional objects
and SCHISM namelist objects.

"""

import logging
from pathlib import Path
from typing import Any, Literal, Optional, Union

from pydantic import Field, ValidationInfo, field_validator, model_validator

from rompy.core.types import RompyBaseModel
from rompy.core.time import TimeRange
from rompy.schism.data import SCHISMData
from rompy.schism.grid import SCHISMGrid
from rompy.schism.namelists import NML

logger = logging.getLogger(__name__)


class DataInterface(RompyBaseModel):
    """SCHISM forcing data interface.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.schism.interface import DataInterface

    """

    model_type: Literal["schism_data_interface", "SCHISM_DATA_INTERFACE"] = Field(
        default="schism_data_interface", description="Model type discriminator"
    )
    data: SCHISMData = Field(..., description="Schism data")
    nml: NML = Field(..., description="Schism namelist")

    def get(self, staging_dir: Path, grid: SCHISMGrid, period: TimeRange):
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


class TimeInterface(RompyBaseModel):
    """Base interface to pass time to group components.

    This class is used to set consistent time parameters in a group component by
    redefining existing `times` component attribute based on the `period` field.

    """

    nml: Union[NML, None] = Field(..., description="Namelist to set times to")
    period: TimeRange = Field(..., description="Time period to write the output over")

    def __call__(self):
        """Update the namelists."""
        update = {
            "param": {
                "opt": {
                    "start_year": self.period.start.year,
                    "start_month": self.period.start.month,
                    "start_day": self.period.start.day,
                    "start_hour": self.period.start.hour,
                    "rnday": self.period.duration.total_seconds() / 86400,
                }
            }
        }
        if self.nml is None:
            logger.warning("No namelist to update, creating param.opt")
            return NML(**update)

        date_format = "%Y-%m-%d %H:%M:%S"
        if hasattr(self.nml, "WWMINPUT"):
            if hasattr(self.nml.wwminput, "proc"):
                # TODO these are currently all the same, but they could be different
                update.update(
                    {
                        "wwminput": {
                            "PROC": {
                                "BEGTC": self.period.start.strftime(date_format),
                                "ENDTC": self.period.end.strftime(date_format),
                            },
                            "WIND": {
                                "BEGTC": self.period.start.strftime(date_format),
                                "ENDTC": self.period.end.strftime(date_format),
                            },
                            "CURR": {
                                "BEGTC": self.period.start.strftime(date_format),
                                "ENDTC": self.period.end.strftime(date_format),
                            },
                            "WALV": {
                                "BEGTC": self.period.start.strftime(date_format),
                                "ENDTC": self.period.end.strftime(date_format),
                            },
                            "STATION": {
                                "BEGTC": self.period.start.strftime(date_format),
                                "ENDTC": self.period.end.strftime(date_format),
                            },
                            "HOTFILE": {
                                "BEGTC": self.period.start.strftime(date_format),
                                "ENDTC": self.period.end.strftime(date_format),
                            },
                        }
                    }
                )
        return self.nml.model_copy(update=update)
