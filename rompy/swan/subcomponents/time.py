"""Time subcomponents."""
import logging
from datetime import datetime, timedelta
from typing import Literal, Optional
from pydantic import Field, validator, root_validator

from rompy.swan.subcomponents.base import BaseSubComponent


logger = logging.getLogger(__name__)


TIME_FORMATS = {
    1: "%Y%m%d.%H%M%S",
    2: "'%d-%b-%y %H:%M:%S'",
    3: "%m/%d/%y.%H:%M:%S",
    4: "%H:%M:%S",
    5: "%y/%m/%d %H:%M:%S'",
    6: "%y%m%d%H%M",
}


class NONSTATIONARY(BaseSubComponent):
    """SWAN Nonstationary specification.

    `NONSTATIONARY [tbeg] [delt] SEC|MIN|HR|DAY [tend]`

    Notes
    -----
    Format to render time specification:

    * 1: ISO-notation 19870530.153000
    * 2: (as in HP compiler) '30-May-87 15:30:00'
    * 3: (as in Lahey compiler) 05/30/87.15:30:00
    * 4: 15:30:00
    * 5: 87/05/30 15:30:00'
    * 6: as in WAM 8705301530

    The datetime types can be specified as:

    * existing datetime object
    * int or float, assumed as Unix time, i.e. seconds (if >= -2e10 or <= 2e10) or
      milliseconds (if < -2e10 or > 2e10) since 1 January 1970.
    * ISO 8601 time string.

    The timedelta type can be specified as:

    * existing timedelta object
    * int or float, assumed as seconds
    * ISO 8601 duration string, following formats work:

      - `[-][DD ][HH:MM]SS[.ffffff]`
      - `[±]P[DD]DT[HH]H[MM]M[SS]S` (ISO 8601 format for timedelta)

    """

    model_type: Literal["nonstationary"] = Field(
        default="nonstationary", description="Model type discriminator"
    )
    tbeg: datetime = Field(
        description="Begin time of the first field of the variable"
    )
    delt: timedelta = Field(
        description="Time interval between fields"
    )
    tend: datetime = Field(
        description="End time of the last field of the variable"
    )
    tfmt: Literal[1, 2, 3, 4, 5, 6] = Field(
        default=1,
        description="Format to render time specification",
    )
    deltfmt: Literal["sec", "min", "hr", "day"] = Field(
        default="sec",
        description="Format to render time interval specification",
    )
    suffix: Optional[str] = Field(
        description="Suffix to append to the variable name when rendering"
    )

    @root_validator
    def set_time_format(cls, values):
        """Set the time format."""
        values["tfmt"] = TIME_FORMATS[values.get("tfmt")]
        return values

    @property
    def delt_string(self):
        """Return the timedelta as a string."""
        dt = self.delt.total_seconds()
        if self.deltfmt == "min":
            dt /= 60
        elif self.deltfmt == "hr":
            dt /= 3600
        elif self.deltfmt == "day":
            dt /= 86400
        return f"{dt} {self.deltfmt.upper()}"

    def cmd(self) -> str:
        """Render NONSTATIONARY string."""
        repr = "NONSTATIONARY"
        if self.suffix is not None:
            repr += f" tbeg{self.suffix}={self.tbeg.strftime(self.tfmt)}"
            repr += f" delt{self.suffix}={self.delt_string}"
            repr += f" tend{self.suffix}={self.tend.strftime(self.tfmt)}"
        else:
            repr += f" {self.tbeg.strftime(self.tfmt)}"
            repr += f" {self.delt_string}"
            repr += f" {self.tend.strftime(self.tfmt)}"
        return repr
