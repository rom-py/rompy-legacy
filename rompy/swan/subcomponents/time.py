"""
SWAN Time Subcomponents

This module contains subcomponents for handling time specifications in SWAN,
including time ranges, intervals, and time format conversions.
"""

from datetime import datetime, timedelta
from typing import Literal, Optional, Union

import pandas as pd
from pydantic import Field, field_validator, model_validator

from rompy.logging import get_logger
from rompy.swan.subcomponents.base import BaseSubComponent

logger = get_logger(__name__)

DEFAULT_TIME = datetime(1970, 1, 1, 0, 0, 0)
DEFAULT_TEND = DEFAULT_TIME + timedelta(days=1)
DEFAULT_DELT = timedelta(hours=1)
TIME_FORMAT = {
    1: "%Y%m%d.%H%M%S",
    2: "'%d-%b-%y %H:%M:%S'",
    3: "%m/%d/%y.%H:%M:%S",
    4: "%H:%M:%S",
    5: "%y/%m/%d %H:%M:%S'",
    6: "%y%m%d%H%M",
}


class Time(BaseSubComponent):
    """Time specification in SWAN.

    .. code-block:: text

        [time]

    Time is rendered in one of the following formats:

    * 1: ISO-notation 19870530.153000
    * 2: (as in HP compiler) '30-May-87 15:30:00'
    * 3: (as in Lahey compiler) 05/30/87.15:30:00
    * 4: 15:30:00
    * 5: 87/05/30 15:30:00'
    * 6: as in WAM 8705301530

    Note
    ----
    The `time` field can be specified as:

    * existing datetime object
    * int or float, assumed as Unix time, i.e. seconds (if >= -2e10 or <= 2e10) or
      milliseconds (if < -2e10 or > 2e10) since 1 January 1970.
    * ISO 8601 time string.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.time import Time
        from datetime import datetime
        time = Time(time=datetime(1990, 1, 1))
        print(time.render())
        time = Time(time="2012-01-01T00:00:00", tfmt=2)
        print(time.render())

    """

    model_type: Literal["time", "Time", "TIME"] = Field(
        default="time", description="Model type discriminator"
    )
    time: datetime = Field(description="Datetime specification")
    tfmt: Union[Literal[1, 2, 3, 4, 5, 6], str] = Field(
        default=1,
        description="Format to render time specification",
        validate_default=True,
    )

    @field_validator("tfmt")
    @classmethod
    def set_time_format(cls, v: int | str) -> str:
        """Set the time format to render."""
        if isinstance(v, str):
            return v
        return TIME_FORMAT[v]

    def cmd(self) -> str:
        """Render subcomponent cmd."""
        return f"{self.time.strftime(self.tfmt)}"


class Delt(BaseSubComponent):
    """Time interval specification in SWAN.

    .. code-block:: text

        [delt] SEC|MIN|HR|DAY

    Note
    ----
    The `tdelta` field can be specified as:

    * existing timedelta object
    * int or float, assumed as seconds
    * ISO 8601 duration string, following formats work:

      * `[-][DD ][HH:MM]SS[.ffffff]`
      * `[±]P[DD]DT[HH]H[MM]M[SS]S` (ISO 8601 format for timedelta)

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.time import Delt
        from datetime import timedelta
        delt = Delt(delt=timedelta(minutes=30))
        print(delt.render())
        delt = Delt(delt="PT1H", dfmt="hr")
        print(delt.render())

    """

    model_type: Literal["delt"] = Field(
        default="delt", description="Model type discriminator"
    )
    delt: timedelta = Field(description="Time interval")
    dfmt: Literal["sec", "min", "hr", "day"] = Field(
        default="sec",
        description="Format to render time interval specification",
    )

    @property
    def delt_float(self):
        delt_scaling = {"sec": 1, "min": 60, "hr": 3600, "day": 86400}
        return self.delt.total_seconds() / delt_scaling[self.dfmt]

    def cmd(self) -> str:
        """Render subcomponent cmd."""
        return f"{self.delt_float} {self.dfmt.upper()}"


class TimeRangeOpen(BaseSubComponent):
    """Regular times with an open boundary.

    .. code-block:: text

        [tbeg] [delt] SEC|MIN|HR|DAY

    Time is rendered in one of the following formats:

    * 1: ISO-notation 19870530.153000
    * 2: (as in HP compiler) '30-May-87 15:30:00'
    * 3: (as in Lahey compiler) 05/30/87.15:30:00
    * 4: 15:30:00
    * 5: 87/05/30 15:30:00'
    * 6: as in WAM 8705301530

    Note
    ----
    The `tbeg` field can be specified as:

    * existing datetime object
    * int or float, assumed as Unix time, i.e. seconds (if >= -2e10 or <= 2e10) or
      milliseconds (if < -2e10 or > 2e10) since 1 January 1970.
    * ISO 8601 time string.

    Note
    ----
    The `tdelta` field can be specified as:

    * existing timedelta object
    * int or float, assumed as seconds
    * ISO 8601 duration string, following formats work:

        * `[-][DD ][HH:MM]SS[.ffffff]`
        * `[±]P[DD]DT[HH]H[MM]M[SS]S` (ISO 8601 format for timedelta)

    Note
    ----
    Default values for the time specification fields are provided for the case where
    the user wants to set times dynamically after instantiating this subcomponent.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.time import TimeRangeOpen
        from datetime import datetime, timedelta
        times = TimeRangeOpen(
            tbeg=datetime(1990, 1, 1), delt=timedelta(minutes=30), dfmt="min"
        )
        print(times.render())
        times = TimeRangeOpen(
            tbeg="2012-01-01T00:00:00", delt="PT1H", tfmt=2, dfmt="hr", suffix="blk"
        )
        print(times.render())

    """

    model_type: Literal["open", "OPEN"] = Field(
        default="open", description="Model type discriminator"
    )
    tbeg: datetime = Field(default=DEFAULT_TIME, description="Start time")
    delt: timedelta = Field(default=DEFAULT_DELT, description="Time interval")
    tfmt: Union[Literal[1, 2, 3, 4, 5, 6], str] = Field(
        default=1,
        description="Format to render time specification",
    )
    dfmt: Literal["sec", "min", "hr", "day"] = Field(
        default="sec",
        description="Format to render time interval specification",
    )
    suffix: str = Field(
        default="",
        description="Suffix to prepend to argument names when rendering",
    )

    def cmd(self) -> str:
        """Render subcomponent cmd."""
        repr = f"tbeg{self.suffix}={Time(time=self.tbeg, tfmt=self.tfmt).render()}"
        repr += f" delt{self.suffix}={Delt(delt=self.delt, dfmt=self.dfmt).render()}"
        return repr


class TimeRangeClosed(TimeRangeOpen):
    """Regular times with a closed boundary.

    .. code-block:: text

        [tbeg] [delt] SEC|MIN|HR|DAY [tend]

    Note
    ----
    Default values for the time specification fields are provided for the case where
    the user wants to set times dynamically after instantiating this subcomponent.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.time import TimeRangeClosed
        from datetime import datetime, timedelta
        times = TimeRangeClosed(
            tbeg=datetime(1990, 1, 1),
            tend=datetime(1990, 1, 7),
            delt=timedelta(minutes=30),
            dfmt="min",
        )
        print(times.render())
        times = TimeRangeClosed(
            tbeg="2012-01-01T00:00:00",
            tend="2012-02-01T00:00:00",
            delt="PT1H",
            tfmt=2,
            dfmt="hr",
            suffix="blk",
        )
        print(times.render())

    """

    model_type: Literal["closed", "CLOSED"] = Field(
        default="closed", description="Model type discriminator"
    )
    tend: datetime = Field(default=DEFAULT_TEND, description="End time")

    def __call__(self) -> list[Time]:
        """Returns the list of Time objects."""
        times = pd.date_range(start=self.tbeg, end=self.tend, freq=self.delt)
        return [time.to_pydatetime() for time in times]

    def __getitem__(self, index) -> datetime | list[datetime]:
        """Slicing from the times array."""
        return self.__call__()[index]

    def __len__(self):
        """Returns the length of the times array."""
        return len(self())

    def cmd(self) -> str:
        """Render subcomponent cmd."""
        repr = super().cmd()
        repr += f" tend{self.suffix}={Time(time=self.tend, tfmt=self.tfmt).render()}"
        return repr


class NONSTATIONARY(TimeRangeClosed):
    """Nonstationary time specification.

    .. code-block:: text

        NONSTATIONARY [tbeg] [delt] SEC|MIN|HR|DAY [tend]

    Note
    ----
    Default values for the time specification fields are provided for the case where
    the user wants to set times dynamically after instantiating this subcomponent.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.time import NONSTATIONARY
        nonstat = NONSTATIONARY(
            tbeg="2012-01-01T00:00:00",
            tend="2012-02-01T00:00:00",
            delt="PT1H",
            dfmt="hr",
        )
        print(nonstat.render())
        from datetime import datetime, timedelta
        nonstat = NONSTATIONARY(
            tbeg=datetime(1990, 1, 1),
            tend=datetime(1990, 1, 7),
            delt=timedelta(minutes=30),
            tfmt=1,
            dfmt="min",
            suffix="tbl",
        )
        print(nonstat.render())

    """

    model_type: Literal["nonstationary", "NONSTATIONARY"] = Field(
        default="nonstationary", description="Model type discriminator"
    )
    tbeg: datetime = Field(default=DEFAULT_TIME, description="Start time")
    tend: datetime = Field(default=DEFAULT_TEND, description="End time")
    delt: timedelta = Field(default=DEFAULT_DELT, description="Time interval")

    def cmd(self) -> str:
        """Render subcomponent cmd."""
        repr = f"NONSTATIONARY {super().cmd()}"
        return repr


class STATIONARY(BaseSubComponent):
    """Stationary time specification.

    .. code-block:: text

        STATIONARY [time]

    Note
    ----
    The field `time` is optional to allow for the case where the user wants to set the
    time dynamically after instantiating this component.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.time import STATIONARY
        stat = STATIONARY(time="2012-01-01T00:00:00")
        print(stat.render())

    """

    model_type: Literal["stationary", "STATIONARY"] = Field(
        default="stationary", description="Model type discriminator"
    )
    time: datetime = Field(default=DEFAULT_TIME, description="Stationary time")
    tfmt: Union[Literal[1, 2, 3, 4, 5, 6], str] = Field(
        default=1,
        description="Format to render time specification",
    )

    def __call__(self) -> list[Time]:
        """Returns the list of Time object for consistency with NONSTATIONARY."""
        return [self.time]

    def __getitem__(self, index) -> Time | list[Time]:
        """Slicing from the times array."""
        return self.__call__()[index]

    def __len__(self):
        """Returns the length of the times array."""
        return len(self())

    def cmd(self) -> str:
        """Render subcomponent cmd."""
        return f"STATIONARY time={Time(time=self.time, tfmt=self.tfmt).render()}"
