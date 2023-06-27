from datetime import datetime, timedelta
from typing import List, Optional, Union

from dateutil.relativedelta import relativedelta
from pydantic import BaseModel, Field, root_validator, validator

from rompy.core.types import RompyBaseModel

time_units = {
    "h": "hours",
    "d": "days",
    "w": "weeks",
    "m": "minutes",
    "s": "seconds",
    "y": "years",
    "H": "hours",
    "D": "days",
    "W": "weeks",
    "M": "minutes",
    "S": "seconds",
    "Y": "years",
}


class TimeRange(BaseModel):
    """
    A time range object

    Examples
    --------
    >>> from rompy import TimeRange
    >>> tr = TimeRange(start="2020-01-01", end="2020-01-02")
    >>> tr
    TimeRange(start=datetime.datetime(2020, 1, 1, 0, 0), end=datetime.datetime(2020, 1, 2, 0, 0), duration=None, interval=None, include_end=True)
    >>> tr = TimeRange(start="2020-01-01", duration="1d")
    >>> tr
    TimeRange(start=datetime.datetime(2020, 1, 1, 0, 0), end=datetime.datetime(2020, 1, 2, 0, 0), duration=timedelta(days=1), interval=None, include_end=True)
    >>> tr = TimeRange(start="2020-01-01", duration="1d", interval="1h")
    >>> tr
    TimeRange(start=datetime.datetime(2020, 1, 1, 0, 0), end=None, duration=timedelta(days=1), interval=timedelta(hours=1), include_end=True)
    """

    start: Optional[datetime] = Field(
        None,
        description="The start date of the time range",
        example="2020-01-01",
    )
    end: Optional[datetime] = Field(
        None,
        description="The end date of the time range",
        example="2020-01-02",
    )
    duration: Optional[Union[str, timedelta]] = Field(
        None,
        description="The duration of the time range",
        example="1d",
    )
    interval: Optional[Union[str, timedelta]] = Field(
        "1h",
        description="The frequency of the time range",
        example="1h or '1h'",
    )
    include_end: bool = Field(
        True,
        description="Determines if the end date should be included in the range",
    )

    class Config:
        validate_all = True

    @validator("interval", "duration", pre=True)
    def valid_duration_interval(cls, v):
        if v == None:
            return v
        if isinstance(v, timedelta):
            return v
        elif isinstance(v, str):
            if v[-1] not in time_units:
                raise ValueError(
                    "Invalid duration unit. Must be one of: h, m, s, d, w, y"
                )
            time_delta_unit = v[-1]
            time_delta_value = int(v[:-1])
            return timedelta(**{time_units[time_delta_unit]: time_delta_value})

    @validator("start", "end", pre=True)
    def validate_start_end(cls, v):
        if v == None:
            return v
        if isinstance(v, datetime):
            return v
        for fmt in [
            "%Y%m%d",
            "%Y%m%dT%H",
            "%Y%m%dT%H%M",
            "%Y%m%dT%H%M%S",
            "%Y%m%d.%H",
            "%Y%m%d.%H%M",
            "%Y%m%d.%H%M%S",
            "%Y-%m-%d",
            "%Y-%m-%dT%H",
            "%Y-%m-%dT%H%M",
        ]:
            try:
                ret = datetime.strptime(v, fmt)
                return ret
            except ValueError:
                pass
        return v

    @root_validator
    def validate_start_end_duration(cls, values):
        # check two out of start, end, duration are provided
        if values.get("start"):
            if not values.get("end") and not values.get("duration"):
                raise ValueError(
                    "start provided, must provide either end or duration")
        if values.get("end"):
            if not values.get("start") and not values.get("duration"):
                raise ValueError(
                    "end provided, must provide either start or duration")
        if values.get("duration"):
            if not values.get("start") and not values.get("end"):
                raise ValueError(
                    "duration provided, must provide either start or end")
        if (
            values.get("start") is None
            and values.get("end") is None
            and values.get("duration") is None
        ):
            raise ValueError("Must provide two of start, end, duration")
        if (
            values.get("start") is not None
            and values.get("end") is not None
            and values.get("duration") is not None
        ):
            raise ValueError("Must provide only two of start, end, duration")
        if values.get("start") is not None and values.get("end") is not None:
            values["duration"] = values["end"] - values["start"]
        if values.get("start") is not None and values.get("duration") is not None:
            values["end"] = values["start"] + values["duration"]
        if values.get("end") is not None and values.get("duration") is not None:
            values["start"] = values["end"] - values["duration"]
        return values

    @property
    def date_range(self) -> List[datetime]:
        start = self.start
        end = self.end
        date_range = []
        while start < end:
            date_range.append(start)
            start += self.interval
            if start + self.interval > end:
                if self.include_end:
                    date_range.append(end)
                break
        return date_range

    def contains(self, date: datetime) -> bool:
        return self.start <= date <= self.end

    def contains_range(self, date_range: "TimeRange") -> bool:
        return self.contains(date_range.start) and self.contains(date_range.end)

    def common_times(self, date_range: "TimeRange") -> List[datetime]:
        common_times = []
        for date in self.date_range:
            if date_range.contains(date):
                common_times.append(date)
        return common_times

    def __str__(self):
        ret = f"\n\tStart: {self.start}\n"
        ret += f"\tEnd: {self.end}\n"
        ret += f"\tDuration: {self.duration}\n"
        ret += f"\tInterval: {self.interval}\n"
        ret += f"\tInclude End: {self.include_end}\n"
        return ret
