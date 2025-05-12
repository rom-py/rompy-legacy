from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional, Set, Union

import isodate
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)
from pydantic.json_schema import core_schema

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
        None, description="The start date of the time range", examples=["2020-01-01"]
    )
    end: Optional[datetime] = Field(
        None, description="The end date of the time range", examples=["2020-01-02"]
    )
    duration: Optional[Union[str, timedelta]] = Field(
        None, description="The duration of the time range", examples=["1d"]
    )
    interval: Optional[Union[str, timedelta]] = Field(
        "1h", description="The frequency of the time range", examples=["1h", "'1h'"]
    )
    include_end: bool = Field(
        True, description="Determines if the end date should be included in the range"
    )
    model_config = ConfigDict(validate_default=True)

    @field_validator("interval", "duration", mode="before")
    @classmethod
    def valid_duration_interval(cls, v):
        if v is None:
            return v
        if isinstance(v, timedelta):
            return v
        elif isinstance(v, str):
            try:
                return isodate.parse_duration(v)
            except isodate.ISO8601Error:
                if v[-1] not in time_units:
                    raise ValueError(
                        "Invalid duration unit. Must be one isoformat or one of: h, m, s, d, w, y"
                    )
                time_delta_unit = v[-1]
                time_delta_value = int(v[:-1])
                return timedelta(**{time_units[time_delta_unit]: time_delta_value})

    @field_validator("start", "end", mode="before")
    @classmethod
    def validate_start_end(cls, v):
        if v is None:
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
                return datetime.strptime(v, fmt)
            except ValueError:
                continue
        return v

    @model_validator(mode="before")
    @classmethod
    def validate_start_end_duration(cls, data: Any) -> Any:
        start, end, duration = data.get("start"), data.get("end"), data.get("duration")
        if start and not (end or duration):
            raise ValueError("start provided, must provide either end or duration")
        if end and not (start or duration):
            raise ValueError("end provided, must provide either start or duration")
        if duration and not (start or end):
            raise ValueError("duration provided, must provide either start or end")
        if not (start or end or duration):
            raise ValueError("Must provide two of start, end, duration")
        if start and end and duration:
            raise ValueError("Must provide only two of start, end, duration")
        return data

    @model_validator(mode="after")
    def parse_start_end_duration(self) -> "TimeRange":
        if self.start and self.end and not self.duration:
            self.duration = self.end - self.start
        elif self.start and self.duration and not self.end:
            self.end = self.start + self.duration
        elif self.end and self.duration and not self.start:
            self.start = self.end - self.duration
        return self

    model_config = ConfigDict(validate_default=True)

    @model_serializer
    def serialize_model(self) -> dict:
        # This replaces default serialization for the whole model
        # It works for both direct serialization and nested serialization
        result = {}
        for key, value in self.__dict__.items():
            # Skip 'duration' if both start and end are present
            if key == "duration" and self.start and self.end:
                continue
            # Include all other fields
            result[key] = value
        return result

    @property
    def date_range(self) -> list[datetime]:
        if not self.start or not self.end or not self.interval:
            return []
        start, end = self.start, self.end
        step_size = (
            self.interval
            if isinstance(self.interval, timedelta)
            else timedelta(**{time_units[self.interval[-1]]: int(self.interval[:-1])})
        )
        date_range = []
        while start < end:
            date_range.append(start)
            start += step_size
        if self.include_end and date_range and date_range[-1] != end:
            date_range.append(end)
        return date_range

    def contains(self, date: datetime) -> bool:
        return self.start <= date <= self.end

    def contains_range(self, date_range: "TimeRange") -> bool:
        return self.contains(date_range.start) and self.contains(date_range.end)

    def common_times(self, date_range: "TimeRange") -> list[datetime]:
        return [date for date in self.date_range if date_range.contains(date)]

    def format_duration(self, duration: timedelta) -> str:
        """Format a timedelta object as a human-readable string.
        
        This method formats a timedelta in a way that's suitable for display
        in logs and other output.
        
        Args:
            duration: The timedelta object to format
            
        Returns:
            A formatted string representation of the duration
        """
        if not duration:
            return "None"
            
        days = duration.days
        seconds = duration.seconds
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        parts = []
        if days > 0:
            parts.append(f"{days} day{'s' if days != 1 else ''}")
        if hours > 0:
            parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
        if minutes > 0:
            parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
        if seconds > 0 or not parts:
            parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")
            
        return ", ".join(parts)
    
    def __str__(self):
        return (
            f"\n\tStart: {self.start}\n"
            f"\tEnd: {self.end}\n"
            f"\tDuration: {self.format_duration(self.duration)}\n"
            f"\tInterval: {str(self.interval)}\n"
            f"\tInclude End: {self.include_end}\n"
        )
