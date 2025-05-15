"""Rompy types."""

import json
from datetime import datetime
from typing import Any, Optional, Union

from pydantic import (BaseModel, ConfigDict, Field, field_validator,
                      model_validator)


class RompyBaseModel(BaseModel):
    # The config below prevents https://github.com/pydantic/pydantic/discussions/7121
    model_config = ConfigDict(protected_namespaces=(), extra="forbid")

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._original_inputs = data

    def dump_inputs_dict(self) -> dict:
        """Return the original inputs as a dictionary."""
        return self._original_inputs

    def dump_inputs_json(self) -> str:
        """Return the original inputs as a JSON string."""
        return json.dumps((self._original_inputs))
        
    def __str__(self) -> str:
        """Return a hierarchical string representation of the model.
        
        This generic implementation provides a consistent string formatting
        for all RompyBaseModel objects, handling nested models recursively.
        Classes can override this if they need custom string representations.
        """
        lines = []
        self._str_helper(lines, name=self.__class__.__name__, obj=self, level=0)
        return "\n".join(lines)
    
    def _format_value(self, obj: Any) -> Optional[str]:
        """Format a value for string representation.
        
        This method can be overridden by subclasses to customize how specific types
        are formatted in the string representation.
        
        Args:
            obj: The object to format
            
        Returns:
            A string representation of the object, or None to use default formatting
        """
        return None
    
    def _str_helper(self, lines: list, name: str, obj: Any, level: int) -> None:
        """Helper method to build a hierarchical string representation.
        
        Args:
            lines: List to append formatted string lines
            name: Name of the current object/field
            obj: The object to format
            level: Current indentation level
        """
        indent = "  " * level
        
        # Handle None values
        if obj is None:
            lines.append(f"{indent}{name}: None")
            return
        
        # Check if there's a custom formatter in the current class
        custom_format = self._format_value(obj)
        if custom_format is not None:
            if "\n" in custom_format:
                # For multi-line string representations
                lines.append(f"{indent}{name}:")
                for line in custom_format.split("\n"):
                    lines.append(f"{indent}  {line}")
            else:
                lines.append(f"{indent}{name}: {custom_format}")
            return
            
        # Check for objects with their own __str__ method (not inherited from object or base classes)
        # But don't use it for RompyBaseModel instances (use our hierarchical formatting instead)
        str_method = getattr(obj.__class__, "__str__", None)
        base_str_method = getattr(RompyBaseModel, "__str__", None)
        object_str_method = getattr(object, "__str__", None)
        
        if (not isinstance(obj, RompyBaseModel) and 
            str_method is not None and 
            str_method is not object_str_method):
            # Use the object's custom __str__ if it has one
            str_val = str(obj)
            if "\n" in str_val:
                # For multi-line string representations
                lines.append(f"{indent}{name}:")
                for line in str_val.split("\n"):
                    lines.append(f"{indent}  {line}")
            else:
                lines.append(f"{indent}{name}: {str_val}")
        elif isinstance(obj, RompyBaseModel):
            lines.append(f"{indent}{name}:")
            for field_name, field_value in obj.model_dump().items():
                if field_name.startswith("_"):
                    continue
                self._str_helper(lines, field_name, field_value, level + 1)
        elif isinstance(obj, dict):
            if not obj:
                lines.append(f"{indent}{name}: {{}}")
            else:
                lines.append(f"{indent}{name}:")
                for key, value in obj.items():
                    self._str_helper(lines, str(key), value, level + 1)
        elif isinstance(obj, (list, tuple)):
            if not obj:
                lines.append(f"{indent}{name}: []")
            else:
                lines.append(f"{indent}{name}:")
                for i, item in enumerate(obj):
                    self._str_helper(lines, f"[{i}]", item, level + 1)
        else:
            lines.append(f"{indent}{name}: {obj}")


class Latitude(BaseModel):
    """Latitude"""

    lat: float = Field(description="Latitude", ge=-90, le=90)

    @field_validator("lat")
    @classmethod
    def validate_lat(cls, v: float) -> float:
        if not (-90 <= v <= 90):
            raise ValueError("latitude must be between -90 and 90")
        return v

    def __repr__(self):
        return f"Latitude(lat={self.lat})"

    def __str__(self):
        return f"{self.lat}"

    def __eq__(self, other):
        return self.lat == other.lat

    def __hash__(self):
        return hash((self.lat,))


class Longitude(BaseModel):
    """Longitude"""

    lon: float = Field(description="Longitude", ge=-180, le=180)

    @field_validator("lon")
    @classmethod
    def validate_lon(cls, v: float) -> float:
        if not (-180 <= v <= 180):
            raise ValueError("longitude must be between -180 and 180")
        return v

    def __repr__(self):
        return f"Longitude(lon={self.lon})"

    def __str__(self):
        return f"{self.lon}"

    def __eq__(self, other):
        return self.lon == other.lon

    def __hash__(self):
        return hash((self.lon,))


class Coordinate(BaseModel):
    """Coordinate"""

    lon: float = Field(description="Longitude")
    lat: float = Field(description="Latitude")

    @field_validator("lon")
    @classmethod
    def validate_lon(cls, v: float) -> float:
        if not (-180 <= v <= 180):
            raise ValueError("longitude must be between -180 and 180")
        return v

    @field_validator("lat")
    @classmethod
    def validate_lat(cls, v: float) -> float:
        if not (-90 <= v <= 90):
            raise ValueError("latitude must be between -90 and 90")
        return v

    def __repr__(self):
        return f"Coordinate(lon={self.lon}, lat={self.lat})"

    def __str__(self):
        return f"({self.lon}, {self.lat})"

    def __eq__(self, other):
        return self.lon == other.lon and self.lat == other.lat

    def __hash__(self):
        return hash((self.lon, self.lat))


class Bbox(BaseModel):
    """Bounding box"""

    minlon: Longitude = Field(description="Minimum longitude")
    minlat: Latitude = Field(description="Minimum latitude")
    maxlon: Longitude = Field(description="Maximum longitude")
    maxlat: Latitude = Field(description="Maximum latitude")

    def __repr__(self):
        return f"Bbox(minlon={self.minlon}, minlat={self.minlat}, maxlon={self.maxlon}, maxlat={self.maxlat})"

    def __str__(self):
        return f"({self.minlon}, {self.minlat}, {self.maxlon}, {self.maxlat})"

    def __eq__(self, other):
        return (
            self.minlon == other.minlon
            and self.minlat == other.minlat
            and self.maxlon == other.maxlon
            and self.maxlat == other.maxlat
        )

    def __hash__(self):
        return hash((self.minlon, self.minlat, self.maxlon, self.maxlat))

    @model_validator(mode="after")
    def validate_coords(self) -> "Bbox":
        if self.minlon.lon >= self.maxlon.lon:
            raise ValueError("minlon must be less than maxlon")
        if self.minlat.lat >= self.maxlat.lat:
            raise ValueError("minlat must be less than maxlat")
        return self

    @property
    def width(self):
        return self.maxlon - self.minlon

    @property
    def height(self):
        return self.maxlat - self.minlat

    @property
    def center(self):
        return Coordinate(
            lon=(self.minlon + self.maxlon) / 2, lat=(self.minlat + self.maxlat) / 2
        )

    @property
    def area(self):
        return self.width * self.height

    def contains(self, other):
        if isinstance(other, Bbox):
            return (
                self.minlon <= other.minlon
                and self.minlat <= other.minlat
                and self.maxlon >= other.maxlon
                and self.maxlat >= other.maxlat
            )
        elif isinstance(other, Coordinate):
            return (
                self.minlon <= other.lon
                and self.minlat <= other.lat
                and self.maxlon >= other.lon
                and self.maxlat >= other.lat
            )
        else:
            raise TypeError("other must be a Bbox or a Coordinate")

    def intersects(self, other):
        if isinstance(other, Bbox):
            return (
                self.minlon <= other.maxlon
                and self.minlat <= other.maxlat
                and self.maxlon >= other.minlon
                and self.maxlat >= other.minlat
            )
        elif isinstance(other, Coordinate):
            return (
                self.minlon <= other.lon
                and self.minlat <= other.lat
                and self.maxlon >= other.lon
                and self.maxlat >= other.lat
            )
        else:
            raise TypeError("other must be a Bbox or a Coordinate")

    def union(self, other):
        if isinstance(other, Bbox):
            return Bbox(
                minlon=min(self.minlon, other.minlon),
                minlat=min(self.minlat, other.minlat),
                maxlon=max(self.maxlon, other.maxlon),
                maxlat=max(self.maxlat, other.maxlat),
            )
        elif isinstance(other, Coordinate):
            return Bbox(
                minlon=min(self.minlon, other.lon),
                minlat=min(self.minlat, other.lat),
                maxlon=max(self.maxlon, other.lon),
                maxlat=max(self.maxlat, other.lat),
            )
        else:
            raise TypeError("other must be a Bbox or a Coordinate")

    def intersection(self, other):
        if isinstance(other, Bbox):
            return Bbox(
                minlon=max(self.minlon, other.minlon),
                minlat=max(self.minlat, other.minlat),
                maxlon=min(self.maxlon, other.maxlon),
                maxlat=min(self.maxlat, other.maxlat),
            )
        elif isinstance(other, Coordinate):
            if self.contains(other):
                return other
            else:
                return None
        else:
            raise TypeError("other must be a Bbox or a Coordinate")

    def expand(self, amount):
        return Bbox(
            minlon=self.minlon - amount,
            minlat=self.minlat - amount,
            maxlon=self.maxlon + amount,
            maxlat=self.maxlat + amount,
        )

    def shrink(self, amount):
        return Bbox(
            minlon=self.minlon + amount,
            minlat=self.minlat + amount,
            maxlon=self.maxlon - amount,
            maxlat=self.maxlat - amount,
        )

    def scale(self, amount):
        return Bbox(
            minlon=self.minlon * amount,
            minlat=self.minlat * amount,
            maxlon=self.maxlon * amount,
            maxlat=self.maxlat * amount,
        )

    def translate(self, amount):
        return Bbox(
            minlon=self.minlon + amount.lon,
            minlat=self.minlat + amount.lat,
            maxlon=self.maxlon + amount.lon,
            maxlat=self.maxlat + amount.lat,
        )

    def to_geojson(self):
        return {
            "type": "Polygon",
            "coordinates": [
                [
                    [self.minlon, self.minlat],
                    [self.minlon, self.maxlat],
                    [self.maxlon, self.maxlat],
                    [self.maxlon, self.minlat],
                    [self.minlon, self.minlat],
                ]
            ],
        }

    def to_wkt(self):
        return f"POLYGON(({self.minlon} {self.minlat}, {self.minlon} {self.maxlat}, {self.maxlon} {self.maxlat}, {self.maxlon} {self.minlat}, {self.minlon} {self.minlat}))"

    def to_wkb(self):
        return wkb.dumps(self.to_geojson())

    @classmethod
    def from_geojson(cls, geojson):
        if geojson["type"] == "Polygon":
            return cls.from_polygon(geojson["coordinates"])
        elif geojson["type"] == "MultiPolygon":
            return cls.from_multipolygon(geojson["coordinates"])
        else:
            raise ValueError("geojson must be a Polygon or a MultiPolygon")

    @classmethod
    def from_wkt(cls, wkt):
        return cls.from_geojson(wkt.loads(wkt))

    @classmethod
    def from_wkb(cls, wkb):
        return cls.from_geojson(wkb.loads(wkb))

    @classmethod
    def from_polygon(cls, polygon):
        if len(polygon) != 1:
            raise ValueError("polygon must have exactly one ring")
        return cls.from_ring(polygon[0])


class Spectrum(RompyBaseModel):
    """A class to represent a wave spectrum."""

    fmin: float = Field(0.0464, description="Minimum frequency in Hz")
    fmax: float = Field(1.0, description="Maximum frequency in Hz")
    nfreqs: int = Field(31, description="Number of frequency components")
    ndirs: int = Field(36, description="Number of directional components")
    # TODO make more flexible.


class DatasetCoords(RompyBaseModel):
    """Coordinates representation."""

    t: Optional[str] = Field("time", description="Name of the time coordinate")
    x: Optional[str] = Field("longitude", description="Name of the x coordinate")
    y: Optional[str] = Field("latitude", description="Name of the y coordinate")
    z: Optional[str] = Field("depth", description="Name of the z coordinate")
    s: Optional[str] = Field("site", description="Name of the site coordinate")


class Slice(BaseModel):
    """Basic float or datetime slice representation"""

    start: Optional[Union[float, datetime, str]] = Field(
        default=None, description="Slice start"
    )
    stop: Optional[Union[float, datetime, str]] = Field(
        default=None, description="Slice stop"
    )

    @classmethod
    def from_dict(cls, d: dict):
        return cls(start=d["start"], stop=d["stop"])

    @classmethod
    def from_slice(cls, s: slice):
        return cls(start=s.start, stop=s.stop)

    def to_slice(self) -> slice:
        return slice(self.start, self.stop)

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if isinstance(v, slice):
            return cls.from_slice(v)
        elif isinstance(v, cls):
            return v
        else:
            raise ValueError(f"Cannot convert {type(v)} to Slice")
