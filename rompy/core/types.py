from pydantic import BaseModel, Field, validator


class RompyBaseModel(BaseModel):
    class Config:
        underscore_attrs_are_private = True


class Latitude(BaseModel):
    """Latitude"""

    lat: float = Field(..., description="Latitude", ge=-90, le=90)

    @validator("lat")
    def validate_lat(cls, v):
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

    lon: float = Field(..., description="Longitude", ge=-180, le=180)

    @validator("lon")
    def validate_lon(cls, v):
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

    lon: float = Field(..., description="Longitude")
    lat: float = Field(..., description="Latitude")

    @validator("lon")
    def validate_lon(cls, v):
        if not (-180 <= v <= 180):
            raise ValueError("longitude must be between -180 and 180")
        return v

    @validator("lat")
    def validate_lat(cls, v):
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

    @validator("minlon")
    def validate_minlon(cls, v, values):
        if v > values["maxlon"]:
            raise ValueError("minlon must be less than maxlon")
        return v

    @validator("minlat")
    def validate_minlat(cls, v, values):
        if v > values["maxlat"]:
            raise ValueError("minlat must be less than maxlat")
        return v

    @validator("maxlon")
    def validate_maxlon(cls, v, values):
        if v < values["minlon"]:
            raise ValueError("maxlon must be greater than minlon")
        return v

    @validator("maxlat")
    def validate_maxlat(cls, v, values):
        if v < values["minlat"]:
            raise ValueError("maxlat must be greater than minlat")
        return v

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
