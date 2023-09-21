import logging
from typing import Any, Union, Literal, Optional

import numpy as np
from pydantic import Field, model_validator
from pydantic_numpy.typing import Np1DArray, Np2DArray
from shapely.geometry import MultiPoint, Polygon

from .types import Bbox, RompyBaseModel


logger = logging.getLogger(__name__)


class BaseGrid(RompyBaseModel):
    """
    Abstract representation of a grid in geographic space

    This is the base class for all Grid objects. The minimum representation of a grid
    are two NumPy array's representing the vertices or nodes of some structured or
    unstructured grid, its bounding box and a boundary polygon. No knowledge of the
    grid connectivity is expected.

    """

    x: Optional[Union[Np1DArray, Np2DArray]] = Field(
        default=None, description="The x coordinates"
    )
    y: Optional[Union[Np1DArray, Np2DArray]] = Field(
        default=None, description="The y coordinates"
    )
    grid_type: Literal["base"] = "base"

    @property
    def minx(self) -> float:
        return np.nanmin(self.x)

    @property
    def maxx(self) -> float:
        return np.nanmax(self.x)

    @property
    def miny(self) -> float:
        return np.nanmin(self.y)

    @property
    def maxy(self) -> float:
        return np.nanmax(self.y)

    def bbox(self, buffer=0.0) -> Bbox:
        """Returns a bounding box for the spatial grid

        This function returns a list [ll_x, ll_y, ur_x, ur_y]
        where ll_x, ll_y (ur_x, ur_y) are the lower left (upper right)
        x and y coordinates bounding box of the model domain

        """
        ll_x = self.minx - buffer
        ll_y = self.miny - buffer
        ur_x = self.maxx + buffer
        ur_y = self.maxy + buffer
        bbox = [ll_x, ll_y, ur_x, ur_y]
        return bbox

    def _get_boundary(self, tolerance=0.2) -> Polygon:
        xys = list(zip(self.x.flatten(), self.y.flatten()))
        polygon = MultiPoint(xys).convex_hull
        polygon = polygon.simplify(tolerance=tolerance)

        return polygon

    def boundary(self, tolerance=0.2):
        """
        Returns a boundary polygon as a Shapely Polygon object. Sub-classes can override the private method Grid._get_boundary() but must return a Shapely polygon object.

        Parameters
        ----------
        tolerance : float
            See https://shapely.readthedocs.io/en/stable/manual.html#object.simplify

        Returns
        -------
        polygon : shapely.Polygon see https://shapely.readthedocs.io/en/stable/manual.html#Polygon

        """
        return self._get_boundary(tolerance=tolerance)

    def boundary_points(self, tolerance=0.2):
        """
        Convenience function to convert boundary Shapely Polygon
        to arrays of coordinates

        Parameters
        ----------
        tolerance : float
            Passed to Grid.boundary
            See https://shapely.readthedocs.io/en/stable/manual.html#object.simplify

        """
        polygon = self.boundary(tolerance=tolerance)
        hull_x, hull_y = polygon.exterior.coords.xy
        return hull_x, hull_y

    def points_along_boundary(self, spacing):
        """Points evenly spaced along the grid boundary.

        Parameters
        ----------
        spacing : float
            The spacing between points along the boundary

        Returns
        -------
        points : MultiPoint
            A Shapely MultiPoint object containing the points along the boundary.

        """
        polygon = self.boundary(tolerance=0)
        perimeter = polygon.length
        if perimeter < spacing:
            raise ValueError(f"Spacing = {spacing} > grid perimeter = {perimeter}")
        num_points = int(np.ceil(perimeter / spacing))
        points = [polygon.boundary.interpolate(i * spacing) for i in range(num_points)]
        return MultiPoint(points)

    def plot(self, fscale=10, ax=None):
        """Plot the grid"""

        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.pyplot as plt
        from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER

        # First set some plot parameters:
        bbox = self.bbox(buffer=0.1)
        minLon, minLat, maxLon, maxLat = bbox
        extents = [minLon, maxLon, minLat, maxLat]

        # create figure and plot/map
        if ax is None:
            fig, ax = plt.subplots(
                1,
                1,
                figsize=(fscale, fscale * (maxLon - minLon) / (maxLat - minLat)),
                subplot_kw={"projection": ccrs.PlateCarree()},
            )
            ax.set_extent(extents, crs=ccrs.PlateCarree())

            coastline = cfeature.GSHHSFeature(
                scale="auto", edgecolor="black", facecolor=cfeature.COLORS["land"]
            )
            ax.add_feature(coastline, zorder=0)
            ax.add_feature(cfeature.BORDERS, linewidth=2)

        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=2,
            color="gray",
            alpha=0.5,
            linestyle="--",
        )

        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        # Plot the model domain
        bx, by = self.boundary_points()
        poly = plt.Polygon(list(zip(bx, by)), facecolor="r", alpha=0.05)
        ax.add_patch(poly)
        ax.plot(bx, by, lw=2, color="k")
        return fig, ax

    def __repr__(self):
        return f"{self.__class__.__name__}({self.x}, {self.y})"

    def __eq__(self, other):
        return self.model_dump() == other.dict()


class RegularGrid(BaseGrid):
    """
    An object which provides an abstract representation of a regular grid in
    some geographic space
    """

    grid_type: Literal["regular"] = Field(
        "regular", description="Type of grid, must be 'regular'"
    )
    x0: Optional[float] = Field(
        default=None, description="X coordinate of the grid origin"
    )
    y0: Optional[float] = Field(
        default=None, description="Y coordinate of the grid origin"
    )
    rot: Optional[float] = Field(
        0.0, description="Rotation angle of the grid in degrees"
    )
    dx: Optional[float] = Field(
        default=None, description="Spacing between grid points in the x direction"
    )
    dy: Optional[float] = Field(
        default=None, description="Spacing between grid points in the y direction"
    )
    nx: Optional[int] = Field(
        default=None, description="Number of grid points in the x direction"
    )
    ny: Optional[int] = Field(
        default=None, description="Number of grid points in the y direction"
    )
    _x0: Optional[float]
    _y0: Optional[float]
    _rot: Optional[float]
    _dx: Optional[float]
    _dy: Optional[float]
    _nx: Optional[int]
    _ny: Optional[int]

    @model_validator(mode="before")
    @classmethod
    def validate_grid(cls, data: Any) -> Any:
        keys = ["x0", "y0", "dx", "dy", "nx", "ny"]
        if data.get("x") is not None or data.get("y") is not None:
            if any(data.get(key) is not None for key in keys):
                raise ValueError(
                    f"x, y provided explicitly, can't process {','.join(keys)}"
                )
            return data
        for var in keys:
            if data.get(var) is None:
                raise ValueError(f"{','.join(keys)} must be provided for REG grid")
        return data

    def __init__(self, **data):
        super().__init__(**data)
        if not isinstance(self.x, np.ndarray) or not isinstance(self.y, np.ndarray):
            self._x0 = self.x0
            self._y0 = self.y0
            self._rot = self.rot
            self._dx = self.dx
            self._dy = self.dy
            self._nx = self.nx
            self._ny = self.ny
            self._regen_grid()

    def _regen_grid(self):
        _x, _y = self._gen_reg_cgrid()
        self.x = _x
        self.y = _y

    def _gen_reg_cgrid(self):
        # Grid at origin
        i = np.arange(0.0, self.dx * self.nx, self.dx)
        j = np.arange(0.0, self.dy * self.ny, self.dy)
        ii, jj = np.meshgrid(i, j)

        # Rotation
        alpha = -self.rot * np.pi / 180.0
        R = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
        gg = np.dot(np.vstack([ii.ravel(), jj.ravel()]).T, R)

        # Translation
        x = gg[:, 0] + self.x0
        y = gg[:, 1] + self.y0

        x = np.reshape(x, ii.shape)
        y = np.reshape(y, ii.shape)
        return x, y

    def __repr__(self):
        return f"{self.__class__.__name__}({self.nx}, {self.ny})"

    def __str__(self):
        return f"{self.__class__.__name__}({self.nx}, {self.ny})"


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # model = BaseModel()
    x = np.array([0, 1, 2, 3])
    y = np.array([0, 1, 2, 3])
    xx, yy = np.meshgrid(x, y)
    grid = BaseGrid(x=xx, y=yy)
    grid.plot()
    plt.show()
