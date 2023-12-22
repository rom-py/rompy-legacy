import logging
from typing import Any, Union, Literal, Optional

import numpy as np
from pydantic import Field, model_validator
from pydantic_numpy.typing import Np1DArray, Np2DArray
from shapely.geometry import MultiPoint, Polygon

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

from rompy.core.types import Bbox, RompyBaseModel


logger = logging.getLogger(__name__)


class BaseGrid(RompyBaseModel):
    """Representation of a grid in geographic space.

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

    def _figsize(self, x0, x1, y0, y1, fscale):
        xlen = abs(x1 - x0)
        ylen = abs(y1 - y0)
        if xlen >= ylen:
            figsize = (fscale, fscale * ylen / xlen or fscale)
        else:
            figsize = (fscale * xlen / ylen or fscale, fscale)
        return figsize

    def plot(
        self,
        ax=None,
        figsize=None,
        fscale=10,
        buffer=0.1,
        borders=True,
        land=True,
        coastline=True,
    ):
        """Plot the grid"""

        projection = ccrs.PlateCarree()
        transform = ccrs.PlateCarree()

        # Set some plot parameters:
        x0, y0, x1, y1 = self.bbox(buffer=buffer)

        # create figure and plot/map
        if ax is None:
            if figsize is None:
                figsize = self._figsize(x0, x1, y0, y1, fscale)
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection=projection)
            ax.set_extent([x0, x1, y0, y1], crs=transform)

            if borders:
                ax.add_feature(cfeature.BORDERS)
            if land:
                ax.add_feature(cfeature.LAND)
            if coastline:
                ax.add_feature(cfeature.COASTLINE)
        else:
            fig = ax.figure

        ax.gridlines(
            crs=transform,
            draw_labels=["left", "bottom"],
            linewidth=1,
            color="gray",
            alpha=0.5,
            linestyle="--",
        )

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
    """Regular grid in geographic space.

    This object provides an abstract representation of a regular grid in some
    geographic space.

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

    @model_validator(mode="after")
    def generate(self) -> "RegularGrid":
        """Generate the grid from the provided parameters."""
        keys = ["x0", "y0", "dx", "dy", "nx", "ny"]
        if self.x is not None and self.y is not None:
            for key in keys:
                if getattr(self, key) is not None:
                    logger.warning(f"x, y provided explicitly, can't process {key}")
            self._attrs_from_xy()
        elif None in [getattr(self, key) for key in keys]:
            raise ValueError(f"All of {','.join(keys)} must be provided for REG grid")
        # Ensure x, y 2D coordinates are defined
        self._regen_grid()
        return self

    def _regen_grid(self):
        _x, _y = self._gen_reg_cgrid()
        self.x = _x
        self.y = _y

    def _attrs_from_xy(self):
        """Generate regular grid attributes from x, y coordinates."""
        self.ny, self.nx = self.x.shape
        self.x0 = self.x[0, 0]
        self.y0 = self.y[0, 0]
        self.rot = np.degrees(
            np.arctan2(self.y[0, 1] - self.y0, self.x[0, 1] - self.x0)
        )
        self.dx = np.sqrt((self.x[0, 1] - self.x0) ** 2 + (self.y[0, 1] - self.y0) ** 2)
        self.dy = np.sqrt((self.x[1, 0] - self.x0) ** 2 + (self.y[1, 0] - self.y0) ** 2)

    @property
    def xlen(self):
        return self.dx * (self.nx - 1)

    @property
    def ylen(self):
        return self.dy * (self.ny - 1)

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

    def __eq__(self, other) -> bool:
        return (
            (self.nx == other.nx)
            & (self.ny == other.ny)
            & (self.rot == other.rot)
            & (self.x0 == other.x0)
            & (self.y0 == other.y0)
            & (self.dx == other.dx)
            & (self.dy == other.dy)
        )

    def __repr__(self):
        return f"{self.__class__.__name__}({self.nx}, {self.ny})"

    def __str__(self):
        return f"{self.__class__.__name__}({self.nx}, {self.ny})"


if __name__ == "__main__":
    import copy

    grid0 = RegularGrid(x0=-1, y0=1, rot=35, nx=10, ny=10, dx=1, dy=2)
    grid1 = RegularGrid(x=grid0.x, y=grid0.y)

    # print(grid0.rot, grid1.rot)
    # print(grid0.dx, grid0.dy, grid1.dx, grid1.dy)
