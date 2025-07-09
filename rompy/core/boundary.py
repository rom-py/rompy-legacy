"""Boundary classes."""

import logging
from importlib.metadata import entry_points
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import xarray as xr
from pydantic import Field, field_validator

from rompy.core.data import DataGrid
from rompy.core.grid import RegularGrid
from rompy.core.time import TimeRange
from rompy.utils import load_entry_points

logger = logging.getLogger(__name__)


def find_minimum_distance(points: list[tuple[float, float]]) -> float:
    """Find the minimum distance between a set of points.

    Parameters
    ----------
    points: list[tuple[float, float]]
        List of points as (x, y) tuples.

    Returns
    -------
    min_distance: float
        Minimum distance between all points.

    """

    def calculate_distance(x1, y1, x2, y2):
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    n = len(points)
    if n <= 1:
        return float("inf")

    # Sort points by x-coordinate
    points.sort()

    # Recursive step
    if n == 2:
        return calculate_distance(*points[0], *points[1])

    mid = n // 2
    left_points = points[:mid]
    right_points = points[mid:]

    # Divide and conquer
    left_min = find_minimum_distance(left_points)
    right_min = find_minimum_distance(right_points)

    min_distance = min(left_min, right_min)

    # Find the closest pair across the dividing line
    strip = []
    for point in points:
        if abs(point[0] - points[mid][0]) < min_distance:
            strip.append(point)

    strip_min = min_distance
    strip_len = len(strip)
    for i in range(strip_len - 1):
        j = i + 1
        while j < strip_len and (strip[j][1] - strip[i][1]) < strip_min:
            distance = calculate_distance(*strip[i], *strip[j])
            if distance < strip_min:
                strip_min = distance
            j += 1

    return min(min_distance, strip_min)


class DataBoundary(DataGrid):
    model_type: Literal["boundary"] = Field(
        default="data_boundary",
        description="Model type discriminator",
    )
    id: str = Field(description="Unique identifier for this data source")
    spacing: Optional[Union[float, Literal["parent"]]] = Field(
        default=None,
        description=(
            "Spacing between points along the grid boundary to retrieve data for. If "
            "None (default), points are defined from the the actual grid object "
            "passed to the `get` method. If 'parent', the resolution of the parent "
            "dataset is used to define the spacing."
        ),
    )
    sel_method: Literal["sel", "interp"] = Field(
        default="sel",
        description=(
            "Xarray method to use for selecting boundary points from the dataset"
        ),
    )
    sel_method_kwargs: dict = Field(
        default={}, description="Keyword arguments for sel_method"
    )
    crop_data: bool = Field(
        default=True,
        description="Update crop filter from Time object if passed to get method",
    )

    @field_validator("spacing")
    @classmethod
    def spacing_gt_zero(cls, v):
        if v not in (None, "parent") and v <= 0.0:
            raise ValueError("Spacing must be greater than zero")
        return v

    def _source_grid_spacing(self) -> float:
        """Return the lowest grid spacing in the source dataset.

        In a gridded dataset this is defined as the lowest spacing between adjacent
        points in the dataset. In other dataset types such as a station dataset this
        method needs to be overriden to return the lowest spacing between points.

        """
        dx = np.diff(sorted(self.ds[self.coords.x].values)).min()
        dy = np.diff(sorted(self.ds[self.coords.y].values)).min()
        return min(dx, dy)

    def _set_spacing(self) -> float:
        """Define spacing from the parent dataset if required."""
        if self.spacing == "parent":
            return self._source_grid_spacing()
        else:
            return self.spacing

    def _boundary_points(self, grid) -> tuple:
        """Returns the x and y arrays representing the boundary points to select.

        This method can be overriden to define custom boundary points.

        """
        xbnd, ybnd = grid.boundary_points(spacing=self._set_spacing())
        return xbnd, ybnd

    def _sel_boundary(self, grid) -> xr.Dataset:
        """Select the boundary points from the dataset."""
        xbnd, ybnd = self._boundary_points(grid=grid)
        coords = {
            self.coords.x: xr.DataArray(xbnd, dims=("site",)),
            self.coords.y: xr.DataArray(ybnd, dims=("site",)),
        }
        ds = getattr(self.ds, self.sel_method)(coords, **self.sel_method_kwargs)
        # rename the coordinates to x, y
        ds = ds.rename({self.coords.x: "x", self.coords.y: "y"})
        return ds

    def get(
        self, destdir: str | Path, grid: RegularGrid, time: Optional[TimeRange] = None
    ) -> str:
        """Write the selected boundary data to a netcdf file.

        Parameters
        ----------
        destdir : str | Path
            Destination directory for the netcdf file.
        grid : RegularGrid
            Grid instance to use for selecting the boundary points.
        time: TimeRange, optional
            The times to filter the data to, only used if `self.crop_data` is True.

        Returns
        -------
        outfile : Path
            Path to the netcdf file.

        """
        if self.crop_data and time is not None:
            self._filter_time(time)
        ds = self._sel_boundary(grid)
        outfile = Path(destdir) / f"{self.id}.nc"
        ds.to_netcdf(outfile)
        return outfile

    def plot(self, model_grid=None, cmap="turbo", fscale=10, ax=None, **kwargs):
        return scatter_plot(
            self, model_grid=model_grid, cmap=cmap, fscale=fscale, ax=ax, **kwargs
        )

    def plot_boundary(self, grid=None, fscale=10, ax=None, **kwargs):
        """Plot the boundary points on a map."""
        ds = self._sel_boundary(grid)
        fig, ax = grid.plot(ax=ax, fscale=fscale, **kwargs)
        return scatter_plot(
            self,
            ds=ds,
            fscale=fscale,
            ax=ax,
            **kwargs,
        )


# Plugin for the source types
SOURCE_TYPES = load_entry_points("rompy.source")


class BoundaryWaveStation(DataBoundary):
    """Wave boundary data from station datasets.

    Note
    ----
    The `tolerance` behaves differently with sel_methods `idw` and `nearest`; in `idw`
    sites with no enough neighbours within `tolerance` are masked whereas in `nearest`
    an exception is raised (see wavespectra documentation for more details).

    Note
    ----
    Be aware that when using `idw` missing values will be returned for sites with less
    than 2 neighbours within `tolerance` in the original dataset. This is okay for land
    mask areas but could cause boundary issues when on an open boundary location. To
    avoid this either use `nearest` or increase `tolerance` to include more neighbours.

    """

    grid_type: Literal["boundary_wave_station"] = Field(
        default="boundary_wave_station",
        description="Model type discriminator",
    )
    source: Union[SOURCE_TYPES] = Field(
        description=(
            "Dataset source reader, must return a wavespectra-enabled "
            "xarray dataset in the open method"
        ),
        discriminator="model_type",
    )
    sel_method: Literal["idw", "nearest"] = Field(
        default="idw",
        description=(
            "Wavespectra method to use for selecting boundary points from the dataset"
        ),
    )
    buffer: float = Field(
        default=2.0,
        description="Space to buffer the grid bounding box if `filter_grid` is True",
    )

    def model_post_init(self, __context):
        self.variables = ["efth", "lon", "lat"]

    # @model_validator(mode="after")
    # def assert_has_wavespectra_accessor(self) -> "BoundaryWaveStation":
    #     dset = self.source.open()
    #     if not hasattr(dset, "spec"):
    #         raise ValueError(f"Wavespectra compatible source is required")
    #     return self

    def _source_grid_spacing(self, grid) -> float:
        """Return the lowest spacing between points in the source dataset."""
        # Select dataset points just outside the actual grid to optimise the search
        xbnd, ybnd = grid.boundary().exterior.coords.xy
        dx = np.diff(xbnd).min()
        dy = np.diff(ybnd).min()
        buffer = 2 * min(dx, dy)
        x0, y0, x1, y1 = grid.bbox(buffer=buffer)
        ds = self.ds.spec.sel([x0, x1], [y0, y1], method="bbox")
        # Return the closest distance between adjacent points in cropped dataset
        points = list(zip(ds.lon.values, ds.lat.values))
        return find_minimum_distance(points)

    def _set_spacing(self, grid) -> float:
        """Define spacing from the parent dataset if required."""
        if self.spacing == "parent":
            return self._source_grid_spacing(grid)
        else:
            return self.spacing

    def _boundary_points(self, grid) -> tuple:
        """Returns the x and y arrays representing the boundary points to select.

        Override the default method to use grid when setting the default spacing.

        """
        xbnd, ybnd = grid.boundary_points(spacing=self._set_spacing(grid))
        return xbnd, ybnd

    def _sel_boundary(self, grid) -> xr.Dataset:
        """Select the boundary points from the dataset."""
        xbnd, ybnd = self._boundary_points(grid=grid)
        ds = self.ds.spec.sel(
            lons=xbnd,
            lats=ybnd,
            method=self.sel_method,
            **self.sel_method_kwargs,
        )
        return ds

    @property
    def ds(self):
        """Return the filtered xarray dataset instance."""
        dset = super().ds
        if dset.efth.size == 0:
            raise ValueError(f"Empty dataset after applying filter {self.filter}")
        return dset

    def get(
        self, destdir: str | Path, grid: RegularGrid, time: Optional[TimeRange] = None
    ) -> str:
        """Write the selected boundary data to a netcdf file.

        Parameters
        ----------
        destdir : str | Path
            Destination directory for the netcdf file.
        grid : RegularGrid
            Grid instance to use for selecting the boundary points.
        time: TimeRange, optional
            The times to filter the data to, only used if `self.crop_data` is True.

        Returns
        -------
        outfile : Path
            Path to the netcdf file.

        """
        if self.crop_data:
            if time is not None:
                self._filter_time(time)
            if grid is not None:
                self._filter_grid(grid)
        ds = self._sel_boundary(grid)
        outfile = Path(destdir) / f"{self.id}.nc"
        ds.spec.to_netcdf(outfile)
        return outfile


def scatter_plot(bnd, ds=None, fscale=10, ax=None, **kwargs):
    """Plot the grid"""

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER

    if ds is None:
        ds = bnd.ds

    # First set some plot parameters:
    minLon, minLat, maxLon, maxLat = (
        min(ds[bnd.coords.x]),
        min(ds[bnd.coords.y]),
        max(ds[bnd.coords.x]),
        max(ds[bnd.coords.y]),
    )
    extents = [minLon, maxLon, minLat, maxLat]

    if ax is None:
        # create figure and plot/map
        figsize = figsize = (fscale, fscale * (maxLat - minLat) / (maxLon - minLon))
        subplot_kw = {"projection": ccrs.PlateCarree()}
        fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw=subplot_kw)
        # ax.set_extent(extents, crs=ccrs.PlateCarree())

        coastline = cfeature.GSHHSFeature(
            scale="auto", edgecolor="black", facecolor=cfeature.COLORS["land"]
        )

        ax.add_feature(coastline)
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

    ax.scatter(ds[bnd.coords.x], ds[bnd.coords.y], transform=ccrs.PlateCarree())
    return ax
