# write pydantic model to read xarray data from intake catalogs, filter, and write to netcdf

import os
import pathlib
from typing import List, Optional

import cloudpathlib
import intake
import xarray as xr
from pydantic import BaseModel, Field, root_validator

from .filters import Filter
from .time import TimeRange
from .types import RompyBaseModel


class DataBlob(RompyBaseModel):
    """Data source for model ingestion. This is intended to be a generic data
    source for files that simply need to be copied to the model directory.

    Must be a local file or a remote file.

    Attributes:
    -----------
    id: str
        Unique identifier for this data source
    path: Optional[pathlib.Path]
        Optional local file path
    url: Optional[cloudpathlib.CloudPath]
        Optional remote file url
    """

    id: str = Field(description="Unique identifier for this data source")
    path: Optional[pathlib.Path] = Field(
        default=None, description="Optional local file path"
    )
    url: Optional[cloudpathlib.CloudPath] = Field(
        default=None, description="Optional remote file url"
    )

    @root_validator
    def check_path_or_url(cls, values):
        if values.get("path") is None and values.get("url") is None:
            raise ValueError("Must provide either a path or a url")
        if values.get("path") is not None and values.get("url") is not None:
            raise ValueError("Must provide either a path or a url, not both")
        return values

    def get(self, dest: str) -> "DataBlob":
        """Copy the data source to a new location"""
        if self.path:
            pathlib.Path(dest).write_bytes(self.path.read_bytes())
        elif self.url:
            pathlib.Path(dest).write_bytes(self.url.read_bytes())
        return DataBlob(id=self.id, path=dest)


class DataGrid(RompyBaseModel):
    """Data source for model ingestion. This is intended to be a generic data
    source for xarray datasets that need to be filtered and written to netcdf.

    Must be a local file (path) or a remote file (url) or intake and dataset
    id combination.

    """

    id: str = Field(description="Unique identifier for this data source")
    path: Optional[str] = Field(None, description="Optional local file path")
    url: Optional[str] = Field(None, description="Optional remote file url")
    catalog: Optional[str] = Field(None, description="Optional intake catalog")
    dataset: Optional[str] = Field(
        None, description="Optional intake dataset id")
    args: Optional[dict] = Field(
        {}, description="Optional arguments to pass to the intake catalog"
    )
    params: Optional[dict] = Field(
        {}, description="Optional parameters to pass to the intake catalog"
    )
    filter: Optional[Filter] = Field(
        Filter(), description="Optional filter specification to apply to the dataset"
    )
    latname: Optional[str] = Field(
        "latitude", description="Name of the latitude variable"
    )
    lonname: Optional[str] = Field(
        "longitude", description="Name of the longitude variable"
    )
    timename: Optional[str] = Field(
        "time", description="Name of the time variable")
    xarray_kwargs: Optional[dict] = Field(
        {}, description="Optional keyword arguments to pass to xarray.open_dataset"
    )
    netcdf_kwargs: Optional[dict] = Field(
        {"mode": "w", "format": "NETCDF4"},
        description="Optional keyword arguments to pass to xarray.Dataset.to_netcdf",
    )

    @root_validator
    def check_path_or_url_or_intake(cls, values):
        if values.get("path") is None and values.get("url") is None:
            if values.get("catalog") is None or values.get("dataset") is None:
                raise ValueError(
                    "Must provide either a path or a url or a catalog and dataset"
                )
        if (
            values.get("path") is not None
            and values.get("url") is not None
            and values.get("catalog") is not None
        ):
            raise ValueError("Must provide only one of a path url or catalog")
        return values

    def _filter_grid(self, grid, buffer=0.1):
        """Define the filters to use to extract data to this grid"""
        minLon, minLat, maxLon, maxLat = grid.bbox()
        self.filter.crop.update(
            {
                self.lonname: slice(minLon - buffer, maxLon + buffer),
                self.latname: slice(minLat - buffer, maxLat + buffer),
            }
        )

    def _filter_time(self, time: TimeRange):
        """Define the filters to use to extract data to this grid"""
        self.filter.crop.update(
            {
                self.timename: slice(time.start, time.end),
            }
        )

    @property
    def ds(self):
        """Return the xarray dataset for this data source."""
        if self.path:
            ds = xr.open_dataset(self.path, **self.xarray_kwargs)
        elif self.url:
            ds = xr.open_dataset(self.url, **self.xarray_kwargs)
        elif self.catalog:
            cat = intake.open_catalog(self.catalog)
            ds = cat[self.dataset](**self.args, **self.params).to_dask()
        if self.filter:
            ds = self.filter(ds)
        return ds

    def plot(self, param, isel={}, model_grid=None, cmap="turbo", fscale=10):
        """Plot the grid"""

        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.pyplot as plt
        from cartopy.mpl.gridliner import (LATITUDE_FORMATTER,
                                           LONGITUDE_FORMATTER)

        ds = self.ds
        if param not in ds:
            raise ValueError(f"Parameter {param} not in dataset")

        # First set some plot parameters:
        minLon, minLat, maxLon, maxLat = (
            ds[self.lonname].values[0],
            ds[self.latname].values[0],
            ds[self.lonname].values[-1],
            ds[self.latname].values[-1],
        )
        extents = [minLon, maxLon, minLat, maxLat]

        # create figure and plot/map
        fig, ax = plt.subplots(
            1,
            1,
            figsize=(fscale, fscale * (maxLat - minLat) / (maxLon - minLon)),
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
        # ax.set_extent(extents, crs=ccrs.PlateCarree())

        coastline = cfeature.GSHHSFeature(
            scale="auto", edgecolor="black", facecolor=cfeature.COLORS["land"]
        )

        ds[param].isel(isel).plot(ax=ax, cmap=cmap)

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

        # Plot the model domain
        if model_grid:
            bx, by = model_grid.boundary_points()
            poly = plt.Polygon(list(zip(bx, by)), facecolor="r", alpha=0.05)
            ax.add_patch(poly)
            ax.plot(bx, by, lw=2, color="k")
        return fig, ax

    def get(self, stage_dir: str) -> "DataGrid":
        """Write the data source to a new location"""
        dest = os.path.join(stage_dir, f"{self.id}.nc")
        self.ds.to_netcdf(dest, **self.netcdf_kwargs)
        return DataGrid(id=self.id, path=dest)
