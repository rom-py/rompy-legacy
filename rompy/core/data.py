# write pydantic model to read xarray data from intake catalogs, filter, and write to netcdf

import os
import pathlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Optional

import cloudpathlib
import intake
import xarray as xr
from pydantic import Field, root_validator

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


class Dataset(RompyBaseModel, ABC):
    """Abstract base class for a dataset."""

    @abstractmethod
    def open(self):
        """Return a dataset instance with the wavespectra accessor."""
        pass


class DatasetXarray(Dataset):
    """Dataset from xarray reader."""

    model_type: Literal["xarray"] = Field(
        default="xarray",
        description="Model type discriminator",
    )
    uri: str | Path = Field(description="Path to the dataset")
    engine: Optional[str] = Field(
        default=None,
        description="Engine to use for reading the dataset with xarray.open_dataset",
    )
    kwargs: dict = Field(
        default={},
        description="Keyword arguments to pass to xarray.open_dataset",
    )

    def open(self):
        return xr.open_dataset(self.uri, engine=self.engine, **self.kwargs)

    def __str__(self):
        return f"DatasetXarray(uri={self.uri}"


class DatasetIntake(Dataset):
    """Wavespectra dataset from intake catalog."""

    model_type: Literal["intake"] = Field(
        default="intake",
        description="Model type discriminator",
    )
    dataset_id: str = Field(
        description="The id of the dataset to read in the catalog")
    catalog_uri: str | Path = Field(
        description="The URI of the catalog to read from")
    kwargs: dict = Field(
        default={},
        description="Keyword arguments to pass to intake.open_catalog",
    )

    def open(self):
        cat = intake.open_catalog(self.catalog_uri)
        return cat[self.dataset_id](**self.kwargs).to_dask()

    def __str__(self):
        return f"DatasetIntake(catalog_uri={self.catalog_uri}, dataset_id={self.dataset_id})"


class DataGrid(RompyBaseModel):
    """Data source for model ingestion. This is intended to be a generic data
    source for xarray datasets that need to be filtered and written to netcdf.

    Must be a local file (path) or a remote file (url) or intake and dataset
    id combination.

    """

    id: str = Field(description="Unique identifier for this data source")
    dataset: DatasetXarray | DatasetIntake = Field(
        description="Dataset reader, must return a wavespectra-enabled xarray dataset in the open method",
        discriminator="model_type",
    )
    filter: Optional[Filter] = Field(
        Filter(), description="Optional filter specification to apply to the dataset"
    )
    variables: Optional[list[str]] = Field(
        [], description="Subset of variables to extract from the dataset"
    )
    latname: Optional[str] = Field(
        "latitude", description="Name of the latitude variable"
    )
    lonname: Optional[str] = Field(
        "longitude", description="Name of the longitude variable"
    )
    timename: Optional[str] = Field(
        "time", description="Name of the time variable")

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
        ds = self.dataset.open()
        if self.variables:
            ds = ds[self.variables]
        if self.filter:
            ds = self.filter(ds)
        return ds

    def plot(self, param, isel={}, model_grid=None, cmap="turbo", fscale=10, **kwargs):
        """Plot the grid"""

        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.pyplot as plt
        from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER

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

        ds[param].isel(isel).plot(ax=ax, cmap=cmap, **kwargs)

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
