"""Rompy core data objects."""

import logging
import os
from abc import ABC, abstractmethod
from functools import cached_property
from importlib.metadata import entry_points
from pathlib import Path
from shutil import copytree
from typing import Literal, Optional, Union

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import fsspec
import intake
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cloudpathlib import AnyPath
from intake.catalog import Catalog
from intake.catalog.local import YAMLFileCatalog
from oceanum.datamesh import Connector
from pydantic import ConfigDict, Field, PrivateAttr, model_validator

from rompy.core.filters import Filter
from rompy.core.grid import BaseGrid, RegularGrid
from rompy.core.time import TimeRange
from rompy.core.types import DatasetCoords, RompyBaseModel, Slice

logger = logging.getLogger(__name__)


class SourceBase(RompyBaseModel, ABC):
    """Abstract base class for a source dataset."""

    model_type: Literal["base_source"] = Field(
        description="Model type discriminator, must be overriden by a subclass",
    )

    @abstractmethod
    def _open(self) -> xr.Dataset:
        """This abstract private method should return a xarray dataset object."""
        pass

    @cached_property
    def coordinates(self) -> xr.Dataset:
        """Return the coordinates of the datasource."""
        return self.open().coords

    def open(self, variables: list = [], filters: Filter = {}, **kwargs) -> xr.Dataset:
        """Return the filtered dataset object.

        Parameters
        ----------
        variables : list, optional
            List of variables to select from the dataset.
        filters : Filter, optional
            Filters to apply to the dataset.

        Notes
        -----
        The kwargs are only a placeholder in case a subclass needs to pass additional
        arguments to the open method.

        """
        ds = self._open()
        if variables:
            ds = ds[variables]
        if filters:
            ds = filters(ds)
        return ds


class SourceDataset(SourceBase):
    """Source dataset from an existing xarray Dataset object."""

    model_type: Literal["dataset"] = Field(
        default="dataset",
        description="Model type discriminator",
    )
    obj: xr.Dataset = Field(
        description="xarray Dataset object",
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __str__(self) -> str:
        return f"SourceDataset(obj={self.obj})"

    def _open(self) -> xr.Dataset:
        return self.obj


class SourceFile(SourceBase):
    """Source dataset from file to open with xarray.open_dataset."""

    model_type: Literal["open_dataset"] = Field(
        default="open_dataset",
        description="Model type discriminator",
    )
    uri: str | Path = Field(description="Path to the dataset")
    kwargs: dict = Field(
        default={},
        description="Keyword arguments to pass to xarray.open_dataset",
    )

    def __str__(self) -> str:
        return f"SourceFile(uri={self.uri})"

    def _open(self) -> xr.Dataset:
        return xr.open_dataset(self.uri, **self.kwargs)


class SourceIntake(SourceBase):
    """Source dataset from intake catalog.

    note
    ----
    The intake catalog can be prescribed either by the URI of an existing catalog file
    or by a YAML string defining the catalog. The YAML string can be obtained from
    calling the `yaml()` method on an intake dataset instance.

    """

    model_type: Literal["intake"] = Field(
        default="intake",
        description="Model type discriminator",
    )
    dataset_id: str = Field(description="The id of the dataset to read in the catalog")
    catalog_uri: Optional[str | Path] = Field(
        default=None,
        description="The URI of the catalog to read from",
    )
    catalog_yaml: Optional[str] = Field(
        default=None,
        description="The YAML string of the catalog to read from",
    )
    kwargs: dict = Field(
        default={},
        description="Keyword arguments to define intake dataset parameters",
    )

    @model_validator(mode="after")
    def check_catalog(self) -> "SourceIntake":
        if self.catalog_uri is None and self.catalog_yaml is None:
            raise ValueError("Either catalog_uri or catalog_yaml must be provided")
        elif self.catalog_uri is not None and self.catalog_yaml is not None:
            raise ValueError("Only one of catalog_uri or catalog_yaml can be provided")
        return self

    def __str__(self) -> str:
        return f"SourceIntake(catalog_uri={self.catalog_uri}, dataset_id={self.dataset_id})"

    @property
    def catalog(self) -> Catalog:
        """The intake catalog instance."""
        if self.catalog_uri:
            return intake.open_catalog(self.catalog_uri)
        else:
            fs = fsspec.filesystem("memory")
            fs_map = fs.get_mapper()
            fs_map[f"/temp.yaml"] = self.catalog_yaml.encode("utf-8")
            return YAMLFileCatalog("temp.yaml", fs=fs)

    def _open(self) -> xr.Dataset:
        return self.catalog[self.dataset_id](**self.kwargs).to_dask()


class SourceDatamesh(SourceBase):
    """Source dataset from Datamesh.

    Datamesh documentation: https://docs.oceanum.io/datamesh/index.html

    """

    model_type: Literal["datamesh"] = Field(
        default="datamesh",
        description="Model type discriminator",
    )
    datasource: str = Field(
        description="The id of the datasource on Datamesh",
    )
    token: Optional[str] = Field(
        description="Datamesh API token, taken from the environment if not provided",
    )
    kwargs: dict = Field(
        default={},
        description="Keyword arguments to pass to `oceanum.datamesh.Connector`",
    )

    def __str__(self) -> str:
        return f"SourceDatamesh(datasource={self.datasource})"

    @cached_property
    def connector(self) -> Connector:
        """The Datamesh connector instance."""
        return Connector(token=self.token, **self.kwargs)

    @cached_property
    def coordinates(self) -> xr.Dataset:
        """Return the coordinates of the datasource."""
        return self._open(variables=[], geofilter=None, timefilter=None).coords

    def _geofilter(self, filters: Filter, coords: DatasetCoords) -> dict:
        """The Datamesh geofilter."""
        xslice = filters.crop.get(coords.x)
        yslice = filters.crop.get(coords.y)
        if xslice is None or yslice is None:
            logger.warning(
                f"No slices found for x={coords.x} and/or y={coords.y} in the crop "
                f"filter {filters.crop}, cannot define a geofilter for querying"
            )
            return None

        x0 = min(xslice.start, xslice.stop)
        x1 = max(xslice.start, xslice.stop)
        y0 = min(yslice.start, yslice.stop)
        y1 = max(yslice.start, yslice.stop)
        return dict(type="bbox", geom=[x0, y0, x1, y1])

    def _timefilter(self, filters: Filter, coords: DatasetCoords) -> dict:
        """The Datamesh timefilter."""
        tslice = filters.crop.get(coords.t)
        if tslice is None:
            logger.info(
                f"No time slice found in the crop filter {filters.crop}, "
                "cannot define a timefilter for querying datamesh"
            )
            return None
        return dict(type="range", times=[tslice.start, tslice.stop])

    def _open(self, variables: list, geofilter: dict, timefilter: dict) -> xr.Dataset:
        query = dict(
            datasource=self.datasource,
            variables=variables,
            geofilter=geofilter,
            timefilter=timefilter,
        )
        return self.connector.query(query)

    def open(
        self, filters: Filter, coords: DatasetCoords, variables: list = []
    ) -> xr.Dataset:
        """Returns the filtered dataset object.

        This method is overriden from the base class because the crop filters need to
        be converted to a geofilter and timefilter for querying Datamesh.

        """
        ds = self._open(
            variables=variables,
            geofilter=self._geofilter(filters, coords),
            timefilter=self._timefilter(filters, coords),
        )
        return ds


class DataBlob(RompyBaseModel):
    """Data source for model ingestion.

    Generic data source for files that either need to be copied to the model directory
    or linked if `link` is set to True.
    """

    model_type: Literal["data_blob", "data_link"] = Field(
        default="data_blob",
        description="Model type discriminator",
    )
    id: str = Field(
        default="data", description="Unique identifier for this data source"
    )
    source: AnyPath = Field(
        description="URI of the data source, either a local file path or a remote uri",
    )
    link: bool = Field(
        default=False,
        description="Whether to create a symbolic link instead of copying the file",
    )
    _copied: str = PrivateAttr(default=None)

    def get(self, destdir: Union[str, Path], name: str = None, *args, **kwargs) -> Path:
        """Copy or link the data source to a new directory.

        Parameters
        ----------
        destdir : str | Path
            The destination directory to copy or link the data source to.

        Returns
        -------
        Path
            The path to the copied file or created symlink.
        """
        destdir = Path(destdir).resolve()

        if self.link:
            # Create a symbolic link
            if name:
                symlink_path = destdir / name
            else:
                symlink_path = destdir / self.source.name

            # Ensure the destination directory exists
            destdir.mkdir(parents=True, exist_ok=True)

            # Remove existing symlink/file if it exists
            if symlink_path.exists():
                symlink_path.unlink()

            # Compute the relative path from destdir to self.source
            relative_source_path = os.path.relpath(self.source.resolve(), destdir)

            # Create symlink
            os.symlink(relative_source_path, symlink_path)
            self._copied = symlink_path

            return symlink_path
        else:
            # Copy the data source
            if self.source.is_dir():
                # Copy directory
                outfile = copytree(self.source, destdir)
            else:
                if name:
                    outfile = destdir / name
                else:
                    outfile = destdir / self.source.name
                if outfile.resolve() != self.source.resolve():
                    outfile.write_bytes(self.source.read_bytes())
            self._copied = outfile
            return outfile


GRID_TYPES = Union[BaseGrid, RegularGrid]


# Plugin for the source types
data_source_eps = entry_points(group="rompy.data_source")
DATA_SOURCE_TYPES = tuple(eps.load() for eps in data_source_eps)


class DataGrid(DataBlob):
    """Data object for model ingestion.

    Generic data object for xarray datasets that need to be filtered and written to
    netcdf.

    Note
    ----
    The fields `filter_grid` and `filter_time` trigger updates to the crop filter from
    the grid and time range objects passed to the get method. This is useful for data
    sources that are not defined on the same grid as the model grid or the same time
    range as the model run.

    """

    model_type: Literal["data_grid"] = Field(
        default="data_grid",
        description="Model type discriminator",
    )
    source: Union[DATA_SOURCE_TYPES] = Field(
        description="Source reader, must return an xarray dataset in the open method",
        discriminator="model_type",
    )

    filter: Optional[Filter] = Field(
        default_factory=Filter,
        description="Optional filter specification to apply to the dataset",
    )
    variables: Optional[list[str]] = Field(
        [], description="Subset of variables to extract from the dataset"
    )
    coords: Optional[DatasetCoords] = Field(
        default=DatasetCoords(),
        description="Names of the coordinates in the dataset",
    )
    crop_data: bool = Field(
        default=True,
        description=(
            "Update crop filters from Grid and Time objects if passed to get method"
        ),
    )
    buffer: float = Field(
        default=0.0,
        description="Space to buffer the grid bounding box if `filter_grid` is True",
    )
    time_buffer: list[int] = Field(
        default=[0, 0],
        description=(
            "Number of source data timesteps to buffer the time range "
            "if `filter_time` is True"
        ),
    )

    def _filter_grid(self, grid: GRID_TYPES):
        """Define the filters to use to extract data to this grid"""
        x0, y0, x1, y1 = grid.bbox(buffer=self.buffer)
        self.filter.crop.update(
            {
                self.coords.x: Slice(start=x0, stop=x1),
                self.coords.y: Slice(start=y0, stop=y1),
            }
        )

    def _filter_time(self, time: TimeRange, end_buffer=1):
        """Define the filters to use to extract data to this grid"""
        start = time.start
        end = time.end
        t = self.coords.t
        if t in self.source.coordinates and self.source.coordinates[t].size > 1:
            times = self.source.coordinates[t].to_index().to_pydatetime()
            dt = times[1] - times[0]
            if self.time_buffer[0]:
                start -= dt * self.time_buffer[0]
            if self.time_buffer[1]:
                end += dt * self.time_buffer[1]
        self.filter.crop.update({t: Slice(start=start, stop=end)})

    @property
    def ds(self):
        """Return the xarray dataset for this data source."""
        ds = self.source.open(
            variables=self.variables, filters=self.filter, coords=self.coords
        )
        # Sort the dataset by all coordinates to avoid cropping issues
        for dim in ds.dims:
            ds = ds.sortby(dim)
        return ds

    def _figsize(self, x0, x1, y0, y1, fscale):
        xlen = abs(x1 - x0)
        ylen = abs(y1 - y0)
        if xlen >= ylen:
            figsize = (fscale, (fscale * ylen / xlen or fscale) * 0.8)
        else:
            figsize = ((fscale * xlen / ylen) * 1.2 or fscale, fscale)
        return figsize

    def plot(
        self,
        param,
        isel={},
        model_grid=None,
        cmap="turbo",
        figsize=None,
        fscale=10,
        borders=True,
        land=True,
        coastline=True,
        **kwargs,
    ):
        """Plot the grid."""

        projection = ccrs.PlateCarree()
        transform = ccrs.PlateCarree()

        # Sanity checks
        try:
            ds = self.ds[param].isel(isel)
        except KeyError as err:
            raise ValueError(f"Parameter {param} not in dataset") from err

        if ds[self.coords.x].size <= 1:
            raise ValueError(f"Cannot plot {param} with only one x coordinate\n\n{ds}")
        if ds[self.coords.y].size <= 1:
            raise ValueError(f"Cannot plot {param} with only one y coordinate\n\n{ds}")

        # Set some plot parameters:
        x0 = ds[self.coords.x].values[0]
        y0 = ds[self.coords.y].values[0]
        x1 = ds[self.coords.x].values[-1]
        y1 = ds[self.coords.y].values[-1]

        # create figure and plot/map
        if figsize is None:
            figsize = self._figsize(x0, x1, y0, y1, fscale)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection=projection)

        ds.plot.pcolormesh(ax=ax, cmap=cmap, **kwargs)

        if borders:
            ax.add_feature(cfeature.BORDERS)
        if land:
            ax.add_feature(cfeature.LAND, zorder=1)
        if coastline:
            ax.add_feature(cfeature.COASTLINE)

        ax.gridlines(
            crs=transform,
            draw_labels=["left", "bottom"],
            linewidth=1,
            color="gray",
            alpha=0.5,
            linestyle="--",
        )

        # Plot the model domain
        if model_grid:
            bx, by = model_grid.boundary_points()
            poly = plt.Polygon(list(zip(bx, by)), facecolor="r", alpha=0.05)
            ax.add_patch(poly)
            ax.plot(bx, by, lw=2, color="k")
        return fig, ax

    @property
    def outfile(self) -> str:
        return f"{self.id}.nc"

    def get(
        self,
        destdir: str | Path,
        grid: Optional[GRID_TYPES] = None,
        time: Optional[TimeRange] = None,
    ) -> Path:
        """Write the data source to a new location.

        Parameters
        ----------
        destdir : str | Path
            The destination directory to write the netcdf data to.
        grid: GRID_TYPES, optional
            The grid to filter the data to, only used if `self.crop_data` is True.
        time: TimeRange, optional
            The times to filter the data to, only used if `self.crop_data` is True.

        Returns
        -------
        outfile: Path
            The path to the written file.

        """
        if self.crop_data:
            if grid is not None:
                self._filter_grid(grid)
            if time is not None:
                self._filter_time(time)
        outfile = Path(destdir) / self.outfile
        self.ds.to_netcdf(outfile)
        return outfile
