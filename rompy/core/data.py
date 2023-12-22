"""Rompy core data objects."""
import logging
from pathlib import Path
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Optional, Union

from cloudpathlib import AnyPath
import intake
from intake.catalog import Catalog
import xarray as xr
from pydantic import ConfigDict, Field
from oceanum.datamesh import Connector

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

from rompy.core.filters import Filter
from rompy.core.grid import BaseGrid, RegularGrid
from rompy.core.time import TimeRange
from rompy.core.types import RompyBaseModel, DatasetCoords, Slice


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
    """Source dataset from intake catalog."""

    model_type: Literal["intake"] = Field(
        default="intake",
        description="Model type discriminator",
    )
    dataset_id: str = Field(description="The id of the dataset to read in the catalog")
    catalog_uri: str | Path = Field(description="The URI of the catalog to read from")
    kwargs: dict = Field(
        default={},
        description="Keyword arguments to define intake dataset parameters",
    )

    def __str__(self) -> str:
        return f"SourceIntake(catalog_uri={self.catalog_uri}, dataset_id={self.dataset_id})"

    @property
    def catalog(self) -> Catalog:
        """The intake catalog instance."""
        return intake.open_catalog(self.catalog_uri)

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

    @property
    def connector(self) -> Connector:
        """The Datamesh connector instance."""
        return Connector(token=self.token, **self.kwargs)

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

        coords = [
            [xslice.start, yslice.start],
            [xslice.stop, yslice.start],
            [xslice.stop, yslice.stop],
            [xslice.start, yslice.stop],
            [xslice.start, yslice.start],
        ]
        geofilter = dict(
            type="feature",
            geom=dict(
                type="Feature",
                geometry=dict(
                    type="Polygon",
                    coordinates=[coords],
                ),
            ),
        )
        return geofilter

    def _timefilter(self, filters: Filter, coords: DatasetCoords) -> dict:
        """The Datamesh timefilter."""
        tslice = filters.crop.get(coords.t)
        if tslice is None:
            logger.warning(
                f"No time slice found in the crop filter {filters.crop}, "
                "cannot define a timefilter for querying datamesh"
            )
            return None
        timefilter = dict(
            type="range",
            times=[tslice.start, tslice.stop],
        )
        return timefilter

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
        if filters:
            ds = filters(ds)
        return ds


class DataBlob(RompyBaseModel):
    """Data source for model ingestion.

    Generic data source for files that simply need to be copied to the model directory.

    """

    model_type: Literal["data_blob"] = Field(
        default="data_blob",
        description="Model type discriminator",
    )
    id: str = Field(
        default="data", description="Unique identifier for this data source"
    )
    source: AnyPath = Field(
        description=(
            "URI of the data source, either a local file path or a remote uri"
        ),
    )

    def get(self, destdir: str | Path) -> Path:
        """Copy the data source to a new directory.

        Parameters
        ----------
        destdir : str | Path
            The destination directory to copy the data source to.

        Returns
        -------
        outfile: Path
            The path to the copied file.

        """
        outfile = Path(destdir) / self.source.name
        if outfile.resolve() != self.source.resolve():
            outfile.write_bytes(self.source.read_bytes())
        return outfile


DATA_SOURCE_TYPES = Union[
    SourceDataset,
    SourceFile,
    SourceIntake,
    SourceDatamesh,
]
GRID_TYPES = Union[BaseGrid, RegularGrid]


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
    source: DATA_SOURCE_TYPES = Field(
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

    def _filter_grid(self, grid: GRID_TYPES):
        """Define the filters to use to extract data to this grid"""
        x0, y0, x1, y1 = grid.bbox(buffer=self.buffer)
        self.filter.crop.update(
            {
                self.coords.x: Slice(start=x0, stop=x1),
                self.coords.y: Slice(start=y0, stop=y1),
            }
        )

    def _filter_time(self, time: TimeRange):
        """Define the filters to use to extract data to this grid"""
        self.filter.crop.update({self.coords.t: Slice(start=time.start, stop=time.end)})

    @property
    def ds(self):
        """Return the xarray dataset for this data source."""
        ds = self.source.open(
            variables=self.variables, filters=self.filter, coords=self.coords
        )
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
        outfile = Path(destdir) / f"{self.id}.nc"
        self.ds.to_netcdf(outfile)
        return outfile
