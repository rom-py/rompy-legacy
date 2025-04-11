"""Rompy source objects."""

import logging
from functools import cached_property
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Optional, Union

import fsspec
import intake
import pandas as pd
import xarray as xr
import wavespectra
from intake.catalog import Catalog
from intake.catalog.local import YAMLFileCatalog
from oceanum.datamesh import Connector
from pydantic import ConfigDict, Field, model_validator, field_validator

from rompy.core.filters import Filter
from rompy.core.types import DatasetCoords, RompyBaseModel


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
            try:
                ds = ds[variables]
            except KeyError as e:
                dataset_variables = list(ds.data_vars.keys())
                missing_variables = list(set(variables) - set(dataset_variables))
                raise ValueError(
                    f"Cannot find requested variables in dataset.\n\n"
                    f"Requested variables in the Data object: {variables}\n"
                    f"Available variables in source dataset: {dataset_variables}\n"
                    f"Missing variables: {missing_variables}\n\n"
                    f"Please check:\n"
                    f"1. The variable names in your Data object, make sure you check for default values\n"
                    f"2. The data source contains the expected variables\n"
                    f"3. If using a custom data source, ensure it creates variables with the correct names"
                ) from e
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

    model_type: Literal["file"] = Field(
        default="file",
        description="Model type discriminator",
    )
    uri: Union[str, Path] = Field(description="Path to the dataset")
    kwargs: dict = Field(
        default={},
        description="Keyword arguments to pass to xarray.open_dataset",
    )
    
    # Enable arbitrary types for Path objects
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __str__(self) -> str:
        return f"SourceFile(uri={self.uri})"

    def _open(self) -> xr.Dataset:
        # Handle Path objects by using str() to ensure compatibility 
        uri_str = str(self.uri) if isinstance(self.uri, Path) else self.uri
        return xr.open_dataset(uri_str, **self.kwargs)


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


class SourceWavespectra(SourceBase):
    """Wavespectra dataset from wavespectra reader."""

    model_type: Literal["wavespectra"] = Field(
        default="wavespectra",
        description="Model type discriminator",
    )
    uri: str | Path = Field(description="Path to the dataset")
    reader: str = Field(
        description="Name of the wavespectra reader to use, e.g., read_swan",
    )
    kwargs: dict = Field(
        default={},
        description="Keyword arguments to pass to the wavespectra reader",
    )

    def __str__(self) -> str:
        return f"SourceWavespectra(uri={self.uri}, reader={self.reader})"

    def _open(self):
        return getattr(wavespectra, self.reader)(self.uri, **self.kwargs)


class SourceTimeseriesCSV(SourceBase):
    """Timeseries source class from CSV file.

    This class should return a timeseries from a CSV file. The dataset variables are
    defined from the column headers, therefore the appropriate read_csv kwargs must be
    passed to allow defining the columns. The time index is defined from column name
    identified by the tcol field.

    """

    model_type: Literal["csv"] = Field(
        default="csv",
        description="Model type discriminator",
    )
    filename: str | Path = Field(description="Path to the csv file")
    tcol: str = Field(
        default="time",
        description="Name of the column containing the time data",
    )
    read_csv_kwargs: dict = Field(
        default={},
        description="Keyword arguments to pass to pandas.read_csv",
    )

    @model_validator(mode="after")
    def validate_kwargs(self) -> "SourceTimeseriesCSV":
        """Validate the keyword arguments."""
        if "parse_dates" not in self.read_csv_kwargs:
            self.read_csv_kwargs["parse_dates"] = [self.tcol]
        if "index_col" not in self.read_csv_kwargs:
            self.read_csv_kwargs["index_col"] = self.tcol
        return self

    def _open_dataframe(self) -> pd.DataFrame:
        """Read the data from the csv file."""
        return pd.read_csv(self.filename, **self.read_csv_kwargs)

    def _open(self) -> xr.Dataset:
        """Interpolate the xyz data onto a regular grid."""
        df = self._open_dataframe()
        ds = xr.Dataset.from_dataframe(df).rename({self.tcol: "time"})
        return ds


class SourceTimeseriesDataFrame(SourceBase):
    """Source dataset from an existing pandas DataFrame timeseries object."""

    model_type: Literal["dataframe"] = Field(
        default="dataframe",
        description="Model type discriminator",
    )
    obj: pd.DataFrame = Field(
        description="pandas DataFrame object",
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("obj")
    @classmethod
    def is_timeseries(cls, obj: pd.DataFrame) -> pd.DataFrame:
        """Check if the DataFrame is a timeseries."""
        if not pd.api.types.is_datetime64_any_dtype(obj.index):
            raise ValueError("The DataFrame index must be datetime dtype")
        if obj.index.name is None:
            raise ValueError("The DataFrame index must have a name")
        return obj

    def __str__(self) -> str:
        return f"SourceTimeseriesDataFrame(obj={self.obj})"

    def _open(self) -> xr.Dataset:
        return xr.Dataset.from_dataframe(self.obj).rename({self.obj.index.name: "time"})
