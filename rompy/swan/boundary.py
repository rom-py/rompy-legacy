"""SWAN boundary classes."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Optional
import logging

import intake
import numpy as np
import wavespectra
import xarray as xr
from pydantic import Field, confloat, root_validator

from rompy.core.data import DataBlob
from rompy.core.filters import Filter
from rompy.core.time import TimeRange
from rompy.core.types import RompyBaseModel
from rompy.swan.grid import SwanGrid

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


class Dataset(RompyBaseModel, ABC):
    """Abstract base class for a dataset."""

    @abstractmethod
    def open(self):
        """Return a dataset instance with the wavespectra accessor."""
        pass


class DatasetXarray(Dataset):
    """Wavespectra dataset from xarray reader."""

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


class DatasetIntake(Dataset):
    """Wavespectra dataset from intake catalog."""

    model_type: Literal["intake"] = Field(
        default="intake",
        description="Model type discriminator",
    )
    dataset_id: str = Field(description="The id of the dataset to read in the catalog")
    catalog_uri: str | Path = Field(description="The URI of the catalog to read from")
    kwargs: dict = Field(
        default={},
        description="Keyword arguments to pass to intake.open_catalog",
    )

    def open(self):
        cat = intake.open_catalog(self.catalog_uri)
        return cat[self.dataset_id](**self.kwargs).to_dask()


class DatasetWavespectra(Dataset):
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
        description="Keyword arguments to pass to wavespectra.read_swan",
    )

    def open(self):
        return getattr(wavespectra, self.reader)(self.uri, **self.kwargs)


class DataBoundary(RompyBaseModel):
    """SWAN BOUNDNEST1 NEST data class.

    Notes
    -----
    The `tolerance` behaves differently with sel_methods `idw` and `nearest`; in `idw`
    sites with no enough neighbours within `tolerance` are masked whereas in `nearest`
    an exception is raised (see wavespectra documentation for more details).

    Be aware that when using `idw` missing values will be returned for sites with less
    than 2 neighbours within `tolerance` in the original dataset. This is okay for land
    mask areas but could cause boundary issues when on an open boundary location. To
    avoid this either use `nearest` or increase `tolerance` to include more neighbours.

    """

    id: str = Field(description="Unique identifier for this data source")
    dataset: DatasetXarray | DatasetIntake | DatasetWavespectra = Field(
        description="Dataset reader, must return a wavespectra-enabled xarray dataset in the open method",
        discriminator="model_type",
    )
    spacing: Optional[float] = Field(
        description="Spacing between boundary points, by default defined as the minimum distance between points in the dataset",
    )
    sel_method: Literal["idw", "nearest"] = Field(
        default="idw",
        description="Wavespectra method to use for selecting boundary points from the dataset",
    )
    tolerance: confloat(ge=0) = Field(
        default=1.0,
        description="Wavespectra tolerance for selecting boundary points from the dataset",
    )
    rectangle: Literal["closed", "open"] = Field(
        default="closed",
        description="Defines whether boundary is defined over an closed or open rectangle",
    )
    filter: Optional[Filter] = Field(
        default=Filter(),
        description="Optional filter specification to apply to the dataset",
    )

    @root_validator
    def assert_has_wavespectra_accessor(cls, values):
        dataset = values.get("dataset")
        if dataset is not None:
            dset = dataset.open()
            if not hasattr(dset, "spec"):
                raise ValueError(f"Wavespectra compatible dataset is required")
        return values

    @property
    def ds(self):
        """Return the filtered xarray dataset instance."""
        dset = self.filter(self.dataset.open())
        if dset.efth.size == 0:
            raise ValueError(f"Empty dataset after applying filter {self.filter}")
        return dset

    def _boundary_resolutions(self, grid):
        """Boundary resolution based on the shortest distance between points.

        The boundary resolution should be based on the dataset resolution instead of
        the grid resolution to avoid creating points unecessarily. Here we find the
        minimum distance between points in the dataset and use that to define the
        boundary resolution ensuring the grid sizes are divisible by the resolution.

        """
        # Find out the minimum distance between points in the original dataset
        buffer = 2 * min(grid.dx, grid.dy)
        x0, y0, x1, y1 = grid.bbox(buffer=buffer)
        # Select dataset points just outside the actual grid to optimise the search
        ds = self.ds.spec.sel([x0, x1], [y0, y1], method="bbox")
        points = list(zip(ds.lon.values, ds.lat.values))
        min_distance = find_minimum_distance(points)
        # Calculate resolutions ensuring at least 3 points per side
        xlen = grid.maxx - grid.minx
        nx = max(xlen // min_distance, 3)
        dx = xlen / nx
        ylen = grid.maxy - grid.miny
        ny = max(ylen // min_distance, 3)
        dy = ylen / ny
        return dx, dy

    def _boundary_points(self, grid):
        """Coordinates of boundary points based on grid bbox and dataset resolution."""
        if self.spacing is None:
            dx, dy = self._boundary_resolutions(grid)
            spacing = min(dx, dy)
        else:
            spacing = self.spacing
        points = grid.points_along_boundary(spacing=spacing)
        if len(points.geoms) < 4:
            logger.warning(
                f"There are only {len(points)} boundary points (less than 1 point per grid side), "
                f"consider setting a smaller spacing (the current spacing is {spacing})"
            )
        xbnd = np.array([p.x for p in points.geoms])
        ybnd = np.array([p.y for p in points.geoms])
        return xbnd, ybnd

    def _filter_grid(self, grid, buffer=0.1):
        """Required by SwanForcing"""
        pass

    def _filter_time(self, time: TimeRange):
        """Time filter"""
        self.filter.crop.update({"time": slice(time.start, time.end)})

    def get(self, stage_dir: str, grid: SwanGrid) -> str:
        """Write the data source to a new location"""
        xbnd, ybnd = self._boundary_points(grid)
        ds = self.ds.spec.sel(
            lons=xbnd,
            lats=ybnd,
            method=self.sel_method,
            tolerance=self.tolerance,
        )
        filename = f"{self.id}.bnd"
        filepath = Path(stage_dir) / filename
        ds.spec.to_swan(filepath)
        cmd = f"BOUNDNEST1 NEST '{filename}' {self.rectangle.upper()}"
        return cmd
