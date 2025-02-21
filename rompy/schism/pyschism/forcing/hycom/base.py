import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, Union

import numpy as np
import tqdm
import tqdm_logging_wrapper
from matplotlib.transforms import Bbox
from netCDF4 import Dataset
from scipy.interpolate import griddata

from rompy.schism.pyschism import dates

from ..base import ModelForcing

logger = logging.getLogger(__name__)


class HycomComponent(ABC):
    """Base class for HYCOM component API"""

    @abstractmethod
    def get_datasets(
        self, start_date, run_days, output_interval=None
    ) -> Dict[datetime, Dataset]:
        """Returns a dictionary of relevant datasets."""

    @abstractmethod
    def put_boundary_ncdata(*args, **kwargs):
        """Interpolates HYCOM data to boundary."""

    @property
    @abstractmethod
    def ncvar(self) -> Union[str, tuple]:
        """Name of the variable or variables in the HYCOM dataset."""

    def _modified_bbox(self, dataset, bbox=None):
        # dataset = list(self.datasets.values())[-1]
        if bbox is None:
            return Bbox.from_extents(
                dataset["lon"][:].min(),
                dataset["lat"][:].min(),
                dataset["lon"][:].max(),
                dataset["lat"][:].max(),
            )
        else:
            xmin = (
                bbox.xmin + 360.0
                if not (bbox.xmin >= dataset["lon"][:].min() and bbox.xmin < 180.0)
                else bbox.xmin
            )

            xmax = (
                bbox.xmax + 360.0
                if not (bbox.xmax >= dataset["lon"][:].min() and bbox.xmax < 180.0)
                else bbox.xmax
            )

            return Bbox.from_extents(
                np.min([xmin, xmax]), bbox.ymin, np.max([xmin, xmax]), bbox.ymax
            )

    def _modified_bbox_indexes(self, bbox, dataset, pixel_buffer=0):
        # dataset = list(self.datasets.values())[-1]
        lat_idxs = np.where(
            (dataset["lat"][:] >= bbox.ymin) & (dataset["lat"][:] <= bbox.ymax)
        )[0]
        lon_idxs = np.where(
            (dataset["lon"][:] >= bbox.xmin) & (dataset["lon"][:] <= bbox.xmax)
        )[0]
        lon_idxs = lon_idxs.tolist()
        lat_idxs = lat_idxs.tolist()
        for i in range(pixel_buffer):
            lon_idxs.insert(0, lon_idxs[0] - 1)
            lon_idxs.append(lon_idxs[-1] + 1)
            lat_idxs.insert(0, lat_idxs[0] - 1)
            lat_idxs.append(lat_idxs[-1] + 1)
        return lon_idxs, lat_idxs


class Hycom(ModelForcing):

    @property
    @abstractmethod
    def elevation(self) -> HycomComponent:
        """Elevation component API of HYCOM dataset."""

    @property
    @abstractmethod
    def velocity(self) -> HycomComponent:
        """Elevation component API of HYCOM dataset."""

    @property
    @abstractmethod
    def temperature(self) -> HycomComponent:
        """Temperature component API of HYCOM dataset."""

    @property
    @abstractmethod
    def salinity(self) -> HycomComponent:
        """Temperature component API of HYCOM dataset."""
