"""Dataset transform functions."""

import logging
from collections.abc import Iterable
import xarray as xr


logger = logging.getLogger(__file__)


def sortby(dset: xr.Dataset, dims: Iterable):
    """Sort dataset along dimensions specified in dims.

    parameters
    ----------
    dset: xr.Dataset
        Input dataset to transform.
    dims: Iterable
        Dimensions along which dataset is sorted.

    Returns
    -------
    dset: xr.Dataset
        Transformed dataset.

    """
    for dim in set(dims).intersection(dset.dims):
        dset = dset.sortby(dim)
    return dset


def subset(dset: xr.Dataset, data_vars: Iterable):
    """Subset data variables from dataset.

    parameters
    ----------
    dset: xr.Dataset
        Input dataset to transform.
    dims: Iterable
        Variables to subset from dset.

    Returns
    -------
    dset: xr.Dataset
        Transformed dataset.

    """
    return dset[data_vars]


def crop(dset: xr.Dataset, **data_slice):
    """Crop dataset.

    parameters
    ----------
    dset: xr.Dataset
        Input dataset to transform.
    data_slice: Iterable
        Variables to subset from dset.

    Returns
    -------
    dset: xr.Dataset
        Transformed dataset.

    """
    this_crop = {k: data_slice[k] for k in data_slice.keys() if k in dset.dims.keys()}
    dset = dset.sel(this_crop)
    for k in data_slice.keys():
        if (k not in dset.dims.keys()) and (k in dset.coords.keys()):
            dset = dset.where(dset[k] > float(data_slice[k][0]), drop=True)
            dset = dset.where(dset[k] < float(data_slice[k][1]), drop=True)
    return dset


def timenorm(ds, interval="hour", reftime=None):
    from pandas import to_timedelta, to_datetime

    dt = to_timedelta("1 " + interval)
    if reftime is None:
        ds["init"] = ds["time"][0]
    else:
        ds["init"] = (("time",), to_datetime(ds[reftime].values))
    ds["lead"] = ((ds["time"] - ds["init"]) / dt).astype("int")
    ds["lead"].attrs["units"] = interval
    ds = ds.set_coords("init")
    ds = ds.swap_dims({"time": "lead"})
    return ds


def rename(ds, **varmap):
    ds = ds.rename(varmap)
    return ds
