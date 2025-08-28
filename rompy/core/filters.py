# -----------------------------------------------------------------------------
# Copyright (c) 2020 - 2021, CSIRO
#
# All rights reserved.
#
# The full license is in the LICENSE file, distributed with this software.
# -----------------------------------------------------------------------------

from typing import Optional

import xarray as xr
from pydantic import field_validator

from .types import RompyBaseModel, Slice


# pydantic class to apply all the filters to the dataset
class Filter(RompyBaseModel):
    sort: Optional[dict] = {}
    subset: Optional[dict] = {}
    crop: Optional[dict] = {}
    timenorm: Optional[dict] = {}
    rename: Optional[dict] = {}
    derived: Optional[dict] = {}

    @field_validator("crop", mode="before")
    def convert_slices(cls, v):
        for key, value in v.items():
            if isinstance(value, slice):
                v[key] = Slice.from_slice(value)
            if isinstance(value, dict):
                v[key] = Slice.from_dict(value)
        return v

    def __call__(self, ds):
        filters = get_filter_fns()
        for fn in filters:
            params = getattr(self, fn)
            if params:
                ds = filters[fn](ds, **params)
        return ds

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Filter(sort={self.sort}, subset={self.subset}, crop={self.crop}, timenorm={self.timenorm}, rename={self.rename}, derived={self.derived})"


def derived_filter(ds, derived_variables):
    """Add derived variable to Dataset.

    Parameters
    ----------
    ds: xarray.Dataset
        Input dataset to add derived variables to.
    derived_variables: dict
        Mapping {`derived_variable_name`: `derived_variable_definition`} where
        `derived_variable_definition` is a string to be evaluated defining some
        transformation based on existing variables in the input dataset `ds`.

    Returns
    -------
    ds: xarray.Dataset
        Input dataset with extra derived variables.

    Example
    -------
    >>> import xarray as xr
    >>> ds = xr.DataArray([-10, -11], coords={"x": [0, 1]}).to_dataset(name="elevation")
    >>> ds = derived_filter(ds, {"depth": "ds.elevation * -1"})

    """
    for var, expr in derived_variables.items():
        ds[var] = eval(expr)
    return ds


def sort_filter(ds, coords: list = []):
    coords = list[coords] if isinstance(coords, str) else coords
    for c in coords:
        if c in ds:
            ds = ds.sortby(c)
    return ds


def subset_filter(ds, data_vars=None) -> xr.Dataset:
    """
    Subset data variables from dataset.

    parameters
    ----------
    ds: xr.Dataset
        Input dataset to transform.
    data_vars: Iterable
        Variables to subset from ds.

    Returns
    -------
    ds: xr.Dataset
    """
    if data_vars is not None:
        ds = ds[data_vars]
    return ds


def crop_filter(ds, **data_slice) -> xr.Dataset:
    """
    Crop dataset.

    parameters
    ----------
    ds: xr.Dataset
        Input dataset to transform.
    data_slice: Iterable
        Data slice to crop

    Returns
    -------
    ds: xr.Dataset

    """
    if data_slice is not None:
        this_crop = {
            k: data_slice[k].to_slice()
            for k in data_slice.keys()
            if k in ds.sizes.keys()
        }
        ds = ds.sel(this_crop)
        for k in data_slice.keys():
            if (k not in ds.sizes.keys()) and (k in ds.coords.keys()):
                ds = ds.where(ds[k] > float(data_slice[k].start), drop=True)
                ds = ds.where(ds[k] < float(data_slice[k].stop), drop=True)
    return ds


def timenorm_filter(ds, interval="hour", reftime=None) -> xr.Dataset:
    """Normalize time to lead time in hours

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset to transform.
    interval : str, optional
        Time interval to normalize to, by default "hour"
    reftime : str, optional
        Reference time variable, by default None

    Returns
    -------
    ds : xr.Dataset
    """
    from pandas import to_datetime, to_timedelta

    dt = to_timedelta("1 " + interval)
    if reftime is None:
        ds["init"] = (
            ("time",),
            [
                ds["time"].values[0],
            ],
        )
    else:
        ds["init"] = (("time",), to_datetime(ds[reftime].values))
    ds["lead"] = ((ds["time"] - ds["init"]) / dt).astype("int")
    ds["lead"].attrs["units"] = interval
    ds = ds.set_coords("init")
    ds = ds.swap_dims({"time": "lead"})
    return ds


def rename_filter(ds, **varmap) -> xr.Dataset:
    """Rename variables in dataset

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset to transform.
    varmap : dict
        Dictionary of variable names to rename

    Returns
    -------
    ds : xr.Dataset
    """
    ds = ds.rename(varmap)
    return ds


def get_filter_fns() -> dict:
    """Get dictionary of filter functions"""
    return {
        "sort": sort_filter,
        "subset": subset_filter,
        "crop": crop_filter,
        "timenorm": timenorm_filter,
        "rename": rename_filter,
        "derived": derived_filter,
    }


def _open_preprocess(url, chunks, filters, xarray_kwargs):
    import xarray as xr

    ds = xr.open_dataset(url, chunks=chunks, **xarray_kwargs)
    filter_fns = get_filter_fns()
    for fn, params in filters.items():
        if isinstance(fn, str):
            fn = filter_fns[fn]
        ds = fn(ds, **params)

    return ds
