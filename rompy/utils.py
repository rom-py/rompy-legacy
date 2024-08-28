# -----------------------------------------------------------------------------
# Copyright (c) 2020 - 2021, CSIRO
#
# All rights reserved.
#
# The full license is in the LICENSE file, distributed with this software.
# -----------------------------------------------------------------------------

import importlib
import logging
from typing import Union

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import KDTree

logger = logging.getLogger("rompy.util")


def dict_product(d):
    from itertools import product

    keys = d.keys()
    for element in product(*d.values()):
        yield dict(zip(keys, element))


def walk_server(urlpath, fn_fmt, fmt_fields, url_replace):
    from functools import reduce
    from operator import iconcat

    from dask import compute, delayed

    # Targetted scans of the file system based on date range
    test_urls = set([urlpath.format(**pv) for pv in dict_product(fmt_fields)])
    test_fns = set([fn_fmt.format(**pv) for pv in dict_product(fmt_fields)])

    logger.debug(f"Test URLS : {test_urls}")

    @delayed
    def check_url(test_url, test_fns):
        from os.path import dirname

        from fsspec import filesystem
        from fsspec.utils import get_protocol

        fs = filesystem(get_protocol(test_url))
        logger.debug(f"testing {test_url}")
        urls = []
        if fs.exists(test_url):
            for url, _, links in fs.walk(test_url):
                # test for case that url is a local directory, otherwise likely a http url
                if fs.isfile(url):
                    url = dirname(url)
                urls += [url + "/" + fn for fn in links if fn in test_fns]
        return urls

    valid_urls = compute(
        *[check_url(test_url, test_fns) for test_url in test_urls],
        traverse=False,
        scheduler="threads",
    )
    # valid_urls = [check_url(test_url,test_fns) for test_url in test_urls]
    valid_urls = sorted(reduce(iconcat, valid_urls, []))

    logger.debug(f"valid_urls : {valid_urls}")

    for f, r in url_replace.items():
        valid_urls = [u.replace(f, r) for u in valid_urls]

    logger.debug(f"valid_urls after replace : {valid_urls}")

    return valid_urls


def find_matchup_data(
    measurement, model, var_map, time_thresh=None, KDtree_kwargs={}, metadata={}
):
    """
    Finds nearest points between observed data and model output and returns corresonding nearest variable.

    Parameters
    ----------
    measurement : xarray.dataset or pandas.dataframe
        Dataset containing measurements
    model : xarray.dataset
        Dataset containing model output - currently only tested for regular grid
    var_map: dict
        Dictionary of key maps from variables in "measurement" to corresponding variable in "model"
    time_thresh: 'None' (default), int or numpy.timedelta64
        Time threshold for finding matching measurements and model outputs.
            None (Defaults): within 30 mins
            int:  passes int to np.timedelta(int,'m')
            np.timedelta
    KDtree_kwargs: dict
        Dictionary passed to scipy.spatial.KDtree function
    metadata: dict
        Dictionary passed to output ds for user-provided metadata

    Returns
    ----------
    ds: xarray.dataset
        Xarray dataset containing measurements and nearest model outputs
    """

    ### Remove case-sensitivity from measurement dataframe/ds by making everything lowercase, try to make the lat/lon/time calls a little more robust
    if type(measurement) == xr.Dataset:
        name_dict = dict(
            zip(
                list(measurement.variables),
                [a.lower() for a in list(measurement.variables)],
            )
        )
        measurement = measurement.rename(name_dict)
    elif type(measurement) == pd.DataFrame:
        name_dict = dict(
            zip(list(measurement.keys()), [a.lower() for a in list(measurement.keys())])
        )
        measurement = measurement.rename(columns=name_dict)

    var_map = dict(
        (meas_key.lower(), model_key) for meas_key, model_key in var_map.items()
    )

    ### Set time threshold
    if not time_thresh:
        time_thresh = np.timedelta64(30, "m")  ## Defaults to 30 mins
    elif type(time_thresh) == int:
        time_thresh = np.timedelta64(
            time_thresh, "m"
        )  ## Otherwise if user passes int or float turn this into td64
    elif type(time_thresh) == np.timedelta64:
        None
    else:
        raise ValueError(
            'Unrecognised input for "time_thresh", must be "int", "np.timedelta64" or "None"'
        )

    #### Find Indices of nearest point
    lats = model["latitude"].values
    lons = model["longitude"].values
    dummy_var = model[list(var_map.items())[0][1]]  ### Pull out the first key

    if (len(lats.shape) == 1) and (len(dummy_var.shape) == 3):  # assumes time, x, y
        grid = "regular"
        mesh_lat, mesh_lon = np.meshgrid(lats, lons, indexing="ij")
    elif (len(lats.shape) == 1) and (
        len(dummy_var.shape) == 2
    ):  # assumes time, element
        grid = "unstructured"
        mesh_lat, mesh_lon = lats, lons
    elif (len(lats.shape) == 2) and (len(dummy_var.shape) == 3):  # assumes time, x, y
        grid = "curvilinear"  # Curvilinear
        mesh_lat, mesh_lon = lats, lons
    else:
        raise ValueError("Model dataset has an unsupported grid type")

    tree = KDTree(list(zip(mesh_lat.ravel(), mesh_lon.ravel())), **KDtree_kwargs)
    dist, grid_idx_r = tree.query(
        list(zip(measurement["latitude"], measurement["longitude"]))
    )

    if grid in ["regular", "curvilinear"]:
        grid_idx_lat, grid_idx_lon = np.unravel_index(grid_idx_r, mesh_lon.shape)

    ##################
    ### Loop through time, check if timestamps are within thresh and get indices
    meas_times = measurement.time.values
    model_times = model.time.values

    ## Initialise a dict that we can append to
    measurement_idx = []
    model_time_idx = []

    for i, time in enumerate(meas_times):
        inds = np.argwhere(
            np.abs(model_times - time) < time_thresh
        )  ## within time_thresh
        if inds.size > 0:
            for time_idx in inds:
                measurement_idx.append(i)
                model_time_idx.append(int(time_idx))

    ######## Now retrieve data from model and measurements for indices
    model_time_idx = xr.DataArray(model_time_idx, dims="observation")
    model_lat_idx = xr.DataArray(grid_idx_lat[measurement_idx], dims="observation")
    model_lon_idx = xr.DataArray(grid_idx_lon[measurement_idx], dims="observation")

    model_results = model[list(var_map.values())].isel(
        time=model_time_idx, latitude=model_lat_idx, longitude=model_lon_idx
    )

    measurement_keys = ["time", "longitude", "latitude"] + list(var_map.keys())
    if type(measurement) == pd.DataFrame:
        measurement_results = measurement[measurement_keys].loc[measurement_idx]
    elif type(measurement) == xr.Dataset:
        meas_time_idx = xr.DataArray(measurement_idx, dims="observation")
        measurement_results = measurement[measurement_keys].isel(time=meas_time_idx)

    ###################################################
    ### Initialise an output xarray dataset
    out_ds = xr.Dataset()

    for key in list(
        measurement_results.keys()
    ):  ### first lets add the measurement data
        out_ds["meas_" + key] = xr.DataArray(
            measurement_results[key].values, dims=["observation"]
        )

    for key in list(model_results.variables):  ### then the model data
        out_ds["model_" + key] = xr.DataArray(model_results[key], dims=["observation"])

    out_ds["dist"] = xr.DataArray(
        dist[measurement_idx],
        dims=["observation"],
        attrs={
            "long_name": "Distance from observation to nearest model cell",
            "units": "dDegrees",
        },
    )  ### Add distances from KD tree

    ### Add in attributes - would be nice to update the drivers to convert straight to nc's with catalog params which we could add here!
    out_ds.attrs["grid"] = grid
    attrs_prepend_map = {"metadata": "metadata_", "KDtree_kwargs": "KDtree_"}
    for attr_source in [metadata, KDtree_kwargs]:
        if attr_source:
            for val, key in attr_source.items():
                if type(val) in [
                    str,
                    int,
                    float,
                    np.int64,
                    np.int32,
                    np.float64,
                    np.float32,
                ]:  # Any other types could go here
                    out_ds.attrs[attrs_prepend_map[str(attr_source)] + key] = val

    return out_ds


def total_seconds(timedelta):
    """Convert timedeltas to seconds

    In Python, time differences can take many formats. This function can take
    timedeltas in any format and return the corresponding number of seconds, as
    a float.

    Beware! Representing timedeltas as floats is not as precise as representing
    them as a timedelta object in datetime, numpy, or pandas.

    Parameters
    ----------
    timedelta : various
        Time delta from python's datetime library or from numpy or pandas. If
        it is from numpy, it can be an ndarray with dtype datetime64. If it is
        from pandas, it can also be a Series of datetimes. However, this
        function cannot operate on entire pandas DataFrames. To convert a
        DataFrame, do df.apply(to_seconds)

    Returns
    -------
    seconds : various
        Returns the total seconds in the input timedelta object(s) as float.
        If the input is a numpy ndarray or pandas Series, the output is the
        same, but with a float datatype.
    """
    try:
        seconds = timedelta.total_seconds()
    except AttributeError:  # no method total_seconds
        one_second = np.timedelta64(1000000000, "ns")
        # use nanoseconds to get highest possible precision in output
        seconds = timedelta / one_second
    return seconds


def get_class_from_module(module_name, class_name):
    """Dynamically import classes from a given module"""
    try:
        # Dynamically import the module
        module = importlib.import_module(module_name)

        # Check for attribute presence and if it's a class
        if hasattr(module, class_name):
            attr = getattr(module, class_name)
            if isinstance(attr, type):  # Make sure it's a class
                return attr
        return None
    except ImportError:
        print(f"Error importing module {module_name}")
        return None
