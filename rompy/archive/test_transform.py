"""Test transform functions."""
import pytest
import xarray as xr

from rompy import transform


@pytest.fixture(scope="function")
def dset():
    dset = xr.Dataset()
    dset["wspd"] = xr.DataArray(
        data=[[1.0, 3.5, 1.2], [2.0, 3.0, 4.0]],
        coords={"y": [10, 20], "x": [0, 2, 1]}
    )
    dset["wdir"] = dset.wspd * 10
    yield dset


def test_sortby(dset):
    dsout = transform.sortby(dset, ["x"])
    assert dsout.equals(dset.sortby("x"))


def test_subset(dset):
    dsout = transform.subset(dset, ["wdir"])
    assert set(dsout.data_vars) == {"wdir"}


def test_crop(dset):
    pass


def test_timenorm(dset):
    pass


def test_rename(dset):
    pass
