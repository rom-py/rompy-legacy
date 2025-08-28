"""Test cgrid component."""

import pytest

# Import test utilities
from test_utils.logging import get_test_logger

# Initialize logger
logger = get_test_logger(__name__)

from rompy.swan.subcomponents.readgrid import GRIDREGULAR
from rompy.swan.components.cgrid import (
    SPECTRUM,
    REGULAR,
    CURVILINEAR,
    UNSTRUCTURED,
)


@pytest.fixture(scope="module")
def spectrum():
    yield dict(mdc=36, flow=0.04, fhigh=0.4)


@pytest.fixture(scope="module")
def curvilinear_kwargs(spectrum):
    readcoord = {"fname": "grid_coord.txt"}
    yield dict(spectrum=spectrum, mxc=10, myc=10, readcoord=readcoord)


def test_spectrum(spectrum):
    spec = SPECTRUM(**spectrum)
    assert spec.mdc == spectrum["mdc"]
    assert spec.flow == spectrum["flow"]
    assert spec.fhigh == spectrum["fhigh"]


def test_spectrum_msc_gt_3():
    with pytest.raises(ValueError):
        SPECTRUM(mdc=36, msc=2)


def test_spectrum_circle(spectrum):
    spec = SPECTRUM(**spectrum)
    assert spec.dir_sector == "CIRCLE"


def test_spectrum_sector():
    spec = SPECTRUM(mdc=36, flow=0.04, fhigh=0.4, dir1=0.0, dir2=180.0)
    assert spec.dir_sector == "SECTOR 0.0 180.0"


def test_spectrum_dir1_and_dir2():
    with pytest.raises(ValueError):
        SPECTRUM(mdc=36, flow=0.04, fhigh=0.4, dir1=45.0)


def test_spectrum_freq_args_at_least_two():
    with pytest.raises(ValueError):
        SPECTRUM(mdc=36, flow=0.04)


def test_spectrum_flow_less_than_fhigh():
    with pytest.raises(ValueError):
        SPECTRUM(mdc=36, flow=0.4, fhigh=0.04)


def test_regular():
    cgrid = REGULAR(
        spectrum=SPECTRUM(
            mdc=36,
            flow=0.04,
            fhigh=0.4,
        ),
        grid=GRIDREGULAR(
            xp=0.0,
            yp=0.0,
            alp=0.0,
            xlen=100.0,
            ylen=100.0,
            mx=10,
            my=10,
        ),
    )


def test_curvilinear(curvilinear_kwargs):
    CURVILINEAR(**curvilinear_kwargs)


def test_curvilinear_exception(curvilinear_kwargs):
    CURVILINEAR(xexc=-999.0, yexc=-999.0, **curvilinear_kwargs)
    with pytest.raises(ValueError):
        CURVILINEAR(xexc=-999.0, **curvilinear_kwargs)
    with pytest.raises(ValueError):
        CURVILINEAR(yexc=-999.0, **curvilinear_kwargs)


def test_read_grid_coord_free_or_fixed_or_unformatted_only(curvilinear_kwargs):
    kwargs = curvilinear_kwargs.copy()
    kwargs["readcoord"] = {
        "fname": "coords.txt",
        "format": "fixed",
        "form": "(10X,12F5.0)",
    }
    CURVILINEAR(**kwargs)
    kwargs["readcoord"] = {"fname": "coords.txt", "format": "free"}
    CURVILINEAR(**kwargs)
    kwargs["readcoord"] = {"fname": "coords.txt", "format": "unformatted"}
    CURVILINEAR(**kwargs)
    with pytest.raises(ValueError):
        kwargs["readcoord"] = {"fname": "coords.txt", "format": "something_else"}
        CURVILINEAR(**kwargs)


def test_read_grid_coord_idfm_options(curvilinear_kwargs):
    kwargs = curvilinear_kwargs.copy()
    for idfm in [1, 5, 6, 8]:
        kwargs["readcoord"] = {"fname": "coords.txt", "format": "fixed", "idfm": idfm}
        CURVILINEAR(**kwargs)
    with pytest.raises(ValueError):
        kwargs["readcoord"] = {"fname": "coords.txt", "format": "fixed", "idfm": 2}
        CURVILINEAR(**kwargs)


def test_read_grid_fixed_format_arguments(curvilinear_kwargs):
    kwargs = curvilinear_kwargs.copy()
    kwargs["readcoord"] = {
        "fname": "coords.txt",
        "format": "fixed",
        "form": "(10X,12F5.0)",
    }
    CURVILINEAR(**kwargs)
    kwargs["readcoord"] = {"fname": "coords.txt", "format": "fixed", "idfm": 1}
    CURVILINEAR(**kwargs)
    with pytest.raises(ValueError):
        kwargs["readcoord"] = {
            "fname": "coords.txt",
            "format": "fixed",
            "form": "(10X,12F5.0)",
            "idfm": 1,
        }
        CURVILINEAR(**kwargs)
    with pytest.raises(ValueError):
        kwargs["readcoord"] = {"fname": "coords.txt", "format": "fixed"}
        CURVILINEAR(**kwargs)


def test_unstructured_adcirc(spectrum):
    UNSTRUCTURED(spectrum=spectrum)


def test_unstructured_triangle_easymesh(spectrum):
    UNSTRUCTURED(spectrum=spectrum, grid_type="triangle", fname="mesh.txt")
    UNSTRUCTURED(spectrum=spectrum, grid_type="easymesh", fname="mesh.txt")


def test_unstructured_grid_types(spectrum):
    with pytest.raises(ValueError):
        UNSTRUCTURED(spectrum=spectrum, grid_type="something_else")
