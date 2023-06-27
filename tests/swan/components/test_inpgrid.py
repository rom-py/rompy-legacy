"""Test inpgrid component."""
import pytest
import logging
from pydantic import ValidationError

from rompy.swan.types import GridOptions
from rompy.swan.components.inpgrid import (
    INPGRID,
    INPGRIDS,
    REGULAR,
    CURVILINEAR,
    UNSTRUCTURED,
    READINP,
)
from rompy.swan.subcomponents.time import NONSTATIONARY


logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def readinp():
    yield READINP(fname1="test.txt")


@pytest.fixture(scope="module")
def nonstat():
    inst = NONSTATIONARY(
        tbeg="2023-01-01T00:00:00",
        delt="PT30M",
        tend="2023-02-01T00:00:00",
        deltfmt="hr",
    )
    yield inst


def test_valid_inpgrid_options(readinp):
    for inpgrid in GridOptions:
        INPGRID(grid_type=inpgrid.value, readinp=readinp)
    with pytest.raises(ValidationError):
        INPGRID(grid_type="invalid", readinp=readinp)


def test_excval(readinp):
    inpgrid = INPGRID(grid_type="bottom", excval=-999, readinp=readinp)
    assert inpgrid.excval == -999.0
    assert isinstance(inpgrid.excval, float)


def test_inpgrid_nonstationary(nonstat, readinp):
    inpgrid = INPGRID(grid_type="bottom", nonstationary=nonstat, readinp=readinp)
    assert isinstance(inpgrid.nonstationary, NONSTATIONARY)
    assert inpgrid.nonstationary.suffix == "inp"


def test_inpgrid_regular(nonstat, readinp):
    inpgrid = REGULAR(
        grid_type=GridOptions.wind,
        xpinp=0.0,
        ypinp=0.0,
        alpinp=0.0,
        mxinp=10,
        myinp=10,
        dxinp=0.1,
        dyinp=0.1,
        excval=-999.0,
        nonstationary=nonstat,
        readinp=readinp,
    )
    logger.info(inpgrid.render())


def test_inpgrid_curvilinear(nonstat, readinp):
    inpgrid = CURVILINEAR(
        grid_type=GridOptions.wind,
        mxinp=10,
        myinp=10,
        excval=-999.0,
        nonstationary=nonstat,
        readinp=readinp,
    )
    logger.info(inpgrid.render())


def test_inpgrid_curvilinear_render(nonstat, readinp):
    inpgrid = UNSTRUCTURED(
        grid_type=GridOptions.wind,
        excval=-999.0,
        nonstationary=nonstat,
        readinp=readinp,
    )
    logger.info(inpgrid.render())


def test_inpgrids(nonstat, readinp):
    bottom = REGULAR(
        grid_type=GridOptions.bottom,
        xpinp=0.0,
        ypinp=0.0,
        alpinp=0.0,
        mxinp=10,
        myinp=10,
        dxinp=0.1,
        dyinp=0.1,
        excval=-999.0,
        readinp=readinp,
    )
    wind = REGULAR(
        grid_type=GridOptions.wind,
        xpinp=0.0,
        ypinp=0.0,
        alpinp=0.0,
        mxinp=10,
        myinp=10,
        dxinp=0.1,
        dyinp=0.1,
        excval=-999.0,
        nonstationary=nonstat,
        readinp=readinp,
    )

    inpgrids = INPGRIDS(
        inpgrids=[bottom, wind]
    )
    logger.info(inpgrids.render())