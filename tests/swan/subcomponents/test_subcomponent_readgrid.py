"""Test readgrid sub-components."""

import logging
import pytest

from pydantic import ValidationError
from rompy.swan.subcomponents.readgrid import READGRID, READCOORD, READINP, GRIDREGULAR

# Import test utilities
from test_utils.logging import get_test_logger

# Initialize logger
logger = get_test_logger(__name__)


def test_readgrid_fac():
    READGRID(grid_type="coordinates", fac=1.0)
    with pytest.raises(ValidationError):
        READGRID(grid_type="coordinates", fac=0.0)
    with pytest.raises(ValidationError):
        READGRID(grid_type="coordinates", fac=-1.0)


def test_readgrid_wrong_format():
    with pytest.raises(ValidationError):
        READGRID(
            grid_type="coordinates",
            format="invalid",
        )


def test_readgrid_free():
    readgrid = READGRID(
        grid_type="coordinates",
        format="free",
    )
    assert readgrid.format_repr == "FREE"


def test_readgrid_unformatted():
    readgrid = READGRID(
        grid_type="coordinates",
        format="unformatted",
    )
    assert readgrid.format_repr == "UNFORMATTED"


def test_readgrid_fixed_form():
    readgrid = READGRID(
        grid_type="coordinates",
        format="fixed",
        form="(10X,12F5.0)",
    )
    assert readgrid.format_repr == f"FORMAT form='(10X,12F5.0)'"


def test_readgrid_fixed_idfm():
    with pytest.raises(ValidationError):
        READGRID(
            grid_type="coordinates",
            format="fixed",
            idfm=2,
        )
    readgrid = READGRID(
        grid_type="coordinates",
        format="fixed",
        idfm=1,
    )
    assert readgrid.format_repr == f"FORMAT idfm=1"


def test_coord():
    readcoord = READCOORD(fname="grid_coords.txt")
    logger.info(readcoord.render())


def test_grid_regular():
    grid = GRIDREGULAR(
        xp=120.0,
        yp=-30.0,
        alp=0.0,
        xlen=10,
        ylen=10,
        mx=20,
        my=20,
        suffix="c",
    )
    assert grid.dx == 0.5 and grid.dy == 0.5
    assert grid.render() == (
        "xpc=120.0 ypc=-30.0 alpc=0.0 xlenc=10.0 ylenc=10.0 mxc=20 myc=20"
    )
