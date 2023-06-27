"""Test readgrid sub-components."""
import pytest
import logging
from pydantic import ValidationError

from rompy.swan.subcomponents.readgrid import READGRID, READCOORD, READINP


logger = logging.getLogger(__name__)


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
