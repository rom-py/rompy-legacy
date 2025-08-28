"""Test swan_types."""

import pytest
# Import test utilities
from test_utils.logging import get_test_logger

# Initialize logger
logger = get_test_logger(__name__)

from rompy.swan.types import IDLA, BoundShapeOptions, GridOptions, SideOptions


def test_idla():
    assert IDLA(1) == IDLA.ONE == 1
    assert IDLA(2) == IDLA.TWO == 2
    assert IDLA(3) == IDLA.THREE == 3
    assert IDLA(4) == IDLA.FOUR == 4
    assert IDLA(5) == IDLA.FIVE == 5
    assert IDLA(6) == IDLA.SIX == 6
    with pytest.raises(ValueError):
        IDLA(7)


def test_grid_options():
    assert GridOptions.BOTTOM.name == "BOTTOM"
    assert GridOptions.BOTTOM.value == "bottom"
    assert GridOptions.WLEVEL.name == "WLEVEL"
    assert GridOptions.WLEVEL.value == "wlevel"
    assert GridOptions.CURRENT.name == "CURRENT"
    assert GridOptions.CURRENT.value == "current"
    assert GridOptions.VX.name == "VX"
    assert GridOptions.VX.value == "vx"
    assert GridOptions.VY.name == "VY"
    assert GridOptions.VY.value == "vy"
    assert GridOptions.WIND.name == "WIND"
    assert GridOptions.WIND.value == "wind"
    assert GridOptions.WX.name == "WX"
    assert GridOptions.WX.value == "wx"
    assert GridOptions.WY.name == "WY"
    assert GridOptions.WY.value == "wy"
    assert GridOptions.FRICTION.name == "FRICTION"
    assert GridOptions.FRICTION.value == "friction"
    assert GridOptions.NPLANTS.name == "NPLANTS"
    assert GridOptions.NPLANTS.value == "nplants"
    assert GridOptions.TURBVISC.name == "TURBVISC"
    assert GridOptions.TURBVISC.value == "turbvisc"
    assert GridOptions.MUDLAYER.name == "MUDLAYER"
    assert GridOptions.MUDLAYER.value == "mudlayer"
    assert GridOptions.AICE.name == "AICE"
    assert GridOptions.AICE.value == "aice"
    assert GridOptions.HICE.name == "HICE"
    assert GridOptions.HICE.value == "hice"
    assert GridOptions.HSS.name == "HSS"
    assert GridOptions.HSS.value == "hss"
    assert GridOptions.TSS.name == "TSS"
    assert GridOptions.TSS.value == "tss"
    for option in GridOptions:
        assert option.name == option.value.upper()
    with pytest.raises(ValueError):
        GridOptions("invalid")


def test_bound_shape_options():
    assert BoundShapeOptions.JONSWAP.name == "JONSWAP"
    assert BoundShapeOptions.JONSWAP.value == "jonswap"
    assert BoundShapeOptions.PM.name == "PM"
    assert BoundShapeOptions.PM.value == "pm"
    assert BoundShapeOptions.GAUSS.name == "GAUSS"
    assert BoundShapeOptions.GAUSS.value == "gauss"
    assert BoundShapeOptions.BIN.name == "BIN"
    assert BoundShapeOptions.BIN.value == "bin"
    assert BoundShapeOptions.TMA.name == "TMA"
    assert BoundShapeOptions.TMA.value == "tma"
    for option in BoundShapeOptions:
        assert option.name == option.value.upper()
    with pytest.raises(ValueError):
        BoundShapeOptions("square")


def test_side_options():
    assert SideOptions.NORTH.name == "NORTH"
    assert SideOptions.NORTH.value == "north"
    assert SideOptions.NW.name == "NW"
    assert SideOptions.NW.value == "nw"
    assert SideOptions.WEST.name == "WEST"
    assert SideOptions.WEST.value == "west"
    assert SideOptions.SW.name == "SW"
    assert SideOptions.SW.value == "sw"
    assert SideOptions.SOUTH.name == "SOUTH"
    assert SideOptions.SOUTH.value == "south"
    assert SideOptions.SE.name == "SE"
    assert SideOptions.SE.value == "se"
    assert SideOptions.EAST.name == "EAST"
    assert SideOptions.EAST.value == "east"
    assert SideOptions.NE.name == "NE"
    assert SideOptions.NE.value == "ne"
    for option in SideOptions:
        assert option.name == option.value.upper()
    with pytest.raises(ValueError):
        SideOptions("center")
