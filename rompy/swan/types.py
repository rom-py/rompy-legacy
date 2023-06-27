"""Types for the swan wrapper."""
from enum import Enum


class GridOptions(str, Enum):
    """Valid options for the input grid type."""
    bottom = "bottom"
    wlevel = "wlevel"
    current = "current"
    vx = "vx"
    vy = "vy"
    wind = "wind"
    wx = "wx"
    wy = "wy"
    friction = "friction"
    nplants = "nplants"
    turbvisc = "turbvisc"
    mudlayer = "mudlayer"
    aice = "aice"
    hice = "hice"
    hss = "hss"
    tss = "tss"


class BoundShapeOptions(str, Enum):
    """Valid options for the boundary shape type."""
    jonswap = "jonswap"
    pm = "pm"
    gauss = "gauss"
    bin = "bin"
    tma = "tma"


class SideOptions(str, Enum):
    """Valid options for the boundary shape type."""
    north = "north"
    nw = "nw"
    west = "west"
    sw = "sw"
    south = "south"
    se = "se"
    east = "east"
    ne = "ne"
    k = "k"