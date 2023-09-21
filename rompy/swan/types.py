"""Types for the swan wrapper."""
from enum import Enum, IntEnum


class IDLA(IntEnum):
    """Order of values in the input files.

    Attributes
    ----------
    ONE: 1
        SWAN reads the map from left to right starting in the upper-left-hand corner of
        the map. A new line in the map should start on a new line in the file.
    TWO: 2
        As `1` but a new line in the map need not start on a new line in the file.
    THREE: 3
        SWAN reads the map from left to right starting in the lower-left-hand corner of
        the map. A new line in the map should start on a new line in the file.
    FOUR: 4
        As `3` but a new line in the map need not start on a new line in the file.
    FIVE: 5
        SWAN reads the map from top to bottom starting in the lower-left-hand corner of
        the map. A new column in the map should start on a new line in the file.
    SIX: 6
        As `5` but a new column in the map need not start on a new line in the file.

    Notes
    -----
    It is assumed that the x-axis of the grid is pointing to the right and the y-axis
    upwards.

    """

    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6


class GridOptions(str, Enum):
    """Valid options for the input grid type.

    Attributes
    ----------
    BOTTOM: "bottom"
        Bottom level grid.
    WLEVEL: "wlevel"
        Water level grid.
    CURRENT: "current"
        Current field grid.
    VX: "vx"
        Current field x-component grid.
    VY: "vy"
        Current field y-component grid.
    WIND: "wind"
        Wind velocity grid.
    WX: "wx"
        Wind velocity x-component grid.
    WY: "wy"
        Wind velocity y-component grid.
    FRICTION: "friction"
        Bottom friction grid.
    NPLANTS: "nplants"
        Horizontally varying vegetation density grid.
    TURBVISC: "turbvisc"
        Horizontally varying turbulent viscosity grid.
    MUDLAYER: "mudlayer"
        Horizontally varying mud layer thickness grid.
    AICE: "aice"
        Areal ice fraction grid, a number between 0 and 1.
    HICE: "hice"
        Ice thickness grid.
    HSS: "hss"
        Sea-swell significant wave height grid.
    TSS: "tss"
        Sea-swell mean wave period.

    """

    BOTTOM = "bottom"
    WLEVEL = "wlevel"
    CURRENT = "current"
    VX = "vx"
    VY = "vy"
    WIND = "wind"
    WX = "wx"
    WY = "wy"
    FRICTION = "friction"
    NPLANTS = "nplants"
    TURBVISC = "turbvisc"
    MUDLAYER = "mudlayer"
    AICE = "aice"
    HICE = "hice"
    HSS = "hss"
    TSS = "tss"


class BoundShapeOptions(str, Enum):
    """Valid options for the boundary shape type.

    Attributes
    ----------
    JONSWAP: "jonswap"
        JONSWAP spectrum.
    PM: "pm"
        Pierson-Moskowitz spectrum.
    GAUSS: "gauss"
        Gaussian spectrum.
    BIN: "bin"
        Energy at a single bin spectrum.
    TMA: "tma"
        TMA spectrum.

    """

    JONSWAP = "jonswap"
    PM = "pm"
    GAUSS = "gauss"
    BIN = "bin"
    TMA = "tma"


class SideOptions(str, Enum):
    """Valid options for the boundary shape type.

    Attributes
    ----------
    NORTH: "north"
        North side.
    NW: "nw"
        North-west side.
    WEST: "west"
        West side.
    SW: "sw"
        South-west side.
    SOUTH: "south"
        South side.
    SE: "se"
        South-east side.
    EAST: "east"
        East side.
    NE: "ne"
        North-east side.

    """

    NORTH = "north"
    NW = "nw"
    WEST = "west"
    SW = "sw"
    SOUTH = "south"
    SE = "se"
    EAST = "east"
    NE = "ne"


class PhysicsOff(str, Enum):
    """Physics commands to be switched off.

    Attributes
    ----------
    WINDGROWTH : str = "windgrowth"
        Switches off wind growth (in commands GEN1, GEN2, GEN3).
    QUADRUPL : str = "quadrupl"
        Switches off quadruplet wave interactions (in command GEN3).
    WCAPPING : str = "wcapping"
        Switches off whitecapping (in command GEN3).
    BREAKING : str = "breaking"
        Switches off wave breaking dissipation.
    REFRAC : str = "refrac"
        Switches off wave refraction (action transport in theta space).
    FSHIFT : str = "fshift"
        Switches off frequency shifting (action transport in sigma space).
    BNDCHK : str = "bndchk"
        Switches off the checking of the delta imposed and computed Hs at the boundary.

    """

    WINDGROWTH = "windgrowth"
    QUADRUPL = "quadrupl"
    WCAPPING = "wcapping"
    BREAKING = "breaking"
    REFRAC = "refrac"
    FSHIFT = "fshift"
    BNDCHK = "bndchk"
