import numpy as np
import pytest
import shapely

from rompy.swan import SwanGrid
from rompy.swan.subcomponents.readgrid import GRIDREGULAR


# Import test utilities
from test_utils.logging import get_test_logger

# Initialize logger
logger = get_test_logger(__name__)


# test class based on pytest fixtures
@pytest.fixture
def grid():
    return SwanGrid(x0=0, y0=0, nx=10, ny=10, dx=1, dy=1)


@pytest.fixture
def grid2():
    return SwanGrid(x0=0, y0=0, dx=1, dy=1, nx=10, ny=10)


def test_bbox(grid):
    assert grid.bbox() == [0.0, 0.0, 9.0, 9.0]


def test_boundary(grid):
    bnd = grid.boundary()
    assert isinstance(bnd, shapely.geometry.polygon.Polygon)


def test_grid_shape(grid):
    assert grid.x.shape == (10, 10)
    assert grid.y.shape == (10, 10)


def test_grid_size(grid):
    assert grid.x.size == 100
    assert grid.y.size == 100


def test_grid_ndim(grid):
    assert grid.x.ndim == 2
    assert grid.y.ndim == 2


def test_grid_minmax(grid):
    assert grid.minx == 0
    assert grid.miny == 0
    assert grid.maxx == 9
    assert grid.maxy == 9


def test_initilisation(grid, grid2):
    assert np.array_equal(grid.x, grid2.x)
    assert np.array_equal(grid.y, grid2.y)


def test_component_attribute(grid):
    regular_grid_component = GRIDREGULAR(
        xp=grid.x0,
        yp=grid.y0,
        alp=grid.rot,
        xlen=grid.xlen,
        ylen=grid.ylen,
        mx=grid.nx - 1,
        my=grid.ny - 1,
    )
    assert grid.component == regular_grid_component


def test_grid_from_component(grid):
    regular_grid_component = GRIDREGULAR(
        xp=grid.x0,
        yp=grid.y0,
        alp=grid.rot,
        xlen=grid.xlen,
        ylen=grid.ylen,
        mx=grid.nx - 1,
        my=grid.ny - 1,
    )
    grid2 = SwanGrid.from_component(regular_grid_component)
    assert grid == grid2
