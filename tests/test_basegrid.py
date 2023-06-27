import numpy as np
import pytest
import shapely

from rompy.core import BaseGrid, RegularGrid


# test class based on pytest fixtures
@pytest.fixture
def grid():
    x = np.arange(10)
    y = np.arange(10)
    xx, yy = np.meshgrid(x, y)
    return BaseGrid(x=xx, y=yy)


@pytest.fixture
def regulargrid():
    x0, y0 = 0, 0
    dx, dy = 1, 1
    nx, ny = 10, 10
    return RegularGrid(x0=x0, y0=y0, dx=dx, dy=dy, nx=nx, ny=ny)


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


def test_equivalence(regulargrid, grid):
    assert np.array_equal(regulargrid.x, grid.x)
    assert np.array_equal(regulargrid.y, grid.y)
