from pathlib import Path

import pytest

from rompy.core import DataBlob
from rompy.schism.grid import VgridGenerator, VGRID_TYPE_2D, VGRID_TYPE_LSC2, VGRID_TYPE_SZ

HERE = Path(__file__).parent


@pytest.fixture
def hgrid():
    return HERE / "test_data/hgrid.gr3"


def test_vgridgenerator2d(tmp_path):
    # Using the default 2D grid type
    vgrid = VgridGenerator(vgrid_type="2d")
    vgrid.generate(tmp_path)


def test_vgridgenerator3dLSC2(tmp_path, hgrid):
    # Using the LSC2 vertical grid type with parameters
    vgrid = VgridGenerator(vgrid_type="lsc2", nvrt=10, hsm=10.0)
    vgrid.generate(tmp_path)


def test_vgridgenerator3dSZ(tmp_path, hgrid):
    # Using the SZ vertical grid type with parameters
    vgrid = VgridGenerator(vgrid_type="sz", nvrt=10, h_c=5.0, theta_b=0.5, theta_f=0.5)
    vgrid.generate(tmp_path)
