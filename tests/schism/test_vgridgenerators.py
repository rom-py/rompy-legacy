from pathlib import Path

import pytest

from rompy.core.data import DataBlob
from rompy.schism.grid import Vgrid3D_LSC2, Vgrid3D_SZ, VgridGenerator

HERE = Path(__file__).parent


@pytest.fixture
def hgrid():
    return HERE / "test_data/hgrid.gr3"


def test_vgridgenerator2d(tmp_path):
    vgrid = VgridGenerator()
    vgrid.generate(tmp_path)


def test_vgridgenerator3dLSC2(tmp_path, hgrid):

    vgrid = VgridGenerator(
        vgrid=Vgrid3D_LSC2(
            hgrid=hgrid,
            hsm=[10],
            nv=[10],
            h_c=5,
            theta_b=0.5,
            theta_f=0.5,
        )
    )
    vgrid.generate(tmp_path)


def test_vgridgenerator3dSZ(tmp_path, hgrid):
    vgrid = VgridGenerator(
        vgrid=Vgrid3D_SZ(
            hgrid=hgrid,
            h_s=10,
            ztot=[10],
            h_c=5,
            theta_b=0.5,
            theta_f=0.5,
            sigma=[0.5],
        )
    )
    vgrid.generate(tmp_path)
