from pathlib import Path
from importlib.metadata import entry_points
import pytest

pytest.importorskip("rompy.schism")

from rompy.core import DataBlob
from rompy.core.grid import BaseGrid
from rompy.schism import SCHISMGrid
from rompy.schism.grid import WWMBNDGR3Generator

here = Path(__file__).parent

eps = entry_points(group="rompy.config")

@pytest.mark.skipif("schism" not in eps.names, reason="requires SCHISM")
def test_SCHISMGrid2D(tmpdir):
    hgrid = DataBlob(source=here / "test_data/hgrid.gr3")
    # drag = DataBlob(source=here / "test_data/drag.gr3")
    # rough = DataBlob(source=here / "test_data/rough.gr3")
    # manning = DataBlob(source=here / "test_data/manning.gr3")
    # hgridll = DataBlob(source=here / "test_data/hgrid.ll")
    # diffmin = DataBlob(source=here / "test_data/diffmin.gr3")
    # diffmax = DataBlob(source=here / "test_data/diffmax.gr3")
    # hgrid_WWM = DataBlob(source=here / "test_data/hgrid_WWM.gr3")
    # wwmbnd = DataBlob(source=here / "test_data/wwmbnd.gr3")

    grid = SCHISMGrid(
        hgrid=hgrid,
        drag=1,
        # rough=rough,
        # manning=1.2,
        # hgridll=hgridll,
        # diffmin=diffmin,
        # diffmax=diffmax,
        # hgrid_WWM=hgrid_WWM,
        # wwmbnd=wwmbnd,
    )

    # assert grid.is_3d == False
    # # assert grid.drag == drag
    # # assert grid.rough == rough
    # assert grid.manning == manning
    # assert grid.hgridll == hgridll
    # assert grid.diffmin == diffmin
    # assert grid.diffmax == diffmax
    # assert grid.hgrid_WWM == hgrid_WWM
    # assert grid.wwmbnd == wwmbnd

    assert grid.validate_rough_drag_manning(grid) == grid
    # assert that the gr3 file for each of the above is in the staging dir
    staging_dir = Path(tmpdir)
    ret = grid.get(staging_dir)

    assert staging_dir.joinpath("hgrid.gr3").exists()
    assert staging_dir.joinpath("hgrid.ll").exists()
    assert staging_dir.joinpath("hgrid.ll").is_symlink()
    assert staging_dir.joinpath("hgrid_WWM.gr3").is_symlink()
    assert staging_dir.joinpath("diffmin.gr3").exists()
    assert staging_dir.joinpath("diffmax.gr3").exists()
    assert staging_dir.joinpath("tvd.prop").exists()
    assert staging_dir.joinpath("vgrid.in").exists()


# def test_generate_wwmbnd():
#     hgrid = "test_data/hgrid.gr3"
#     wwmbnd = WWMBNDGR3Generator(hgrid=hgrid)
#     wwmbnd.get("./")
#
#     # assert contents of wwmbnd.gr3 and wwmbnd_ref.gr3 are the same
#     with open("wwmbnd.gr3", "r") as f:
#         wwmbnd_lines = f.readlines()
#     with open("wwmbnd_ref.gr3", "r") as f:
#         wwmbnd_ref_lines = f.readlines()
#     assert wwmbnd_lines == wwmbnd_ref_lines
