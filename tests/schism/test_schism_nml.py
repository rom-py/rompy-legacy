from datetime import datetime
from pathlib import Path
from shutil import rmtree

# Import test utilities
from test_utils.logging import get_test_logger

# Initialize logger
logger = get_test_logger(__name__)

# pytest.importorskip("rompy.schism")

from rompy.core.data import DataBlob
from rompy.core.time import TimeRange
from rompy.model import ModelRun
from rompy.schism import SCHISMConfig, SCHISMGrid
from rompy.schism.namelists import NML, Param, Wwminput

here = Path(__file__).parent


def test_schism_render(tmpdir):
    """Test the swantemplate function."""
    run_id = "test_nml"
    period = TimeRange(
        start=datetime(2022, 8, 1, 0), end=datetime(2022, 8, 3, 0), interval="15M"
    )
    runtime = ModelRun(
        period=period,
        run_id=run_id,
        output_dir=str(tmpdir),
        config=SCHISMConfig(
            grid=SCHISMGrid(
                hgrid=DataBlob(id="hgrid", source=here / "test_data" / "hgrid.gr3"),
                drag=1,
            ),
            nml=NML(
                param=Param(**{"core": {"ipre": 1}}),
                wwminput=Wwminput(**{"proc": {"dimmode": 2}}),
            ),
        ),
    )
    runtime.generate()
    # compare_files(
    #     runtime.output_dir / runtime.run_id / "param.nml",
    #     here / "reference_nmls" / runtime.run_id / "param.nml",
    # )
    # compare_files(
    #     runtime.output_dir / runtime.run_id / "wwminput.nml",
    #     here / "reference_nmls" / runtime.run_id / "wwminput.nml",
    # )

    # for fname in ["param.nml", "wwminput.nml"]:
    #     compare_files(
    #         here / "reference_files" / runtime.run_id / fname,
    #         tmpdir / runtime.run_id / fname,
    #     )
    #     # assert file exists
    #     for fname in [
    #         "diffmax.gr3",
    #         "diffmin.gr3",
    #         "hgrid.gr3",
    #         "hgrid_WWM.gr3",
    #         # "drag.gr3",
    #         "manning.gr3",
    #         "schism_bnd_spec_SWAN_500m_use_in_schism_2021Aug-Nov.nc",
    #         "vgrid.in",
    #         "wwmbnd.gr3",
    #     ]:
    #         assert (tmpdir / runtime.run_id / fname).exists()


if __name__ == "__main__":
    tmpdir = "test_schism"
    rmtree(tmpdir, ignore_errors=True)
    test_schism_render(tmpdir)
