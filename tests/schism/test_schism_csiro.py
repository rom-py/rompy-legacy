import sys
from datetime import datetime
from pathlib import Path

import pytest

pytest.importorskip("rompy.schism")
from rompy.core.data import DataBlob
from rompy.core.time import TimeRange
from rompy.model import ModelRun
from rompy.schism.config_legacy import Inputs, SchismCSIROConfig
from rompy.schism.grid import SCHISMGrid
from tests.utils import compare_files

here = Path(__file__).parent


@pytest.mark.skip(reason="Needs to be updated")
def test_schism_render(tmpdir):
    """Test the swantemplate function."""
    run_id = "test_schism"
    period = TimeRange(
        start=datetime(2021, 8, 1, 0), end=datetime(2021, 11, 29, 0), interval="15M"
    )
    runtime = ModelRun(
        period=period,
        run_id=run_id,
        output_dir=str(tmpdir),
        config=SchismCSIROConfig(
            grid=SCHISMGrid(
                hgrid=DataBlob(id="hgrid", source=here / "test_data" / "hgrid.gr3"),
                vgrid=DataBlob(id="vgrid", source=here / "test_data" / "vgrid.in"),
                diffmin=DataBlob(
                    id="diffmin", source=here / "test_data" / "diffmin.gr3"
                ),
                diffmax=DataBlob(
                    id="diffmax", source=here / "test_data" / "diffmax.gr3"
                ),
                # drag=DataBlob(id="drag", source=here /
                #               "test_data" / "drag.gr3"),
                manning=DataBlob(
                    id="manning", source=here / "test_data" / "manning.gr3"
                ),
                # rough=DataBlob(id="rough", source=here /
                #                "test_data" / "rough.gr3"),
                # hgridll=DataBlob(
                #     id="hgridll", source=here / "test_data" / "hgridll.gr3"
                # ),
                hgrid_WWM=DataBlob(
                    id="hgrid_WWM", source=here / "test_data" / "hgrid_WWM.gr3"
                ),
                wwmbnd=DataBlob(id="wwmbnd", source=here / "test_data" / "wwmbnd.gr3"),
            ),
            inputs=Inputs(
                filewave=DataBlob(
                    id="filewave",
                    source=here
                    / "test_data"
                    / "schism_bnd_spec_SWAN_500m_use_in_schism_2021Aug-Nov.nc",
                ),
            ),
        ),
    )
    runtime.generate()
    for fname in ["param.nml", "wwminput.nml"]:
        compare_files(
            here / "reference_files" / runtime.run_id / fname,
            tmpdir / runtime.run_id / fname,
        )
        # assert file exists
        for fname in [
            "diffmax.gr3",
            "diffmin.gr3",
            "hgrid.gr3",
            "hgrid_WWM.gr3",
            # "drag.gr3",
            "manning.gr3",
            "schism_bnd_spec_SWAN_500m_use_in_schism_2021Aug-Nov.nc",
            "vgrid.in",
            "wwmbnd.gr3",
        ]:
            assert (tmpdir / runtime.run_id / fname).exists()
