from pathlib import Path

import pytest
from utils import compare_nmls


# Import test utilities
from test_utils.logging import get_test_logger

# Initialize logger
logger = get_test_logger(__name__)

pytest.importorskip("rompy.schism")

from rompy.schism.namelists import Ice, Icm, Mice, Param, Sediment

SAMPLE_DIR = (
    Path(__file__).parent
    / ".."
    / ".."
    / "rompy"
    / "schism"
    / "namelists"
    / "sample_inputs"
)


def test_namelists(tmp_path):
    for nml in [Icm, Param, Sediment, Mice, Ice]:
        instance = nml()
        instance.write_nml(tmp_path)
        name = instance.__class__.__name__.lower()
        compare_nmls(
            tmp_path / f"{name}.nml", SAMPLE_DIR / f"{name}.nml", raise_missing=True
        )
