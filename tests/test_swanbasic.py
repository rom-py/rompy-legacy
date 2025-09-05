from datetime import datetime
from pathlib import Path

import pytest

# Import test utilities
from test_utils.logging import get_test_logger
from utils import compare_files

from rompy.core.config import BaseConfig
from rompy.core.time import TimeRange
from rompy.model import ModelRun

# Initialize logger
logger = get_test_logger(__name__)

here = Path(__file__).parent


@pytest.mark.skip(reason="frequency removed from ModelRun so this template breaks")
def test_swanbasic(tmpdir):
    """Test the swantemplate function."""
    TimeRange(
        start=datetime(2020, 2, 21, 4), end=datetime(2020, 2, 24, 4), interval="15M"
    )
    runtime = ModelRun(
        run_id="test_swantemplatebasic",
        output_dir=str(tmpdir),
        config=BaseConfig(
            friction_coefficient=0.1,
            model_type="base",
            template=str(here.parent / "rompy/templates/swanbasic"),
        ),
    )
    runtime.generate()
    compare_files(
        here / "simulations" / "test_swan_ref" / "INPUT_NEW",
        tmpdir / runtime.run_id / "INPUT",
    )
