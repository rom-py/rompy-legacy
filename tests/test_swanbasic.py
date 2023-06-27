import os
import shutil
from datetime import datetime

import pytest
from utils import compare_files

from rompy import ModelRun
from rompy.core import BaseConfig, TimeRange
from rompy.swan import SwanConfig, SwanDataGrid, SwanGrid

here = os.path.dirname(os.path.abspath(__file__))


def test_swanbasic():
    """Test the swantemplate function."""
    time = TimeRange(
        start=datetime(2020, 2, 21, 4), end=datetime(2020, 2, 24, 4), interval="15M"
    )
    runtime = ModelRun(
        run_id="test_swantemplatebasic",
        output_dir=os.path.join(here, "simulations"),
        config=BaseConfig(
            friction_coefficient=0.1,
            model_type="base",
            template=os.path.join(here, "../rompy/templates/swanbasic"),
        ),
    )
    runtime.generate()
    compare_files(
        os.path.join(here, "simulations/test_swan_ref/INPUT_NEW"),
        os.path.join(here, "simulations/test_swantemplatebasic/INPUT"),
    )
    shutil.rmtree(os.path.join(here, "simulations/test_swantemplatebasic"))
