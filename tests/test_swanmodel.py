from pathlib import Path

import pytest
# Import test utilities
from test_utils.logging import get_test_logger

from rompy.model import ModelRun
from tests.utils import compare_files

# Initialize logger
logger = get_test_logger(__name__)

from rompy.swan import SwanConfig, SwanGrid

here = Path(__file__).parent


@pytest.fixture
def grid():
    return SwanGrid(x0=115.68, y0=-32.76, dx=0.001, dy=0.001, nx=390, ny=150, rot=77)


@pytest.fixture
def model(tmp_path):
    return ModelRun(
        run_id="test_swan",
        output_dir=str(tmp_path),
    )


@pytest.fixture
def nesting(tmp_path):
    return ModelRun(
        run_id="test_nesting",
        output_dir=str(tmp_path),
        config=SwanConfig(
            subnests=[
                SwanConfig(),
                SwanConfig(subnests=[SwanConfig()]),
            ]
        ),
    )


@pytest.mark.skip(reason="Overlap here with swan temlate tests - need to consolidate")
def test_generate(tmp_path, model):
    model.config.write(
        ModelRun(
            run_id="test_swan",
            output_dir=str(tmp_path),
        )
    )
    compare_files(
        tmp_path / model.run_id / "INPUT",
        here / "simulations/test_swan_ref/INPUT",
    )


def test_swan_input(tmp_path, grid):
    model = ModelRun(
        run_id="test_swan",
        output_dir=str(tmp_path),
        config=SwanConfig(grid=grid, physics=dict(friction="MAD")),
    )
    assert model.config.physics.friction == "MAD"


def test_failing_friction(tmp_path):
    with pytest.raises(ValueError):
        ModelRun(
            run_id="test_swan",
            output_dir=str(tmp_path),
            config=dict(friction="BAD", model_type="swan"),
        )


@pytest.mark.skip(reason="not working, need to investigate")
def test_nesting(nesting):
    nesting.generate()
    # compare_files(
    #     os.path.join(here, "simulations/test_swan/INPUT"),
    #     os.path.join(here, "simulations/test_swan_ref/INPUT"),
    # )
