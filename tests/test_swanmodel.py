import os
import shutil

import pytest
from utils import compare_files

from rompy.swan import SwanConfig, SwanGrid, SwanModel

here = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def grid():
    return SwanGrid(x0=115.68, y0=-32.76, dx=0.001, dy=0.001, nx=390, ny=150, rot=77)


@pytest.fixture
def model():
    return SwanModel(
        run_id="test_swan",
        output_dir=os.path.join(here, "simulations"),
    )


@pytest.fixture
def nesting():
    return SwanModel(
        run_id="test_nesting",
        output_dir=os.path.join(here, "simulations"),
        config=SwanConfig(subnests=[SwanConfig(), SwanConfig(subnests=[SwanConfig()])]),
    )


@pytest.mark.skip(reason="Overlap here with swan temlate tests - need to consolidate")
def test_generate(model):
    model.config.write(
        SwanModel(run_id="test_swan", output_dir=os.path.join(here, "simulations"))
    )
    compare_files(
        os.path.join(here, "simulations/test_swan/INPUT"),
        os.path.join(here, "simulations/test_swan_ref/INPUT"),
    )
    # shutil.rmtree(os.path.join(here, "simulations/test_swan"))


def test_swan_input(grid):
    model = SwanModel(
        run_id="test_swan",
        output_dir="simulations",
        config=SwanConfig(grid=grid, physics=dict(friction="MAD")),
    )
    assert model.config.physics.friction == "MAD"


def test_failing_friction():
    with pytest.raises(ValueError):
        model = SwanModel(
            run_id="test_swan",
            output_dir="simulations",
            config=dict(friction="BAD", model_type="swan"),
        )


@pytest.mark.skip(reason="not working, need to investigate")
def test_nesting(nesting):
    nesting.generate()
    # compare_files(
    #     os.path.join(here, "simulations/test_swan/INPUT"),
    #     os.path.join(here, "simulations/test_swan_ref/INPUT"),
    # )
    # shutil.rmtree(os.path.join(here, "simulations/test_swan"))
