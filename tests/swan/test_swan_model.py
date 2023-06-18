"""Test swan_config class."""
import logging
import os
from pathlib import Path

import pytest
import yaml

from rompy.core.model import ModelRun
from rompy.swan.config import SwanConfigComponents as SwanConfig

logger = logging.getLogger(__name__)

HERE = Path(__file__).parent


@pytest.fixture(scope="module")
def config():
    yield yaml.load((HERE / "swan_model.yml").read_text(), Loader=yaml.Loader)


def test_swan_model(tmpdir, config):
    config = SwanConfig(
        template=os.path.join(HERE, "../../rompy/templates/swan2"),
        project=config["project"],
        set=config["set"],
        mode=config["mode"],
        coordinates=config["coordinates"],
        cgrid=config["cgrid"],
        inpgrid=config["inpgrid"],
        boundary=config["boundary"],
        initial=config["initial"],
        physics=config["physics"],
    )
    model = ModelRun(
        run_id="test",
        output_dir=str(tmpdir),
        config=config,
    )
    # import ipdb; ipdb.set_trace()
    model.generate()
