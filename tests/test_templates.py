import os
import shutil
from datetime import datetime

import pytest
from utils import compare_files

from rompy import TEMPLATES_DIR, ModelRun
from rompy.core import BaseConfig, TimeRange

here = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def template():
    return BaseConfig(
        template="../rompy/templates/base",
    )


# def test_template():
#     config = BaseConfig()
#     assert config.template == os.path.join(TEMPLATES_DIR, "base")


def test_newbaseconfig():
    """Test the swantemplate function."""
    run_dir = os.path.join(here, "simulations")
    run_id = "test_base"
    config = BaseConfig(arg1="foo", arg2="bar")
    runtime = ModelRun(
        run_id=run_id,
        output_dir=run_dir,
        config=config,
    )
    runtime.generate()
    compare_files(
        os.path.join(here, "simulations/test_base_ref/INPUT"),
        os.path.join(run_dir, run_id, "INPUT"),
    )
    shutil.rmtree(os.path.join(run_dir, run_id))


def test_custom_template():
    run_dir = os.path.join(here, "simulations")
    run_id = "test_base"
    config = BaseConfig(arg1="foo", arg2="bar")
    runtime = ModelRun(
        run_id=run_id,
        output_dir=run_dir,
        template="simple_templates/base",
        config=config,
    )
    runtime.generate()
    compare_files(
        os.path.join(here, "simulations/test_base_ref/INPUT"),
        os.path.join(run_dir, run_id, "INPUT"),
    )
    # shutil.rmtree(os.path.join(run_dir, run_id))
