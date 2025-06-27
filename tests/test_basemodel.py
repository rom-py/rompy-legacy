from datetime import datetime
from pathlib import Path

import pytest
from tests.utils import compare_files

from rompy.core.config import BaseConfig
from rompy.core.time import TimeRange
from rompy.model import ModelRun


# Import test utilities
from test_utils.logging import get_test_logger

# Initialize logger
logger = get_test_logger(__name__)

here = Path(__file__).parent


@pytest.fixture
def model(tmpdir):
    return ModelRun(
        run_id="test_base",
        output_dir=str(tmpdir),
        config=BaseConfig(arg1="foo", arg2="bar"),
        # template=BaseConfig(),
    )


@pytest.fixture
def gitlab_template(tmpdir):
    return ModelRun(
        output_dir=str(tmpdir),
        template=BaseConfig(
            template="git@gitlab.com:oceanum/models/test-rompy-template.git",
        ),
    )


def test_datetime_parse(tmpdir):
    end = datetime(2022, 2, 21, 4)
    for format in [
        "%Y%m%d.%H%M%S",
        "%Y%m%d.%H%M",
        "%Y%m%dT%H%M%S",
        "%Y%m%dT%H%M",
    ]:
        model = ModelRun(
            period=TimeRange(end=end.strftime(format), duration="1d"),
            output_dir=str(tmpdir),
        )
        for period in ["year", "month", "day", "hour"]:
            assert getattr(model.period.end, period) == getattr(end, period)


def test_datetime_parse_fail(tmpdir):
    end = datetime(2022, 2, 21, 4)
    for format in [
        "%Y%m%d.%Hhello",
        "%Y%m%dhello",
    ]:
        try:
            ModelRun(
                period=TimeRange(end=end.strftime(format), duration="1d"),
                output_dir=str(tmpdir),
            )
        except ValueError:
            pass
        else:
            raise ValueError("Should not be able to parse {format}")


# test generate
def test_generate(model):
    model.generate()
    compare_files(
        Path(model.output_dir) / model.run_id / "INPUT",
        here / "simulations/test_base_ref/INPUT",
    )


# repeat suite for gitlab template
@pytest.mark.skip(reason="gitlab template not ready following restructure")
def test_gitlab_template(gitlab_template):
    gitlab_template.generate()
    compare_files(
        Path(gitlab_template.output_dir) / gitlab_template.run_id / "INPUT",
        here / "simulations/test_base_ref/INPUT",
    )
