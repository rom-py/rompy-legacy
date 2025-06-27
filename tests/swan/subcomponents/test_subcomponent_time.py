"""Test time sub-component."""

import pytest

# Import test utilities
from test_utils.logging import get_test_logger

# Initialize logger
logger = get_test_logger(__name__)

from datetime import datetime, timedelta

from rompy.swan.subcomponents.time import (
    Time,
    Delt,
    TimeRangeOpen,
    TimeRangeClosed,
    STATIONARY,
    NONSTATIONARY,
)


@pytest.mark.parametrize(
    "tfmt, tmpl",
    [
        (1, "%Y%m%d.%H%M%S"),
        (2, "'%d-%b-%y %H:%M:%S'"),
        (3, "%m/%d/%y.%H:%M:%S"),
        (4, "%H:%M:%S"),
        (5, "%y/%m/%d %H:%M:%S'"),
        (6, "%y%m%d%H%M"),
    ],
)
def test_time(tfmt, tmpl):
    t = datetime(1990, 1, 1, 0, 0, 0)
    time = Time(time=t, tfmt=tfmt)
    assert time.render() == t.strftime(tmpl)


@pytest.mark.parametrize(
    "delt, dfmt, rendered",
    [
        (1800, "sec", "1800.0 SEC"),
        (timedelta(hours=2), "min", "120.0 MIN"),
        ("PT60M", "hr", "1.0 HR"),
    ],
)
def test_delt(delt, dfmt, rendered):
    delta = Delt(delt=delt, dfmt=dfmt)
    assert delta.render() == rendered


def test_timerange_open():
    tr = TimeRangeOpen(tbeg="2023-01-01T00:00:00", delt="PT30M", dfmt="min")
    assert tr.render() == "tbeg=20230101.000000 delt=30.0 MIN"


def test_timerange_closed():
    tr = TimeRangeClosed(
        tbeg="2023-01-01T00:00:00",
        tend="2023-02-01T00:00:00",
        delt="PT30M",
        dfmt="min",
    )
    assert tr.render() == "tbeg=20230101.000000 delt=30.0 MIN tend=20230201.000000"


def test_nonstationary():
    tr = NONSTATIONARY(
        tbeg="2023-01-01T00:00:00",
        tend="2023-02-01T00:00:00",
        delt="PT30M",
        tfmt=6,
        dfmt="sec",
        suffix="tbl",
    )
    assert tr.render() == (
        "NONSTATIONARY tbegtbl=2301010000 delttbl=1800.0 SEC tendtbl=2302010000"
    )


def test_stationary():
    stat = STATIONARY(time="2023-01-01T00:00:00", tfmt=1)
    assert stat.render() == "STATIONARY time=20230101.000000"
