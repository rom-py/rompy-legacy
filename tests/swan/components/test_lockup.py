"""Test lockup components."""

from copy import deepcopy

import pytest

# Import test utilities
from test_utils.logging import get_test_logger

# Initialize logger
logger = get_test_logger(__name__)

from rompy.swan.components.group import LOCKUP
from rompy.swan.components.lockup import (
    COMPUTE,
    COMPUTE_NONSTAT,
    COMPUTE_STAT,
    HOTFILE,
    STOP,
)
from rompy.swan.subcomponents.time import NONSTATIONARY, STATIONARY


@pytest.fixture(scope="module")
def times():
    yield NONSTATIONARY(
        tbeg="1990-01-01T00:00:00",
        tend="1990-02-01T00:00:00",
        delt="PT1H",
        tfmt=1,
        dfmt="hr",
    )


def test_compute():
    comp = COMPUTE()
    assert comp.render() == "COMPUTE"


def test_compute_stationary():
    comp = COMPUTE(times=dict(model_type="stationary", time="1990-01-01T00:00:00"))
    assert comp.render() == "COMPUTE STATIONARY time=19900101.000000"


def test_compute_nonstationary(times):
    comp = COMPUTE(times=times)
    assert comp.render() == (
        "COMPUTE NONSTATIONARY tbegc=19900101.000000 deltc=1.0 HR "
        "tendc=19900201.000000"
    )


def test_hotfile_default():
    hotfile = HOTFILE(fname="hotfile")
    assert hotfile.render() == "HOTFILE fname='hotfile'"


def test_hotfile_unformatted():
    hotfile = HOTFILE(fname="hotfile", format="unformatted")
    assert hotfile.render() == "HOTFILE fname='hotfile' UNFORMATTED"


def test_stop():
    stop = STOP()
    assert stop.render() == "STOP"


def test_compute_nonstationary_single_no_hotfiles(times):
    compute = COMPUTE_NONSTAT(times=times)
    assert compute.render() == f"COMPUTE {times.render()}"


def test_compute_nonstationary_single_with_hotfile(times):
    compute = COMPUTE_NONSTAT(times=times, hotfile={"fname": "hotfile"}, hottimes=[-1])
    cmds = compute.render().split("\n")
    assert len(cmds) == 2
    assert cmds[0].startswith("COMPUTE NONSTAT") and cmds[1].startswith("HOTFILE")


def test_compute_nonstationary_multiple_with_hotfiles(times):
    hottimes = ["1990-01-01T06:00:00", "1990-01-01T12:00:00", "1990-01-01T18:00:00"]
    compute = COMPUTE_NONSTAT(
        times=times,
        hotfile={"fname": "hotfile"},
        hottimes=hottimes,
    )
    cmds = compute.render().split("\n")
    assert len(cmds) == 7
    assert cmds[1] == "HOTFILE fname='hotfile_19900101T060000'"
    assert cmds[3] == "HOTFILE fname='hotfile_19900101T120000'"
    assert cmds[5] == "HOTFILE fname='hotfile_19900101T180000'"
    assert cmds[6].startswith("COMPUTE NONSTATIONARY tbegc=19900101.180000")


def test_compute_nonstationary_initstat(times):
    compute = COMPUTE_NONSTAT(times=times, initstat=True)
    cmds = compute.render().split("\n")
    assert len(cmds) == 2
    assert cmds[0].startswith("COMPUTE STATIONARY")
    assert cmds[1].startswith("COMPUTE NONSTATIONARY")


def test_compute_stationary_single_no_hotfile():
    compute = COMPUTE_STAT(times=STATIONARY(time="1990-01-01T00:00:00"))
    assert compute.render() == "COMPUTE STATIONARY time=19900101.000000"


def test_compute_stationary_single_with_hotfile(times):
    t = deepcopy(times)
    t.tend = t.tbeg
    compute = COMPUTE_STAT(times=t, hotfile=dict(fname="hotfile"), hottimes=[-1])
    cmds = compute.render().split("\n")
    assert len(cmds) == 2
    assert cmds[0] == "COMPUTE STATIONARY time=19900101.000000"
    assert cmds[1] == "HOTFILE fname='hotfile_19900101T000000'"


def test_compute_stationary_multiple_no_hotfiles(times):
    t = deepcopy(times)
    t.tend = times()[6]
    compute = COMPUTE_STAT(times=t)
    cmds = compute.render().split("\n")
    assert len(cmds) == 7
    for cmd in cmds:
        assert cmd.startswith("COMPUTE STATIONARY")


def test_compute_stationary_multiple_with_hotfiles(times):
    t = deepcopy(times)
    t.tend = times()[6]
    hottimes = ["1990-01-01T03:00:00", "1990-01-01T06:00:00"]
    compute = COMPUTE_STAT(times=t, hotfile={"fname": "hotfile"}, hottimes=hottimes)
    cmds = compute.render().split("\n")
    assert len(cmds) == 9
    for ind in [0, 1, 2, 3, 5, 6, 7]:
        assert cmds[ind].startswith("COMPUTE STATIONARY")
    for ind in [4, 8]:
        assert cmds[ind].startswith("HOTFILE")


def test_lockup_compute_stat(times):
    lockup = LOCKUP(
        compute=dict(
            model_type="stat",
            times=STATIONARY(time="1990-01-01T00:00:00"),
        ),
    )
    assert lockup.compute.model_type == "stat"
    assert "STOP" in lockup.render()


def test_lockup_compute_nonstat(times):
    lockup = LOCKUP(compute=dict(model_type="nonstat", times=times))
    assert lockup.compute.model_type == "nonstat"
    assert "STOP" in lockup.render()


def test_compute_stationary_no_times_passed():
    compute = COMPUTE_STAT()
    assert compute.render() == "COMPUTE STATIONARY time=19700101.000000"


def test_compute_nonstationary_no_times_passed():
    compute = COMPUTE_NONSTAT()
    assert compute.render() == f"COMPUTE {compute.times.render()}"
