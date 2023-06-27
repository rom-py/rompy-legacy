"""Test time sub-component."""
import pytest

from rompy.swan.subcomponents.time import NONSTATIONARY


def test_nonstationary():
    nonstat = NONSTATIONARY(
        tbeg="2023-01-01T00:00:00",
        delt="PT30M",
        tend="2023-02-01T00:00:00",
        suffix="inp",
    )
    assert (
        nonstat.render()
        == "NONSTATIONARY tbeginp=20230101.000000 deltinp=1800.0 SEC tendinp=20230201.000000"
    )
