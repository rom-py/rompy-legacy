"""Test startup components."""
from pydantic import ValidationError

from rompy.swan.components.physics import GEN1, GEN2, GEN3


def test_gen1_default():
    phys = GEN1()
    assert phys.render() == "GEN1"


def test_gen1_set_all():
    phys = GEN1(
        cf10=188,
        cf20=0.59,
        cf30=0.12,
        cf40=250,
        edmlpm=0.0036,
        cdrag=0.0012,
        umin=1,
        cfpm=0.13,
    )
    assert phys.render() == "GEN1 cf10=188.0 cf20=0.59 cf30=0.12 cf40=250.0 edmlpm=0.0036 cdrag=0.0012 umin=1.0 cfpm=0.13"


def test_gen2_default():
    phys = GEN2()
    assert phys.render() == "GEN2"


def test_gen2_set_all():
    phys = GEN2(
        cf10=188,
        cf20=0.59,
        cf30=0.12,
        cf40=250,
        cf50=0.0023,
        cf60=-0.223,
        edmlpm=0.0036,
        cdrag=0.0012,
        umin=1,
        cfpm=0.13,
    )
    assert phys.render() == "GEN2 cf10=188.0 cf20=0.59 cf30=0.12 cf40=250.0 cf50=0.0023 cf60=-0.223 edmlpm=0.0036 cdrag=0.0012 umin=1.0 cfpm=0.13"
