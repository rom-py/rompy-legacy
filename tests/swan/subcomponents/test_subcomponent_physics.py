"""Test physics sub-components."""
import pytest

from rompy.swan.subcomponents.physics import JANSSEN, KOMEN, WESTHUYSEN, ST6, ST6C1, ST6C2, ST6C3, ST6C4, ST6C5


def test_janssen():
    jans = JANSSEN()
    assert jans.render() == "JANSSEN DRAG WU"
    jans = JANSSEN(cds1=4.5, delta=0.5, wind_drag="fit", agrow=True, a=0.0015)
    assert jans.render() == "JANSSEN cds1=4.5 delta=0.5 DRAG FIT AGROW a=0.0015"


def test_komen():
    kom = KOMEN()
    assert kom.render() == "KOMEN DRAG WU"
    kom = KOMEN(cds2=2.36e-5, stpm=3.02e-3, wind_drag="fit", agrow=True, a=0.0020)
    assert kom.render() == "KOMEN cds2=2.36e-05 stpm=0.00302 DRAG FIT AGROW a=0.002"


def test_westhuysen():
    west = WESTHUYSEN()
    assert west.render() == "WESTHUYSEN DRAG WU"
    west = WESTHUYSEN(cds2=5.0e-5, br=1.7e-3, wind_drag="fit", agrow=True)
    assert west.render() == "WESTHUYSEN cds2=5e-05 br=0.0017 DRAG FIT AGROW a=0.0015"


def test_st6_default():
    st6 = ST6(
        a1sds=4.7e-7,
        a2sds=6.6e-6,
    )
    assert st6.render() == "ST6 a1sds=4.7e-07 a2sds=6.6e-06 UP HWANG VECTAU U10PROXY windscaling=32.0"


def test_st6_c1():
    st6 = ST6C1()
    assert st6.render() == "ST6 a1sds=4.7e-07 a2sds=6.6e-06 p1sds=4.0 p2sds=4.0 UP HWANG VECTAU U10PROXY windscaling=28.0 AGROW a=0.0015"


def test_st6_c2():
    st6 = ST6C2()
    assert st6.render() == "ST6 a1sds=4.7e-07 a2sds=6.6e-06 p1sds=4.0 p2sds=4.0 UP FAN VECTAU U10PROXY windscaling=28.0 AGROW a=0.0015"


def test_st6_c3():
    st6 = ST6C3()
    assert st6.render() == "ST6 a1sds=2.8e-06 a2sds=3.5e-05 p1sds=4.0 p2sds=4.0 UP HWANG VECTAU U10PROXY windscaling=32.0 AGROW a=0.0015"


def test_st6_c4():
    st6 = ST6C4()
    assert st6.render() == "ST6 a1sds=2.8e-06 a2sds=3.5e-05 p1sds=4.0 p2sds=4.0 UP HWANG VECTAU U10PROXY windscaling=32.0 DEBIAS cdfac=0.89 AGROW a=0.0015"


def test_st6_c5():
    st6 = ST6C5()
    assert st6.render() == "ST6 a1sds=6.5e-06 a2sds=8.5e-05 p1sds=4.0 p2sds=4.0 UP HWANG VECTAU U10PROXY windscaling=35.0 DEBIAS cdfac=0.89 AGROW a=0.0015"
