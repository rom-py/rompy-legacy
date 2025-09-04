"""Test physics sub-components."""

import pytest
from pydantic import ValidationError
# Import test utilities
from test_utils.logging import get_test_logger

# Initialize logger
logger = get_test_logger(__name__)
from rompy.swan.subcomponents.physics import (DANGREMOND, DEWIT, ELDEBERKY,
                                              FREEBOARD, GODA, JANSSEN, KOMEN,
                                              LINE, RDIFF, REFL, RSPEC, ST6,
                                              ST6C1, ST6C2, ST6C3, ST6C4,
                                              ST6C5, TRANS1D, TRANS2D, TRANSM,
                                              WESTHUYSEN)


# =====================================================================================
# Source terms
# =====================================================================================
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
    west = WESTHUYSEN(cds2=5.0e-5, br=1.7e-3, wind_drag="fit", agrow=True, a=0.0015)
    assert west.render() == "WESTHUYSEN cds2=5e-05 br=0.0017 DRAG FIT AGROW a=0.0015"


def test_st6_default():
    st6 = ST6(
        a1sds=4.7e-7,
        a2sds=6.6e-6,
    )
    assert st6.render() == (
        "ST6 a1sds=4.7e-07 a2sds=6.6e-06 UP HWANG VECTAU U10PROXY windscaling=32.0"
    )


def test_st6_c1():
    st6 = ST6C1()
    assert st6.render() == (
        "ST6 a1sds=4.7e-07 a2sds=6.6e-06 p1sds=4.0 p2sds=4.0 UP HWANG VECTAU U10PROXY "
        "windscaling=28.0 AGROW"
    )


def test_st6_c2():
    st6 = ST6C2()
    assert st6.render() == (
        "ST6 a1sds=4.7e-07 a2sds=6.6e-06 p1sds=4.0 p2sds=4.0 UP FAN VECTAU U10PROXY "
        "windscaling=28.0 AGROW"
    )


def test_st6_c3():
    st6 = ST6C3()
    assert st6.render() == (
        "ST6 a1sds=2.8e-06 a2sds=3.5e-05 p1sds=4.0 p2sds=4.0 UP HWANG VECTAU U10PROXY "
        "windscaling=32.0 AGROW"
    )


def test_st6_c4():
    st6 = ST6C4()
    assert st6.render() == (
        "ST6 a1sds=2.8e-06 a2sds=3.5e-05 p1sds=4.0 p2sds=4.0 UP HWANG VECTAU U10PROXY "
        "windscaling=32.0 DEBIAS cdfac=0.89 AGROW"
    )


def test_st6_c5():
    st6 = ST6C5()
    assert st6.render() == (
        "ST6 a1sds=6.5e-06 a2sds=8.5e-05 p1sds=4.0 p2sds=4.0 UP HWANG VECTAU U10PROXY "
        "windscaling=35.0 DEBIAS cdfac=0.89 AGROW"
    )


# =====================================================================================
# Triads
# =====================================================================================
def test_biphase_elderberky():
    biphase = ELDEBERKY()
    assert biphase.render() == "BIPHASE ELDEBERKY"
    biphase = ELDEBERKY(urcrit=0.63)
    assert biphase.render() == "BIPHASE ELDEBERKY urcrit=0.63"


def test_biphase_dewit():
    biphase = DEWIT()
    assert biphase.render() == "BIPHASE DEWIT"
    biphase = DEWIT(lpar=0)
    assert biphase.render() == "BIPHASE DEWIT lpar=0.0"


# =====================================================================================
# Obstacle
# =====================================================================================
def test_transm():
    trans = TRANSM()
    assert trans.render() == "TRANSM"
    trans = TRANSM(trcoef=0.0)
    assert trans.render() == "TRANSM trcoef=0.0"
    with pytest.raises(ValidationError):
        TRANSM(trcoef=1.1)


def test_trans1d():
    trans = TRANS1D(trcoef=[0.0, 0.0, 0.3, 0.2])
    assert trans.render() == "TRANS1D 0.0 0.0 0.3 0.2"
    with pytest.raises(ValidationError):
        TRANS1D(trcoef=[1.1, 0.0, 0.3, 0.2, 0.1])


def test_trans2d():
    trans = TRANS2D(trcoef=[[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]])
    assert "TRANS2D" in trans.render()
    "0.0 0.0 0.0" in trans.render()
    "0.1 0.1 0.1" in trans.render()
    with pytest.raises(ValidationError):
        TRANS2D(trcoef=[[0.0, 0.0, 0.0], [0.1, 0.1, 0.1, 0.1]])
    with pytest.raises(ValidationError):
        TRANS2D(trcoef=[[0.0, 0.0, 0.0], [0.1, 0.1, 1.1]])


def test_goda():
    trans = GODA(hgt=1.0)
    assert trans.render() == "DAM GODA hgt=1.0"
    trans = GODA(hgt=-1.0, alpha=2.6, beta=0.15)
    assert trans.render() == "DAM GODA hgt=-1.0 alpha=2.6 beta=0.15"


def test_dangremond():
    trans = DANGREMOND(hgt=3.0, slope=60, Bk=10.0)
    assert trans.render() == "DAM DANGREMOND hgt=3.0 slope=60.0 Bk=10.0"
    with pytest.raises(ValidationError):
        DANGREMOND(hgt=3.0, slope=120, Bk=1.0)


def test_refl():
    refl = REFL()
    assert refl.render() == "REFL"
    refl = REFL(reflc=0.5)
    assert refl.render() == "REFL reflc=0.5"


def test_rspec():
    refl = RSPEC()
    assert refl.render() == "RSPEC"


def test_rdiff():
    refl = RDIFF()
    assert refl.render() == "RDIFF"
    refl = RDIFF(pown=1.0)
    assert refl.render() == "RDIFF pown=1.0"


def test_freeboard():
    free = FREEBOARD(
        hgt=2.0,
        gammat=0.5,
        gammar=0.5,
        quay=True,
    )
    assert free.render() == "FREEBOARD hgt=2.0 gammat=0.5 gammar=0.5 QUAY"


def test_freeboard_no_quay():
    free = FREEBOARD(
        hgt=2.0,
        gammat=0.5,
        gammar=0.5,
        quay=False,
    )
    assert free.render() == "FREEBOARD hgt=2.0 gammat=0.5 gammar=0.5"


def test_freeboard_gamma_gt_0():
    with pytest.raises(ValidationError):
        FREEBOARD(hgt=2.0, gammat=0.0)
    with pytest.raises(ValidationError):
        FREEBOARD(hgt=2.0, gammar=0.0)


def test_line():
    line = LINE(xp=[174.1, 174.2, 174.3], yp=[-39.1, -39.1, -39.1])
    assert [f"{x} {y}" in line.render() for x, y in zip(line.xp, line.yp)]
