# Import test utilities
from test_utils.logging import get_test_logger

# Initialize logger
logger = get_test_logger(__name__)


"""Test SWAN subcomponents."""

from rompy.swan.subcomponents.spectrum import BIN, GAUSS, JONSWAP, PM, SHAPESPEC, TMA


def test_jonswap():
    shape = JONSWAP()
    assert shape.render() == "JONSWAP gamma=3.3"


def test_tma():
    shape = TMA(d=10.0)
    assert shape.render() == "TMA gamma=3.3 d=10.0"


def test_gauss():
    shape = GAUSS(sigfr=0.05)
    assert shape.render() == "GAUSS sigfr=0.05"


def test_pm():
    shape = PM()
    assert shape.render() == "PM"


def test_pm():
    shape = BIN()
    assert shape.render() == "BIN"


def test_shapespec():
    shape = SHAPESPEC()
    assert shape.render() == "BOUND SHAPESPEC JONSWAP gamma=3.3 PEAK DSPR POWER"
