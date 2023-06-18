"""Test subcomponents."""
import pytest
from pydantic import ValidationError

from rompy.swan.subcomponents.boundary import (
    CONSTANTPAR,
    VARIABLEPAR,
    CONSTANTFILE,
    VARIABLEFILE,
    SIDE,
    SEGMENTXY,
    SEGMENTIJ,
)


def test_side():
    side = SIDE(side="north")
    assert side.render().strip() == "SIDE NORTH CCW"
    with pytest.raises(ValidationError):
        SIDE(side="wrong_dir")


def test_segment_xy():
    seg = SEGMENTXY(points=((0, 0), (1, 1), (2, 2)))
    assert seg.render().startswith("SEGMENT XY")
    with pytest.raises(ValidationError):
        SEGMENTXY(points=((0, 0, 0), (1, 1, 1), (2, 2, 2)))


def test_segment_ij():
    seg = SEGMENTIJ(points=((0, 0), (1, 1), (2, 2)))
    with pytest.raises(ValidationError):
        SEGMENTIJ(points=((0, 0, 0), (1, 1, 1), (2, 2, 2)))


def test_par_constant():
    par = CONSTANTPAR(hs=1.0, per=10.0, dir=0.0, dd=0.0)
    assert par.render() == "CONSTANT PAR hs=1.0 per=10.0 dir=0.0 dd=0.0"


def test_par_variable():
    par = VARIABLEPAR(
        hs=[1.0, 1.0], per=[10.0, 10.0], dir=[0.0, 0.0], dd=[0.0, 0.0], len=[0, 1]
    )
    assert par.render().startswith("VARIABLE PAR &")


def test_varpar_all_same_size():
    with pytest.raises(ValidationError):
        VARIABLEPAR(hs=[1.0], per=[10.0], dir=[0.0], dd=[0.0, 0.0], len=[0.0])


def test_file_constant():
    file = CONSTANTFILE(fname="tpar.txt")
    assert file.render() == "CONSTANT FILE fname='tpar.txt'"


def test_file_variable():
    file = VARIABLEFILE(fname=["tpar0.txt", "tpar1.txt"], len=[0, 1])
    assert file.render().startswith("VARIABLE FILE &")


def test_varfile_all_same_size():
    with pytest.raises(ValidationError):
        VARIABLEFILE(fname=["tpar0.txt", "tpar1.txt"], len=[0, 1, 2])
