"""Test filesystem functions."""

from pathlib import Path
import pytest

from rompy import filesystem as fs


HERE = Path(__file__).parent


def test_ls():
    # Test with pathlib instance
    test_files = fs.ls(HERE)
    assert str(Path(__file__)) in test_files
    # Test with fsspec-like str protocol
    test_files = fs.ls("file://" + str(HERE))
    assert str(Path(__file__)) in test_files


def test_exists():
    assert fs.exists(HERE)
    assert fs.exists(__file__)


def test_isdir():
    assert fs.isdir(HERE)
    assert not fs.isdir(__file__)


def test_isfile():
    assert not fs.isfile(HERE)
    assert fs.isfile(__file__)


def test_makedirs(tmpdir):
    dirname = tmpdir / "dir1"
    fs.makedirs(dirname)
    dirname = tmpdir / "parent" / "dir2"
    with pytest.raises(FileNotFoundError):
        fs.makedirs(dirname)
    fs.makedirs(dirname, exist_ok=True)


def test_rm():
    pass


def test_get():
    pass


def test_put():
    pass
