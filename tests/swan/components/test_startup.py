"""Test startup components."""
from pydantic import ValidationError

from rompy.swan.components.startup import PROJECT, SET, MODE, COORDINATES


def test_project():
    proj = PROJECT(
        name="Test project",
        nr="0001",
        title1="Title 1",
        title2="Title 2",
        title3="Title 3",
    )
    assert proj.render() == (
        "PROJECT name='Test project' nr='0001' "
        "title1='Title 1' title2='Title 2' title3='Title 3'"
    )


def test_project_no_titles():
    proj = PROJECT(
        name="Test project",
        nr="0001",
    )
    assert proj.render() == ("PROJECT name='Test project' nr='0001'")


def test_set():
    s = SET(direction_convention="nautical")
    assert s.render() == "SET NAUTICAL"


def test_mode_default():
    mode = MODE()
    assert mode.render() == "MODE STATIONARY TWODIMENSIONAL"


def test_mode_non_default():
    mode = MODE(kind="stationary", dim="onedimensional")
    assert mode.render() == "MODE STATIONARY ONEDIMENSIONAL"


def test_coord_default():
    coord = COORDINATES()
    assert coord.render() == "COORDINATES CARTESIAN"
