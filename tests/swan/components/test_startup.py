"""Test startup components."""

import pytest
from pydantic import ValidationError


# Import test utilities
from test_utils.logging import get_test_logger

# Initialize logger
logger = get_test_logger(__name__)

from rompy.swan.components.startup import PROJECT, SET, MODE, COORDINATES
from rompy.swan.components.group import STARTUP


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
    assert proj.render() == "PROJECT name='Test project' nr='0001'"


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


def test_startup():
    proj = PROJECT(nr="01")
    set = SET(level=0.5, direction_convention="nautical")
    mode = MODE(kind="nonstationary", dim="twodimensional")
    coords = COORDINATES(kind=dict(model_type="spherical", projection="ccm"))
    startup = STARTUP(
        project=proj,
        set=set,
        mode=mode,
        coordinates=coords,
    )
    assert "PROJECT nr='01'" in startup.render()
    assert "SET level=0.5 NAUTICAL" in startup.render()
    assert "MODE NONSTATIONARY TWODIMENSIONAL" in startup.render()
    assert "COORDINATES SPHERICAL CCM" in startup.render()
