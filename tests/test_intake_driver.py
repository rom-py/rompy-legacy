import os

# Import test utilities
from test_utils.logging import get_test_logger

# Initialize logger
logger = get_test_logger(__name__)

from datetime import datetime, timedelta

import pytest

import rompy
from rompy.core.data import DataGrid

# round now to the nearest 6 hours
cycle = datetime.utcnow().replace(
    hour=0, minute=0, second=0, microsecond=0
) - timedelta(days=2)


@pytest.fixture
def gfs():
    return DataGrid(
        id="gfs_wind",
        catalog=os.path.join(rompy.__path__[0], "catalogs", "gfs.yml"),
        dataset="gfs_glob05",
        params={"cycle": cycle},
        filter={
            "crop": {
                "time": slice(
                    cycle,
                    cycle + timedelta(days=1),
                ),  # Change dates to available dates
                "lat": slice(0, 10),
                "lon": slice(0, 10),
            },
            "subset": {"data_vars": ["ugrd10m", "vgrd10m"]},
        },
    )


@pytest.fixture
def csiro():
    return DataGrid(
        id="TODO",
        catalog=os.path.join(rompy.__path__[0], "catalogs", "gfs.yml"),
        dataset="TODO",
        params={"cycle": cycle},
        filter={
            "crop": {
                "time": slice(
                    cycle,
                    cycle + timedelta(days=1),
                ),  # Change dates to available dates
                "lat": slice(0, 10),
                "lon": slice(0, 10),
            },
            "subset": {"data_vars": ["ugrd10m", "vgrd10m"]},
        },
    )


# mark as slow
@pytest.mark.skipif(
    "not config.getoption('--run-slow')",
    reason="Only run when --run-slow is given",
)
def test_gfs(tmpdir, gfs):
    """Test that the GFS catalog works"""
    assert gfs.ds.lat.max() == 10
    assert gfs.ds.lat.min() == 0
    assert gfs.ds.lon.max() == 10
    assert gfs.ds.lon.min() == 0
    downloaded = gfs.get(tmpdir)
    assert downloaded.ds.lat.max() == 10
    assert downloaded.ds.lat.min() == 0
    assert downloaded.ds.lon.max() == 10
    assert downloaded.ds.lon.min() == 0


@pytest.mark.skip(reason="Not yet implemented - Ben to have a look")
def test_csiro(tmpdir, csiro):
    """Test that the CSIRO catalog works"""
    assert csiro.ds.lat.max() == 10
    assert csiro.ds.lat.min() == 0
    assert csiro.ds.lon.max() == 10
    assert csiro.ds.lon.min() == 0
    gfs.get(tmpdir)
    # These may not be exact, may need to fine tune
    # assert downloaded.ds.lat.max() == 10
    # assert downloaded.ds.lat.min() == 0
    # assert downloaded.ds.lon.max() == 10
    # assert downloaded.ds.lon.min() == 0
