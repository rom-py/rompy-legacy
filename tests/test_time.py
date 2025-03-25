from datetime import datetime, timedelta

import pytest

from rompy.core import TimeRange


@pytest.fixture
def dtr_hourly():
    return TimeRange(end="2019-01-02", duration="1d")


@pytest.fixture
def dtr_daily():
    return TimeRange(start="2019-01-01", duration="3d", interval="1d")


def test_dtr(dtr_hourly):
    assert dtr_hourly.start == datetime(2019, 1, 1)
    assert dtr_hourly.end == datetime(2019, 1, 2)
    assert dtr_hourly.duration == timedelta(days=1)
    assert dtr_hourly.interval == timedelta(hours=1)


def test_bad_args():
    with pytest.raises(ValueError):
        t = TimeRange(start="2019-01-01")
    with pytest.raises(ValueError):
        t = TimeRange(end="2019-01-01")
    with pytest.raises(ValueError):
        t = TimeRange(start="2019-01-01", end="2019-01-01", duration="1d")
    with pytest.raises(ValueError):
        t = TimeRange()


def test_date_range_hourly(dtr_hourly):
    assert dtr_hourly.date_range[0] == datetime(2019, 1, 1)
    assert dtr_hourly.date_range[-1] == datetime(2019, 1, 2)
    assert len(dtr_hourly.date_range) == 25


def test_date_range_daily(dtr_daily):
    dtr_daily = TimeRange(start="2019-01-01", duration="3d", interval="1d")
    assert dtr_daily.date_range[0] == datetime(2019, 1, 1)
    assert dtr_daily.date_range[-1] == datetime(2019, 1, 4)
    assert len(dtr_daily.date_range) == 4
    dtr_daily = TimeRange(
        start="2019-01-01", duration="3d", interval="1d", include_end=False
    )
    assert dtr_daily.date_range[0] == datetime(2019, 1, 1)
    assert dtr_daily.date_range[-1] == datetime(2019, 1, 3)
    assert len(dtr_daily.date_range) == 3


def test_nested_serialization():
    """Test that duration is excluded during serialization when nested in another model."""
    from pydantic import BaseModel
    
    # Create a TimeRange with both start and end
    tr = TimeRange(start="2020-01-01", end="2020-01-02")
    
    # Verify that duration is computed but excluded from direct serialization
    assert tr.duration == timedelta(days=1)
    assert 'duration' not in tr.model_dump()
    
    # Create a container class with a TimeRange field
    class Container(BaseModel):
        time_range: TimeRange
    
    # Test that duration is excluded when TimeRange is nested inside another model
    container = Container(time_range=tr)
    serialized = container.model_dump()
    
    # Verify that duration is excluded from nested serialization
    assert 'duration' not in serialized['time_range']
    
    # Ensure other fields are still present
    assert 'start' in serialized['time_range']
    assert 'end' in serialized['time_range']
    assert 'interval' in serialized['time_range']
    assert 'include_end' in serialized['time_range']
    
    # Verify we can deserialize back without issues
    container2 = Container(**serialized)
    assert container2.time_range.start == datetime(2020, 1, 1, 0, 0)
    assert container2.time_range.end == datetime(2020, 1, 2, 0, 0)
    assert container2.time_range.duration == timedelta(days=1)
