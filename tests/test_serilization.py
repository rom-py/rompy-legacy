
# Import test utilities
from test_utils.logging import get_test_logger

from rompy.core.time import TimeRange

# Initialize logger
logger = get_test_logger(__name__)


def test_time_range_model_dump():
    """Test TimeRange serialization with model_dump."""

    # Create a TimeRange instance
    time_range = TimeRange(start="2025-01-01T00:00:00", end="2025-01-02T00:00:00")

    # Serialize the TimeRange instance
    serialized = time_range.model_dump()

    # Deserialize back into a TimeRange instance
    deserialized = TimeRange(**serialized)

    # Verify the serialization and deserialization
    assert time_range.start == deserialized.start
    assert time_range.end == deserialized.end


def test_time_range_model_dump_json():
    """Test TimeRange serialization with model_dump_json."""

    # Create a TimeRange instance
    time_range = TimeRange(start="2025-01-01T00:00:00", end="2025-01-02T00:00:00")

    # Serialize the TimeRange instance to JSON
    serialized_json = time_range.model_dump_json()

    # Deserialize the JSON back into a TimeRange instance
    deserialized = TimeRange.model_validate_json(serialized_json)

    # Verify the serialization and deserialization
    assert time_range.start == deserialized.start
    assert time_range.end == deserialized.end
