import os
import sys
import pytest

# Add the tests directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import test utilities for logging
from test_utils.logging import configure_test_logging


def pytest_addoption(parser):
    """Add command-line options for pytest."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests",
    )
    parser.addoption(
        "--rompy-log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level for ROMPY tests",
    )


def pytest_configure(config):
    """Configure pytest with plugins and settings."""
    import logging

    # Get log level from command line or use default
    log_level_str = config.getoption("--rompy-log-level")
    log_level = getattr(logging, log_level_str)

    # Configure logging for tests
    configure_test_logging(level=log_level_str)


@pytest.fixture(scope="session", autouse=True)
def setup_logging():
    """Set up logging for all tests.

    This fixture runs once per test session and ensures that logging is properly
    configured for all tests.
    """
    # Import here to avoid circular imports
    from test_utils.logging import configure_test_logging

    # Configure logging with default settings
    configure_test_logging()

    # Return a function to reconfigure logging if needed
    return configure_test_logging
