import os
import subprocess
import sys

import pytest

# Add the tests directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import test utilities for logging
from test_utils.logging import configure_test_logging


@pytest.fixture(scope="session", autouse=True)
def docker_available() -> bool:
    try:
        result = subprocess.run(
            [
                "docker",
                "info",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


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
    getattr(logging, log_level_str)

    # Configure logging for tests
    configure_test_logging(level=log_level_str)


# Helper to detect CI environment


def is_ci_environment():
    """Detect if running in a CI environment."""
    ci_vars = [
        "CI",
        "GITHUB_ACTIONS",
        "GITLAB_CI",
        "TRAVIS",
        "CIRCLECI",
        "APPVEYOR",
        "BUILDKITE",
        "TEAMCITY_VERSION",
    ]
    return any(
        os.environ.get(var, "").lower() in ("1", "true", "yes", "on") for var in ci_vars
    )


@pytest.fixture(scope="session")
def should_skip_docker_builds():
    """Fixture to determine if Docker builds should be skipped."""
    if os.environ.get("SKIP_DOCKER_BUILDS", "").lower() in ("1", "true", "yes", "on"):
        return True
    if os.environ.get("ROMPY_SKIP_DOCKER_BUILDS", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return True
    if os.environ.get("ROMPY_ENABLE_DOCKER_IN_CI", "").lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return is_ci_environment()
    return False


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
